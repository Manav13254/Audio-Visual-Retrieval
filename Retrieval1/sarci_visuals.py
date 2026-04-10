import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchaudio
import torchaudio.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Import from your modular files ---
from datasets import vision_transform
from quaternion_attn import QUATER_ATTENTION
from ca_attn import CA_Block
from iclm import ICLM

# ==========================================
# SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    # Point this to your saved ICLM model
    parser.add_argument("--model_path", type=str, default="best_iclm_model.pth")
    parser.add_argument("--output_dir", type=str, default="qualitative_results")
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.output_dir, exist_ok=True)

# --- Audioset Repo Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(CURRENT_DIR, "audioset_tagging_cnn")
sys.path.append(REPO_PATH)
sys.path.append(os.path.join(REPO_PATH, 'pytorch'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print("Error loading Cnn14. Ensure audioset_tagging_cnn is cloned.")
    sys.exit()

# ==========================================
# VISUALIZATION ARCHITECTURE (With Spatial Return)
# ==========================================
class IclmVisVisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Use Weights=None because we are loading your custom trained weights anyway
        resnet = resnet18(weights=None) 
        
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.f0conv = nn.Conv2d(64, 64, 3, 2, 1)
        self.f01conv = nn.Conv2d(128, 128, 7, 4, 3)
        self.f2conv = nn.Conv2d(128, 128, 3, 2, 1)
        self.fusionconv = nn.Conv2d(512, 512, 3, 2, 1)
        
        self.quater_att_fusion = QUATER_ATTENTION(512)
        self.quater_att_f4 = QUATER_ATTENTION(512)
        self.project = nn.Linear(512, embedding_dim)

    def forward(self, x, return_spatial=False):
        x = self.stem[0](x); x = self.stem[1](x); f0 = self.stem[2](x); x = self.stem[3](f0)
        f1 = self.layer1(x); f2 = self.layer2(f1); f3 = self.layer3(f2); f4 = self.layer4(f3)
        
        f0_d = self.f0conv(f0)
        f01 = self.f01conv(torch.cat([f0_d, f1], 1))
        f2_d = self.f2conv(f2)
        f23 = torch.cat([f2_d, f3], 1)
        
        fusion = self.fusionconv(torch.cat([f01, f23], 1))
        fusion = self.quater_att_fusion(fusion)
        f4_att = self.quater_att_f4(f4)
        
        final = f4_att * torch.sigmoid(fusion) + torch.sigmoid(f4_att) * fusion + f4_att
        out = F.adaptive_avg_pool2d(final, (1, 1)).flatten(1)
        v_base = F.normalize(self.project(out), p=2, dim=1)
        
        if return_spatial:
            B, C, H, W = final.shape
            spatial_flat = final.view(B, C, -1).transpose(1, 2)
            spatial_proj = self.project(spatial_flat)
            spatial_norm = F.normalize(spatial_proj, p=2, dim=-1)
            spatial_features = spatial_norm.transpose(1, 2).view(B, -1, H, W)
            return v_base, spatial_features
            
        return v_base

class IclmVisAudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.ca_block = CA_Block(2048)
        self.attn_pool = nn.Sequential(nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1))
        self.project = nn.Linear(4096, embedding_dim)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        x = self.base_model.spectrogram_extractor(x)
        x = self.base_model.logmel_extractor(x)
        x = x.transpose(1, 3); x = self.base_model.bn0(x); x = x.transpose(1, 3)
        
        x = self.base_model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        
        x = self.ca_block(x)
        x = torch.mean(x, dim=3).transpose(1, 2)
        w = self.attn_pool(x)
        mu = torch.sum(x * w, dim=1)
        var = torch.sum(w * (x - mu.unsqueeze(1))**2, dim=1)
        std = torch.sqrt(var.clamp(min=1e-5))
        
        out = torch.cat([mu, std], 1)
        return F.normalize(self.project(out), p=2, dim=1)

class IclmVisCrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_model = IclmVisVisionEncoder(embedding_dim)
        self.audio_model = IclmVisAudioEncoder(embedding_dim)
        self.iclm = ICLM(embed_dim=embedding_dim)


# ==========================================
# RETRIEVAL & HEATMAP GENERATION LOGIC
# ==========================================
def main():
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    print(f"Loading SARCI Model from {args.model_path}...")
    model = IclmVisCrossModalModel().to(DEVICE)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=False))
    else:
        print(f"ERROR: {args.model_path} not found. Please train the model first.")
        sys.exit()
    model.eval()

    all_classes = sorted(os.listdir(VAL_V))
    print(f"Generating 2-row overlay retrievals for all {len(all_classes)} classes...")

    # ----------------------------------------
    # STEP 1: PRE-COMPUTE FULL GALLERY
    # ----------------------------------------
    gallery_paths = []
    gallery_labels = []
    for cls in all_classes:
        v_files = sorted(glob.glob(os.path.join(VAL_V, cls, "*.*")))
        gallery_paths.extend(v_files)
        gallery_labels.extend([cls] * len(v_files))

    gallery_v_base = []
    batch_size = 128
    
    with torch.no_grad():
        for i in tqdm(range(0, len(gallery_paths), batch_size), desc="Extracting Image Gallery Base"):
            paths = gallery_paths[i:i+batch_size]
            imgs = [vision_transform(Image.open(p).convert('RGB')) for p in paths]
            imgs = torch.stack(imgs).to(DEVICE)
            v_base = model.vision_model(imgs) # Standard forward pass (pooled)
            gallery_v_base.append(v_base.cpu())
            
    gallery_v_base = torch.cat(gallery_v_base, dim=0)

    # ----------------------------------------
    # STEP 2: PROCESS QUERIES
    # ----------------------------------------
    random.seed(100) # Ensure identical query selection for comparisons
    
    for q_cls in all_classes:
        print(f"\nProcessing Query: {q_cls}...")
        
        a_files = sorted(glob.glob(os.path.join(VAL_A, q_cls, "*.wav")))
        if not a_files: continue
        
        wav_path = random.choice(a_files)
        wav, sr = torchaudio.load(wav_path) 
        if sr != 32000: wav = T.Resample(sr, 32000)(wav)
        if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
        else: wav = wav[:, :320000]
        if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
        wav_tensor = wav.to(DEVICE)
        
        with torch.no_grad():
            a_base = model.audio_model(wav_tensor).cpu()
            sims = []
            
            # --- FULL GALLERY RETRIEVAL ---
            for i in range(0, len(gallery_v_base), batch_size):
                v_chunk = gallery_v_base[i:i+batch_size].to(DEVICE)
                a_chunk = a_base.to(DEVICE)
                a_exp = a_chunk.expand(v_chunk.size(0), -1)
                
                v_final, a_final = model.iclm(v_chunk, a_exp)
                batch_sims = F.cosine_similarity(v_final, a_final, dim=1)
                sims.extend(batch_sims.cpu().numpy())
                
            # Get Top 5 Indices
            top5_indices = np.argsort(sims)[::-1][:5]
            
            # --- HEATMAP COMPUTATION FOR TOP 5 ONLY ---
            top5_paths = [gallery_paths[idx] for idx in top5_indices]
            top5_labels = [gallery_labels[idx] for idx in top5_indices]
            top5_sims = [sims[idx] for idx in top5_indices]
            
            top5_imgs = [vision_transform(Image.open(p).convert('RGB')) for p in top5_paths]
            top5_tensor = torch.stack(top5_imgs).to(DEVICE)
            
            # Re-run vision encoder on Top 5 WITH spatial tracking
            v_base_top5, v_spatial_top5 = model.vision_model(top5_tensor, return_spatial=True)
            
            # Re-run ICLM on Top 5 to get specific audio queries adapted to each image
            a_chunk_top5 = a_base.to(DEVICE).expand(5, -1)
            _, a_final_top5 = model.iclm(v_base_top5, a_chunk_top5)
            
            # Compute Heatmaps: Dot product of audio query against every spatial pixel
            a_expanded = a_final_top5.unsqueeze(-1).unsqueeze(-1)
            raw_heatmaps = torch.sum(a_expanded * v_spatial_top5, dim=1).cpu().numpy()
            
        # --- PLOTTING (2x6 Grid) ---
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        # Query Audio Column
        axes[0, 0].text(0.5, 0.6, "QUERY AUDIO", fontsize=18, ha='center', weight='bold')
        axes[0, 0].text(0.5, 0.4, f"Class:\n{q_cls}", fontsize=16, ha='center', color='darkblue')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Row Labels
        axes[0, 0].set_ylabel("Original Image", fontsize=16, weight='bold', visible=True)
        axes[1, 0].set_ylabel("Overlay", fontsize=16, weight='bold', visible=True)
        
        # Top 5 Image Columns
        for j in range(5):
            col = j + 1
            pil_img = Image.open(top5_paths[j]).convert('RGB')
            img_class = top5_labels[j]
            sim_score = top5_sims[j]
            color = 'limegreen' if img_class == q_cls else 'red'
            
            # 1. Process Heatmap
            hm = raw_heatmaps[j]
            hm = np.maximum(hm, 0) # ReLU equivalent
            hm = hm / (np.max(hm) + 1e-8) # Normalize 0-1
            heatmap_img = Image.fromarray(np.uint8(255 * hm)).resize(pil_img.size, Image.BILINEAR)
            
            # Row 1: Original Image
            axes[0, col].imshow(pil_img)
            axes[0, col].axis('off')
            rect = patches.Rectangle((0,0), pil_img.size[0], pil_img.size[1], 
                                     linewidth=16, edgecolor=color, facecolor='none')
            axes[0, col].add_patch(rect)
            axes[0, col].set_title(f"Rank {j+1} | {img_class}\nSim: {sim_score:.3f}", 
                                   fontsize=14, color=color, weight='bold')
            
            # Row 2: Overlay
            axes[1, col].imshow(pil_img)
            axes[1, col].imshow(heatmap_img, cmap='jet', alpha=0.5)
            axes[1, col].axis('off')

        plt.tight_layout()
        save_name = os.path.join(args.output_dir, f"SARCI_Overlay_Retrieval_{q_cls}.png")
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
        plt.close()
        
    print(f"\n✅ All overlay retrievals saved to the '{args.output_dir}' directory.")

if __name__ == "__main__":
    main()