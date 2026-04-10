
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
import torchvision.transforms as transforms
import torchaudio
import torchaudio.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    # Ensure this points to your trained PVSE weights
    parser.add_argument("--model_path", type=str, default="best_pvse_k2.pth")
    parser.add_argument("--output_dir", type=str, default="qualitative_results_pvse")
    parser.add_argument("--K", type=int, default=2, help="Must match training K")
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
    print("Error loading Cnn14. Ensure audioset_tagging_cnn is in path.")
    sys.exit()

# ==========================================
# 1. PVSE ARCHITECTURE (Must match training)
# ==========================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_in, d_hidden):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn = self.w_2(self.tanh(self.w_1(x)))
        attn = self.softmax(attn) 
        output = torch.bmm(attn.transpose(1,2), x) 
        return output, attn

class PIENet(nn.Module):
    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.1):
        super().__init__()
        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, out, x):
        residual, attn = self.attention(x)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.num_embeds > 1:
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
        else:
            out = out.unsqueeze(1)
        out = self.layer_norm(out + residual)
        return out, attn

class VisionEncoderPVSE(nn.Module):
    def __init__(self, embedding_dim=128, K=2):
        super().__init__()
        resnet = resnet18(weights=None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)
        self.pie_net = PIENet(n_embeds=K, d_in=512, d_out=embedding_dim, d_h=256)

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x_7x7 = self.layer4(x)
        global_out = self.fc(self.avgpool(x_7x7).view(-1, 512))
        x_local = x_7x7.view(x_7x7.size(0), 512, -1).transpose(1, 2)
        out, attn = self.pie_net(global_out, x_local)
        return F.normalize(out, p=2, dim=2), attn

class AudioEncoderPVSE(nn.Module):
    def __init__(self, embedding_dim=128, K=2):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.fc = nn.Linear(2048, embedding_dim)
        self.pie_net = PIENet(n_embeds=K, d_in=2048, d_out=embedding_dim, d_h=1024)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        x = self.base_model.spectrogram_extractor(x)
        x = self.base_model.logmel_extractor(x)
        x = x.transpose(1, 3); x = self.base_model.bn0(x); x = x.transpose(1, 3)
        x = self.base_model.conv_block1(x, (2,2), 'avg'); x = self.base_model.conv_block2(x, (2,2), 'avg')
        x = self.base_model.conv_block3(x, (2,2), 'avg'); x = self.base_model.conv_block4(x, (2,2), 'avg')
        x = self.base_model.conv_block5(x, (2,2), 'avg'); x = self.base_model.conv_block6(x, (1,1), 'avg')
        x_temporal = torch.mean(x, dim=3)
        global_out = self.fc(torch.mean(x_temporal, dim=2))
        x_local = x_temporal.transpose(1, 2)
        out, attn = self.pie_net(global_out, x_local)
        return F.normalize(out, p=2, dim=2), attn

class PVSECrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128, K=2):
        super().__init__()
        self.vision_model = VisionEncoderPVSE(embedding_dim, K)
        self.audio_model = AudioEncoderPVSE(embedding_dim, K)

# ==========================================
# 2. RETRIEVAL LOGIC
# ==========================================
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    print(f"Loading PVSE Model (K={args.K}) from {args.model_path}...")
    model = PVSECrossModalModel(K=args.K).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    all_classes = sorted(os.listdir(VAL_V))
    target_classes = all_classes
    
    # ----------------------------------------
    # STEP 1: PRE-COMPUTE FULL GALLERY
    # ----------------------------------------
    gallery_paths, gallery_labels = [], []
    for cls in all_classes:
        v_files = sorted(glob.glob(os.path.join(VAL_V, cls, "*.*")))
        gallery_paths.extend(v_files)
        gallery_labels.extend([cls] * len(v_files))

    gallery_feats = [] # Will store (N, K, 128)
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, len(gallery_paths), batch_size), desc="Extracting Image Gallery"):
            paths = gallery_paths[i:i+batch_size]
            imgs = [vision_transform(Image.open(p).convert('RGB')) for p in paths]
            # Unpack: PVSE returns (Embeddings, Attention)
            emb, _ = model.vision_model(torch.stack(imgs).to(DEVICE))
            gallery_feats.append(emb.cpu())
            
    gallery_feats = torch.cat(gallery_feats, dim=0) # (N, K, 128)

    # ----------------------------------------
    # STEP 2: PROCESS QUERIES
    # ----------------------------------------
    random.seed(100) # Sync query selection across experiments
    
    for q_cls in target_classes:
        a_files = sorted(glob.glob(os.path.join(VAL_A, q_cls, "*.wav")))
        if not a_files: continue
        
        wav_path = random.choice(a_files)
        wav, sr = torchaudio.load(wav_path) 
        if sr != 32000: wav = T.Resample(sr, 32000)(wav)
        if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
        else: wav = wav[:, :320000]
        if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
        
        with torch.no_grad():
            # Query Embedding: (1, K, 128)
            q_emb, _ = model.audio_model(wav.to(DEVICE))
            q_emb = q_emb.cpu()
            
            # MIL Similarity: Max similarity over all possible K x K combinations
            # q_emb: (1, K, 128), gallery: (N, K, 128)
            # Dot product -> (N, K, K)
            sim_blocks = torch.einsum('ikd,njd->nkj', q_emb, gallery_feats)
            # Max pooling over both embedding dimensions
            sims = sim_blocks.max(dim=1)[0].max(dim=1)[0].numpy()
            
            top5_indices = np.argsort(sims)[::-1][:5]
            
        # ----------------------------------------
        # STEP 3: PLOTTING
        # ----------------------------------------
        fig, axes = plt.subplots(1, 6, figsize=(24, 5))
        axes[0].text(0.5, 0.6, "QUERY AUDIO", fontsize=18, ha='center', weight='bold')
        axes[0].text(0.5, 0.4, f"Class: {q_cls}\n(PVSE K={args.K})", fontsize=14, ha='center', color='darkblue')
        axes[0].axis('off')
        
        for j, idx in enumerate(top5_indices):
            ax = axes[j+1]
            img = Image.open(gallery_paths[idx]).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            
            img_class = gallery_labels[idx]
            color = 'limegreen' if img_class == q_cls else 'red'
            rect = patches.Rectangle((0,0), img.size[0], img.size[1], 
                                     linewidth=14, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f"Rank {j+1}\n{img_class}\nSim: {sims[idx]:.3f}", 
                         fontsize=13, color=color, weight='bold')

        plt.tight_layout()
        save_name = os.path.join(args.output_dir, f"PVSE_result_{q_cls}.png")
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
        
    print(f"\n✅ PVSE Retrieval results saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()