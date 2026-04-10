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
import torchvision.models as models
from torchvision.models.resnet import resnet18
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
    # Ensure this points to your trained AMFMN model weights
    parser.add_argument("--model_path", type=str, default="best_amfmn_soft.pth")
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
sys.path.append(os.path.join(REPO_PATH, 'utils'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print(f"Error loading Cnn14. Ensure '{REPO_PATH}' contains the model definitions.")
    sys.exit()

# ==========================================
# 1. AMFMN ARCHITECTURE
# ==========================================
def l2norm(X, dim=-1, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ExtractFeature(nn.Module):
    def __init__(self, opt):
        super(ExtractFeature, self).__init__()
        self.embed_dim = opt['embed']['embed_dim']
        self.resnet = resnet18(weights=None)
        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.linear = nn.Linear(in_features=512, out_features=self.embed_dim)

    def forward(self, img):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        f2_up = self.up_sample_2(f2)
        lower_feature = torch.cat([f1, f2_up], dim=1)

        f4_up = self.up_sample_2(f4)
        higher_feature = torch.cat([f3, f4_up], dim=1)

        feature = f4.view(f4.shape[0], 512, -1)
        solo_feature = self.linear(torch.mean(feature, dim=-1))

        return lower_feature, higher_feature, solo_feature

class VSA_Module(nn.Module):
    def __init__(self, opt):
        super(VSA_Module, self).__init__()
        channel_size = opt['multiscale']['multiscale_input_channel']
        out_channels = opt['multiscale']['multiscale_output_channel']
        embed_dim = opt['embed']['embed_dim']

        self.LF_conv = nn.Conv2d(in_channels=192, out_channels=channel_size, kernel_size=3, stride=4)
        self.HF_conv = nn.Conv2d(in_channels=768, out_channels=channel_size, kernel_size=1, stride=1)

        self.conv1x1_1 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)

        self.solo_attention = nn.Linear(in_features=256, out_features=embed_dim)

    def forward(self, lower_feature, higher_feature, solo_feature):
        lower_feature = self.LF_conv(lower_feature)
        higher_feature = self.HF_conv(higher_feature)

        concat_feature = torch.cat([lower_feature, higher_feature], dim=1)
        concat_feature = higher_feature.mean(dim=1, keepdim=True).expand_as(concat_feature) + concat_feature

        main_feature = self.conv1x1_1(concat_feature)
        attn_feature = torch.sigmoid(self.conv1x1_2(concat_feature))
        
        atted_feature = (main_feature * attn_feature).view(concat_feature.shape[0], -1)
        solo_att = torch.sigmoid(self.solo_attention(atted_feature))
        solo_feature = solo_feature * solo_att

        return solo_feature

class EncoderAudio(nn.Module):
    def __init__(self, embed_size):
        super(EncoderAudio, self).__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, 
                                mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.fc = nn.Linear(2048, embed_size)

    def forward(self, audio):
        if audio.dim() == 3: audio = audio.squeeze(1)
        out = self.base_model(audio)['embedding']
        return self.fc(out)

class CrossAttention(nn.Module):
    def __init__(self, opt):
        super(CrossAttention, self).__init__()
        self.att_type = opt['cross_attention']['att_type']
        dim = opt['embed']['embed_dim']

        if self.att_type == "soft_att":
            self.cross_attention = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        elif self.att_type == "fusion_att":
            self.cross_attention_fc1 = nn.Sequential(nn.Linear(2*dim, dim), nn.Sigmoid())
            self.cross_attention_fc2 = nn.Sequential(nn.Linear(2*dim, dim))
            self.cross_attention = lambda x: self.cross_attention_fc1(x) * self.cross_attention_fc2(x)
        elif self.att_type == "similarity_att":
            self.fc_visual = nn.Sequential(nn.Linear(dim, dim))
            self.fc_text = nn.Sequential(nn.Linear(dim, dim))

    def forward(self, visual, text):
        batch_v = visual.shape[0]
        batch_t = text.shape[0]

        if self.att_type == "soft_att":
            visual_gate = self.cross_attention(visual)
            visual_gate = visual_gate.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            return visual_gate * text
            
        elif self.att_type == "fusion_att":
            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            fusion_vec = torch.cat([visual, text], dim=-1)
            return self.cross_attention(fusion_vec)
            
        elif self.att_type == "similarity_att":
            visual = self.fc_visual(visual)
            text = self.fc_text(text)
            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)
            sims = visual * text
            return torch.sigmoid(sims) * text

class ClassAwareContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, max_violation=True):
        super(ClassAwareContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores, labels):
        pass # Not needed for inference

class AMFMN_AudioVisual(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_enc = ExtractFeature(opt)
        self.vsa = VSA_Module(opt)
        self.aud_enc = EncoderAudio(opt['embed']['embed_dim'])
        self.cross_attn = CrossAttention(opt)
        self.criterion = ClassAwareContrastiveLoss(margin=opt['loss']['margin'], max_violation=True)

    def forward_img(self, img):
        lf, hf, sf = self.img_enc(img)
        v_feat = self.vsa(lf, hf, sf)
        return l2norm(v_feat, dim=-1)

    def forward_aud(self, aud):
        a_feat = self.aud_enc(aud)
        return l2norm(a_feat, dim=-1)


# ==========================================
# 2. QUALITATIVE INFERENCE LOGIC
# ==========================================
# CRITICAL: Must be 256x256 for AMFMN spatial math
vision_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    opt = {
        'embed': {'embed_dim': 128},
        'multiscale': {
            'multiscale_input_channel': 64,
            'multiscale_output_channel': 1 
        },
        # Ensure this matches the version of the weights you are loading
        'cross_attention': {'att_type': 'soft_att'}, 
        'loss': {'margin': 0.2}
    }
    
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")
    
    # ----------------------------------------
    # A. INSTANTIATE AMFMN MODEL
    # ----------------------------------------
    print(f"Loading AMFMN Model from {args.model_path}...")
    model = AMFMN_AudioVisual(opt).to(DEVICE)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True))
    else:
        print(f"ERROR: {args.model_path} not found. Train the model first.")
        sys.exit()
    model.eval()

    # ----------------------------------------
    # B. SELECT ALL CLASSES
    # ----------------------------------------
    all_classes = sorted(os.listdir(VAL_V))
    target_classes = all_classes
    print(f"Evaluating all {len(target_classes)} classes...")

    # ----------------------------------------
    # C. LOAD AUDIO QUERIES
    # ----------------------------------------
    queries = {}
    
    # FIXED SEED: Guarantees the exact same random audio is picked across model scripts!
    random.seed(90) 
    
    for cls in target_classes:
        a_files = sorted(glob.glob(os.path.join(VAL_A, cls, "*.wav")))
        if a_files:
            wav_path = random.choice(a_files) # Deterministic random selection
            wav, sr = torchaudio.load(wav_path) 
            if sr != 32000: wav = T.Resample(sr, 32000)(wav)
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            queries[cls] = wav

    # ----------------------------------------
    # D. LOAD IMAGE GALLERY & PRE-COMPUTE BASE
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
        for i in tqdm(range(0, len(gallery_paths), batch_size), desc="Extracting Image Gallery"):
            paths = gallery_paths[i:i+batch_size]
            imgs = [vision_transform(Image.open(p).convert('RGB')) for p in paths]
            imgs = torch.stack(imgs).to(DEVICE)
            
            # Base features processed through VSA module
            v_base = model.forward_img(imgs)
            gallery_v_base.append(v_base.cpu())
            
    gallery_v_base = torch.cat(gallery_v_base, dim=0)

    # ----------------------------------------
    # E. RETRIEVAL & PLOTTING
    # ----------------------------------------
    for q_cls, wav_tensor in queries.items():
        print(f"Generating visual for: {q_cls}...")
        wav_tensor = wav_tensor.to(DEVICE)
        
        with torch.no_grad():
            # Get the single audio query base feature
            a_base = model.forward_aud(wav_tensor).cpu()
            sims = []
            
            # AMFMN Cross Attention Math
            for i in range(0, len(gallery_v_base), batch_size):
                v_chunk = gallery_v_base[i:i+batch_size].to(DEVICE)
                a_chunk = a_base.to(DEVICE)
                
                # Image acts as a gate to guide the audio features
                a_guided = model.cross_attn(v_chunk, a_chunk) 
                
                # Compute Cosine Similarity between visual and guided audio
                batch_sims = F.cosine_similarity(v_chunk.unsqueeze(1), a_guided, dim=-1).squeeze(1)
                sims.extend(batch_sims.cpu().numpy())
                
        # Get Top 5 Indices
        top5_indices = np.argsort(sims)[::-1][:5]
        
        # Plotting
        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        
        axes[0].text(0.5, 0.6, "QUERY AUDIO", fontsize=18, ha='center', weight='bold')
        axes[0].text(0.5, 0.4, f"Class:\n{q_cls}", fontsize=16, ha='center', color='darkblue')
        axes[0].axis('off')
        
        for j, idx in enumerate(top5_indices):
            ax = axes[j+1]
            img = Image.open(gallery_paths[idx]).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            
            img_class = gallery_labels[idx]
            sim_score = sims[idx]
            
            color = 'limegreen' if img_class == q_cls else 'red'
            rect = patches.Rectangle((0,0), img.size[0], img.size[1], 
                                     linewidth=16, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            ax.set_title(f"Rank {j+1} | {img_class}\nSim: {sim_score:.3f}", 
                         fontsize=14, color=color, weight='bold')
                         
        plt.tight_layout()
        save_name = os.path.join(args.output_dir, f"AMFMN_query_{q_cls}.png")
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
        
    print(f"\n✅ All {len(target_classes)} visualizations saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()