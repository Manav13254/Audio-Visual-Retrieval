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
from torchvision.models import resnet18
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
    # Ensure this points to the exact VSE++ model weights you trained
    parser.add_argument("--model_path", type=str, default="best_vse_baseline.pth")
    parser.add_argument("--output_dir", type=str, default="qualitative_results")
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.output_dir, exist_ok=True)

# --- Audioset Repo Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(CURRENT_DIR, "audioset_tagging_cnn")
sys.path.append(REPO_PATH)
sys.path.append(os.path.join(REPO_PATH, 'utils'))
sys.path.append(os.path.join(REPO_PATH, 'pytorch'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print("Error loading Cnn14. Ensure the audioset repository is correctly cloned.")
    sys.exit()


# ==========================================
# 1. VSE++ MODEL ARCHITECTURE
# ==========================================
class VisionEncoderBaseline(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.cnn = resnet18(weights=None) 
        self.cnn.fc = nn.Identity() 
        self.project = nn.Linear(512, embedding_dim)

    def forward(self, x):
        features = self.cnn(x)
        out = self.project(features)
        return F.normalize(out, p=2, dim=1)

class AudioEncoderBaseline(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.project = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        output_dict = self.base_model(x)
        features = output_dict['embedding']
        out = self.project(features)
        return F.normalize(out, p=2, dim=1)

class BaselineCrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_model = VisionEncoderBaseline(embedding_dim)
        self.audio_model = AudioEncoderBaseline(embedding_dim)


# ==========================================
# 2. QUALITATIVE INFERENCE LOGIC
# ==========================================
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")
    
    # ----------------------------------------
    # A. INSTANTIATE VSE++ MODEL
    # ----------------------------------------
    print(f"Loading VSE++ Model from {args.model_path}...")
    model = BaselineCrossModalModel().to(DEVICE)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    else:
        print(f"ERROR: {args.model_path} not found. Train the model first.")
        sys.exit()
    model.eval()

    # ----------------------------------------
    # B. SET SPECIFIC TARGET CLASSES
    # ----------------------------------------
    all_classes = sorted(os.listdir(VAL_V))
    
    # Manually defined target classes as requested
    target_classes = ["airport", "beach", "bridge", "grassland", "residential", "sports land"]
    print(f"Target Classes for Comparison: {target_classes}")

    # ----------------------------------------
    # C. LOAD AUDIO QUERIES
    # ----------------------------------------
    queries = {}
    
    # FIXED SEED: Guarantees the same random audio clip selection
    random.seed(100) 
    
    for cls in target_classes:
        if cls not in all_classes:
            print(f"Warning: Class '{cls}' not found in directory.")
            continue
            
        a_files = sorted(glob.glob(os.path.join(VAL_A, cls, "*.wav")))
        if a_files:
            wav_path = random.choice(a_files) 
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
            
            v_base = model.vision_model(imgs)
            gallery_v_base.append(v_base.cpu())
            
    gallery_v_base = torch.cat(gallery_v_base, dim=0)

    # ----------------------------------------
    # E. RETRIEVAL & PLOTTING
    # ----------------------------------------
    for q_cls, wav_tensor in queries.items():
        print(f"Evaluating Query: {q_cls}...")
        wav_tensor = wav_tensor.to(DEVICE)
        
        with torch.no_grad():
            a_base = model.audio_model(wav_tensor).cpu()
            
            # VSE++ Similarity Math: Simple Dot Product of L2 Normalized Features
            sims = (gallery_v_base @ a_base.T).squeeze(1).numpy()
                
        # Get Top 5 Indices
        top5_indices = np.argsort(sims)[::-1][:5]
        
        # Plotting
        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        
        # Query Audio Description Box
        axes[0].text(0.5, 0.6, "QUERY AUDIO", fontsize=18, ha='center', weight='bold')
        axes[0].text(0.5, 0.4, f"Class:\n{q_cls}", fontsize=16, ha='center', color='darkblue')
        axes[0].axis('off')
        
        # Top 5 Images
        for j, idx in enumerate(top5_indices):
            ax = axes[j+1]
            img = Image.open(gallery_paths[idx]).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            
            img_class = gallery_labels[idx]
            sim_score = sims[idx]
            
            # Green border if correct, Red if wrong
            color = 'limegreen' if img_class == q_cls else 'red'
            rect = patches.Rectangle((0,0), img.size[0], img.size[1], 
                                     linewidth=12, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Subtitle with Rank and Score
            ax.set_title(f"Rank {j+1} | {img_class}\nSim: {sim_score:.3f}", 
                         fontsize=14, color=color, weight='bold')
                         
        plt.tight_layout()
        save_name = os.path.join(args.output_dir, f"VSE_query_{q_cls}.png")
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {save_name}")

if __name__ == "__main__":
    main()