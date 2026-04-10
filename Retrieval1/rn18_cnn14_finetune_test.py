import os
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Import shared data logic from your modular file
from datasets import ValVisionDataset, ValAudioDataset, vision_transform

def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    # Point this to your fully trained fine-tuned weights
    parser.add_argument("--model_path", type=str, default="best_cross_modal_finetune.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Audioset Repo Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(CURRENT_DIR, "audioset_tagging_cnn")
sys.path.append(REPO_PATH)
sys.path.append(os.path.join(REPO_PATH, 'pytorch'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print(f"Error loading Cnn14. Ensure '{REPO_PATH}' contains the model definitions.")
    sys.exit()

# ==========================================
# FINETUNED ARCHITECTURES
# ==========================================
class FinetuneVisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)

class FinetuneAudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, 
                                mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        output_dict = self.base_model(x)
        out = self.head(output_dict['embedding'])
        return F.normalize(out, p=2, dim=1)

class FinetuneCrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_model = FinetuneVisionEncoder(embedding_dim)
        self.audio_model = FinetuneAudioEncoder(embedding_dim)

# ==========================================
# EVALUATION SCRIPT
# ==========================================
def main():
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    print(f"Loading Model from: {args.model_path}")
    model = FinetuneCrossModalModel().to(DEVICE)
    
    if os.path.exists(args.model_path):
        # weights_only=False required if the original save included legacy structures
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=False))
        print("Model weights loaded successfully.")
    else:
        print(f"CRITICAL ERROR: Could not find saved weights at {args.model_path}")
        print("Please run train_finetune.py first to generate the model.")
        sys.exit()

    model.eval()
    
    v_ds = ValVisionDataset(VAL_V, transform=vision_transform)
    a_ds = ValAudioDataset(VAL_A)
    
    v_loader = DataLoader(v_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    a_loader = DataLoader(a_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    
    v_feats, v_labels = [], []
    a_feats, a_labels = [], []
    
    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        for imgs, lbls in tqdm(v_loader, desc="Extracting Vision Features"):
            v_feats.append(model.vision_model(imgs.to(DEVICE)).cpu().numpy())
            v_labels.extend(lbls.numpy())
            
        for auds, lbls in tqdm(a_loader, desc="Extracting Audio Features "):
            a_feats.append(model.audio_model(auds.to(DEVICE)).cpu().numpy())
            a_labels.extend(lbls.numpy())
            
    v_feats = np.vstack(v_feats)
    a_feats = np.vstack(a_feats)
    v_labels = np.array(v_labels)
    a_labels = np.array(a_labels)
    
    print("Computing Cosine Similarity Matrix...")
    sim_matrix = cosine_similarity(v_feats, a_feats)
    
    def calculate_r_at_k(matrix, q_labels, g_labels, k_values=[1, 5, 10]):
        scores = {k: 0 for k in k_values}
        for idx in range(len(q_labels)):
            sorted_indices = np.argsort(matrix[idx])[::-1]
            top_labels = g_labels[sorted_indices[:10]]
            for k in k_values:
                if q_labels[idx] in top_labels[:k]: scores[k] += 1
        return {k: (v / len(q_labels)) * 100 for k, v in scores.items()}

    i2a = calculate_r_at_k(sim_matrix, v_labels, a_labels)
    a2i = calculate_r_at_k(sim_matrix.T, a_labels, v_labels)
    
    print("\n==========================================")
    print("FINAL FINE-TUNED RESULTS:")
    print("==========================================")
    print(f"Image-to-Audio Retrieval:")
    print(f"  R@1:  {i2a[1]:.2f}%")
    print(f"  R@5:  {i2a[5]:.2f}%")
    print(f"  R@10: {i2a[10]:.2f}%")
    print("-" * 42)
    print(f"Audio-to-Image Retrieval:")
    print(f"  R@1:  {a2i[1]:.2f}%")
    print(f"  R@5:  {a2i[5]:.2f}%")
    print(f"  R@10: {a2i[10]:.2f}%")
    print("==========================================\n")

if __name__ == "__main__":
    main()