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

# Import shared modules
from datasets import ValVisionDataset, ValAudioDataset, vision_transform
from quaternion_attn import QUATER_ATTENTION
from ca_attn import CA_Block
from iclm import ICLM

def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--model_path", type=str, default="best_iclm_model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# ARCHITECTURES
# ==========================================
class IclmVisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

    def forward(self, x):
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
        return F.normalize(self.project(out), p=2, dim=1)

class IclmAudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.ca_block = CA_Block(2048)
        self.attn_pool = nn.Sequential(
            nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1)
        )
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

class IclmCrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_model = IclmVisionEncoder(embedding_dim)
        self.audio_model = IclmAudioEncoder(embedding_dim)
        self.iclm = ICLM(embed_dim=embedding_dim)

# ==========================================
# EVALUATION SCRIPT
# ==========================================
def main():
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    print(f"Loading Model from: {args.model_path}")
    model = IclmCrossModalModel().to(DEVICE)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=False))
        print("Model weights loaded successfully.")
    else:
        print(f"CRITICAL ERROR: Could not find saved weights at {args.model_path}")
        print("Please run train_iclm.py first to generate the model.")
        sys.exit()

    model.eval()
    
    v_ds = ValVisionDataset(VAL_V, transform=vision_transform)
    a_ds = ValAudioDataset(VAL_A)
    
    v_loader = DataLoader(v_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    a_loader = DataLoader(a_ds, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    
    v_base_feats, v_labels = [], []
    a_base_feats, a_labels = [], []
    
    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        for imgs, lbls in tqdm(v_loader, desc="Extracting Vision Features"):
            v_base_feats.append(model.vision_model(imgs.to(DEVICE)).cpu())
            v_labels.extend(lbls.numpy())
            
        for auds, lbls in tqdm(a_loader, desc="Extracting Audio Features "):
            a_base_feats.append(model.audio_model(auds.to(DEVICE)).cpu())
            a_labels.extend(lbls.numpy())
            
    v_base_feats = torch.cat(v_base_feats, dim=0)
    a_base_feats = torch.cat(a_base_feats, dim=0)
    v_labels = np.array(v_labels)
    a_labels = np.array(a_labels)
    
    N, M = v_base_feats.size(0), a_base_feats.size(0)
    sim_matrix = np.zeros((N, M))
    
    print("Applying Cross-Modal ICLM Interactions...")
    with torch.no_grad():
        for i in tqdm(range(N), leave=False):
            v_row = v_base_feats[i].unsqueeze(0).to(DEVICE)
            
            for j in range(0, M, args.batch_size):
                a_chunk = a_base_feats[j:j+args.batch_size].to(DEVICE)
                v_chunk = v_row.expand(a_chunk.size(0), -1)
                
                v_final, a_final = model.iclm(v_chunk, a_chunk)
                sims = F.cosine_similarity(v_final, a_final, dim=1).cpu().numpy()
                sim_matrix[i, j:j+args.batch_size] = sims
    
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
    print("FINAL ICLM MODEL RESULTS:")
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