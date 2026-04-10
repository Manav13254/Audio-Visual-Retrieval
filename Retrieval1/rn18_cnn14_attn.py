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

# Import shared data logic and attention blocks from your modular files
from datasets import BidirectionalDataset, ValVisionDataset, ValAudioDataset, vision_transform
from quaternion_attn import QUATER_ATTENTION
from ca_attn import CA_Block

def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_attention_encoders.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
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
# ATTENTION ENCODER ARCHITECTURES
# ==========================================
class AttnVisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze Backbone
        for param in resnet.parameters():
            param.requires_grad = False
            
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Fusion Layers 
        self.f0conv = nn.Conv2d(64, 64, 3, 2, 1)
        self.f01conv = nn.Conv2d(128, 128, 7, 4, 3)
        self.f2conv = nn.Conv2d(128, 128, 3, 2, 1)
        self.fusionconv = nn.Conv2d(512, 512, 3, 2, 1)
        
        # Imported Quaternion Attention Modules
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

class AttnAudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        # Freeze Backbone
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Imported Coordinate Attention
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

class AttnCrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_model = AttnVisionEncoder(embedding_dim)
        self.audio_model = AttnAudioEncoder(embedding_dim)


# ==========================================
# EVALUATION ROUTINE
# ==========================================
def compute_attn_metrics(model, v_dir, a_dir, device, k_values=[1, 5, 10], batch_size=64):
    model.eval()
    v_ds = ValVisionDataset(v_dir, transform=vision_transform)
    a_ds = ValAudioDataset(a_dir)
    
    v_loader = DataLoader(v_ds, batch_size=batch_size, num_workers=8, pin_memory=True)
    a_loader = DataLoader(a_ds, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    v_feats, v_labels = [], []
    a_feats, a_labels = [], []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(v_loader, leave=False, desc="Extracting Vision"):
            v_feats.append(model.vision_model(imgs.to(device)).cpu().numpy())
            v_labels.extend(lbls.numpy())
            
        for auds, lbls in tqdm(a_loader, leave=False, desc="Extracting Audio"):
            a_feats.append(model.audio_model(auds.to(device)).cpu().numpy())
            a_labels.extend(lbls.numpy())
            
    v_feats = np.vstack(v_feats)
    a_feats = np.vstack(a_feats)
    v_labels = np.array(v_labels)
    a_labels = np.array(a_labels)
    
    sim_matrix = cosine_similarity(v_feats, a_feats)
    
    def calculate_r_at_k(matrix, q_labels, g_labels):
        scores = {k: 0 for k in k_values}
        for idx in range(len(q_labels)):
            sorted_indices = np.argsort(matrix[idx])[::-1]
            top_labels = g_labels[sorted_indices[:10]]
            for k in k_values:
                if q_labels[idx] in top_labels[:k]: scores[k] += 1
        return {k: (v / len(q_labels)) * 100 for k, v in scores.items()}

    i2a = calculate_r_at_k(sim_matrix, v_labels, a_labels)
    a2i = calculate_r_at_k(sim_matrix.T, a_labels, v_labels)
    
    return i2a, a2i


# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def main():
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = AttnCrossModalModel().to(DEVICE)
    
    if os.path.exists(args.audio_weights):
        print(f"Loading AudioSet Weights: {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.audio_model.base_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"WARNING: Weights not found at {args.audio_weights}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    train_ds = BidirectionalDataset(TRAIN_V, TRAIN_A, transform=vision_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    best_avg_r5 = 0.0
    print(f"Starting Attention Encoders (Backbones Frozen) Training...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for img_anc, img_neg, aud_pos, aud_neg in pbar:
            img_anc, img_neg = img_anc.to(DEVICE), img_neg.to(DEVICE)
            aud_pos, aud_neg = aud_pos.to(DEVICE), aud_neg.to(DEVICE)
            
            optimizer.zero_grad()
            
            v_a = model.vision_model(img_anc)
            v_n = model.vision_model(img_neg)
            a_p = model.audio_model(aud_pos)
            a_n = model.audio_model(aud_neg)
            
            loss = criterion(v_a, a_p, a_n) + criterion(a_p, v_a, v_n)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print("Validating model...")
        i2a, a2i = compute_attn_metrics(model, VAL_V, VAL_A, DEVICE, batch_size=128)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        print(f"Epoch {epoch+1} Results: Loss {train_loss/len(train_loader):.4f}")
        print(f"Image-to-Audio: R@1 {i2a[1]:.1f} | R@5 {i2a[5]:.1f} | R@10 {i2a[10]:.1f}")
        print(f"Audio-to-Image: R@1 {a2i[1]:.1f} | R@5 {a2i[5]:.1f} | R@10 {a2i[10]:.1f}")
        
        if avg_r5 > best_avg_r5:
            best_avg_r5 = avg_r5
            torch.save(model.state_dict(), args.save_path)
            print(f"Validation improved. Model saved to {args.save_path}")

if __name__ == "__main__":
    main()