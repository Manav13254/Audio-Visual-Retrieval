import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import glob
import argparse
import sys
import numpy as np

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_pvse_model.pth")
    # PVSE K Parameter (Number of polysemous embeddings)
    parser.add_argument("--K", type=int, default=2, help="Number of polysemous embeddings per instance (e.g., 1 or 2)")
    # Diversity Loss Weight
    parser.add_argument("--lambda_div", type=float, default=0.1, help="Weight for the diversity regularization loss")
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. PIE-NET (Polysemous Instance Embedding)
# ==========================================

class MultiHeadSelfAttention(nn.Module):
    """Self-attention module to generate K different contextual embeddings"""
    def __init__(self, n_head, d_in, d_hidden):
        super().__init__()
        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):
        # x: (Batch, SeqLen, d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        attn = self.softmax(attn) # (Batch, SeqLen, K)
        
        # Aggregate local features based on K attention heads
        output = torch.bmm(attn.transpose(1,2), x) # -> (Batch, K, d_feat)
        
        # RETURN BOTH OUTPUT AND ATTN
        return output, attn

class PIENet(nn.Module):
    """Combines Global features with K Local context features"""
    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.1):
        super().__init__()
        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x):
        # out: Global feature (Batch, d_out)
        # x: Local features (Batch, SeqLen, d_in)
        
        residual, attn = self.attention(x) # (Batch, K, d_in) AND (Batch, SeqLen, K)
        residual = self.dropout(self.sigmoid(self.fc(residual))) # (Batch, K, d_out)
        
        # Expand Global feature to K dimensions
        if self.num_embeds > 1:
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
        else:
            out = out.unsqueeze(1)
            
        # Combine and Normalize
        out = self.layer_norm(out + residual)
        return out, attn # (Batch, K, d_out), (Batch, SeqLen, K)

# ==========================================
# 2. ENCODERS & MODEL ARCHITECTURE
# ==========================================

class VisionEncoderPVSE(nn.Module):
    def __init__(self, embedding_dim=128, K=2):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # We need the 7x7 feature map before pooling for local features
        self.stem = nn.Sequential(self.cnn.conv1, self.cnn.bn1, self.cnn.relu, self.cnn.maxpool)
        self.layer1 = self.cnn.layer1
        self.layer2 = self.cnn.layer2
        self.layer3 = self.cnn.layer3
        self.layer4 = self.cnn.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)
        
        # PIE-Net takes 512-d local features and 128-d global features
        self.pie_net = PIENet(n_embeds=K, d_in=512, d_out=embedding_dim, d_h=256)

    def forward(self, x):
        # 1. Extract features manually to get the pre-pool 7x7 grid
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_7x7 = self.layer4(x) # Shape: (B, 512, 7, 7)
        
        # 2. Global Feature (Average Pool -> FC)
        global_out = self.avgpool(x_7x7).view(-1, 512)
        global_out = self.fc(global_out) # Shape: (B, 128)
        
        # 3. Local Features (Flatten 7x7 grid)
        x_local = x_7x7.view(x_7x7.size(0), 512, -1).transpose(1, 2) # Shape: (B, 49, 512)
        
        # 4. Generate K Polysemous Embeddings & Attention Maps
        out, attn = self.pie_net(global_out, x_local)
        return F.normalize(out, p=2, dim=2), attn # Shape: (B, K, 128), (B, SeqLen, K)

class AudioEncoderPVSE(nn.Module):
    def __init__(self, embedding_dim=128, K=2):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.fc = nn.Linear(2048, embedding_dim)
        
        # PIE-Net takes 2048-d local temporal features
        self.pie_net = PIENet(n_embeds=K, d_in=2048, d_out=embedding_dim, d_h=1024)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        
        # Extract features layer by layer to get the temporal map
        x = self.base_model.spectrogram_extractor(x)
        x = self.base_model.logmel_extractor(x)
        x = x.transpose(1, 3); x = self.base_model.bn0(x); x = x.transpose(1, 3)
        
        x = self.base_model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        
        # Shape is (B, 2048, T, F). We mean over frequency F to get a Temporal sequence
        x_temporal = torch.mean(x, dim=3) # Shape: (B, 2048, T)
        
        # 1. Global Feature
        global_out = torch.mean(x_temporal, dim=2) # Shape: (B, 2048)
        global_out = self.fc(global_out) # Shape: (B, 128)
        
        # 2. Local Features (Temporal sequence)
        x_local = x_temporal.transpose(1, 2) # Shape: (B, T, 2048)
        
        # 3. Generate K Polysemous Embeddings & Attention Maps
        out, attn = self.pie_net(global_out, x_local)
        return F.normalize(out, p=2, dim=2), attn # Shape: (B, K, 128), (B, SeqLen, K)

class PVSECrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128, K=2):
        super().__init__()
        self.vision_model = VisionEncoderPVSE(embedding_dim, K)
        self.audio_model = AudioEncoderPVSE(embedding_dim, K)

# ==========================================
# 3. VSE++ LOSS & DATASETS (MIL ADAPTED)
# ==========================================

class VSELoss(nn.Module):
    def __init__(self, margin=0.2, max_violation=True):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores, labels):
        # scores is already the MIL computed N x N similarity matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        if torch.cuda.is_available(): mask = mask.to(DEVICE)
            
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BidirectionalDataset(Dataset):
    def __init__(self, vision_dir, audio_dir, transform=None):
        self.vision_dir = vision_dir; self.audio_dir = audio_dir; self.transform = transform
        v_classes = set(os.listdir(vision_dir)); a_classes = set(os.listdir(audio_dir))
        self.classes = sorted(list(v_classes.intersection(a_classes)))
        
        self.class_to_imgs = {c: glob.glob(os.path.join(vision_dir, c, "*.*")) for c in self.classes}
        self.class_to_auds = {c: glob.glob(os.path.join(audio_dir, c, "*.wav")) for c in self.classes}

        self.samples = []
        for cls_idx, cls_name in enumerate(self.classes):
            for img_path in self.class_to_imgs[cls_name]:
                self.samples.append((img_path, cls_name, cls_idx))

    def __len__(self): return len(self.samples)

    def _process_audio(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if sr != 32000: wav = T.Resample(sr, 32000)(wav)
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            return wav
        except: return torch.zeros(1, 320000)

    def __getitem__(self, index):
        img_path, cls_name, label = self.samples[index]
        aud_pos_path = random.choice(self.class_to_auds[cls_name])
        
        img_anchor = Image.open(img_path).convert('RGB')
        if self.transform: img_anchor = self.transform(img_anchor)
        aud_pos = self._process_audio(aud_pos_path)
        
        return img_anchor, aud_pos, label

class ValVisionDataset(Dataset):
    def __init__(self, v_dir, transform=None):
        self.files, self.labels = [], []
        classes = sorted(os.listdir(v_dir))
        for i, cls in enumerate(classes):
            for f in glob.glob(os.path.join(v_dir, cls, "*.*")):
                self.files.append(f)
                self.labels.append(i)
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img) if self.transform else img, self.labels[idx]

class ValAudioDataset(Dataset):
    def __init__(self, a_dir):
        self.files, self.labels = [], []
        classes = sorted(os.listdir(a_dir))
        for i, cls in enumerate(classes):
            for f in glob.glob(os.path.join(a_dir, cls, "*.wav")):
                self.files.append(f)
                self.labels.append(i)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            wav, sr = torchaudio.load(self.files[idx])
            if sr != 32000: wav = T.Resample(sr, 32000)(wav)
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            return wav.squeeze(0), self.labels[idx]
        except: return torch.zeros(320000), self.labels[idx]


# ==========================================
# 4. FAST MIL VALIDATION LOOP
# ==========================================

def compute_metrics(model, v_dir, a_dir, k_values=[1, 5, 10], batch_size=128):
    model.eval()
    v_ds = ValVisionDataset(v_dir, transform=vision_transform)
    a_ds = ValAudioDataset(a_dir)
    
    v_loader = DataLoader(v_ds, batch_size=batch_size, num_workers=4)
    a_loader = DataLoader(a_ds, batch_size=batch_size, num_workers=4)
    
    v_base_feats, v_labels = [], []
    a_base_feats, a_labels = [], []
    
    print(f"  -> Extracting PVSE Features (K={args.K})...")
    with torch.no_grad():
        for imgs, lbls in tqdm(v_loader, leave=False):
            # Unpack to ignore attention maps during evaluation
            emb, _ = model.vision_model(imgs.to(DEVICE))
            v_base_feats.append(emb.cpu())
            v_labels.extend(lbls.numpy())
            
        for auds, lbls in tqdm(a_loader, leave=False):
            # Unpack to ignore attention maps during evaluation
            emb, _ = model.audio_model(auds.to(DEVICE))
            a_base_feats.append(emb.cpu())
            a_labels.extend(lbls.numpy())
            
    v_base_feats = torch.cat(v_base_feats, dim=0) # (N, K, 128)
    a_base_feats = torch.cat(a_base_feats, dim=0) # (M, K, 128)
    v_labels = np.array(v_labels)
    a_labels = np.array(a_labels)
    
    print("  -> Computing MIL Similarities...")
    N, K, D = v_base_feats.shape
    M = a_base_feats.shape[0]

    # Efficiently compute all K x K similarities and take the Maximum (MIL)
    v_flat = v_base_feats.view(N * K, D)
    a_flat = a_base_feats.view(M * K, D)
    
    sim_flat = (v_flat @ a_flat.T) # (N*K) x (M*K)
    sim_matrix = sim_flat.view(N, K, M, K)
    
    # Max pooling over the K embedding dimensions for Image and Audio
    sim_matrix = sim_matrix.max(dim=1)[0].max(dim=2)[0].numpy() # (N, M)
            
    def calc_r(matrix, q_lbl, g_lbl):
        res = {k: 0 for k in k_values}
        for idx in range(len(q_lbl)):
            indices = np.argsort(matrix[idx])[::-1]
            top = g_lbl[indices[:10]]
            for k in k_values:
                if q_lbl[idx] in top[:k]: res[k] += 1
        return {k: (v/len(q_lbl))*100 for k,v in res.items()}

    i2a = calc_r(sim_matrix, v_labels, a_labels)
    a2i = calc_r(sim_matrix.T, a_labels, v_labels)
    return i2a, a2i

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
def main():
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = PVSECrossModalModel(K=args.K).to(DEVICE)
    
    if os.path.exists(args.audio_weights):
        print(f"Loading AudioSet Weights: {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.audio_model.base_model.load_state_dict(state_dict, strict=False)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = VSELoss(margin=0.2, max_violation=True)
    
    train_ds = BidirectionalDataset(TRAIN_V, TRAIN_A, transform=vision_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    
    best_avg_r5 = 0.0
    max_epochs = 40
    patience = 10
    epochs_no_improve = 0
    
    print(f"--- Training PVSE Baseline (K={args.K}) with Diversity Loss ---")
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for img, aud, labels in pbar:
            img = img.to(DEVICE)
            aud = aud.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. Polysemous Feature Extraction (Outputs B x K x 128 AND Attention Maps)
            v_embed, v_attn = model.vision_model(img)
            a_embed, a_attn = model.audio_model(aud)
            
            # 2. Multiple Instance Learning (MIL) Similarity Computation
            B, K, D = v_embed.shape
            v_flat = v_embed.view(B * K, D)
            a_flat = a_embed.view(B * K, D)
            
            # Compute all pairwise combinations in the batch
            sim_flat = v_flat @ a_flat.T
            sim_matrix_full = sim_flat.view(B, K, B, K)
            
            # Take the max similarity across the K embeddings
            sim_matrix = sim_matrix_full.max(dim=1)[0].max(dim=2)[0]
            
            # 3. VSE++ Loss
            vse_loss = criterion(sim_matrix, labels)
            
            # 4. Diversity Loss (Orthogonal Regularization)
            if args.K > 1:
                # Identity matrix of size K x K for the batch
                identity = torch.eye(K).unsqueeze(0).expand(B, K, K).to(DEVICE)
                
                # Image Diversity: penalize overlap between attention maps
                v_attn_t = v_attn.transpose(1, 2) # (B, K, SeqLen)
                v_overlap = torch.bmm(v_attn_t, v_attn) # (B, K, K)
                v_div_loss = torch.norm(v_overlap - identity, p='fro', dim=(1,2)).mean()
                
                # Audio Diversity
                a_attn_t = a_attn.transpose(1, 2)
                a_overlap = torch.bmm(a_attn_t, a_attn)
                a_div_loss = torch.norm(a_overlap - identity, p='fro', dim=(1,2)).mean()
                
                total_loss = vse_loss + args.lambda_div * (v_div_loss + a_div_loss)
            else:
                total_loss = vse_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())
            
        print("Validating...")
        i2a, a2i = compute_metrics(model, VAL_V, VAL_A, batch_size=128)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f}")
        print(f"  I->A: R@1 {i2a[1]:.1f}% | R@5 {i2a[5]:.1f}% | R@10 {i2a[10]:.1f}%")
        print(f"  A->I: R@1 {a2i[1]:.1f}% | R@5 {a2i[5]:.1f}% | R@10 {a2i[10]:.1f}%")
        
        # --- Save current epoch ---
        epoch_save_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_save_path)
        
        # --- Early Stopping Logic ---
        if avg_r5 > best_avg_r5:
            best_avg_r5 = avg_r5
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  🔥 New best PVSE model found! Saved to '{args.save_path}'")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"  🛑 Early stopping triggered! Training stopped after {epoch+1} epochs.")
            break

if __name__ == "__main__":
    main()