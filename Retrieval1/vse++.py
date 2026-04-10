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
    parser.add_argument("--save_path", type=str, default="best_vse_baseline.pth")
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
# 1. PURE BASELINE ENCODERS (No Custom Attention)
# ==========================================

class VisionEncoderBaseline(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze Backbone
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # Replace the final classification layer with an Identity layer
        self.cnn.fc = nn.Identity() 
        
        # ResNet18 outputs 512-dim features
        self.project = nn.Linear(512, embedding_dim)

    def forward(self, x):
        features = self.cnn(x)
        out = self.project(features)
        return F.normalize(out, p=2, dim=1)

class AudioEncoderBaseline(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        # Freeze Backbone
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Cnn14 outputs a 2048-dim feature vector named 'embedding'
        self.project = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        
        # Standard Cnn14 forward pass
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
# 2. VSE++ LOSS & DATASETS
# ==========================================

class VSELoss(nn.Module):
    def __init__(self, margin=0.2, max_violation=True):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores, labels):
        # scores is an N x N matrix of cosine similarities
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # Mask: Zero out loss for intra-class items to prevent false penalties
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        if torch.cuda.is_available(): 
            mask = mask.to(DEVICE)
            
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        # Hardest Negative Mining
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
            # Ensure proper shape comparison (wav.shape[1])
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
            # Ensure proper shape comparison (wav.shape[1])
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            return wav.squeeze(0), self.labels[idx]
        except: return torch.zeros(320000), self.labels[idx]


# ==========================================
# 3. FAST VALIDATION LOOP
# ==========================================

def compute_metrics(model, v_dir, a_dir, k_values=[1, 5, 10], batch_size=128):
    model.eval()
    v_ds = ValVisionDataset(v_dir, transform=vision_transform)
    a_ds = ValAudioDataset(a_dir)
    
    v_loader = DataLoader(v_ds, batch_size=batch_size, num_workers=4)
    a_loader = DataLoader(a_ds, batch_size=batch_size, num_workers=4)
    
    v_base_feats, v_labels = [], []
    a_base_feats, a_labels = [], []
    
    print("  -> Extracting Features...")
    with torch.no_grad():
        for imgs, lbls in tqdm(v_loader, leave=False):
            v_base_feats.append(model.vision_model(imgs.to(DEVICE)).cpu())
            v_labels.extend(lbls.numpy())
            
        for auds, lbls in tqdm(a_loader, leave=False):
            a_base_feats.append(model.audio_model(auds.to(DEVICE)).cpu())
            a_labels.extend(lbls.numpy())
            
    v_base_feats = torch.cat(v_base_feats, dim=0) 
    a_base_feats = torch.cat(a_base_feats, dim=0) 
    v_labels = np.array(v_labels)
    a_labels = np.array(a_labels)
    
    print("  -> Computing Similarities...")
    # Because both vectors are L2 normalized, cosine similarity is just the dot product
    sim_matrix = (v_base_feats @ a_base_feats.T).numpy()
            
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
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = BaselineCrossModalModel().to(DEVICE)
    
    if os.path.exists(args.audio_weights):
        print(f"Loading AudioSet Weights: {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.audio_model.base_model.load_state_dict(state_dict, strict=False)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    # Cosine Triplet Loss initialized with Max Violation (Hard Negatives)
    criterion = VSELoss(margin=0.2, max_violation=True)
    
    train_ds = BidirectionalDataset(TRAIN_V, TRAIN_A, transform=vision_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    
    best_avg_r5 = 0.0
    
    # --- Early Stopping Setup ---
    max_epochs = 40
    patience = 10
    epochs_no_improve = 0
    
    print(f"--- Training Pure VSE++ Baseline ---")
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for img, aud, labels in pbar:
            img = img.to(DEVICE)
            aud = aud.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. Base Feature Extraction
            v_embed = model.vision_model(img)
            a_embed = model.audio_model(aud)
            
            # 2. Compute N x N Cosine Similarities (Dot product of L2 normalized features)
            sim_matrix = v_embed @ a_embed.T
            
            # 3. VSE++ Loss
            loss = criterion(sim_matrix, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print("Validating...")
        i2a, a2i = compute_metrics(model, VAL_V, VAL_A, batch_size=128)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f}")
        print(f"  I->A: R@1 {i2a[1]:.1f}% | R@5 {i2a[5]:.1f}% | R@10 {i2a[10]:.1f}%")
        print(f"  A->I: R@1 {a2i[1]:.1f}% | R@5 {a2i[5]:.1f}% | R@10 {a2i[10]:.1f}%")
        
        # --- Save current epoch ---
        epoch_save_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_save_path)
        
        # --- Early Stopping & Best Model Logic ---
        if avg_r5 > best_avg_r5:
            best_avg_r5 = avg_r5
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  🔥 New best baseline found! Saved to '{args.save_path}'")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"  🛑 Early stopping triggered! Training stopped after {epoch+1} epochs.")
            break

if __name__ == "__main__":
    main()