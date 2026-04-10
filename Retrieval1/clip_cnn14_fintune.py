import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from tqdm import tqdm
import glob
import argparse
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Install CLIP if not present
try:
    import clip
except ImportError:
    print("Error: CLIP not found. Please install via: pip install git+https://github.com/openai/CLIP.git")
    sys.exit()

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_finetune_clip_cnn14.pth")
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
    print("Error loading Cnn14.")
    sys.exit()

# ==========================================
# 1. MODELS (Unfrozen Last Layers)
# ==========================================

class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        print("Loading CLIP (ViT-B/32)...")
        self.clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        self.clip_model = self.clip_model.float() # Ensure float32 for training stability

        # --- Fine-Tuning Logic ---
        # 1. Freeze everything first
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the last Transformer Block
        # CLIP Visual structure: visual -> transformer -> resblocks (List)
        last_block = self.clip_model.visual.transformer.resblocks[-1]
        for param in last_block.parameters():
            param.requires_grad = True
            
        # 3. Unfreeze final LayerNorm
        for param in self.clip_model.visual.ln_post.parameters():
            param.requires_grad = True
            
        print("✅ CLIP: Unfrozen Last Transformer Block & LayerNorm")

        # Projection Head
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        features = features.float() 
        return F.normalize(self.head(features), p=2, dim=1)

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, 
                                mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        # --- Fine-Tuning Logic ---
        # 1. Freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze Last Conv Block (conv_block6)
        for param in self.base_model.conv_block6.parameters():
            param.requires_grad = True
            
        print("✅ Cnn14: Unfrozen conv_block6")

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        output_dict = self.base_model(x)
        return F.normalize(self.head(output_dict['embedding']), p=2, dim=1)

class CrossModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = VisionEncoder()
        self.audio_model = AudioEncoder()

# ==========================================
# 2. DATASET & TRANSFORMS
# ==========================================
# CLIP Specific Transforms
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

class BidirectionalDataset(Dataset):
    def __init__(self, vision_dir, audio_dir, transform=None):
        self.vision_dir = vision_dir; self.audio_dir = audio_dir; self.transform = transform
        v_classes = set(os.listdir(vision_dir)); a_classes = set(os.listdir(audio_dir))
        self.classes = sorted(list(v_classes.intersection(a_classes)))
        self.class_to_imgs = {c: glob.glob(os.path.join(vision_dir, c, "*.*")) for c in self.classes}
        self.class_to_auds = {c: glob.glob(os.path.join(audio_dir, c, "*.wav")) for c in self.classes}

    def __len__(self): return sum(len(imgs) for imgs in self.class_to_imgs.values())

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
        target_class = random.choice(self.classes)
        neg_class = random.choice([c for c in self.classes if c != target_class])
        
        img_path = random.choice(self.class_to_imgs[target_class])
        aud_pos_path = random.choice(self.class_to_auds[target_class])
        img_neg_path = random.choice(self.class_to_imgs[neg_class])
        aud_neg_path = random.choice(self.class_to_auds[neg_class])
        
        img_anchor = Image.open(img_path).convert('RGB')
        img_neg = Image.open(img_neg_path).convert('RGB')
        
        if self.transform: img_anchor = self.transform(img_anchor); img_neg = self.transform(img_neg)
        aud_pos = self._process_audio(aud_pos_path); aud_neg = self._process_audio(aud_neg_path)
        return img_anchor, img_neg, aud_pos, aud_neg

# ==========================================
# 3. METRICS
# ==========================================
def compute_metrics(model, v_dir, a_dir, k_values=[1, 5, 10]):
    model.eval()
    classes = sorted(os.listdir(v_dir))
    v_feats, a_feats, labels = [], [], []
    
    with torch.no_grad():
        for i, cls in enumerate(classes):
            for f in glob.glob(os.path.join(v_dir, cls, "*.*")):
                try:
                    img = Image.open(f).convert('RGB')
                    img = clip_transform(img).unsqueeze(0).to(DEVICE)
                    v_feats.append(model.vision_model(img).cpu().numpy())
                    labels.append(i)
                except: continue
                
    a_feats_dict = {i: [] for i in range(len(classes))}
    for i, cls in enumerate(classes):
        for f in glob.glob(os.path.join(a_dir, cls, "*.wav")):
            try:
                wav, sr = torchaudio.load(f)
                if sr!=32000: wav=T.Resample(sr,32000)(wav)
                if wav.shape[1]<320000: wav=F.pad(wav,(0,320000-wav.shape[1]))
                else: wav=wav[:,:320000]
                if wav.shape[0]>1: wav=torch.mean(wav,dim=0,keepdim=True)
                wav = wav.to(DEVICE)
                with torch.no_grad():
                    a_feats_dict[i].append(model.audio_model(wav).cpu().numpy())
            except: continue

    final_a_feats, final_a_labels = [], []
    for cls_idx, feats in a_feats_dict.items():
        for f in feats:
            final_a_feats.append(f)
            final_a_labels.append(cls_idx)
            
    v_feats = np.vstack(v_feats)
    final_a_feats = np.vstack(final_a_feats)
    v_labels, a_labels = np.array(labels), np.array(final_a_labels)
    
    sim_matrix = cosine_similarity(v_feats, final_a_feats)
    
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
# 4. MAIN
# ==========================================
def main():
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = CrossModalModel().to(DEVICE)
    
    # Load Cnn14 Weights
    if os.path.exists(args.audio_weights):
        print(f"Loading AudioSet Weights: {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.audio_model.base_model.load_state_dict(state_dict, strict=False)
    
    # Verify Trainable Parameters
    v_params = sum(p.numel() for p in model.vision_model.parameters() if p.requires_grad)
    a_params = sum(p.numel() for p in model.audio_model.parameters() if p.requires_grad)
    print(f"Trainable Params -> Vision (CLIP): {v_params:,} | Audio (Cnn14): {a_params:,}")

    # Optimize only unfrozen parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    train_ds = BidirectionalDataset(TRAIN_V, TRAIN_A, transform=clip_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    
    best_avg_r5 = 0.0
    print(f"--- Starting Fine-Tuning (CLIP + Cnn14) ---")
    
    for epoch in range(20):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")
        
        for img_anc, img_neg, aud_pos, aud_neg in pbar:
            img_anc, img_neg = img_anc.to(DEVICE), img_neg.to(DEVICE)
            aud_pos, aud_neg = aud_pos.to(DEVICE), aud_neg.to(DEVICE)
            
            optimizer.zero_grad()
            v_a, v_n = model.vision_model(img_anc), model.vision_model(img_neg)
            a_p, a_n = model.audio_model(aud_pos), model.audio_model(aud_neg)
            
            loss = criterion(v_a, a_p, a_n) + criterion(a_p, v_a, v_n)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print("Validating...")
        i2a, a2i = compute_metrics(model, VAL_V, VAL_A)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f}")
        print(f"  I->A: R@1 {i2a[1]:.1f}% | R@5 {i2a[5]:.1f}% | R@10 {i2a[10]:.1f}%")
        print(f"  A->I: R@1 {a2i[1]:.1f}% | R@5 {a2i[5]:.1f}% | R@10 {a2i[10]:.1f}%")
        
        if avg_r5 > best_avg_r5:
            best_avg_r5 = avg_r5
            torch.save(model.state_dict(), args.save_path)
            print("  🔥 Best Fine-Tuned CLIP-Cnn14 Saved!")

if __name__ == "__main__":
    main()