import os
import glob
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import resnet18
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import torchvision.transforms as transforms
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 0. SETUP & PATHS
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_amfmn_soft.pth")
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
# 1. AMFMN MODELS
# ==========================================
def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ExtractFeature(nn.Module):
    def __init__(self, opt):
        super(ExtractFeature, self).__init__()
        self.embed_dim = opt['embed']['embed_dim']
        
        # Safely load pretrained weights
        try:
            self.resnet = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            self.resnet = resnet18(pretrained=True)
            
        # --- STRICTLY FREEZE RESNET BACKBONE ---
        for param in self.resnet.parameters():
            param.requires_grad = False

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
        
        # Flatten for spatial attention
        atted_feature = (main_feature * attn_feature).view(concat_feature.shape[0], -1)

        solo_att = torch.sigmoid(self.solo_attention(atted_feature))
        solo_feature = solo_feature * solo_att

        return solo_feature

class EncoderAudio(nn.Module):
    def __init__(self, embed_size):
        super(EncoderAudio, self).__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, 
                                mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        # --- STRICTLY FREEZE CNN14 BACKBONE ---
        for param in self.base_model.parameters():
            param.requires_grad = False
            
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
        # scores is already a B x B matrix computed by AMFMN
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        if torch.cuda.is_available(): mask = mask.cuda()
            
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class AMFMN_AudioVisual(nn.Module):
    """The unified AMFMN architecture adapted for Audio-Visual."""
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

    def forward(self, img, aud, labels=None):
        v_feat = self.forward_img(img)
        a_feat = self.forward_aud(aud)

        # Cross Attention: Image acts as a gate to guide the Audio
        a_guided = self.cross_attn(v_feat, a_feat) # Shape: (B, B, D)

        # Calculate Score Matrix
        v_feat_exp = v_feat.unsqueeze(1) # Shape: (B, 1, D)
        scores = F.cosine_similarity(v_feat_exp, a_guided, dim=-1) # Shape: (B, B)

        if labels is not None:
            loss = self.criterion(scores, labels)
            return loss, scores
        return scores

# ==========================================
# 2. DATASET & TRANSFORMS
# ==========================================
# IMPORTANT: Resized to 256x256 to match the exact mathematical bounds of the VSA spatial tensor flattening
vision_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ClassAwareBatchDataset(Dataset):
    def __init__(self, vision_dir, audio_dir, transform=None):
        self.vision_dir = vision_dir
        self.audio_dir = audio_dir
        self.transform = transform
        
        v_classes = set(os.listdir(vision_dir))
        a_classes = set(os.listdir(audio_dir))
        self.classes = sorted(list(v_classes.intersection(a_classes)))
        
        self.class_to_imgs = {c: glob.glob(os.path.join(vision_dir, c, "*.*")) for c in self.classes}
        self.class_to_auds = {c: glob.glob(os.path.join(audio_dir, c, "*.wav")) for c in self.classes}
        
        self.samples = []
        for cls_idx, cls_name in enumerate(self.classes):
            for img_path in self.class_to_imgs[cls_name]:
                self.samples.append((img_path, cls_name, cls_idx))

    def __len__(self): 
        return len(self.samples)

    def _process_audio(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if sr != 32000: wav = T.Resample(sr, 32000)(wav)
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            return wav
        except: 
            return torch.zeros(1, 320000)

    def __getitem__(self, index):
        img_path, cls_name, label = self.samples[index]
        aud_path = random.choice(self.class_to_auds[cls_name])
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        aud = self._process_audio(aud_path)
        return img, aud, label

# ==========================================
# 3. METRICS (Memory-Safe Validation)
# ==========================================
def compute_metrics(model, v_dir, a_dir, k_values=[1, 5, 10]):
    model.eval()
    classes = sorted(os.listdir(v_dir))
    v_feats, a_base_feats, labels = [], [], []
    
    print("Extracting Evaluation Features...")
    with torch.no_grad():
        for i, cls in enumerate(classes):
            # Process Images
            v_files = glob.glob(os.path.join(v_dir, cls, "*.*"))
            for f in v_files:
                try:
                    img = Image.open(f).convert('RGB')
                    img = vision_transform(img).unsqueeze(0).to(DEVICE)
                    # Move to CPU immediately to prevent OOM
                    v_feats.append(model.forward_img(img).cpu())
                    labels.append(i)
                except: continue
                
    a_feats_dict = {i: [] for i in range(len(classes))}
    with torch.no_grad():
        for i, cls in enumerate(classes):
            a_files = glob.glob(os.path.join(a_dir, cls, "*.wav"))
            for f in a_files:
                try:
                    wav, sr = torchaudio.load(f)
                    if sr!=32000: wav=T.Resample(sr,32000)(wav)
                    if wav.shape[1]<320000: wav=torch.nn.functional.pad(wav,(0,320000-wav.shape[1]))
                    else: wav=wav[:,:320000]
                    if wav.shape[0]>1: wav=torch.mean(wav,dim=0,keepdim=True)
                    wav = wav.to(DEVICE)
                    # Move to CPU immediately to prevent OOM
                    a_feats_dict[i].append(model.forward_aud(wav).cpu())
                except: continue

    for cls_idx, feats in a_feats_dict.items():
        for f in feats:
            a_base_feats.append(f)
            
    v_feats = torch.cat(v_feats)        # (N_v, D) on CPU
    a_base_feats = torch.cat(a_base_feats) # (N_a, D) on CPU
    v_labels = np.array(labels)
    a_labels = np.repeat(np.arange(len(classes)), [len(a_feats_dict[i]) for i in range(len(classes))])
    
    print("Computing Cross-Modal Matrix...")
    sim_matrix = np.zeros((len(v_feats), len(a_base_feats)))
    chunk_size = 32
    
    with torch.no_grad():
        # Move the audio gallery to GPU once
        a_base_feats_gpu = a_base_feats.to(DEVICE)
        
        for i in range(0, len(v_feats), chunk_size):
            # Only move the current image chunk to GPU
            v_chunk = v_feats[i:i+chunk_size].to(DEVICE)
            
            # Image acts as a gate to guide ALL audio features in the database
            a_guided = model.cross_attn(v_chunk, a_base_feats_gpu) 
            scores = F.cosine_similarity(v_chunk.unsqueeze(1), a_guided, dim=-1)
            
            # Move the resulting scores back to CPU numpy array
            sim_matrix[i:i+chunk_size] = scores.cpu().numpy()
            
    def calculate_r_at_k(matrix, q_labels, g_labels):
        scores = {k: 0 for k in k_values}
        for idx in range(len(q_labels)):
            sorted_indices = np.argsort(matrix[idx])[::-1]
            for k in k_values:
                if q_labels[idx] in g_labels[sorted_indices[:k]]: scores[k] += 1
        return {k: (v / len(q_labels)) * 100 for k, v in scores.items()}

    i2a = calculate_r_at_k(sim_matrix, v_labels, a_labels)
    a2i = calculate_r_at_k(sim_matrix.T, a_labels, v_labels)
    
    return i2a, a2i

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    opt = {
        'embed': {'embed_dim': 128},
        'multiscale': {
            'multiscale_input_channel': 64,
            'multiscale_output_channel': 1 # Critical for spatial flattening shape (256 parameters)
        },
        # OPTIONS: 'soft_att', 'fusion_att', 'similarity_att'
        'cross_attention': {'att_type': 'soft_att'}, 
        'loss': {'margin': 0.2}
    }
    
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = AMFMN_AudioVisual(opt).to(DEVICE)
    
    if os.path.exists(args.audio_weights):
        print(f"Loading AudioSet Weights: {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.aud_enc.base_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"WARNING: AudioSet Weights not found at {args.audio_weights}")
        
    # Verify Trainable Params - Should be very small now!
    v_params = sum(p.numel() for p in model.img_enc.parameters() if p.requires_grad)
    a_params = sum(p.numel() for p in model.aud_enc.parameters() if p.requires_grad)
    print(f"Trainable Params -> Vision Backbone: {v_params:,} | Audio Backbone: {a_params:,}")

    train_ds = ClassAwareBatchDataset(TRAIN_V, TRAIN_A, transform=vision_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    
    # Only optimize the parameters that require gradients
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    best_avg_r5 = 0.0
    max_epochs = 40
    patience = 10
    epochs_no_improve = 0
    
    print(f"--- Starting Audio-Visual AMFMN Training ({opt['cross_attention']['att_type']}) ---")
    
    for epoch in range(max_epochs):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for img, aud, labels in pbar:
            if torch.cuda.is_available():
                img, aud, labels = img.cuda(), aud.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            loss, _ = model(img, aud, labels)
            
            loss.backward()
            clip_grad_norm(model.parameters(), 2.0)
            optimizer.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        print("Validating Cross-Modal Retrieval...")
        i2a, a2i = compute_metrics(model, VAL_V, VAL_A)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        print(f"Epoch {epoch+1} Results:")
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
            print(f"  🔥 Best AMFMN Model Saved to '{args.save_path}'!")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"  🛑 Early stopping triggered! Training stopped after {epoch+1} epochs.")
            break

if __name__ == "__main__":
    main()