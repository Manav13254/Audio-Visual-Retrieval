import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import glob
import argparse
import sys
import numpy as np

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_sarci_mim2.pth")
    parser.add_argument("--mask_ratio", type=float, default=0.20)
    parser.add_argument("--lambda_recon", type=float, default=0.5)
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audioset setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(CURRENT_DIR, "audioset_tagging_cnn")
sys.path.append(REPO_PATH)
sys.path.append(os.path.join(REPO_PATH, 'utils'))
sys.path.append(os.path.join(REPO_PATH, 'pytorch'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print("Error: Could not import Cnn14. Ensure audioset_tagging_cnn is in the path.")
    sys.exit()

# Image Masking Utility
def apply_random_mask(imgs, mask_ratio=0.20, patch_size=16):
    if mask_ratio <= 0.0:
        return imgs
        
    B, C, H, W = imgs.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w
    num_masked = int(total_patches * mask_ratio)
    
    masked_imgs = imgs.clone()
    
    for i in range(B):
        mask_indices = torch.randperm(total_patches)[:num_masked]
        for idx in mask_indices:
            row = (idx // num_patches_w) * patch_size
            col = (idx % num_patches_w) * patch_size
            masked_imgs[i, :, row:row+patch_size, col:col+patch_size] = 0.0
            
    return masked_imgs

# Visualization Utility - Modified to save a Grid of target classes
def save_reconstruction_grid(sample_list, epoch, out_dir="reconstruction_samples"):
    os.makedirs(out_dir, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    
    all_rows = []
    for orig, masked, recon in sample_list:
        # Denormalize
        orig_dn = (orig * std + mean).clamp(0, 1)
        masked_dn = (masked * std + mean).clamp(0, 1)
        recon_dn = (recon * std + mean).clamp(0, 1)
        
        # Row: Original | Masked | Reconstructed
        row = torch.cat([orig_dn, masked_dn, recon_dn], dim=3)
        all_rows.append(row)
        
    if all_rows:
        grid = torch.cat(all_rows, dim=2) # Stack rows vertically
        save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_comparison.png")
        save_image(grid, save_path)


# Attention & Interaction Modules
class QUATER_ATTENTION(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(QUATER_ATTENTION, self).__init__()
        self.fc_h = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn_h = nn.BatchNorm2d(in_planes // ratio)
        self.relu_h = nn.ReLU()
        self.conv_h_sptial = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio, 7, padding=3, bias=False)

        self.fc_w = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn_w = nn.BatchNorm2d(in_planes // ratio)
        self.relu_w = nn.ReLU()
        self.conv_w_sptial = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio, 7, padding=3, bias=False)

        self.fc_general = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x_h_avg = torch.mean(x, dim=3, keepdim=True)
        x_h_max, _ = torch.max(x, dim=3, keepdim=True)
        x_w_avg = torch.mean(x, dim=2, keepdim=True)
        x_w_max, _ = torch.max(x, dim=2, keepdim=True)

        x_h_avg = self.relu_h(self.bn_h(self.fc_h(x_h_avg)))
        x_h_max = self.relu_h(self.bn_h(self.fc_h(x_h_max)))
        x_w_avg = self.relu_w(self.bn_w(self.fc_w(x_w_avg)))
        x_w_max = self.relu_w(self.bn_w(self.fc_w(x_w_max)))

        x_h_cat_sp = self.conv_h_sptial(torch.cat([x_h_avg, x_h_max], dim=1))
        x_w_cat_sp = self.conv_w_sptial(torch.cat([x_w_avg, x_w_max], dim=1))

        x_general = self.fc_general(x_h_cat_sp * x_w_cat_sp)
        return x * self.sigmoid(x_general)

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        self.conv_1x1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        self.F_w = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()   

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        return x * s_h.expand_as(x) * s_w.expand_as(x)

class ICLM(nn.Module):
    def __init__(self, embed_dim=128, heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.scale = (embed_dim // heads) ** -0.5

        self.v_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.a_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, v_o, a_o):
        B = v_o.size(0)

        def reshape_heads(x):
            return x.view(B, self.heads, 1, self.embed_dim // self.heads)

        vq = reshape_heads(self.v_q(v_o))
        vk = reshape_heads(self.v_k(v_o))
        vv = reshape_heads(self.v_v(v_o))

        aq = reshape_heads(self.a_q(a_o))
        ak = reshape_heads(self.a_k(a_o))
        av = reshape_heads(self.a_v(a_o))

        dots_v_ai = torch.einsum('bhid,bhjd->bhij', aq, vk) * self.scale
        attn_v_ai = dots_v_ai.softmax(dim=-1)
        v_ai = torch.einsum('bhij,bhjd->bhid', attn_v_ai, vv).reshape(B, self.embed_dim)

        dots_a_vi = torch.einsum('bhid,bhjd->bhij', vq, ak) * self.scale
        attn_a_vi = dots_a_vi.softmax(dim=-1)
        a_vi = torch.einsum('bhij,bhjd->bhid', attn_a_vi, av).reshape(B, self.embed_dim)

        v_local = torch.sigmoid(a_vi) * v_ai
        v_global = torch.sigmoid(a_o) * v_ai
        v_final = v_local + v_global + v_o

        a_fine = torch.sigmoid(v_ai) * a_o
        a_iteration = torch.sigmoid(a_vi) * a_o
        a_final = a_fine + a_iteration + a_o

        return F.normalize(v_final, p=2, dim=1), F.normalize(a_final, p=2, dim=1)


# Encoders & Decoder Architecture
class PixelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ) 
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ) 
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) 
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ) 
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        ) 

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x

class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        for param in resnet.parameters():
            param.requires_grad = False
            
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
        
        self.decoder = PixelDecoder()

    def forward(self, x, reconstruct=False):
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
        retrieval_embed = F.normalize(self.project(out), p=2, dim=1)
        
        if reconstruct:
            reconstructed_img = self.decoder(final)
            return retrieval_embed, reconstructed_img
            
        return retrieval_embed

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
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

class CrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_model = VisionEncoder(embedding_dim)
        self.audio_model = AudioEncoder(embedding_dim)
        self.iclm = ICLM(embed_dim=embedding_dim)


# ==========================================
# SUPERVISED CONTRASTIVE LOSS
# ==========================================
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v_features, a_features, labels):
        """
        v_features: (Batch, Dim)
        a_features: (Batch, Dim)
        labels: (Batch) - Integer class labels
        """
        device = v_features.device
        
        # 1. Create a mask where mask[i, j] = 1 if they share the SAME label
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 2. Calculate similarities
        logits = torch.matmul(v_features, a_features.T) / self.temperature
        
        # 3. Apply Log-Softmax
        log_softmax_v2a = F.log_softmax(logits, dim=1)
        log_softmax_a2v = F.log_softmax(logits.T, dim=1)
        
        # 4. Calculate loss only over the true positives (using the mask)
        # Using .clamp(min=1e-5) prevents division by zero if something weird happens
        loss_v2a = - (mask * log_softmax_v2a).sum(1) / mask.sum(1).clamp(min=1e-5)
        loss_a2v = - (mask * log_softmax_a2v).sum(1) / mask.sum(1).clamp(min=1e-5)
        
        return (loss_v2a.mean() + loss_a2v.mean()) / 2


vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# DATASETS
# ==========================================
class BidirectionalDataset(Dataset):
    def __init__(self, vision_dir, audio_dir, transform=None):
        self.vision_dir = vision_dir
        self.audio_dir = audio_dir
        self.transform = transform
        v_classes = set(os.listdir(vision_dir))
        a_classes = set(os.listdir(audio_dir))
        self.classes = sorted(list(v_classes.intersection(a_classes)))
        self.class_to_imgs = {c: glob.glob(os.path.join(vision_dir, c, "*.*")) for c in self.classes}
        self.class_to_auds = {c: glob.glob(os.path.join(audio_dir, c, "*.wav")) for c in self.classes}

    def __len__(self): 
        return sum(len(imgs) for imgs in self.class_to_imgs.values())

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
        # Pick the target class and get its integer index for the SupCon Loss
        target_class = random.choice(self.classes)
        class_idx = self.classes.index(target_class)
        
        img_path = random.choice(self.class_to_imgs[target_class])
        aud_pos_path = random.choice(self.class_to_auds[target_class])
        
        img_anchor = Image.open(img_path).convert('RGB')
        if self.transform: 
            img_anchor = self.transform(img_anchor)
            
        aud_pos = self._process_audio(aud_pos_path)
        
        # Return anchor, positive audio, and the class label
        return img_anchor, aud_pos, class_idx


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
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            return wav.squeeze(0), self.labels[idx]
        except: 
            return torch.zeros(320000), self.labels[idx]


# Validation Routine
def compute_metrics(model, v_dir, a_dir, k_values=[1, 5, 10], batch_size=64):
    model.eval()
    v_ds = ValVisionDataset(v_dir, transform=vision_transform)
    a_ds = ValAudioDataset(a_dir)
    
    v_loader = DataLoader(v_ds, batch_size=batch_size, num_workers=8, pin_memory=True)
    a_loader = DataLoader(a_ds, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    v_base_feats, v_labels = [], []
    a_base_feats, a_labels = [], []
    
    print("Extracting base features...")
    with torch.no_grad():
        for imgs, lbls in tqdm(v_loader, leave=False):
            v_base_feats.append(model.vision_model(imgs.to(DEVICE), reconstruct=False).cpu())
            v_labels.extend(lbls.numpy())
            
        for auds, lbls in tqdm(a_loader, leave=False):
            a_base_feats.append(model.audio_model(auds.to(DEVICE)).cpu())
            a_labels.extend(lbls.numpy())
            
    v_base_feats = torch.cat(v_base_feats, dim=0)
    a_base_feats = torch.cat(a_base_feats, dim=0)
    v_labels = np.array(v_labels)
    a_labels = np.array(a_labels)
    
    N, M = v_base_feats.size(0), a_base_feats.size(0)
    sim_matrix = np.zeros((N, M))
    
    print("Computing cross-modal interactions...")
    with torch.no_grad():
        for i in tqdm(range(N), leave=False):
            v_row = v_base_feats[i].unsqueeze(0).to(DEVICE)
            for j in range(0, M, batch_size):
                a_chunk = a_base_feats[j:j+batch_size].to(DEVICE)
                v_chunk = v_row.expand(a_chunk.size(0), -1)
                
                v_final, a_final = model.iclm(v_chunk, a_chunk)
                sims = F.cosine_similarity(v_final, a_final, dim=1).cpu().numpy()
                sim_matrix[i, j:j+batch_size] = sims
            
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


# Main Training Loop
# Main Training Loop
def main():
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = CrossModalModel().to(DEVICE)
    
    if os.path.exists(args.audio_weights):
        print(f"Loading pre-trained weights from {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.audio_model.base_model.load_state_dict(state_dict, strict=False)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    max_epochs = 60
    
    # ---------------------------------------------------------
    # ADDED: Initialize the Cosine Annealing Scheduler
    # ---------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    # Init SupCon Loss
    criterion_retrieval = SupervisedContrastiveLoss(temperature=0.07)
    criterion_recon = nn.MSELoss()
    
    train_ds = BidirectionalDataset(TRAIN_V, TRAIN_A, transform=vision_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    
    # Target classes for visual monitoring
    monitor_classes = ['airport', 'beach', 'forest', 'grassland', 'sportsland', 'residential', 'harbour']

    best_avg_r5 = 0.0
    patience = 15
    epochs_no_improve = 0
    
    print(f"Starting training. Mask ratio: {args.mask_ratio}")
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- Visual Monitoring Block (Every 5 epochs) ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            plot_samples = []
            with torch.no_grad():
                for cls in monitor_classes:
                    if cls in train_ds.classes:
                        img_path = random.choice(train_ds.class_to_imgs[cls])
                        img_pil = Image.open(img_path).convert('RGB')
                        img_t = vision_transform(img_pil).unsqueeze(0).to(DEVICE)
                        
                        masked_t = apply_random_mask(img_t, mask_ratio=args.mask_ratio)
                        _, recon_t = model.vision_model(masked_t, reconstruct=True)
                        
                        plot_samples.append((img_t, masked_t, recon_t))
            
            if plot_samples:
                save_reconstruction_grid(plot_samples, epoch + 1)
            model.train()

        # Added current LR to the progress bar description
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [LR: {current_lr:.6f}]")
        
        for batch_idx, (img_anc, aud_pos, labels) in enumerate(pbar):
            img_anc = img_anc.to(DEVICE)
            aud_pos = aud_pos.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            masked_img_anc = apply_random_mask(img_anc, mask_ratio=args.mask_ratio)
            
            v_o_anc, recon_anc = model.vision_model(masked_img_anc, reconstruct=True)
            a_o_pos = model.audio_model(aud_pos)
            
            v_f_pos, a_f_pos = model.iclm(v_o_anc, a_o_pos)
            
            # Pass the labels to the SupCon loss
            retrieval_loss = criterion_retrieval(v_f_pos, a_f_pos, labels)
            recon_loss = criterion_recon(recon_anc, img_anc)
            
            # Loss Decay: smoothly reduce reconstruction penalty
            current_lambda = args.lambda_recon * (1.0 - (epoch / max_epochs))
            
            loss = retrieval_loss + (current_lambda * recon_loss)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pbar.set_postfix(Loss=f"{loss.item():.3f}", Ret=f"{retrieval_loss.item():.3f}", Rec=f"{recon_loss.item():.3f}")
            
        # ---------------------------------------------------------
        # ADDED: Step the scheduler at the end of the epoch
        # ---------------------------------------------------------
        scheduler.step()
            
        print("Running evaluation...")
        i2a, a2i = compute_metrics(model, VAL_V, VAL_A, batch_size=128)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        # Added current LR to the print output
        print(f"Epoch {epoch+1} Results - Total Loss: {train_loss/len(train_loader):.4f} | LR: {current_lr:.6f}")
        print(f"Image-to-Audio: R@1: {i2a[1]:.1f} | R@5: {i2a[5]:.1f} | R@10: {i2a[10]:.1f}")
        print(f"Audio-to-Image: R@1: {a2i[1]:.1f} | R@5: {a2i[5]:.1f} | R@10: {a2i[10]:.1f}")
        
        if avg_r5 > best_avg_r5:
            best_avg_r5 = avg_r5
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"Validation performance improved. Model saved to {args.save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} consecutive epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping condition met. Training halted at epoch {epoch+1}.")
            break


if __name__ == "__main__":
    main()