import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.utils import save_image
from tqdm import tqdm

# Import from your modular files
from datasets import BidirectionalDataset, vision_transform
from quaternion_attn import QUATER_ATTENTION
from ca_attn import CA_Block
from iclm import ICLM
from losses import CosineTripletLoss
from evaluate import compute_metrics

def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_sarci_mim.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--mask_ratio", type=float, default=0.20)
    parser.add_argument("--lambda_recon", type=float, default=0.5)
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
# MASKING & VISUALIZATION UTILITIES
# ==========================================
def apply_random_mask(imgs, mask_ratio=0.20, patch_size=16):
    if mask_ratio <= 0.0: return imgs
        
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

def save_reconstruction_sample(orig, masked, recon, epoch, out_dir="reconstruction_samples"):
    os.makedirs(out_dir, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(orig.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(orig.device)
    
    orig_denorm = (orig[0:1] * std + mean).clamp(0, 1)
    masked_denorm = (masked[0:1] * std + mean).clamp(0, 1)
    recon_denorm = (recon[0:1] * std + mean).clamp(0, 1)
    
    comparison = torch.cat([orig_denorm, masked_denorm, recon_denorm], dim=3)
    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_recon.png")
    save_image(comparison, save_path)

# ==========================================
# ARCHITECTURES
# ==========================================
class PixelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)) 
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)) 
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) 
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)) 
        self.up5 = nn.Sequential(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)) 

    def forward(self, x):
        return self.up5(self.up4(self.up3(self.up2(self.up1(x)))))

class SarciVisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet.parameters(): param.requires_grad = False
            
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

class SarciAudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        for param in self.base_model.parameters(): param.requires_grad = False
            
        self.ca_block = CA_Block(2048)
        self.attn_pool = nn.Sequential(nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1))
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

class SarciCrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_model = SarciVisionEncoder(embedding_dim)
        self.audio_model = SarciAudioEncoder(embedding_dim)
        self.iclm = ICLM(embed_dim=embedding_dim)


# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def main():
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = SarciCrossModalModel().to(DEVICE)
    
    if os.path.exists(args.audio_weights):
        print(f"Loading pre-trained weights from {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.audio_model.base_model.load_state_dict(state_dict, strict=False)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    criterion_retrieval = CosineTripletLoss(margin=0.2)
    criterion_recon = nn.MSELoss()
    
    train_ds = BidirectionalDataset(TRAIN_V, TRAIN_A, transform=vision_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    best_avg_r5 = 0.0
    epochs_no_improve = 0
    
    print(f"Starting SARCI MIM training. Mask ratio: {args.mask_ratio}")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (img_anc, img_neg, aud_pos, aud_neg) in enumerate(pbar):
            img_anc, img_neg = img_anc.to(DEVICE), img_neg.to(DEVICE)
            aud_pos, aud_neg = aud_pos.to(DEVICE), aud_neg.to(DEVICE)
            
            optimizer.zero_grad()
            
            masked_img_anc = apply_random_mask(img_anc, mask_ratio=args.mask_ratio)
            masked_img_neg = apply_random_mask(img_neg, mask_ratio=args.mask_ratio)
            
            v_o_anc, recon_anc = model.vision_model(masked_img_anc, reconstruct=True)
            v_o_neg = model.vision_model(masked_img_neg, reconstruct=False)
            a_o_pos = model.audio_model(aud_pos)
            a_o_neg = model.audio_model(aud_neg)
            
            v_f_anc_pos, a_f_pos = model.iclm(v_o_anc, a_o_pos)
            v_f_anc_neg, a_f_neg_a = model.iclm(v_o_anc, a_o_neg) 
            v_f_neg_v, a_f_pos_neg = model.iclm(v_o_neg, a_o_pos) 
            
            sim_pos = F.cosine_similarity(v_f_anc_pos, a_f_pos, dim=1)
            sim_neg_aud = F.cosine_similarity(v_f_anc_neg, a_f_neg_a, dim=1)
            sim_neg_img = F.cosine_similarity(v_f_neg_v, a_f_pos_neg, dim=1)
            
            loss_i2a = criterion_retrieval(sim_pos, sim_neg_aud)
            loss_a2i = criterion_retrieval(sim_pos, sim_neg_img)
            retrieval_loss = loss_i2a + loss_a2i
            
            recon_loss = criterion_recon(recon_anc, img_anc)
            loss = retrieval_loss + (args.lambda_recon * recon_loss)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pbar.set_postfix(Loss=f"{loss.item():.3f}", Ret=f"{retrieval_loss.item():.3f}", Rec=f"{recon_loss.item():.3f}")
            
            if (epoch + 1) % 5 == 0 and batch_idx == 0:
                save_reconstruction_sample(img_anc, masked_img_anc, recon_anc, epoch + 1)
            
        print("Running evaluation...")
        i2a, a2i = compute_metrics(model, VAL_V, VAL_A, DEVICE, batch_size=128)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        print(f"Epoch {epoch+1} Results - Total Loss: {train_loss/len(train_loader):.4f}")
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
            
        if epochs_no_improve >= args.patience:
            print(f"Early stopping condition met. Training halted at epoch {epoch+1}.")
            break

if __name__ == "__main__":
    main()