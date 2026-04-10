import os
import glob
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        target_class = random.choice(self.classes)
        neg_class = random.choice([c for c in self.classes if c != target_class])
        
        img_path = random.choice(self.class_to_imgs[target_class])
        aud_pos_path = random.choice(self.class_to_auds[target_class])
        img_neg_path = random.choice(self.class_to_imgs[neg_class])
        aud_neg_path = random.choice(self.class_to_auds[neg_class])
        
        img_anchor = Image.open(img_path).convert('RGB')
        img_neg = Image.open(img_neg_path).convert('RGB')
        
        if self.transform: 
            img_anchor = self.transform(img_anchor)
            img_neg = self.transform(img_neg)
            
        aud_pos = self._process_audio(aud_pos_path)
        aud_neg = self._process_audio(aud_neg_path)
        return img_anchor, img_neg, aud_pos, aud_neg

class ValVisionDataset(Dataset):
    def __init__(self, v_dir, transform=None):
        self.files, self.labels = [], []
        classes = sorted(os.listdir(v_dir))
        for i, cls in enumerate(classes):
            for f in glob.glob(os.path.join(v_dir, cls, "*.*")):
                self.files.append(f)
                self.labels.append(i)
        self.transform = transform
        
    def __len__(self): 
        return len(self.files)
        
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
                
    def __len__(self): 
        return len(self.files)
        
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