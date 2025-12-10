import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast # Optimisation GPU
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# --- Configuration Optimisée pour RTX 4090 ---
IMG_SIZE = 64
BATCH_SIZE = 2048    # Augmenté drastiquement pour remplir la VRAM de la 4090
TIMESTEPS = 16
LR = 1e-4
EPOCHS = 100         # Tu peux viser plus haut maintenant
SAVE_INTERVAL = 10   # Sauvegarde tous les 10 epochs pour gagner du temps I/O

# Configuration Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Appareil utilisé : {DEVICE}")

# Optimisation backend CUDNN
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True # Accélère si la taille des images est constante

# --- 1. Dataset Robuste ---

class LocalCelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        # Recherche récursive
        search_path = os.path.join(root_dir, '**', '*.jpg')
        self.image_paths = glob.glob(search_path, recursive=True)
        
        if len(self.image_paths) == 0:
            search_path_png = os.path.join(root_dir, '**', '*.png')
            self.image_paths = glob.glob(search_path_png, recursive=True)

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"❌ Aucune image trouvée dans '{root_dir}'")
        
        print(f"✅ Dataset chargé : {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, 0 
        except Exception as e:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), 0

# --- 2. Diffusion Logic ---
time_bar = 1 - np.linspace(0, 1.0, TIMESTEPS + 1)

def forward_noise(x, t):
    noise = torch.randn_like(x).to(DEVICE)
    # Récupération optimisée des coefficients
    a = torch.tensor(time_bar, device=DEVICE)[t].view(-1, 1, 1, 1).float()
    b = torch.tensor(time_bar, device=DEVICE)[t + 1].view(-1, 1, 1, 1).float()
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b

# --- 3. Modèle U-Net BOOSTÉ (Celui de la 5090) ---

class TimeBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim=256): # Notez le 256 ici (avant c'était 128)
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.relu = nn.ReLU()
        self.time_dense = nn.Linear(time_emb_dim, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_c) 
        
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.relu(h)
        time_scale = self.time_dense(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_scale 
        h = self.conv2(h)
        h = self.norm(h)
        h = self.relu(h)
        if x.shape[1] == h.shape[1]:
            return x + h
        return h

class QuickDiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding temporel plus précis (256)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        
        # Architecture élargie (Base 128)
        self.b1 = TimeBlock(3, 128)      # 64x64
        self.pool1 = nn.MaxPool2d(2)
        self.b2 = TimeBlock(128, 256)    # 32x32
        self.pool2 = nn.MaxPool2d(2)
        self.b3 = TimeBlock(256, 512)    # 16x16
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck (8x8)
        self.flatten_dim = 8 * 8 * 512
        self.bottleneck_mlp = nn.Sequential(
            nn.Linear(self.flatten_dim + 256, 1024),
            nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, self.flatten_dim),
            nn.LayerNorm(self.flatten_dim), nn.ReLU()
        )
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2)
        self.b4 = TimeBlock(512 + 512, 256)
        self.up2 = nn.Upsample(scale_factor=2)
        self.b5 = TimeBlock(256 + 256, 128)
        self.up3 = nn.Upsample(scale_factor=2)
        self.b6 = TimeBlock(128 + 128, 128)
        self.final_conv = nn.Conv2d(128, 3, 1)

    def forward(self, x, t):
        t_norm = t.view(-1, 1).float() / TIMESTEPS
        t_emb = self.time_mlp(t_norm)
        
        x1 = self.b1(x, t_emb)
        p1 = self.pool1(x1)
        x2 = self.b2(p1, t_emb)
        p2 = self.pool2(x2)
        x3 = self.b3(p2, t_emb)
        p3 = self.pool3(x3)
        
        flat = p3.view(-1, self.flatten_dim)
        bot = torch.cat([flat, t_emb], dim=1)
        bot = self.bottleneck_mlp(bot)
        bot = bot.view(-1, 512, 8, 8)
        
        u1 = self.up1(bot)
        u1 = torch.cat([u1, x3], dim=1)
        x4 = self.b4(u1, t_emb)
        u2 = self.up2(x4)
        u2 = torch.cat([u2, x2], dim=1)
        x5 = self.b5(u2, t_emb)
        u3 = self.up3(x5)
        u3 = torch.cat([u3, x1], dim=1)
        x6 = self.b6(u3, t_emb)
        
        return self.final_conv(x6)

# --- 4. Entraînement avec AMP (Mixed Precision) ---

def train(dataloader, model, optimizer, criterion):
    model.train()
    os.makedirs("checkpoints", exist_ok=True)
    
    # Scaler pour la précision mixte (float16)
    scaler = GradScaler() 

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        epoch_loss = 0
        
        for x, _ in pbar:
            x = x.to(DEVICE, non_blocking=True) # Transfert rapide
            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE)
            
            # Context AMP : calcule en float16, stocke en float32
            with autocast():
                img_a, img_b = forward_noise(x, t)
                pred = model(img_a, t)
                loss = criterion(pred, img_b)
            
            # Backpropagation optimisée
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Sauvegarde périodique
        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == EPOCHS:
            torch.save(model.state_dict(), f"checkpoints/model_ep{epoch+1}.pth")
            print(f"💾 Modèle sauvegardé (Epoch {epoch+1})")
            predict_preview(model, epoch+1)

def predict_preview(model, epoch_num):
    model.eval()
    os.makedirs("results", exist_ok=True)
    with torch.no_grad():
        x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        for i in range(TIMESTEPS):
            t_input = torch.full((16,), i, device=DEVICE, dtype=torch.long)
            x = model(x, t_input)
        
        x = x.cpu().clamp(-1, 1)
        grid = torchvision.utils.make_grid(x, nrow=4, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Epoch {epoch_num}")
        plt.savefig(f"results/epoch_{epoch_num}.png")
        plt.close()

# --- 5. Main ---

if __name__ == "__main__":
    # Détection du dossier data (relatif au script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data') # Assure-toi que ton dossier s'appelle 'data' sur le serveur

    print(f"📂 Dossier données : {data_dir}")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        dataset = LocalCelebADataset(root_dir=data_dir, transform=transform)
        
        # OPTIMISATION CPU -> GPU
        # num_workers=8 car les serveurs GPU ont souvent beaucoup de coeurs CPU
        # pin_memory=True accélère le transfert RAM -> VRAM
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=8,      # Mets 8 sur un serveur Linux
            pin_memory=True,    # Indispensable sur GPU
            persistent_workers=True # Garde les workers vivants entre les epochs
        )

        model = QuickDiffusionUNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss()

        print(f"🚀 Démarrage sur {DEVICE} avec Batch Size {BATCH_SIZE}...")
        train(dataloader, model, optimizer, criterion)
        print("✅ Terminé !")

    except Exception as e:
        print(f"❌ Erreur : {e}")