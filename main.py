import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# --- Configuration ---
IMG_SIZE = 64
BATCH_SIZE = 64
TIMESTEPS = 16
LR = 1e-4
EPOCHS = 10
# Détection automatique : GPU (Nvidia), MPS (Mac M1/M2/M3) ou CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"✅ Appareil utilisé : {DEVICE}")

# --- 1. Gestion du Dataset (Locale et Robuste) ---

class LocalCelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Le dossier 'data' (le script cherchera les jpg dedans et dans les sous-dossiers)
        """
        self.transform = transform
        # Recherche récursive de tous les fichiers .jpg dans root_dir
        # Cela marche que les images soient dans data/ ou data/img_align_celeba/
        search_path = os.path.join(root_dir, '**', '*.jpg')
        self.image_paths = glob.glob(search_path, recursive=True)
        
        if len(self.image_paths) == 0:
            # Fallback : essaie l'extension .png au cas où
            search_path_png = os.path.join(root_dir, '**', '*.png')
            self.image_paths = glob.glob(search_path_png, recursive=True)

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"❌ Aucune image trouvée dans '{root_dir}' ou ses sous-dossiers.")
        
        print(f"✅ Dataset chargé : {len(self.image_paths)} images trouvées dans '{root_dir}'.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, 0 
        except Exception as e:
            print(f"⚠️ Erreur fichier {img_path}: {e}")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), 0

# --- 2. Logique de Diffusion ---
time_bar = 1 - np.linspace(0, 1.0, TIMESTEPS + 1)

def forward_noise(x, t):
    noise = torch.randn_like(x).to(DEVICE)
    a = torch.tensor(time_bar, device=DEVICE)[t].view(-1, 1, 1, 1).float()
    b = torch.tensor(time_bar, device=DEVICE)[t + 1].view(-1, 1, 1, 1).float()
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b

# --- 3. Modèle U-Net ---

class TimeBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim=128):
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
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 192), nn.ReLU(),
            nn.Linear(192, 128), nn.ReLU()
        )
        # Encoder
        self.b1 = TimeBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.b2 = TimeBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.b3 = TimeBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck (8x8)
        self.flatten_dim = 8 * 8 * 256
        self.bottleneck_mlp = nn.Sequential(
            nn.Linear(self.flatten_dim + 128, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, self.flatten_dim),
            nn.LayerNorm(self.flatten_dim), nn.ReLU()
        )
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2)
        self.b4 = TimeBlock(256 + 256, 128)
        self.up2 = nn.Upsample(scale_factor=2)
        self.b5 = TimeBlock(128 + 128, 64)
        self.up3 = nn.Upsample(scale_factor=2)
        self.b6 = TimeBlock(64 + 64, 64)
        self.final_conv = nn.Conv2d(64, 3, 1)

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
        bot = bot.view(-1, 256, 8, 8)
        
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

# --- 4. Fonctions d'Entraînement et Prédiction ---

def train(dataloader, model, optimizer, criterion):
    model.train()
    
    # Création dossier de sauvegarde
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for x, _ in pbar:
            x = x.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE)
            img_a, img_b = forward_noise(x, t)
            
            optimizer.zero_grad()
            pred = model(img_a, t)
            loss = criterion(pred, img_b)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())

        # Sauvegarde et Preview
        if (epoch + 1) % 1 == 0: # Sauvegarde à chaque époque
            torch.save(model.state_dict(), f"checkpoints/model_ep{epoch+1}.pth")
            predict_preview(model, epoch+1)

def predict_preview(model, epoch_num):
    model.eval()
    os.makedirs("results", exist_ok=True)
    with torch.no_grad():
        # Générer 16 images
        x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        
        for i in range(TIMESTEPS):
            t_input = torch.full((16,), i, device=DEVICE, dtype=torch.long)
            x = model(x, t_input)
        
        # Sauvegarde de l'image
        x = x.cpu().clamp(-1, 1)
        grid = torchvision.utils.make_grid(x, nrow=4, normalize=True)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Result Epoch {epoch_num}")
        # En local, on sauvegarde l'image plutôt que de juste l'afficher
        plt.savefig(f"results/epoch_{epoch_num}.png")
        plt.close() # Ferme la figure pour libérer la mémoire
        print(f"🖼️ Image générée sauvegardée dans 'results/epoch_{epoch_num}.png'")

# --- 5. Main (Point d'entrée) ---

if __name__ == "__main__":
    # 1. Définition du chemin relatif
    # Récupère le dossier où se trouve ce fichier main.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    print(f"📂 Recherche des données dans : {data_dir}")

    # 2. Préparation
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # Initialisation du Dataset
        dataset = LocalCelebADataset(root_dir=data_dir, transform=transform)
        # num_workers=0 est plus sûr sur Windows pour éviter les erreurs de multiprocessing
        # Si vous êtes sur Linux/Mac, vous pouvez mettre num_workers=2
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # Modèle
        model = QuickDiffusionUNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss()

        print("🚀 Lancement de l'entraînement...")
        train(dataloader, model, optimizer, criterion)
        print("✅ Entraînement terminé !")

    except FileNotFoundError as e:
        print("\n❌ ERREUR CRITIQUE :")
        print(e)
        print(f"Assurez-vous d'avoir créé le dossier '{data_dir}' et d'y avoir mis vos images.")
    except Exception as e:
        print(f"\n❌ Une erreur inattendue est survenue : {e}")