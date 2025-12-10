import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Importez votre modèle depuis votre fichier main.py
# Assurez-vous que main.py est dans le même dossier
from main import QuickDiffusionUNet, IMG_SIZE, TIMESTEPS, DEVICE

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/model_ep650.pth" # Le chemin vers votre modèle entraîné
IMAGE_GUIDE = "image2.png"              # L'image à modifier (dessin ou photo)
START_STEP = 8                              # Entre 0 (Bruit total) et 16 (Image nette)
                                            # Essayez 5, 8, 10 pour voir la différence.

# --- CHARGEMENT DU MODÈLE ---
print(f"Chargement du modèle depuis {MODEL_PATH}...")
model = QuickDiffusionUNet().to(DEVICE)

# 1. Charger le dictionnaire brut (avec le warning de sécurité qui est normal)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# 2. CORRECTION DES CLÉS (Hack pour torch.compile)
# On retire le préfixe "_orig_mod." que torch.compile a ajouté pendant l'entraînement
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("_orig_mod."):
        new_key = key.replace("_orig_mod.", "") # On enlève le préfixe
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# 3. Chargement propre
model.load_state_dict(new_state_dict)
model.eval()
print("✅ Modèle chargé avec succès !")

# --- PRÉPARATION DE LA BARRE DE TEMPS (Même logique que l'entraînement) ---
# Rappel : 0 = Bruit pur, 1.0 = Image nette dans votre logique de code précédente ?
# Vérifions votre code : time_bar = 1 - np.linspace(0, 1.0, TIMESTEPS + 1)
# Donc time_bar[0] = 1.0 (Alpha=1, Bruit pur)
# Donc time_bar[16] = 0.0 (Alpha=0, Image nette)
time_bar = 1 - np.linspace(0, 1.0, TIMESTEPS + 1)

def load_and_process_image(path):
    """Charge une image, la redimensionne et la transforme en tenseur"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    try:
        img = Image.open(path).convert("RGB")
        return transform(img).unsqueeze(0).to(DEVICE) # Ajoute la dimension Batch
    except FileNotFoundError:
        print(f"Erreur : Impossible de trouver l'image {path}")
        return None

def sdedit_inpainting(model, x_guide, start_step):
    """
    x_guide : L'image originale (tenseurs)
    start_step : L'étape où on commence (0=Bruit, 16=Net)
    """
    
    # 1. BRUITAGE PARTIEL (Forward Process)
    # On ajoute du bruit correspondant à l'étape 'start_step'
    noise = torch.randn_like(x_guide).to(DEVICE)
    
    # On récupère le coefficient de bruit pour cette étape
    # Attention : dans votre time_bar, l'index 0 est le plus bruité
    # Donc si on veut commencer "un peu bruité", il faut viser un index vers 6-10.
    t_idx = start_step 
    
    a = time_bar[t_idx] # Coefficient de bruit
    
    # Formule de mélange : Image * (1-a) + Bruit * a
    # Plus 'a' est grand, plus on détruit l'image
    x_noisy = x_guide * (1 - a) + noise * a
    
    current_img = x_noisy
    
    print(f"Début de la génération à l'étape {start_step} (Alpha={a:.2f})...")

    # 2. DÉBRUITAGE (Reverse Process)
    # On part de l'étape start_step et on remonte jusqu'à la fin (TIMESTEPS)
    with torch.no_grad():
        for i in range(start_step, TIMESTEPS):
            # On crée le tenseur de temps pour le batch
            t_input = torch.full((1,), i, device=DEVICE, dtype=torch.long)
            
            # Le modèle prédit l'étape suivante (plus nette)
            current_img = model(current_img, t_input)
            
    return current_img

# --- EXÉCUTION ---

# 1. Charger l'image guide
x_original = load_and_process_image(IMAGE_GUIDE)

if x_original is not None:
    # 2. Lancer l'Inpainting
    # On essaie plusieurs niveaux pour comparer
    steps_to_test = [0, 5, 10, 13] 
    # 0 = Le modèle invente tout (ignore l'image)
    # 10 = Le modèle modifie un peu (garde la structure)
    
    results = []
    titles = []
    
    # On ajoute l'original pour comparer
    results.append(x_original)
    titles.append("Original")

    for step in steps_to_test:
        out = sdedit_inpainting(model, x_original, step)
        results.append(out)
        titles.append(f"Start Step {step}")

    # 3. Affichage
    plt.figure(figsize=(15, 5))
    for i in range(len(results)):
        plt.subplot(1, len(results), i+1)
        
        # Conversion pour affichage
        img_disp = results[i].squeeze(0).cpu().clamp(-1, 1)
        img_disp = (img_disp + 1) / 2 # Denormalize -1,1 -> 0,1
        img_disp = img_disp.permute(1, 2, 0).numpy()
        
        plt.imshow(img_disp)
        plt.title(titles[i])
        plt.axis('off')
    
    plt.show()
    plt.savefig("inpainting_result.png")
    print("Résultat sauvegardé dans inpainting_result.png")