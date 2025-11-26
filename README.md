# Quick Diffusion — README

Ce dépôt contient un script Python simple et autonome (`main.py`) implémentant un petit modèle de diffusion (U-Net simplifié) pour entraîner et générer des images à partir d'un dataset local de type CelebA.

## 🎯 Objectif

Documenter `main.py` :
- comment préparer les données localement ;
- quelles sont les dépendances ;
- comment lancer l'entraînement ;
- où retrouver les sorties (checkpoints & images prévisualisées).

## 🧭 Résumé de `main.py`

- Le script crée un dataset local (`LocalCelebADataset`) qui cherche récursivement des images `.jpg` (ou `.png` si none found) dans le dossier `data/`.
- Un modèle simple de type U-Net temporel (`QuickDiffusionUNet`) est défini avec un bloc temporel (`TimeBlock`) pour injecter l'embedding du pas temporel.
- L'entraînement ajoute progressivement du bruit aux images (fonction `forward_noise`) et apprend à prédire un niveau de bruit suivant pour chaque pas.
- Pendant l'entraînement, des checkpoints sont sauvegardés dans `checkpoints/` et une prévisualisation globale est sauvegardée dans `results/` (images d'exemple par époque).
- Le script détecte automatiquement l'appareil : GPU CUDA, MPS (Mac M1/M2/M3) ou CPU.

## ⚙️ Hyperparamètres et options (définis en haut de `main.py`)
- IMG_SIZE = 64 — taille (H×W) des images trainées
- BATCH_SIZE = 64
- TIMESTEPS = 16 — nombre de pas de diffusion
- LR = 1e-4 — learning rate
- EPOCHS = 10

Ces variables peuvent être modifiées directement dans `main.py` pour expérimenter.

## 📁 Structure attendue du projet

- data/  <-- placez vos images ici (ex: `img_align_celeba/`)
- checkpoints/  <-- créé automatiquement par le script (sauvegarde `.pth`)
- results/  <-- créé automatiquement (sauvegarde `epoch_N.png`)

Remarque : Evitez d'ajouter `data/`, `*.pth` ou `results/` au dépôt Git — `.gitignore` a été ajouté pour ces fichiers.

## ✅ Dépendances

Le script utilise (extrait depuis `main.py`):
- Python 3.8+ (recommandé)
- torch
- torchvision
- numpy
- pillow (PIL)
- matplotlib
- tqdm

Exemple d'installation :

```powershell
pip install torch torchvision numpy pillow matplotlib tqdm
```

Si vous utilisez un GPU, installez la version de `torch` compatible avec votre CUDA.

## 🚀 Comment lancer l'entraînement

1. Mettez vos images dans le dossier `data/` (ou un sous-dossier : script cherche récursivement `*.jpg` / `*.png`).
2. (Optionnel) ajustez les hyperparamètres en tête de `main.py`.
3. Exécutez :

```powershell
python main.py
```

Remarques :
- Sur Windows, `DataLoader` utilise `num_workers=0` pour éviter des erreurs de multiprocessing. Sur Linux/Mac vous pouvez augmenter `num_workers`.
- Si vous sentez des problèmes de mémoire, réduisez `BATCH_SIZE` ou `IMG_SIZE`.

## 💾 Sorties / Checkpoints

- `checkpoints/model_ep{N}.pth` — états du modèle enregistrés après chaque époque
- `results/epoch_{N}.png` — grille d'images générées en fin d'époque (prévisualisation)

## 📌 Astuces et suggestions

- Les modèles et datasets peuvent être volumineux : si vous souhaitez suivre les `.pth` dans Git, configurez Git LFS (`git lfs track "*.pth"`) pour éviter d'avoir de gros fichiers git historiques.
- Si vous avez accidentellement committé de gros fichiers, je peux vous aider à les supprimer de l'historique (avec `git filter-repo` ou `bfg`).

## ❓ Prochaine étape — amélioration possible

- Ajouter un fichier `requirements.txt` pour simplifier l'installation.
- Ajouter des scripts CLI pour configurer hyperparamètres via des flags.
- Ajouter un petit notebook pour visualiser les images générées et l'évolution du training.

---

Si tu veux, je peux maintenant :
- ajouter `requirements.txt`,
- configurer Git LFS pour les `.pth`,
- ou commit & push ce `README.md` sur ton repo (je peux faire ça tout de suite).