# Génération d'images par modèle de diffusion

Ce dépôt contient un prototype de modèle de diffusion et des scripts pour entraîner et produire des images (inpainting / génération conditionnelle simple) à partir d'un dataset local (ex. CelebA).

L'objectif du projet : expérimenter une version simple d'un diffusion model (U-Net temporel) pour comprendre la chaîne complète — préparation des données, entraînement, sauvegarde de checkpoints et génération d'images de démonstration.

**Langue** : français

## Contenu principal

- `main.py` — script principal pour entraîner le modèle et sauvegarder checkpoints + images de prévisualisation.
- `inpainting.py` — script/utilitaire pour exécuter des tâches d'inpainting (si présent dans le dépôt).
- `data/` — dossier attendu contenant les images d'entraînement (ex. `img_align_celeba/`).
- `checkpoints/` — dossiers de sauvegarde des modèles (fichiers `.pth`).
- `results/` — images de sortie (grilles d'exemples générées pendant/à la fin des époques).

## Ce que fait le code

- Prépare un dataset local en recherchant des images dans `data/`.
- Définit un petit U-Net temporel qui reçoit l'embedding du pas de diffusion.
- Entraîne le modèle à prédire le bruit ajouté à une image (schéma de diffusion simplifié).
- Sauvegarde périodiquement les checkpoints dans `checkpoints/` et des images d'exemple dans `results/`.

## Structure du projet

- `data/` — placez votre dataset ici (structure récursive acceptée).
- `checkpoints/` — créé automatiquement par le script.
- `results/` — créé automatiquement par le script.

Ne commitez pas vos datasets ou checkpoints volumineux dans Git. Utilisez `.gitignore` (déjà présent) et, si besoin, Git LFS pour les fichiers `.pth`.

## Dépendances

Recommandé : Python 3.8+.

Bibliothèques principales :

- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `Pillow`
- `matplotlib`
- `tqdm`

Exemple d'installation (Windows PowerShell) :

```powershell
python -m pip install --upgrade pip
pip install torch torchvision numpy pillow matplotlib tqdm
```

Pour utiliser CUDA, installez la version de `torch` compatible avec votre version de CUDA (voir https://pytorch.org/).

## Exécution

1. Placez vos images dans `data/` (le script recherche récursivement `*.jpg` / `*.png`).
2. (Optionnel) modifiez les hyperparamètres directement dans `main.py` (taille d'image, batch, timesteps, epochs, etc.).
3. Lancez l'entraînement :

```powershell
python main.py
```

Notes :
- Sur Windows, `DataLoader` peut être configuré avec `num_workers=0` pour éviter des soucis de multiprocessing.
- Si vous manquez de mémoire GPU/CPU, diminuez `BATCH_SIZE` ou `IMG_SIZE`.

## Génération / Inpainting

Si `inpainting.py` est fourni, il offre des utilitaires pour lancer de l'inpainting en chargeant un checkpoint existant depuis `checkpoints/`. Exemple d'usage (varie selon l'implémentation) :

```powershell
python inpainting.py --checkpoint checkpoints/model_ep200.pth --input examples/masked.png --output results/inpainted.png
```

Consultez l'entête des scripts pour les options disponibles.

## Sorties

- `checkpoints/model_ep{N}.pth` — états du modèle sauvegardés par époque.
- `results/epoch_{N}.png` — images d'exemple générées pour visualiser la progression de l'entraînement.

## Bonnes pratiques Git

- Ne versionnez pas `data/` ni de gros fichiers de checkpoints.
- Pour suivre les checkpoints volumineux, utilisez Git LFS :

```powershell
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

Si vous avez déjà committé de gros fichiers, on peut nettoyer l'historique sur demande.

## Suggestions d'amélioration

- Ajouter un `requirements.txt` et/ou `environment.yml`.
- Ajouter des notebooks d'analyse et d'évaluation.
- Ajouter des scripts CLI pour entraînement/génération indépendants.

## Contact / Aide

Si tu souhaites, je peux :
- ajouter un `requirements.txt` ;
- configurer Git LFS pour les `.pth` ;
- committer et pousser ce README vers GitHub (je peux le faire maintenant).

---

Merci — dis-moi si tu veux que j'ajoute aussi `requirements.txt` ou que je pousse ce changement sur GitHub maintenant.