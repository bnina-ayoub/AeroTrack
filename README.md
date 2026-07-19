# 🚁 AeroTrack: Dynamic Early-Exit Architecture for UAV Tracking

AeroTrack est une architecture de suivi d'objets (Object Tracking) optimisée pour les véhicules aériens sans pilote (UAV). Basée sur le framework YOLOX, notre approche introduit un mécanisme de routage dynamique innovant (**Early Exit**) piloté par un **Decision Gate**.

Cette architecture permet de court-circuiter les couches profondes du réseau lorsque la confiance de détection est suffisante, offrant une réduction drastique du coût computationnel (GFLOPs) sans sacrifier la précision du suivi grâce aux métriques de distance IoU et NWD.

---

## 🎥 Démonstration Vidéo

Regardez l'architecture AeroTrack (Early Exit + NWD) en action :



https://github.com/user-attachments/assets/ca8e1d86-d2df-4852-928f-89b6337f349b





---

## ⚙️ Installation et Configuration

Pour exécuter AeroTrack, vous devez configurer l'environnement Python et installer YOLOX.

### 1. Prérequis environnementaux

Il est recommandé d'utiliser un environnement virtuel (Conda ou venv) avec **Python 3.8+** et **PyTorch** compatible avec votre version de CUDA.

```bash
# Exemple avec venv
python3 -m venv venv
source venv/bin/activate
```

### 2. Installation des dépendances et de YOLOX

AeroTrack s'appuie sur le moteur YOLOX. Exécutez les commandes suivantes à la racine du projet pour installer les dépendances requises et lier le projet.

```bash
# Installation des dépendances de base
pip install -r requirements.txt

# Installation de YOLOX en mode développement
pip install -v -e .

# Installation des dépendances spécifiques au tracking (MOT)
pip install cython
pip install cython_bbox
pip install motmetrics
```

### 3. Préparation des poids (Weights)

Assurez-vous de placer votre fichier de poids entraîné (`early_exit_weights.pth`) dans le dossier `weights/` à la racine du projet.

---

## 🚀 Évaluation et Inférence

Nous avons mis en place un script d'automatisation robuste pour tester l'architecture sous toutes ses configurations de manière fluide.

### Exécution automatisée (Recommandé)

Le script `run_evaluations.sh` exécute automatiquement 4 expériences en croisant les métriques de distance (IoU / NWD) et l'activation du routage dynamique (Baseline / Early Exit).

Pour lancer l'évaluation complète :

```bash
# 1. Donner les droits d'exécution au script
chmod +x run_evaluations.sh

# 2. Lancer l'évaluation
./run_evaluations.sh
```

Le script exécutera la commande suivante en coulisse pour chaque mode :

```bash
python tools/track.py --fp16 --fuse -d 1 -b 1 -f exps/aerotrack_proposed.py -c weights/early_exit_weights.pth --distance <metric> [--early_exit] --save_vis
```

### Analyse des Résultats

Pour chaque expérience, AeroTrack générera un dossier de résultats spécifique contenant :

- **`mot_evaluation_metrics.csv`** : Les résultats détaillés du tracking (MOTA, IDF1, FPS, etc.).
- **`early_exit_stats.csv`** : Le ratio exact des trames ayant emprunté le chemin court et les GFLOPs effectifs économisés.
- **`track_vis/`** : Un dossier contenant les visualisations image par image du suivi de l'UAV avec l'indication du chemin emprunté (Early Exit ou Full).

---

## 📝 Structure du Projet

- **`tools/track.py`** : Script principal pour lancer l'inférence et le suivi MOT.
- **`exps/aerotrack_proposed.py`** : Fichier de définition de notre architecture unifiée.
- **`run_evaluations.sh`** : Script Bash pour l'automatisation des évaluations comparatives.
- **`yolox/`** : Code source du modèle contenant la logique du `DecisionGate` et de l'`EarlyHead`.
