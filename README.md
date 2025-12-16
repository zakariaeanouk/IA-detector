# Projet Transformation Digital — AI Detector

Résumé
-------
Ce dépôt présente un projet de détection de texte généré par intelligence artificielle.
Il inclut des notebooks pour la collecte, le nettoyage et l'entraînement, ainsi qu'une application Flask pour tester des textes localement.

Structure du dépôt
------------------
- `AI-Detector-main/` : service Flask, fonctions utilitaires, modèles et templates.
- `cleaned_data.csv` : jeu de données nettoyé (CSV).
- `Data_collection&Transformation.ipynb` : extraction et transformation des données brutes.
- `Data_Analysis_and_Cleaning.ipynb` : exploration et nettoyage des données.
- `Model_Training.ipynb` : entraînement et export des artefacts modèles.
- `datasets.zip` (optionnel) : archive des jeux de données bruts si fournie.

Prérequis
---------
- Python 3.8+
- Virtualenv recommandé
- macOS / Linux

Installation (rapide)
---------------------
1. Créer et activer un environnement virtuel :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installer les dépendances :

```bash
pip install -r AI-Detector-main/requirements.txt
```

3. (Optionnel) Installer les données NLTK si nécessaire :

```bash
python -m nltk.downloader punkt wordnet stopwords
```

Modèles et artefacts requis
---------------------------
L'application Flask attend les fichiers suivants dans `AI-Detector-main/Model/` :

- `logistic_regression_model.pkl` — modèle entraîné
- `model_vectorizer.pkl` — vectoriseur (ex: `TfidfVectorizer`)
- `scaler.pkl` — scaler pour normalisation des features

Si ces artefacts manquent, exécutez `Model_Training.ipynb` pour entraîner et exporter
les fichiers dans le dossier `AI-Detector-main/Model/`.

Exécution de l'application
--------------------------
Depuis la racine du projet :

```bash
cd "AI-Detector-main"
python app.py
```

Ouvrir la page web à l'adresse : http://127.0.0.1:5000

Notebooks et reproduction
-------------------------
- `Data_collection&Transformation.ipynb` : reconstruire/collecter les données brutes.
- `Data_Analysis_and_Cleaning.ipynb` : appliquer le pipeline de nettoyage et produire `cleaned_data.csv`.
- `Model_Training.ipynb` : entraîner le modèle et sauvegarder les artefacts.

Jeu de données
---------------
- `cleaned_data.csv` est fourni (jeu nettoyé). Si vous avez `datasets.zip`, décompressez-le
  pour récupérer les sources brutes :

```bash
unzip datasets.zip
```

Bonnes pratiques et code
------------------------
- Les fonctions de prétraitement sont dans `AI-Detector-main/Functions/` (`Cleaning.py`, `Features.py`).
- Les notebooks servent également de documentation reproductible.
