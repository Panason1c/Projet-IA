"""
config.py — Constantes globales du projet AISCA.
Centralise tous les chemins, paramètres du modèle, seuils et poids.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins du projet
# ---------------------------------------------------------------------------

# Racine du projet (dossier parent de src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Données de référence
DATA_DIR = ROOT_DIR / "data"
COMPETENCES_CSV = DATA_DIR / "competences.csv"
METIERS_CSV = DATA_DIR / "metiers.csv"
RESPONSES_DIR = DATA_DIR / "responses"

# Cache GenAI
CACHE_DIR = ROOT_DIR / "cache"
GENAI_CACHE_FILE = CACHE_DIR / "genai_cache.json"

# Embeddings pré-calculés du référentiel (optionnel, accélère les relances)
EMBEDDINGS_CACHE_FILE = DATA_DIR / "competences_embeddings.npy"
COMPETENCY_IDS_CACHE_FILE = DATA_DIR / "competences_ids.json"

# ---------------------------------------------------------------------------
# Modèle NLP local (SBERT)
# ---------------------------------------------------------------------------

# Modèle sentence-transformers utilisé pour les embeddings
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Scoring — seuils d'interprétation du score de couverture
# ---------------------------------------------------------------------------

SEUIL_BON = 0.7        # Score >= 0.7 : bonne couverture
SEUIL_MOYEN = 0.5      # Score 0.5–0.7 : couverture partielle
# Score < 0.5 : couverture insuffisante

# ---------------------------------------------------------------------------
# Poids des blocs (Wi dans la formule Coverage = Σ(Wi·Si) / Σ(Wi))
# Clés = BlockID (entier), valeurs = poids (float)
# Modifier ici pour personnaliser l'importance relative de chaque bloc.
# ---------------------------------------------------------------------------

BLOCK_WEIGHTS = {
    1: 1.0,  # Data Analysis
    2: 1.0,  # Machine Learning
    3: 1.0,  # NLP
    4: 1.0,  # Data Engineering
    5: 1.0,  # Software Development & Programming
    6: 1.0,  # Project Management & Soft Skills
}

# ---------------------------------------------------------------------------
# GenAI — paramètres Gemini
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"

# Seuil de détection d'une saisie trop courte (enrichissement optionnel)
MIN_WORDS_THRESHOLD = 5

# Nombre de métiers recommandés dans le top-N
TOP_N_JOBS = 3

# Nombre de compétences faibles à inclure dans le prompt de progression
TOP_WEAK_COMPETENCIES = 5

# ---------------------------------------------------------------------------
# Stockage des réponses utilisateur
# ---------------------------------------------------------------------------

# Format de sauvegarde : "csv" ou "json"
STORAGE_FORMAT = "json"
