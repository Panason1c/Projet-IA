# AISCA — Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences

Projet EFREI — RNCP40875 — Bloc 2 (BC2 : Piloter et implémenter des solutions d'IA en s'aidant de l'IA générative).

Mini-agent RAG : questionnaire hybride → analyse sémantique SBERT locale → score de couverture pondéré → recommandation top-3 métiers → plan de progression et bio via Gemini (avec cache).

---

## Prérequis

- Python 3.10 ou supérieur
- pip

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Panason1c/Projet-IA.git
cd Projet-IA

# 2. Créer un environnement virtuel (isole les dépendances du système)
python -m venv .venv

# 3. L'activer
#   Windows (PowerShell) :
.\.venv\Scripts\Activate.ps1
#   Windows (invite CMD)  :   .venv\Scripts\activate.bat
#   Linux / macOS         :   source .venv/bin/activate

# 4. Installer les dépendances
python -m pip install -r requirements.txt

# 5. (Optionnel) Configurer la clé API Gemini — l'app fonctionne sans
#   Windows : copy .env.example .env   |   Linux/macOS : cp .env.example .env
#   'https://aistudio.google.com/app/apikey'
# Puis éditer .env et renseigner :  GEMINI_API_KEY=<votre_clé>
```

> **Windows — PowerShell bloque `Activate.ps1` ?** Soit autoriser pour la session :
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, soit ne pas
> activer du tout et préfixer chaque commande par le Python du venv (voir ci-dessous).

---

## Lancement

```bash
# Méthode recommandée (robuste : ne dépend ni du PATH ni de l'activation)
python -m streamlit run app.py
```

> Si `streamlit` n'est pas reconnu comme commande, c'est normal : le dossier `Scripts`
> de pip n'est pas toujours dans le PATH. Passer par `python -m streamlit` règle le
> problème. Sur Windows **sans activer le venv**, utiliser directement son interpréteur :
> `.\.venv\Scripts\python.exe -m streamlit run app.py`

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`.

---

## Fonctionnement sans clé API

Si `GEMINI_API_KEY` n'est pas définie, l'application fonctionne en mode dégradé :
- Le questionnaire, l'analyse SBERT, le scoring et la recommandation sont pleinement opérationnels.
- Le plan de progression et la bio professionnelle affichent un message de remplacement.

---

## Structure du projet

```
AISCA/
│
├── app.py                      # Interface Streamlit — point d'entrée de l'application
├── requirements.txt            # Dépendances Python
├── README.md                   # Présentation, installation, lancement (ce fichier)
├── ARCHITECTURE.md             # Architecture détaillée et découpage en lots
├── .env.example                # Modèle de configuration (clé API Gemini)
│
├── src/                        # Code métier (testable indépendamment de Streamlit)
│   ├── config.py               # Constantes : chemins, seuils, poids des blocs
│   ├── questionnaire.py        # Questions (Likert / ouvertes / choix multiples)
│   ├── preprocessing.py        # Nettoyage léger du texte utilisateur
│   ├── referentiel.py          # Chargement et validation des CSV
│   ├── nlp_engine.py           # SBERT : encodage et similarité cosinus
│   ├── scoring.py              # Score de couverture pondéré  Σ(Wi·Si) / Σ(Wi)
│   ├── recommender.py          # Score par métier, top-3, contexte RAG
│   ├── genai.py                # Appels Gemini (plan + bio) avec repli déterministe
│   ├── cache.py                # Cache JSON des appels GenAI (clé SHA-256)
│   └── storage.py              # Persistance des sessions utilisateur
│
├── data/                       # Données de référence
│   ├── competences.csv         # 36 compétences réparties en 6 blocs
│   ├── metiers.csv             # 8 métiers et leurs compétences requises
│   └── README.md               # Origine et structure des données (ROME / e-CF)
│
├── docs/                       # Livrables documentaires
│   ├── RAPPORT.md              # Rapport de projet (+ mapping RNCP Bloc 2)
│   └── PRESENTATION.md         # Trame de présentation (≈ 12 slides)
│
└── tests/                      # Tests unitaires pytest
    ├── test_referentiel.py
    ├── test_nlp_engine.py
    ├── test_scoring.py
    ├── test_recommender.py
    └── test_cache.py
```

> **Généré automatiquement** (exclu du dépôt via `.gitignore`, ne pas versionner) :
> `cache/genai_cache.json`, `data/competences_embeddings.npy`, `data/responses/`,
> `.venv/`, `__pycache__/`, `.pytest_cache/`.

---

## Pipeline RAG

1. Questionnaire hybride (Likert + texte libre + cases à cocher)
2. Prétraitement du texte
3. Embeddings SBERT locaux (`all-MiniLM-L6-v2`) — coût zéro
4. Similarité cosinus réponses utilisateur ↔ compétences
5. Score de couverture pondéré par bloc : `Coverage = Σ(Wi·Si) / Σ(Wi)`
6. Recommandation top-3 métiers
7. GenAI Gemini (plan de progression + bio) avec cache obligatoire
8. Visualisations radar et barres

---

## Variables d'environnement

| Variable | Description | Requis |
|----------|-------------|--------|
| `GEMINI_API_KEY` | Clé API Google Gemini (free-tier) | Non (dégradation propre) |
