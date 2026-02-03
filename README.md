# AISCA - Agent Intelligent Semantique pour la Cartographie des Competences

Application d'evaluation des competences en NLP/IA avec recommandation de metiers.

## Installation

```bash
# Creer un environnement virtuel
python -m venv .venv

# Activer l'environnement
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Installer les dependances
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur a l'adresse `http://localhost:8501`

## Utilisation

### 1. Remplir le questionnaire
- Entrez votre nom et prenom
- Repondez aux 9 questions sur vos competences (Python, ML, NLP, reseaux neuronaux)
- Cliquez sur **Submit**

### 2. Analyser un profil
- Selectionnez un profil dans la liste deroulante
- Cliquez sur **Analyse**
- Consultez les resultats :
  - Scores par bloc (Data Analysis, ML, NLP, Software Engineering)
  - Graphique radar et heatmap
  - Score de couverture global
  - Top 3 metiers recommandes
  - Competences a developper

### 3. Generer une biographie (optionnel)
- Cliquez sur **Generer la biographie**
- Necessite Ollama avec le modele Gemma3 installe
https://docs.ollama.com/quickstart#python

## Structure du projet

```
projet analytique/
├── app.py              # Interface Streamlit
├── semantic_engine.py  # Moteur d'analyse SBERT
├── referentiel.py      # Referentiel des competences
├── ai.py               # Generation IA (Ollama)
├── results.csv         # Base de donnees candidats
└── requirements.txt    # Dependances
```

## Technologies

- **Streamlit** - Interface web
- **Sentence-Transformers** - Embeddings semantiques (all-MiniLM-L6-v2)
- **Pandas/NumPy** - Traitement des donnees
- **Matplotlib** - Visualisations
