"""
questionnaire.py — Définition des questions du questionnaire hybride AISCA.

Le questionnaire est structuré en 3 types de questions :
  1. Questions Likert (échelle 1-5) : auto-évaluation par bloc de compétences
  2. Questions ouvertes (texte libre) : expériences concrètes, projets
  3. Questions à choix multiples : sélection de compétences maîtrisées

Les questions sont définies comme des structures de données pures (sans Streamlit)
pour rester testables indépendamment.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Questions Likert (auto-évaluation 1-5 par domaine)
# ---------------------------------------------------------------------------

QUESTIONS_LIKERT: List[Dict] = [
    {
        "id": "L01",
        "bloc_id": 1,
        "texte": "Quel est votre niveau en analyse de données et statistiques ?",
        "description": "Exploration, visualisation, métriques statistiques, SQL analytique.",
        "type": "likert",
        "min_label": "Débutant",
        "max_label": "Expert",
    },
    {
        "id": "L02",
        "bloc_id": 2,
        "texte": "Comment évaluez-vous vos compétences en Machine Learning ?",
        "description": "Régression, classification, évaluation de modèles, deep learning.",
        "type": "likert",
        "min_label": "Débutant",
        "max_label": "Expert",
    },
    {
        "id": "L03",
        "bloc_id": 3,
        "texte": "Quel est votre niveau en Traitement du Langage Naturel (NLP) ?",
        "description": "Tokenisation, classification textuelle, analyse de sentiment, LLMs.",
        "type": "likert",
        "min_label": "Débutant",
        "max_label": "Expert",
    },
    {
        "id": "L04",
        "bloc_id": 4,
        "texte": "Comment maîtrisez-vous l'ingénierie des données (Data Engineering) ?",
        "description": "Pipelines ETL/ELT, bases de données, big data, orchestration.",
        "type": "likert",
        "min_label": "Débutant",
        "max_label": "Expert",
    },
    {
        "id": "L05",
        "bloc_id": 5,
        "texte": "Évaluez votre niveau en développement logiciel et programmation.",
        "description": "Python, Git, tests unitaires, APIs REST, conteneurs Docker.",
        "type": "likert",
        "min_label": "Débutant",
        "max_label": "Expert",
    },
    {
        "id": "L06",
        "bloc_id": 6,
        "texte": "Comment évaluez-vous vos compétences en gestion de projet et soft skills ?",
        "description": "Planification, communication, travail en équipe, gestion des risques.",
        "type": "likert",
        "min_label": "Débutant",
        "max_label": "Expert",
    },
]

# ---------------------------------------------------------------------------
# Questions ouvertes (texte libre)
# ---------------------------------------------------------------------------

QUESTIONS_OUVERTES: List[Dict] = [
    {
        "id": "O01",
        "texte": "Décrivez votre expérience la plus significative en analyse ou manipulation de données.",
        "placeholder": "Ex : J'ai analysé des données de ventes pour identifier des tendances...",
        "type": "ouverte",
        "aide": "Mentionnez les outils utilisés, les résultats obtenus et les défis surmontés.",
    },
    {
        "id": "O02",
        "texte": "Avez-vous réalisé des projets de Machine Learning ou d'IA ? Si oui, décrivez le plus représentatif.",
        "placeholder": "Ex : J'ai construit un modèle de classification pour détecter des fraudes...",
        "type": "ouverte",
        "aide": "Précisez les algorithmes, frameworks et métriques d'évaluation utilisés.",
    },
    {
        "id": "O03",
        "texte": "Décrivez vos expériences avec le traitement du langage naturel ou le texte.",
        "placeholder": "Ex : J'ai implémenté une analyse de sentiment sur des avis clients...",
        "type": "ouverte",
        "aide": "Indiquez les bibliothèques (spaCy, HuggingFace, NLTK) et les cas d'usage.",
    },
    {
        "id": "O04",
        "texte": "Quelles sont vos compétences en infrastructure et ingénierie des données ?",
        "placeholder": "Ex : J'ai conçu un pipeline ETL avec Airflow et BigQuery...",
        "type": "ouverte",
        "aide": "Mentionnez les technologies de stockage, traitement et orchestration maîtrisées.",
    },
    {
        "id": "O05",
        "texte": "Présentez un projet où vous avez développé du code de qualité production.",
        "placeholder": "Ex : J'ai développé une API REST avec FastAPI déployée avec Docker...",
        "type": "ouverte",
        "aide": "Incluez des détails sur les tests, la documentation et les bonnes pratiques.",
    },
    {
        "id": "O06",
        "texte": "Décrivez une expérience de collaboration ou de gestion de projet data.",
        "placeholder": "Ex : J'ai coordonné une équipe de 3 personnes pour livrer un dashboard BI...",
        "type": "ouverte",
        "aide": "Mentionnez votre rôle, les méthodologies (Agile, Scrum) et les parties prenantes.",
    },
]

# ---------------------------------------------------------------------------
# Questions à choix multiples (compétences spécifiques)
# ---------------------------------------------------------------------------

QUESTIONS_CHOIX: List[Dict] = [
    {
        "id": "CM01",
        "texte": "Quels outils et langages maîtrisez-vous pour l'analyse de données ?",
        "type": "choix_multiple",
        "options": [
            "Python (pandas, numpy)",
            "SQL avancé et optimisation de requêtes",
            "Visualisation avec Matplotlib / Seaborn / Plotly",
            "Tableaux de bord BI (Power BI, Tableau, Looker)",
            "Statistiques descriptives et inférentielles",
            "R pour l'analyse statistique",
        ],
    },
    {
        "id": "CM02",
        "texte": "Quels frameworks ou algorithmes ML avez-vous utilisés en production ?",
        "type": "choix_multiple",
        "options": [
            "scikit-learn (régression, classification, clustering)",
            "TensorFlow / Keras",
            "PyTorch",
            "XGBoost / LightGBM / CatBoost",
            "AutoML (H2O, AutoSklearn)",
            "MLflow ou autre outil de tracking d'expériences",
        ],
    },
    {
        "id": "CM03",
        "texte": "Quelles technologies NLP avez-vous pratiquées ?",
        "type": "choix_multiple",
        "options": [
            "Hugging Face Transformers",
            "spaCy ou NLTK",
            "Fine-tuning de modèles BERT/GPT",
            "Systèmes RAG (Retrieval-Augmented Generation)",
            "Embeddings textuels (Word2Vec, FastText, SBERT)",
            "LangChain ou frameworks LLM",
        ],
    },
    {
        "id": "CM04",
        "texte": "Quelles technologies de Data Engineering avez-vous utilisées ?",
        "type": "choix_multiple",
        "options": [
            "Apache Spark ou PySpark",
            "Airflow ou autre orchestrateur de workflows",
            "dbt (data build tool)",
            "Kafka ou systèmes de streaming",
            "Cloud data warehouses (BigQuery, Redshift, Snowflake)",
            "Hadoop ou ecosystème Big Data",
        ],
    },
    {
        "id": "CM05",
        "texte": "Quelles pratiques de développement logiciel appliquez-vous ?",
        "type": "choix_multiple",
        "options": [
            "Git et GitFlow",
            "Tests unitaires (pytest, unittest)",
            "Docker et containerisation",
            "APIs REST ou gRPC",
            "CI/CD (GitHub Actions, GitLab CI)",
            "Kubernetes ou orchestration de conteneurs",
        ],
    },
]

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def valider_reponses(
    reponses_likert: Dict[str, int],
    reponses_ouvertes: Dict[str, str],
    reponses_choix: Dict[str, List[str]]
) -> List[str]:
    """
    Valide les réponses soumises par l'utilisateur.

    Retourne une liste d'erreurs (vide = tout est valide).
    """
    erreurs = []

    # Validation Likert : valeurs entre 1 et 5
    for q_id, valeur in reponses_likert.items():
        if not isinstance(valeur, (int, float)) or not (1 <= valeur <= 5):
            erreurs.append(f"Question {q_id} : la note doit être entre 1 et 5 (reçu : {valeur})")

    # Validation ouvertes : au moins quelques questions renseignées
    nb_renseignees = sum(
        1 for v in reponses_ouvertes.values()
        if isinstance(v, str) and v.strip()
    )
    if nb_renseignees == 0:
        erreurs.append(
            "Veuillez renseigner au moins une réponse ouverte pour permettre l'analyse sémantique."
        )

    return erreurs


def reponses_vers_textes(
    reponses_ouvertes: Dict[str, str],
    reponses_likert: Dict[str, int],
    reponses_choix: Dict[str, List[str]]
) -> List[str]:
    """
    Convertit toutes les réponses du questionnaire en une liste de textes
    exploitables par le moteur NLP.

    Délègue à preprocessing.fusionner_reponses() pour la logique de fusion.
    """
    from src.preprocessing import fusionner_reponses

    textes_libres = [v for v in reponses_ouvertes.values() if v and v.strip()]
    return fusionner_reponses(textes_libres, reponses_likert, reponses_choix)
