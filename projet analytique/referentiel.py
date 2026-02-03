
# Chaque bloc contient des phrases-cles decrivant les competences attendues.
# Ces phrases servent de cibles pour la comparaison semantique via SBERT.

COMPETENCY_BLOCKS = {
    "Data Analysis": [
        "data cleaning and preprocessing",
        "data visualization and dashboards",
        "Python programming for data analysis",
        "statistical analysis and hypothesis testing",
        "data wrangling and transformation",
        "exploratory data analysis",
        "reporting and business intelligence",
    ],
    "Machine Learning": [
        "classification algorithms and supervised learning",
        "regression models and prediction",
        "neural networks and deep learning",
        "model evaluation and validation",
        "model training and optimization",
        "feature engineering and selection",
        "ensemble methods and boosting",
    ],
    "NLP": [
        "tokenization and text preprocessing",
        "word embeddings and vector representations",
        "transformers and attention mechanisms",
        "semantic analysis and text understanding",
        "text classification and sentiment analysis",
        "language models and text generation",
        "named entity recognition and information extraction",
    ],
    "Software Engineering": [
        "API development and integration",
        "ETL pipelines and data engineering",
        "automation and scripting",
        "deployment and production systems",
        "CI/CD and version control",
        "software architecture and design patterns",
        "testing and quality assurance",
    ],
}


# Chaque profil definit les poids par bloc de competences.
# Poids plus eleve = bloc plus important pour ce poste.

JOB_PROFILES = {
    "Data Analyst": {
        "Data Analysis": 2.0,
        "Machine Learning": 1.0,
        "NLP": 0.5,
        "Software Engineering": 1.0,
    },
    "ML Engineer": {
        "Data Analysis": 1.0,
        "Machine Learning": 1.5,
        "NLP": 1.0,
        "Software Engineering": 1.5,
    },
    "Data Scientist": {
        "Data Analysis": 1.5,
        "Machine Learning": 1.5,
        "NLP": 1.5,
        "Software Engineering": 1.0,
    },
    "NLP Engineer": {
        "Data Analysis": 1.0,
        "Machine Learning": 1.0,
        "NLP": 2.0,
        "Software Engineering": 1.0,
    },
    "Data Engineer": {
        "Data Analysis": 1.0,
        "Machine Learning": 1.0,
        "NLP": 0.3,
        "Software Engineering": 1.5,
    },
}


# Indique quelles colonnes du CSV alimentent quels blocs pour l'analyse semantique.
# Une colonne peut alimenter plusieurs blocs.

CSV_TO_BLOCKS = {
    "Use Case Python": ["Data Analysis", "Software Engineering"],
    "Neural Network": ["Machine Learning", "NLP"],
    "Tokenisation": ["NLP"],
    "Regression": ["Machine Learning"],
    "Neural Work": ["Machine Learning", "NLP"],
    "NLP": ["NLP"],
    "Tokenisation Method": ["NLP"],
}

# --- Colonnes numeriques ---
# Colonnes du CSV qui fournissent un score numerique (non semantique).

NUMERIC_COLUMNS = {
    "Years Python": {"block": "Data Analysis", "max_value": 20, "weight": 0.3},
    "Python Skill": {"block": "Data Analysis", "max_value": 20, "weight": 0.3},
}
