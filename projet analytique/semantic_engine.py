
import numpy as np
from sentence_transformers import SentenceTransformer, util
from referentiel import COMPETENCY_BLOCKS, JOB_PROFILES, CSV_TO_BLOCKS, NUMERIC_COLUMNS

# Cache global du modele pour eviter de le recharger a chaque appel
_model = None


def get_model():
    #Charge le modele SBERT une seule fois (singleton).
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_block_scores(user_texts: dict) -> dict:
    model = get_model()
    block_scores = {block: [] for block in COMPETENCY_BLOCKS}

    for col_name, text in user_texts.items():
        if not text or not isinstance(text, str) or len(text.strip()) < 3:
            continue

        # Trouver quels blocs cette colonne alimente
        target_blocks = CSV_TO_BLOCKS.get(col_name, [])
        if not target_blocks:
            continue

        # Encoder le texte utilisateur
        user_embedding = model.encode(text, convert_to_tensor=True)

        for block_name in target_blocks:
            if block_name not in COMPETENCY_BLOCKS:
                continue

            competencies = COMPETENCY_BLOCKS[block_name]
            block_embeddings = model.encode(competencies, convert_to_tensor=True)

            # Similarite cosinus entre le texte et chaque competence du bloc
            similarities = util.cos_sim(user_embedding, block_embeddings)

            # Prendre la similarite max (meilleure correspondance dans le bloc)
            max_sim = float(similarities.max())
            block_scores[block_name].append(max_sim)

    # Moyenne des scores pour chaque bloc
    final_scores = {}
    for block, scores in block_scores.items():
        if scores:
            final_scores[block] = np.mean(scores)
        else:
            final_scores[block] = 0.0

    return final_scores


def integrate_numeric_scores(block_scores: dict, row: dict) -> dict:

    for col_name, config in NUMERIC_COLUMNS.items():
        value = row.get(col_name, 0)
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        block = config["block"]
        max_val = config["max_value"]
        weight = config["weight"]

        # Normaliser entre 0 et 1
        normalized = min(value / max_val, 1.0)

        # Combiner avec le score semantique existant
        if block in block_scores:
            semantic = block_scores[block]
            block_scores[block] = (1 - weight) * semantic + weight * normalized

    return block_scores


def compute_coverage_score(block_scores: dict, weights: dict = None) -> float:

    if weights is None:
        weights = {block: 1.0 for block in block_scores}

    numerator = sum(weights.get(b, 1.0) * s for b, s in block_scores.items())
    denominator = sum(weights.get(b, 1.0) for b in block_scores)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def recommend_jobs(block_scores: dict, top_n: int = 3) -> list:
    
    job_scores = {}

    for job_title, job_weights in JOB_PROFILES.items():
        score = compute_coverage_score(block_scores, job_weights)
        job_scores[job_title] = score

    # Trier par score decroissant et retourner le top N
    sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_jobs[:top_n]


def identify_weak_blocks(block_scores: dict, threshold: float = 0.5) -> list:
    weak = [(b, s) for b, s in block_scores.items() if s < threshold]
    return sorted(weak, key=lambda x: x[1])


def analyze_profile(row: dict) -> dict:
    # Extraire les colonnes textuelles
    user_texts = {}
    for col in CSV_TO_BLOCKS:
        if col in row and row[col]:
            user_texts[col] = str(row[col])

    # Calcul des scores semantiques par bloc
    block_scores = compute_block_scores(user_texts)

    # Integration des scores numeriques
    block_scores = integrate_numeric_scores(block_scores, row)

    # Score de couverture global
    coverage_score = compute_coverage_score(block_scores)

    # Recommandation de metiers (top 3)
    top_jobs = recommend_jobs(block_scores)

    # Blocs faibles
    weak_blocks = identify_weak_blocks(block_scores)

    return {
        "block_scores": block_scores,
        "coverage_score": coverage_score,
        "top_jobs": top_jobs,
        "weak_blocks": weak_blocks,
    }
