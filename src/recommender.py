"""
recommender.py — Recommandation des métiers les mieux couverts par le profil utilisateur.

Algorithme :
  Pour chaque métier :
    1. Récupérer les scores de similarité de ses compétences requises
       (depuis scores_competences_detail de nlp_engine.py).
    2. Score du métier = moyenne des scores de ses compétences requises.
  Trier les métiers par score décroissant → top-3.

Expose aussi l'analyse des écarts (compétences à renforcer).
"""

from typing import Dict, List

from src.config import TOP_N_JOBS, TOP_WEAK_COMPETENCIES


def calculer_score_metier(
    metier_info: Dict,
    scores_competences: Dict[str, float]
) -> float:
    """
    Calcule le score de couverture d'un métier à partir des scores
    individuels de compétences.

    Score métier = moyenne des scores des compétences requises du métier.

    Paramètres :
      - metier_info       : dict {title, competency_ids, ...}
      - scores_competences: dict {CompetencyID: score_float}

    Retourne un float dans [0, 1].
    """
    ids_requis = metier_info.get("competency_ids", [])
    if not ids_requis:
        return 0.0

    scores = [scores_competences.get(cid, 0.0) for cid in ids_requis]
    return sum(scores) / len(scores)


def top_n_metiers(
    metiers: Dict[str, Dict],
    scores_competences: Dict[str, float],
    n: int = TOP_N_JOBS
) -> List[Dict]:
    """
    Calcule le score de chaque métier et retourne les n mieux couverts.

    Paramètres :
      - metiers            : dict {JobID: {title, competency_ids, ...}}
      - scores_competences : dict {CompetencyID: score_float}
      - n                  : nombre de métiers à retourner

    Retourne une liste de dicts triée par score décroissant :
    [
        {
            "job_id": str,
            "title": str,
            "score": float,
            "competency_ids": [...],
            "competency_scores": {CompetencyID: score}
        },
        ...
    ]
    """
    resultats = []

    for job_id, metier_info in metiers.items():
        score = calculer_score_metier(metier_info, scores_competences)
        comp_scores = {
            cid: scores_competences.get(cid, 0.0)
            for cid in metier_info.get("competency_ids", [])
        }
        resultats.append({
            "job_id": job_id,
            "title": metier_info["title"],
            "score": score,
            "competency_ids": metier_info.get("competency_ids", []),
            "competency_scores": comp_scores
        })

    # Tri décroissant par score, puis par title pour l'égalité
    resultats.sort(key=lambda x: (-x["score"], x["title"]))

    return resultats[:n]


def analyser_ecarts(
    scores_competences: Dict[str, float],
    competency_texts: Dict[str, str],
    n_faibles: int = TOP_WEAK_COMPETENCIES
) -> List[Dict]:
    """
    Identifie les compétences les moins bien couvertes par le profil utilisateur.
    Utilisé pour construire le plan de progression.

    Paramètres :
      - scores_competences : dict {CompetencyID: score_float}
      - competency_texts   : dict {CompetencyID: texte_de_la_competence}
      - n_faibles          : nombre de compétences faibles à retourner

    Retourne une liste de dicts triée par score croissant :
    [
        {"id": str, "texte": str, "score": float},
        ...
    ]
    """
    competences = [
        {
            "id": cid,
            "texte": competency_texts.get(cid, cid),
            "score": score
        }
        for cid, score in scores_competences.items()
    ]

    # Tri croissant par score
    competences.sort(key=lambda x: x["score"])

    return competences[:n_faibles]


def construire_contexte_rag(
    resume_scores: Dict,
    top_metiers: List[Dict],
    competences_faibles: List[Dict]
) -> str:
    """
    Construit le contexte augmenté (RAG) injecté dans les prompts GenAI.

    Ce contexte contient uniquement des faits issus du pipeline NLP,
    évitant ainsi les hallucinations du LLM.

    Retourne une chaîne de texte formatée.
    """
    lignes = []

    # Score global
    sg = resume_scores.get("score_global", 0.0)
    desc = resume_scores.get("description_globale", "")
    lignes.append(f"Score de couverture global : {sg:.2%} ({desc})")
    lignes.append("")

    # Scores par bloc
    lignes.append("Scores par bloc de compétences :")
    for bloc in resume_scores.get("blocs", []):
        lignes.append(
            f"  - {bloc['nom']} : {bloc['score']:.2%} ({bloc['description']})"
        )
    lignes.append("")

    # Top métiers recommandés
    lignes.append("Métiers les mieux couverts (top 3) :")
    for i, metier in enumerate(top_metiers, 1):
        lignes.append(f"  {i}. {metier['title']} — score : {metier['score']:.2%}")
    lignes.append("")

    # Compétences à renforcer
    lignes.append(f"Compétences prioritaires à développer (les {len(competences_faibles)} plus faibles) :")
    for comp in competences_faibles:
        lignes.append(f"  - {comp['texte']} (score : {comp['score']:.2%})")

    return "\n".join(lignes)
