"""
scoring.py — Calcul du score de couverture pondéré.

Formule centrale :
    Coverage = Σ(Wi · Si) / Σ(Wi)

Où :
  - Si = score du bloc i (similarité cosinus moyenne, cf. nlp_engine.py)
  - Wi = poids du bloc i (configurable dans config.py, défaut = 1)

Expose également l'interprétation qualitative des scores via les seuils
définis dans config.py.
"""

from typing import Dict, Tuple

from src.config import BLOCK_WEIGHTS, SEUIL_BON, SEUIL_MOYEN


def score_global(
    scores_blocs: Dict[int, float],
    poids: Dict[int, float] = None
) -> float:
    """
    Calcule le score de couverture global pondéré.

    Paramètres :
      - scores_blocs : dict {BlockID: score_float} retourné par nlp_engine
      - poids        : dict {BlockID: poids_float} (défaut : BLOCK_WEIGHTS)

    Retourne un float dans [0, 1].
    """
    if poids is None:
        poids = BLOCK_WEIGHTS

    somme_ponderee = 0.0
    somme_poids = 0.0

    for block_id, score in scores_blocs.items():
        w = poids.get(block_id, 1.0)
        somme_ponderee += w * score
        somme_poids += w

    if somme_poids == 0.0:
        return 0.0

    return somme_ponderee / somme_poids


def interpreter_score(score: float) -> Tuple[str, str]:
    """
    Retourne une interprétation qualitative d'un score de couverture.

    Retourne un tuple (niveau, description) :
      - ("bon",      "Bonne couverture")       si score >= SEUIL_BON
      - ("moyen",    "Couverture partielle")   si SEUIL_MOYEN <= score < SEUIL_BON
      - ("faible",   "Couverture insuffisante") si score < SEUIL_MOYEN
    """
    if score >= SEUIL_BON:
        return ("bon", "Bonne couverture")
    elif score >= SEUIL_MOYEN:
        return ("moyen", "Couverture partielle")
    else:
        return ("faible", "Couverture insuffisante")


def resume_scores(
    scores_blocs: Dict[int, float],
    noms_blocs: Dict[int, str],
    poids: Dict[int, float] = None
) -> Dict:
    """
    Construit un résumé complet des scores pour l'affichage.

    Paramètres :
      - scores_blocs : dict {BlockID: score}
      - noms_blocs   : dict {BlockID: nom_du_bloc}
      - poids        : dict {BlockID: poids} (défaut BLOCK_WEIGHTS)

    Retourne un dict :
    {
        "score_global": float,
        "niveau_global": str,
        "description_globale": str,
        "blocs": [
            {
                "id": int,
                "nom": str,
                "score": float,
                "poids": float,
                "niveau": str,
                "description": str
            },
            ...
        ]
    }
    """
    if poids is None:
        poids = BLOCK_WEIGHTS

    sg = score_global(scores_blocs, poids)
    niveau, description = interpreter_score(sg)

    blocs_detail = []
    for block_id in sorted(scores_blocs.keys()):
        score = scores_blocs[block_id]
        niv, desc = interpreter_score(score)
        blocs_detail.append({
            "id": block_id,
            "nom": noms_blocs.get(block_id, f"Bloc {block_id}"),
            "score": score,
            "poids": poids.get(block_id, 1.0),
            "niveau": niv,
            "description": desc
        })

    return {
        "score_global": sg,
        "niveau_global": niveau,
        "description_globale": description,
        "blocs": blocs_detail
    }
