"""
nlp_engine.py — Moteur NLP sémantique local (coût zéro).

Fonctionnalités :
  - Chargement du modèle SBERT en singleton (une seule instance en mémoire)
  - Encodage des phrases (utilisateur et référentiel)
  - Cache des embeddings du référentiel sur disque (.npy) pour éviter
    de les recalculer à chaque session
  - Calcul de la similarité cosinus via sentence_transformers.util.cos_sim

Aucun appel réseau payant : le modèle tourne entièrement en local.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import (
    SBERT_MODEL_NAME,
    EMBEDDINGS_CACHE_FILE,
    COMPETENCY_IDS_CACHE_FILE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton du modèle SBERT
# ---------------------------------------------------------------------------

_model = None  # Instance unique partagée pour toute la session


def get_model():
    """
    Charge le modèle SBERT une seule fois (singleton).
    Les appels suivants retournent l'instance déjà en mémoire.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Chargement du modèle SBERT : {SBERT_MODEL_NAME}")
            _model = SentenceTransformer(SBERT_MODEL_NAME)
            logger.info("Modèle SBERT chargé avec succès.")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers n'est pas installé. "
                "Exécuter : pip install sentence-transformers"
            ) from e
    return _model


def encoder_phrases(phrases: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode une liste de phrases en vecteurs d'embeddings.

    Paramètres :
      - phrases    : liste de textes à encoder
      - batch_size : taille des lots pour l'encodage (performance)

    Retourne un tableau numpy de forme (n_phrases, dim_embedding).
    """
    if not phrases:
        return np.array([])

    model = get_model()
    embeddings = model.encode(
        phrases,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  # normalisation L2 pour cosinus = produit scalaire
    )
    return embeddings


def encoder_referentiel(
    competency_ids: List[str],
    competency_texts: List[str],
    forcer_recalcul: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode le référentiel de compétences en embeddings, avec cache disque.

    Si le fichier .npy existe et correspond aux mêmes IDs, il est rechargé
    directement sans réencoder (gain de temps significatif).

    Paramètres :
      - competency_ids   : liste des identifiants de compétences (ex. ['C01', 'C02', ...])
      - competency_texts : liste des textes correspondants
      - forcer_recalcul  : si True, ignore le cache et recalcule

    Retourne :
      - embeddings  : tableau numpy (n_competences, dim_embedding)
      - ids_ordonnes: liste des IDs dans le même ordre que les embeddings
    """
    cache_path = Path(EMBEDDINGS_CACHE_FILE)
    ids_path = Path(COMPETENCY_IDS_CACHE_FILE)

    # Tentative de chargement depuis le cache disque
    if not forcer_recalcul and cache_path.exists() and ids_path.exists():
        try:
            ids_en_cache = json.loads(ids_path.read_text(encoding="utf-8"))
            if ids_en_cache == competency_ids:
                embeddings = np.load(str(cache_path))
                logger.info(
                    f"Embeddings du référentiel chargés depuis le cache "
                    f"({len(competency_ids)} compétences)."
                )
                return embeddings, ids_en_cache
        except Exception as e:
            logger.warning(f"Impossible de lire le cache d'embeddings : {e}. Recalcul en cours.")

    # Calcul des embeddings
    logger.info(f"Encodage de {len(competency_texts)} compétences via SBERT...")
    embeddings = encoder_phrases(competency_texts)

    # Sauvegarde du cache sur disque
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embeddings)
        ids_path.write_text(json.dumps(competency_ids), encoding="utf-8")
        logger.info("Cache d'embeddings sauvegardé sur disque.")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le cache d'embeddings : {e}")

    return embeddings, competency_ids


def calculer_similarites(
    embeddings_utilisateur: np.ndarray,
    embeddings_referentiel: np.ndarray
) -> np.ndarray:
    """
    Calcule la matrice de similarité cosinus entre les embeddings utilisateur
    et les embeddings du référentiel.

    Les embeddings doivent être normalisés (normalize_embeddings=True dans encode).
    Dans ce cas, cos_sim = produit scalaire = matmul.

    Paramètres :
      - embeddings_utilisateur  : tableau (n_reponses, dim)
      - embeddings_referentiel  : tableau (n_competences, dim)

    Retourne une matrice numpy de forme (n_reponses, n_competences)
    avec des valeurs dans [-1, 1].
    """
    try:
        from sentence_transformers import util as sbert_util
        # Conversion en tenseurs pour utiliser l'API officielle cos_sim
        import torch
        t_user = torch.tensor(embeddings_utilisateur)
        t_ref = torch.tensor(embeddings_referentiel)
        sim_matrix = sbert_util.cos_sim(t_user, t_ref)
        return sim_matrix.numpy()
    except Exception:
        # Fallback numpy si PyTorch non disponible (ne devrait pas arriver
        # avec sentence-transformers, mais défense en profondeur)
        return np.dot(embeddings_utilisateur, embeddings_referentiel.T)


def scores_par_bloc(
    reponses_texte: List[str],
    blocs: Dict[int, Dict]
) -> Dict[int, float]:
    """
    Calcule le score de similarité pour chaque bloc de compétences.

    Algorithme (tel que spécifié dans le pipeline) :
      Pour chaque bloc B :
        1. Encoder les compétences du bloc.
        2. Pour chaque réponse utilisateur, calculer la similarité max
           avec toutes les compétences du bloc.
        3. Score du bloc = moyenne des max par réponse.

    Paramètres :
      - reponses_texte : liste des textes utilisateur nettoyés
      - blocs          : dict {BlockID: {name, competencies: [{id, text}]}}

    Retourne un dict {BlockID: score_float}.
    """
    if not reponses_texte:
        return {block_id: 0.0 for block_id in blocs}

    # Encodage des réponses utilisateur (une seule fois pour tous les blocs)
    emb_utilisateur = encoder_phrases(reponses_texte)

    scores = {}
    for block_id, bloc_info in blocs.items():
        textes_comp = [c["text"] for c in bloc_info["competencies"]]
        ids_comp = [c["id"] for c in bloc_info["competencies"]]

        if not textes_comp:
            scores[block_id] = 0.0
            continue

        # Encodage des compétences du bloc
        emb_bloc = encoder_phrases(textes_comp)

        # Matrice de similarité (n_reponses x n_competences)
        matrice_sim = calculer_similarites(emb_utilisateur, emb_bloc)

        # Pour chaque compétence : max de similarité avec toutes les réponses
        # (mesure de couverture : est-ce qu'au moins une réponse couvre la compétence ?)
        max_par_competence = matrice_sim.max(axis=0)

        # Score du bloc = moyenne des couvertures par compétence
        score_bloc = float(np.mean(max_par_competence))
        scores[block_id] = max(0.0, score_bloc)  # plancher à 0

    return scores


def scores_competences_detail(
    reponses_texte: List[str],
    competency_ids: List[str],
    competency_texts: List[str]
) -> Dict[str, float]:
    """
    Calcule un score de similarité individuel pour chaque compétence.

    Pour chaque compétence : score = max de similarité cosinus avec
    toutes les réponses utilisateur.

    Utilisé pour identifier les compétences les plus et les moins couvertes.

    Retourne un dict {CompetencyID: score_float}.
    """
    if not reponses_texte or not competency_texts:
        return {cid: 0.0 for cid in competency_ids}

    emb_utilisateur = encoder_phrases(reponses_texte)
    emb_competences, ids_ordonnes = encoder_referentiel(competency_ids, competency_texts)

    # Matrice (n_reponses x n_competences)
    matrice_sim = calculer_similarites(emb_utilisateur, emb_competences)

    # Pour chaque compétence : max sur toutes les réponses
    max_par_competence = matrice_sim.max(axis=0)

    return {
        cid: float(max(0.0, score))
        for cid, score in zip(ids_ordonnes, max_par_competence)
    }
