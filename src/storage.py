"""
storage.py — Persistance des réponses utilisateur.

Format configurable dans config.py (STORAGE_FORMAT = "json" ou "csv").
Les réponses sont stockées dans data/responses/ avec un nom de fichier
horodaté pour éviter les collisions.

Structure d'une session sauvegardée :
{
    "session_id": str (timestamp),
    "prenom": str,
    "timestamp": str (ISO 8601),
    "reponses_likert": {question_id: valeur_int},
    "reponses_ouvertes": {question_id: texte},
    "reponses_choix": {question_id: [options_selectionnees]},
    "scores_blocs": {block_id: score_float},
    "score_global": float,
    "top_metiers": [{job_id, title, score}]
}
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import RESPONSES_DIR, STORAGE_FORMAT

logger = logging.getLogger(__name__)


def _generer_session_id() -> str:
    """Génère un identifiant unique de session basé sur l'horodatage."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sauvegarder_session(
    prenom: str,
    reponses_likert: Dict[str, Any],
    reponses_ouvertes: Dict[str, str],
    reponses_choix: Dict[str, List[str]],
    scores_blocs: Optional[Dict] = None,
    score_global: Optional[float] = None,
    top_metiers: Optional[List[Dict]] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Sauvegarde les réponses et résultats d'une session utilisateur.

    Retourne le session_id utilisé.
    """
    if session_id is None:
        session_id = _generer_session_id()

    donnees = {
        "session_id": session_id,
        "prenom": prenom,
        "timestamp": datetime.now().isoformat(),
        "reponses_likert": {str(k): v for k, v in reponses_likert.items()},
        "reponses_ouvertes": {str(k): v for k, v in reponses_ouvertes.items()},
        "reponses_choix": {str(k): v for k, v in reponses_choix.items()},
        "scores_blocs": {str(k): v for k, v in (scores_blocs or {}).items()},
        "score_global": score_global,
        "top_metiers": top_metiers or [],
    }

    # Création du dossier si nécessaire
    repertoire = Path(RESPONSES_DIR)
    repertoire.mkdir(parents=True, exist_ok=True)

    if STORAGE_FORMAT == "json":
        chemin = repertoire / f"session_{session_id}.json"
        try:
            chemin.write_text(
                json.dumps(donnees, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info(f"Session sauvegardée : {chemin}")
        except OSError as e:
            logger.error(f"Impossible de sauvegarder la session : {e}")

    elif STORAGE_FORMAT == "csv":
        chemin = repertoire / f"session_{session_id}.csv"
        try:
            # Aplatissement pour CSV : une ligne par question
            lignes = []
            for q_id, valeur in reponses_likert.items():
                lignes.append({"session_id": session_id, "type": "likert",
                                "question_id": q_id, "valeur": str(valeur)})
            for q_id, texte in reponses_ouvertes.items():
                lignes.append({"session_id": session_id, "type": "ouverte",
                                "question_id": q_id, "valeur": texte})
            for q_id, options in reponses_choix.items():
                lignes.append({"session_id": session_id, "type": "choix",
                                "question_id": q_id, "valeur": ";".join(options)})

            df = pd.DataFrame(lignes)
            df.to_csv(chemin, index=False, encoding="utf-8")
            logger.info(f"Session sauvegardée : {chemin}")
        except OSError as e:
            logger.error(f"Impossible de sauvegarder la session CSV : {e}")

    return session_id


def charger_session(session_id: str) -> Optional[Dict]:
    """
    Charge une session précédemment sauvegardée par son identifiant.

    Retourne None si la session n'existe pas.
    """
    repertoire = Path(RESPONSES_DIR)

    if STORAGE_FORMAT == "json":
        chemin = repertoire / f"session_{session_id}.json"
        if not chemin.exists():
            return None
        try:
            return json.loads(chemin.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Impossible de charger la session {session_id} : {e}")
            return None

    return None


def lister_sessions() -> List[str]:
    """
    Liste tous les identifiants de sessions sauvegardées.
    """
    repertoire = Path(RESPONSES_DIR)
    if not repertoire.exists():
        return []

    sessions = []
    pattern = "session_*.json" if STORAGE_FORMAT == "json" else "session_*.csv"
    for fichier in sorted(repertoire.glob(pattern)):
        # Extraction de l'ID depuis le nom de fichier
        session_id = fichier.stem.replace("session_", "")
        sessions.append(session_id)

    return sessions
