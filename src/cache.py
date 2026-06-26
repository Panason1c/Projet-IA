"""
cache.py — Cache local des réponses GenAI.

Principe :
  - Chaque appel GenAI est identifié par le hash SHA-256 de son prompt.
  - Si le hash est déjà présent dans cache/genai_cache.json, la réponse
    est retournée directement sans appel réseau.
  - Sinon, la réponse est stockée après l'appel API.

Cela garantit :
  - Respect du free-tier (aucun appel inutile)
  - Reproductibilité des résultats pour les mêmes entrées
  - Résilience aux coupures réseau après le premier appel
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from src.config import GENAI_CACHE_FILE

logger = logging.getLogger(__name__)


def _hash_prompt(prompt: str) -> str:
    """
    Calcule un identifiant unique pour un prompt donné.
    Utilise SHA-256 pour minimiser les collisions.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _charger_cache() -> dict:
    """
    Charge le cache depuis le fichier JSON.
    Retourne un dict vide si le fichier n'existe pas ou est corrompu.
    """
    cache_path = Path(GENAI_CACHE_FILE)
    if not cache_path.exists():
        return {}

    try:
        contenu = cache_path.read_text(encoding="utf-8")
        return json.loads(contenu)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cache GenAI illisible ({e}), démarrage avec un cache vide.")
        return {}


def _sauvegarder_cache(cache: dict) -> None:
    """
    Sauvegarde le cache dans le fichier JSON.
    Crée le dossier parent si nécessaire.
    """
    cache_path = Path(GENAI_CACHE_FILE)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except OSError as e:
        logger.error(f"Impossible de sauvegarder le cache GenAI : {e}")


def lire_cache(prompt: str) -> Optional[str]:
    """
    Cherche une réponse en cache pour le prompt donné.

    Retourne la réponse mise en cache, ou None si absente.
    """
    cle = _hash_prompt(prompt)
    cache = _charger_cache()
    reponse = cache.get(cle)
    if reponse is not None:
        logger.info(f"Cache GenAI HIT (clé={cle[:12]}...)")
    return reponse


def ecrire_cache(prompt: str, reponse: str) -> None:
    """
    Enregistre une paire (prompt, réponse) dans le cache.
    """
    cle = _hash_prompt(prompt)
    cache = _charger_cache()
    cache[cle] = reponse
    _sauvegarder_cache(cache)
    logger.info(f"Cache GenAI écrit (clé={cle[:12]}...)")


def appel_avec_cache(prompt: str, fn_appel_api) -> str:
    """
    Wrapper générique : cherche dans le cache, sinon appelle fn_appel_api(prompt).

    Paramètres :
      - prompt       : texte du prompt envoyé au LLM
      - fn_appel_api : callable(prompt: str) -> str — fonction d'appel réel à l'API

    Retourne la réponse (depuis le cache ou depuis l'API).
    """
    # 1. Vérification du cache
    reponse_cachee = lire_cache(prompt)
    if reponse_cachee is not None:
        return reponse_cachee

    # 2. Appel API (un seul)
    logger.info("Cache GenAI MISS — appel à l'API.")
    reponse = fn_appel_api(prompt)

    # 3. Mise en cache pour les appels futurs
    if reponse:
        ecrire_cache(prompt, reponse)

    return reponse


def vider_cache() -> int:
    """
    Vide le cache GenAI et retourne le nombre d'entrées supprimées.
    Utile pour les tests ou la maintenance.
    """
    cache = _charger_cache()
    nb_entrees = len(cache)
    _sauvegarder_cache({})
    logger.info(f"Cache GenAI vidé ({nb_entrees} entrées supprimées).")
    return nb_entrees


def taille_cache() -> int:
    """Retourne le nombre d'entrées actuellement dans le cache."""
    return len(_charger_cache())
