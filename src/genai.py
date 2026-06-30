"""
genai.py — Intégration Gemini 2.5 Flash (free-tier) avec cache obligatoire.

Trois usages stratégiques et LIMITÉS :
  1. enrichir_saisie()      : enrichissement conditionnel d'une phrase < 5 mots (optionnel)
  2. generer_plan()         : UN SEUL appel pour le plan de progression personnalisé
  3. generer_bio()          : UN SEUL appel pour la bio professionnelle

Toutes les fonctions passent par cache.py :
  - Si le prompt a déjà été soumis, la réponse est retournée sans appel réseau.
  - Si GEMINI_API_KEY est absente ou si google-generativeai est absent,
    les fonctions retournent un fallback déterministe sans planter.

Clé API lue depuis la variable d'environnement GEMINI_API_KEY (jamais en dur).
"""

import logging
import os
from typing import Optional

from src.cache import appel_avec_cache
from src.config import GEMINI_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chargement de la clé API et du client Gemini
# ---------------------------------------------------------------------------

def _charger_cle_api() -> Optional[str]:
    """
    Lit la clé API Gemini depuis les variables d'environnement.
    Tente aussi de charger .env via python-dotenv si disponible.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv absent : on lit directement os.environ

    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def _creer_client():
    """
    Crée et configure le client Gemini.
    Retourne None si la bibliothèque ou la clé est absente.
    """
    cle = _charger_cle_api()
    if not cle:
        logger.warning(
            "GEMINI_API_KEY non définie. Les fonctions GenAI retourneront des fallbacks."
        )
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=cle)
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
    except ImportError:
        logger.warning(
            "google-generativeai non installé. "
            "Exécuter : pip install google-generativeai"
        )
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client Gemini : {e}")
        return None


# ---------------------------------------------------------------------------
# Fonction d'appel bas niveau (utilisée par le cache)
# ---------------------------------------------------------------------------

def _appeler_gemini(prompt: str) -> str:
    """
    Appelle l'API Gemini et retourne le texte de la réponse.
    Retourne une chaîne vide en cas d'erreur.
    """
    client = _creer_client()
    if client is None:
        return ""

    try:
        reponse = client.generate_content(prompt)

        # Accès direct à .text (peut lever ValueError si contenu bloqué)
        texte = reponse.text
        if texte:
            return texte.strip()

        # Fallback : parcourir les candidates si .text est vide
        if hasattr(reponse, "candidates") and reponse.candidates:
            for candidate in reponse.candidates:
                try:
                    parts = candidate.content.parts
                    if parts:
                        return parts[0].text.strip()
                except Exception:
                    continue

        logger.warning("Gemini a retourné une réponse vide ou bloquée.")
        return ""

    except Exception as e:
        logger.error(f"Erreur Gemini : {type(e).__name__} — {e}")
        return ""


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def enrichir_saisie(texte_court: str) -> str:
    """
    Enrichit une saisie trop courte (< 5 mots) pour améliorer
    la qualité des embeddings SBERT. Usage OPTIONNEL et CONDITIONNEL.

    Si pas de clé API ou erreur : retourne le texte original.
    Passe par le cache (même phrase courte = même enrichissement).
    """
    if not texte_court.strip():
        return texte_court

    prompt = (
        f"Tu es un assistant RH. Reformule la compétence suivante en une phrase "
        f"professionnelle complète (15-25 mots), sans ajouter d'informations inventées. "
        f"Compétence courte : \"{texte_court}\"\n"
        f"Phrase reformulée :"
    )

    reponse = appel_avec_cache(prompt, _appeler_gemini)

    if not reponse:
        logger.info("Enrichissement GenAI non disponible, texte original conservé.")
        return texte_court

    return reponse


def generer_plan(contexte_rag: str, prenom: str = "") -> str:
    """
    Génère UN SEUL plan de progression personnalisé basé sur le contexte RAG.

    Le contexte RAG contient les scores, les compétences faibles et les métiers
    cibles — tout est issu du pipeline NLP local, pas du LLM (anti-hallucination).

    Paramètres :
      - contexte_rag : texte structuré construit par recommender.construire_contexte_rag()
      - prenom       : prénom de l'utilisateur (optionnel, pour personnaliser le ton)

    Retourne le plan en markdown, ou un fallback déterministe si pas de clé.
    """
    destinataire = f"pour {prenom}" if prenom else "pour le candidat"

    prompt = f"""Tu es un conseiller en orientation professionnelle data/IA.
À partir du bilan de compétences ci-dessous, génère un plan de progression structuré {destinataire}.

RÈGLES STRICTES :
- Ne jamais inventer de compétences ou de scores non mentionnés dans le contexte.
- Structurer le plan en 3 phases (court terme 0-3 mois, moyen terme 3-6 mois, long terme 6-12 mois).
- Pour chaque phase : 2-3 actions concrètes avec des ressources réalistes (MOOC, projets pratiques, certifications).
- Rédiger en français, ton professionnel et encourageant.
- Format : markdown structuré.

CONTEXTE DU BILAN :
{contexte_rag}

PLAN DE PROGRESSION :"""

    reponse = appel_avec_cache(prompt, _appeler_gemini)

    if not reponse:
        return _fallback_plan(contexte_rag)

    return reponse


def generer_bio(contexte_rag: str, prenom: str = "", metier_cible: str = "") -> str:
    """
    Génère UN SEUL bio professionnelle résumant le profil utilisateur.

    Paramètres :
      - contexte_rag  : texte structuré du pipeline RAG
      - prenom        : prénom (optionnel)
      - metier_cible  : titre du métier recommandé en premier (optionnel)

    Retourne la bio (3-5 phrases), ou un fallback déterministe si pas de clé.
    """
    candidat = prenom if prenom else "le candidat"
    cible = f"pour un poste de {metier_cible}" if metier_cible else "dans le domaine data/IA"

    prompt = f"""Tu es un expert en personal branding RH.
Rédige une bio professionnelle courte (3 à 5 phrases) {cible} pour {candidat}.

RÈGLES STRICTES :
- S'appuyer UNIQUEMENT sur les compétences et scores fournis dans le contexte.
- Ne pas inventer de diplômes, d'expériences ou de projets non mentionnés.
- Ton professionnel, positif et synthétique.
- Mettre en valeur les points forts (blocs à score élevé).
- Format : texte continu, pas de bullet points.

CONTEXTE DU PROFIL :
{contexte_rag}

BIO PROFESSIONNELLE :"""

    reponse = appel_avec_cache(prompt, _appeler_gemini)

    if not reponse:
        return _fallback_bio(contexte_rag, metier_cible)

    return reponse


# ---------------------------------------------------------------------------
# Fallbacks déterministes (mode dégradé sans clé API)
# ---------------------------------------------------------------------------

def _fallback_plan(contexte_rag: str) -> str:
    """
    Plan de progression générique lorsque l'API Gemini n'est pas disponible.
    Extrait les informations brutes du contexte RAG pour rester utile.
    """
    return (
        "## Plan de progression (mode hors-ligne)\n\n"
        "La génération personnalisée par IA n'est pas disponible "
        "(clé GEMINI_API_KEY absente ou quota dépassé).\n\n"
        "**Recommandations génériques basées sur votre bilan :**\n\n"
        "### Phase 1 — Court terme (0-3 mois)\n"
        "- Identifier les compétences à score faible dans votre bilan et les travailler en priorité.\n"
        "- Suivre un MOOC ciblé (Coursera, DataCamp, OpenClassrooms) sur les lacunes identifiées.\n\n"
        "### Phase 2 — Moyen terme (3-6 mois)\n"
        "- Réaliser un projet pratique personnel ou contribuer à un projet open source.\n"
        "- Préparer une certification reconnue dans votre domaine cible.\n\n"
        "### Phase 3 — Long terme (6-12 mois)\n"
        "- Postuler à des stages ou alternances dans les métiers recommandés.\n"
        "- Construire un portfolio démontrant vos compétences renforcées.\n\n"
        f"**Votre bilan détaillé :**\n```\n{contexte_rag}\n```"
    )


def _fallback_bio(contexte_rag: str, metier_cible: str = "") -> str:
    """
    Bio générique lorsque l'API Gemini n'est pas disponible.
    """
    cible = f"dans le domaine {metier_cible}" if metier_cible else "dans le domaine data/IA"

    return (
        f"Professionnel(le) en développement {cible}, je possède un profil technique "
        "solide avec des compétences couvrant plusieurs blocs du référentiel analysé. "
        "Mon parcours me permet d'aborder des projets data avec méthode et rigueur. "
        "Je cherche à renforcer mes compétences prioritaires identifiées dans mon bilan "
        "pour atteindre mes objectifs professionnels.\n\n"
        "*(Bio personnalisée non disponible : configurez GEMINI_API_KEY dans .env)*"
    )
