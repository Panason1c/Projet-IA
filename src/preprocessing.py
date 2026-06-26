"""
preprocessing.py — Nettoyage et préparation du texte utilisateur.

Principes :
  - Nettoyage LÉGER : SBERT fonctionne mieux avec le contexte naturel préservé.
  - Pas de stemming, pas de suppression des stopwords (le modèle les gère).
  - Détection des phrases trop courtes (< MIN_WORDS_THRESHOLD mots) pour
    déclencher l'enrichissement GenAI conditionnel (optionnel).
"""

import re
from typing import List, Tuple

from src.config import MIN_WORDS_THRESHOLD


def nettoyer_texte(texte: str) -> str:
    """
    Nettoie légèrement un texte utilisateur sans en altérer le sens.

    Opérations :
      - Suppression des espaces en début/fin
      - Normalisation des espaces multiples
      - Suppression des caractères de contrôle (sauf ponctuation courante)

    Le texte reste en minuscules naturelles (SBERT gère la casse).
    """
    if not isinstance(texte, str):
        return ""

    # Suppression des espaces extrêmes
    texte = texte.strip()

    # Normalisation des espaces multiples / tabulations / sauts de ligne
    texte = re.sub(r"\s+", " ", texte)

    # Suppression des caractères de contrôle (sauf \n, \t déjà normalisés)
    texte = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texte)

    return texte


def compter_mots(texte: str) -> int:
    """
    Compte le nombre de mots dans un texte nettoyé.
    Un mot est une séquence de caractères non-espaces.
    """
    texte = texte.strip()
    if not texte:
        return 0
    return len(texte.split())


def est_trop_court(texte: str, seuil: int = MIN_WORDS_THRESHOLD) -> bool:
    """
    Retourne True si le texte a strictement moins de `seuil` mots.
    Utilisé pour déclencher l'enrichissement GenAI conditionnel.
    """
    return compter_mots(texte) < seuil


def preparer_reponses_ouvertes(
    reponses: List[str],
    enrichir_fn=None
) -> Tuple[List[str], List[bool]]:
    """
    Prépare une liste de réponses texte libres pour l'analyse NLP.

    Paramètres :
      - reponses   : liste de textes bruts saisis par l'utilisateur
      - enrichir_fn: fonction optionnelle (str -> str) pour enrichir les phrases
                     trop courtes via GenAI. Si None, les phrases courtes sont
                     conservées telles quelles (pas d'appel API).

    Retourne :
      - reponses_nettoyees : liste des textes prêts pour SBERT
      - enrichies          : liste de booléens indiquant si chaque réponse
                             a été enrichie par GenAI
    """
    reponses_nettoyees = []
    enrichies = []

    for rep in reponses:
        texte = nettoyer_texte(rep)

        # Ignorer les réponses vides
        if not texte:
            continue

        # Détection et enrichissement conditionnel
        if est_trop_court(texte) and enrichir_fn is not None:
            try:
                texte_enrichi = enrichir_fn(texte)
                texte = nettoyer_texte(texte_enrichi) if texte_enrichi else texte
                enrichies.append(True)
            except Exception:
                # En cas d'échec de l'enrichissement, on conserve le texte original
                enrichies.append(False)
        else:
            enrichies.append(False)

        reponses_nettoyees.append(texte)

    return reponses_nettoyees, enrichies


def fusionner_reponses(
    reponses_ouvertes: List[str],
    reponses_likert: dict,
    reponses_choix: dict
) -> List[str]:
    """
    Fusionne toutes les réponses du questionnaire en une liste de textes
    exploitables par SBERT.

    Paramètres :
      - reponses_ouvertes : liste de textes libres saisis par l'utilisateur
      - reponses_likert   : dict {question_id: valeur_int (1-5)}
                            Les notes Likert sont converties en phrases courtes.
      - reponses_choix    : dict {question_id: [compétence_id_1, ...]}
                            Les cases cochées deviennent des textes de compétences.

    Retourne une liste unifiée de textes pour l'encodage SBERT.
    """
    textes = []

    # Textes libres (après nettoyage)
    for rep in reponses_ouvertes:
        texte = nettoyer_texte(rep)
        if texte:
            textes.append(texte)

    # Conversion Likert -> phrase (le score numérique est contextualisé)
    LIKERT_PHRASES = {
        1: "je ne maîtrise pas du tout cette compétence",
        2: "j'ai une très faible connaissance de cette compétence",
        3: "j'ai quelques notions de base dans ce domaine",
        4: "je maîtrise plutôt bien cette compétence",
        5: "je suis très compétent et expérimenté dans ce domaine",
    }
    for question_id, valeur in reponses_likert.items():
        phrase = LIKERT_PHRASES.get(int(valeur), "")
        if phrase:
            textes.append(phrase)

    # Choix multiples -> textes de compétences sélectionnées
    for _, selections in reponses_choix.items():
        for texte_comp in selections:
            texte = nettoyer_texte(texte_comp)
            if texte:
                textes.append(texte)

    return textes
