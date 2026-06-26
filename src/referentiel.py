"""
referentiel.py — Chargement et validation des référentiels de compétences et de métiers.

Expose :
  - charger_competences() -> DataFrame
  - charger_metiers()     -> DataFrame
  - get_blocs()           -> dict[int, dict]  {BlockID: {name, competencies: [...]}}
  - get_metiers()         -> dict[str, dict]  {JobID: {title, competency_ids: [...]}}
  - valider_integrite()   -> list[str]  (liste des anomalies, vide si OK)
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List

from src.config import COMPETENCES_CSV, METIERS_CSV


def charger_competences(chemin: Path = COMPETENCES_CSV) -> pd.DataFrame:
    """
    Charge le CSV des compétences et valide les colonnes attendues.

    Retourne un DataFrame avec colonnes :
      CompetencyID (str), Competency (str), BlockID (int), BlockName (str)
    """
    df = pd.read_csv(chemin, dtype=str)

    # Vérification des colonnes obligatoires
    colonnes_attendues = {"CompetencyID", "Competency", "BlockID", "BlockName"}
    colonnes_manquantes = colonnes_attendues - set(df.columns)
    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes manquantes dans {chemin} : {colonnes_manquantes}"
        )

    # Nettoyage des espaces superflus (utilise .map() compatible pandas >= 2.0)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Conversion du BlockID en entier pour les calculs de scoring
    df["BlockID"] = df["BlockID"].astype(int)

    return df


def charger_metiers(chemin: Path = METIERS_CSV) -> pd.DataFrame:
    """
    Charge le CSV des métiers et valide les colonnes attendues.

    Retourne un DataFrame avec colonnes :
      JobID (str), JobTitle (str), RequiredCompetencies (str, séparateur ';')
    """
    df = pd.read_csv(chemin, dtype=str)

    colonnes_attendues = {"JobID", "JobTitle", "RequiredCompetencies"}
    colonnes_manquantes = colonnes_attendues - set(df.columns)
    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes manquantes dans {chemin} : {colonnes_manquantes}"
        )

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def get_blocs(df_competences: pd.DataFrame = None) -> Dict[int, Dict]:
    """
    Construit un dictionnaire de blocs à partir du DataFrame de compétences.

    Structure retournée :
    {
        1: {
            "name": "Data Analysis",
            "competencies": [
                {"id": "C01", "text": "Analyse descriptive et exploration de données"},
                ...
            ]
        },
        ...
    }
    """
    if df_competences is None:
        df_competences = charger_competences()

    blocs = {}
    for _, row in df_competences.iterrows():
        block_id = int(row["BlockID"])
        if block_id not in blocs:
            blocs[block_id] = {
                "name": row["BlockName"],
                "competencies": []
            }
        blocs[block_id]["competencies"].append({
            "id": row["CompetencyID"],
            "text": row["Competency"]
        })

    return blocs


def get_metiers(
    df_metiers: pd.DataFrame = None,
    df_competences: pd.DataFrame = None
) -> Dict[str, Dict]:
    """
    Construit un dictionnaire de métiers avec leurs compétences requises.

    Structure retournée :
    {
        "J01": {
            "title": "Data Analyst",
            "competency_ids": ["C01", "C02", ...],
            "competency_texts": ["Analyse descriptive...", ...]
        },
        ...
    }
    """
    if df_metiers is None:
        df_metiers = charger_metiers()
    if df_competences is None:
        df_competences = charger_competences()

    # Index rapide CompetencyID -> texte
    comp_index = df_competences.set_index("CompetencyID")["Competency"].to_dict()

    metiers = {}
    for _, row in df_metiers.iterrows():
        # Parsing tolérant aux espaces autour du séparateur ';'
        ids = [cid.strip() for cid in row["RequiredCompetencies"].split(";") if cid.strip()]
        textes = [comp_index.get(cid, f"[Compétence inconnue : {cid}]") for cid in ids]

        metiers[row["JobID"]] = {
            "title": row["JobTitle"],
            "competency_ids": ids,
            "competency_texts": textes
        }

    return metiers


def valider_integrite(
    df_competences: pd.DataFrame = None,
    df_metiers: pd.DataFrame = None
) -> List[str]:
    """
    Vérifie l'intégrité référentielle des deux CSV.

    Retourne une liste d'anomalies (chaînes de texte).
    Une liste vide signifie que tout est cohérent.
    """
    if df_competences is None:
        df_competences = charger_competences()
    if df_metiers is None:
        df_metiers = charger_metiers()

    anomalies = []

    # 1. Unicité des CompetencyID
    doublons_comp = df_competences[df_competences["CompetencyID"].duplicated()]["CompetencyID"].tolist()
    if doublons_comp:
        anomalies.append(f"CompetencyID en doublon : {doublons_comp}")

    # 2. Unicité des JobID
    doublons_job = df_metiers[df_metiers["JobID"].duplicated()]["JobID"].tolist()
    if doublons_job:
        anomalies.append(f"JobID en doublon : {doublons_job}")

    # 3. Tous les IDs dans RequiredCompetencies existent dans competences.csv
    ids_valides = set(df_competences["CompetencyID"].tolist())
    for _, row in df_metiers.iterrows():
        ids_requis = [cid.strip() for cid in row["RequiredCompetencies"].split(";") if cid.strip()]
        ids_invalides = [cid for cid in ids_requis if cid not in ids_valides]
        if ids_invalides:
            anomalies.append(
                f"Métier '{row['JobID']}' référence des compétences inexistantes : {ids_invalides}"
            )

    return anomalies
