# Données de Référence - Projet AISCA

## Origine et Sources

Ce référentiel de compétences et de métiers a été construit en s'inspirant de deux standards de référence majeurs :

1. **ROME (Répertoire Opérationnel des Métiers et des Emplois)** - version 4.0
   - Source: [data.gouv.fr - ROME](https://www.data.gouv.fr/datasets/repertoire-operationnel-des-metiers-et-des-emplois-rome)
   - Gestionnaire: France Travail
   - Actualisé semestriellement avec ~1900 fiches métiers descriptives

2. **European e-Competence Framework (e-CF) 3.0**
   - Source: [Official e-CF Documentation](https://www.ecompetences.eu/)
   - Standard: EN 16234 (2016), maintenu par CEN TC 428
   - Définit 40 compétences ICT organisées selon 5 domaines: PLAN, BUILD, RUN, ENABLE, MANAGE

Le référentiel AISCA **enrichit ces standards** avec des compétences spécialisées en **Data Science, Machine Learning, NLP et Data Engineering**, domaines en émergence rapide.

## Structure des Données

### 1. competences.csv
**Colonnes:** `CompetencyID, Competency, BlockID, BlockName`

Contient **36 compétences** organisées en **6 blocs thématiques** :

| BlockID | BlockName | Nb Compétences | Exemples |
|---------|-----------|----------------|----------|
| 1 | Data Analysis | 6 | Analyse descriptive, SQL, visualisations |
| 2 | Machine Learning | 6 | Modèles supervisés, deep learning, feature engineering |
| 3 | NLP | 6 | Tokenisation, transformers, sentiment analysis |
| 4 | Data Engineering | 6 | Pipelines ETL, big data, qualité données |
| 5 | Software Development & Programming | 6 | Python, Git, Docker, APIs REST |
| 6 | Project Management & Soft Skills | 6 | Communication, collaboration, RGPD |

**Format CompetencyID:** C01 à C36 (identifiants uniques alphanumériques)

**Caractéristiques:**
- Phrases courtes et explicites (adaptées aux embeddings SBERT)
- Mélange français courant + anglais technique où pertinent
- Pas de doublons ; termes normalisés

### 2. metiers.csv
**Colonnes:** `JobID, JobTitle, RequiredCompetencies`

Contient **8 métiers** clés du domaine Data/IA :

| JobID | JobTitle | Nb Compétences |
|-------|----------|----------------|
| J01 | Data Analyst | 9 |
| J02 | Machine Learning Engineer | 10 |
| J03 | Data Scientist | 13 |
| J04 | NLP Engineer | 13 |
| J05 | Data Engineer | 14 |
| J06 | Business Intelligence Analyst | 9 |
| J07 | MLOps Engineer | 11 |
| J08 | Research Scientist in AI | 12 |

**Format RequiredCompetencies:** Listes d'IDs séparées par `;` (ex. `C01;C02;C03`)

**Propriétés:**
- Chaque CompetencyID référencé existe dans `competences.csv`
- Les combinaisons reflètent les chemins de carrière réels du secteur data/IA
- Données cohérentes et sans référence orpheline

## Utilisation

### Intégration AISCA
Ces fichiers alimentent le pipeline RAG du mini-agent AISCA :
1. **Chargement:** Les données CSV sont parsées et structurées
2. **Embeddings:** Chaque compétence est convertie via SBERT pour la similarité sémantique
3. **Requête utilisateur:** Un utilisateur demande une analyse de compétences
4. **Récupération:** RAG retrouve les compétences pertinentes et les métiers associés
5. **Curation:** L'agent synthétise les résultats en insights actionnables

### Format de fichier
- **Encodage:** UTF-8 (avec BOM optionnel)
- **Délimiteur CSV:** Virgule (`,`)
- **Guillemets:** Pas de guillemets autour des valeurs standard

## Extensibilité

La structure est conçue pour croître :
- **Ajouter une compétence:** Incrémenter CompetencyID, assigner un BlockID existant ou créer un nouveau bloc
- **Ajouter un métier:** Incrémenter JobID, composer la liste de CompetencyID
- **Ajouter un bloc:** Augmenter BlockID, regrouper les compétences associées

## Cohérence et Qualité

- Dédoublonnage : Chaque compétence est unique et non répétée entre blocs
- Validation des références : Tous les CompetencyID cités dans `metiers.csv` existent dans `competences.csv`
- Phrases canoniques : Optimisées pour la recherche sémantique et la récupération contextuelle

---

**Version:** 1.0 | **Date:** Juin 2026 | **Licence:** Domaine public (inspiré de données ouvertes)
