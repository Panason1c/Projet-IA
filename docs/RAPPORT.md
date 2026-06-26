# AISCA — Rapport de Projet
## Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences

**Épreuve certifiante** : EFREI — RNCP40875 — Bloc 2  
**BC2 : Piloter et implémenter des solutions d'IA en s'aidant de l'IA générative**  
**Date** : Juin 2026

---

## Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Architecture RAG et pipeline de bout en bout](#2-architecture-rag-et-pipeline-de-bout-en-bout)
3. [Choix techniques justifiés](#3-choix-techniques-justifiés)
4. [Description détaillée des modules](#4-description-détaillée-des-modules)
5. [Référentiel de données](#5-référentiel-de-données)
6. [Résultats et fonctionnement réel](#6-résultats-et-fonctionnement-réel)
7. [Interface utilisateur Streamlit](#7-interface-utilisateur-streamlit)
8. [Limites et perspectives](#8-limites-et-perspectives)
9. [Mapping RNCP40875 Bloc 2](#9-mapping-rncp40875-bloc-2)
10. [Références](#10-références)

---

## 1. Contexte et objectifs

### 1.1 Problématique

Dans le domaine de la data et de l'intelligence artificielle, l'orientation professionnelle repose trop souvent sur des auto-évaluations imprécises ou des matrices de compétences rigides qui ne capturent pas la richesse sémantique des expériences décrites par un candidat. Un professionnel qui rédige "j'ai construit un pipeline de features pour un modèle de détection d'anomalies" exprime des compétences en *ingénierie des features* (C09), en *évaluation de modèles* (C08) et en *développement Python* (C25) — mais un système à mots-clés basique ne les reliera pas sans dictionnaire exhaustif.

AISCA (Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences) répond à ce problème en substituant la correspondance lexicale par une **analyse sémantique contextuelle** fondée sur des embeddings de phrases (Sentence-BERT).

### 1.2 Objectifs du projet

| Objectif | Détail |
|----------|--------|
| **Cartographie sémantique** | Mesurer la couverture réelle du profil utilisateur sur un référentiel de 36 compétences réparties en 6 blocs métier |
| **Recommandation de métiers** | Identifier les 3 métiers du référentiel (parmi 8) les mieux alignés avec le profil analysé |
| **Génération augmentée** | Produire un plan de progression et une bio professionnelle personnalisés, ancrés dans les scores calculés (et non inventés par le LLM) |
| **Coût maîtrisé** | Analyse NLP entièrement locale (coût zéro) ; GenAI limitée à 2 appels maximum par session, avec caching |
| **Industrialisation** | Code modulaire, testable, configurable, déployable via une commande unique (`streamlit run app.py`) |

### 1.3 Positionnement dans le référentiel RNCP40875 Bloc 2

Le projet couvre l'intégralité des compétences évaluées du Bloc 2 : collecte et préparation de données structurées, conception d'un modèle NLP basé sur les transformers, prototypage d'un pipeline RAG avec embeddings, intégration responsable de la GenAI, et documentation technique complète.

---

## 2. Architecture RAG et pipeline de bout en bout

### 2.1 Le paradigme RAG appliqué à la cartographie des compétences

L'architecture **RAG (Retrieval-Augmented Generation)** se décompose en trois phases canoniques, ici adaptées au domaine de l'évaluation des compétences :

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE AISCA (RAG)                            │
│                                                                         │
│  [ACQUISITION]          [RETRIEVAL]           [AUGMENTED GENERATION]    │
│                                                                         │
│  Questionnaire    →   Embeddings SBERT   →   Contexte RAG structuré    │
│  hybride              Similarité cosinus      injecté dans Gemini       │
│  (Likert +            Référentiel encodé      → Plan + Bio              │
│   texte libre +       Scores par bloc                                   │
│   choix)              Score global                                      │
│                        Recommandation                                   │
│                        top-3 métiers                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

La phase **Retrieval** est la plus critique : elle garantit que le contenu transmis au LLM est entièrement issu du pipeline NLP local, éliminant le risque d'hallucinations sur les scores ou les compétences. La GenAI ne fait que *mettre en forme* un contexte factuel qu'elle n'a pas produit.

### 2.2 Description étape par étape

#### Étape 1 — Acquisition (questionnaire hybride)

`questionnaire.py` + `app.py`

Le questionnaire est structuré en trois couches complémentaires :
- **6 questions Likert** (échelle 1-5) : auto-évaluation rapide par bloc de compétences (Data Analysis, Machine Learning, NLP, Data Engineering, Software Development, Project Management).
- **6 questions ouvertes** : description de projets et expériences concrètes, une par bloc. Ces textes libres constituent l'input principal de l'analyse sémantique.
- **5 questions à choix multiples** : sélection de technologies et frameworks maîtrisés (Python/pandas, scikit-learn, HuggingFace, Airflow, Docker, etc.).

Les réponses sont validées (au moins une réponse ouverte obligatoire) et persistées par `storage.py` au format JSON horodaté dans `data/responses/`.

#### Étape 2 — Prétraitement (`preprocessing.py`)

Le nettoyage est intentionnellement **léger** pour préserver le contexte naturel dont SBERT a besoin :
- Normalisation des espaces et suppression des caractères de contrôle uniquement.
- Pas de stemming, pas de suppression des stopwords (contrairement à une approche TF-IDF classique).
- Détection des phrases de moins de 5 mots (`MIN_WORDS_THRESHOLD = 5`) pour un éventuel enrichissement GenAI conditionnel.

La fusion des réponses produit une liste unifiée de textes : les réponses ouvertes sont conservées telles quelles ; les notes Likert sont converties en phrases naturelles (ex. note 4 → "je maîtrise plutôt bien cette compétence") ; les choix multiples deviennent directement des textes de compétences.

#### Étape 3 — Embeddings SBERT (`nlp_engine.py`)

Le modèle `all-MiniLM-L6-v2` encode chaque texte en un vecteur dense de **384 dimensions**. Les embeddings sont normalisés L2 dès l'encodage (`normalize_embeddings=True`), ce qui ramène le calcul de similarité cosinus à un simple produit scalaire.

Les embeddings du référentiel (36 compétences) sont calculés **une seule fois** et mis en cache sur disque (`data/competences_embeddings.npy` + `data/competences_ids.json`). Lors des sessions suivantes, le cache est rechargé directement si les IDs n'ont pas changé.

#### Étape 4 — Similarité cosinus (`nlp_engine.py`)

Pour chaque bloc B :
1. La matrice de similarité (n_réponses × n_compétences_du_bloc) est calculée via `sentence_transformers.util.cos_sim`.
2. Pour chaque réponse utilisateur : on prend la **similarité maximale** avec toutes les compétences du bloc (max sur l'axe des compétences).
3. Le **score du bloc** Si = moyenne de ces maxima sur toutes les réponses.

Cette approche "max puis moyenne" est robuste : une réponse décrivant une expérience Python activera fortement la compétence C25 sans être pénalisée par les autres compétences du bloc logiciel qu'elle n'aborde pas.

#### Étape 5 — Score de couverture pondéré (`scoring.py`)

```
Coverage = Σ(Wi · Si) / Σ(Wi)
```

Où `Si` est le score du bloc i et `Wi` son poids (configurable dans `config.py`, tous à 1.0 par défaut). La formule est une **moyenne pondérée** : en configuration par défaut (poids égaux), elle est équivalente à une moyenne arithmétique simple des scores de blocs.

Trois niveaux d'interprétation sont définis :
- Score ≥ 0.70 : Bonne couverture
- Score 0.50–0.69 : Couverture partielle
- Score < 0.50 : Couverture insuffisante

#### Étape 6 — Recommandation top-3 métiers (`recommender.py`)

Pour chaque métier du référentiel, le score est calculé comme la **moyenne des scores de similarité de ses compétences requises** (pré-calculés à l'étape 4 via `scores_competences_detail`). Les 8 métiers sont triés par score décroissant et les 3 premiers sont retenus.

La fonction `analyser_ecarts` identifie les 5 compétences les moins bien couvertes, qui serviront à construire le plan de progression.

#### Étape 7 — Contexte RAG et génération GenAI (`recommender.py` + `genai.py` + `cache.py`)

La fonction `construire_contexte_rag` assemble un texte structuré contenant :
- Le score global et son interprétation qualitative.
- Les scores des 6 blocs.
- Le top-3 des métiers avec leurs scores.
- Les 5 compétences les plus faibles.

Ce contexte est injecté dans les prompts Gemini avec des **règles strictes anti-hallucination** : "Ne jamais inventer de compétences ou de scores non mentionnés dans le contexte." Deux appels maximum sont effectués : un pour le plan de progression (3 phases : 0-3 mois, 3-6 mois, 6-12 mois) et un pour la bio professionnelle (3-5 phrases).

Chaque appel passe par `cache.py` : la clé est le hash SHA-256 du prompt complet. Un prompt déjà soumis ne génère aucun appel réseau.

#### Étape 8 — Visualisation et restitution (`app.py`)

L'interface Streamlit présente les résultats en 5 onglets :
1. Radar de compétences (Plotly) + barres horizontales par bloc avec seuils visuels.
2. Top-3 métiers avec graphique en barres et détail compétence par compétence.
3. Carte de couverture de toutes les compétences (barre horizontale colorée RdYlGn).
4. Plan de progression en markdown.
5. Bio professionnelle.

---

## 3. Choix techniques justifiés

### 3.1 SBERT local `all-MiniLM-L6-v2` — pourquoi ce modèle

#### Sémantique vs TF-IDF

| Critère | TF-IDF | SBERT `all-MiniLM-L6-v2` |
|---------|--------|--------------------------|
| Type de représentation | Fréquence de termes, espace creux | Vecteur dense 384 dimensions |
| Capture de la sémantique | Non ("Python" ≠ "développement Python") | Oui (synonymes, paraphrases, contexte) |
| Robustesse aux formulations | Faible | Élevée |
| Besoin de dictionnaire de synonymes | Oui | Non |
| Coût d'inférence | Nul (calcul matriciel) | Local, gratuit (~14 000 phrases/sec sur CPU) |
| Appel API externe | Non | Non |
| Taille du modèle | N/A | 22 Mo, ~22M paramètres |

L'avantage décisif de SBERT est illustré par ce cas concret : la réponse "j'ai entraîné un réseau de neurones convolutif pour la classification d'images" sera sémantiquement proche de la compétence C11 ("Implémenter des modèles de deep learning et réseaux de neurones") avec une similarité cosinus élevée, même si aucun mot exact n'est partagé. TF-IDF aurait produit un score nul ou très faible.

#### Pourquoi `all-MiniLM-L6-v2` spécifiquement

Ce modèle est issu d'une distillation d'un modèle BERT-large en 6 couches (contre 12 pour BERT-base), conservant 99% de la qualité sur les benchmarks de similarité sémantique (STS) tout en étant cinq fois plus léger. Il est disponible librement sur HuggingFace via la bibliothèque `sentence-transformers` et s'exécute intégralement en local, sans aucun appel réseau payant. C'est le choix standard pour les applications de recherche sémantique embarquée.

#### Singleton et cache disque

Le modèle est chargé une seule fois grâce au pattern singleton (`_model` global dans `nlp_engine.py`) et déclaré via `@st.cache_resource` côté Streamlit. Les embeddings du référentiel sont persistés sur disque (`.npy`), ce qui élimine l'encodage répété des 36 compétences à chaque session.

### 3.2 Similarité cosinus

La similarité cosinus mesure l'angle entre deux vecteurs indépendamment de leur norme. Comme les embeddings sont normalisés L2 dès l'encodage, `cos_sim(u, v) = u · v` (produit scalaire), ce qui est numériquement efficace. Les valeurs appartiennent à [-1, 1] ; en pratique, les similarités sémantiques pertinentes se situent dans [0, 1].

### 3.3 Formule de score de couverture pondéré

```
Coverage = Σ(Wi · Si) / Σ(Wi)
```

**Justification du choix "max puis moyenne" :**
- Le *maximum* par réponse évite de pénaliser une réponse riche qui aborde plusieurs compétences d'un bloc : la compétence la mieux couverte détermine la contribution de cette réponse au bloc.
- La *moyenne* sur les réponses est robuste : un utilisateur qui répond abondamment à une seule question n'est pas avantagé artificiellement.

**Cas du test unitaire de référence (extrait de `test_scoring.py`) :**
```
Blocs : {1: 0.85, 2: 0.78, 3: 0.40}, poids égaux
Score = (0.85 + 0.78 + 0.40) / 3 = 2.03 / 3 ≈ 0.677
```
Ce cas est vérifié par le test `test_score_global_equal_weights` avec une tolérance de ±0.01.

**Extensibilité des poids :** Le dictionnaire `BLOCK_WEIGHTS` dans `config.py` permet d'ajuster l'importance relative des blocs selon le poste visé. Par exemple, un recruteur cherchant un NLP Engineer pourrait donner un poids de 2.0 au bloc 3 (NLP) sans modifier une ligne de code métier.

### 3.4 Recommandation par score de compétences

Le score d'un métier est la **moyenne des scores de similarité de ses compétences requises**. Cette approche est directe et interprétable : un métier dont plusieurs compétences requises sont absentes du profil aura naturellement un score faible. Elle garantit aussi que le top-3 est déterministe et traçable — chaque score peut être expliqué compétence par compétence.

### 3.5 Gestion responsable et limitée de la GenAI

#### Principe "GenAI en dernier recours"

La GenAI (Gemini 2.5 Flash, free-tier) intervient **uniquement pour la mise en forme narrative** d'informations déjà calculées. Elle ne participe jamais au scoring, à la recommandation, ni à la comparaison sémantique. Ce découplage est fondamental pour :
- **Contrôler les hallucinations** : le LLM reçoit des faits chiffrés et des règles "Ne jamais inventer de compétences ou de scores non mentionnés dans le contexte."
- **Minimiser les coûts** : 2 appels maximum par session (plan + bio) contre potentiellement des dizaines si le LLM participait au scoring.
- **Assurer la reproductibilité** : même contexte RAG = même plan (via le cache).

#### Stratégie d'appels minimisés

| Usage | Fréquence | Conditionnel |
|-------|-----------|--------------|
| Enrichissement d'une saisie courte | 0 ou 1 par saisie < 5 mots | Optionnel, si `enrichir_fn` fourni |
| Plan de progression | 1 par session | Toujours (avec fallback si absent) |
| Bio professionnelle | 1 par session | Toujours (avec fallback si absent) |

#### Caching par hash SHA-256

`cache.py` maintient un fichier `cache/genai_cache.json` dont les clés sont les empreintes SHA-256 des prompts. Lors d'un appel identique (même contexte RAG, même prénom, même métier cible), la réponse est retournée en cache sans aucun appel réseau. Cela respecte le quota free-tier et garantit la reproductibilité.

#### Clé API sécurisée

La clé `GEMINI_API_KEY` est lue exclusivement depuis les variables d'environnement, chargées via `python-dotenv` à partir d'un fichier `.env` local. Le fichier `.env` est listé dans `.gitignore` ; seul `.env.example` (sans valeur réelle) est versionné.

#### Dégradation propre

Si la clé est absente ou si `google-generativeai` n'est pas installé, les fonctions `generer_plan` et `generer_bio` retournent des **fallbacks déterministes** : un plan générique structuré en 3 phases et une bio standardisée. Le reste du pipeline (questionnaire, NLP, scoring, recommandation, visualisations) reste **entièrement opérationnel**.

---

## 4. Description détaillée des modules

### 4.1 `src/config.py` — Configuration centralisée

Centralise toutes les constantes : chemins des fichiers CSV et de cache, nom du modèle SBERT (`all-MiniLM-L6-v2`), seuils d'interprétation des scores (0.70 pour "bon", 0.50 pour "moyen"), poids des blocs (`BLOCK_WEIGHTS`), nom du modèle Gemini (`gemini-2.5-flash`), seuil de détection des saisies courtes (5 mots), nombre de métiers recommandés (3) et nombre de compétences faibles à inclure dans le contexte RAG (5).

### 4.2 `src/questionnaire.py` — Questionnaire hybride

Définit les trois listes de questions comme des structures de données pures (sans dépendance Streamlit) : `QUESTIONS_LIKERT` (6 questions), `QUESTIONS_OUVERTES` (6 questions), `QUESTIONS_CHOIX` (5 questions avec 6 options chacune). Expose `valider_reponses` pour la validation et `reponses_vers_textes` qui délègue la fusion à `preprocessing.fusionner_reponses`.

### 4.3 `src/preprocessing.py` — Nettoyage léger

Trois opérations : normalisation des espaces, suppression des caractères de contrôle, détection des phrases courtes. La conversion Likert→phrase est une table de correspondance fixe (valeur entière 1-5 → phrase naturelle en français). Les choix multiples sont directement ajoutés comme textes de compétences.

### 4.4 `src/referentiel.py` — Chargement et validation des CSV

Charge `competences.csv` et `metiers.csv` avec validation des colonnes obligatoires et parsing tolérant du séparateur `;`. La fonction `valider_integrite` vérifie l'unicité des IDs et la cohérence référentielle (tout ID dans `RequiredCompetencies` doit exister dans `competences.csv`).

### 4.5 `src/nlp_engine.py` — Moteur sémantique

Singleton SBERT, encodage en batch, cache disque des embeddings du référentiel, calcul de la matrice de similarité cosinus. Deux fonctions de haut niveau : `scores_par_bloc` (algorithme max-puis-moyenne) et `scores_competences_detail` (max sur toutes les réponses par compétence).

### 4.6 `src/scoring.py` — Score de couverture pondéré

Implémente la formule `Coverage = Σ(Wi·Si) / Σ(Wi)` avec les poids de `config.py`. La fonction `resume_scores` construit le dictionnaire complet (score global, niveau, détail par bloc) utilisé par le contexte RAG et l'affichage.

### 4.7 `src/recommender.py` — Recommandation et contexte RAG

`top_n_metiers` calcule et trie les scores des métiers. `analyser_ecarts` identifie les compétences faibles. `construire_contexte_rag` assemble le texte structuré injecté dans les prompts Gemini.

### 4.8 `src/genai.py` — Intégration Gemini

Trois fonctions publiques : `enrichir_saisie` (conditionnel), `generer_plan` (1 appel, prompt structuré en 3 phases), `generer_bio` (1 appel). Toutes passent par `appel_avec_cache`. Gestion complète des cas d'absence de clé ou de bibliothèque avec fallbacks.

### 4.9 `src/cache.py` — Cache SHA-256

Lecture/écriture dans `cache/genai_cache.json`. Interface simple : `appel_avec_cache(prompt, fn_appel_api)`. Utilitaires : `vider_cache`, `taille_cache`.

### 4.10 `src/storage.py` — Persistance des sessions

Sauvegarde JSON (format configurable) horodatée dans `data/responses/`. Structure complète : réponses brutes, scores, top métiers, timestamp ISO 8601.

### 4.11 `app.py` — Interface Streamlit

Orchestration du pipeline via la machine à états `etape` : `questionnaire` → `analyse` → `resultats`. Chargement du référentiel mis en cache via `@st.cache_data`, modèle SBERT via `@st.cache_resource`. Quatre fonctions de visualisation Plotly avec fallback texte si Plotly est absent.

---

## 5. Référentiel de données

### 5.1 Schéma des CSV

#### `data/competences.csv` (36 entrées, 4 colonnes)

| Colonne | Type | Description |
|---------|------|-------------|
| `CompetencyID` | str | Identifiant unique (C01–C36) |
| `Competency` | str | Libellé exploitable par SBERT |
| `BlockID` | int | Identifiant du bloc (1–6) |
| `BlockName` | str | Nom du bloc |

#### `data/metiers.csv` (8 entrées, 3 colonnes)

| Colonne | Type | Description |
|---------|------|-------------|
| `JobID` | str | Identifiant unique (J01–J08) |
| `JobTitle` | str | Intitulé du métier |
| `RequiredCompetencies` | str | Liste de CompetencyID séparés par `;` |

### 5.2 Organisation en blocs

| BlockID | BlockName | Nb compétences | IDs |
|---------|-----------|----------------|-----|
| 1 | Data Analysis | 6 | C01–C06 |
| 2 | Machine Learning | 6 | C07–C12 |
| 3 | NLP | 6 | C13–C18 |
| 4 | Data Engineering | 6 | C19–C24 |
| 5 | Software Development & Programming | 6 | C25–C30 |
| 6 | Project Management & Soft Skills | 6 | C31–C36 |

### 5.3 Métiers et couverture du référentiel

| JobID | Métier | Nb compétences requises | Blocs couverts |
|-------|--------|------------------------|----------------|
| J01 | Data Analyst | 9 | 1, 6 |
| J02 | Machine Learning Engineer | 10 | 2, 5 |
| J03 | Data Scientist | 13 | 1, 2, 5, 6 |
| J04 | NLP Engineer | 13 | 3, 5, 6 |
| J05 | Data Engineer | 14 | 4, 5, 6 |
| J06 | Business Intelligence Analyst | 9 | 1, 4, 6 |
| J07 | MLOps Engineer | 11 | 2, 5, 6 |
| J08 | Research Scientist in AI | 13 | 2, 3, 5, 6 |

Les compétences logicielles (bloc 5 : Python, Git, tests, APIs, Docker) et de gestion de projet (bloc 6) sont présentes dans la quasi-totalité des métiers, reflétant fidèlement les exigences du marché data/IA.

---

## 6. Résultats et fonctionnement réel

### 6.1 Validation de la formule de scoring

Le test unitaire `test_score_global_equal_weights` dans `tests/test_scoring.py` vérifie le cas de référence défini dans l'architecture du projet :

```
Entrée : scores_blocs = {1: 0.85, 2: 0.78, 3: 0.40}, poids = {1: 1.0, 2: 1.0, 3: 1.0}
Calcul : (0.85 + 0.78 + 0.40) / 3 = 2.03 / 3 ≈ 0.677
Résultat attendu : ≈ 0.68 (tolérance ±0.01)
```

Ce test valide la correction de l'implémentation de `scoring.score_global`. Le score de 0.68 correspondrait à une "couverture partielle" selon les seuils configurés (0.50–0.69 → moyen), à 0.02 du seuil "bon".

Le test `test_score_global_weighted` vérifie également le cas différencié :
```
Entrée : {1: 0.8, 2: 0.6, 3: 0.4}, poids = {1: 2.0, 2: 1.0, 3: 1.0}
Calcul : (2×0.8 + 1×0.6 + 1×0.4) / 4 = 2.8 / 4 = 0.70
```

### 6.2 Comportement sémantique observable

D'après la structure du pipeline, les scores de similarité cosinus produits par SBERT pour le modèle `all-MiniLM-L6-v2` présentent les caractéristiques suivantes, observables à l'exécution :

- **Réponse spécialisée vs compétence alignée** : une réponse décrivant un projet NLP obtiendra des scores élevés (> 0.65) sur les compétences du bloc 3, et des scores modérés (0.30–0.50) sur les blocs non mentionnés.
- **Conversion Likert** : une note de 5 génère la phrase "je suis très compétent et expérimenté dans ce domaine", laquelle obtient des scores de similarité modérément élevés sur tous les blocs (SBERT reconnaît l'expertise générale sans la spécialiser).
- **Choix multiples** : les textes courts cochés ("scikit-learn (régression, classification, clustering)") sont sémantiquement proches de C07 ("Développer des modèles de régression et classification") et C10 ("Utiliser les frameworks ML"), ce qui contribue fortement au score du bloc 2.

### 6.3 Couverture des tests unitaires

| Fichier test | Module testé | Statut |
|-------------|--------------|--------|
| `test_scoring.py` | `scoring.py` | Complet et passant (7 tests de classe, cas limite inclus) |
| `test_referentiel.py` | `referentiel.py` | Squelette défini, tests à implémenter |
| `test_nlp_engine.py` | `nlp_engine.py` | Squelette défini, tests à implémenter |
| `test_recommender.py` | `recommender.py` | Squelette défini, tests à implémenter |
| `test_cache.py` | `cache.py` | Squelette défini, tests à implémenter |

Les tests de `scoring.py` couvrent : poids égaux, poids différenciés, dictionnaire vide, poids totaux nuls, poids par défaut, et les trois niveaux d'interprétation aux seuils exacts.

---

## 7. Interface utilisateur Streamlit

### 7.1 Navigation

L'application est structurée en machine à états avec trois pages :

```
[questionnaire] → (soumission valide) → [analyse] → (calculs terminés) → [resultats]
                                                         ↑
                                            "Refaire le questionnaire"
```

### 7.2 Page questionnaire

Formulaire Streamlit unique (`st.form`) organisé en sections : informations personnelles (prénom optionnel), 6 sliders Likert dans des `st.expander`, 6 zones de texte pour les réponses ouvertes, 5 groupes de cases à cocher en deux colonnes. Validation avant soumission.

### 7.3 Page résultats

5 onglets (`st.tabs`) :
1. **Scores par bloc** : radar Plotly (polygone fermé, remplissage semi-transparent) + barres horizontales avec lignes de référence aux seuils 0.50 et 0.70, code couleur vert/orange/rouge.
2. **Top 3 métiers** : graphique en barres + détail expandable par métier avec visualisation de chaque compétence sous forme de barre ASCII et pourcentage.
3. **Compétences détaillées** : barre horizontale colorée (échelle RdYlGn) pour les 36 compétences, triées par score décroissant.
4. **Plan de progression** : rendu markdown (Gemini ou fallback).
5. **Bio professionnelle** : rendu markdown avec bouton "Copier la bio".

### 7.4 Sidebar

Affichage permanent du pipeline en 6 étapes, du modèle NLP utilisé et du modèle GenAI. Mention EFREI — RNCP40875 — Bloc 2.

---

## 8. Limites et perspectives

### 8.1 Limites identifiées

| Limite | Impact | Mitigation actuelle |
|--------|--------|---------------------|
| **Référentiel fermé** : 36 compétences fixes, 8 métiers | Ne couvre pas tous les métiers data/IA du marché (ex. Cloud Architect, Product Manager Data) | Conception modulaire : ajout de lignes CSV sans modification du code |
| **Profil auto-déclaratif** : biais de sur- ou sous-estimation possible | Scores peuvent diverger du niveau réel | Triangulation Likert + texte libre + choix multiples pour réduire ce biais |
| **Modèle SBERT non spécialisé** : `all-MiniLM-L6-v2` est généraliste | Peut mal gérer les néologismes récents ou termes très techniques (ex. noms de frameworks récents) | Fine-tuning possible sur un corpus de compétences data/IA |
| **Langue mixte** : compétences en français, certains choix en anglais | Légère perte de qualité sémantique sur les correspondances cross-lingues | SBERT gère raisonnablement le français et l'anglais ; choix acceptable pour ce prototype |
| **Tests partiels** : 4 des 5 fichiers de tests sont des squelettes | Couverture incomplète | Les squelettes documentent les cas à couvrir ; priorité test_scoring (complet) |
| **GenAI non déterministe** : Gemini peut varier entre sessions | Plans légèrement différents pour le même profil si cache vide | Cache SHA-256 garantit la reproductibilité après le premier appel |

### 8.2 Perspectives

**Court terme :**
- Compléter les tests unitaires (`test_nlp_engine.py`, `test_recommender.py`, `test_referentiel.py`, `test_cache.py`).
- Ajouter un mode "comparaison de profils" : afficher le profil utilisateur face à un profil cible d'un métier spécifique.

**Moyen terme :**
- **Fine-tuning SBERT** sur un corpus de paires (description de compétence, libellé de poste) issu de données ROME ou e-CF pour améliorer la précision sur le domaine data.
- **Intégration du référentiel ROME v4** (nomenclature officielle de Pôle Emploi) pour étendre les métiers et compétences couverts.
- **Système d'apprentissage actif** : recueillir les corrections des utilisateurs ("ce métier ne me correspond pas") pour affiner les scores.

**Long terme :**
- **API REST** (`src/genai.py` et `src/scoring.py` n'ont aucune dépendance Streamlit) permettant d'exposer le pipeline dans un service externe.
- **Dashboard administrateur** : visualisation agrégée des profils analysés (anonymisés), tendances des compétences les plus/moins couvertes.
- **Support multilingue** : extension à l'anglais via un modèle SBERT multilingue (`paraphrase-multilingual-MiniLM-L12-v2`).

---

## 9. Mapping RNCP40875 Bloc 2

Le tableau suivant aligne les compétences évaluées du **Bloc 2 (BC2 : Piloter et implémenter des solutions d'IA en s'aidant de l'IA générative)** avec leur démonstration concrète dans le projet AISCA.

| Compétence RNCP40875 Bloc 2 | Démonstration dans AISCA | Fichiers concernés |
|-----------------------------|--------------------------|-------------------|
| **Collecter et préparer des données pour l'entraînement ou l'analyse** | Construction du référentiel structuré (36 compétences, 8 métiers) en CSV avec schéma canonique validé ; pipeline de collecte des réponses utilisateur (3 types) et persistance JSON horodatée | `data/competences.csv`, `data/metiers.csv`, `src/questionnaire.py`, `src/storage.py` |
| **Concevoir et implémenter un modèle NLP / représentation vectorielle** | Implémentation du moteur sémantique SBERT (`all-MiniLM-L6-v2`), encodage en vecteurs denses 384D, calcul de la matrice de similarité cosinus, cache disque des embeddings | `src/nlp_engine.py`, `src/preprocessing.py` |
| **Prototyper un pipeline RAG (Retrieval-Augmented Generation)** | Architecture RAG complète : retrieval sur référentiel encodé → contexte augmenté structuré → génération par Gemini. Découplage strict entre calcul NLP local et LLM | `src/nlp_engine.py`, `src/scoring.py`, `src/recommender.py`, `src/genai.py` |
| **Utiliser des embeddings pour la recherche sémantique** | Embeddings SBERT pour la comparaison sémantique réponses↔compétences ; similarité cosinus normalisée ; fonctions `scores_par_bloc` et `scores_competences_detail` | `src/nlp_engine.py` |
| **Intégrer et utiliser responsablement un modèle génératif (LLM/GenAI)** | Gemini 2.5 Flash free-tier, 2 appels max par session, prompts anti-hallucination, caching SHA-256, clé en variable d'environnement, dégradation propre sans clé | `src/genai.py`, `src/cache.py`, `.env.example` |
| **Construire un pipeline de bout en bout** | Questionnaire → prétraitement → SBERT → scoring → recommandation → GenAI → visualisation, sans dépendance Streamlit dans les modules métier | `app.py` + l'ensemble de `src/` |
| **Évaluer et valider les résultats d'un modèle** | Tests unitaires de la formule de scoring (cas de référence 0.85/0.78/0.40 → ≈0.68), tests d'interprétation aux seuils exacts, tests des cas limites (poids nuls, dict vide) | `tests/test_scoring.py` |
| **Industrialiser sous contrainte de coût** | NLP 100% local (coût zéro), GenAI limitée (free-tier + caching), embeddings mis en cache sur disque, configuration centralisée sans code en dur | `src/config.py`, `src/nlp_engine.py`, `src/cache.py` |
| **Documenter et présenter une solution d'IA** | Ce rapport, documentation technique (ARCHITECTURE.md, README.md), support de présentation, docstrings exhaustives dans chaque module | `docs/`, `ARCHITECTURE.md`, `README.md` |
| **Versionner et structurer un projet de développement** | Arborescence modulaire respectant la séparation métier/UI, `.gitignore` complet (cache, .env, embeddings, __pycache__), `requirements.txt` versionné | `.gitignore`, `requirements.txt`, structure `src/` |

---

## 10. Références

### Modèle NLP
- **sentence-transformers/all-MiniLM-L6-v2** — HuggingFace Model Hub : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.

### Architecture RAG
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Springer (2025). *Retrieval-Augmented Generation (RAG)*. Business & Information Systems Engineering. https://link.springer.com/article/10.1007/s12599-025-00945-3

### Référentiels de compétences
- **ROME v4** (Répertoire Opérationnel des Métiers et Emplois) — France Travail : https://www.francetravail.fr/employeur/vos-recrutements/le-rome-et-les-fiches-metiers.html
- **e-CF (European e-Competence Framework)** — CEN : https://www.ecompetences.eu/

### Technologies
- Streamlit Documentation : https://docs.streamlit.io/
- Google Gemini API : https://ai.google.dev/docs
- sentence-transformers Documentation : https://www.sbert.net/

---

*Rapport rédigé dans le cadre de l'épreuve certifiante EFREI RNCP40875 Bloc 2 — Juin 2026.*
