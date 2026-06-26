# ARCHITECTURE — AISCA

**Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences**
Épreuve certifiante EFREI — RNCP40875 — Bloc 2 (BC2 : Piloter et implémenter des solutions d'IA en s'aidant de l'IA générative).
Mini-agent **RAG** : questionnaire hybride → analyse sémantique SBERT locale → score de couverture pondéré → recommandation top-3 métiers → GenAI (plan + bio) avec cache.

Document de cadrage destiné aux agents `developpeur`, `testeur`, `rapporteur`, `sourceur`. Il fait foi sur l'arborescence, le pipeline, le schéma de données et les contraintes.

---

## 1. Arborescence du projet

```
AISCA/
├── app.py                      # Point d'entrée Streamlit (UI : questionnaire + résultats + viz)
├── README.md                   # Lancement, install, variables d'env
├── ARCHITECTURE.md             # Ce document
├── requirements.txt            # streamlit, sentence-transformers, pandas, numpy, plotly, google-generativeai, python-dotenv
├── .env.example                # GEMINI_API_KEY=... (modèle, jamais committer le vrai .env)
├── .gitignore                  # .env, __pycache__/, cache/genai_cache.json, embeddings .npy, *.pyc
│
├── src/                        # Code métier (importable, testable, sans Streamlit)
│   ├── __init__.py
│   ├── config.py               # Constantes : chemins, nom du modèle SBERT, seuils, poids des blocs, paramètres GenAI
│   ├── questionnaire.py        # EF1 : définition des questions (Likert + ouvertes + guidées), structure des réponses
│   ├── storage.py              # EF1.2 : persistance des réponses (CSV/JSON/SQLite) dans data/responses/
│   ├── preprocessing.py        # Nettoyage texte, fusion réponses ouvertes, détection phrases trop courtes (<5 mots)
│   ├── referentiel.py          # EF2.1 : chargement + validation de data/competences.csv et data/metiers.csv
│   ├── nlp_engine.py           # EF2.2/EF2.3 : modèle SBERT (singleton), encodage, similarité cosinus
│   ├── scoring.py              # EF3.1 : score de couverture pondéré par bloc (formule Σ Wi·Si / Σ Wi)
│   ├── recommender.py          # EF3.2 : score par métier + classement top-3
│   ├── genai.py                # EF4 : appels Gemini (enrichissement / plan / bio) — UN appel max par sortie
│   └── cache.py                # Caching obligatoire des appels GenAI (clé = hash du prompt)
│
├── data/
│   ├── competences.csv         # Référentiel des compétences (schéma canonique — voir §4)
│   ├── metiers.csv             # Référentiel des métiers (schéma canonique — voir §4)
│   └── responses/              # Réponses utilisateur stockées (CSV/JSON/SQLite) — gitignored si données réelles
│
├── cache/
│   └── genai_cache.json        # Cache local des réponses GenAI (réutilisé si prompt identique)
│
├── tests/                      # Tests unitaires (pytest)
│   ├── test_referentiel.py     # Schéma/intégrité des CSV, cohérence des IDs métiers ↔ compétences
│   ├── test_nlp_engine.py      # Encodage non vide, cosinus ∈ [-1,1], cohérence sémantique de base
│   ├── test_scoring.py         # Formule pondérée (cas du sujet : 0.85/0.78/0.40 → 0.68)
│   ├── test_recommender.py     # Top-3 trié décroissant, pas de doublon
│   └── test_cache.py           # 2e appel identique → 0 appel réseau (mock GenAI)
│
├── docs/
│   ├── rapport.md              # Rapport projet (architecture, choix, justifications, RNCP Bloc 2)
│   └── presentation.pdf|pptx   # Support de présentation/démo
│
└── notebooks/                  # (optionnel) exploration SBERT / calibrage seuils
    └── exploration.ipynb
```

**Principe de séparation** : tout le métier est dans `src/` (sans dépendance Streamlit) afin d'être testable indépendamment ; `app.py` n'orchestre que l'UI et appelle `src/`.

---

## 2. Pipeline de bout en bout (RAG)

```
[1] ACQUISITION (EF1) ── questionnaire.py + app.py
    Questionnaire hybride : échelles de Likert + questions ouvertes (texte libre) + questions guidées.
        │  réponses → storage.py → data/responses/ (format structuré CSV/JSON/SQLite)
        ▼
[2] PRÉTRAITEMENT ── preprocessing.py
    Fusion des réponses ouvertes, nettoyage léger (pas de sur-nettoyage : SBERT a besoin du contexte).
    Détection des phrases < 5 mots → enrichissement GenAI CONDITIONNEL (EF4.1, optionnel, 0 ou 1 appel).
        ▼
[3] EMBEDDINGS SBERT (EF2.2) ── nlp_engine.py        ┌── RETRIEVAL (RAG) ──┐
    Modèle local open-source `all-MiniLM-L6-v2`.      │  Référentiel encodé │
    Encodage des réponses utilisateur ET des phrases  │  une fois, mis en   │
    de compétences (referentiel.py).                  │  cache mémoire.     │
        ▼                                             └─────────────────────┘
[4] SIMILARITÉ COSINUS (EF2.3) ── nlp_engine.py
    cos_sim(user_inputs, compétences). Pour chaque bloc : max par input, moyenne sur les inputs → score du bloc Si.
        ▼
[5] SCORE DE COUVERTURE PONDÉRÉ (EF3.1) ── scoring.py
    Score par bloc Si ; score global = Σ(Wi·Si) / Σ(Wi)  (Wi = poids du bloc, défaut 1, configurable dans config.py).
        ▼
[6] RECOMMANDATION TOP-3 MÉTIERS (EF3.2) ── recommender.py
    Pour chaque métier : agrégation des similarités sur ses RequiredCompetencies → score métier.
    Tri décroissant → 3 métiers les mieux couverts. (Analyse des écarts = compétences à plus faible score.)
        ▼
[7] GENAI — CONTEXTE AUGMENTÉ + GÉNÉRATION (EF4) ── genai.py + cache.py
    Augmented context = scores de blocs + compétences faibles + top-3 métiers, injectés dans le prompt.
    • Plan de progression  : UN seul appel (EF4.2)
    • Bio professionnelle  : UN seul appel (EF4.3)
    Chaque appel passe par cache.py : si prompt déjà vu → réponse rejouée, aucun appel réseau.
        ▼
[8] RESTITUTION / VISUALISATION ── app.py
    Radar (scores par bloc), barres (top-3 métiers), tableau de similarités, plan + bio affichés.
```

Le caractère **RAG** est garanti par les étapes [3]→[6] (Retrieval sur le référentiel maîtrisé) qui construisent le contexte fiable injecté en [7], évitant les hallucinations et limitant les appels.

---

## 3. Découpage en lots de travail

| Lot | Périmètre | Fichiers principaux | Agent | Dépend de |
|-----|-----------|---------------------|-------|-----------|
| **L0 — Setup** | Squelette repo, `requirements.txt`, `.env.example`, `.gitignore`, `config.py` | racine + `src/config.py` | `developpeur` | — |
| **L1 — Référentiel de données** | Construire `data/competences.csv` et `data/metiers.csv` (schéma §4) inspirés ROME / e-CF ; ≥ 3 blocs, ≥ 15 compétences, ≥ 6 métiers | `data/*.csv` | `sourceur` | L0 |
| **L2 — Acquisition / questionnaire** | Questions Likert + ouvertes + guidées, stockage structuré | `questionnaire.py`, `storage.py` | `developpeur` | L0 |
| **L3 — Prétraitement** | Fusion/nettoyage, détection phrases courtes | `preprocessing.py` | `developpeur` | L2 |
| **L4 — Moteur NLP SBERT** | Chargement référentiel, modèle SBERT singleton, encodage, cosinus | `referentiel.py`, `nlp_engine.py` | `developpeur` | L1, L3 |
| **L5 — Scoring** | Score de bloc + couverture pondérée | `scoring.py` | `developpeur` | L4 |
| **L6 — Recommandation** | Score par métier, top-3, écarts | `recommender.py` | `developpeur` | L4, L5 |
| **L7 — GenAI + cache** | Client Gemini, prompts (enrichissement/plan/bio), cache JSON | `genai.py`, `cache.py` | `developpeur` | L5, L6 |
| **L8 — UI Streamlit + viz** | Questionnaire web, radar/barres, affichage plan+bio | `app.py` | `developpeur` | L2→L7 |
| **L9 — Tests** | Unitaires sur référentiel, NLP, scoring, reco, cache | `tests/` | `testeur` | L4→L7 |
| **L10 — Rapport** | Rapport technique, justification des choix, mapping RNCP Bloc 2 | `docs/rapport.md` | `rapporteur` | L1→L9 |
| **L11 — Présentation** | Slides + scénario de démo | `docs/presentation.*` | `rapporteur` | L8, L10 |
| **L12 — Sources** | Liens ROME / e-CF, modèle SBERT, doc Gemini ; bibliographie pour rapport | `docs/rapport.md` (annexe) | `sourceur` | transverse |

**Séquencement** : L0 → L1/L2 (parallélisables) → L3 → L4 → L5 → L6 → L7 → L8 ; L9 dès qu'un module métier est prêt ; L10/L11/L12 en continu, finalisés en fin de cycle.

---

## 4. Schéma de données canonique (CONTRAT — à respecter par tous les agents)

### `data/competences.csv`
| Colonne | Type | Description |
|---------|------|-------------|
| `CompetencyID` | str | Identifiant unique, ex. `C01` |
| `Competency` | str | Libellé court de la compétence (phrase exploitable par SBERT), ex. `data cleaning` |
| `BlockID` | int | Identifiant du bloc, ex. `1` |
| `BlockName` | str | Nom du bloc, ex. `Data Analysis` |

### `data/metiers.csv`
| Colonne | Type | Description |
|---------|------|-------------|
| `JobID` | str | Identifiant unique, ex. `J01` |
| `JobTitle` | str | Intitulé du métier, ex. `Data Analyst` |
| `RequiredCompetencies` | str | Liste de `CompetencyID` séparés par `;`, ex. `C01;C02;C03` |

**Règles d'intégrité** (vérifiées en L9) :
- `CompetencyID` et `JobID` uniques.
- Tout ID listé dans `RequiredCompetencies` existe dans `competences.csv`.
- Un même `CompetencyID` peut apparaître dans plusieurs métiers (ex. Python ∈ plusieurs postes).
- Séparateur strict `;` (sans espace imposé, mais parsing tolérant aux espaces).

**Exemple minimal de référence (à étendre par le `sourceur`)** :

```
# competences.csv
CompetencyID,Competency,BlockID,BlockName
C01,data cleaning,1,Data Analysis
C02,data visualization,1,Data Analysis
C03,Python programming,1,Data Analysis
C04,regression,2,Machine Learning
C05,neural networks,2,Machine Learning
C06,tokenization,3,NLP
C07,transformers,3,NLP

# metiers.csv
JobID,JobTitle,RequiredCompetencies
J01,Data Analyst,C01;C02;C03
J02,ML Engineer,C03;C04;C05
J03,Data Scientist,C01;C02;C03;C04;C05;C06;C07
J04,NLP Engineer,C03;C05;C06;C07
```

---

## 5. Contraintes clés à respecter

### NLP local — coût zéro (EF2)
- Embeddings via **`sentence-transformers`**, modèle **`all-MiniLM-L6-v2`** local et open-source. **Aucun** appel payant/réseau pour le cœur NLP.
- Similarité = **cosinus** (`util.cos_sim`). Référentiel encodé **une seule fois** (cache mémoire / `.npy` optionnel) pour la performance.
- Le modèle SBERT est chargé en **singleton** (`@st.cache_resource` côté UI) — pas de rechargement par interaction.

### GenAI strictement limitée + caching obligatoire (EF4)
- **Gemini 2.5 Flash** free-tier. Usage **minimal** : enrichissement = conditionnel (phrase < 5 mots, optionnel) ; **plan = 1 appel** ; **bio = 1 appel**.
- **Caching obligatoire** dans `cache.py` (`cache/genai_cache.json`) : clé = hash du prompt ; un prompt déjà vu n'engendre **aucun** nouvel appel.
- La GenAI ne décide jamais des scores/recommandations : elle ne fait que **rédiger** à partir du contexte RAG (anti-hallucination, contrôle total).
- Le système doit **fonctionner sans clé API** (dégradation propre : pas de plan/bio générés, le reste du pipeline reste opérationnel).

### Sécurité / configuration
- **Clé API en variable d'environnement** (`GEMINI_API_KEY`), chargée via `.env` (python-dotenv) ; **jamais** en dur dans le code, `.env` dans `.gitignore`.
- `.env.example` documente les variables sans secret.

### Livrables & qualité (RNCP Bloc 2)
- Livrables : **code fonctionnel** + **documentation technique** + **rapport** + **support de présentation**, poussés sur **Git/GitHub**.
- Versioning Git régulier (commits par lot), `README.md` clair (install + run : `streamlit run app.py`).
- Rapport explicite le **mapping avec les compétences RNCP Bloc 2** (collecte de données, modèle NLP, évaluation, pipeline end-to-end, industrialisation/coût, GenAI responsable).
- Travail en binôme ; présentation/démo finale avec participation de tous.
```
