# AISCA — Support de Présentation
## Trame de slides pour la démo en classe

**Projet EFREI — RNCP40875 — Bloc 2**  
**BC2 : Piloter et implémenter des solutions d'IA en s'aidant de l'IA générative**

---

## Slide 1 — Titre et contexte

**Titre principal :** AISCA — Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences

**Sous-titre :** Analyse sémantique de profils data/IA et recommandation de métiers par RAG

**Informations :**
- EFREI — RNCP40875 — Bloc 2
- Juin 2026

**Points clés à l'oral :**
- Présenter le projet en une phrase : "Un système qui analyse ce que vous savez vraiment faire, pas ce que vous écrivez mot pour mot."
- Annoncer le plan : problème → solution → démo → résultats → conclusion.

---

## Slide 2 — La problématique

**Titre :** Pourquoi l'auto-évaluation classique ne suffit pas

**Contenu :**

Le problème des systèmes de matching classiques :

| Approche classique | Limite |
|--------------------|--------|
| Mots-clés / TF-IDF | "j'ai entraîné un réseau de neurones" ne matche pas "deep learning" |
| Formulaire à cases | Biais déclaratif, manque de nuance |
| Score manuel | Non reproductible, subjectif |

**Ce qu'on veut :**
- Comprendre le *sens* des expériences décrites, pas les mots exacts.
- Produire un score objectif, traçable et expliqué.
- Recommander des métiers pertinents et un plan d'action concret.

**Point clé à l'oral :** Illustrer avec l'exemple : "pipeline de features pour détection d'anomalies" → couvre C09, C08, C25 mais aucun mot exact en commun.

---

## Slide 3 — La solution : architecture RAG

**Titre :** AISCA — Pipeline RAG en 3 phases

**Schéma central :**

```
RETRIEVAL                AUGMENTED CONTEXT         GENERATION
─────────────────────    ─────────────────────────  ──────────────────
Embeddings SBERT     →   Scores par bloc         →  Plan de progression
Référentiel encodé       Score global pondéré       Bio professionnelle
Similarité cosinus        Top-3 métiers              (Gemini 2.5 Flash)
                         Compétences faibles
```

**Points clés :**
- Phase Retrieval : 100% locale, coût zéro.
- Contexte Augmenté : faits calculés, pas inventés.
- Génération : LLM contraint par le contexte (anti-hallucination).

**Point clé à l'oral :** Le RAG garantit que Gemini ne peut pas inventer de scores. Il reçoit des chiffres calculés et des règles strictes.

---

## Slide 4 — Le référentiel de données

**Titre :** 36 compétences × 8 métiers — la base de connaissance

**Tableau des blocs :**

| Bloc | Domaine | Exemples de compétences |
|------|---------|------------------------|
| 1 | Data Analysis | Visualisation, SQL analytique, métriques statistiques |
| 2 | Machine Learning | Régression, évaluation de modèles, deep learning |
| 3 | NLP | Tokenisation, BERT/GPT, systèmes Q&A |
| 4 | Data Engineering | ETL/ELT, Spark, pipelines de données |
| 5 | Software Dev. | Python, Git, Docker, APIs REST, tests unitaires |
| 6 | Project Mgmt. | Communication, RGPD, gestion d'équipe |

**8 métiers couverts :** Data Analyst, ML Engineer, Data Scientist, NLP Engineer, Data Engineer, BI Analyst, MLOps Engineer, Research Scientist in AI.

**Point clé à l'oral :** Inspiré du référentiel ROME et de l'e-CF européen. Extensible : ajouter un métier = ajouter une ligne CSV, zéro modification de code.

---

## Slide 5 — Le questionnaire hybride

**Titre :** Acquisition multi-modale des compétences

**3 types de questions :**

1. **6 questions Likert (1-5)** — auto-évaluation rapide par bloc
   - Exemple : "Quel est votre niveau en Machine Learning ?"
   - Conversion automatique : note 5 → "je suis très compétent et expérimenté dans ce domaine"

2. **6 questions ouvertes** — description d'expériences concrètes
   - Exemple : "Décrivez votre projet Machine Learning le plus représentatif."
   - Input principal de l'analyse sémantique SBERT.

3. **5 questions à choix multiples** — technologies maîtrisées
   - Exemple : "Quels frameworks ML avez-vous utilisés ?"
   - Les labels cochés deviennent directement des textes de compétences.

**Pourquoi hybride ?** Triangulation pour réduire les biais déclaratifs. Validé : au moins une réponse ouverte obligatoire.

---

## Slide 6 — Le moteur NLP : SBERT local

**Titre :** Analyse sémantique à coût zéro avec `all-MiniLM-L6-v2`

**Pourquoi SBERT plutôt que TF-IDF :**

| Critère | TF-IDF | SBERT |
|---------|--------|-------|
| Représentation | Vecteur creux (fréquences) | Vecteur dense (384 dimensions) |
| Synonymes | Non capturés | Capturés |
| Paraphrases | Non capturées | Capturées |
| Coût | Nul | Nul (100% local) |
| Taille modèle | N/A | 22 Mo, ~22M paramètres |

**Algorithme scores de bloc :**
1. Encoder toutes les réponses utilisateur.
2. Pour chaque réponse : similarité **max** avec les compétences du bloc.
3. Score du bloc = **moyenne** des maxima.

**Points clés à l'oral :** Le modèle tourne entièrement en local, aucun appel réseau pour le NLP. Singleton en mémoire, cache disque des embeddings du référentiel.

---

## Slide 7 — Le score de couverture pondéré

**Titre :** `Coverage = Σ(Wi · Si) / Σ(Wi)`

**Cas de référence (testé unitairement) :**

```
Blocs     : Data Analysis=0.85, Machine Learning=0.78, NLP=0.40
Poids     : Wi = 1.0 pour chaque bloc
Score     : (0.85 + 0.78 + 0.40) / 3 = 0.677 ≈ 0.68
Niveau    : Couverture partielle (seuil "bon" à 0.70)
```

**Grille d'interprétation :**

| Score | Niveau | Couleur |
|-------|--------|---------|
| ≥ 0.70 | Bonne couverture | Vert |
| 0.50 – 0.69 | Couverture partielle | Orange |
| < 0.50 | Couverture insuffisante | Rouge |

**Extensibilité :** Modifier `BLOCK_WEIGHTS` dans `config.py` pour pondérer un bloc selon le poste visé (ex. NLP Engineer → poids 2.0 sur le bloc NLP).

---

## Slide 8 — La recommandation de métiers

**Titre :** Top-3 métiers : matching par couverture de compétences

**Algorithme :**
1. Pour chaque métier : récupérer les scores de ses compétences requises.
2. Score métier = **moyenne** des scores de compétences requises.
3. Trier les 8 métiers par score décroissant.
4. Retenir les 3 premiers.

**Exemple illustratif :**

| Métier | Compétences requises | Score indicatif |
|--------|---------------------|-----------------|
| NLP Engineer | C13–C18 + Python, Git, Docker... | Élevé si profil NLP fort |
| Data Scientist | C01–C12 + Python, Git... | Élevé si profil Data + ML |
| Data Engineer | C19–C28 + Python, Git, Cloud... | Élevé si profil infra fort |

**Analyse des écarts :** Les 5 compétences les moins bien couvertes alimentent le plan de progression. Entièrement traçable par compétence.

---

## Slide 9 — La GenAI responsable

**Titre :** Gemini 2.5 Flash — 2 appels maximum, toujours contrôlés

**Principe fondamental :**

```
NLP LOCAL          →   CONTEXTE RAG     →   GEMINI
(calcule les faits)    (faits structurés)   (met en forme uniquement)
```

**3 mesures de contrôle des coûts et de la qualité :**

1. **Appels minimisés** : max 2 par session (plan + bio). Enrichissement court texte : conditionnel et optionnel.

2. **Caching SHA-256** : `cache/genai_cache.json`. Même prompt = zéro appel réseau. Persistant entre sessions.

3. **Sécurité** : `GEMINI_API_KEY` en variable d'environnement, jamais dans le code. `.env` dans `.gitignore`.

**Dégradation propre :** Sans clé API, le pipeline NLP, le scoring et la recommandation restent opérationnels. Plan et bio affichent un fallback structuré.

**Prompts anti-hallucination :** "Ne jamais inventer de compétences ou de scores non mentionnés dans le contexte."

---

## Slide 10 — Démo live

**Titre :** Démonstration — `streamlit run app.py`

**Scénario de démo recommandé (profil NLP/ML) :**

**Étape 1 — Questionnaire (2 min) :**
- Prénom : à saisir en direct.
- Likert : NLP = 4, ML = 3, autres = 2.
- Ouverte O02 : "J'ai implémenté un classifieur BERT pour la détection de spam en fine-tuning avec HuggingFace Transformers sur 50 000 emails."
- Ouverte O03 : "J'ai réalisé une analyse de sentiment sur des avis clients avec spaCy et un modèle CamemBERT."
- Choix CM02 : cocher scikit-learn + PyTorch. Choix CM03 : cocher HuggingFace + embeddings SBERT.

**Étape 2 — Analyse (1 min, spinners visibles) :**
- Montrer les étapes successives (NLP, scoring, recommandation, GenAI).

**Étape 3 — Résultats (3 min) :**
- Onglet 1 : Radar (NLP et ML en tête).
- Onglet 2 : Top-3 (NLP Engineer en premier attendu).
- Onglet 3 : Carte de compétences.
- Onglet 4 : Plan de progression Gemini.
- Onglet 5 : Bio professionnelle Gemini.

**Points à commenter :** Score de chaque bloc, cohérence du top-3 avec le profil saisi, plan en 3 phases.

---

## Slide 11 — Architecture technique et qualité

**Titre :** Code modulaire, testable, industrialisé

**Arborescence en 3 couches :**

```
app.py (UI Streamlit)
    └── src/ (logique métier, sans dépendance Streamlit)
            ├── config.py         (configuration centralisée)
            ├── questionnaire.py  (questions : données pures)
            ├── preprocessing.py  (nettoyage léger)
            ├── referentiel.py    (chargement CSV + validation)
            ├── nlp_engine.py     (SBERT singleton + cosinus)
            ├── scoring.py        (Coverage = Σ(Wi·Si)/Σ(Wi))
            ├── recommender.py    (top-3 + contexte RAG)
            ├── genai.py          (Gemini + fallbacks)
            └── cache.py          (SHA-256, JSON)
```

**Tests unitaires :** `tests/test_scoring.py` complet et passant (formule, cas limites, niveaux d'interprétation). Squelettes définis pour les 4 autres modules.

**Installation :** `pip install -r requirements.txt` + `streamlit run app.py`.

---

## Slide 12 — Mapping RNCP40875 Bloc 2 et conclusion

**Titre :** Compétences RNCP démontrées dans AISCA

| Compétence Bloc 2 | Démonstration |
|-------------------|---------------|
| Collecter et préparer des données | Référentiel CSV + questionnaire + storage JSON |
| Concevoir un modèle NLP | SBERT `all-MiniLM-L6-v2`, embeddings 384D, cosinus |
| Prototyper un pipeline RAG | Retrieval → contexte augmenté → Gemini |
| Utiliser des embeddings sémantiques | `scores_par_bloc`, `scores_competences_detail` |
| Intégrer la GenAI responsablement | 2 appels max, cache, clé env, dégradation propre |
| Pipeline de bout en bout | Questionnaire → NLP → scoring → reco → viz |
| Évaluer et valider | Tests scoring (cas 0.85/0.78/0.40 → ≈0.68) |
| Industrialiser sous contrainte de coût | NLP local gratuit, cache, configuration externe |
| Documenter | Rapport, ARCHITECTURE.md, README, docstrings |

**Conclusion :**
- AISCA démontre qu'un pipeline RAG complet est réalisable avec des outils open-source gratuits (SBERT) et un usage minimal et maîtrisé de la GenAI.
- La séparation stricte NLP local / GenAI garantit fiabilité, coût maîtrisé et absence d'hallucinations dans les scores.
- Perspectives : fine-tuning SBERT sur corpus ROME, extension du référentiel, API REST.

**Questions ?**

---

*Support de présentation — AISCA — EFREI RNCP40875 Bloc 2 — Juin 2026*
