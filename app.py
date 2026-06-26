"""
app.py — Interface Streamlit du projet AISCA.

Point d'entrée de l'application.
Lancement : streamlit run app.py

Pipeline orchestré :
  1. Questionnaire hybride (Likert + texte libre + cases à cocher)
  2. Prétraitement et préparation des textes
  3. Analyse sémantique SBERT (locale, coût zéro)
  4. Scoring de couverture pondéré
  5. Recommandation top-3 métiers
  6. Génération GenAI (plan + bio) avec cache
  7. Visualisations (radar + barres)
"""

import sys
import logging
from pathlib import Path

# Ajout du répertoire racine au path pour les imports src.*
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# Imports métier (sans dépendance Streamlit)
from src.config import TOP_N_JOBS, SEUIL_BON, SEUIL_MOYEN
from src.referentiel import charger_competences, charger_metiers, get_blocs, get_metiers
from src.questionnaire import (
    QUESTIONS_LIKERT,
    QUESTIONS_OUVERTES,
    QUESTIONS_CHOIX,
    valider_reponses,
    reponses_vers_textes,
)
from src.preprocessing import preparer_reponses_ouvertes
from src.nlp_engine import scores_par_bloc, scores_competences_detail
from src.scoring import resume_scores
from src.recommender import top_n_metiers, analyser_ecarts, construire_contexte_rag
from src.storage import sauvegarder_session
from src.genai import generer_plan, generer_bio

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration de la page Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AISCA — Cartographie des Compétences",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Chargement des données (avec cache Streamlit pour éviter les rechargements)
# ---------------------------------------------------------------------------

@st.cache_data
def _charger_referentiel():
    """Charge le référentiel une seule fois et le met en cache Streamlit."""
    df_comp = charger_competences()
    df_met = charger_metiers()
    blocs = get_blocs(df_comp)
    metiers = get_metiers(df_met, df_comp)
    return df_comp, df_met, blocs, metiers


@st.cache_resource
def _charger_modele_sbert():
    """Charge le modèle SBERT en singleton (une seule instance par session serveur)."""
    from src.nlp_engine import get_model
    return get_model()


# ---------------------------------------------------------------------------
# Initialisation de l'état de session Streamlit
# ---------------------------------------------------------------------------

def _init_session_state():
    """Initialise les variables de session avec des valeurs par défaut."""
    defaults = {
        "etape": "questionnaire",  # "questionnaire" | "resultats"
        "prenom": "",
        "reponses_likert": {},
        "reponses_ouvertes": {},
        "reponses_choix": {},
        "resultats": None,
    }
    for cle, valeur in defaults.items():
        if cle not in st.session_state:
            st.session_state[cle] = valeur


# ---------------------------------------------------------------------------
# Page : Questionnaire
# ---------------------------------------------------------------------------

def page_questionnaire(blocs: dict, metiers: dict):
    """Affiche le formulaire de questionnaire hybride."""

    st.title("AISCA — Analyse Sémantique des Compétences")
    st.markdown(
        "Répondez au questionnaire ci-dessous pour obtenir une analyse personnalisée "
        "de vos compétences et des recommandations de métiers adaptées à votre profil."
    )
    st.markdown("---")

    with st.form(key="questionnaire_form"):

        # ---- Informations personnelles ----
        st.subheader("Informations")
        prenom = st.text_input(
            "Votre prénom (optionnel)",
            placeholder="Ex : Marie",
            help="Utilisé uniquement pour personnaliser les résultats."
        )

        st.markdown("---")

        # ---- Questions Likert ----
        st.subheader("Auto-évaluation par domaine")
        st.caption(
            "Évaluez votre niveau sur une échelle de 1 (débutant) à 5 (expert)."
        )

        reponses_likert = {}
        for q in QUESTIONS_LIKERT:
            bloc_info = blocs.get(q["bloc_id"], {})
            with st.expander(f"{q['texte']}", expanded=True):
                if q.get("description"):
                    st.caption(q["description"])
                valeur = st.slider(
                    label=f"Niveau — {bloc_info.get('name', '')}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"likert_{q['id']}",
                    help=f"{q['min_label']} → {q['max_label']}"
                )
                reponses_likert[q["id"]] = valeur

        st.markdown("---")

        # ---- Questions ouvertes ----
        st.subheader("Expériences et projets")
        st.caption(
            "Décrivez vos expériences concrètes. Plus votre description est précise, "
            "meilleure sera l'analyse sémantique."
        )

        reponses_ouvertes = {}
        for q in QUESTIONS_OUVERTES:
            texte = st.text_area(
                label=q["texte"],
                placeholder=q.get("placeholder", ""),
                help=q.get("aide", ""),
                height=120,
                key=f"ouverte_{q['id']}"
            )
            reponses_ouvertes[q["id"]] = texte

        st.markdown("---")

        # ---- Questions à choix multiples ----
        st.subheader("Technologies et outils maîtrisés")
        st.caption("Cochez toutes les technologies que vous avez utilisées dans un contexte professionnel ou académique.")

        reponses_choix = {}
        for q in QUESTIONS_CHOIX:
            st.markdown(f"**{q['texte']}**")
            selections = []
            cols = st.columns(2)
            for i, option in enumerate(q["options"]):
                col = cols[i % 2]
                if col.checkbox(option, key=f"choix_{q['id']}_{i}"):
                    selections.append(option)
            reponses_choix[q["id"]] = selections
            st.markdown("")

        st.markdown("---")

        # ---- Bouton de soumission ----
        soumis = st.form_submit_button(
            label="Analyser mon profil",
            type="primary",
            use_container_width=True
        )

    if soumis:
        erreurs = valider_reponses(reponses_likert, reponses_ouvertes, reponses_choix)
        if erreurs:
            for err in erreurs:
                st.error(err)
        else:
            st.session_state["prenom"] = prenom.strip()
            st.session_state["reponses_likert"] = reponses_likert
            st.session_state["reponses_ouvertes"] = reponses_ouvertes
            st.session_state["reponses_choix"] = reponses_choix
            st.session_state["etape"] = "analyse"
            st.rerun()


# ---------------------------------------------------------------------------
# Page : Analyse (phase de calcul avec spinner)
# ---------------------------------------------------------------------------

def page_analyse(df_comp, blocs, metiers):
    """Lance l'analyse et stocke les résultats dans session_state."""

    st.title("Analyse en cours...")

    prenom = st.session_state["prenom"]
    reponses_likert = st.session_state["reponses_likert"]
    reponses_ouvertes = st.session_state["reponses_ouvertes"]
    reponses_choix = st.session_state["reponses_choix"]

    with st.spinner("Chargement du modèle NLP..."):
        _charger_modele_sbert()

    with st.spinner("Prétraitement des réponses..."):
        textes_bruts = reponses_vers_textes(reponses_ouvertes, reponses_likert, reponses_choix)
        textes_prepares, _ = preparer_reponses_ouvertes(textes_bruts)
        if not textes_prepares:
            textes_prepares = textes_bruts  # fallback si tout filtré

    with st.spinner("Analyse sémantique SBERT en cours (coût zéro)..."):
        # Scores par bloc
        scores_blocs = scores_par_bloc(textes_prepares, blocs)

        # Scores par compétence individuelle
        comp_ids = df_comp["CompetencyID"].tolist()
        comp_texts = df_comp["Competency"].tolist()
        scores_comp = scores_competences_detail(textes_prepares, comp_ids, comp_texts)

    with st.spinner("Calcul des scores de couverture..."):
        noms_blocs = {bid: info["name"] for bid, info in blocs.items()}
        resume = resume_scores(scores_blocs, noms_blocs)

    with st.spinner("Recommandation des métiers..."):
        top_metiers = top_n_metiers(metiers, scores_comp, n=TOP_N_JOBS)

    with st.spinner("Identification des compétences à renforcer..."):
        comp_texts_dict = dict(zip(comp_ids, comp_texts))
        competences_faibles = analyser_ecarts(scores_comp, comp_texts_dict)
        contexte_rag = construire_contexte_rag(resume, top_metiers, competences_faibles)

    with st.spinner("Génération du plan de progression (GenAI)..."):
        plan = generer_plan(contexte_rag, prenom)

    with st.spinner("Génération de la bio professionnelle (GenAI)..."):
        metier_cible = top_metiers[0]["title"] if top_metiers else ""
        bio = generer_bio(contexte_rag, prenom, metier_cible)

    # Sauvegarde de la session
    sauvegarder_session(
        prenom=prenom,
        reponses_likert=reponses_likert,
        reponses_ouvertes=reponses_ouvertes,
        reponses_choix=reponses_choix,
        scores_blocs=scores_blocs,
        score_global=resume["score_global"],
        top_metiers=[
            {"job_id": m["job_id"], "title": m["title"], "score": m["score"]}
            for m in top_metiers
        ]
    )

    # Stockage des résultats dans l'état de session
    st.session_state["resultats"] = {
        "resume": resume,
        "scores_blocs": scores_blocs,
        "scores_comp": scores_comp,
        "top_metiers": top_metiers,
        "competences_faibles": competences_faibles,
        "contexte_rag": contexte_rag,
        "plan": plan,
        "bio": bio,
    }
    st.session_state["etape"] = "resultats"
    st.rerun()


# ---------------------------------------------------------------------------
# Page : Résultats
# ---------------------------------------------------------------------------

def page_resultats(blocs: dict):
    """Affiche les résultats de l'analyse avec visualisations."""

    resultats = st.session_state["resultats"]
    prenom = st.session_state["prenom"]
    resume = resultats["resume"]
    top_metiers = resultats["top_metiers"]
    competences_faibles = resultats["competences_faibles"]
    plan = resultats["plan"]
    bio = resultats["bio"]

    # Titre
    salutation = f"Bonjour {prenom} !" if prenom else "Voici votre analyse !"
    st.title(f"Résultats AISCA — {salutation}")

    # Bouton pour refaire le questionnaire
    if st.button("Refaire le questionnaire", type="secondary"):
        for cle in ["etape", "prenom", "reponses_likert", "reponses_ouvertes",
                    "reponses_choix", "resultats"]:
            if cle in st.session_state:
                del st.session_state[cle]
        st.rerun()

    st.markdown("---")

    # ---- Score global ----
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        score = resume["score_global"]
        niveau = resume["niveau_global"]
        couleur = {"bon": "green", "moyen": "orange", "faible": "red"}.get(niveau, "gray")

        st.metric(
            label="Score de couverture global",
            value=f"{score:.1%}",
            delta=resume["description_globale"]
        )

        # Barre de progression colorée
        st.progress(score)

    st.markdown("---")

    # ---- Tabs pour organiser les résultats ----
    onglets = st.tabs([
        "Scores par bloc",
        "Top 3 métiers",
        "Compétences détaillées",
        "Plan de progression",
        "Bio professionnelle"
    ])

    # ---- Onglet 1 : Radar + barres des scores par bloc ----
    with onglets[0]:
        st.subheader("Scores de couverture par bloc de compétences")

        blocs_data = resume["blocs"]
        noms = [b["nom"] for b in blocs_data]
        scores_vals = [b["score"] for b in blocs_data]

        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            _afficher_radar(noms, scores_vals)

        with col_graph2:
            _afficher_barres_blocs(blocs_data)

        # Tableau de détail
        st.markdown("**Détail par bloc :**")
        for bloc in blocs_data:
            emoji_niveau = {"bon": "✅", "moyen": "⚠️", "faible": "❌"}.get(bloc["niveau"], "")
            st.markdown(
                f"{emoji_niveau} **{bloc['nom']}** : {bloc['score']:.1%} — {bloc['description']}"
            )

    # ---- Onglet 2 : Top 3 métiers ----
    with onglets[1]:
        st.subheader("Métiers les mieux couverts par votre profil")

        _afficher_barres_metiers(top_metiers)

        st.markdown("---")
        for i, metier in enumerate(top_metiers, 1):
            with st.expander(f"#{i} — {metier['title']} ({metier['score']:.1%})", expanded=(i == 1)):
                st.markdown(f"**Score de couverture : {metier['score']:.1%}**")
                st.markdown("Compétences requises et leur couverture :")
                for cid, score in sorted(metier["competency_scores"].items(),
                                          key=lambda x: -x[1]):
                    barre = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                    st.markdown(f"- `{cid}` {barre} {score:.1%}")

    # ---- Onglet 3 : Carte de similarité compétences ----
    with onglets[2]:
        st.subheader("Couverture de toutes les compétences")
        _afficher_heatmap_competences(resultats["scores_comp"])

        st.markdown("---")
        st.markdown("**Compétences prioritaires à renforcer :**")
        for comp in competences_faibles:
            st.markdown(f"- {comp['texte']} *(score : {comp['score']:.1%})*")

    # ---- Onglet 4 : Plan de progression ----
    with onglets[3]:
        st.subheader("Plan de progression personnalisé")
        st.markdown(plan)

    # ---- Onglet 5 : Bio professionnelle ----
    with onglets[4]:
        st.subheader("Bio professionnelle")
        st.markdown(bio)


# ---------------------------------------------------------------------------
# Fonctions de visualisation
# ---------------------------------------------------------------------------

def _afficher_radar(noms: list, scores: list):
    """Affiche un graphique radar des scores par bloc."""
    try:
        import plotly.graph_objects as go

        # Fermeture du polygone
        noms_fermes = noms + [noms[0]]
        scores_fermes = scores + [scores[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores_fermes,
            theta=noms_fermes,
            fill="toself",
            name="Profil",
            line_color="#4A90D9",
            fillcolor="rgba(74, 144, 217, 0.3)"
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat=".0%"
                )
            ),
            showlegend=False,
            title="Radar de compétences",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly non disponible — graphique radar non affiché.")
        for nom, score in zip(noms, scores):
            st.text(f"{nom}: {score:.1%}")


def _afficher_barres_blocs(blocs_data: list):
    """Affiche un graphique en barres horizontales des scores par bloc."""
    try:
        import plotly.graph_objects as go

        noms = [b["nom"] for b in blocs_data]
        scores = [b["score"] for b in blocs_data]
        couleurs = [
            "#2ECC71" if b["niveau"] == "bon"
            else "#F39C12" if b["niveau"] == "moyen"
            else "#E74C3C"
            for b in blocs_data
        ]

        fig = go.Figure(go.Bar(
            x=scores,
            y=noms,
            orientation="h",
            marker_color=couleurs,
            text=[f"{s:.1%}" for s in scores],
            textposition="outside"
        ))
        fig.update_layout(
            title="Scores par bloc",
            xaxis=dict(range=[0, 1.1], tickformat=".0%"),
            height=400,
            margin=dict(l=10, r=60, t=50, b=40)
        )
        # Lignes de référence des seuils
        fig.add_vline(x=SEUIL_BON, line_dash="dash", line_color="green",
                      annotation_text="Bon", annotation_position="top right")
        fig.add_vline(x=SEUIL_MOYEN, line_dash="dash", line_color="orange",
                      annotation_text="Moyen", annotation_position="top right")

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly non disponible.")


def _afficher_barres_metiers(top_metiers: list):
    """Affiche un graphique en barres des scores des métiers recommandés."""
    try:
        import plotly.graph_objects as go

        titres = [m["title"] for m in top_metiers]
        scores = [m["score"] for m in top_metiers]
        couleurs = ["#4A90D9", "#5BA85A", "#9B59B6"][:len(top_metiers)]

        fig = go.Figure(go.Bar(
            x=titres,
            y=scores,
            marker_color=couleurs,
            text=[f"{s:.1%}" for s in scores],
            textposition="outside"
        ))
        fig.update_layout(
            title="Top 3 métiers recommandés",
            yaxis=dict(range=[0, 1.1], tickformat=".0%"),
            height=350,
            margin=dict(l=40, r=40, t=50, b=100)
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly non disponible.")
        for m in top_metiers:
            st.text(f"{m['title']}: {m['score']:.1%}")


def _afficher_heatmap_competences(scores_comp: dict):
    """Affiche une carte de chaleur des scores de toutes les compétences."""
    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame([
            {"Compétence": cid, "Score": score}
            for cid, score in sorted(scores_comp.items(), key=lambda x: -x[1])
        ])

        fig = px.bar(
            df,
            x="Score",
            y="Compétence",
            orientation="h",
            color="Score",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            title="Score de couverture par compétence"
        )
        fig.update_layout(
            height=max(400, len(scores_comp) * 18),
            margin=dict(l=10, r=40, t=50, b=40),
            xaxis=dict(tickformat=".0%", range=[0, 1.05])
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly non disponible.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def afficher_sidebar():
    """Affiche la barre latérale avec informations et navigation."""
    with st.sidebar:
        st.markdown("## AISCA")
        st.markdown("Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences")
        st.markdown("---")
        st.markdown("**Pipeline :**")
        st.markdown("1. Questionnaire hybride")
        st.markdown("2. Prétraitement NLP")
        st.markdown("3. Embeddings SBERT (local)")
        st.markdown("4. Score de couverture pondéré")
        st.markdown("5. Recommandation top-3 métiers")
        st.markdown("6. Plan + Bio (Gemini)")
        st.markdown("---")
        st.markdown("**Modèle NLP :** `all-MiniLM-L6-v2`")
        st.markdown("**GenAI :** Gemini 2.5 Flash (free-tier)")
        st.markdown("---")
        st.caption("EFREI — RNCP40875 — Bloc 2")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main():
    _init_session_state()
    afficher_sidebar()

    # Chargement du référentiel
    try:
        df_comp, _, blocs, metiers = _charger_referentiel()
    except Exception as e:
        st.error(f"Erreur lors du chargement du référentiel : {e}")
        st.stop()

    # Routage par étape
    etape = st.session_state["etape"]

    if etape == "questionnaire":
        page_questionnaire(blocs, metiers)

    elif etape == "analyse":
        page_analyse(df_comp, blocs, metiers)

    elif etape == "resultats":
        if st.session_state.get("resultats") is None:
            st.session_state["etape"] = "questionnaire"
            st.rerun()
        page_resultats(blocs)


if __name__ == "__main__":
    main()
