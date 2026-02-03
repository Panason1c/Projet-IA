import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from semantic_engine import analyze_profile
from ai import generate_profile_summary



CSV_FILE = "results.csv"


with st.form("form"):

    st.write("What is your name ?")
    Name = st.text_input("Name")
    Familly_Name = st.text_input("Familly_Name")
    

    st.write("Question 1")
    slider_val = st.slider("How many years did you use python in a professional environment ?")
    Question1 = st.text_input("And what was it used for ?")


    st.write("Question 2")
    Question2 = st.text_input("Are you at familiar with neural networks ?")

    st.write("Question 3")
    Question3 = st.text_input("On a scale of 0 to 20, how are you good in python by your own judgement ?")


    st.write("Question 4")
    Question4 = st.text_input("What the word “tokenisation”  means to you ?")

    st.write("Question 5")
    Question5 = st.text_input("what the word “regression”  means to you ?")

    st.write("Question 6")
    Question6 = st.text_input("Do you work on neural networks ? If yes describe your work.")

   
    st.write("Question 7")
    Question7 = st.text_input("In your word, describe “NLP” ?")
    # Every form must have a submit button.
    
    st.write("Question 9")
    Question8 = st.text_input("Did you use any tokenisation method ? If yes, describe it.")
   

    submitted = st.form_submit_button("Submit")

if submitted:
    data = {
        "Name": [Name],
        "Family Name": [Familly_Name],
        "Years Python": [slider_val],
        "Use Case Python": [Question1],
        "Neural Network": [Question2],
        "Python Skill": [Question3],
        "Tokenisation": [Question4],
        "Regression": [Question5],
        "Neural Work": [Question6],
        "NLP": [Question7],
        "Tokenisation Method": [Question8],
        "Timestamp": [datetime.now().isoformat()]
    }
    df_new = pd.DataFrame(data)
    if not os.path.exists(CSV_FILE):
        df_new.to_csv(CSV_FILE, index=False)
    else:
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
    st.success("Your answers have been successfully saved!")
    st.write("Saved data:")
    st.dataframe(df_new)


st.header("Analyse AISCA")

# Initialiser session_state
if "results" not in st.session_state:
    st.session_state.results = None
if "selected_profile" not in st.session_state:
    st.session_state.selected_profile = None
if "bio" not in st.session_state:
    st.session_state.bio = None
if "avg_score" not in st.session_state:
    st.session_state.avg_score = None

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
    df_valid = df.dropna(subset=["Name"])
    df_valid = df_valid[df_valid["Name"].astype(str).str.strip() != ""]

    if not df_valid.empty:
        df_valid["label"] = df_valid["Name"].astype(str) + " " + df_valid["Family Name"].astype(str)
        df_unique = df_valid.drop_duplicates(subset=["label"], keep="last")

        selected = st.selectbox("Select a profil", df_unique["label"].tolist())

        if st.button("Analyse"):
            row = df_unique[df_unique["label"] == selected].iloc[0].to_dict()

            with st.spinner("Loading..."):
                results = analyze_profile(row)

                # Calculer la moyenne de tous les candidats
                all_coverage_scores = []
                for _, r in df_unique.iterrows():
                    try:
                        row_data = r.to_dict()
                        profile_results = analyze_profile(row_data)
                        score = profile_results["coverage_score"]
                        if score is not None and not np.isnan(score):
                            all_coverage_scores.append(score)
                    except:
                        pass
                avg_score = np.nanmean(all_coverage_scores) if all_coverage_scores else None

            # Sauvegarder dans session_state
            st.session_state.results = results
            st.session_state.selected_profile = selected
            st.session_state.avg_score = avg_score
            st.session_state.bio = None  # Reset bio

        # Afficher les résultats si disponibles
        if st.session_state.results is not None:
            results = st.session_state.results

            st.subheader("Scores par bloc")
            for block, score in results["block_scores"].items():
                st.write(f"- **{block}** : {score:.2f}")
            # Graphique radar
            categories = list(results["block_scores"].keys())
            values = list(results["block_scores"].values())

            # Fermer le polygone
            values_closed = values + [values[0]]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles_closed = angles + [angles[0]]

            fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax_radar.fill(angles_closed, values_closed, color='#4CAF50', alpha=0.25)
            ax_radar.plot(angles_closed, values_closed, color='#4CAF50', linewidth=2)
            ax_radar.scatter(angles, values, color='#4CAF50', s=50, zorder=5)

            # Labels
            ax_radar.set_xticks(angles)
            ax_radar.set_xticklabels(categories, fontsize=10)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

            plt.tight_layout()
            st.pyplot(fig_radar)

            st.subheader("Heatmap des scores")

            # Créer la heatmap
            fig, ax = plt.subplots(figsize=(8, 2))

            blocks = list(results["block_scores"].keys())
            scores = list(results["block_scores"].values())

            # Matrice 1xN pour la heatmap
            data = np.array([scores])

            im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            # Labels
            ax.set_xticks(np.arange(len(blocks)))
            ax.set_xticklabels(blocks, rotation=45, ha='right')
            ax.set_yticks([])

            # Afficher les valeurs dans les cellules
            for i, score in enumerate(scores):
                color = 'white' if score < 0.5 else 'black'
                ax.text(i, 0, f'{score:.2f}', ha='center', va='center', color=color, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Score de couverture global")
            st.write(f"**{results['coverage_score']:.2f}**")

            if st.session_state.avg_score is not None and not np.isnan(st.session_state.avg_score):
                st.write(f"Moyenne de tous les candidats : **{st.session_state.avg_score:.2f}**")

                # Graphique comparaison
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                labels = ['Profil actuel', 'Moyenne candidats']
                values = [results['coverage_score'], st.session_state.avg_score]
                colors = ['#4CAF50', '#2196F3']
                bars = ax2.bar(labels, values, color=colors, width=0.5)
                ax2.set_ylim(0, 1)
                ax2.set_ylabel('Score')
                ax2.set_title('Comparaison avec la moyenne')
                for bar, val in zip(bars, values):
                    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig2)

            st.subheader("Top 3 metiers recommandes")
            for rank, (job, score) in enumerate(results["top_jobs"], 1):
                st.write(f"{rank}. **{job}** (score : {score:.2f})")

            if results["weak_blocks"]:
                st.subheader("Competences a developper")
                for block, score in results["weak_blocks"]:
                    st.write(f"- **{block}** : {score:.2f}")

            st.divider()

            st.subheader("Synthèse de Profil (IA)")
            if st.button("Générer la biographie", key="btn_bio"):
                with st.spinner("Génération en cours avec Ollama (Gemma3)..."):
                    try:
                        bio = generate_profile_summary(st.session_state.selected_profile)
                        st.session_state.bio = bio
                    except Exception as e:
                        st.error(f"Erreur lors de la génération : {e}")

            if st.session_state.bio:
                st.success("Biographie générée !")
                st.markdown(st.session_state.bio)

else:
    st.info("Remplissez le questionnaire pour lancer une analyse.")
