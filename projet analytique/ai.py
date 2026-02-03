from ollama import chat
from ollama import ChatResponse
import pandas as pd
import numpy as np
from semantic_engine import analyze_profile

CSV_FILE = "results.csv"


def generate_profile_summary(profile_label: str) -> str:
    
    # Charger les données
    df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
    df_valid = df.dropna(subset=["Name"])
    df_valid = df_valid[df_valid["Name"].astype(str).str.strip() != ""]
    df_valid["label"] = df_valid["Name"].astype(str) + " " + df_valid["Family Name"].astype(str)

    # Trouver le profil
    df_unique = df_valid.drop_duplicates(subset=["label"], keep="last")
    row = df_unique[df_unique["label"] == profile_label].iloc[0].to_dict()

    # Analyser le profil
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
    avg_score = np.nanmean(all_coverage_scores) if all_coverage_scores else 0

    # Comparaison avec la moyenne
    diff = results['coverage_score'] - avg_score
    if diff > 0.05:
        position_vs_moyenne = f"au-dessus de la moyenne (+{diff:.2f})"
    elif diff < -0.05:
        position_vs_moyenne = f"en-dessous de la moyenne ({diff:.2f})"
    else:
        position_vs_moyenne = "dans la moyenne"

    # Construire le contexte pour le LLM
    block_scores_text = "\n".join([f"- {block}: {score:.2f}/1.00" for block, score in results["block_scores"].items()])
    top_jobs_text = "\n".join([f"- {job} (score: {score:.2f})" for job, score in results["top_jobs"]])
    weak_blocks_text = "\n".join([f"- {block}: {score:.2f}" for block, score in results["weak_blocks"]]) if results["weak_blocks"] else "Aucun"

    prompt = f"""Tu es un expert en rédaction de profils professionnels.
À partir des données d'analyse suivantes, génère une Synthèse de Profil professionnelle.
La Synthèse de Profil  doit être percutante, valorisante et mettre en avant les points forts du candidat.

## Informations du candidat
- Nom: {row.get('Name', '')} {row.get('Family Name', '')}
- Années d'expérience Python: {row.get('Years Python', 'N/A')}
- Cas d'usage Python: {row.get('Use Case Python', 'N/A')}

## Scores par bloc de compétences (sur 1.00)
{block_scores_text}

## Score de couverture global: {results['coverage_score']:.2f}/1.00
## Moyenne de tous les candidats: {avg_score:.2f}/1.00
## Position par rapport à la moyenne: {position_vs_moyenne}

## Top 3 des métiers recommandés
{top_jobs_text}

## Compétences à développer
{weak_blocks_text}

Génère maintenant une Synthèse de Profil  professionnelle. Commente la moyonne du candidat par rapport au groupe"""

    response: ChatResponse = chat(model='gemma3', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response.message.content


if __name__ == "__main__":
    # Exemple d'utilisation
    df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
    df_valid = df.dropna(subset=["Name"])
    df_valid = df_valid[df_valid["Name"].astype(str).str.strip() != ""]
    df_valid["label"] = df_valid["Name"].astype(str) + " " + df_valid["Family Name"].astype(str)

    print("Profils disponibles:")
    for label in df_valid["label"].unique():
        print(f"  - {label}")

    # Prendre le dernier profil pour la démo
    last_profile = df_valid["label"].iloc[-1]
    print(f"\nGénération de la Synthèse de Profil  pour: {last_profile}")
    print("-" * 50)

    bio = generate_profile_summary(last_profile)
    print(bio)