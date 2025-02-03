import numpy as np
import pandas as pd
import argparse

# Initialize ELO ratings dictionary with a starting value
INITIAL_ELO = 1500
K_FACTOR = 15  # ELO rating adjustment factor

def expected_score(ra, rb):
    """Calculates the expected score between two players."""
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ratings, player_a, player_b, outcome):
    """Updates ELO ratings based on match outcome."""
    ea, eb = expected_score(ratings[player_a], ratings[player_b]), 1 - expected_score(ratings[player_a], ratings[player_b])

    if outcome == "win":
        ratings[player_a] += K_FACTOR * (1 - ea)
        ratings[player_b] += K_FACTOR * (0 - eb)
    elif outcome == "draw":
        ratings[player_a] += K_FACTOR * (0.5 - ea)
        ratings[player_b] += K_FACTOR * (0.5 - eb)

def refined_update_elo(ratings, player_a, player_b, result):
    """Determines match result from annotations and updates ELO."""
    if "A > B" in result or "A >> B" in result:
        update_elo(ratings, player_a, player_b, "win")
    elif "B > A" in result or "B >> A" in result:
        update_elo(ratings, player_b, player_a, "win")
    elif "None" in result or "A = B" in result:
        update_elo(ratings, player_a, player_b, "draw")

def preprocess_audio_labels(df):
    """Standardizes naming conventions for audio models."""
    model_mapping = {
        "musicgen.*baseline": "MusicGenBaseline",
        "musicgen.*fine": "MusicGenFinetuned",
        "mustango.*baseline": "MustangoBaseline",
        "mustango.*fine": "MustangoFinetuned"
    }
    for col in ["audioA", "audioB"]:
        for pattern, replacement in model_mapping.items():
            df.loc[df[col].str.contains(pattern, regex=True, na=False), col] = replacement
    return df

def compute_elo_ratings(df):
    """Computes ELO ratings for each agreement column."""
    elo_dict = {model: INITIAL_ELO for model in ["MusicGenBaseline", "MusicGenFinetuned", "MustangoBaseline", "MustangoFinetuned"]}
    agreement_columns = ['agreement_1', 'agreement_2', 'agreement_3', 'agreement_4', 'agreement_5']
    column_mapping = {"agreement_1": "overall", "agreement_2": "instruments", "agreement_3": "melody", "agreement_4": "rhythm", "agreement_5": "creativity"}

    for column in agreement_columns:
        for _, row in df.iterrows():
            refined_update_elo(elo_dict, row["audioA"], row["audioB"], row[column])
        print(f"For category **{column_mapping[column]}**, updated ELO ratings: {elo_dict}")

def main(args):
    """Main function to load data, preprocess labels, and compute ELO ratings."""
    df = pd.read_excel(args.input_file, sheet_name=args.sheet_name)
    df = df[df["prompt_type"] == "creativity"]
    df = preprocess_audio_labels(df)

    compute_elo_ratings(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ELO ratings for music annotation rankings.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the Excel file containing rankings.")
    parser.add_argument("--sheet_name", type=str, default="Sheet2", help="Sheet name to process (default: Sheet2).")

    args = parser.parse_args()
    main(args)
