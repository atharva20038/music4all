import pandas as pd
import numpy as np
import os
import argparse

def parse_annotation(annotation):
    """Extracts the main comparison symbol from the annotation string."""
    if pd.isna(annotation) or "None" in annotation:
        return None
    annotation_map = {
        "A >> B": ">>", "B << A": ">>",
        "B >> A": "<<", "A << B": "<<",
        "A > B": ">", "B < A": ">",
        "A < B": "<", "B > A": "<",
        "=": "="
    }
    for key, value in annotation_map.items():
        if key in annotation:
            return value
    return None

def compare_annotations_distance(val1, val2):
    """Compares two annotations and returns a distance score based on agreement."""
    parsed_val1, parsed_val2 = parse_annotation(val1), parse_annotation(val2)

    if parsed_val1 is None or parsed_val2 is None:
        return -1
    if parsed_val1 == parsed_val2:
        return 0
    scoring_rules = [
        ({">>", "<<"}, 3), ({">>", "<"}, 3), ({">>", "="}, 2),
        ({">", "<"}, 2), ({"<", "="}, 1), ({">", "="}, 1),
        ({">>", ">"}, 0), ({"<<", "<"}, 0)
    ]
    for conditions, score in scoring_rules:
        if {parsed_val1, parsed_val2} == conditions:
            return score
    return -1

def compare_annotations_direction(val1, val2):
    """Compares two annotations and returns a direction score."""
    parsed_val1, parsed_val2 = parse_annotation(val1), parse_annotation(val2)

    if parsed_val1 is None or parsed_val2 is None:
        return -1
    if parsed_val1 == parsed_val2:
        return 0
    scoring_rules = [
        ({">>", "<<"}, 1), ({">>", "<"}, 1), ({">", "<"}, 1),
        ({"<", "="}, 0), ({">", "="}, 0), ({">>", "="}, 0),
        ({">>", ">"}, 0), ({"<<", "<"}, 0)
    ]
    for conditions, score in scoring_rules:
        if {parsed_val1, parsed_val2} == conditions:
            return score
    return -1

def get_kappa_distance(actual, random_baseline):
    """Computes kappa distance based on agreement scores."""
    return (actual - random_baseline) / (1 - random_baseline)

def preprocess_audio_columns(df):
    """Standardizes naming conventions for audio models."""
    model_map = {
        "musicgen.*baseline": "MusicGenBaseline",
        "musicgen.*fine": "MusicGenFinetuned",
        "mustango.*baseline": "MustangoBaseline",
        "mustango.*fine": "MustangoFinetuned"
    }
    for col in ["audioA", "audioB"]:
        for pattern, replacement in model_map.items():
            df.loc[df[col].str.contains(pattern, regex=True, na=False), col] = replacement
    return df

def calculate_iaa_scores(df, agreement_columns, random_baseline=0.55):
    """Computes inter-annotator agreement scores for given columns."""
    rename_dict = {
        "agreement_1": "overall",
        "agreement_2": "instruments",
        "agreement_3": "melody",
        "agreement_4": "rhythm",
        "agreement_5": "creativity"
    }

    for column in agreement_columns:
        total_count, isNA_count, total_agreement = 0, 0, 0

        for _, row in df.iterrows():
            dist = compare_annotations_distance(row[column+"_x"], row[column+"_y"])

            if dist == -1:
                isNA_count += 1
            else:
                total_agreement += (3 - dist) / 3

            total_count += 1

        agree_percent = total_agreement / (total_count - isNA_count)
        kappa_score = get_kappa_distance(agree_percent, random_baseline)

        print(f"For column **{rename_dict[column]}**:")
        print(f"   - Absolute Agreement: {agree_percent:.3f}")
        print(f"   - IAA Kappa Score: {kappa_score:.3f}\n")

def main(args):
    """Main function to load data and compute IAA scores."""
    df = pd.read_excel(args.input_file, sheet_name=args.sheet_name)
    df = preprocess_audio_columns(df)

    agreement_columns = ['agreement_1', 'agreement_2', 'agreement_3', 'agreement_4', 'agreement_5']
    calculate_iaa_scores(df, agreement_columns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IAA Scores for Music Annotations")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the Excel file containing annotations")
    parser.add_argument("--sheet_name", type=str, default="iaa", help="Sheet name to process (default: iaa)")

    args = parser.parse_args()
    main(args)
