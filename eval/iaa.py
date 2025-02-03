import pandas as pd
import numpy as np
import os

df = pd.read_excel("..\ELO Ratings\hindustani_2.xlsx", sheet_name="iaa")

def parse_annotation(annotation):
    """Extracts the main comparison symbol from the annotation string."""
    if pd.isna(annotation) or "None" in annotation:
        return None
    elif "A >> B" in annotation:
        return ">>"
    elif "B << A" in annotation:
        return ">>"
    elif "B >> A" in annotation:
        return "<<"
    elif "A << B" in annotation:
        return "<<"
    elif "A > B" in annotation:
        return ">"
    elif "B < A" in annotation:
        return ">"
    elif "A < B" in annotation:
        return "<"
    elif "B > A" in annotation:
        return "<"
    elif "=" in annotation:
        return "="
    else:
        return None

def compare_annotations_distance(column1, column2):
    """
    Compares two columns of parsed annotations and returns a score based on the following logic:
    - ">>" and "<" (or vice versa): output 3
    - "<" and ">" (or vice versa): output 2
    - "<" and "=" (or vice versa): output 1
    - ">>" and ">" (or vice versa): output 0
    - Match (both annotations are the same, e.g., both ">", both "="): output 0
    - None or NA: output -1
    """
    def compare_values(val1, val2):
        parsed_val1 = parse_annotation(val1)
        parsed_val2 = parse_annotation(val2)
        
        # Handle None or missing values
        if parsed_val1 is None or parsed_val2 is None:
            # print(parsed_val1, parsed_val2)
            return -1
        # Return 0 if both values are the same (e.g., both ">", both "=")
        elif parsed_val1 == parsed_val2:
            return 0
        # Check specific cases using unordered sets for symmetry
        elif {parsed_val1, parsed_val2} == {">>", "<<"} or {parsed_val1, parsed_val2} == {"<<", ">>"}:
            return 3
        elif {parsed_val1, parsed_val2} == {">>", "<"} or {parsed_val1, parsed_val2} == {"<<", ">"}:
            return 3
        elif {parsed_val1, parsed_val2} == {">", "<"}:
            return 2
        elif {parsed_val1, parsed_val2} == {"<", "="} or {parsed_val1, parsed_val2} == {">", "="}:
            return 1
        elif {parsed_val1, parsed_val2} == {"<<", "="} or {parsed_val1, parsed_val2} == {">>", "="}:
            return 2
        elif {parsed_val1, parsed_val2} == {">>", ">"} or {parsed_val1, parsed_val2} == {"<<", "<"}:
            return 0
        else:
            # print(parsed_val1, parsed_val2)
            return -1
        
    return compare_values(column1,column2)

def compare_annotations_direction(column1, column2):
    """
    Compares two columns of parsed annotations and returns a score based on the following logic:
    - ">>" and "<" (or vice versa): output 3
    - "<" and ">" (or vice versa): output 2
    - "<" and "=" (or vice versa): output 1
    - ">>" and ">" (or vice versa): output 0
    - Match (both annotations are the same, e.g., both ">", both "="): output 0
    - None or NA: output -1
    """
    def compare_values(val1, val2):
        parsed_val1 = parse_annotation(val1)
        parsed_val2 = parse_annotation(val2)
        
        # Handle None or missing values
        if parsed_val1 is None or parsed_val2 is None:
            # print(parsed_val1, parsed_val2)
            return -1
        # Return 0 if both values are the same (e.g., both ">", both "=")
        elif parsed_val1 == parsed_val2:
            return 0
        # Check specific cases using unordered sets for symmetry
        elif {parsed_val1, parsed_val2} == {">>", "<<"} or {parsed_val1, parsed_val2} == {"<<", ">>"}:
            return 1
        elif {parsed_val1, parsed_val2} == {">>", "<"} or {parsed_val1, parsed_val2} == {"<<", ">"}:
            return 1
        elif {parsed_val1, parsed_val2} == {">", "<"}:
            return 1
        elif {parsed_val1, parsed_val2} == {"<", "="} or {parsed_val1, parsed_val2} == {">", "="}:
            return 0
        elif {parsed_val1, parsed_val2} == {"<<", "="} or {parsed_val1, parsed_val2} == {">>", "="}:
            return 0
        elif {parsed_val1, parsed_val2} == {">>", ">"} or {parsed_val1, parsed_val2} == {"<<", "<"}:
            return 0
        else:
            # print(parsed_val1, parsed_val2)
            return -1
        
    return compare_values(column1,column2)

def get_iaa_distance_score(result1, result2):
    # if player_a.find("MusicGenBaseline") != -1 or player_b.find("MusicGenBaseline") != -1 : 
    #     print("MSB")
    
    # if player_a.find("MusicGenFinetuned") != -1 or player_b.find("MusicGenFinetuned") != -1 : 
    #     print("MSF")
    
    # if player_a.find("MustangoBaseline") != -1 or player_b.find("MustangoBaseline") != -1 : 
    #     print("MTB")
    
    # if player_a.find("MustangoFinetuned") != -1 or player_b.find("MustangoFinetuned") != -1 : 
    #     print("MTF")
        
    
    # Parse result and assign player names accordingly
    return compare_annotations_distance(result1, result2)


def get_iaa_direction_score(result1, result2) : 
    return compare_annotations_direction(result1, result2)
    
    

# Preprocess The audioA and audioB columns
# Define replacements for audioA and audioB based on conditions
def get_kappa_distance(actual,random) : 
    return (actual-random)/(1-random)

replace_dict = {
    "agreement_1" : "overall",
    "agreement_2" : "instruments",
    "agreement_3" : "melody",
    "agreement_4" : "rhythm",
    "agreement_5" : "creativity"
}

# df= df[df["prompt_type"] == "creativity"]

df.loc[df["audioA"].str.contains("musicgen") & df["audioA"].str.contains("baseline"), "audioA"] = "MusicGenBaseline"
df.loc[df["audioA"].str.contains("musicgen") & df["audioA"].str.contains("fine"), "audioA"] = "MusicGenFinetuned"
df.loc[df["audioA"].str.contains("mustango") & df["audioA"].str.contains("baseline"), "audioA"] = "MustangoBaseline"
df.loc[df["audioA"].str.contains("mustango") & df["audioA"].str.contains("fine"), "audioA"] = "MustangoFinetuned"

df.loc[df["audioB"].str.contains("musicgen") & df["audioB"].str.contains("baseline"), "audioB"] = "MusicGenBaseline"
df.loc[df["audioB"].str.contains("musicgen") & df["audioB"].str.contains("fine"), "audioB"] = "MusicGenFinetuned"
df.loc[df["audioB"].str.contains("mustango") & df["audioB"].str.contains("baseline"), "audioB"] = "MustangoBaseline"
df.loc[df["audioB"].str.contains("mustango") & df["audioB"].str.contains("fine"), "audioB"] = "MustangoFinetuned"


# Iterate over each agreement column to update ELO ratings with refined logic
for column in ['agreement_1', 'agreement_2', 'agreement_3', 'agreement_4', 'agreement_5']:
    isNA_count = 0
    total_count = 0
    total_agreement = 0
    for _, row in df.iterrows():
        result1 = row[column+"_x"]
        result2 = row[column+"_y"]
        dist = get_iaa_distance_score(result1, result2)
        
        if dist == -1 : 
            isNA_count+=1
        else : 
            total_agreement += (3-dist)/3
    
        total_count += 1
    agree_percent = total_agreement/(total_count - isNA_count)
    kappa_score = get_kappa_distance(agree_percent, 0.55)
    print("For column {} absolute percent {}".format(replace_dict[column], agree_percent))
    print("For column {} iaa ratings {}".format(replace_dict[column], kappa_score))