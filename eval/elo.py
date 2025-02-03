import numpy as np
import pandas as pd

df = pd.read_excel("..\IAA\makam.xlsx", sheet_name="Sheet2")

# Initialize ELO ratings dictionary with a starting value of 1500 for all unique entries in agreement columns
initial_elo = 1500

# Set constant K-factor for ELO calculations
K = 15

# Function to calculate the expected score
def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

# Function to update ELO ratings based on match result
def update_elo(ratings, player_a, player_b, outcome):
    # Determine the expected scores
    ea = expected_score(ratings[player_a], ratings[player_b])
    eb = 1 - ea
    
    # Update ratings based on outcome
    if outcome == "win":
        ratings[player_a] += K * (1 - ea)  # Win for player A
        ratings[player_b] += K * (0 - eb)  # Loss for player B
        
    elif outcome == "draw":
        ratings[player_a] += K * (0.5 - ea)  # Draw
        ratings[player_b] += K * (0.5 - eb)  # Draw

# Adjusting the function to recognize the match result labels more accurately

# Refined function to update ELO ratings based on match result from each comparison result label
def refined_update_elo(ratings, player_a, player_b, result):
    # if player_a.find("MusicGenBaseline") != -1 or player_b.find("MusicGenBaseline") != -1 : 
    #     print("MSB")
    
    # if player_a.find("MusicGenFinetuned") != -1 or player_b.find("MusicGenFinetuned") != -1 : 
    #     print("MSF")
    
    # if player_a.find("MustangoBaseline") != -1 or player_b.find("MustangoBaseline") != -1 : 
    #     print("MTB")
    
    # if player_a.find("MustangoFinetuned") != -1 or player_b.find("MustangoFinetuned") != -1 : 
    #     print("MTF")
        
    
    # Parse result and assign player names accordingly
    if "A > B" in result or "A >> B" in result:
        update_elo(ratings, player_a, player_b, "win")
    elif "B > A" in result or "B >> A" in result:
        update_elo(ratings, player_b, player_a , "win")
    elif "None" in result or "A = B" in result:
        update_elo(ratings, player_a, player_b, "draw")

# Preprocess The audioA and audioB columns
# Define replacements for audioA and audioB based on conditions

elo_dict = {
    "MusicGenBaseline" : initial_elo,
    "MusicGenFinetuned" : initial_elo,
    "MustangoBaseline" : initial_elo,
    "MustangoFinetuned" : initial_elo
}

replace_dict = {
    "agreement_1" : "overall",
    "agreement_2" : "instruments",
    "agreement_3" : "melody",
    "agreement_4" : "rhythm",
    "agreement_5" : "creativity"
}

df= df[df["prompt_type"] == "creativity"]

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
    for _, row in df.iterrows():
        result = row[column]
        refined_update_elo(elo_dict, row["audioA"], row["audioB"], result)
        
    print("For column {} elo ratings {}".format(replace_dict[column], elo_dict))

# Display the resulting ELO ratings for each agreement type
