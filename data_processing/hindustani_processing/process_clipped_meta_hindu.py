import pandas as pd
import random

# Load the CSV file
file_path = 'new_metadata_hindustani.csv'
df = pd.read_csv(file_path)

# Function to generate custom prompts while skipping NaN values and avoiding duplicates
def generate_custom_prompts(instrument, raaga, taal, laaya):
    # Function to split comma-separated values and get unique entries
    def get_unique_values(field):
        if pd.notnull(field):
            return list(set([item.strip() for item in str(field).split(',')]))
        return []

    # Get unique entities for each field
    instruments = get_unique_values(instrument)
    ragas = get_unique_values(raaga)
    taals = get_unique_values(taal)
    laayas = get_unique_values(laaya)

    # Create entity descriptions without duplicates
    entities = []
    if instruments:
        entities.append(f"{', '.join(instruments)} instruments")
    if ragas:
        entities.append(f"{', '.join(ragas)} raga")
    if taals:
        entities.append(f"{', '.join(taals)} taal")
    if laayas:
        entities.append(f"{', '.join(laayas)} laaya")
    
    # Generating the prompts dynamically based on available unique information
    prompts = [
        f"Compose a soulful Hindustani Classical piece that blends the rich timbre of {', '.join(entities)}.",
        f"Design a captivating Hindustani Classical composition where the soothing tones of {', '.join(entities)} explore new depths.",
        f"Imagine a traditional Hindustani Classical performance that brings together {', '.join(entities)}, flowing effortlessly.",
        f"Craft a mesmerizing Hindustani Classical melody that intertwines the essence of {', '.join(entities)}.",
        f"Envision a timeless Hindustani Classical recital where the harmonious blend of {', '.join(entities)}.",
    ]
    
    return prompts

# Apply the function to each row in the dataframe and select one random prompt
df['Final Prompt'] = df.apply(
    lambda row: random.choice(generate_custom_prompts(row['instrument'], row['raga'], row['taal'], row['laya'])),
    axis=1
)

# Shuffle the dataframe before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Perform 70-10-20 train-validation-test split based on 'mbid' (assuming it's a unique identifier)
unique_mbids = df['mbid'].unique()
total_mbids = len(unique_mbids)

train_split = int(0.7 * total_mbids)
val_split = int(0.1 * total_mbids)

train_mbids = unique_mbids[:train_split]
val_mbids = unique_mbids[train_split:train_split + val_split]
test_mbids = unique_mbids[train_split + val_split:]

train_df = df[df['mbid'].isin(train_mbids)]
val_df = df[df['mbid'].isin(val_mbids)]
test_df = df[df['mbid'].isin(test_mbids)]

# Save the splits to new CSV files
train_output_path = 'train_Curated_Hindustani_Prompts.csv'
val_output_path = 'val_Curated_Hindustani_Prompts.csv'
test_output_path = 'test_Curated_Hindustani_Prompts.csv'

train_df.to_csv(train_output_path, index=False)
val_df.to_csv(val_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

train_output_path, val_output_path, test_output_path

