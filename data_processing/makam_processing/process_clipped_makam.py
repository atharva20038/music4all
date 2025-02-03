import pandas as pd
import random

# Load the CSV file
file_path = 'new_metadata_makam.csv'
df = pd.read_csv(file_path)

# Split the 'audio_path' column by "/" and get the last element in each list
df['audio_path'] = df['audio_path'].str.split("/").str[-1]

# Function to generate custom prompts while skipping NaN values and avoiding duplicates
def generate_custom_prompts(instrument, makam, usul):
    # Function to split comma-separated values and get unique entries
    def get_unique_values(field):
        if pd.notnull(field):
            return list(set([item.strip() for item in str(field).split(',')]))
        return []

    # Get unique entities for each field
    instruments = get_unique_values(instrument)
    makams = get_unique_values(makam)
    usuls = get_unique_values(usul)

    # Create entity descriptions without duplicates
    entities = []
    if instruments:
        entities.append(f"{', '.join(instruments)} instruments")
    if makams:
        entities.append(f"{', '.join(makams)} makam")
    if usuls:
        entities.append(f"{', '.join(usuls)} usul")
    
    # Generating the prompts dynamically based on available unique information
    prompts = [
        f"Compose a soulful Turkish Makam piece that blends the rich timbre of {', '.join(entities)}.",
        f"Design a captivating Turkish Makam composition where the soothing tones of {', '.join(entities)} explore new depths.",
        f"Imagine a traditional Turkish Makam performance that brings together {', '.join(entities)}, flowing effortlessly.",
        f"Craft a mesmerizing Turkish Makam melody that intertwines the essence of {', '.join(entities)}.",
        f"Envision a timeless Turkish Makam recital where the harmonious blend of {', '.join(entities)} stands out."
    ]
    
    return prompts

# Apply the function to each row in the dataframe and select one random prompt
df['Final Prompt'] = df.apply(
    lambda row: random.choice(generate_custom_prompts(row['instrument'], row['makam'], row['usul'])),
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
train_output_path = 'train_Curated_Makam_Music_Prompts.csv'
val_output_path = 'val_Curated_Makam_Music_Prompts.csv'
test_output_path = 'test_Curated_Makam_Music_Prompts.csv'

train_df.to_csv(train_output_path, index=False)
val_df.to_csv(val_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

train_output_path, val_output_path, test_output_path
