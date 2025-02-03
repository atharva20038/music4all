import json
import random

# Load the original JSON file
with open('data/metadata_test_hindustani.json', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# Randomly sample 400 lines from the data
sampled_data = random.sample(data, min(400, len(data)))

# Process each sampled item to extract "main_caption" and "location"
processed_data = []
for item in sampled_data:
    processed_item = {
        "captions": item.get("main_caption"),
        "location": item.get("location").split("/")[-1]
    }
    processed_data.append(processed_item)

# Save the processed sampled data to a new JSON file
with open('processed_sampled_data.json', 'w', encoding='utf-8') as output_file:
    for item in processed_data:
        json.dump(item, output_file, ensure_ascii=False)
        output_file.write('\n')  # Write each JSON object on a new line

print("Random sampling and data extraction complete. Sampled data saved to 'processed_sampled_data.json'.")
