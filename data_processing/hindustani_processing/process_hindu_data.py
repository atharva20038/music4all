import os 
import pandas as pd
import json
import pydub
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

metadata = pd.read_csv("../hindustani.csv")

def process_audio_chunk(row):
    try:
        if len(row) > 1:
            print("More than one audio file")
        
        if len(row) == 0:
            return []

        chunks = []

        for file in os.listdir(row["audio_path"]):
            audio_path = os.path.join(row["audio_path"], file)
            audio = pydub.AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(32000)

            length = len(audio)

            start = int(length * 20 / 100)
            end = int(length * 90 / 100)

            for i in range(start, end, 30 * 1000):
                title = audio_path.split("/")[-1].split(".")[0]
                chunk_path = f"chunks/{title}_{row['mbid']}_{i}.wav"

                if os.path.exists(chunk_path):
                    chunks.append([row["mbid"], row["raga"], row["taal"], row["laya"], row["instrument"], chunk_path])
                    continue

                if len(audio) < (i + 30000):
                    continue

                chunk = audio[i:i + 30 * 1000]
                chunk.export(chunk_path, format="wav")

                print(f"Creating chunk for {chunk_path}")
                chunks.append([row["mbid"], row["raga"], row["taal"], row["laya"], row["instrument"], chunk_path])

        return chunks

    except Exception as e:
        print(f"Exception occurred: {e}")
        return []

def get_hindustani_data():
    if not os.path.exists("chunks"):
        os.mkdir("chunks")

    new_df = pd.DataFrame(columns=["mbid", "raga", "taal", "laya", "instrument", "audio_path"])

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_audio_chunk, row) for _, row in metadata.iterrows()]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                new_df = pd.concat([new_df, pd.DataFrame(result, columns=["mbid", "raga", "taal", "laya", "instrument", "audio_path"])])

    new_df.to_csv("new_metadata_hindustani.csv", index=False)

get_hindustani_data()
