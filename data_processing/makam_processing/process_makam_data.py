import os 
import pandas as pd
import json
import pydub
from tqdm import tqdm

metadata = pd.read_csv("../makam.csv")

def get_makam_data():
    if not os.path.exists("chunks") : 
        os.mkdir("chunks")

    audio_paths = []

    new_df = pd.DataFrame(columns = ["mbid","makam","usul","instrument","audio_path"])

    for index, row in tqdm(metadata.iterrows()) : 
        if len(row) > 1 : 
            print("More than one audio file")
        
        if len(row) == 0 : 
            continue
        
        for file in os.listdir(row["audio_path"]):
            audio_path = os.path.join(row["audio_path"],file)
            audio = pydub.AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(32000)

            ## Length of the audio
            # length = int(float(pydub.utils.mediainfo(audio_path)["duration"])*1000)
            length = len(audio)

            # print(length)
            # print(len(audio))

            ## Change the sample rate to 32000
            # audio = audio.set_frame_rate(32000)
            
            ## Divide the audio into 30 second chunk
            start = int(length*20/100)
            end = int(length*90/100)

            for i in range(start, end, 30*1000) : 
                title = audio_path.split("/")[-1].split(".")[0]
                chunk_path = f"chunks/{title}_{row['mbid']}_{i}.wav"

                if os.path.exists(chunk_path) : 
                    # print("Already exists")
                    new_df = pd.concat([new_df, pd.DataFrame([[row["mbid"], row["makam"], row["usul"], row["instrument"] , chunk_path]], columns = ["mbid","makam","usul","instrument","audio_path"])])
                    continue

                if len(audio) < (i + 30000) : 
                    continue

                chunk = audio[i:i+30*1000]
                chunk.export(chunk_path, format="wav")

                print(f"Creating chunk for {chunk_path}")

                audio_paths.append(chunk_path)
                new_df = pd.concat([new_df, pd.DataFrame([[row["mbid"], row["makam"], row["usul"], row["instrument"] , chunk_path]], columns = ["mbid","makam","usul","instrument","audio_path"])])


    new_df.to_csv("new_metadata_makam.csv", index = False)

get_makam_data()