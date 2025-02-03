import json
import pandas as pd
import numpy as np
import os

metadata = pd.read_csv("../Training Dataset/Hindustani Classical/test_Curated_Hindustani_Prompts.csv")

music_bench = pd.read_json("data/MusicBench_train.json", lines=True)

music_bench.info()

## The metadata file should have the same structure as the MusicBench file
new_meta = pd.DataFrame(columns=music_bench.columns)

new_meta["location"] = metadata["audio_path"].apply(lambda x: os.path.join("../Training Dataset","Hindustani Classical",x))
new_meta["main_caption"] = metadata["Final Prompt"]
new_meta["prompt_aug"] = metadata["Final Prompt"]
new_meta["alt_caption"] = metadata["Final Prompt"]

print(music_bench["beats"].head())

new_meta["dataset"] = "Hindustani"

new_meta["prompt_ch"] = ""
new_meta["prompt_bt"] = ""
new_meta["prompt_bpm"] = ""
new_meta["prompt_key"] = ""
new_meta["beats"] = [[[0.0],[0.0]]]*len(metadata)
new_meta["bpm"] = [0.0]*len(metadata)
new_meta["key"] = [[""]]*len(metadata)
new_meta["chords"] = [[""]]*len(metadata)
new_meta["chords_time"] = [[0.0]]*len(metadata)
new_meta["keyprob"] = [[0.0]]*len(metadata)
new_meta["is_audioset_eval_mcaps"] = False


new_meta.to_json("metadata_test_hindustani.json", orient="records", lines=True)