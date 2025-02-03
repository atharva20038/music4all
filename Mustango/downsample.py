import os 
import numpy as np
import pydub
from tqdm import tqdm

def downsample_audio(audio_path, output_path, target_sr):
    audio = pydub.AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_sr)
    audio.export(output_path, format="wav")


def downsample_audio_folder(audio_folder, output_folder, target_sr):
    os.makedirs(output_folder, exist_ok=True)
    for audio_file in tqdm(os.listdir(audio_folder)):
        try: 
            audio_path = os.path.join(audio_folder, audio_file)
            output_path = os.path.join(output_folder, audio_file)

            if os.path.exists(output_path):
                continue
            downsample_audio(audio_path, output_path, target_sr)

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")    


downsample_audio_folder("dataset_finetune/Makam_32", "dataset_finetune/Makam_16", 16000)