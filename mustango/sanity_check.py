import os 
import pydub
import pandas as pd
import numpy as np

data_folder = "dataset_finetune/Indian Classical Music_16KHz"

for file in os.listdir(data_folder) : 
    audio = pydub.AudioSegment.from_file(file = os.path.join(data_folder, file), format = "wav")
    # print(len(audio))
    if len(audio) < 30000 : 
        print("Insane data")
        
    if audio.max == 0 : 
        print("Insane data")
        
    # To know about channels of file 
    # print(audio.channels)  
    # OUTPUT: 1 
    
    # Find the number of bytes per sample  
    # print(audio.sample_width )  
    # OUTPUT : 2 
    
    
    # Find Maximum amplitude  
    # print(audio.max) 
    # OUTPUT 17106 