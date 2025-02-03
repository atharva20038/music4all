import os
import pandas as pd
import numpy as np
import pydub
import shutil


df = pd.read_json("processed_sampled_data.json", lines=True)

if not os.path.exists("test_reference_data_hindustani_new_version") : 
    os.makedirs("test_reference_data_hindustani_new_version")

for iter,row in df.iterrows() : 
    try : 
        src_dir = os.path.join("../Training Dataset/Hindustani Classical/chunks",row["location"])
        tar_dir = "test_reference_data_hindustani_new_version/" + src_dir.split("/")[-1]
        print(src_dir)
        print(tar_dir)

        shutil.copyfile(src_dir, tar_dir)
        
    except Exception as e : 
        print(e)



