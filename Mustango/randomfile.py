import os 
import pydub
import shutil

output_list = os.listdir("outputs/out_1728665577_1727867905_best_steps_200_guidance_3.0")

os.makedirs("test_reference_data_makam_kld_32", exist_ok=True)

for files in output_list : 
    src_dir = os.path.join("dataset_finetune/Makam_32KHz",files)
    tar_dir = os.path.join("test_reference_data_makam_kld_32",files)
    
    shutil.copyfile(src_dir, tar_dir)

# print(len(os.listdir("test_reference_data_hindustani_kld")))
    
