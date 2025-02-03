import os
import subprocess
import argparse

def automate_safetensor_to_bin(directory, baseline=False):
    # Get all filenames from the directory
    files = os.listdir(directory)
    print(files)
    
    # Filter to get .safetensors and .bin files
    safetensor_files = [f for f in files if f.endswith('.safetensors')]
    bin_files = [f for f in files if f.endswith('.bin')]
    
    # Sort files to get the latest ones (optional)
    safetensor_files.sort()
    bin_files.sort()
    
    # Check if there are files available
    if len(safetensor_files) == 0 and len(bin_files) == 0:
        print("No .safetensors or .bin files found.")
        return


    # Paths for other files (adjust these as needed)
    test_files = ["processed_sampled_data.json"]
    original_args = "saved/1734429910/summary.jsonl"
    test_references = "test_reference_data_hindustani_new_version"
    for test_file in test_files : 
        if baseline : 
            print(test_file)
            bin_file_path = os.path.join(directory, bin_files[2])
            inference_cmd = (f"python inference.py --model \"{bin_file_path}\" "
                        f"--test_file \"{test_file}\" --original_args \"{original_args}\" "
                        f"--guidance 3 --test_references \"{test_references}\" --baseline True")

            subprocess.run(inference_cmd, shell=True)
            
            continue
 

        for safetensor in safetensor_files : 
            if safetensor.find("model") == -1 and len(safetensor) > 0: 
                print("No model found")
                continue

            safetensor_path = os.path.join(directory, safetensor)  # Latest .safetensors file
            bin_file_path = os.path.join(directory, safetensor.replace(".safetensors",".bin"))  # Latest .bin file

            if len(safetensor) > 0 :
                # Command 1: safetensor to bin
                safetensor_cmd = f"python safetensor_to_bin.py --path \"{safetensor_path}\""
                subprocess.run(safetensor_cmd, shell=True)
            
        files = os.listdir(directory)
        print(files)
        
        # Filter to get .safetensors and .bin files
        safetensor_files = [f for f in files if f.endswith('.safetensors')]
        bin_files = [f for f in files if f.endswith('.bin')]
        
        # Sort files to get the latest ones (optional)
        safetensor_files.sort()
        bin_files.sort()

        for bin in bin_files : 
            bin_file_path = os.path.join(directory, bin)
            if bin_file_path.find("model_2") == -1 : 
                continue

            # Command 2: inference
            if not baseline :
                print("Calling for inference") 
                inference_cmd = (f"python inference.py --model \"{bin_file_path}\" "
                            f"--test_file \"{test_file}\" --original_args \"{original_args}\" "
                            f"--guidance 3 --test_references \"{test_references}\"")
                print(inference_cmd)
        
            subprocess.run(inference_cmd, shell=True)

if __name__ == "__main__":
    # Replace 'saved/1727867905/epoch_40/' with your actual directory
    directory = "saved/1734429910"
    parser = argparse.ArgumentParser()

    parser.add_argument(
		"--baseline", type=bool, default=False,
		help="Baseline inference or not"
	)
    sub_dirs = os.listdir(directory)
    # for epoch_dir in sub_dirs : 
    #     automate_safetensor_to_bin(os.path.join(directory,epoch_dir))

    args = parser.parse_args()

    automate_safetensor_to_bin("saved/1734429910/best", args.baseline)
