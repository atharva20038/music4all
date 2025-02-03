import os
import subprocess
import argparse

def run_inference(model_path, test_files, original_args, test_references, baseline=False):
    """Runs inference, converting safetensor to bin if needed."""
    
    # Convert safetensor to bin if the model is in .safetensors format
    if model_path.endswith('.safetensors'):
        print(f"Converting {model_path} to .bin format...")
        conversion_cmd = f"python safetensor_to_bin.py --path \"{model_path}\""
        subprocess.run(conversion_cmd, shell=True)
        
        # Replace .safetensors with .bin for inference
        model_path = model_path.replace('.safetensors', '.bin')

    if not model_path.endswith('.bin'):
        print("Invalid model file format. Please provide a .bin or .safetensors file.")
        return

    for test_file in test_files:
        test_file_path = os.path.abspath(test_file)

        print(f"Running inference for {test_file}")
        inference_cmd = (
            f"python inference.py --model \"{model_path}\" "
            f"--test_file \"{test_file_path}\" --original_args \"{original_args}\" "
            f"--guidance 3 --test_references \"{test_references}\""
        )

        if baseline:
            inference_cmd += " --baseline True"

        subprocess.run(inference_cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a specified model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (.bin or .safetensors).")
    parser.add_argument("--test_files", nargs="+", required=True, help="List of test files for inference.")
    parser.add_argument("--original_args", type=str, required=True, help="Path to the original args file.")
    parser.add_argument("--test_references", type=str, required=True, help="Path to the test reference data.")
    parser.add_argument("--baseline", action="store_true", help="Run baseline inference.")

    args = parser.parse_args()

    run_inference(args.model, args.test_files, args.original_args, args.test_references, args.baseline)
