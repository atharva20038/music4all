import torch
from safetensors.torch import load_file
import argparse

parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
parser.add_argument(
    "--path", type=str, default="saved/1727169396/epoch_40/model_2.safetensors",
    help="Path for summary jsonl file saved during training."
)

args = parser.parse_args()

# Path to the safetensors model
safetensors_path = args.path

# Load the model from safetensors
state_dict = load_file(safetensors_path)

# Now save the state_dict as a .bin file using torch.save
bin_path = args.path.replace(".safetensors",".bin")
torch.save(state_dict, bin_path)

print(f"Model saved as {bin_path}")
