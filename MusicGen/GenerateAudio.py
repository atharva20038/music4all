import torch
import torch.nn as nn
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os
import matplotlib.pyplot as plt
 
# Paths and configurations
pretrained_model_name = "facebook/musicgen-medium"  # Pre-trained MusicGen model name
model_save_path = "./ModelsFinetuned/New/MusicgenMedium_with_adapters_EncoderDecoder.pt"  # Path to the fine-tuned model
output_audio_path = "./GeneratedAudios/1.wav"  # Path to save the generated audio
AudioWaveform_graph_path = "./GeneratedGraphs/1.jpeg"  # Path to save the plot of generated audio
sample_rate = 16000  # Desired sample rate for the output audio
adapter_bottleneck_dim = 32  # Use the same dimension as training
max_new_tokens = 512 # To control length of music piece generated 512 = 10 sec
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# Configuration to choose between pre-trained and fine-tuned model
use_finetuned_model = True  # Set to True to use the fine-tuned model, False to use pre-trained model
 
# Custom adapter class (same as in training)
class Adapter(nn.Module):
    def __init__(self, bottleneck_channels=256, input_channels=1, seq_len=32000):
        super(Adapter, self).__init__()
        self.adapter_down = nn.Linear(seq_len, bottleneck_channels)
        self.activation = nn.ReLU()
        self.adapter_up = nn.Linear(bottleneck_channels, seq_len)
        self.dropout = nn.Dropout(p=0.0)
 
    def forward(self, residual):
        x = self.adapter_down(residual.squeeze(1))
        x = self.activation(x)
        x = self.adapter_up(x)
        x = self.dropout(x + residual.squeeze(1))
        return x.unsqueeze(1)
 
# MusicGen Model with Adapter (same as in training)
class MusicGenWithAdapters(nn.Module):
    def __init__(self, musicgen_model, processor, adapter_bottleneck_dim=256, device='cpu'):
        super(MusicGenWithAdapters, self).__init__()
        self.musicgen = musicgen_model
        self.adapter = Adapter(bottleneck_channels=adapter_bottleneck_dim, input_channels=1, seq_len=32000).to(device)
 
    def forward(self, audio_text):
        encoder_output = self.musicgen.generate(**audio_text, max_new_tokens=max_new_tokens)
        encoder_output = encoder_output.to('cpu')
        encoder_output = torchaudio.transforms.Resample(orig_freq=encoder_output.size(2), new_freq=32000)(encoder_output)
        encoder_output = encoder_output.to(self.adapter.adapter_down.weight.device)
        adapted = self.adapter(encoder_output)
        return adapted
 
# Function to load the model based on the configuration
def load_model(use_finetuned_model, model_save_path, device):
    if use_finetuned_model:
        # Load the fine-tuned model (MusicGen + Adapters)
        processor = AutoProcessor.from_pretrained(pretrained_model_name)
        musicgen_model = MusicgenForConditionalGeneration.from_pretrained(pretrained_model_name).to(device)
        model_with_adapters = MusicGenWithAdapters(musicgen_model, processor, adapter_bottleneck_dim=adapter_bottleneck_dim, device=device).to(device)
       
        # Load the state dicts for both the MusicGen model and the adapter
        checkpoint = torch.load(model_save_path, map_location=device)
        model_with_adapters.musicgen.load_state_dict(checkpoint['musicgen_state_dict'])
        model_with_adapters.adapter.load_state_dict(checkpoint['adapter_state_dict'])
 
        model_with_adapters.eval()
        total_params = sum(p.numel() for p in model_with_adapters.parameters())
        print(f"Total number of parameters in the fine-tuned model: {total_params}")
        return model_with_adapters
    else:
        # Load the pre-trained MusicGen model
        musicgen_model = MusicgenForConditionalGeneration.from_pretrained(pretrained_model_name).to(device)
        musicgen_model.eval()
        total_params = sum(p.numel() for p in musicgen_model.parameters())
        print(f"Total number of parameters in the Original model: {total_params}")
        return musicgen_model
 
# Function to generate audio from a text prompt
def generate_audio(model, text_prompt, sample_rate=32000):
    processor = AutoProcessor.from_pretrained(pretrained_model_name)
 
    # Generate input tensor for the text prompt
    input_data = processor(text=[text_prompt], return_tensors="pt").to(device)
 
    # Generate audio using the fine-tuned or pre-trained model
    if isinstance(model, MusicGenWithAdapters):
        musicgen = model.musicgen
    else:
        musicgen = model
 
    with torch.no_grad():
        generated_output = musicgen.generate(**input_data, max_new_tokens=max_new_tokens)
 
    waveform = generated_output.squeeze(0).cpu()
 
    if sample_rate != 32000:
        resampler = torchaudio.transforms.Resample(orig_freq=32000, new_freq=sample_rate)
        waveform = resampler(waveform)
 
    return waveform
 
# Main inference code
if __name__ == "__main__":
    # Get text prompt from the user
    text_prompt = input("Enter a text prompt for music generation: ")
 
    # Load the appropriate model (pre-trained or fine-tuned) based on the setting
    model = load_model(use_finetuned_model, model_save_path, device)
 
    # Generate audio
    waveform = generate_audio(
        model,
        text_prompt,
        sample_rate=sample_rate
    )
 
    # Save the generated audio
    torchaudio.save(output_audio_path, waveform, sample_rate)
    print(f"Generated audio saved at {output_audio_path}")
 
    # Optional: Visualize the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(waveform.t().numpy())
    plt.savefig(AudioWaveform_graph_path)
    plt.show()
    print(f"Waveform graph saved at {AudioWaveform_graph_path}")
