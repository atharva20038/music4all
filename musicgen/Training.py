import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchaudio
from audiocraft.models import MusicGen
import datasets
from tensorboardX import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os, random, numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
 
# Hyperparameters and Paths
log_dir = './runs/MusicGen_EncoderDecoder_Adaptor/'  # Directory for TensorBoard logs
adapter_bottleneck_dim = 32  # Adapter bottleneck dimension size
batch_size = 4  # Training batch size
shuffle_data = True  # Shuffle dataset during training
learning_rate = 5e-5  # Learning rate for optimizer
weight_decay = 0.05  # Weight decay (L2 regularization) for optimizer
num_epochs = 30  # Number of training epochs
dropout_prob = 0.1  # Dropout probability in Adapter Layers
max_grad_norm = 1.0  # Set the maximum norm for gradient clipping
train_test_split_size = 0.1  # Train-validation split
patience = 5  # Number of epochs to wait for improvement on validation loss
pretrained_model_name = "facebook/musicgen-medium"  # Pre-trained MusicGen model name
train_data_file = r"/home/shivam.chauhan/Music/Atharva/Processed_Dataset/Makam_32KHz/metadata.csv"  # Path to training dataset
audio_base_path = r"/home/shivam.chauhan/Music/Atharva/Processed_Dataset/Makam_32KHz/"  # Base path for audio files
model_save_path = "./ModelsFinetuned/MusicgenMedium_with_adapters_EncoderDecoder_newMaqam.pt"  # Save path for fine-tuned model
 
# Maqam -- /home/shivam.chauhan/Music/Atharva/Processed_Dataset/Makam_32KHz/
# Indian Classical -- /home/shivam.chauhan/Music/Atharva/Processed_Dataset/Indian Classical Music 32KHz/
 
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(142)
 
class Adapter(nn.Module):
    def __init__(self, bottleneck_channels=256, input_channels=1, seq_len=32000):
        super(Adapter, self).__init__()
 
        # Define the down-projection (reduce the dimensionality)
        self.adapter_down = nn.Linear(seq_len, bottleneck_channels)
        self.activation = nn.GELU()
       
        # Define the up-projection (restore the dimensionality)
        self.adapter_up = nn.Linear(bottleneck_channels, seq_len)
       
        # Optional dropout for regularization
        self.dropout = nn.Dropout(p=dropout_prob)
 
    def forward(self, residual):
        # Flattening the input for Linear transformation
        x = self.adapter_down(residual.squeeze(1))  # Remove the channel dimension to apply linear layer
       
        # Apply activation
        x = self.activation(x)
       
        # Restore the sequence length
        x = self.adapter_up(x)
       
        # Add residual connection and dropout
        x = self.dropout(x + residual.squeeze(1))  # Apply residual connection and dropout
       
        return x.unsqueeze(1)  # Add back the channel dimension
 
 
 
# MusicGen Model with Adapter
class MusicGenWithAdapters(nn.Module):
    def __init__(self, musicgen_model, processor, adapter_bottleneck_dim=adapter_bottleneck_dim, device='cpu'):
        super(MusicGenWithAdapters, self).__init__()
        self.musicgen = musicgen_model
       
        # Get the size of the generated music to determine the input channels
        self.adapter = Adapter(bottleneck_channels=adapter_bottleneck_dim, input_channels=1, seq_len=32000).to(device)
 
    def train(self, mode=True):
        self.adapter.train(mode)
 
    def eval(self):
        self.train(False)
 
 
    def forward(self, audio_text):
        encoder_output = self.musicgen.generate(**audio_text, max_new_tokens=128)
       
        # Move the encoder output to CPU needed for Resample
        encoder_output = encoder_output.to('cpu')
        encoder_output = torchaudio.transforms.Resample(orig_freq=encoder_output.size(2), new_freq=32000)(encoder_output)
        # Move back to the original device
        encoder_output = encoder_output.to(self.adapter.adapter_down.weight.device)
       
        # Pass through the adapter
        adapted = self.adapter(encoder_output)
        return adapted
 
# Distributed setup
def setup():
    dist.init_process_group(backend='nccl')
 
def cleanup():
    dist.destroy_process_group()
 
def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
 
    setup()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
 
   
 
    # Move model to device and wrap in DDP
    model_with_adapters = model_with_adapters.to(device)
    model_with_adapters = DDP(model_with_adapters, device_ids=[local_rank], output_device=local_rank)
    model_with_adapters.module.adapter.train()
 
    # Update the optimizer to include only adapter parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model_with_adapters.parameters()))
    total_params = sum(p.numel() for p in model_with_adapters.parameters() if p.requires_grad)
    print(f"Total number of trainable elements: {total_params}")
 
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    data_files = {"train": train_data_file}
    dataset = datasets.load_dataset("csv", data_files=data_files)
   
    def preprocess_function(examples):
        examples["audio"] = audio_base_path + examples["audio_filename"]
        examples["text_prompt"] = examples["text_prompt"]
        return examples
 
 
    dataset = dataset.map(preprocess_function)
    dataset = dataset.filter(lambda x: x["audio"] is not None)
 
    train_val_split = dataset['train'].train_test_split(test_size=train_test_split_size)
    train_sampler = DistributedSampler(
        train_val_split['train'],
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        train_val_split['test'],
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
 
    train_dataloader = DataLoader(
        train_val_split['train'],
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda x: [(b['audio'], b['text_prompt']) for b in x]
    )
 
    val_dataloader = DataLoader(
        train_val_split['test'],
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=lambda x: [(b['audio'], b['text_prompt']) for b in x]
    )
 
 
 
    mse_loss_fn = nn.MSELoss()
    best_loss = float('inf')
    no_improvement_counter = 0
 
    if rank == 0:
        writer = SummaryWriter(logdir=log_dir)
    else:
        writer = None  # Only the main process writes to TensorBoard
 
    for epoch in range(num_epochs):
        model_with_adapters.module.adapter.train()
        total_loss = 0
        train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_dataloader):
            target_audios = []
            target_text = []
            for audio_text in batch:
                waveform, sample_rate = torchaudio.load(audio_text[0])
                target_audios.append(waveform.to(device))
                target_text.append(audio_text[1])
            target_audios = torch.stack(target_audios, dim=0)
            inputs = processor(text=target_text, padding=True, return_tensors="pt").to(device)
           
            reconstructed_audios = model_with_adapters(inputs)
            min_length = min(reconstructed_audios.size(-1), target_audios.size(-1))
            reconstructed_audios = reconstructed_audios[:, :, :min_length]
            target_audios = target_audios[:, :, :min_length]
 
            loss = mse_loss_fn(reconstructed_audios, target_audios)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
 
            if writer is not None:
                writer.add_scalar('Loss/Training per Batch', loss.item(), epoch * len(train_dataloader) + batch_idx)
 
        avg_loss = total_loss / len(train_dataloader)
        val_loss = 0
        model_with_adapters.module.adapter.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                target_audios = []
                target_text = []
                for audio_text in batch:
                    waveform, sample_rate = torchaudio.load(audio_text[0])
                    target_audios.append(waveform.to(device))
                    target_text.append(audio_text[1])
                target_audios = torch.stack(target_audios, dim=0)
                inputs = processor(text=target_text, padding=True, return_tensors="pt").to(device)
 
                reconstructed_audios = model_with_adapters(inputs)
                min_length = min(reconstructed_audios.size(-1), target_audios.size(-1))
                reconstructed_audios = reconstructed_audios[:, :, :min_length]
                target_audios = target_audios[:, :, :min_length]
                loss = mse_loss_fn(reconstructed_audios, target_audios)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
 
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_loss} - Validation Loss: {avg_val_loss}")
 
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                no_improvement_counter = 0
                # Save both the musicgen model and the adapter
                torch.save({
                    'musicgen_state_dict': model_with_adapters.module.musicgen.state_dict(),
                    'adapter_state_dict': model_with_adapters.module.adapter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_save_path)
            else:
                no_improvement_counter += 1
 
 
            if no_improvement_counter >= patience:
                print("Early stopping: no improvement in validation loss for", patience, "epochs.")
                break
 
            writer.add_scalar('Loss/Training per Epoch', avg_loss, epoch)
            writer.add_scalar('Loss/Validation per Epoch', avg_val_loss, epoch)
 
    if writer is not None:
        writer.close()
    cleanup()
 
if __name__ == "__main__":
    main()
 
 
