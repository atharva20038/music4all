<div align="center">

# MusicGen

</div>

## Quickstart Guide

### Generate music from a text prompt  
Run the following command to train the model:  

```sh
python Training.py

```

# Training Configuration

Below table provides an overview of the key hyperparameters and paths used in the training process.

| Parameter                 | Description                                              | Value                                                      |
|---------------------------|----------------------------------------------------------|------------------------------------------------------------|
| **Pretrained Model**      | Name of the pre-trained MusicGen model used for fine-tuning. | `facebook/musicgen-medium`                                  |
| **Dataset Path**          | Path to the CSV file containing metadata for training.  | `/home/shivam.chauhan/Music/Atharva/Processed_Dataset/Makam_32KHz/metadata.csv` |
| **Audio Base Path**       | Directory containing audio files for training.          | `/home/shivam.chauhan/Music/Atharva/Processed_Dataset/Makam_32KHz/` |
| **Model Save Path**       | Path where the fine-tuned model will be saved.         | `./ModelsFinetuned/MusicgenMedium_with_adapters_EncoderDecoder_newMaqam.pt` |
| **Adapter Bottleneck Dim**| Size of the bottleneck layer in the adapter.            | `32`                                                        |
| **Batch Size**            | Number of samples per training batch.                    | `4`                                                         |
| **Learning Rate**         | Step size for updating model weights.                    | `5e-5`                                                      |
| **Weight Decay**          | Regularization parameter to prevent overfitting.        | `0.05`                                                      |
| **Number of Epochs**      | Total number of training iterations over the dataset.    | `30`                                                        |
| **Dropout Probability**   | Probability of dropping units in adapter layers.        | `0.1`                                                       |
| **Max Gradient Norm**     | Maximum norm for gradient clipping to prevent explosion. | `1.0`                                                       |
| **Train-Test Split Ratio**| Proportion of data used for training vs validation.      | `90:10`                                                     |
| **Early Stopping Patience** | Number of epochs without improvement before stopping training. | `5 epochs`                                                 |

## Explanation of Key Components:
- **Pretrained Model**: A foundation model (`facebook/musicgen-medium`) that is fine-tuned for a specific task.
- **Adapter Bottleneck**: A technique to introduce lightweight modifications without retraining the entire model.
- **Batch Size**: A lower batch size (4) is used, likely due to memory constraints with large audio models.
- **Dropout**: Helps prevent overfitting by randomly deactivating parts of the model during training.
- **Gradient Clipping**: Ensures stability in training by capping large gradient updates.
- **Early Stopping**: Prevents unnecessary training epochs if validation loss stops improving.

This configuration is optimized for fine-tuning **MusicGen** with **adapter-based modifications** for improved music generation capabilities.


### Example Scenarios
Run the following command to generate audio's using your model:  

```sh
python GenerateAudio.py

```

# Inference Configuration

This table provides an overview of the key parameters used in the **inference process** for generating music.

| Parameter                 | Description                                               | Value                                                      |
|---------------------------|-----------------------------------------------------------|------------------------------------------------------------|
| **Pretrained Model**      | Name of the pre-trained MusicGen model used for inference. | `facebook/musicgen-medium`                                  |
| **Fine-tuned Model Path** | Path where the fine-tuned model is stored.                | `./ModelsFinetuned/New/MusicgenMedium_with_adapters_EncoderDecoder.pt` |
| **Output Audio Path**     | Path where the generated audio file is saved.             | `./GeneratedAudios/1.wav`                                  |
| **Waveform Graph Path**   | Path where the waveform visualization is stored.          | `./GeneratedGraphs/1.jpeg`                                 |
| **Sample Rate**           | Desired sample rate for the generated audio.              | `16,000 Hz`                                                |
| **Adapter Bottleneck Dim**| Size of the bottleneck layer in the adapter network.      | `32`                                                     |
| **Max New Tokens**        | Controls the length of the generated music (512 ≈ 10 sec). | `512`                                                      |
| **Device**               | Specifies whether to use GPU or CPU for inference.        | `CUDA if available, else CPU`                             |
| **Use Fine-tuned Model**  | Determines whether to use the fine-tuned model or pre-trained. | `True` (uses fine-tuned model)                            |

## Explanation of Key Components:
- **Pretrained Model**: Uses `facebook/musicgen-medium`, which is fine-tuned for customized music generation.
- **Fine-tuned Model Path**: If `use_finetuned_model = True`, the model loads from this path.
- **Waveform Graph Path**: Saves the waveform visualization as an image.
- **Max New Tokens**: Higher values generate longer music samples.
- **Device Selection**: Automatically chooses GPU (if available) for faster inference.

### 🔥 How the Inference Works:
1. The **model is loaded** (`pre-trained` or `fine-tuned` based on configuration).
2. The user **inputs a text prompt** describing the music to be generated.
3. The model **generates an audio waveform** based on the text input.
4. The generated music is **saved as a `.wav` file**.
5. A **waveform graph** is plotted and saved for visualization.

This setup ensures **efficient, high-quality music generation** using **MusicGen with adapter-based fine-tuning**. 🚀



## Installation

```bash
git clone https://github.com/atharva20038/music4all
cd Mustango
pip install -r requirements.txt
cd diffusers
pip install -e .
```


## Training

We use the `accelerate` package from Hugging Face for multi-gpu training. Run `accelerate config` from terminal and set up your run configuration by the answering the questions asked.

You can now train **Mustango** on the Hindustani Classical/Turkish Makam dataset using:

```bash
accelerate launch --text_encoder_name="google/flan-t5-large" 
                  --scheduler_name="stabilityai/stable-diffusion-2-1" 
                  --unet_model_config="configs/diffusion_model_config_munet.json" 
                  --model_type Mustango 
                  --freeze_text_encoder 
                  --uncondition_all --uncondition_single --drop_sentences --snr_gamma 5 
                  --train_file "data/metadata_train_hindustani.json" 
                  --validation_file "data/metadata_val_hindustani.json" 
                  --validation_file2 "data/metadata_val_hindustani.json" 
                  --test_file "data/metadata_test_hindustani.json"
```

The `--model_type` flag allows to choose either Mustango, or Tango to be trained with the same code. However, do note that you also need to change `--unet_model_config` to the relevant config: diffusion_model_config_munet for Mustango; diffusion_model_config for Tango.

The arguments `--uncondition_all`, `--uncondition_single`, `--drop_sentences` control the dropout functions. The `--train_file`, `--validation_file`, `--validation_file2`, `--test_file` files control the train/val/test splits. We have added samples of our files in `data` folder for format reference of input prompts in training. For audio only training, the data needs to be modified by keeping chords empty and beats 0 or other variables false. They are excluded from our training code as well during training by making these attributes empty during training.    

Recommended training time from scratch on Hindustani Classical/Turkish Makam is at least 10-15 epochs.

## Evaluation

The inference scripts above help in computing the evaluation metrics with the given model outputs. The `audioldm_eval` folder contains the scripts for evaluation metrics. For the prompts used for Bloom's taxonomy based evaluation the folders `bloom_prompt_hindustani` & `bloom_prompt_makam` contain the prompts based on which these models are evaluated with the help of human evaluations.


## Model Zoo

We have released the following models:

Mustango Adapted: https://huggingface.co/athi180202/music4all_mustango



