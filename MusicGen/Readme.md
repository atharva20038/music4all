<div align="center">

# MusicGen

</div>

## Quickstart Guide

### Generate music from a text prompt  
Run the following command to automate safetensor-to-bin conversion and generate music using your model:  

```sh
python Training.py

```

### Arguments Explained  

| Argument            | Description |
|---------------------|-------------|
| `--model`      | **(Required)** Path to the directory containing `.safetensors` and `.bin` model files. This is where the models are stored for conversion and inference. |
| `--test_files`     | **(Required)** One or more test JSON files containing input prompts for generating music. You can provide multiple test files by separating them with spaces. |
| `--original_args`  | **(Required)** Path to the JSON file containing original model arguments (e.g., training configurations). |
| `--test_references`| **(Required)** Path to the test reference dataset, which provides ground truth or benchmark data for evaluating model outputs. |
| `--baseline`       | **(Optional)** If included, runs inference using a baseline model. Exclude this flag to run inference on all models in the directory. |

### Example Scenarios  

#### 1️⃣ Run inference on trained model  
```sh
python automate_inference.py --model saved/1734429910/model_2.bin \
                 --test_files processed_sampled_data.json test_data.json \
                 --original_args saved/1734429910/summary.jsonl \
                 --test_references test_reference_data
```

#### 2️⃣ Run baseline inference only  
```sh
python automate_inference.py --model saved/1734429910/model_2.bin \
                 --test_files processed_sampled_data.json \
                 --original_args config.json \
                 --test_references reference_dataset \
                 --baseline
```

This guide ensures that you can quickly get started with generating and evaluating music models with minimal setup! 🚀


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



