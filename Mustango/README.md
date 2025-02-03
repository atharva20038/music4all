<div align="center">

# Mustango

</div>

## Quickstart Guide

### Generate music from a text prompt  
Run the following command to automate safetensor-to-bin conversion and generate music using your model:  

```sh
python script.py --model saved/1734429910/model_2.bin \
                 --test_files processed_sampled_data.json test_data.json \
                 --original_args saved/1734429910/summary.jsonl \
                 --test_references test_reference_data

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
python script.py --model saved/1734429910/model_2.bin \
                 --test_files processed_sampled_data.json test_data.json \
                 --original_args saved/1734429910/summary.jsonl \
                 --test_references test_reference_data
```

#### 2️⃣ Run baseline inference only  
```sh
python script.py --model saved/1734429910/model_2.bin \
                 --test_files test_data.json \
                 --original_args config.json \
                 --test_references reference_dataset \
                 --baseline
```

This guide ensures that you can quickly get started with generating and evaluating music models with minimal setup! 🚀
```

This markdown format ensures proper rendering in GitHub and other markdown-compatible platforms. Let me know if you need further refinements! 🚀

## Installation

```bash
git clone https://github.com/AMAAI-Lab/mustango
cd mustango
pip install -r requirements.txt
cd diffusers
pip install -e .
```


## Training

We use the `accelerate` package from Hugging Face for multi-gpu training. Run `accelerate config` from terminal and set up your run configuration by the answering the questions asked.

You can now train **Mustango** on the MusicBench dataset using:

```bash
accelerate launch train.py \
--text_encoder_name="google/flan-t5-large" \
--scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config_munet.json" \
--model_type Mustango --freeze_text_encoder --uncondition_all --uncondition_single \
--drop_sentences --random_pick_text_column --snr_gamma 5 \
```

The `--model_type` flag allows to choose either Mustango, or Tango to be trained with the same code. However, do note that you also need to change `--unet_model_config` to the relevant config: diffusion_model_config_munet for Mustango; diffusion_model_config for Tango.

The arguments `--uncondition_all`, `--uncondition_single`, `--drop_sentences` control the dropout functions as per Section 5.2 in our paper. The argument of `--random_pick_text_column` allows to randomly pick between two input text prompts - in the case of MusicBench, we pick between ChatGPT rephrased captions and original enhanced MusicCaps prompts, as depicted in Figure 1 in our paper.

Recommended training time from scratch on MusicBench is at least 40 epochs.


## Model Zoo

We have released the following models:

Mustango Pretrained: https://huggingface.co/declare-lab/mustango-pretrained
Mustango Adapted: https://huggingface.co/athi180202/music4all_mustango



