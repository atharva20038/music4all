# Music4All
This repository contains our code for the paper:  "Music for All : Representational bias and cross-cultural adaptability in Music generation models."

[Survey](https://atharva20038.github.io/aimusicexamples.github.io/) | [Model](https://huggingface.co/collections/athi180202/music4all-67a0778b5b562859c2a9a8e1) | [Paper]()

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/collections/athi180202/music4all-67a0778b5b562859c2a9a8e1)

We present a study of the datasets and research papers for music generation and quantify the bias and under-representation of genres. We find that only **5.7\%** of the total hours of existing music datasets come from non-Western genres, which naturally leads to disparate performance of the models across genres. 
We then investigate the efficacy of *Parameter-Efficient Fine-Tuning (PEFT)* techniques in mitigating this bias. Our experiments with two popular models -- **[MusicGen](https://arxiv.org/abs/2306.05284)** and **[Mustango](https://arxiv.org/abs/2311.08355)**, for two underrepresented non-Western music traditions --  **Hindustani Classical** and **Turkish Makam** music, highlight the promises as well as the non-triviality of cross-genre adaptation of music through small datasets, implying the need for more equitable baseline music-language models that are designed for cross-cultural transfer learning.

## Global Music Generation Analysis
<div align="center">
  <img src="img/dataset_infographic_v7.png" width="900"/>
</div>

## Datasets

The [Compmusic](https://compmusic.upf.edu/datasets) dataset contains 120+ hours of Turkish Makam and Hindustani Classical data.

The [MTG-Saraga](https://mtg.github.io/saraga/) dataset contains 40+ hours of Hindustani Classical annotated data. 

## Adapter Positioning

<div align="center">
  <img src="img/model_settings-1.png" width="900"/>
</div>

### Mustango
To enhance this process, a Bottleneck Residual Adapter with convolution layers is integrated into the up-sampling, middle, and down-sampling blocks of the UNet, positioned just after the cross-attention block. This design facilitates cultural adaptation while preserving computational efficiency. The adapters reduce channel dimensions by a factor of 8, using a kernel size of 1 and GeLU activation after the down-projection layers to introduce non-linearity.

### MusicGen
In MusicGen, we enhance the model with an additional 2 million parameters by integrating Linear Bottleneck Residual Adapter after the transformer decoder within the MusicGen architecture after thorough experimentation with other placements.

The total parameter count of both the models is ~2 billion, making the adapter only 0.1% of the total size (2M params).

For both models we used two RTX A6000 GPUs over a period of around 10 hours. The adapter block was fine-tuned, using the AdamW optimizer using MSE (Reconstruction Loss).

## Evaluations
## **Objective Evaluation Metrics for Music Models**

The table below presents the **objective evaluation metrics** for **Hindustani Classical Music** and **Turkish Makam**, assessing the quality of generated music based on **Fréchet Audio Distance (FAD)**, **Fréchet Distance (FD)**, **Kullback-Leibler Divergence (KLD)**, and **Peak Signal-to-Noise Ratio (PSNR)**.

| **Hindustani Classical Music** | **FAD ↓** | **FD ↓** | **KLD ↓** | **PSNR ↑** || **Turkish Makam** | **FAD ↓** | **FD ↓** | **KLD ↓** | **PSNR ↑** |
|--------------------------------|----------|----------|----------|----------||----------------|----------|----------|----------|----------|
| **MGB**  | 40.05  | 75.76  | 6.53  | 16.23  || **MGB**  | 39.65  | 57.29  | 7.35  | 14.60  |
| **MGF**  | 40.04  | 72.65  | 6.12  | 16.18  || **MGF**  | 39.68  | 56.71  | 7.21  | 14.46  |
| **MTB**  | 6.36  | 45.31  | 2.73  | 16.78  || **MTB**  | 8.65  | 75.21  | 6.01  | **16.60**  |
| **MTF**  | **5.18**  | **22.03**  | **1.26**  | **17.70**  || **MTF**  | **2.57**  | **20.56**  | **4.81**  | 16.17  |




## Citation
Please consider citing the following article if you found our work useful:
```
@misc{melechovsky2023mustango,
      title={Mustango: Toward Controllable Text-to-Music Generation}, 
      author={Jan Melechovsky and Zixun Guo and Deepanway Ghosal and Navonil Majumder and Dorien Herremans and Soujanya Poria},
      year={2023},
      eprint={2311.08355},
      archivePrefix={arXiv},
}
```