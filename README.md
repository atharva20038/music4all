# Music4All
This repository contains our code for the paper:  "Music for All : Representational bias and cross-cultural adaptability in music generation models."

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

For Hindustani Classical, the dataset includes five instrument types—sarangi, harmonium, tabla, violin, and tanpura—along with voice. It comprises 41 ragas across two laya types: Madhya and Vilambit.
The dataset features 15 instruments specific to Turkish makam, including the oud, tanbur, ney, davul, clarinet, kös, kudüm, yaylı tanbur, tef, kanun, zurna, bendir, darbuka, classical kemençe, rebab, and çevgen. It encompasses 93 different makams and 63 distinct usuls.

## Adapter Positioning

<div align="center">
  <img src="img/model_settings-1.png" width="900"/>
</div>

### Mustango
To enhance this process, a Bottleneck Residual Adapter with convolution layers is integrated into the up-sampling, middle, and down-sampling blocks of the UNet, positioned just after the cross-attention block. This design facilitates cultural adaptation while preserving computational efficiency. The adapters reduce channel dimensions by a factor of 8, using a kernel size of 1 and GeLU activation after the down-projection layers to introduce non-linearity.

### MusicGen
In MusicGen, we enhance the model with an additional 2 million parameters by integrating Linear Bottleneck Residual Adapter after the transformer decoder within the MusicGen architecture after thorough experimentation with other placements.

The total parameter count of both the models is ~2 billion, making the adapter only 0.1% of the total size (2M params).
For both models, we used two RTX A6000 GPUs over a period of around 10 hours. The adapter block was fine-tuned, using the AdamW optimizer using MSE (Reconstruction Loss).

## Evaluations
### **Objective Evaluation Metrics for Music Models**

The table below presents the **objective evaluation metrics** for **Hindustani Classical Music** and **Turkish Makam**, assessing the quality of generated music based on **Fréchet Audio Distance (FAD)**, **Fréchet Distance (FD)**, **Kullback-Leibler Divergence (KLD)**, and **Peak Signal-to-Noise Ratio (PSNR)**.

<div style="display: flex; flex-direction: row; justify-content: space-between;">

<!-- Hindustani Classical Music Table -->
<table style="width:45%">
<caption><b>Hindustani Classical Music</b></caption>
<tr>
<th>Model</th><th>FAD ↓</th><th>FD ↓</th><th>KLD ↓</th><th>PSNR ↑</th>
</tr>
<tr><td><b>MusicGen Baseline</b></td><td>40.05</td><td>75.76</td><td>6.53</td><td>16.23</td></tr>
<tr><td><b>MusicGen Finetuned</b></td><td>40.04</td><td>72.65</td><td>6.12</td><td>16.18</td></tr>
<tr><td><b>Mustango Baseline</b></td><td>6.36</td><td>45.31</td><td>2.73</td><td>16.78</td></tr>
<tr><td><b>Mustango Finetuned</b></td><td><b>5.18</b></td><td><b>22.03</b></td><td><b>1.26</b></td><td><b>17.70</b></td></tr>
</table>
<!-- Turkish Makam Table -->
<table style="width:45%">
<caption><b>Turkish Makam</b></caption>
<tr>
<th>Model</th><th>FAD ↓</th><th>FD ↓</th><th>KLD ↓</th><th>PSNR ↑</th>
</tr>
<tr><td><b>MusicGen Baseline</b></td><td>39.65</td><td>57.29</td><td>7.35</td><td>14.60</td></tr>
<tr><td><b>MusicGen Finetuned</b></td><td>39.68</td><td>56.71</td><td>7.21</td><td>14.46</td></tr>
<tr><td><b>Mustango Baseline</b></td><td>8.65</td><td>75.21</td><td>6.01</td><td><b>16.60</b></td></tr>
<tr><td><b>Mustango Finetuned</b></td><td><b>2.57</b></td><td><b>20.56</b></td><td><b>4.81</b></td><td>16.17</td></tr>
</table>

</div>

### **Human Evaluation (ELO Ratings, ↑)**  

The table below presents the **human evaluation scores (ELO Ratings)** for **Hindustani Classical Music** and **Turkish Makam**, where **higher values indicate better performance**.

#### **Hindustani Classical Music - All Queries**
| **Model**  | **OA ↑**  | **Inst. ↑**  | **MC ↑**  | **RC ↑**  | **CR ↑**  |
|------------|----------|----------|----------|----------|----------|
| **MusicGen Baseline**  | 1525  | 1520  | 1540  | 1552  | 1546  |
| **Mustango Baseline**  | 1449  | 1466  | 1409  | 1470  | 1518  |
| **MusicGen Finetuned**  | 1448  | 1454  | 1428  | 1439  | 1448  |
| **Mustango Finetuned**  | **1577**  | **1559**  | **1623**  | **1538**  | **1487**  |

#### **Turkish Makam - All Queries**
| **Model**  | **OA ↑**  | **Inst. ↑**  | **MC ↑**  | **RC ↑**  | **CR ↑**  |
|------------|----------|----------|----------|----------|----------|
| **MusicGen Baseline**  | 1539  | 1562  | 1597  | 1622  | 1603  |
| **Mustango Baseline**  | 1527  | 1531  | 1499  | 1523  | 1560  |
| **MusicGen Finetuned**  | **1597**  | 1529  | 1570  | 1570  | 1541  |
| **Mustango Finetuned**  | 1337  | 1377  | 1334  | 1286  | 1297  |

#### **Legend:**
- **OA (Overall Accuracy)**  
- **Inst. (Instrumentation)**  
- **MC (Melodic Consistency)**  
- **RC (Rhythmic Consistency)**  
- **CR (Creativity)**  



## Citation
Please consider citing the following article if you found our work useful:
```
@inproceedings{mehta-etal-2025-music,
    title = "Music for All: Representational Bias and Cross-Cultural Adaptability of Music Generation Models",
    author = "Mehta, Atharva  and
      Chauhan, Shivam  and
      Djanibekov, Amirbek  and
      Kulkarni, Atharva  and
      Xia, Gus  and
      Choudhury, Monojit",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.258/",
    doi = "10.18653/v1/2025.findings-naacl.258",
    pages = "4569--4585",
    ISBN = "979-8-89176-195-7",
    abstract = "The advent of Music-Language Models has greatly enhanced the automatic music generation capability of AI systems, but they are also limited in their coverage of the musical genres and cultures of the world. We present a study of the datasets and research papers for music generation and quantify the bias and under-representation of genres. We find that only 5.7{\%} of the total hours of existing music datasets come from non-Western genres, which naturally leads to disparate performance of the models across genres.We then investigate the efficacy of Parameter-Efficient Fine-Tuning (PEFT) techniques in mitigating this bias. Our experiments with two popular models {--} MusicGen and Mustango, for two underrepresented non-Western music traditions {--} Hindustani Classical and Turkish Makam music, highlight the promises as well as the non-triviality of cross-genre adaptation of music through small datasets, implying the need for more equitable baseline music-language models that are designed for cross-cultural transfer learning."
}
```
