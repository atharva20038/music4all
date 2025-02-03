
# **Inter-Annotator Agreement (IAA) Score Computation for Music Annotations**

## **Overview**
This project computes **Inter-Annotator Agreement (IAA) Scores** for music annotation comparisons using a **distance-based scoring system**. It processes an Excel file containing annotation comparisons and evaluates agreement using **Kappa scores**.

## **Features**
Parses annotation comparisons using a structured symbol-based system.  
Computes **distance-based** and **direction-based** agreement scores.  
Uses **Kappa scoring** to measure annotator agreement beyond chance.  

## **Requirements**
This project requires **Python 3.7+** and the following dependencies:

```sh
pip install pandas numpy openpyxl
```

## **Usage**
Run the script with the required **Excel file** and specify the sheet name (default: `"iaa"`).

```sh
python iaa.py --input_file "sample_annotations.xlsx" --sheet_name "iaa"
```

### **Arguments**
| Argument       | Description |
|---------------|-------------|
| `--input_file` | **(Required)** Path to the Excel file containing annotation comparisons. |
| `--sheet_name` | **(Optional)** Sheet name to process (default: `"iaa"`). |

---

## **Sample Data**
To demonstrate how the script works, we have provided a **sample Excel file (`sample_annotations.xlsx`)** in the repository. This file contains mock annotation comparisons for testing the IAA computation.

The script processes **five ranking categories**, each corresponding to different aspects of the music evaluation:

| Column Name    | Represents |
|---------------|------------|
| `agreement_1` | **Overall Quality** |
| `agreement_2` | **Instrumentation** (clarity, richness, and arrangement of instruments) |
| `agreement_3` | **Melody** (musicality, harmonic progression, and tune) |
| `agreement_4` | **Rhythm** (beat structure, tempo consistency, and groove) |
| `agreement_5` | **Creativity** (originality, genre blending, and uniqueness) |

Each of these categories is compared between two music models (e.g., `audioA` vs. `audioB`), and **ELO scores are adjusted accordingly**.

---

## **Understanding the Output**
After running the script, you will see output similar to:

```sh
For column **overall**:
   - Absolute Agreement: 0.756
   - IAA Kappa Score: 0.457

For column **instruments**:
   - Absolute Agreement: 0.682
   - IAA Kappa Score: 0.312
```

- **Absolute Agreement**: Measures direct agreement between annotators.
- **IAA Kappa Score**: Adjusts agreement by considering random chance (higher is better).

<div align="center">
  <img src="img/iaa_all.png" width="900"/>
</div>


# **ELO Rating System for Music Model Comparisons**

## **Overview**
This project calculates **ELO ratings** for different music generation models based on **annotator rankings**. The system dynamically updates **ratings based on head-to-head comparisons** and helps analyze model performance across different aspects like **melody, rhythm, instruments, and creativity**.

## **Features**
Uses **ELO rating system** for fair ranking adjustments.  
Filters data based on **specific prompts (e.g., "creativity")**.  

---

## **Requirements**
Ensure you have **Python 3.7+** and install the required dependencies:

```sh
pip install pandas numpy openpyxl
```

---

## **Usage**
To compute ELO ratings from an **Excel file**, run:


### **Arguments**
| Argument       | Description |
|---------------|-------------|
| `--input_file` | **(Required)** Path to the Excel file containing rankings. |
| `--sheet_name` | **(Optional)** Sheet name to process (default: `"Sheet2"`). |

---

## **Sample Data**
To help you understand how the script works, we have included **`sample_rankings.xlsx`**, which contains example **head-to-head music model rankings**.

---

## **Understanding the Output**
After running the script, you will see output similar to:

```sh
For category **overall**, updated ELO ratings: {'MusicGenBaseline': 1520, 'MusicGenFinetuned': 1485, 'MustangoBaseline': 1500, 'MustangoFinetuned': 1510}
For category **instruments**, updated ELO ratings: {'MusicGenBaseline': 1510, 'MusicGenFinetuned': 1490, 'MustangoBaseline': 1505, 'MustangoFinetuned': 1495}
```

- **Higher ELO scores** indicate better performance in that category.
- **Lower scores** mean the model performed worse in comparisons.

<div align="center">
  <img src="img/elo.png" width="900"/>
</div>
