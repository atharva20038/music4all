
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

