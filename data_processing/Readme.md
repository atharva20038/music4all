# **Dataset Processing**

## **Downloader**
This script downloads **Hindustani classical/Turkish Makam music recordings** from the **CompMusic Dunya API**, extracts metadata, and saves it into a structured CSV file.

---

## **Requirements**
Ensure you have **Python 3.7+** installed, then install the dependencies:

```sh
pip install compmusic tqdm pandas concurrent.futures
```

---

## **Usage**
### **1️⃣ Set up the API Token**
Obtain a **Dunya API token** from [CompMusic](https://compmusic.upf.edu/) and add it to the script:

```python
DUNYA_API_TOKEN = "your_token_here"
```

### **2️⃣ Run the Script**
To download the dataset, execute:

```sh
python download_hindustani.py
```

or 

```sh
python download_makam.py
```

---

## **Metadata Output**
After running the script, **hindustani.csv** will contain:

| **Column**    | **Description** |
|--------------|-------------|
| `title`      | Name of the track |
| `mbid`       | Unique MusicBrainz ID |
| `raga`       | Raga of the composition |
| `taal`       | Taal (rhythmic cycle) |
| `laya`       | Laya (tempo classification) |
| `instrument` | Instruments used in the recording |
| `audio_path` | File path where the MP3 is saved |

After running the script, **makam.csv** will contain:

| **Column**    | **Description** |
|--------------|-------------|
| `title`      | Name of the track |
| `mbid`       | Unique MusicBrainz ID |
| `makam`       | Makam (modal structure) |
| `usul`       | Usul (rhythmic pattern) |
| `instrument` | Instruments used in the recording |
| `audio_path` | File path where the MP3 is saved |

## Chunking & Metadata Generation

This script **processes Hindustani Classical music recordings**, **splits them into 30-second audio chunks**, and **saves the metadata** into a structured CSV file.

---

### **🔹 Features**  
**Processes and chunks audio files** while maintaining metadata  
**Ensures uniform frame rate (32kHz) for all chunks**  
**Extracts meaningful 30-second segments** from 20% to 90% of the track length  

---

### **📌 Output Files**  
- **Audio Chunks** → Stored in the `chunks/` directory  
- **Updated Metadata** → `new_metadata_hindustani.csv`  

---

### **⚡ Execution Example**  
To run the script, navigate to the directory and execute:

```sh
python process_hindu_data.py
```

or 

```sh
python process_hindu_data.py
```

**After execution:**  
- The processed **30-second chunks** will be stored in the `chunks/` folder.  
- The **new metadata file (`new_metadata_hindustani.csv`)** will be created, containing information about each chunk.  

## **Prompt Generation & Dataset Split**  

This script **automatically generates curated prompts** for **Turkish Makam music compositions** based on **instrument, makam, usul** metadata. It then **splits the dataset into train (70%), validation (10%), and test (20%)** sets.

---

### **📌 Output Files**  
- **`train_Curated_Turkish_Makam_Prompts.csv`** (70%)  
- **`val_Curated_Turkish_Makam_Prompts.csv`** (10%)  
- **`test_Curated_Turkish_Makam_Prompts.csv`** (20%)  

---

### **⚡ Execution Example**  
To run the script, navigate to the directory and execute:

```sh
python process_clipped_meta_hindu.py
```

or 

```sh
python process_clipped_meta_makam.py
```

**After execution:**  
- **Curated prompts** will be generated for Turkish Makam music.  
- The dataset will be **split into train, validation, and test sets**.  
- The **new metadata files (`train_Curated_Turkish_Makam_Prompts.csv`, etc.)** will be saved.  
