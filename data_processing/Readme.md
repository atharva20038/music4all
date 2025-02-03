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
