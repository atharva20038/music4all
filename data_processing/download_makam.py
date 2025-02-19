import compmusic
from tqdm import tqdm
import os
from compmusic import dunya
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set Dunya API Token
DUNYA_API_TOKEN = ""  # Add your token here
dunya.set_token(DUNYA_API_TOKEN)

# Output directory for Turkish Makam music data
OUTPUT_DIR = "Turkish_Makam"
METADATA_FILE = "turkish_makam.csv"

# Initialize metadata storage
metadata_df = pd.DataFrame(columns=["title", "mbid", "makam", "usul", "instrument", "audio_path"])
mbid_to_path = {}

print("Starting Turkish Makam music dataset download...")

def fetch_recording(release):
    """Fetches a recording from the CompMusic API."""
    try:
        return dunya.makam.get_release(release["mbid"])
    except Exception as e:
        print(f"Error fetching release {release['mbid']}: {e}")
        return None

def process_recording(recording):
    """Processes and downloads individual recordings."""
    global metadata_df
    
    try:
        release_title = recording["title"]
        
        for rec in recording["recordings"]:
            track_title = rec["title"]
            track_mbid = rec["mbid"]
            save_path = os.path.join(OUTPUT_DIR, release_title, track_title)

            os.makedirs(save_path, exist_ok=True)

            # Fetch metadata for the track
            meta = dunya.makam.get_recording(track_mbid)

            # Download MP3 file
            try:
                dunya.makam.download_mp3(track_mbid, save_path)
            except Exception as e:
                print(f"Error downloading {track_mbid}: {e}")
                continue

            # Extract metadata attributes
            instruments = [p['instrument']['name'] for p in meta.get('artists', []) if 'instrument' in p]
            makams = [m["name"] for m in meta.get('makams', [])]
            usuls = [u["name"] for u in meta.get('usuls', [])]

            mbid_to_path[track_mbid] = save_path

            # Append to metadata dataframe
            metadata_df = metadata_df.append({
                "title": track_title,
                "mbid": track_mbid,
                "makam": ",".join(makams),
                "usul": ",".join(usuls),
                "instrument": ",".join(instruments),
                "audio_path": save_path
            }, ignore_index=True)

    except Exception as e:
        print(f"Error processing recording: {e}")

# Fetch releases
print("Fetching releases from Dunya...")
releases = dunya.makam.get_releases()

# Fetch recordings concurrently
print("Fetching recording metadata...")
recordings = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_recording, r): r for r in releases}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Metadata"):
        result = future.result()
        if result:
            recordings.append(result)

# Process recordings concurrently
print("Processing and downloading recordings...")
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_recording, r) for r in recordings]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading & Processing"):
        pass

# Save metadata to CSV
metadata_df.to_csv(METADATA_FILE, index=False)
print(f"Metadata saved to {METADATA_FILE}")

# Print downloaded files
print("Download complete!")
print(mbid_to_path)
