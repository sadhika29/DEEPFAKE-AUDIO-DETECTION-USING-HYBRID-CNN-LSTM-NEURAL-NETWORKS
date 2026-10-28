"""
ASVspoof 2019 Logical Access (LA) Dataset Utility
Assists in downloading, unpacking, and structuring the ASVspoof 2019 dataset.
"""

import os
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_DIR, DATA_DIR, LABEL_BONAFIDE, LABEL_SPOOF

ASVSPOOF_OFFICIAL_URL = "https://datashare.ed.ac.uk/handle/10283/3336"
ZENODO_MIRROR = "https://zenodo.org/records/4835108"

def display_instructions():
    print("""
================================================================================
ASVSPOOF 2019 LOGICAL ACCESS (LA) DATASET INSTRUCTIONS
================================================================================
The official ASVspoof 2019 Logical Access dataset contains genuine human speech
and synthetic speech produced by state-of-the-art Text-to-Speech (TTS) and 
Voice Conversion (VC) algorithms.

Official Links:
1. University of Edinburgh DataShare:
   https://datashare.ed.ac.uk/handle/10283/3336
2. Academic Zenodo Mirror:
   https://zenodo.org/records/4835108
3. Kaggle Mirror (Search: 'ASVspoof 2019 LA'):
   https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

Expected File Structure:
- LA/ASVspoof2019_LA_train/flac/*.flac
- LA/ASVspoof2019_LA_dev/flac/*.flac
- LA/ASVspoof2019_LA_eval/flac/*.flac
- LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt

Quick Ingestion:
If you have extracted the dataset locally, you can use `prepare_asvspoof_subset()`
to automatically link the files into the project's data directory.
================================================================================
""")

def prepare_asvspoof_subset(
    protocol_file,
    flac_dir,
    output_dir=RAW_DATA_DIR,
    max_samples_per_class=500
):
    """
    Reads an ASVspoof protocol file and copies/links a balanced subset of files
    into data/raw/bonafide and data/raw/spoof for fast training.
    """
    protocol_file = Path(protocol_file)
    flac_dir = Path(flac_dir)
    output_dir = Path(output_dir)

    bonafide_dir = output_dir / "bonafide"
    spoof_dir = output_dir / "spoof"
    bonafide_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    if not protocol_file.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
    if not flac_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {flac_dir}")

    bona_count = 0
    spoof_count = 0
    records = []

    print(f"Reading {protocol_file.name} and extracting up to {max_samples_per_class} per class...")

    with open(protocol_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_id = parts[1]
                key = parts[4].lower()
                src_flac = flac_dir / f"{file_id}.flac"

                if src_flac.exists():
                    if key == "bonafide" and bona_count < max_samples_per_class:
                        dest = bonafide_dir / f"{file_id}.flac"
                        shutil.copy2(src_flac, dest)
                        records.append({"file_path": str(dest), "label": LABEL_BONAFIDE, "class_name": "bonafide"})
                        bona_count += 1
                    elif key == "spoof" and spoof_count < max_samples_per_class:
                        dest = spoof_dir / f"{file_id}.flac"
                        shutil.copy2(src_flac, dest)
                        records.append({"file_path": str(dest), "label": LABEL_SPOOF, "class_name": "spoof"})
                        spoof_count += 1

                if bona_count >= max_samples_per_class and spoof_count >= max_samples_per_class:
                    break

    df = pd.DataFrame(records)
    protocol_path = DATA_DIR / "protocol.csv"
    df.to_csv(protocol_path, index=False)
    print(f"Subset prepared: {bona_count} bonafide, {spoof_count} spoof samples logged to {protocol_path}")
    return df

if __name__ == "__main__":
    display_instructions()
