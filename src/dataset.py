import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    SAMPLE_RATE, DURATION, NUM_SAMPLES, RAW_DATA_DIR, DATA_DIR,
    LABEL_BONAFIDE, LABEL_SPOOF, RANDOM_SEED, BATCH_SIZE
)
from src.preprocess import extract_features_from_file

def generate_synthetic_audio(label_type="bonafide", duration=DURATION, sr=SAMPLE_RATE):
    """
    Synthesizes audio waveforms designed to mimic speech characteristics:
    - Bona fide (Genuine): Dynamic pitch contours, natural harmonic formants (F0, F1, F2),
      smooth amplitude envelopes, and low background breath noise.
    - Spoof (Synthetic/Vocoded Deepfake): Rigid/constant pitch, robotic phase buzz,
      metallic high-frequency quantization noise, and synthetic artifacts.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if label_type == "bonafide":
        # Realistic vocal pitch (F0) with natural vibrato and prosodic drift
        base_f0 = np.random.uniform(110.0, 240.0)  # Human voice range
        f0_drift = np.sin(2 * np.pi * np.random.uniform(1.5, 4.0) * t) * 8.0
        instantaneous_f0 = base_f0 + f0_drift
        phase_f0 = 2 * np.pi * np.cumsum(instantaneous_f0) / sr

        # Natural speech formants (vocal tract resonance peaks)
        f1_mult = np.random.uniform(2.5, 3.5)
        f2_mult = np.random.uniform(5.0, 7.5)
        f3_mult = np.random.uniform(9.0, 12.0)

        audio = (
            1.00 * np.sin(phase_f0) +
            0.50 * np.sin(f1_mult * phase_f0) +
            0.25 * np.sin(f2_mult * phase_f0) +
            0.12 * np.sin(f3_mult * phase_f0)
        )
        
        # Smooth attack-decay envelope mimicking syllabic articulation
        num_syllables = np.random.randint(4, 8)
        envelope = np.zeros_like(t)
        for i in range(num_syllables):
            center = (i + 0.5) * (duration / num_syllables)
            width = (duration / num_syllables) * 0.4
            envelope += np.exp(-((t - center) ** 2) / (2 * width ** 2))
        envelope = envelope / (np.max(envelope) + 1e-6)
        audio = audio * envelope

        # Subtle natural breathiness
        audio += np.random.normal(0, 0.015, len(t))

    else:
        # Spoofed audio: Vocoder phase distortion, robotic pitch, synthetic buzz
        rigid_f0 = np.random.choice([120.0, 150.0, 180.0, 210.0])  # Static pitch
        audio = np.zeros_like(t)

        # Flat harmonic series with unnatural high-frequency energy
        for harmonic in range(1, 14):
            amp = (1.0 / np.sqrt(harmonic)) * np.random.uniform(0.7, 1.2)
            audio += amp * np.sin(2 * np.pi * (rigid_f0 * harmonic) * t)

        # Vocoder phase glitches / synthesis discontinuities
        glitch_times = np.random.choice(len(t), size=int(duration * 6), replace=False)
        audio[glitch_times] *= np.random.uniform(-1.5, 1.5, size=len(glitch_times))

        # High-frequency vocoder hiss and quantization artifact
        audio += np.random.normal(0, 0.045, len(t)) * (np.sin(2 * np.pi * 3200 * t) ** 2)

    # Normalize audio amplitude to avoid clipping
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-6:
        audio = audio / max_amp * 0.85
    return audio.astype(np.float32)

def generate_demo_dataset(output_dir=RAW_DATA_DIR, num_per_class=60):
    """
    Generates a starter dataset of genuine and synthetic audio files
    for immediate training and validation without needing large ASVspoof downloads.
    """
    bonafide_dir = Path(output_dir) / "bonafide"
    spoof_dir = Path(output_dir) / "spoof"
    bonafide_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"Generating {num_per_class} genuine (Bona Fide) audio samples...")
    for i in range(num_per_class):
        audio = generate_synthetic_audio(label_type="bonafide")
        filepath = bonafide_dir / f"bonafide_{i:04d}.wav"
        sf.write(str(filepath), audio, SAMPLE_RATE)
        records.append({"file_path": str(filepath), "label": LABEL_BONAFIDE, "class_name": "bonafide"})

    print(f"Generating {num_per_class} spoofed (Deepfake) audio samples...")
    for i in range(num_per_class):
        audio = generate_synthetic_audio(label_type="spoof")
        filepath = spoof_dir / f"spoof_{i:04d}.wav"
        sf.write(str(filepath), audio, SAMPLE_RATE)
        records.append({"file_path": str(filepath), "label": LABEL_SPOOF, "class_name": "spoof"})

    df = pd.DataFrame(records)
    protocol_path = DATA_DIR / "protocol.csv"
    df.to_csv(protocol_path, index=False)
    print(f"Demo dataset created successfully: {len(df)} samples logged to {protocol_path}")
    return df

def parse_asvspoof_protocol(protocol_file_path, audio_dir):
    """
    Parses official ASVspoof 2019 Logical Access protocol text file.
    Protocol columns: [SPEAKER_ID, AUDIO_FILE_NAME, SYSTEM_ID, ATTACK_ID, KEY]
    """
    records = []
    audio_dir = Path(audio_dir)
    
    with open(protocol_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_id = parts[1]
                key = parts[4].lower()
                
                # Check for either .flac or .wav extension
                candidate_paths = [
                    audio_dir / f"{file_id}.flac",
                    audio_dir / f"{file_id}.wav"
                ]
                found_path = next((p for p in candidate_paths if p.exists()), None)
                if found_path:
                    label = LABEL_BONAFIDE if key == "bonafide" else LABEL_SPOOF
                    records.append({
                        "file_path": str(found_path),
                        "label": label,
                        "class_name": key,
                        "attack_id": parts[3]
                    })
    return pd.DataFrame(records)

def load_dataset_dataframe(protocol_csv=None, auto_generate=True):
    """
    Loads dataset DataFrame from CSV or directory, with fallback to demo generator.
    """
    if protocol_csv and Path(protocol_csv).exists():
        return pd.read_csv(protocol_csv)

    default_csv = DATA_DIR / "protocol.csv"
    if default_csv.exists():
        return pd.read_csv(default_csv)

    # Check for files in raw directories
    bonafide_dir = RAW_DATA_DIR / "bonafide"
    spoof_dir = RAW_DATA_DIR / "spoof"
    if bonafide_dir.exists() and spoof_dir.exists():
        bonafide_files = list(bonafide_dir.glob("*.wav")) + list(bonafide_dir.glob("*.flac"))
        spoof_files = list(spoof_dir.glob("*.wav")) + list(spoof_dir.glob("*.flac"))
        if len(bonafide_files) > 0 and len(spoof_files) > 0:
            records = [{"file_path": str(p), "label": LABEL_BONAFIDE, "class_name": "bonafide"} for p in bonafide_files]
            records += [{"file_path": str(p), "label": LABEL_SPOOF, "class_name": "spoof"} for p in spoof_files]
            df = pd.DataFrame(records)
            df.to_csv(default_csv, index=False)
            return df

    if auto_generate:
        return generate_demo_dataset()
    else:
        raise FileNotFoundError("No dataset found and auto_generate is False.")

class AudioDataGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient Keras Sequence generator that streams audio files in batches,
    extracts Mel-spectrogram features on the fly, and yields (X_batch, y_batch).
    """
    def __init__(self, df, batch_size=BATCH_SIZE, shuffle=True):
        self.df = df.copy().reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        X_batch = []
        y_batch = []

        for _, row in batch_df.iterrows():
            try:
                features = extract_features_from_file(row["file_path"])
                X_batch.append(features)
                y_batch.append(row["label"])
            except Exception as e:
                print(f"Warning: Failed to load {row['file_path']}: {e}")
                continue

        if len(X_batch) == 0:
            # Fallback dummy batch if all files in batch failed
            from src.config import INPUT_SHAPE
            return np.zeros((1, *INPUT_SHAPE), dtype=np.float32), np.zeros((1,), dtype=np.float32)

        return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)
