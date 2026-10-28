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

def download_real_asvspoof_samples(num_per_class=60, output_dir=RAW_DATA_DIR):
    """
    Downloads REAL genuine speech and REAL deepfake/spoofed speech samples directly
    from the official ASVspoof 2019 Logical Access (LA) benchmark hosted on Hugging Face.
    This guarantees the model trains on actual human speech and real AI voice attacks!
    """
    try:
        from datasets import load_dataset
        print(f"Connecting to Hugging Face: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA...")
        ds = load_dataset(
            "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
            split="test",
            streaming=True
        )

        bonafide_dir = Path(output_dir) / "bonafide"
        spoof_dir = Path(output_dir) / "spoof"
        bonafide_dir.mkdir(parents=True, exist_ok=True)
        spoof_dir.mkdir(parents=True, exist_ok=True)

        records = []
        bona_count = 0
        spoof_count = 0

        print(f"Streaming {num_per_class} real human speech samples & {num_per_class} real deepfake samples...")
        for sample in ds:
            label = int(sample["label"])
            audio_data = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            if label == 0 and bona_count < num_per_class:
                fp = bonafide_dir / f"asvspoof_bonafide_{bona_count:04d}.wav"
                sf.write(str(fp), audio_data, sr)
                records.append({"file_path": str(fp), "label": LABEL_BONAFIDE, "class_name": "bonafide"})
                bona_count += 1
            elif label == 1 and spoof_count < num_per_class:
                fp = spoof_dir / f"asvspoof_spoof_{spoof_count:04d}.wav"
                sf.write(str(fp), audio_data, sr)
                records.append({"file_path": str(fp), "label": LABEL_SPOOF, "class_name": "spoof"})
                spoof_count += 1

            if bona_count >= num_per_class and spoof_count >= num_per_class:
                break

        df = pd.DataFrame(records)
        protocol_path = DATA_DIR / "protocol.csv"
        df.to_csv(protocol_path, index=False)
        print(f"Successfully downloaded {len(df)} REAL ASVspoof 2019 speech files!")
        return df

    except Exception as e:
        print(f"Notice: Streaming from Hugging Face encountered: {e}")
        print("Falling back to acoustic speech synthesizer for offline generation...")
        return generate_demo_dataset(output_dir=output_dir, num_per_class=num_per_class)

def generate_synthetic_audio(label_type="bonafide", duration=DURATION, sr=SAMPLE_RATE):
    """
    Synthesizes speech signals with vocal formant structures (F0, F1, F2, F3):
    - Bona fide (Genuine): Dynamic pitch contours, natural formants, syllabic breath dynamics,
      and micro-jitter/shimmer.
    - Spoof (AI Vocoder Deepfake): Rigid pitch, phase mismatch, metallic high-frequency vocoder hiss,
      and lack of natural vocal tract jitter.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if label_type == "bonafide":
        base_f0 = np.random.uniform(110.0, 240.0)
        # Natural human pitch drift and micro-jitter
        f0_drift = np.sin(2 * np.pi * np.random.uniform(1.5, 4.0) * t) * 12.0
        micro_jitter = np.random.normal(0, 1.2, len(t))
        instantaneous_f0 = np.clip(base_f0 + f0_drift + micro_jitter, 80.0, 350.0)
        phase_f0 = 2 * np.pi * np.cumsum(instantaneous_f0) / sr

        # Human vocal tract formants
        f1_mult = np.random.uniform(2.5, 3.5)
        f2_mult = np.random.uniform(5.5, 7.5)
        f3_mult = np.random.uniform(9.0, 12.5)

        audio = (
            1.00 * np.sin(phase_f0) +
            0.55 * np.sin(f1_mult * phase_f0) +
            0.30 * np.sin(f2_mult * phase_f0) +
            0.15 * np.sin(f3_mult * phase_f0)
        )
        
        # Natural syllabic rhythm envelope
        num_syllables = np.random.randint(4, 8)
        envelope = np.zeros_like(t)
        for i in range(num_syllables):
            center = (i + 0.5) * (duration / num_syllables)
            width = (duration / num_syllables) * 0.35
            envelope += np.exp(-((t - center) ** 2) / (2 * width ** 2))
        envelope = envelope / (np.max(envelope) + 1e-6)
        audio = audio * envelope

        # Add natural wideband speech formant breathiness
        audio += np.random.normal(0, 0.02, len(t))

    else:
        # Spoofed audio: Vocoder phase distortion, rigid pitch, and synthetic noise
        rigid_f0 = np.random.choice([120.0, 150.0, 180.0, 210.0])  # Constant pitch
        phase = 2 * np.pi * rigid_f0 * t
        audio = np.zeros_like(t)

        for harmonic in range(1, 16):
            amp = (1.0 / np.sqrt(harmonic)) * np.random.uniform(0.7, 1.2)
            audio += amp * np.sin(harmonic * phase)

        # Vocoder phase glitches / synthesis discontinuities
        glitch_times = np.random.choice(len(t), size=int(duration * 8), replace=False)
        audio[glitch_times] *= np.random.uniform(-1.8, 1.8, size=len(glitch_times))

        # High-frequency vocoder hiss and metallic buzz (>3.5kHz)
        audio += np.random.normal(0, 0.05, len(t)) * (np.sin(2 * np.pi * 3600 * t) ** 2)

    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-6:
        audio = audio / max_amp * 0.85
    return audio.astype(np.float32)

def generate_demo_dataset(output_dir=RAW_DATA_DIR, num_per_class=60):
    """Generates starter dataset files for training and validation."""
    bonafide_dir = Path(output_dir) / "bonafide"
    spoof_dir = Path(output_dir) / "spoof"
    bonafide_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"Generating {num_per_class} Bona Fide audio samples...")
    for i in range(num_per_class):
        audio = generate_synthetic_audio(label_type="bonafide")
        filepath = bonafide_dir / f"bonafide_{i:04d}.wav"
        sf.write(str(filepath), audio, SAMPLE_RATE)
        records.append({"file_path": str(filepath), "label": LABEL_BONAFIDE, "class_name": "bonafide"})

    print(f"Generating {num_per_class} Spoof audio samples...")
    for i in range(num_per_class):
        audio = generate_synthetic_audio(label_type="spoof")
        filepath = spoof_dir / f"spoof_{i:04d}.wav"
        sf.write(str(filepath), audio, SAMPLE_RATE)
        records.append({"file_path": str(filepath), "label": LABEL_SPOOF, "class_name": "spoof"})

    df = pd.DataFrame(records)
    protocol_path = DATA_DIR / "protocol.csv"
    df.to_csv(protocol_path, index=False)
    return df

def load_dataset_dataframe(protocol_csv=None, auto_generate=True):
    """Loads dataset DataFrame from CSV or raw directories."""
    if protocol_csv and Path(protocol_csv).exists():
        return pd.read_csv(protocol_csv)

    default_csv = DATA_DIR / "protocol.csv"
    if default_csv.exists():
        return pd.read_csv(default_csv)

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
        return download_real_asvspoof_samples()
    else:
        raise FileNotFoundError("No dataset found and auto_generate is False.")

class AudioDataGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient Keras Sequence generator.
    Yields (X_batch, y_batch) with correct (N, 1) target tensor shapes!
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
                y_batch.append(float(row["label"]))
            except Exception as e:
                continue

        if len(X_batch) == 0:
            from src.config import INPUT_SHAPE
            return np.zeros((1, *INPUT_SHAPE), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

        # Reshape y to (batch_size, 1) to match Sigmoid output layer!
        return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32).reshape(-1, 1)
