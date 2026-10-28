import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Audio Parameters
SAMPLE_RATE = 16000          # Standard 16kHz for speech recognition & ASVspoof
DURATION = 3.0              # Fixed duration in seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)  # 48,000 samples

# Feature Extraction Parameters (Mel-Spectrogram)
N_MELS = 128                # Number of Mel frequency bands
N_FFT = 1024                # Length of FFT window
HOP_LENGTH = 512            # Number of samples between successive frames
FMIN = 20                   # Minimum frequency
FMAX = 8000                 # Maximum frequency (Nyquist frequency for 16kHz)

# Derived Feature Dimensions
# Time frames: ceil(NUM_SAMPLES / HOP_LENGTH) approx 94
TIME_FRAMES = 94
INPUT_SHAPE = (N_MELS, TIME_FRAMES, 1)  # (128, 94, 1)

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Label Definitions
LABEL_BONAFIDE = 0          # Genuine human voice
LABEL_SPOOF = 1             # Synthetic or voice converted (deepfake)
CLASS_NAMES = ["Bona Fide (Genuine)", "Spoof (Deepfake)"]

# Model Checkpoints
BEST_MODEL_PATH = MODELS_DIR / "best_cnn_lstm_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_cnn_lstm_model.keras"
