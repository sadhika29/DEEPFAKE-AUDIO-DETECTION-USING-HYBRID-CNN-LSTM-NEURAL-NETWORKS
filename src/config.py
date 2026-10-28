import os
from pathlib import Path

# Project Identity: DFADD (Diffusion and Flow-matching based Audio Deepfake Detection)
PROJECT_NAME = "DFADD: Diffusion & Flow-Matching Audio Deepfake Detection"
PROJECT_GITHUB = "https://github.com/isjwdu/DFADD"
HUGGINGFACE_DATASET = "isjwdu/DFADD"
PAPER_REFERENCE = "DFADD: The Diffusion and Flow-Matching Based Audio Deepfake Dataset (SLT 2024)"

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

# Audio Parameters (16kHz standard for VCTK & DFADD speech)
SAMPLE_RATE = 16000
DURATION = 3.0              # 3.0 second fixed evaluation window
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)  # 48,000 samples

# Feature Extraction Parameters (Log-Mel Spectrogram)
N_MELS = 128                # 128 Mel frequency bands
N_FFT = 1024                # FFT window size
HOP_LENGTH = 512            # Hop length (512 samples -> ~32ms step)
FMIN = 20                   # Min frequency
FMAX = 8000                 # Max frequency (Nyquist limit at 16kHz)

# Derived Feature Dimensions: (128, 94, 1)
TIME_FRAMES = 94
INPUT_SHAPE = (N_MELS, TIME_FRAMES, 1)

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Label Definitions
LABEL_BONAFIDE = 0          # Authentic human speech (VCTK bona fide)
LABEL_SPOOF = 1             # Diffusion / Flow-matching synthetic speech
CLASS_NAMES = ["Bona Fide (Authentic)", "Spoof (Diffusion/Flow-Matching Deepfake)"]

# Modern Diffusion / Flow-Matching TTS Models in DFADD
DFADD_MODELS = [
    "GradTTS",              # Score-based Diffusion Model
    "MatchaTTS",            # Fast Flow-Matching ODE
    "PFlow",                # Flow-matching TTS
    "StyleTTS2",            # Style-based Diffusion / Adversarial
    "NaturalSpeech2"        # Latent Diffusion TTS
]

# Model Checkpoint Paths
BEST_MODEL_PATH = MODELS_DIR / "best_dfadd_cnn_lstm.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_dfadd_cnn_lstm.keras"
