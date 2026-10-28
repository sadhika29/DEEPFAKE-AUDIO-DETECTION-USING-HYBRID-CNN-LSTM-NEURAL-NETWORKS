import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.config import BEST_MODEL_PATH, FINAL_MODEL_PATH, CLASS_NAMES
from src.preprocess import extract_features_from_file

_CACHED_MODEL = None

def get_model(model_path=None):
    """Loads and caches the model in memory for fast inference."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and model_path is None:
        return _CACHED_MODEL

    if model_path is None:
        model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    model = tf.keras.models.load_model(str(model_path))
    if model_path == (BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH):
        _CACHED_MODEL = model
    return model

def predict_audio(file_path, model_path=None):
    """
    Runs end-to-end inference on a single audio recording:
    1. Extracts normalized Mel-Spectrogram features
    2. Passes features through the Hybrid CNN-LSTM network
    3. Returns classification decision, confidence percentage, and risk level.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Extract spectrogram: shape (128, 94, 1)
    spec = extract_features_from_file(file_path)
    
    # Expand batch dimension: shape (1, 128, 94, 1)
    batch_input = np.expand_dims(spec, axis=0)

    # Predict
    model = get_model(model_path)
    p_spoof = float(model.predict(batch_input, verbose=0)[0][0])
    p_bonafide = 1.0 - p_spoof

    is_spoof = p_spoof >= 0.5
    prediction = CLASS_NAMES[1] if is_spoof else CLASS_NAMES[0]
    confidence = (p_spoof if is_spoof else p_bonafide) * 100.0

    # Risk Assessment Level
    if p_spoof > 0.75:
        risk_level = "High (Synthetic Voice Detected)"
        risk_color = "#dc2626"
    elif p_spoof > 0.40:
        risk_level = "Medium (Suspicious Acoustic Artifacts)"
        risk_color = "#ea580c"
    else:
        risk_level = "Low (Natural Vocal Dynamics)"
        risk_color = "#16a34a"

    return {
        "file_name": file_path.name,
        "prediction": prediction,
        "is_spoof": bool(is_spoof),
        "confidence": confidence,
        "p_spoof": p_spoof,
        "p_bonafide": p_bonafide,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "spectrogram": spec[:, :, 0]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single audio file authenticity")
    parser.add_argument("audio_path", type=str, help="Path to audio file (WAV, FLAC, MP3)")
    parser.add_argument("--model", type=str, default=None, help="Path to model (.keras)")
    args = parser.parse_args()

    result = predict_audio(args.audio_path, model_path=args.model)
    print("\n" + "=" * 50)
    print(f"FILE:          {result['file_name']}")
    print(f"PREDICTION:    {result['prediction']}")
    print(f"CONFIDENCE:    {result['confidence']:.2f}%")
    print(f"SPOOF PROB:    {result['p_spoof']:.4f}")
    print(f"BONAFIDE PROB: {result['p_bonafide']:.4f}")
    print(f"RISK LEVEL:    {result['risk_level']}")
    print("=" * 50 + "\n")
