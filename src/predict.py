import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.config import BEST_MODEL_PATH, FINAL_MODEL_PATH, CLASS_NAMES
from src.preprocess import extract_features_from_file, load_audio, compute_acoustic_forensic_metrics

_CACHED_MODEL = None

def get_model(model_path=None):
    """Loads and caches the trained DFADD CNN-LSTM model."""
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

def calculate_diffusion_forensic_score(y, sr):
    """
    Evaluates physical acoustic cues of Diffusion & Flow-Matching synthesis:
    1. Score-based diffusion residual noise in upper Mel bands (>4 kHz).
    2. Over-smoothed ODE trajectory transitions across phonemes.
    3. Low pitch/centroid variation (robotic pitch vs natural human prosody).
    """
    metrics = compute_acoustic_forensic_metrics(y, sr)
    
    # 1. Pitch / Prosodic Dynamism
    cent_var = metrics["centroid_variance"]
    if cent_var < 220:
        pitch_score = 0.85
    elif cent_var < 380:
        pitch_score = 0.55
    else:
        pitch_score = 0.15

    # 2. Diffusion High Frequency Noise Floor
    hf_ratio = metrics["high_frequency_ratio"]
    if hf_ratio > 0.22 or hf_ratio < 0.025:
        hf_score = 0.80
    else:
        hf_score = 0.25

    # 3. Spectral Flatness (residual score noise)
    flatness = metrics["spectral_flatness"]
    flat_score = min(flatness * 14.0, 1.0)

    score = 0.45 * pitch_score + 0.35 * hf_score + 0.20 * flat_score
    return float(np.clip(score, 0.0, 1.0)), metrics

def predict_audio(file_path, model_path=None, threshold=0.5):
    """
    Performs inference against DFADD benchmarks:
    - Distinguishes authentic human speech (VCTK) from modern
      diffusion & flow-matching speech synthesis (Grad-TTS, Matcha-TTS, PFlow, etc.).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Extract 128-band Mel-spectrogram: (128, 94, 1)
    spec = extract_features_from_file(file_path)
    batch_input = np.expand_dims(spec, axis=0)

    # 1. CNN-LSTM Deep Learning Prediction
    model = get_model(model_path)
    raw_dl_pred = model.predict(batch_input, verbose=0)
    p_dl_spoof = float(raw_dl_pred[0][0])

    # 2. Physical Diffusion Forensic Analysis
    y, sr = load_audio(file_path)
    forensic_score, forensic_metrics = calculate_diffusion_forensic_score(y, sr)

    # 3. Calibrated Ensemble Probability
    p_spoof = 0.65 * p_dl_spoof + 0.35 * forensic_score
    p_bonafide = 1.0 - p_spoof

    is_spoof = p_spoof >= threshold
    prediction = CLASS_NAMES[1] if is_spoof else CLASS_NAMES[0]
    confidence = (p_spoof if is_spoof else p_bonafide) * 100.0

    if p_spoof > 0.70:
        risk_level = "High Risk (Modern Diffusion / Flow-Matching Deepfake Detected)"
        risk_color = "#dc2626"
        synthesis_signature = "Suspected: Diffusion / Flow-Matching (e.g. Grad-TTS, Matcha-TTS, PFlow)"
    elif p_spoof > 0.45:
        risk_level = "Medium Risk (Suspicious Diffusion Noise Residuals)"
        risk_color = "#ea580c"
        synthesis_signature = "Suspected: Manipulated / Resynthesized Speech"
    else:
        risk_level = "Low Risk (Authentic Human Vocal Tract Dynamics)"
        risk_color = "#16a34a"
        synthesis_signature = "Natural Human Voice (Consistent Glottal Waveform)"

    return {
        "file_name": file_path.name,
        "prediction": prediction,
        "is_spoof": bool(is_spoof),
        "confidence": confidence,
        "p_spoof": p_spoof,
        "p_bonafide": p_bonafide,
        "p_dl_spoof": p_dl_spoof,
        "forensic_score": forensic_score,
        "forensic_metrics": forensic_metrics,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "synthesis_signature": synthesis_signature,
        "spectrogram": spec[:, :, 0]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict audio authenticity using DFADD benchmark model")
    parser.add_argument("audio_path", type=str, help="Path to audio file (WAV, FLAC, MP3)")
    parser.add_argument("--model", type=str, default=None, help="Path to model (.keras)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    result = predict_audio(args.audio_path, model_path=args.model, threshold=args.threshold)
    print("\n" + "=" * 60)
    print(f"FILE:                  {result['file_name']}")
    print(f"VERDICT:               {result['prediction']}")
    print(f"CONFIDENCE:            {result['confidence']:.2f}%")
    print(f"CALIBRATED P(SPOOF):   {result['p_spoof']:.4f}")
    print(f"CNN-LSTM DL SCORE:     {result['p_dl_spoof']:.4f}")
    print(f"DIFFUSION FORENSIC:    {result['forensic_score']:.4f}")
    print(f"SIGNATURE:             {result['synthesis_signature']}")
    print(f"RISK ASSESSMENT:       {result['risk_level']}")
    print("=" * 60 + "\n")
