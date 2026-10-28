import streamlit as st
import tempfile
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_audio
from src.preprocess import load_audio, pad_or_trim_audio, compute_mel_spectrogram
from src.config import BEST_MODEL_PATH, FINAL_MODEL_PATH, SAMPLE_RATE, FMIN, FMAX

st.set_page_config(
    page_title="Deepfake Audio Detection | CNN-LSTM",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Deepfake Audio Detection System")
st.markdown(
    """
    **Detecting AI-Generated and Voice-Converted Synthetic Speech using Hybrid CNN-LSTM Neural Networks.**
    """
)

# Sidebar Information
with st.sidebar:
    st.header("⚙️ Model & Pipeline Info")
    model_exists = BEST_MODEL_PATH.exists() or FINAL_MODEL_PATH.exists()
    if model_exists:
        active_model = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH
        st.success(f"Model Active: `{active_model.name}`")
    else:
        st.warning("⚠️ No trained model found yet. Please run training first.")

    st.markdown("---")
    st.subheader("Architecture Details")
    st.markdown(
        """
        - **Preprocessing**: 16 kHz resampled, 3.0s window
        - **Feature**: 128-band Log-Mel Spectrogram
        - **Front-end**: 3-layer 2D-CNN (spectral patterns)
        - **Back-end**: 2-layer Bidirectional LSTM (temporal patterns)
        - **Benchmark**: ASVspoof 2019 Logical Access
        """
    )
    st.markdown("---")
    threshold = st.slider("Spoof Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Main Content
uploaded_file = st.file_uploader("Upload an Audio File for Deepfake Inspection", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔍 Analyze Audio Recording", type="primary", use_container_width=True):
            with st.spinner("Extracting Mel-spectrogram & performing CNN-LSTM inference..."):
                try:
                    res = predict_audio(temp_audio_path)
                    
                    # Apply custom threshold if user adjusted slider
                    p_spoof = res["p_spoof"]
                    p_bona = res["p_bonafide"]
                    is_spoof = p_spoof >= threshold
                    verdict = "Spoof (Deepfake)" if is_spoof else "Bona Fide (Genuine)"
                    conf = (p_spoof if is_spoof else p_bona) * 100.0

                    st.markdown("### Detection Summary")
                    if is_spoof:
                        st.error(f"### Verdict: {verdict}")
                    else:
                        st.success(f"### Verdict: {verdict}")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Confidence", f"{conf:.2f}%")
                    metric_col2.metric("P(Deepfake)", f"{p_spoof:.4f}")
                    metric_col3.metric("P(Genuine)", f"{p_bona:.4f}")

                    st.markdown(f"**Risk Level:** `{res['risk_level']}`")
                    st.progress(float(p_spoof), text=f"Spoof Probability: {p_spoof*100:.1f}%")

                except Exception as e:
                    st.error(f"Error during audio processing: {e}")

    with col2:
        st.markdown("### Acoustic Spectral Analysis")
        try:
            y, sr = load_audio(temp_audio_path)
            y_fixed = pad_or_trim_audio(y)
            mel_spec = compute_mel_spectrogram(y_fixed, sr)[:, :, 0]

            fig, axes = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={'height_ratios': [1, 1.3]})
            t = np.linspace(0, len(y_fixed) / sr, len(y_fixed))
            axes[0].plot(t, y_fixed, color="#2563eb", lw=0.8)
            axes[0].set_title("Waveform (3.0s Normalized)")
            axes[0].set_ylabel("Amplitude")
            axes[0].grid(True, linestyle="--", alpha=0.4)

            img = axes[1].imshow(
                mel_spec,
                origin="lower",
                aspect="auto",
                cmap="magma",
                extent=[0, len(y_fixed) / sr, FMIN, FMAX]
            )
            axes[1].set_title("Log-Mel Spectrogram")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Frequency (Hz)")
            fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.info("Spectrogram visualizer ready upon file upload.")
