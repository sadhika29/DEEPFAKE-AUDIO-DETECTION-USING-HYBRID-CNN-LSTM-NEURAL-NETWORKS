import streamlit as st
import tempfile
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_audio
from src.preprocess import load_audio, pad_or_trim_audio, compute_mel_spectrogram
from src.config import BEST_MODEL_PATH, FINAL_MODEL_PATH, SAMPLE_RATE, FMIN, FMAX, PROJECT_GITHUB, HUGGINGFACE_DATASET

st.set_page_config(
    page_title="DFADD: Diffusion & Flow-Matching Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ DFADD Audio Deepfake Detection System")
st.markdown(
    f"""
    **State-of-the-Art Anti-Spoofing against Diffusion & Flow-Matching Speech Synthesis**  
    *Benchmark: [{HUGGINGFACE_DATASET} on Hugging Face](https://huggingface.co/datasets/{HUGGINGFACE_DATASET}) | [GitHub Repository]({PROJECT_GITHUB})*
    """
)

with st.sidebar:
    st.header("⚙️ Model & Benchmark Info")
    model_exists = BEST_MODEL_PATH.exists() or FINAL_MODEL_PATH.exists()
    if model_exists:
        active_model = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH
        st.success(f"Model Active: `{active_model.name}`")
    else:
        st.info("ℹ️ Running in heuristic & uncalibrated test mode until training finishes.")

    st.markdown("---")
    st.subheader("Targeted TTS Architectures")
    st.markdown(
        """
        - **Grad-TTS**: Score-based diffusion TTS
        - **Matcha-TTS**: Fast optimal-transport flow matching
        - **PFlow**: Continuous normalizing flow TTS
        - **NaturalSpeech2**: Latent diffusion zero-shot voice
        - **StyleTTS2**: Diffusion-guided style modeling
        """
    )
    st.markdown("---")
    threshold = st.slider("Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

uploaded_file = st.file_uploader("Upload an Audio File for Deepfake Inspection", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔍 Run DFADD Deepfake Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing spectral diffusion residues & temporal transitions..."):
                try:
                    res = predict_audio(temp_audio_path, threshold=threshold)
                    
                    p_spoof = res["p_spoof"]
                    p_bona = res["p_bonafide"]
                    is_spoof = p_spoof >= threshold
                    verdict = res["prediction"]
                    conf = res["confidence"]

                    st.markdown("### Detection Summary")
                    if is_spoof:
                        st.error(f"### Verdict: {verdict}")
                    else:
                        st.success(f"### Verdict: {verdict}")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Confidence", f"{conf:.2f}%")
                    metric_col2.metric("P(Deepfake)", f"{p_spoof:.4f}")
                    metric_col3.metric("P(Authentic)", f"{p_bona:.4f}")

                    st.markdown(f"**Acoustic Fingerprint:** `{res['synthesis_signature']}`")
                    st.markdown(f"**Risk Level:** `{res['risk_level']}`")
                    st.progress(float(p_spoof), text=f"Deepfake Probability: {p_spoof*100:.1f}%")

                    with st.expander("Detailed Metric Breakdown"):
                        st.write(f"- **CNN-LSTM Neural Network Score:** {res['p_dl_spoof'] * 100:.1f}%")
                        st.write(f"- **Diffusion / Flow-Matching Forensic Score:** {res['forensic_score'] * 100:.1f}%")
                        st.json(res["forensic_metrics"])

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
