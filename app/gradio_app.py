import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_audio, get_model
from src.preprocess import load_audio, pad_or_trim_audio, compute_mel_spectrogram
from src.config import BEST_MODEL_PATH, FINAL_MODEL_PATH, SAMPLE_RATE, FMIN, FMAX

def analyze_audio_file(audio_path):
    """
    Callback function for Gradio interface.
    Accepts uploaded or recorded audio path, runs CNN-LSTM model,
    and returns predictions and feature visualization figure.
    """
    if audio_path is None:
        return "Please upload or record an audio file.", None, None

    try:
        # Run inference
        res = predict_audio(audio_path)
        
        # Format prediction text
        pred_label = res["prediction"]
        conf = res["confidence"]
        p_spoof = res["p_spoof"]
        p_bona = res["p_bonafide"]
        risk = res["risk_level"]

        result_html = f"""
        <div style="padding: 16px; border-radius: 8px; background-color: #f8fafc; border: 2px solid {res['risk_color']};">
            <h2 style="color: {res['risk_color']}; margin-top: 0;">Verdict: {pred_label}</h2>
            <p style="font-size: 16px; margin: 4px 0;"><strong>Confidence:</strong> {conf:.2f}%</p>
            <p style="font-size: 16px; margin: 4px 0;"><strong>Risk Assessment:</strong> {risk}</p>
            <div style="margin-top: 12px; background: #e2e8f0; border-radius: 4px; overflow: hidden; height: 16px;">
                <div style="width: {p_spoof * 100:.1f}%; background-color: {res['risk_color']}; height: 100%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 4px; color: #64748b;">
                <span>Genuine Voice ({p_bona * 100:.1f}%)</span>
                <span>Deepfake Voice ({p_spoof * 100:.1f}%)</span>
            </div>
        </div>
        """

        # Generate Waveform & Spectrogram Plot
        y, sr = load_audio(audio_path)
        y_fixed = pad_or_trim_audio(y)
        mel_spec = compute_mel_spectrogram(y_fixed, sr)[:, :, 0]

        fig, axes = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={'height_ratios': [1, 1.4]})
        
        # Waveform
        t = np.linspace(0, len(y_fixed) / sr, len(y_fixed))
        axes[0].plot(t, y_fixed, color="#2563eb", lw=0.8)
        axes[0].set_title("Input Audio Waveform (3.0s Normalized)", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("Amplitude", fontsize=9)
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # Spectrogram
        img = axes[1].imshow(
            mel_spec,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, len(y_fixed) / sr, FMIN, FMAX]
        )
        axes[1].set_title("Extracted Log-Mel Spectrogram (Frequency vs. Time)", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Time (seconds)", fontsize=9)
        axes[1].set_ylabel("Frequency (Hz)", fontsize=9)
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

        plt.tight_layout()

        conf_dict = {
            "Bona Fide (Genuine)": float(p_bona),
            "Spoof (Deepfake)": float(p_spoof)
        }

        return result_html, conf_dict, fig

    except Exception as e:
        error_html = f"<div style='color: red; padding: 10px;'>Error processing audio: {str(e)}</div>"
        return error_html, None, None

def launch_gradio_app(share=False, server_port=7860):
    """Launches Gradio application."""
    custom_css = """
    .gradio-container { max-width: 960px !important; margin: auto; }
    """

    with gr.Blocks(title="Deepfake Audio Detector | CNN-LSTM") as demo:
        gr.Markdown(
            """
            # 🎙️ Deepfake Audio Detection System
            ### Hybrid CNN-LSTM Neural Network for Genuine vs. Synthetic Speech Classification
            Upload a WAV/MP3 recording or use your microphone to test whether an audio sample is genuine human speech or an AI-synthesized deepfake.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Upload or Record Speech Audio (.wav, .mp3, .flac)"
                )
                analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
                gr.Markdown(
                    """
                    **Note:** The model evaluates spectral anomalies, harmonic distortion, and temporal phase transitions.
                    Audio is automatically resampled to 16 kHz and adjusted to 3.0 seconds.
                    """
                )

            with gr.Column(scale=1):
                verdict_output = gr.HTML(label="Classification Verdict")
                label_output = gr.Label(label="Confidence Distribution", num_top_classes=2)
                plot_output = gr.Plot(label="Acoustic Analysis (Waveform & Mel-Spectrogram)")

        analyze_btn.click(
            fn=analyze_audio_file,
            inputs=[audio_input],
            outputs=[verdict_output, label_output, plot_output]
        )

    print("Launching Gradio interface...")
    demo.launch(share=share, server_port=server_port)

if __name__ == "__main__":
    launch_gradio_app(share=False)
