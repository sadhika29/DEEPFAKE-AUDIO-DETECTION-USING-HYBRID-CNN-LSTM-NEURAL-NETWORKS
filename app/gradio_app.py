import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_audio
from src.preprocess import load_audio, pad_or_trim_audio, compute_mel_spectrogram
from src.config import SAMPLE_RATE, FMIN, FMAX, PROJECT_NAME, PROJECT_GITHUB, HUGGINGFACE_DATASET

def analyze_audio_file(audio_path, threshold=0.5):
    """
    Inference callback for Gradio interface.
    """
    if audio_path is None:
        return "<div style='color: orange; font-weight: bold;'>Please upload or record an audio file first.</div>", None, None

    try:
        res = predict_audio(audio_path, threshold=threshold)
        
        pred_label = res["prediction"]
        conf = res["confidence"]
        p_spoof = res["p_spoof"]
        p_bona = res["p_bonafide"]
        risk = res["risk_level"]
        dl_score = res["p_dl_spoof"]
        forensic_score = res["forensic_score"]
        signature = res["synthesis_signature"]

        result_html = f"""
        <div style="padding: 18px; border-radius: 10px; background-color: #f8fafc; border: 2.5px solid {res['risk_color']};">
            <h2 style="color: {res['risk_color']}; margin-top: 0; margin-bottom: 6px;">Verdict: {pred_label}</h2>
            <p style="font-size: 16px; margin: 4px 0;"><strong>Confidence:</strong> {conf:.2f}%</p>
            <p style="font-size: 14px; margin: 4px 0;"><strong>Risk Assessment:</strong> {risk}</p>
            <p style="font-size: 13px; color: #334155; margin: 4px 0;"><strong>Acoustic Fingerprint:</strong> {signature}</p>
            
            <div style="margin: 12px 0;">
                <div style="font-size: 13px; color: #475569; margin-bottom: 4px;"><strong>Deepfake Likelihood:</strong> {p_spoof * 100:.1f}%</div>
                <div style="background: #e2e8f0; border-radius: 5px; overflow: hidden; height: 18px;">
                    <div style="width: {p_spoof * 100:.1f}%; background-color: {res['risk_color']}; height: 100%;"></div>
                </div>
            </div>

            <div style="background: #ffffff; padding: 10px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 13px; line-height: 1.5;">
                <div>🧠 <strong>CNN-LSTM Deep Learning Score:</strong> {dl_score * 100:.1f}%</div>
                <div>🌊 <strong>Diffusion/Flow-Matching Score:</strong> {forensic_score * 100:.1f}%</div>
                <div style="margin-top: 4px; font-size: 11px; color: #64748b;">Trained on DFADD (IEEE SLT 2024 / isjwdu/DFADD)</div>
            </div>
        </div>
        """

        y, sr = load_audio(audio_path)
        y_fixed = pad_or_trim_audio(y)
        mel_spec = compute_mel_spectrogram(y_fixed, sr)[:, :, 0]

        fig, axes = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={'height_ratios': [1, 1.4]})
        
        t = np.linspace(0, len(y_fixed) / sr, len(y_fixed))
        axes[0].plot(t, y_fixed, color="#2563eb", lw=0.8)
        axes[0].set_title("Input Audio Waveform (3.0s Normalized)", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("Amplitude", fontsize=9)
        axes[0].grid(True, linestyle="--", alpha=0.4)

        img = axes[1].imshow(
            mel_spec,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, len(y_fixed) / sr, FMIN, FMAX]
        )
        axes[1].set_title("Log-Mel Spectrogram (Frequency vs. Time)", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Time (seconds)", fontsize=9)
        axes[1].set_ylabel("Frequency (Hz)", fontsize=9)
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

        plt.tight_layout()

        conf_dict = {
            "Bona Fide (Authentic Human)": float(p_bona),
            "Spoof (Diffusion/Flow-Matching)": float(p_spoof)
        }

        return result_html, conf_dict, fig

    except Exception as e:
        error_html = f"<div style='color: red; padding: 10px;'>Error processing audio: {str(e)}</div>"
        return error_html, None, None

def launch_gradio_app(share=False, server_port=7860):
    with gr.Blocks(title="DFADD Audio Deepfake Detector | CNN-LSTM") as demo:
        gr.Markdown(
            f"""
            # 🎙️ DFADD Audio Deepfake Detection System
            ### Modern Anti-Spoofing against Diffusion & Flow-Matching Speech Synthesis (IEEE SLT 2024)
            **Dataset & Benchmark**: [{HUGGINGFACE_DATASET} on HuggingFace](https://huggingface.co/datasets/{HUGGINGFACE_DATASET}) | [GitHub Repository]({PROJECT_GITHUB})
            
            Detect whether an audio recording is **authentic human speech (VCTK)** or synthesized by modern state-of-the-art **Diffusion & Flow-Matching models** (Grad-TTS, Matcha-TTS, PFlow, etc.).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Upload or Record Speech Audio (.wav, .mp3, .flac)"
                )
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Decision Threshold (Default: 0.5)"
                )
                analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
                gr.Markdown(
                    """
                    **Detection Capabilities:**
                    - **Score-based Diffusion Noise**: Detects residual high-frequency denoising noise.
                    - **Flow-Matching Discretization**: Detects ODE vector-field step transitions.
                    - **Prosodic Drift**: Verifies natural human glottal and vocal tract dynamics.
                    """
                )

            with gr.Column(scale=1):
                verdict_output = gr.HTML(label="Classification Verdict")
                label_output = gr.Label(label="Confidence Distribution", num_top_classes=2)
                plot_output = gr.Plot(label="Spectrogram Inspection")

        analyze_btn.click(
            fn=analyze_audio_file,
            inputs=[audio_input, threshold_slider],
            outputs=[verdict_output, label_output, plot_output]
        )

    print("Launching DFADD Gradio interface...")
    demo.launch(share=share, server_port=server_port)

if __name__ == "__main__":
    launch_gradio_app(share=False)
