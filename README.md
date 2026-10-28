# 🎙️ Deepfake Audio Detection Using Hybrid CNN-LSTM Neural Networks

A complete end-to-end Deep Learning system designed to detect AI-generated synthetic and voice-converted speech recordings by analyzing **Log-Mel Spectrograms** through a **Hybrid CNN-LSTM Neural Network**.

---

## 🚀 Quick Start: Google Colab (Recommended)

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Click **Upload** and select `notebooks/Deepfake_Audio_Detection_CNN_LSTM.ipynb`.
3. Enable GPU acceleration:
   - Navigate to **Runtime** → **Change runtime type** → select **T4 GPU** (or CPU).
4. Run all cells sequentially (**Runtime** → **Run all**).
5. The notebook will:
   - Install required packages (`librosa`, `soundfile`, `tensorflow`, `gradio`, etc.)
   - Generate an immediate demo dataset with genuine vocal formants and synthetic vocoder noise
   - Extract and visualize Log-Mel Spectrograms
   - Build and train the Hybrid CNN-LSTM model with early stopping
   - Evaluate model metrics (Accuracy, Precision, Recall, F1, ROC-AUC, EER)
   - Launch an interactive **Gradio web interface** directly in the Colab cell with a public share link!

---

## 💻 Local Setup & Execution

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: Virtual environment (`venv` or `conda`)

```bash
# Clone or navigate to the project directory
cd deepfake-audio-detection

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Model
```bash
# Train model (automatically creates starter dataset if none exists)
python -m src.train --epochs 20 --batch-size 16
```

### 3. Evaluate Model
```bash
# Evaluate model performance on validation data
python -m src.evaluate
```
This generates:
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/score_distribution.png`
- `reports/evaluation_metrics.json`

### 4. Run Single Audio Inference
```bash
python -m src.predict path/to/sample_audio.wav
```

### 5. Launch Web Interfaces
- **Gradio App (Colab & Web Share):**
  ```bash
  python app/gradio_app.py
  ```
- **Streamlit Dashboard:**
  ```bash
  streamlit run app/streamlit_app.py
  ```

---

## 📁 Project Structure

```
deepfake-audio-detection/
├── notebooks/
│   └── Deepfake_Audio_Detection_CNN_LSTM.ipynb   # Complete, turnkey Google Colab notebook
├── src/
│   ├── __init__.py
│   ├── config.py                                 # Audio sample rates, FFT, Mel bands, paths
│   ├── preprocess.py                             # Audio padding/trimming & Log-Mel extraction
│   ├── dataset.py                                # ASVspoof parser & demo data generator
│   ├── model.py                                  # Hybrid CNN-LSTM Keras neural network
│   ├── train.py                                  # Training loop with callbacks & checkpoints
│   ├── evaluate.py                               # Metrics, Confusion Matrix, ROC-AUC, EER
│   ├── predict.py                                # Single audio file inference pipeline
│   └── download_asvspoof.py                      # ASVspoof 2019 dataset helper utility
├── app/
│   ├── gradio_app.py                             # Gradio interface (Microphone + Upload + Plots)
│   └── streamlit_app.py                          # Streamlit dashboard interface
├── data/
│   ├── raw/                                      # Genuine & spoofed audio WAV/FLAC files
│   └── protocol.csv                              # Dataset manifest & labels
├── models/                                       # Saved model weights (.keras)
├── reports/
│   ├── project_report.md                         # Full academic project report
│   ├── presentation_slides.md                    # 10-slide PowerPoint presentation outline
│   └── *.png                                     # Generated evaluation & history charts
├── requirements.txt                              # Python package dependencies
└── README.md                                     # Project guide
```

---

## 🧠 Model Architecture

The system uses a **CRNN (Convolutional Recurrent Neural Network)** design:
1. **Input**: `(128, 94, 1)` Log-Mel Spectrogram representing 3 seconds of 16 kHz audio.
2. **CNN Front-End**: 3 Conv2D blocks with Batch Normalization, ReLU, and MaxPooling to capture fine-grained spatial and spectral anomalies.
3. **Reshape & Permute**: Converts spatial representation to a sequence of `(47 time frames, 2048 features)`.
4. **BiLSTM Back-End**: 2 stacked Bidirectional LSTM layers (64 & 32 units) to track temporal phase transitions and vocal tract dynamics across consecutive speech frames.
5. **Classification Head**: Dense(64, ReLU) $\to$ Dropout(0.4) $\to$ Dense(1, Sigmoid) outputting $P(\text{Spoof})$.

---

## 📊 Evaluation Metrics

Standard biometric speaker verification metrics are computed:
- **Accuracy, Precision, Recall, and F1-Score**
- **Receiver Operating Characteristic (ROC-AUC)**
- **Equal Error Rate (EER)**: The operating threshold where False Acceptance Rate equals False Rejection Rate.

---

## 📜 Deliverables Included
- **Google Colab Notebook**: `notebooks/Deepfake_Audio_Detection_CNN_LSTM.ipynb`
- **Academic Project Report**: `reports/project_report.md`
- **Slide Presentation Outline**: `reports/presentation_slides.md`
