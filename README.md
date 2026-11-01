# Deepfake Audio Detection with CNN-LSTM

A Python project for detecting synthetic speech with a hybrid convolutional neural network and bidirectional LSTM model. It extracts log-Mel spectrogram features, trains on streamed DFADD data, evaluates the trained model, and provides Gradio and Streamlit interfaces for prediction.

## Features

- Hybrid CNN-BiLSTM classifier for audio authenticity detection
- Log-Mel spectrogram preprocessing
- Dataset streaming and protocol generation
- Training, evaluation, and single-file prediction commands
- Gradio and Streamlit user interfaces

## Setup

```bash
git clone https://github.com/veeruveerendra05/deepfake-audio-detection.git
cd deepfake-audio-detection

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Usage

### Train

```bash
python -m src.train --epochs 20 --batch-size 16
```

### Evaluate

```bash
python -m src.evaluate
```

### Predict from an audio file

```bash
python -m src.predict path/to/sample_audio.wav
```

### Launch an interface

```bash
python app/gradio_app.py
streamlit run app/streamlit_app.py
```

## Project structure

```
deepfake-audio-detection/
├── app/           # Gradio and Streamlit interfaces
├── data/          # Generated protocol data
├── notebooks/     # Exploration notebook
├── reports/       # Project report and presentation notes
├── src/           # Training, evaluation, prediction, and preprocessing code
└── requirements.txt
```

## Notes

Training creates model checkpoints in `models/` and evaluation writes plots to `reports/`. The dataset is fetched by the project code, so an internet connection is required for first-time data access.
