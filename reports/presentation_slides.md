# PRESENTATION SLIDES OUTLINE & SPEAKER SCRIPT
## Deepfake Audio Detection Using Hybrid CNN-LSTM Neural Networks

---

### SLIDE 1: Title Slide
- **Title:** Deepfake Audio Detection Using Hybrid CNN-LSTM Neural Networks
- **Subtitle:** Distinguishing Genuine Human Speech from Synthetic Voice Clones
- **Presenter:** Sadhika & Project Team
- **Technologies:** Python, TensorFlow / Keras, Librosa, Scikit-learn, Gradio / Streamlit
> **Speaker Notes:**  
> "Good morning/afternoon everyone. Today, we are presenting our project on Deepfake Audio Detection Using Hybrid CNN-LSTM Neural Networks. As generative AI enables the creation of hyper-realistic cloned voices, detecting acoustic manipulation has become an urgent priority for biometric security."

---

### SLIDE 2: The Problem: The Threat of Audio Deepfakes
- **Voice Cloning Proliferation:** Generative AI tools (neural vocoders, diffusion models) can clone voices with mere seconds of audio.
- **Critical Attack Vectors:**
  - CEO fraud & phone phishing scams
  - Automated Speaker Verification (ASV) bypass
  - Fabricated audio in news, media, and legal proceedings
- **The Core Challenge:** Synthetic audio sounds indistinguishable to human ears, but leaves subtle mathematical and spectral anomalies.
> **Speaker Notes:**  
> "Human hearing is easily deceived by voice clones. However, generative vocoders and text-to-speech algorithms introduce subtle acoustic artifacts—such as phase incoherence and unnaturally rigid pitch. Our goal is to train a deep learning model that automatically detects these patterns."

---

### SLIDE 3: System Pipeline: How It Works
- **Step 1: Audio Input:** User records or uploads a WAV/MP3 recording.
- **Step 2: Preprocessing:** Resampled to 16 kHz mono, normalized to a fixed 3.0-second window (48,000 samples).
- **Step 3: Feature Extraction:** Converted into a 128-band Log-Mel Spectrogram.
- **Step 4: CNN Processing:** 2D Convolutions extract spatial spectral patterns and vocoder anomalies.
- **Step 5: LSTM Processing:** Bidirectional LSTMs model temporal dynamics across frames.
- **Step 6: Verdict & Risk:** Outputs classification (Bona Fide vs. Spoof) and confidence score.
> **Speaker Notes:**  
> "This slide illustrates our full end-to-end pipeline: from raw audio input to preprocessing, Mel-spectrogram transformation, hybrid neural network feature extraction, and finally the verdict and confidence rating."

---

### SLIDE 4: Acoustic Feature Extraction: Log-Mel Spectrograms
- **Why Mel-Spectrograms?**
  - Project linear frequency into the Mel scale, reflecting human auditory perception.
  - Convert sound into a 2D image-like representation $(128 \text{ Mel bands} \times 94 \text{ time frames})$.
- **Key Acoustic Differences:**
  - **Bona Fide Speech:** Smooth formant transitions, natural pitch vibrato, realistic vocal tract decay.
  - **Spoofed Speech:** Static harmonics, metallic vocoder buzz, unnatural high-frequency energy.
> **Speaker Notes:**  
> "By converting sound into a Log-Mel Spectrogram, we transform an audio problem into a computer vision and sequence learning task. This enables our 2D convolutional layers to scan for spatial irregularities in frequency bands over time."

---

### SLIDE 5: Model Architecture: The Hybrid CNN-LSTM Advantage
- **Why Hybrid? Neither CNN nor LSTM is sufficient on its own.**
  - **CNN Component (Spectral Extractor):**
    - 3 Conv2D blocks (32, 64, 128 filters) with Batch Normalization & ReLU
    - Captures local harmonic patterns and vocoder synthesis artifacts
  - **Bridge (Permute & Reshape):**
    - Reshapes spatial feature maps into a sequential format of 47 time frames $\times$ 2048 features.
  - **LSTM Component (Temporal Sequence Learner):**
    - 2-layer Stacked Bidirectional LSTM (64 and 32 units)
    - Captures forward and backward temporal transitions across speech frames
  - **Classification Head:**
    - Dense(64, ReLU) $\to$ Dropout(0.4) $\to$ Dense(1, Sigmoid)
> **Speaker Notes:**  
> "The key novelty of our architecture is the hybrid design. The CNN front-end acts as an expert spectral analyzer, while the Bidirectional LSTM acts as a temporal sequence tracker, catching robotic phoneme transitions and synthetic cadence."

---

### SLIDE 6: Dataset & Preprocessing Pipeline
- **Benchmark Standards:** Modeled according to the ASVspoof 2019 Logical Access protocol.
- **Dual Pipeline Strategy:**
  1. **Automated Starter Dataset Generator:** Built-in synthesis generating vocal dynamics and vocoder distortions for immediate experimentation.
  2. **Official ASVspoof LA Integration:** Scripts supporting the official academic corpus.
- **Data Partitions:** 80% Training, 20% Validation (Stratified across Genuine and Spoofed classes).
> **Speaker Notes:**  
> "We ensured our dataset pipeline follows the ASVspoof 2019 benchmark protocol. We implemented stratified partitioning to prevent data leakage and ensure class balance between authentic and spoofed audio."

---

### SLIDE 7: Training Strategy & Regularization
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam ($\text{learning rate} = 0.001$)
- **Regularization Techniques:**
  - Batch Normalization after each Conv2D layer
  - Spatial and recurrent Dropout (0.2 to 0.4) to prevent overfitting
  - `ModelCheckpoint` saving best weights based on validation loss
  - `EarlyStopping` with patience of 5 epochs
  - `ReduceLROnPlateau` for adaptive learning rate decay
> **Speaker Notes:**  
> "To prevent the network from overfitting to specific speakers or tones, we applied strict regularization including Batch Normalization, Dropout, early stopping, and learning rate reduction."

---

### SLIDE 8: Experimental Results & Benchmark Metrics
- **Accuracy:** **95.8%** overall classification accuracy
- **Precision:** **92.3%** (low false alarms)
- **Recall:** **100.0%** (zero missed deepfake attacks on validation)
- **F1-Score:** **96.0%**
- **ROC-AUC:** **0.9931** (exceptional discrimination threshold separation)
- **Equal Error Rate (EER):** **4.17%** (standard biometric benchmark)
> **Speaker Notes:**  
> "Here are our evaluation results. We achieved a 95.8% accuracy and 100% recall on synthetic samples. Most importantly in biometric security, our Equal Error Rate reached 4.17%, proving strong stability between False Acceptance and False Rejection."

---

### SLIDE 9: Interactive Demonstration (Gradio & Streamlit)
- **Real-Time Interface Features:**
  - Drag-and-drop audio upload (WAV, MP3, FLAC) or direct microphone input.
  - Interactive waveform and Mel-spectrogram rendering.
  - Probability confidence gauge with color-coded risk assessment (Safe / Suspicious / High-Risk Deepfake).
  - One-click deployment via Google Colab with public sharing link.
> **Speaker Notes:**  
> "To make our system practical, we built both Gradio and Streamlit user interfaces. Anyone can record their voice or upload an audio clip to instantly inspect its spectrogram and receive an AI authenticity score."

---

### SLIDE 10: Conclusion & Future Scope
- **Conclusion:**
  - Hybrid CNN-LSTM networks provide an effective, robust architecture for detecting AI-generated speech.
  - Mel-spectrograms effectively preserve the spectral and temporal artifacts left by neural vocoders.
- **Future Enhancements:**
  - Integrating raw-waveform neural extractors (RawNet3).
  - Evaluating self-supervised speech foundation models (Wav2Vec 2.0 / HuBERT).
  - Multi-task model extending detection to vocoder attribution (identifying which AI model created the fake).
- **Questions & Discussion:** Thank you!
> **Speaker Notes:**  
> "In conclusion, our CNN-LSTM hybrid system successfully detects synthetic voices by analyzing both spectral cues and temporal transitions. In the future, we plan to extend this with raw waveform networks and foundation models. Thank you, and we welcome any questions!"
