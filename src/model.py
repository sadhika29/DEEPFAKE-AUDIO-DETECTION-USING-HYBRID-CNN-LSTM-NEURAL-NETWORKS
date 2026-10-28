import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import io

from src.config import INPUT_SHAPE, LEARNING_RATE

def build_cnn_lstm_model(input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE):
    """
    Constructs a Hybrid CNN-LSTM Neural Network for Deepfake Audio Detection:
    
    1. Spatial / Spectral Feature Extractor (CNN):
       - Conv2D + BatchNorm + ReLU + MaxPool blocks extract local spectral
         cues, harmonic structures, and frequency anomalies.
       - Selective pooling pools frequency while preserving sequential time resolution.
    
    2. Temporal Feature Extractor (Bidirectional LSTM):
       - Features are transposed to (time_steps, features) and fed into
         stacked Bidirectional LSTM layers to learn sequential continuity
         and temporal phase transitions across phonemes.
    
    3. Classification Head:
       - Dense layers with Dropout and Sigmoid activation output probability:
         P(Spoof) where 0 = Bona Fide (genuine) and 1 = Spoof (deepfake).
    """
    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")
    
    # --- CNN Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)  # (64, 47, 32)
    x = layers.Dropout(0.2, name="drop1")(x)

    # --- CNN Block 2 ---
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name="pool2")(x)  # (32, 47, 64)
    x = layers.Dropout(0.25, name="drop2")(x)

    # --- CNN Block 3 ---
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name="pool3")(x)  # (16, 47, 128)
    x = layers.Dropout(0.25, name="drop3")(x)

    # --- Reshape to Sequential Format (Batch, Time, Features) ---
    # Current shape: (batch, freq=16, time=47, channels=128)
    # Permute to: (batch, time=47, freq=16, channels=128)
    x = layers.Permute((2, 1, 3), name="permute_time_first")(x)
    
    # Calculate flattened feature dimension per time step: 16 * 128 = 2048
    time_steps = x.shape[1]
    features_per_step = x.shape[2] * x.shape[3]
    x = layers.Reshape((time_steps, features_per_step), name="flatten_features_per_step")(x)

    # --- LSTM Block (Temporal Dynamics) ---
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True),
        name="bilstm_1"
    )(x)
    x = layers.Dropout(0.3, name="drop_lstm1")(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(0.3, name="drop_lstm2")(x)

    # --- Fully Connected Classification Head ---
    x = layers.Dense(64, activation="relu", name="dense_fc1")(x)
    x = layers.Dropout(0.4, name="drop_fc1")(x)
    
    # Output layer: Sigmoid probability for binary classification
    outputs = layers.Dense(1, activation="sigmoid", name="prediction_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Deepfake_Detector")

    # Compile with Adam optimizer and binary crossentropy loss
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model

def get_model_summary_string(model):
    """Returns the Keras model summary as a clean multi-line string."""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()
