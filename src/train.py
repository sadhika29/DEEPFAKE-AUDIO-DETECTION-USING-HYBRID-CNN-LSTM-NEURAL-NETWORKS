import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT, RANDOM_SEED,
    BEST_MODEL_PATH, FINAL_MODEL_PATH, REPORTS_DIR, DATA_DIR
)
from src.dataset import load_dataset_dataframe, AudioDataGenerator
from src.model import build_cnn_lstm_model

def plot_and_save_history(history, save_path=REPORTS_DIR / "training_history.png"):
    """
    Plots training and validation loss and accuracy across epochs.
    """
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs_range, history.history["loss"], label="Train Loss", color="#dc2626", linewidth=2)
    axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#ea580c", linestyle="--", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Accuracy Curve
    axes[1].plot(epochs_range, history.history["accuracy"], label="Train Accuracy", color="#2563eb", linewidth=2)
    axes[1].plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy", color="#0284c7", linestyle="--", linewidth=2)
    axes[1].set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Training history plot saved to: {save_path}")
    return fig

def train_model(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    protocol_csv=None,
    auto_generate=True
):
    """
    Executes end-to-end model training with early stopping and learning rate decay.
    """
    print("=" * 60)
    print("STARTING DEEPFAKE AUDIO DETECTION MODEL TRAINING")
    print("=" * 60)

    # 1. Load Dataset
    df = load_dataset_dataframe(protocol_csv=protocol_csv, auto_generate=auto_generate)
    print(f"Total dataset samples: {len(df)}")
    print(f"Class distribution:\n{df['class_name'].value_counts()}\n")

    # 2. Stratified Train / Validation Split
    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )
    print(f"Training samples: {len(train_df)} | Validation samples: {len(val_df)}")
    
    # Save split manifests
    train_df.to_csv(DATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_split.csv", index=False)

    # 3. Initialize Data Generators
    train_gen = AudioDataGenerator(train_df, batch_size=batch_size, shuffle=True)
    val_gen = AudioDataGenerator(val_df, batch_size=batch_size, shuffle=False)

    # 4. Build Model
    model = build_cnn_lstm_model(learning_rate=learning_rate)
    model.summary()

    # 5. Define Training Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 6. Fit Model
    print("\nBeginning Training Loop...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # 7. Save Final Model & History
    model.save(str(FINAL_MODEL_PATH))
    print(f"\nFinal model saved to: {FINAL_MODEL_PATH}")
    print(f"Best checkpoint saved to: {BEST_MODEL_PATH}")

    # Plot & Save Learning Curves
    plot_and_save_history(history)

    # Export history dictionary
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(REPORTS_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Deepfake Audio Detector")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--protocol", type=str, default=None, help="Path to protocol CSV")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        protocol_csv=args.protocol
    )
