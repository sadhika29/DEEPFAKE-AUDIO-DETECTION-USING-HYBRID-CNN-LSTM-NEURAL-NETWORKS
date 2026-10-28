import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from src.config import (
    BEST_MODEL_PATH, FINAL_MODEL_PATH, REPORTS_DIR, DATA_DIR,
    CLASS_NAMES, BATCH_SIZE
)
from src.dataset import AudioDataGenerator

def compute_eer(y_true, y_scores):
    """
    Computes the Equal Error Rate (EER) - the standard ASVspoof benchmark metric
    where False Acceptance Rate (FAR / FPR) equals False Rejection Rate (FRR / FNR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find index where |fpr - fnr| is minimized
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]
    return eer, eer_threshold, fpr, tpr

def plot_confusion_matrix(cm, save_path=REPORTS_DIR / "confusion_matrix.png"):
    """Plots an annotated Confusion Matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False
    )
    plt.title("Confusion Matrix: Deepfake Audio Detection", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Ground Truth Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion Matrix plot saved to: {save_path}")

def plot_roc_curve(fpr, tpr, roc_auc, eer, save_path=REPORTS_DIR / "roc_curve.png"):
    """Plots Receiver Operating Characteristic (ROC) curve with EER line."""
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance")
    
    # Plot EER point
    plt.scatter([eer], [1 - eer], color="#dc2626", s=60, zorder=5, label=f"EER = {eer * 100:.2f}%")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Spoof marked as Genuine)", fontsize=10)
    plt.ylabel("True Positive Rate (Spoof correctly flagged)", fontsize=10)
    plt.title("ROC Curve & Equal Error Rate (EER)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"ROC curve plot saved to: {save_path}")

def plot_score_distribution(y_true, y_scores, save_path=REPORTS_DIR / "score_distribution.png"):
    """Plots model prediction score distributions for Bona Fide vs Spoof samples."""
    plt.figure(figsize=(8, 5))
    bona_scores = y_scores[y_true == 0]
    spoof_scores = y_scores[y_true == 1]

    plt.hist(bona_scores, bins=25, alpha=0.6, color="#16a34a", label="Bona Fide (Genuine)", density=True)
    plt.hist(spoof_scores, bins=25, alpha=0.6, color="#dc2626", label="Spoof (Deepfake)", density=True)
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")

    plt.title("Prediction Probability Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Probability P(Spoof)", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Score distribution plot saved to: {save_path}")

def evaluate_model(model_path=None, test_csv=None):
    """
    Runs full evaluation benchmark on validation/test split.
    """
    # 1. Resolve model path
    if model_path is None:
        model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else FINAL_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # 2. Resolve dataset
    if test_csv is None:
        test_csv = DATA_DIR / "val_split.csv"
        if not test_csv.exists():
            test_csv = DATA_DIR / "protocol.csv"
    test_csv = Path(test_csv)
    if not test_csv.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {test_csv}.")

    df = pd.read_csv(test_csv)
    print(f"Evaluating on {len(df)} samples from {test_csv.name}")

    # 3. Stream data and compute predictions
    test_gen = AudioDataGenerator(df, batch_size=BATCH_SIZE, shuffle=False)
    
    y_true_list = []
    y_scores_list = []

    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        preds = model.predict(X_batch, verbose=0).flatten()
        y_scores_list.extend(preds)
        y_true_list.extend(y_batch[:len(preds)])

    y_true = np.array(y_true_list)
    y_scores = np.array(y_scores_list)
    y_pred = (y_scores >= 0.5).astype(int)

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    eer, eer_threshold, fpr, tpr = compute_eer(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    # 5. Output Summary
    print("\n" + "=" * 55)
    print("EVALUATION RESULTS & BENCHMARK PERFORMANCE")
    print("=" * 55)
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Precision:        {prec * 100:.2f}%")
    print(f"Recall:           {rec * 100:.2f}%")
    print(f"F1-Score:         {f1 * 100:.2f}%")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Equal Error Rate: {eer * 100:.2f}% (Threshold: {eer_threshold:.4f})")
    print("\nClassification Report:\n", cls_report)
    print("Confusion Matrix:\n", cm)
    print("=" * 55)

    # 6. Generate Diagnostic Plots
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc, eer)
    plot_score_distribution(y_true, y_scores)

    # 7. Save metrics JSON
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "equal_error_rate": float(eer),
        "eer_threshold": float(eer_threshold),
        "confusion_matrix": cm.tolist()
    }
    with open(REPORTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Deepfake Detector")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model (.keras)")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test CSV")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, test_csv=args.test_csv)
