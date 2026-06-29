import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from src.utils import save_figure, get_logger

log = get_logger("evaluation")

def evaluate_model(model, X_test, y_test, model_name: str, threshold: float = 0.5):
    """Full evaluation suite for a trained model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name.upper()}  |  Threshold={threshold}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision : {average_precision_score(y_test, y_prob):.4f}")

    _plot_confusion_matrix(y_test, y_pred, model_name)
    _plot_roc_curve(y_test, y_prob, model_name)
    _plot_pr_curve(y_test, y_prob, model_name)

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }

def _plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    save_figure(fig, f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.show()

def _plot_roc_curve(y_test, y_prob, model_name: str):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_figure(fig, f"roc_{model_name.lower().replace(' ', '_')}.png")
    plt.show()

def _plot_pr_curve(y_test, y_prob, model_name: str):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#3498db", lw=2, label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#3498db")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, f"pr_{model_name.lower().replace(' ', '_')}.png")
    plt.show()

def compare_models(results: dict):
    """Bar chart comparing ROC-AUC across models."""
    names = list(results.keys())
    aucs  = [v["roc_auc"] for v in results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, aucs, color=["#2ecc71", "#3498db"], edgecolor="black", width=0.4)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontweight="bold")
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Model Comparison — ROC-AUC", fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "model_comparison.png")
    plt.show()