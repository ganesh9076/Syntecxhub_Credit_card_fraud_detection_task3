import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from src.utils import save_figure, get_logger

log = get_logger("threshold")

def plot_threshold_tradeoff(model, X_test, y_test, model_name: str):
    """
    Plot Precision, Recall, F1 across thresholds.
    Helps businesses pick the right operating point.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, precisions, "b-",  lw=2, label="Precision")
    ax.plot(thresholds, recalls,    "r-",  lw=2, label="Recall")
    ax.plot(thresholds, f1s,        "g--", lw=2, label="F1-Score")

    # Mark recommended business threshold
    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(x=best_t, color="orange", linestyle="--",
               label=f"Best F1 Threshold ≈ {best_t:.2f}")
    ax.axvline(x=0.3, color="purple", linestyle=":",
               label="Low Threshold (High Recall for fraud)")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Precision / Recall / F1 vs Threshold — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_figure(fig, f"threshold_tradeoff_{model_name.lower().replace(' ','_')}.png")
    plt.show()

    print("\n" + "="*60)
    print("  BUSINESS THRESHOLD DISCUSSION")
    print("="*60)
    print("""
  📌 FRAUD DETECTION THRESHOLD TRADEOFFS:

  ▸ LOW THRESHOLD (e.g., 0.2–0.3):
    • High Recall → Catches more real fraud
    • Low Precision → More false positives (legit txns blocked)
    • Best for: Banks wanting to minimize fraud losses

  ▸ HIGH THRESHOLD (e.g., 0.6–0.7):
    • High Precision → Fewer false alarms
    • Low Recall → Some fraud slips through
    • Best for: Payment processors valuing customer experience

  ▸ RECOMMENDED (F1-Optimal ≈ {:.2f}):
    • Balances both concerns
    • Adjust based on cost of false positive vs. false negative
    """.format(best_t))