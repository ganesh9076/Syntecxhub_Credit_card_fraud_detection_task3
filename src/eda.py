import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from src.utils import save_figure

sns.set_theme(style="darkgrid", palette="muted")

def plot_class_imbalance(df: pd.DataFrame):
    """Bar + Pie showing class imbalance."""
    counts = df["Class"].value_counts()
    labels = ["Legit (0)", "Fraud (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution — Credit Card Fraud", fontsize=15, fontweight="bold")

    # Bar
    axes[0].bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}\n({v/len(df)*100:.2f}%)",
                     ha="center", fontweight="bold")
    axes[0].set_title("Transaction Count")
    axes[0].set_ylabel("Count")

    # Pie
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.2f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Proportion")

    plt.tight_layout()
    save_figure(fig, "01_class_imbalance.png")
    plt.show()

def plot_transaction_amount(df: pd.DataFrame):
    """Distribution of transaction amount by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount by Class", fontsize=14, fontweight="bold")

    for cls, color, name, ax in zip(
        [0, 1], ["#2ecc71", "#e74c3c"], ["Legit", "Fraud"], axes
    ):
        data = df[df["Class"] == cls]["Amount"]
        ax.hist(data, bins=50, color=color, alpha=0.8, edgecolor="black")
        ax.set_title(f"{name} Transactions\nMean=${data.mean():.2f}, Max=${data.max():.2f}")
        ax.set_xlabel("Amount (USD)")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    save_figure(fig, "02_amount_distribution.png")
    plt.show()

def plot_time_distribution(df: pd.DataFrame):
    """Fraud vs legit over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Transaction Time Distribution", fontsize=14, fontweight="bold")

    for cls, color, name, ax in zip(
        [0, 1], ["steelblue", "#e74c3c"], ["Legit", "Fraud"], axes
    ):
        data = df[df["Class"] == cls]["Time"] / 3600  # convert to hours
        ax.plot(data.values, ".", markersize=0.8, color=color, alpha=0.4)
        ax.set_title(f"{name} Transactions Over Time")
        ax.set_ylabel("Count")
        ax.set_xlabel("Time (hours)")

    plt.tight_layout()
    save_figure(fig, "03_time_distribution.png")
    plt.show()

def plot_feature_correlation(df: pd.DataFrame):
    """Heatmap of top correlated features with Class."""
    corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top_features = corr.head(15).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[top_features + ["Class"]].corr(),
        annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, ax=ax, linewidths=0.5
    )
    ax.set_title("Top Feature Correlations with Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "04_correlation_heatmap.png")
    plt.show()
def plot_top_features_boxplot(df: pd.DataFrame):
    """Boxplot of top distinguishing features."""
    corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top6 = corr.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Top Distinguishing Features: Fraud vs Legit", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, feat in enumerate(top6):
        sns.boxplot(
            data=df, x="Class", y=feat,
            hue="Class",
            palette=["#2ecc71", "#e74c3c"],   # ← list instead of dict, no key issues
            legend=False,
            ax=axes[i], showfliers=False
        )
        axes[i].set_title(feat)
        axes[i].set_xticklabels(["Legit", "Fraud"])

    plt.tight_layout()
    save_figure(fig, "05_top_features_boxplot.png")
    plt.show()

def run_eda(df: pd.DataFrame):
    print("\n[EDA] Starting Exploratory Data Analysis...\n")
    plot_class_imbalance(df)
    plot_transaction_amount(df)
    plot_time_distribution(df)
    plot_feature_correlation(df)
    plot_top_features_boxplot(df)
    print("[EDA] Complete. Plots saved to outputs/\n")