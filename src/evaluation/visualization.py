"""Visualization: error distributions, ROC/PR curves, confusion matrix, per-attack DR."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_error_distribution(
    benign_errors: np.ndarray,
    attack_errors: np.ndarray,
    threshold: float,
    title: str = "Error Distribution",
    save_path: str | None = None,
) -> None:
    """KDE plot of benign vs attack reconstruction errors."""
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.kdeplot(benign_errors, ax=ax, label="Benign", fill=True, alpha=0.4, color="green")
    sns.kdeplot(attack_errors, ax=ax, label="Attack", fill=True, alpha=0.4, color="red")
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_violin(
    scores: np.ndarray,
    labels_str: np.ndarray | pd.Series,
    threshold: float,
    top_n: int = 10,
    title: str = "Error by Attack Type",
    save_path: str | None = None,
) -> None:
    """Violin plot of errors by attack category."""
    if isinstance(labels_str, pd.Series):
        labels_str = labels_str.values
    labels_str = np.array([str(l).strip() for l in labels_str])

    df = pd.DataFrame({"score": scores, "label": labels_str})
    counts = df["label"].value_counts()
    top_labels = counts.head(top_n).index.tolist()
    df_top = df[df["label"].isin(top_labels)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df_top, x="label", y="score", ax=ax, cut=0, scale="width")
    ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    curves: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Plot ROC curves for multiple datasets on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in curves.items():
        ax.plot(data["fpr"], data["tpr"],
                label=f"{name} (AUC={data['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    curves: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Plot PR curves for multiple datasets on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in curves.items():
        ax.plot(data["recall"], data["precision"],
                label=f"{name} (AUC={data['auc']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Heatmap of confusion matrix."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_attack_dr(
    dr_df: pd.DataFrame,
    title: str = "Detection Rate per Attack",
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of per-attack detection rate."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(dr_df) * 0.4)))
    colors = ["#2ecc71" if dr >= 0.7 else "#f39c12" if dr >= 0.4 else "#e74c3c"
              for dr in dr_df["detection_rate"]]
    ax.barh(range(len(dr_df)), dr_df["detection_rate"].values, color=colors)
    ax.set_yticks(range(len(dr_df)))
    ax.set_yticklabels(
        [f"{row['attack_type']} (n={row['n_samples']})"
         for _, row in dr_df.iterrows()],
        fontsize=8,
    )
    ax.set_xlabel("Detection Rate")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_comparison(
    comparison_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Table visualization of threshold comparison."""
    fig, ax = plt.subplots(figsize=(14, max(3, len(comparison_df) * 0.5)))
    ax.axis("off")
    table = ax.table(
        cellText=comparison_df.round(4).values,
        colLabels=comparison_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.4)
    ax.set_title("Threshold Strategy Comparison", fontsize=12, pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
