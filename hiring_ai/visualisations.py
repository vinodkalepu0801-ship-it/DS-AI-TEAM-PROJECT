from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_model_comparison(output_dir):
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    ensure_dir(figures)
    path = output_dir / "model_comparison_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metric_cols = [c for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mean_ndcg_at_10"] if c in df.columns]
    melted = df.melt(id_vars="model", value_vars=metric_cols, var_name="Metric", value_name="Score")
    pivot = melted.pivot(index="Metric", columns="model", values="Score")
    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Baseline vs Transformer vs Final Weighted Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "model_comparison.png", dpi=300)
    plt.close()


def plot_score_distribution(output_dir):
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    ensure_dir(figures)
    path = output_dir / "candidate_rankings_all_jobs.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    plt.figure(figsize=(9, 6))
    plt.hist(df["final_score"], bins=25)
    plt.title("Candidate Final Match Score Distribution")
    plt.xlabel("Final Match Score")
    plt.ylabel("Number of Candidate-Job Pairs")
    plt.tight_layout()
    plt.savefig(figures / "score_distribution.png", dpi=300)
    plt.close()


def plot_top_candidates(output_dir, top_n=10):
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    ensure_dir(figures)
    path = output_dir / "candidate_rankings_all_jobs.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    first_job = df["job_id"].iloc[0]
    top = df[df["job_id"] == first_job].sort_values("final_score", ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top["candidate_id"].astype(str), top["final_score"])
    plt.title(f"Top {top_n} Candidates for {top['job_title'].iloc[0]}")
    plt.xlabel("Final Score")
    plt.ylabel("Candidate")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(figures / "top_candidates_first_job.png", dpi=300)
    plt.close()


def plot_confusion_matrices(output_dir):
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    ensure_dir(figures)
    path = output_dir / "evaluation_report.json"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    for key, title in [("baseline_tfidf", "TF-IDF Baseline"), ("advanced_transformer", "Transformer Embeddings"), ("weighted_final_scoring", "Weighted Final Scoring")]:
        cm = report.get(key, {}).get("confusion_matrix")
        if cm is None:
            continue
        cm = np.array(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Suitable", "Suitable"])
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix - {title}")
        plt.tight_layout()
        plt.savefig(figures / f"confusion_matrix_{key}.png", dpi=300)
        plt.close()


def plot_fairness(output_dir):
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    ensure_dir(figures)
    path = output_dir / "fairness_report.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    for attr in df["attribute"].unique():
        sub = df[df["attribute"] == attr]
        plt.figure(figsize=(9, 6))
        plt.bar(sub["group"].astype(str), sub["selection_rate_top_n"])
        plt.title(f"Fairness Analysis: Top-N Selection Rate by {attr}")
        plt.xlabel(attr)
        plt.ylabel("Top-N Selection Rate")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(figures / f"fairness_{attr}.png", dpi=300)
        plt.close()


def generate_all_visualisations(output_dir):
    plot_model_comparison(output_dir)
    plot_score_distribution(output_dir)
    plot_top_candidates(output_dir)
    plot_confusion_matrices(output_dir)
    plot_fairness(output_dir)
