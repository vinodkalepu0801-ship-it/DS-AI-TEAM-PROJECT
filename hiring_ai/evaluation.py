from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def ndcg_at_k(labels, scores, k=10) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    gains = labels[order]
    discounts = np.log2(np.arange(len(gains)) + 2)
    dcg = np.sum(gains / discounts)
    ideal = np.sort(labels)[::-1][:k]
    idcg = np.sum(ideal / discounts[:len(ideal)])
    return float(dcg / idcg) if idcg > 0 else 0.0


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    }


def evaluate_rankings(rankings: pd.DataFrame, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y = rankings["actual_relevance_label"].astype(int)
    baseline_metrics = classification_metrics(y, rankings["predicted_baseline_label"].astype(int))
    transformer_metrics = classification_metrics(y, rankings["predicted_transformer_label"].astype(int))
    final_metrics = classification_metrics(y, rankings["predicted_final_label"].astype(int))
    ndcg_rows = []
    for job_id, group in rankings.groupby("job_id"):
        ndcg_rows.append({
            "job_id": job_id,
            "job_title": group["job_title"].iloc[0],
            "baseline_ndcg_at_10": ndcg_at_k(group["actual_relevance_label"], group["tfidf_similarity"], 10),
            "transformer_ndcg_at_10": ndcg_at_k(group["actual_relevance_label"], group["transformer_similarity"], 10),
            "final_ndcg_at_10": ndcg_at_k(group["actual_relevance_label"], group["final_score"], 10),
            "num_candidates": len(group),
            "positive_labels": int(group["actual_relevance_label"].sum())
        })
    ndcg_df = pd.DataFrame(ndcg_rows)
    ndcg_df.to_csv(output_dir / "ranking_metrics_ndcg.csv", index=False)
    comparison = pd.DataFrame([
        {"model": "TF-IDF Baseline", **{k: v for k, v in baseline_metrics.items() if isinstance(v, float)}},
        {"model": "Transformer Embeddings", **{k: v for k, v in transformer_metrics.items() if isinstance(v, float)}},
        {"model": "Weighted Final Scoring", **{k: v for k, v in final_metrics.items() if isinstance(v, float)}},
    ])
    comparison["mean_ndcg_at_10"] = [
        ndcg_df["baseline_ndcg_at_10"].mean() if not ndcg_df.empty else 0,
        ndcg_df["transformer_ndcg_at_10"].mean() if not ndcg_df.empty else 0,
        ndcg_df["final_ndcg_at_10"].mean() if not ndcg_df.empty else 0,
    ]
    comparison.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    errors = rankings[rankings["actual_relevance_label"].astype(int) != rankings["predicted_final_label"].astype(int)].copy()
    errors.to_csv(output_dir / "error_analysis.csv", index=False)
    role_perf = rankings.groupby(["job_id", "job_title"]).agg(
        num_candidates=("candidate_id", "count"),
        mean_tfidf_score=("tfidf_similarity", "mean"),
        mean_transformer_score=("transformer_similarity", "mean"),
        mean_final_score=("final_score", "mean"),
        relevant_candidates=("actual_relevance_label", "sum")
    ).reset_index()
    role_perf.to_csv(output_dir / "job_role_performance.csv", index=False)
    report = {
        "baseline_tfidf": baseline_metrics,
        "advanced_transformer": transformer_metrics,
        "weighted_final_scoring": final_metrics,
        "mean_ndcg_at_10": {
            "baseline_tfidf": float(comparison.loc[0, "mean_ndcg_at_10"]),
            "advanced_transformer": float(comparison.loc[1, "mean_ndcg_at_10"]),
            "weighted_final_scoring": float(comparison.loc[2, "mean_ndcg_at_10"]),
        }
    }
    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
