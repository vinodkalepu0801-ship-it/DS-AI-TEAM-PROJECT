from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

SENSITIVE_COLUMNS = ["gender", "age_group"]


def generate_fairness_report(rankings: pd.DataFrame, output_dir: str | Path, top_n: int = 10) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    rows = []
    for attr in SENSITIVE_COLUMNS:
        if attr not in rankings.columns:
            continue
        tmp = rankings.copy()
        tmp[attr] = tmp[attr].fillna("Unknown").astype(str)
        if tmp[attr].nunique() <= 1:
            continue
        selected = tmp[tmp["rank_for_job"] <= top_n]
        total_by_group = tmp.groupby(attr)["candidate_id"].nunique()
        selected_by_group = selected.groupby(attr)["candidate_id"].nunique()
        for group, total in total_by_group.items():
            sel = int(selected_by_group.get(group, 0))
            group_rows = tmp[tmp[attr] == group]
            rows.append({
                "attribute": attr,
                "group": group,
                "num_unique_candidates": int(total),
                "selected_top_n_unique_candidates": sel,
                "selection_rate_top_n": float(sel / total) if total else 0,
                "mean_final_score": float(group_rows["final_score"].mean()),
                "mean_transformer_similarity": float(group_rows["transformer_similarity"].mean()),
            })
    fairness_df = pd.DataFrame(rows)
    if not fairness_df.empty:
        fairness_df.to_csv(output_dir / "fairness_report.csv", index=False)
    reports["note"] = "Sensitive attributes are used only for post-hoc fairness analysis, not for candidate scoring."
    reports["available_attributes"] = sorted(set(fairness_df["attribute"])) if not fairness_df.empty else []
    reports["rows"] = rows
    with open(output_dir / "fairness_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    return reports
