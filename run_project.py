from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from hiring_ai.data_loader import load_resumes, load_jobs, load_interviews, attach_interview_data
from hiring_ai.models import compute_similarity
from hiring_ai.scoring import build_rankings
from hiring_ai.evaluation import evaluate_rankings
from hiring_ai.fairness import generate_fairness_report
from hiring_ai.visualisations import generate_all_visualisations


def parse_args():
    parser = argparse.ArgumentParser(description="AI-Powered Automated Hiring and Interview Screening System")
    parser.add_argument("--data-dir", default="hiring_screening_data", help="Directory containing the datasets")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of resumes/jobs/interviews to use")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional maximum number of candidates")
    parser.add_argument("--max-jobs", type=int, default=25, help="Optional maximum number of job descriptions; default keeps full run practical")
    parser.add_argument("--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Transformer embedding batch size")
    parser.add_argument("--strict-transformer", action="store_true", help="Fail if SentenceTransformer cannot load instead of using offline fallback")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    candidates = load_resumes(data_dir, max_rows=args.max_candidates, sample_ratio=args.sample_ratio)
    jobs = load_jobs(data_dir, max_rows=args.max_jobs, sample_ratio=args.sample_ratio)
    interviews = load_interviews(data_dir, n_candidates=len(candidates), sample_ratio=args.sample_ratio)
    candidates = attach_interview_data(candidates, interviews)

    if candidates.empty or jobs.empty:
        raise RuntimeError("No candidates or jobs loaded. Check the hiring_screening_data folder.")

    print(f"Candidates loaded: {len(candidates)}")
    print(f"Jobs loaded: {len(jobs)}")
    print("Computing TF-IDF baseline and transformer semantic similarities...")

    candidate_texts = candidates["resume_text"].fillna("").astype(str).tolist()
    job_texts = jobs["job_text"].fillna("").astype(str).tolist()
    sim = compute_similarity(
        candidate_texts,
        job_texts,
        model_name=args.transformer_model,
        batch_size=args.batch_size,
        allow_fallback=not args.strict_transformer,
    )

    print(f"Advanced model backend used: {sim.transformer_backend}")
    print("Building candidate rankings...")
    rankings = build_rankings(candidates, jobs, sim.baseline_scores, sim.transformer_scores)
    rankings.to_csv(output_dir / "candidate_rankings_all_jobs.csv", index=False)
    for job_id, group in rankings.groupby("job_id"):
        safe_title = "".join(ch if ch.isalnum() else "_" for ch in str(group["job_title"].iloc[0]))[:45]
        group.to_csv(output_dir / f"ranking_{job_id}_{safe_title}.csv", index=False)

    print("Evaluating baseline, transformer, and weighted scoring models...")
    evaluation = evaluate_rankings(rankings, output_dir)
    print("Generating fairness report...")
    fairness = generate_fairness_report(rankings, output_dir)
    print("Generating report-ready graphs...")
    generate_all_visualisations(output_dir)

    summary = {
        "candidates_loaded": int(len(candidates)),
        "jobs_loaded": int(len(jobs)),
        "candidate_job_pairs": int(len(rankings)),
        "advanced_model_backend": sim.transformer_backend,
        "transformer_model_requested": args.transformer_model,
        "outputs_directory": str(output_dir),
        "important_outputs": [
            "candidate_rankings_all_jobs.csv",
            "model_comparison_metrics.csv",
            "evaluation_report.json",
            "ranking_metrics_ndcg.csv",
            "error_analysis.csv",
            "job_role_performance.csv",
            "fairness_report.csv/json",
            "figures/*.png"
        ],
        "evaluation_summary": {
            "baseline_accuracy": evaluation["baseline_tfidf"]["accuracy"],
            "transformer_accuracy": evaluation["advanced_transformer"]["accuracy"],
            "weighted_final_accuracy": evaluation["weighted_final_scoring"]["accuracy"],
            "baseline_ndcg_at_10": evaluation["mean_ndcg_at_10"]["baseline_tfidf"],
            "transformer_ndcg_at_10": evaluation["mean_ndcg_at_10"]["advanced_transformer"],
            "weighted_final_ndcg_at_10": evaluation["mean_ndcg_at_10"]["weighted_final_scoring"],
        }
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
