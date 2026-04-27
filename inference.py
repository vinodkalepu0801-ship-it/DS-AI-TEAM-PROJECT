from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from hiring_ai.data_loader import load_jobs
from hiring_ai.models import TfidfBaselineMatcher, TransformerEmbeddingMatcher
from hiring_ai.preprocessing import (
    clean_text,
    extract_skills,
    extract_education,
    extract_experience_years,
    skill_match_score,
    matched_missing_skills,
)
from hiring_ai.scoring import experience_score


def read_text_from_file(path: Optional[str]) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.read_text(encoding="utf-8", errors="ignore")


def score_interview_text(text: str) -> float:
    cleaned = clean_text(text)
    if not cleaned:
        return 0.5

    positive_terms = [
        "excellent", "strong", "confident", "clear", "experienced", "leadership",
        "communication", "problem solving", "teamwork", "project", "delivered",
        "implemented", "managed", "achieved", "successful", "hire", "selected",
    ]
    negative_terms = [
        "poor", "weak", "unclear", "limited", "no experience", "unable", "failed",
        "confused", "not confident", "lack", "missing",
    ]
    pos = sum(term in cleaned for term in positive_terms)
    neg = sum(term in cleaned for term in negative_terms)
    raw = 0.5 + (pos * 0.04) - (neg * 0.05)
    return round(max(0.0, min(1.0, raw)), 5)


def select_jobs(jobs: pd.DataFrame, job_id: Optional[str], job_title: Optional[str]) -> pd.DataFrame:
    selected = jobs.copy()
    if job_id:
        selected = selected[selected["job_id"].astype(str).str.lower() == job_id.lower()]
    if job_title:
        selected = selected[selected["job_title"].astype(str).str.lower().str.contains(job_title.lower(), na=False)]
    if selected.empty:
        raise ValueError("No matching job found. Check --job-id or --job-title, or run without them to compare against all jobs.")
    return selected.reset_index(drop=True)


def make_recommendation(final_score: float) -> str:
    if final_score >= 0.75:
        return "Strong Match - shortlist for next stage"
    if final_score >= 0.60:
        return "Good Match - review by recruiter"
    if final_score >= 0.45:
        return "Moderate Match - consider if applicant pool is limited"
    return "Low Match - not recommended for this role"


def run_inference(
    resume_text: str,
    interview_text: str = "",
    data_dir: str = "hiring_screening_data",
    output_dir: str = "inference_outputs",
    job_id: Optional[str] = None,
    job_title: Optional[str] = None,
    transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    allow_fallback: bool = True,
    top_k: int = 10,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs(data_dir, max_rows=None, sample_ratio=1.0)
    jobs = select_jobs(jobs, job_id=job_id, job_title=job_title)

    cleaned_resume = clean_text(resume_text)
    if not cleaned_resume:
        raise ValueError("Resume/CV text is empty. Provide --resume-file or --resume-text.")

    candidate_skills = extract_skills(cleaned_resume)
    education = extract_education(cleaned_resume)
    candidate_experience = extract_experience_years(cleaned_resume)
    interview_score = score_interview_text(interview_text)

    candidate_texts = [cleaned_resume]
    job_texts = jobs["job_text"].fillna("").astype(str).tolist()

    baseline_scores = TfidfBaselineMatcher().score(candidate_texts, job_texts)[0]
    transformer_matcher = TransformerEmbeddingMatcher(
        model_name=transformer_model,
        batch_size=batch_size,
        allow_fallback=allow_fallback,
    )
    transformer_scores = transformer_matcher.score(candidate_texts, job_texts)[0]

    rows = []
    weights = {"transformer": 0.40, "skills": 0.25, "interview": 0.20, "experience": 0.15}

    for idx, job in jobs.iterrows():
        job_skills = job["job_skills"]
        skill_score = skill_match_score(candidate_skills, job_skills)
        exp_score = experience_score(candidate_experience, float(job.get("required_experience", 0) or 0))
        tfidf_score = float(baseline_scores[idx])
        transformer_score = float(transformer_scores[idx])
        final_score = (
            weights["transformer"] * transformer_score
            + weights["skills"] * skill_score
            + weights["interview"] * interview_score
            + weights["experience"] * exp_score
        )
        matched, missing = matched_missing_skills(candidate_skills, job_skills)
        rows.append(
            {
                "job_id": job["job_id"],
                "job_title": job["job_title"],
                "tfidf_similarity": round(tfidf_score, 5),
                "transformer_similarity": round(transformer_score, 5),
                "skill_match_score": round(skill_score, 5),
                "interview_score": round(interview_score, 5),
                "experience_score": round(exp_score, 5),
                "final_score": round(final_score, 5),
                "match_percentage": round(final_score * 100, 2),
                "recommendation": make_recommendation(final_score),
                "candidate_experience_years": candidate_experience,
                "candidate_skills": "; ".join(candidate_skills),
                "education_entities": "; ".join(education),
                "matched_skills": matched,
                "missing_skills": missing,
                "explanation": (
                    f"Transformer semantic similarity={transformer_score:.2f}; "
                    f"TF-IDF similarity={tfidf_score:.2f}; skill overlap={skill_score:.2f}; "
                    f"interview score={interview_score:.2f}; experience score={exp_score:.2f}. "
                    f"Matched skills: {matched if matched else 'None found'}. "
                    f"Missing skills: {missing if missing else 'No major extracted skill gaps'}."
                ),
                "advanced_model_backend": transformer_matcher.backend,
            }
        )

    results = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)
    cols = ["rank"] + [c for c in results.columns if c != "rank"]
    results = results[cols]

    top_results = results.head(top_k)
    results.to_csv(output_path / "single_candidate_inference_all_jobs.csv", index=False)
    top_results.to_csv(output_path / "single_candidate_inference_top_matches.csv", index=False)

    summary = {
        "jobs_compared": int(len(results)),
        "top_k_saved": int(min(top_k, len(results))),
        "advanced_model_backend": transformer_matcher.backend,
        "transformer_model_requested": transformer_model,
        "best_match": top_results.iloc[0].to_dict() if not top_results.empty else {},
        "output_files": [
            str(output_path / "single_candidate_inference_all_jobs.csv"),
            str(output_path / "single_candidate_inference_top_matches.csv"),
            str(output_path / "single_candidate_inference_summary.json"),
        ],
    }
    with open(output_path / "single_candidate_inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return top_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for a new CV/resume against job descriptions.")
    parser.add_argument("--data-dir", default="hiring_screening_data")
    parser.add_argument("--output-dir", default="inference_outputs")
    parser.add_argument("--resume-file", default=None, help="Path to a plain text resume/CV file")
    parser.add_argument("--resume-text", default=None, help="Resume/CV text directly from the command line")
    parser.add_argument("--interview-file", default=None, help="Optional path to interview transcript text file")
    parser.add_argument("--interview-text", default=None, help="Optional interview transcript text")
    parser.add_argument("--job-id", default=None, help="Optional job_id. If omitted, compares against all jobs.")
    parser.add_argument("--job-title", default=None, help="Optional partial job title search")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--transformer-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--strict-transformer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    resume_text = args.resume_text or read_text_from_file(args.resume_file)
    interview_text = args.interview_text or read_text_from_file(args.interview_file)

    results = run_inference(
        resume_text=resume_text,
        interview_text=interview_text,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        job_id=args.job_id,
        job_title=args.job_title,
        transformer_model=args.transformer_model,
        batch_size=args.batch_size,
        allow_fallback=not args.strict_transformer,
        top_k=args.top_k,
    )

    print("\nTop matching jobs for the uploaded candidate:\n")
    display_cols = [
        "rank", "job_id", "job_title", "match_percentage", "recommendation",
        "transformer_similarity", "skill_match_score", "interview_score", "experience_score",
    ]
    print(results[display_cols].to_string(index=False))
    print("\nDetailed outputs saved in:", args.output_dir)


if __name__ == "__main__":
    main()
