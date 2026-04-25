from __future__ import annotations
import numpy as np
import pandas as pd
from .preprocessing import skill_match_score, matched_missing_skills
from .data_loader import derive_relevance_label


def experience_score(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return min(candidate_years / 5.0, 1.0) if candidate_years else 0.5
    return min(candidate_years / required_years, 1.0)


def build_rankings(candidates: pd.DataFrame, jobs: pd.DataFrame, baseline_scores, transformer_scores, weights: dict | None = None) -> pd.DataFrame:
    weights = weights or {"transformer": 0.40, "skills": 0.25, "interview": 0.20, "experience": 0.15}
    rows = []
    for ci, cand in candidates.iterrows():
        for ji, job in jobs.iterrows():
            skill_score = skill_match_score(cand["candidate_skills"], job["job_skills"])
            exp_score = experience_score(float(cand.get("experience_years", 0) or 0), float(job.get("required_experience", 0) or 0))
            interview = float(cand.get("interview_score", 0.5) or 0.5)
            transformer = float(transformer_scores[ci, ji])
            baseline = float(baseline_scores[ci, ji])
            final_score = (
                weights["transformer"] * transformer +
                weights["skills"] * skill_score +
                weights["interview"] * interview +
                weights["experience"] * exp_score
            )
            matched, missing = matched_missing_skills(cand["candidate_skills"], job["job_skills"])
            label = derive_relevance_label(cand.get("candidate_category", ""), job.get("job_title", ""), job.get("job_text", ""), skill_score)
            rows.append({
                "candidate_id": cand["candidate_id"],
                "job_id": job["job_id"],
                "job_title": job["job_title"],
                "candidate_category": cand.get("candidate_category", "Unknown"),
                "tfidf_similarity": round(baseline, 5),
                "transformer_similarity": round(transformer, 5),
                "skill_match_score": round(skill_score, 5),
                "interview_score": round(interview, 5),
                "experience_score": round(exp_score, 5),
                "final_score": round(final_score, 5),
                "match_percentage": round(final_score * 100, 2),
                "matched_skills": matched,
                "missing_skills": missing,
                "actual_relevance_label": int(label),
                "predicted_baseline_label": int(baseline >= 0.20),
                "predicted_transformer_label": int(transformer >= 0.35),
                "predicted_final_label": int(final_score >= 0.45),
                "explanation": _make_explanation(matched, missing, transformer, skill_score, interview, exp_score),
                "gender": cand.get("gender", "Unknown"),
                "age_group": cand.get("age_group", "Unknown")
            })
    out = pd.DataFrame(rows)
    out["rank_for_job"] = out.groupby("job_id")["final_score"].rank(method="first", ascending=False).astype(int)
    return out.sort_values(["job_id", "rank_for_job"]).reset_index(drop=True)


def _make_explanation(matched: str, missing: str, transformer: float, skill_score: float, interview: float, exp_score: float) -> str:
    matched_text = matched if matched else "No explicit key skills matched"
    missing_text = missing if missing else "No major extracted skill gaps"
    return (
        f"Semantic score={transformer:.2f}; skill overlap={skill_score:.2f}; "
        f"interview score={interview:.2f}; experience score={exp_score:.2f}. "
        f"Matched skills: {matched_text}. Missing skills: {missing_text}."
    )
