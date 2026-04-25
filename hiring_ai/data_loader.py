from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from .preprocessing import clean_text, extract_skills, extract_education, extract_experience_years


def _safe_read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, nrows=nrows)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", nrows=nrows)


def _first_existing(base: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        p = base / rel
        if p.exists():
            return p
    return None


def load_resumes(data_dir: str | Path, max_rows: int | None = None, sample_ratio: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    base = Path(data_dir)
    path = _first_existing(base, [
        "resume_skills/UpdatedResumeDataSet.csv",
        "processed/resume_skills/updatedresumedataset.csv",
        "resume_job_description/Resume.csv",
        "processed/resume_job_description/resume.csv",
    ])
    if path is None:
        raise FileNotFoundError("Could not find resume dataset in hiring_screening_data.")
    df = _safe_read_csv(path)
    if "Resume" in df.columns:
        text_col = "Resume"
    elif "resume_text" in df.columns:
        text_col = "resume_text"
    elif "Resume_str" in df.columns:
        text_col = "Resume_str"
    else:
        text_col = df.select_dtypes(include="object").columns[-1]
    category_col = "Category" if "Category" in df.columns else None
    if sample_ratio < 1.0 and len(df) > 0:
        df = df.sample(frac=sample_ratio, random_state=random_state)
    if max_rows:
        df = df.head(max_rows)
    out = pd.DataFrame()
    out["candidate_id"] = [f"CAND-{i+1:05d}" for i in range(len(df))]
    out["resume_text"] = df[text_col].fillna("").astype(str).map(clean_text)
    out["candidate_category"] = df[category_col].fillna("Unknown").astype(str) if category_col else "Unknown"
    out["candidate_skills"] = out["resume_text"].map(extract_skills)
    out["education_entities"] = out["resume_text"].map(extract_education)
    out["experience_years"] = out["resume_text"].map(extract_experience_years)
    return out.reset_index(drop=True)


def load_jobs(data_dir: str | Path, max_rows: int | None = None, sample_ratio: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    base = Path(data_dir)
    path = _first_existing(base, [
        "resume_job_description/training_data.csv",
        "processed/resume_job_description/training_data.csv",
    ])
    if path is None:
        raise FileNotFoundError("Could not find job description dataset in hiring_screening_data.")
    df = _safe_read_csv(path)
    title_col = "position_title" if "position_title" in df.columns else None
    desc_col = "job_description" if "job_description" in df.columns else None
    if desc_col is None:
        obj_cols = list(df.select_dtypes(include="object").columns)
        desc_col = obj_cols[-1]
    if title_col is None:
        title_col = df.select_dtypes(include="object").columns[0]
    df = df.dropna(subset=[desc_col]).drop_duplicates(subset=[title_col, desc_col])
    if sample_ratio < 1.0 and len(df) > 0:
        df = df.sample(frac=sample_ratio, random_state=random_state)
    if max_rows:
        df = df.head(max_rows)
    out = pd.DataFrame()
    out["job_id"] = [f"JOB-{i+1:05d}" for i in range(len(df))]
    out["job_title"] = df[title_col].fillna("Unknown Role").astype(str)
    out["job_title"] = out["job_title"].replace({"nan": "Unknown Role", "None": "Unknown Role", "": "Unknown Role"})
    out["job_description"] = df[desc_col].fillna("").astype(str).map(clean_text)
    out["job_text"] = (out["job_title"].astype(str) + " " + out["job_description"].astype(str)).map(clean_text)
    out["job_skills"] = out["job_text"].map(extract_skills)
    out["required_experience"] = out["job_text"].map(extract_experience_years)
    return out.reset_index(drop=True)


def load_interviews(data_dir: str | Path, n_candidates: int, sample_ratio: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    base = Path(data_dir)
    path = _first_existing(base, [
        "interview_selection/Data - Base.csv",
        "processed/interview_selection/data___base.csv",
    ])
    if path is None:
        return pd.DataFrame({"candidate_id": [f"CAND-{i+1:05d}" for i in range(n_candidates)], "interview_score": 0.5})
    df = _safe_read_csv(path)
    if sample_ratio < 1.0 and len(df) > 0:
        df = df.sample(frac=sample_ratio, random_state=random_state)
    df = df.head(n_candidates).copy()
    numeric_score_cols = [c for c in df.columns if "score" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_score_cols:
        raw = df[numeric_score_cols].fillna(0).sum(axis=1)
        denom = raw.max() - raw.min()
        interview_score = (raw - raw.min()) / denom if denom else raw * 0 + 0.5
    else:
        text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
        joined = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        positive_terms = ["selected", "good", "excellent", "confident", "fluent", "strong", "hire", "joined"]
        interview_score = joined.map(lambda x: min(sum(t in x.lower() for t in positive_terms) / 5.0, 1.0))
    out = pd.DataFrame()
    out["candidate_id"] = [f"CAND-{i+1:05d}" for i in range(len(df))]
    out["interview_score"] = interview_score.astype(float).clip(0, 1).values
    if "Gender" in df.columns:
        out["gender"] = df["Gender"].fillna("Unknown").astype(str).values
    if "Age" in df.columns:
        out["age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0).values
        out["age_group"] = pd.cut(out["age"], bins=[0, 25, 35, 45, 60, 100], labels=["<=25", "26-35", "36-45", "46-60", "60+"]).astype(str)
    status_cols = [c for c in df.columns if "status" in c.lower() or "verdict" in c.lower() or "joined" in c.lower()]
    if status_cols:
        joined_status = df[status_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        out["interview_positive_label"] = joined_status.map(lambda x: int(any(t in x for t in ["select", "selected", "joined", "yes", "hire"])))
    return out.reset_index(drop=True)


def attach_interview_data(candidates: pd.DataFrame, interviews: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if interviews.empty:
        out["interview_score"] = 0.5
        return out
    out = out.merge(interviews, on="candidate_id", how="left")
    out["interview_score"] = out["interview_score"].fillna(out["interview_score"].median() if out["interview_score"].notna().any() else 0.5)
    return out


def derive_relevance_label(candidate_category: str, job_title: str, job_text: str, skill_score: float) -> int:
    cat = clean_text(candidate_category)
    combined = clean_text(str(job_title) + " " + str(job_text))
    if cat and cat != "unknown" and any(part for part in cat.split() if len(part) > 3 and part in combined):
        return 1
    return int(skill_score >= 0.25)
