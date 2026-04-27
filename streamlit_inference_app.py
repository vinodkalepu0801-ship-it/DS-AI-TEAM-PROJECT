from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from inference import run_inference
from hiring_ai.data_loader import load_jobs

st.set_page_config(page_title="AI Hiring Screening Inference", layout="wide")
st.title("AI-Powered Hiring Screening - Candidate Inference")
st.write("Upload a candidate CV/resume and optionally an interview transcript to generate job match scores and recommendations.")

DATA_DIR = "hiring_screening_data"
OUTPUT_DIR = "inference_outputs"

@st.cache_data(show_spinner=False)
def load_job_options():
    jobs = load_jobs(DATA_DIR, max_rows=None, sample_ratio=1.0)
    jobs["label"] = jobs["job_id"].astype(str) + " - " + jobs["job_title"].astype(str)
    return jobs

jobs = load_job_options()

with st.sidebar:
    st.header("Settings")
    compare_mode = st.radio("Compare candidate against", ["All jobs", "One selected job"])
    selected_job_id = None
    if compare_mode == "One selected job":
        selected_label = st.selectbox("Select job", jobs["label"].tolist())
        selected_job_id = selected_label.split(" - ")[0]
    top_k = st.slider("Number of results to show", min_value=1, max_value=25, value=10)
    strict_transformer = st.checkbox("Strict transformer mode", value=False)

resume_file = st.file_uploader("Upload CV/Resume text file (.txt)", type=["txt"])
resume_text_manual = st.text_area("Or paste CV/Resume text here", height=220)
interview_file = st.file_uploader("Optional: upload interview transcript (.txt)", type=["txt"])
interview_text_manual = st.text_area("Optional: paste interview transcript here", height=120)

if st.button("Generate Candidate Screening Results", type="primary"):
    resume_text = ""
    if resume_file is not None:
        resume_text = resume_file.read().decode("utf-8", errors="ignore")
    elif resume_text_manual.strip():
        resume_text = resume_text_manual

    interview_text = ""
    if interview_file is not None:
        interview_text = interview_file.read().decode("utf-8", errors="ignore")
    elif interview_text_manual.strip():
        interview_text = interview_text_manual

    if not resume_text.strip():
        st.error("Please upload or paste a CV/resume before running inference.")
    else:
        with st.spinner("Generating match scores using TF-IDF and transformer similarity..."):
            results = run_inference(
                resume_text=resume_text,
                interview_text=interview_text,
                data_dir=DATA_DIR,
                output_dir=OUTPUT_DIR,
                job_id=selected_job_id,
                top_k=top_k,
                allow_fallback=not strict_transformer,
            )
        st.success("Inference completed.")
        st.subheader("Top Candidate Matching Results")
        visible_cols = [
            "rank", "job_id", "job_title", "match_percentage", "recommendation",
            "tfidf_similarity", "transformer_similarity", "skill_match_score",
            "interview_score", "experience_score", "matched_skills", "missing_skills",
        ]
        st.dataframe(results[visible_cols], use_container_width=True)

        best = results.iloc[0]
        st.subheader("Best Match Explanation")
        st.write(best["explanation"])
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top Results CSV", csv, "candidate_inference_top_results.csv", "text/csv")

st.caption("Outputs are also saved in inference_outputs/.")
