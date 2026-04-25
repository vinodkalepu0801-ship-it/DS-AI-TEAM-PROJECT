import re
from typing import Iterable, List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)

SKILL_DICTIONARY = sorted(set([
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "mysql", "postgresql",
    "mongodb", "oracle", "excel", "power bi", "tableau", "aws", "azure", "gcp", "docker",
    "kubernetes", "linux", "git", "github", "machine learning", "deep learning", "nlp",
    "natural language processing", "data science", "data analysis", "data engineering", "etl",
    "spark", "pyspark", "hadoop", "airflow", "dbt", "redshift", "snowflake", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "bert", "transformer",
    "xgboost", "lightgbm", "statistics", "regression", "classification", "clustering",
    "communication", "leadership", "teamwork", "problem solving", "project management",
    "agile", "scrum", "sales", "customer service", "marketing", "finance", "accounting",
    "html", "css", "react", "node", "fastapi", "flask", "django", "api", "rest api",
    "devops", "ci/cd", "terraform", "jenkins", "langchain", "openai", "llm"
]))

EDUCATION_TERMS = [
    "bachelor", "bachelors", "b.sc", "bsc", "b.tech", "btech", "degree", "graduate",
    "master", "masters", "m.sc", "msc", "m.tech", "mba", "phd", "doctorate", "diploma"
]


def clean_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[^A-Za-z0-9+#./\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: object) -> List[str]:
    cleaned = clean_text(text)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#.\-]*", cleaned)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def remove_stopwords(text: object) -> str:
    return " ".join(tokenize(text))


def extract_skills(text: object, extra_terms: Iterable[str] | None = None) -> List[str]:
    cleaned = clean_text(text)
    skills = set()
    dictionary = list(SKILL_DICTIONARY)
    if extra_terms:
        dictionary.extend([clean_text(t) for t in extra_terms if str(t).strip()])
    for skill in dictionary:
        if not skill:
            continue
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, cleaned):
            skills.add(skill)
    return sorted(skills)


def extract_education(text: object) -> List[str]:
    cleaned = clean_text(text)
    return sorted({term for term in EDUCATION_TERMS if term in cleaned})


def extract_experience_years(text: object) -> float:
    cleaned = clean_text(text)
    values = []
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\s+(?:of\s+)?experience",
        r"experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)",
        r"(\d+)\s*(?:months|month)\s+(?:of\s+)?experience"
    ]
    for pat in patterns:
        for match in re.findall(pat, cleaned):
            try:
                val = float(match)
                if "month" in pat:
                    val = val / 12.0
                if 0 <= val <= 50:
                    values.append(val)
            except ValueError:
                pass
    return round(max(values), 2) if values else 0.0


def skill_match_score(candidate_skills: List[str], job_skills: List[str]) -> float:
    if not job_skills:
        return 0.0
    return len(set(candidate_skills).intersection(job_skills)) / max(len(set(job_skills)), 1)


def matched_missing_skills(candidate_skills: List[str], job_skills: List[str]) -> tuple[str, str]:
    cset, jset = set(candidate_skills), set(job_skills)
    matched = sorted(cset.intersection(jset))
    missing = sorted(jset - cset)
    return "; ".join(matched), "; ".join(missing)
