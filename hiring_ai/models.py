from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


@dataclass
class SimilarityResult:
    baseline_scores: np.ndarray
    transformer_scores: np.ndarray
    transformer_backend: str


class TfidfBaselineMatcher:
    def __init__(self, max_features: int = 12000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)

    def score(self, candidate_texts: List[str], job_texts: List[str]) -> np.ndarray:
        corpus = list(candidate_texts) + list(job_texts)
        matrix = self.vectorizer.fit_transform(corpus)
        cand_matrix = matrix[:len(candidate_texts)]
        job_matrix = matrix[len(candidate_texts):]
        return cosine_similarity(cand_matrix, job_matrix)


class TransformerEmbeddingMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32, allow_fallback: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.allow_fallback = allow_fallback
        self.backend = "sentence-transformers"
        self.model = None

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "SentenceTransformer model could not be loaded. Install requirements and ensure internet/model cache is available."
                ) from exc
            self.backend = "fallback-lsa-semantic-vectors"
            self.model = None

    def score(self, candidate_texts: List[str], job_texts: List[str]) -> np.ndarray:
        if self.model is None and self.backend == "sentence-transformers":
            self._load_model()
        if self.backend == "sentence-transformers":
            cand_emb = self.model.encode(candidate_texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
            job_emb = self.model.encode(job_texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
            return cosine_similarity(cand_emb, job_emb)
        # fallback keeps the project runnable offline but reports the backend clearly
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
        corpus = list(candidate_texts) + list(job_texts)
        tfidf = vectorizer.fit_transform(corpus)
        n_components = min(256, max(2, min(tfidf.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = normalize(svd.fit_transform(tfidf))
        cand_emb = dense[:len(candidate_texts)]
        job_emb = dense[len(candidate_texts):]
        return cosine_similarity(cand_emb, job_emb)


def compute_similarity(candidate_texts: List[str], job_texts: List[str], model_name: str, batch_size: int, allow_fallback: bool = True) -> SimilarityResult:
    baseline = TfidfBaselineMatcher().score(candidate_texts, job_texts)
    transformer = TransformerEmbeddingMatcher(model_name=model_name, batch_size=batch_size, allow_fallback=allow_fallback)
    advanced = transformer.score(candidate_texts, job_texts)
    return SimilarityResult(baseline, advanced, transformer.backend)
