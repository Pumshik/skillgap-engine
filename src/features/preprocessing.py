import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from loguru import logger
from typing import List

class SkillSplitter(BaseEstimator, TransformerMixin):
    """Разделяет строку навыков в список для мульти-лейбла"""
    def fit(self, X, y=None):
        return self
    def transform(self, X: pd.Series) -> List[List[str]]:
        return [str(x).lower().replace(" ", "").split(",") for x in X]

class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """Извлекает статистики из текста"""
    def fit(self, X, y=None): return self
    def transform(self, X: pd.Series) -> pd.DataFrame:
        lengths = X.str.len()
        unique_words = X.str.split().apply(lambda x: len(set(x)) if isinstance(x, list) else 0)
        return pd.DataFrame({"text_length": lengths, "unique_skills": unique_words})

def build_preprocessor(config: dict) -> ColumnTransformer:
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=2))
    ])
    
    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipe, "description"),
            ("num", num_pipe, ["salary_min"])
        ],
        remainder="drop"
    )
    logger.info("Preprocessor pipeline initialized")
    return preprocessor

def build_text_preprocessor(max_features=2000):
    """Возвращает ColumnTransformer для извлечения TF-IDF из текстов резюме и вакансии."""
    text_transformer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("resume_tfidf", text_transformer, "resume_text"),
            ("job_tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1,2), min_df=2), "job_text")
        ],
        remainder='drop'
    )
    return preprocessor