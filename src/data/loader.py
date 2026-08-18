import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def prepare_data_with_text(cfg: dict):
    """
    Загружает датасет, извлекает TF-IDF из текстов, добавляет числовые признаки,
    масштабирует и возвращает сплиты.
    """
    seed = cfg.get("data", {}).get("seed", 42)
    test_size = cfg.get("data", {}).get("test_size", 0.15)
    val_size = cfg.get("data", {}).get("val_size", 0.15)
    
    df = pd.read_csv("src/data/job_resume_fit.csv")
    
    resume_skills = df["resume_skill_list"].apply(_parse_skills)
    required_skills = df["job_required_skills"].apply(_parse_skills)
    
    df["resume_skill_count"] = resume_skills.apply(len)
    df["required_skill_count"] = required_skills.apply(len)
    df["matched_skill_count"] = [len(r & req) for r, req in zip(resume_skills, required_skills)]
    df["skill_match_ratio"] = df["matched_skill_count"] / df["required_skill_count"].replace(0, 1)
    df["resume_text_length"] = df["resume_text"].fillna("").astype(str).str.len()
    df["job_text_length"] = df["job_text"].fillna("").astype(str).str.len()
    
    # numeric_features = [
    #     "resume_skill_count", "required_skill_count", "matched_skill_count",
    #     "skill_match_ratio", "resume_text_length", "job_text_length"
    # ]
    numeric_features = [
        "resume_skill_count", "required_skill_count", "matched_skill_count",
        "skill_match_ratio"
    ]
    
    y = ((df["ai_match_score"] >= 70)).astype(int)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("text_resume", TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=2), "resume_text"),
            ("text_job", TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=2), "job_text"),
            ("numeric", StandardScaler(), numeric_features)
        ],
        remainder='drop'
    )
    
    X = preprocessor.fit_transform(df)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=seed, stratify=y_train_val
    )

    return (X_train, y_train.to_numpy()), (X_val, y_val.to_numpy()), (X_test, y_test.to_numpy()), preprocessor


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

import ast

def _parse_skills(value) -> set:
    if pd.isna(value):
        return set()
    if isinstance(value, str) and value.strip().startswith('[') and value.strip().endswith(']'):
        try:
            lst = ast.literal_eval(value)
            if isinstance(lst, list):
                return {str(item).strip().lower() for item in lst if item}
        except:
            pass
    return {
        skill.strip().lower()
        for skill in str(value).split(",")
        if skill.strip()
    }


def prepare_data(cfg: dict):
    """
    Загружает job_resume_fit, формирует признаки для
    оценки соответствия резюме вакансии и выполняет
    train/validation/test split.

    Возвращает:
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        scaler
    """

    seed = cfg.get("data", {}).get("seed", 42)
    test_size = cfg.get("data", {}).get("test_size", 0.15)
    val_size = cfg.get("data", {}).get("val_size", 0.15)

    df = pd.read_csv("src/data/job_resume_fit.csv")

    required_columns = [
        "resume_text",
        "job_text",
        "job_required_skills",
        "resume_skill_list",
        "ai_match_score",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"В датасете отсутствуют необходимые колонки: {missing_columns}"
        )

    resume_skills = df["resume_skill_list"].apply(_parse_skills)
    required_skills = df["job_required_skills"].apply(_parse_skills)

    df["resume_skill_count"] = resume_skills.apply(len)

    df["required_skill_count"] = required_skills.apply(len)

    df["matched_skill_count"] = [
        len(resume & required)
        for resume, required in zip(resume_skills, required_skills)
    ]

    df["skill_match_ratio"] = (
        df["matched_skill_count"]
        / df["required_skill_count"].replace(0, 1)
    )

    df["resume_text_length"] = (
        df["resume_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["job_text_length"] = (
        df["job_text"]
        .fillna("")
        .astype(str)
        .str.len()
    )
    y = ((df["ai_match_score"] >= 40)).astype(int)

    feature_columns = [
        "resume_skill_count",
        "required_skill_count",
        "matched_skill_count",
        "skill_match_ratio",
        "resume_text_length",
        "job_text_length",
    ]

    X = df[feature_columns].astype(float)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_val,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Dataset size: {len(df)}")
    print(f"Features: {feature_columns}")
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    print(f"Positive class ratio: {y.mean():.3f}")

    return (
        (X_train, y_train.to_numpy()),
        (X_val, y_val.to_numpy()),
        (X_test, y_test.to_numpy()),
        scaler,
    )