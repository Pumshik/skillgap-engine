import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path

def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_data(cfg: dict):
    """
    Загружает датасет, масштабирует признаки и возвращает сплиты и скалер.
    """
    seed = cfg.get("data", {}).get("seed", 42)
    test_size = cfg.get("data", {}).get("test_size", 0.15)
    val_size = cfg.get("data", {}).get("val_size", 0.15)

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=seed, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train.values), (X_val, y_val.values), (X_test, y_test.values), scaler