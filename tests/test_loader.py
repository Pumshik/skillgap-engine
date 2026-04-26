import pytest
from src.data.loader import load_config, load_and_split_data

def test_config_loading():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg and "model" in cfg

def test_demo_data_shape():
    cfg = load_config()
    cfg["data"]["source"] = "nonexistent.csv"
    df = load_and_split_data(cfg)
    assert df.shape[0] == 1000
    assert set(["title", "description", "required_skills", "salary_min"]).issubset(df.columns)