import sys
import os
import json
import logging
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.makedirs("artifacts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

from src.data.loader import load_config, prepare_data
from src.models.classical import MyLogisticRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("logs/train.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def train(cfg_path: str = "configs/default.yaml"):
    cfg = load_config(cfg_path)
    
    logger.info("Loading & preprocessing data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = prepare_data(cfg)

    logger.info("Initializing MyLogisticRegression...")
    model_cfg = cfg.get("model", {})
    model = MyLogisticRegression(
        lr=model_cfg.get("lr", 0.01),
        max_iter=model_cfg.get("max_iter", 1000),
        tol=model_cfg.get("tol", 1e-4),
        batch_size=model_cfg.get("batch_size", 64),
        method=model_cfg.get("method", "sgd"),
        seed=cfg.get("data", {}).get("seed", 42)
    )

    model.fit(X_train, y_train)
    logger.info("Training converged.")

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, model.predict(X_val))),
        "val_f1": float(f1_score(y_val, model.predict(X_val))),
        "test_accuracy": float(accuracy_score(y_test, model.predict(X_test))),
        "test_f1": float(f1_score(y_test, model.predict(X_test))),
        "final_loss": float(model.loss_history_[-1])
    }

    logger.info(f"📊 Metrics: {metrics}")
    logger.info(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, cfg["paths"]["model"])
    joblib.dump(scaler, cfg["paths"]["scaler"])
    with open(cfg["paths"]["metrics"], "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Artifacts saved to artifacts/")
    return metrics

if __name__ == "__main__":
    train()