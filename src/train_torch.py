import sys
import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.makedirs("artifacts", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("runs", exist_ok=True)

from src.data.loader import load_config, prepare_data_with_text
from src.models.pytorch_net import MLPClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("logs/torch_train.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def train_torch(cfg_path: str = "configs/default.yaml"):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    (X_train, y_train), (X_val, y_val), _, preprocessor = prepare_data_with_text(cfg)
    
    X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_ds, batch_size=cfg["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    
    writer = SummaryWriter(log_dir="runs/torch_mlp")
    
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=16,
        dropout=0.5
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["model"]["lr"], weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    patience, patience_counter = 15, 0
    epochs = 50
    
    logger.info("Starting PyTorch training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)
            
        train_loss = epoch_loss / len(train_ds)
        
        model.eval()
        val_loss, val_preds, val_true = 0.0, [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                val_loss += criterion(logits, y_b).item() * X_b.size(0)
                probs = torch.sigmoid(logits)
                val_preds.extend((probs > 0.5).float().cpu().numpy())
                val_true.extend(y_b.cpu().numpy())
                
        val_loss /= len(val_ds)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("Metrics/val_accuracy", val_acc, epoch)
        writer.add_scalar("Metrics/val_f1", val_f1, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "artifacts/model_torch.pth")
            joblib.dump(preprocessor, cfg["paths"]["scaler"])
            logger.info(f"New best model saved with val_acc = {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("artifacts/model_torch.pth", map_location=device))
    model.eval()
    val_logits = []
    val_true_calib = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            val_logits.extend(logits.cpu().numpy().flatten())
            val_true_calib.extend(y_b.numpy())
            
    calibrator = LogisticRegression(C=1.0, random_state=42)
    X_calib = np.array(val_logits).reshape(-1, 1)
    y_calib = np.array(val_true_calib)
    calibrator.fit(X_calib, y_calib)
    joblib.dump(calibrator, "artifacts/calibrator.joblib")
    logger.info("Platt scaling calibrator trained and saved.")

    metrics = {
        "best_val_accuracy": float(best_val_acc),
        "val_accuracy": float(val_acc),
        "val_f1": float(val_f1),
    }
    
    input_dim = int(X_train.shape[1])
    
    model_config = {
        "input_dim": input_dim,
        "hidden_dim": 16,
        "dropout": 0.5
    }
    with open("artifacts/model_config.json", "w") as f:
        json.dump(model_config, f)
        
    with open("artifacts/input_dim.json", "w") as f:
        json.dump({"input_dim": input_dim}, f)
        
    with open("artifacts/metrics_torch.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("Training finished. Artifacts saved to artifacts/")
    writer.close()
    return metrics

if __name__ == "__main__":
    train_torch()