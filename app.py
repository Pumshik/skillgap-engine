import os
import logging
from typing import List
import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SkillGap Engine Inference API", version="1.0.0")

class FeaturesInput(BaseModel):
    features: List[float]

SCALER_PATH = "artifacts/scaler.joblib"
MODEL_PATH = "artifacts/model_torch.pth"
scaler = None
model = None
device = torch.device("cpu")

@app.on_event("startup")
async def load_models():
    global scaler, model
    if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Artifacts not found. Run `python src/train_torch.py` first.")
    
    scaler = joblib.load(SCALER_PATH)
    input_dim = scaler.n_features_in_
    
    from src.models.pytorch_net import MLPClassifier
    model = MLPClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info("Models loaded successfully")

@app.post("/predict")
async def predict(input_data: FeaturesInput):
    try:
        features = np.array(input_data.features).reshape(1, -1)
        if features.shape[1] != scaler.n_features_in_:
            raise HTTPException(status_code=400, detail=f"Expected {scaler.n_features_in_} features, got {features.shape[1]}")
        
        features_scaled = scaler.transform(features)
        tensor_input = torch.tensor(features_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = model(tensor_input)
            prob = torch.sigmoid(logits).item()
            pred_class = int(prob >= 0.5)
            
        return {"prediction": pred_class, "probability": round(prob, 4), "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": scaler is not None and model is not None}