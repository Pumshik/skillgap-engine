# import os
# import logging
# import json
# import joblib
# import torch
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# from src.data.loader import _parse_skills
# from src.models.pytorch_net import MLPClassifier

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="SkillGap Engine Inference API", version="1.0.0")

# class ResumeJobInput(BaseModel):
#     resume_text: str
#     job_text: str
#     resume_skill_list: str
#     job_required_skills: str
#     # По умолчанию используем ансамбль, но можно выбрать 'mlp' или 'linear'
#     model_choice: str = "ensemble"

# PREPROCESSOR_PATH = "artifacts/scaler.joblib"
# LINEAR_MODEL_PATH = "artifacts/model.joblib"
# MLP_MODEL_PATH = "artifacts/model_torch.pth"
# INPUT_DIM_PATH = "artifacts/input_dim.json"
# MODEL_CONFIG_PATH = "artifacts/model_config.json"  # Важно для избежания size mismatch
# CALIBRATOR_PATH = "artifacts/calibrator.joblib"

# preprocessor = None
# linear_model = None
# mlp_model = None
# calibrator = None
# device = torch.device("cpu")

# @app.on_event("startup")
# async def load_models():
#     global preprocessor, linear_model, mlp_model, calibrator
    
#     if not os.path.exists(PREPROCESSOR_PATH):
#         raise RuntimeError("Preprocessor not found. Run training first.")
#     preprocessor = joblib.load(PREPROCESSOR_PATH)
#     logger.info("Preprocessor loaded")

#     # Загрузка линейной модели
#     if os.path.exists(LINEAR_MODEL_PATH):
#         linear_model = joblib.load(LINEAR_MODEL_PATH)
#         logger.info("Linear model (Logistic Regression) loaded")
#     else:
#         logger.warning("Linear model not found – linear/ensemble predictions will be unavailable")

#     # Загрузка MLP модели с учетом её реальной архитектуры из конфига
#     if os.path.exists(MLP_MODEL_PATH):
#         if os.path.exists(MODEL_CONFIG_PATH):
#             with open(MODEL_CONFIG_PATH, "r") as f:
#                 model_cfg = json.load(f)
#         else:
#             # Fallback на старые значения, если конфиг не был сохранен
#             with open(INPUT_DIM_PATH, "r") as f:
#                 model_cfg = {"input_dim": json.load(f)["input_dim"], "hidden_dim": 32, "dropout": 0.4}
        
#         mlp_model = MLPClassifier(
#             input_dim=model_cfg["input_dim"],
#             hidden_dim=model_cfg.get("hidden_dim", 32),
#             dropout=model_cfg.get("dropout", 0.4)
#         ).to(device)
        
#         mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device, weights_only=False))
#         mlp_model.eval()
#         logger.info(f"MLP model loaded with config: {model_cfg}")
        
#         if os.path.exists(CALIBRATOR_PATH):
#             calibrator = joblib.load(CALIBRATOR_PATH)
#             logger.info("Calibrator loaded")
#         else:
#             logger.warning("Calibrator not found. Raw sigmoid probabilities will be used for MLP.")
#     else:
#         logger.warning("MLP model not found – mlp/ensemble predictions will be unavailable")

# @app.get("/")
# async def root():
#     return {"message": "SkillGap Engine API is running. Visit /docs for Swagger UI."}

# @app.post("/predict")
# async def predict(input_data: ResumeJobInput):
#     try:
#         model_choice = input_data.model_choice.lower()
#         if model_choice not in ("linear", "mlp", "ensemble"):
#             raise HTTPException(status_code=400, detail="Invalid model_choice. Use 'linear', 'mlp', or 'ensemble'.")

#         df = pd.DataFrame([{
#             "resume_text": input_data.resume_text,
#             "job_text": input_data.job_text,
#             "resume_skill_list": input_data.resume_skill_list,
#             "job_required_skills": input_data.job_required_skills
#         }])
        
#         resume_skills = df["resume_skill_list"].apply(_parse_skills)
#         required_skills = df["job_required_skills"].apply(_parse_skills)
        
#         df["resume_skill_count"] = resume_skills.apply(len)
#         df["required_skill_count"] = required_skills.apply(len)
#         df["matched_skill_count"] = [len(r & req) for r, req in zip(resume_skills, required_skills)]
#         df["skill_match_ratio"] = df["matched_skill_count"] / df["required_skill_count"].replace(0, 1)
        
#         # Оставляем признаки длины на случай, если ваш текущий preprocessor всё еще ожидает их
#         df["resume_text_length"] = df["resume_text"].fillna("").astype(str).str.len()
#         df["job_text_length"] = df["job_text"].fillna("").astype(str).str.len()
        
#         features_sparse = preprocessor.transform(df)
#         features = features_sparse.toarray() if hasattr(features_sparse, "toarray") else features_sparse
        
#         prob_linear = None
#         prob_mlp = None
        
#         # 1. Предсказание линейной модели
#         if model_choice in ("linear", "ensemble"):
#             if linear_model is None:
#                 raise HTTPException(status_code=500, detail="Linear model not loaded, but requested.")
#             prob_linear = float(linear_model.predict_proba(features)[0, 1])
            
#         # 2. Предсказание MLP модели
#         if model_choice in ("mlp", "ensemble"):
#             if mlp_model is None:
#                 raise HTTPException(status_code=500, detail="MLP model not loaded, but requested.")
#             with torch.no_grad():
#                 tensor = torch.tensor(features, dtype=torch.float32).to(device)
#                 logits = mlp_model(tensor)
#                 raw_logit = logits.item()
                
#                 if calibrator is not None:
#                     # Platt scaling: логистическая регрессия от логита
#                     prob_mlp = float(calibrator.predict_proba(np.array([[raw_logit]]))[:, 1][0])
#                 else:
#                     prob_mlp = float(torch.sigmoid(logits).item())
                    
#         # 3. Формирование финального ответа
#         if model_choice == "linear":
#             final_prob = prob_linear
#             model_used = "Linear"
#         elif model_choice == "mlp":
#             final_prob = prob_mlp
#             model_used = "MLP"
#         else:  # ensemble
#             if prob_linear is not None and prob_mlp is not None:
#                 final_prob = (prob_linear + prob_mlp) / 2.0
#                 model_used = "Ensemble (Linear + MLP)"
#             elif prob_linear is not None:
#                 final_prob = prob_linear
#                 model_used = "Linear (Ensemble Fallback)"
#             else:
#                 final_prob = prob_mlp
#                 model_used = "MLP (Ensemble Fallback)"
                
#         pred_class = int(final_prob >= 0.5)
        
#         return {
#             "prediction": pred_class,
#             "probability": round(final_prob, 4),
#             "model_used": model_used,
#             "status": "success"
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Prediction failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     models_loaded = preprocessor is not None and (linear_model is not None or mlp_model is not None)
#     return {"status": "healthy", "models_loaded": models_loaded}

import os
import logging
import json
import joblib
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.loader import _parse_skills
from src.models.pytorch_net import MLPClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SkillGap Engine Inference API", version="1.0.0")

class ResumeJobInput(BaseModel):
    resume_text: str
    job_text: str
    resume_skill_list: str
    job_required_skills: str
    model_choice: str = "ensemble"  # По умолчанию ансамбль

PREPROCESSOR_PATH = "artifacts/scaler.joblib"
LINEAR_MODEL_PATH = "artifacts/model.joblib"
MLP_MODEL_PATH = "artifacts/model_torch.pth"
INPUT_DIM_PATH = "artifacts/input_dim.json"
MODEL_CONFIG_PATH = "artifacts/model_config.json"
CALIBRATOR_PATH = "artifacts/calibrator.joblib"

preprocessor = None
linear_model = None
mlp_model = None
calibrator = None
device = torch.device("cpu")

@app.on_event("startup")
async def load_models():
    global preprocessor, linear_model, mlp_model, calibrator
    
    if not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError("Preprocessor not found. Run training first.")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Preprocessor loaded")

    if os.path.exists(LINEAR_MODEL_PATH):
        linear_model = joblib.load(LINEAR_MODEL_PATH)
        logger.info("Linear model (Logistic Regression) loaded")
    else:
        logger.warning("Linear model not found – linear/ensemble predictions will be unavailable")

    if os.path.exists(MLP_MODEL_PATH):
        # ИСПРАВЛЕНИЕ: Читаем реальную архитектуру из конфига
        if os.path.exists(MODEL_CONFIG_PATH):
            with open(MODEL_CONFIG_PATH, "r") as f:
                model_cfg = json.load(f)
        else:
            with open(INPUT_DIM_PATH, "r") as f:
                model_cfg = {"input_dim": json.load(f)["input_dim"], "hidden_dim": 32, "dropout": 0.4}
        
        mlp_model = MLPClassifier(
            input_dim=model_cfg["input_dim"],
            hidden_dim=model_cfg.get("hidden_dim", 32),
            dropout=model_cfg.get("dropout", 0.4)
        ).to(device)
        
        mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device, weights_only=False))
        mlp_model.eval()
        logger.info(f"MLP model loaded with config: {model_cfg}")
        
        if os.path.exists(CALIBRATOR_PATH):
            calibrator = joblib.load(CALIBRATOR_PATH)
            logger.info("Calibrator loaded")
        else:
            logger.warning("Calibrator not found. Raw sigmoid probabilities will be used for MLP.")
    else:
        logger.warning("MLP model not found – mlp/ensemble predictions will be unavailable")

@app.get("/")
async def root():
    return {"message": "SkillGap Engine API is running. Visit /docs for Swagger UI."}

@app.post("/predict")
async def predict(input_data: ResumeJobInput):
    try:
        model_choice = input_data.model_choice.lower()
        if model_choice not in ("linear", "mlp", "ensemble"):
            raise HTTPException(status_code=400, detail="Invalid model_choice. Use 'linear', 'mlp', or 'ensemble'.")

        df = pd.DataFrame([{
            "resume_text": input_data.resume_text,
            "job_text": input_data.job_text,
            "resume_skill_list": input_data.resume_skill_list,
            "job_required_skills": input_data.job_required_skills
        }])
        
        resume_skills = df["resume_skill_list"].apply(_parse_skills)
        required_skills = df["job_required_skills"].apply(_parse_skills)
        
        df["resume_skill_count"] = resume_skills.apply(len)
        df["required_skill_count"] = required_skills.apply(len)
        df["matched_skill_count"] = [len(r & req) for r, req in zip(resume_skills, required_skills)]
        df["skill_match_ratio"] = df["matched_skill_count"] / df["required_skill_count"].replace(0, 1)
        
        # Оставляем для совместимости, но preprocessor их проигнорирует, так как их нет в numeric_features
        df["resume_text_length"] = df["resume_text"].fillna("").astype(str).str.len()
        df["job_text_length"] = df["job_text"].fillna("").astype(str).str.len()
        
        features_sparse = preprocessor.transform(df)
        features = features_sparse.toarray() if hasattr(features_sparse, "toarray") else features_sparse
        
        # ИСПРАВЛЕНИЕ: Защитный клип последних 4 числовых признаков
        if features.shape[1] >= 4:
            features[:, -4:] = np.clip(features[:, -4:], -3.0, 3.0)
        
        prob_linear = None
        prob_mlp = None
        
        # 1. Предсказание линейной модели
        if model_choice in ("linear", "ensemble"):
            if linear_model is None:
                raise HTTPException(status_code=500, detail="Linear model not loaded, but requested.")
            prob_linear = float(linear_model.predict_proba(features)[0, 1])
            
        # 2. Предсказание MLP модели
        if model_choice in ("mlp", "ensemble"):
            if mlp_model is None:
                raise HTTPException(status_code=500, detail="MLP model not loaded, but requested.")
            with torch.no_grad():
                tensor = torch.tensor(features, dtype=torch.float32).to(device)
                logits = mlp_model(tensor)
                raw_logit = logits.item()
                
                if calibrator is not None:
                    prob_mlp = float(calibrator.predict_proba(np.array([[raw_logit]]))[:, 1][0])
                else:
                    prob_mlp = float(torch.sigmoid(logits).item())
                    
        # 3. Формирование финального ответа
        if model_choice == "linear":
            final_prob = prob_linear
            model_used = "Linear"
        elif model_choice == "mlp":
            final_prob = prob_mlp
            model_used = "MLP"
        else:  # ensemble
            if prob_linear is not None and prob_mlp is not None:
                final_prob = (prob_linear + prob_mlp) / 2.0
                model_used = "Ensemble (Linear + MLP)"
            elif prob_linear is not None:
                final_prob = prob_linear
                model_used = "Linear (Ensemble Fallback)"
            else:
                final_prob = prob_mlp
                model_used = "MLP (Ensemble Fallback)"
                
        pred_class = int(final_prob >= 0.5)
        
        return {
            "prediction": pred_class,
            "probability": round(final_prob, 4),
            "model_used": model_used,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    models_loaded = preprocessor is not None and (linear_model is not None or mlp_model is not None)
    return {"status": "healthy", "models_loaded": models_loaded}