"""
FastAPI Inference Service
Endpoints:
  POST /predict/rul      → RUL prediction for a sequence
  POST /predict/anomaly  → Anomaly score for a feature vector
  GET  /schedule         → Run MILP scheduler on latest predictions
  GET  /health           → Health check
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from src.models.rul_model import RULLSTMModel
from src.models.anomaly import AnomalyEnsemble
from src.scheduler.milp_scheduler import EngineUnit, schedule_maintenance

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

# ── Global model holders ──────────────────────────────────────────────────
_rul_model: RULLSTMModel | None = None
_anomaly_ensemble: AnomalyEnsemble | None = None
_feature_cols: list[str] = []
_cfg: dict = {}


@app.on_event("startup")
async def startup_event():
    global _rul_model, _anomaly_ensemble, _feature_cols, _cfg

    with open("config/config.yaml") as f:
        _cfg = yaml.safe_load(f)

    proc_dir = Path(_cfg["data"]["processed_dir"])
    ckpt_dir = Path(_cfg["model"]["checkpoint_dir"])

    import pandas as pd
    _feature_cols = pd.read_csv(proc_dir / "feature_cols.csv", header=None)[0].tolist()

    # Load latest checkpoint
    ckpts = sorted(ckpt_dir.rglob("*.ckpt"))
    if ckpts:
        _rul_model = RULLSTMModel.load_from_checkpoint(str(ckpts[-1]))
        _rul_model.eval()
        logger.info(f"RUL model loaded from {ckpts[-1]}")

    anomaly_path = proc_dir / "anomaly_ensemble.pkl"
    if anomaly_path.exists():
        _anomaly_ensemble = AnomalyEnsemble.load(anomaly_path)
        logger.info("Anomaly ensemble loaded.")


# ── Schemas ──────────────────────────────────────────────────────────────
class RULRequest(BaseModel):
    sequence: list[list[float]]    # shape: (seq_len, n_features)
    unit_id: str = "unknown"


class RULResponse(BaseModel):
    unit_id: str
    predicted_rul: float
    status: str                    # "critical" | "warning" | "normal"


class AnomalyRequest(BaseModel):
    features: list[float]          # single time-step feature vector
    unit_id: str = "unknown"


class AnomalyResponse(BaseModel):
    unit_id: str
    anomaly_score: float
    is_anomaly: bool


class ScheduleRequest(BaseModel):
    units: list[dict[str, Any]]    # list of {unit_id, predicted_rul}


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rul_model_loaded": _rul_model is not None,
        "anomaly_model_loaded": _anomaly_ensemble is not None,
    }


@app.post("/predict/rul", response_model=RULResponse)
async def predict_rul(req: RULRequest):
    if _rul_model is None:
        raise HTTPException(503, "RUL model not loaded. Run training first.")

    x = torch.tensor([req.sequence], dtype=torch.float32)  # (1, T, F)
    with torch.no_grad():
        rul = float(_rul_model(x).item())

    rul_threshold = _cfg.get("model", {}).get("rul_threshold", 30)
    if rul <= rul_threshold * 0.5:
        status = "critical"
    elif rul <= rul_threshold:
        status = "warning"
    else:
        status = "normal"

    return RULResponse(unit_id=req.unit_id, predicted_rul=round(rul, 2), status=status)


@app.post("/predict/anomaly", response_model=AnomalyResponse)
async def predict_anomaly(req: AnomalyRequest):
    if _anomaly_ensemble is None:
        raise HTTPException(503, "Anomaly ensemble not loaded.")

    X = np.array([req.features])
    score = float(_anomaly_ensemble.score_samples(X)[0])
    is_anomaly = bool(_anomaly_ensemble.predict(X)[0] == 1)

    return AnomalyResponse(
        unit_id=req.unit_id, anomaly_score=round(score, 4), is_anomaly=is_anomaly
    )


@app.post("/schedule")
async def get_schedule(req: ScheduleRequest):
    units = [
        EngineUnit(
            unit_id=u["unit_id"],
            predicted_rul=u["predicted_rul"],
            repair_cost=_cfg["scheduler"]["cost_repair"],
            failure_cost=_cfg["scheduler"]["cost_failure"],
        )
        for u in req.units
    ]
    sched_cfg = _cfg["scheduler"]
    df = schedule_maintenance(
        units,
        planning_horizon=sched_cfg["maintenance_window_hours"],
        max_per_day=sched_cfg["max_units_per_day"],
    )
    return df.to_dict(orient="records")
