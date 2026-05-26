from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import joblib
from loguru import logger
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF

DETECTOR_MAP = {"IForest": IForest, "LOF": LOF, "HBOS": HBOS}

class AnomalyEnsemble:
    def __init__(self, detectors: list, contamination: float = 0.05):
        self.detectors = {name: DETECTOR_MAP[name](contamination=contamination) for name in detectors}

    def fit(self, X):
        for name, det in self.detectors.items():
            logger.info(f"Fitting {name} ...")
            det.fit(X)
        return self

    def predict(self, X):
        votes = np.stack([det.predict(X) for det in self.detectors.values()], axis=1)
        return (votes.mean(axis=1) >= 0.5).astype(int)

    def score_samples(self, X):
        scores = np.stack([det.decision_function(X) for det in self.detectors.values()], axis=1)
        return scores.mean(axis=1)

    def save(self, path):
        data = {"detectors": self.detectors}
        joblib.dump(data, path)
        logger.info(f"Ensemble saved -> {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.detectors = data["detectors"]
        return obj

def train_anomaly(config_path: str = "config/config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    proc_dir = Path(cfg["data"]["processed_dir"])
    an_cfg = cfg["anomaly"]
    feature_cols = pd.read_csv(proc_dir / "feature_cols.csv", header=None)[0].tolist()
    train_df = pd.read_parquet(proc_dir / "train.parquet")
    X_train = train_df[feature_cols].values
    ensemble = AnomalyEnsemble(detectors=an_cfg["detectors"], contamination=an_cfg["contamination"])
    ensemble.fit(X_train)
    ensemble.save(proc_dir / "anomaly_ensemble.pkl")

if __name__ == "__main__":
    train_anomaly()
