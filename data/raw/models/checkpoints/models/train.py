"""
Training script — PyTorch Lightning + MLflow.

Usage:
    python -m src.models.train
    python -m src.models.train --config config/config.yaml
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import yaml
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from src.data.dataset import CMAPSSDataset
from src.models.rul_model import RULLSTMModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    proc_dir = Path(cfg["data"]["processed_dir"])
    seq_len = cfg["data"]["sequence_length"]
    model_cfg = cfg["model"]
    ckpt_dir = Path(model_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature columns ──────────────────────────────────────────────────────
    feature_cols = pd.read_csv(
        proc_dir / "feature_cols.csv", header=None
    )[0].tolist()
    input_size = len(feature_cols)
    logger.info(f"Features ({input_size}): {feature_cols}")

    # ── Datasets & Loaders ───────────────────────────────────────────────────
    train_ds = CMAPSSDataset(proc_dir / "train.parquet", feature_cols, seq_len)
    val_ds = CMAPSSDataset(proc_dir / "val.parquet", feature_cols, seq_len)

    batch = model_cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    logger.info(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    lstm_cfg = model_cfg["lstm"]
    model = RULLSTMModel(
        input_size=input_size,
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        bidirectional=lstm_cfg["bidirectional"],
        learning_rate=model_cfg["training"]["learning_rate"],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="rul_lstm_{epoch:02d}_{val_loss:.4f}",
        monitor="val/loss",
        save_top_k=3,
        mode="min",
    )
    early_cb = EarlyStopping(
        monitor="val/loss",
        patience=model_cfg["training"]["patience"],
        mode="min",
    )

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "predmaint"))

    with mlflow.start_run():
        mlflow.log_params({**lstm_cfg, **model_cfg["training"]})

        trainer = pl.Trainer(
            max_epochs=model_cfg["training"]["max_epochs"],
            callbacks=[ckpt_cb, early_cb],
            log_every_n_steps=10,
            enable_progress_bar=True,
        )
        trainer.fit(model, train_loader, val_loader)

        # Log best checkpoint
        best_path = ckpt_cb.best_model_path
        mlflow.log_artifact(best_path, artifact_path="checkpoints")
        mlflow.pytorch.log_model(model, artifact_path="model")
        logger.info(f"Best checkpoint → {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    train(args.config)
