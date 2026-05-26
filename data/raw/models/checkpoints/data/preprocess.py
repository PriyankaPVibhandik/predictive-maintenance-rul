"""
NASA C-MAPSS Dataset Preprocessing
Downloads FD001-FD004 subsets, engineers features, computes RUL labels,
and saves train/val/test splits as parquet files.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import joblib

# ── Column names ────────────────────────────────────────────────────────────
COLS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance (dropped)
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]

CMAPSS_URL = (
    "https://ti.arc.nasa.gov/c/6/"  # official NASA mirror
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_cmapss(raw_dir: Path) -> None:
    """Download C-MAPSS zip if not already present."""
    zip_path = raw_dir / "CMAPSSData.zip"
    if zip_path.exists():
        logger.info("C-MAPSS zip already present, skipping download.")
        return
    logger.info("Downloading C-MAPSS dataset …")
    urllib.request.urlretrieve(CMAPSS_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    logger.info("Download complete.")


def read_cmapss(raw_dir: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train and test txt files for a given subset (e.g. FD001)."""
    train_path = raw_dir / f"train_{subset}.txt"
    test_path = raw_dir / f"test_{subset}.txt"
    rul_path = raw_dir / f"RUL_{subset}.txt"

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLS)
    rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    return train, test, rul_true


def compute_rul(df: pd.DataFrame, rul_clip: int) -> pd.DataFrame:
    """Add piece-wise linear RUL column."""
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_clip)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def drop_low_variance(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_SENSORS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def select_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("s_") or c.startswith("op_")]


def preprocess(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])
    subset = cfg["data"]["cmapss_subset"]
    rul_clip = cfg["data"]["rul_clip"]
    seq_len = cfg["data"]["sequence_length"]
    val_size = cfg["data"]["val_size"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    logger.info(f"Loading {subset} …")
    train_df, test_df, rul_true = read_cmapss(raw_dir, subset)

    # ── RUL labels ──────────────────────────────────────────────────────────
    train_df = compute_rul(train_df, rul_clip)

    # For test set, last cycle RUL is provided in RUL_FDxxx.txt
    last_test = test_df.groupby("unit", as_index=False)["cycle"].max()
    last_test["RUL"] = rul_true["RUL"].values
    test_df = test_df.merge(last_test[["unit", "cycle", "RUL"]], on=["unit", "cycle"], how="left")

    # ── Feature engineering ─────────────────────────────────────────────────
    for df in (train_df, test_df):
        df.drop(columns=DROP_SENSORS, inplace=True, errors="ignore")

    feature_cols = [c for c in train_df.columns if c.startswith("s_") or c.startswith("op_")]

    # ── Normalise ───────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].fillna(0))

    scaler_path = proc_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved → {scaler_path}")

    # ── Train / val split (by unit) ──────────────────────────────────────────
    units = train_df["unit"].unique()
    np.random.seed(42)
    val_units = np.random.choice(units, size=int(len(units) * val_size), replace=False)
    val_df = train_df[train_df["unit"].isin(val_units)].copy()
    train_df = train_df[~train_df["unit"].isin(val_units)].copy()

    # ── Save ────────────────────────────────────────────────────────────────
    train_df.to_parquet(proc_dir / "train.parquet", index=False)
    val_df.to_parquet(proc_dir / "val.parquet", index=False)
    test_df.to_parquet(proc_dir / "test.parquet", index=False)

    # Save feature list
    pd.Series(feature_cols).to_csv(proc_dir / "feature_cols.csv", index=False, header=False)

    logger.info(
        f"Saved → train={len(train_df)}, val={len(val_df)}, test={len(test_df)} rows"
    )


if __name__ == "__main__":
    preprocess()
