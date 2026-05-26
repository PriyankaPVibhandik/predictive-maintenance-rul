"""
PyTorch Dataset — sliding-window sequences over C-MAPSS data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CMAPSSDataset(Dataset):
    """
    Builds fixed-length sliding windows per engine unit.

    Args:
        parquet_path: path to processed parquet file.
        feature_cols: list of sensor/op feature column names.
        seq_len: window length (time-steps).
        target_col: column name for RUL label.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        feature_cols: list[str],
        seq_len: int = 30,
        target_col: str = "RUL",
    ):
        df = pd.read_parquet(parquet_path)
        self.seq_len = seq_len
        self.feature_cols = feature_cols

        self.sequences: list[np.ndarray] = []
        self.labels: list[float] = []

        for unit_id, group in df.groupby("unit"):
            group = group.sort_values("cycle")
            feats = group[feature_cols].values.astype(np.float32)
            ruls = group[target_col].values.astype(np.float32)

            # Only rows where RUL is not NaN (test set has NaN for non-last rows)
            valid = ~np.isnan(ruls)
            if valid.sum() == 0:
                continue

            for end_idx in range(seq_len, len(feats) + 1):
                window = feats[end_idx - seq_len : end_idx]
                label = ruls[end_idx - 1]
                if np.isnan(label):
                    continue
                self.sequences.append(window)
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
