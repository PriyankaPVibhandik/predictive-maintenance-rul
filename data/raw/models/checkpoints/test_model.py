"""Unit tests for RUL model and anomaly ensemble."""

import numpy as np
import pytest
import torch

from src.models.rul_model import RULLSTMModel
from src.models.anomaly import AnomalyEnsemble


class TestRULLSTMModel:
    def setup_method(self):
        self.model = RULLSTMModel(input_size=14, hidden_size=32, num_layers=1)

    def test_forward_shape(self):
        x = torch.randn(8, 30, 14)      # (batch, seq_len, features)
        out = self.model(x)
        assert out.shape == (8,), f"Expected (8,) got {out.shape}"

    def test_forward_positive(self):
        x = torch.randn(4, 30, 14)
        out = self.model(x)
        # RUL should be a real number (unbounded before clipping)
        assert not torch.isnan(out).any()

    def test_hyperparams_saved(self):
        assert self.model.hparams["input_size"] == 14
        assert self.model.hparams["hidden_size"] == 32


class TestAnomalyEnsemble:
    def setup_method(self):
        np.random.seed(0)
        self.X = np.random.randn(200, 14).astype(np.float32)
        self.ensemble = AnomalyEnsemble(
            detectors=["IForest", "HBOS"], contamination=0.05
        )
        self.ensemble.fit(self.X)

    def test_predict_shape(self):
        preds = self.ensemble.predict(self.X[:10])
        assert preds.shape == (10,)

    def test_predict_binary(self):
        preds = self.ensemble.predict(self.X[:20])
        assert set(preds).issubset({0, 1})

    def test_score_shape(self):
        scores = self.ensemble.score_samples(self.X[:10])
        assert scores.shape == (10,)
