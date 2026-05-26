"""
PyTorch Lightning LSTM module for Remaining Useful Life (RUL) regression.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class RULLSTMModel(pl.LightningModule):
    """
    Bidirectional-optional LSTM that predicts scalar RUL from a
    (batch, seq_len, n_features) input tensor.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out = hidden_size * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.lr = learning_rate
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)            # (B, T, H)
        last = out[:, -1, :]            # take last time-step
        return self.head(last).squeeze(-1)

    # ── Steps ───────────────────────────────────────────────────────────────
    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/mae", self.mae(y_hat, y), prog_bar=True)
        self.log(f"{stage}/rmse", self.rmse(y_hat, y), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ── Optimiser ───────────────────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
