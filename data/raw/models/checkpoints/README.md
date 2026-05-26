# Model Checkpoints

Best checkpoint: rul_lstm_epoch=10_val_loss=148.0062.ckpt

Model architecture: PyTorch Lightning LSTM
- Input size: 17 features
- Hidden size: 128
- Num layers: 2
- Dropout: 0.2
- Val Loss (MSE): 148.0062
- Training epochs: 20

Checkpoint files are not uploaded due to GitHub file size limits.
To reproduce: run `python -m src.models.train` after preprocessing.

data/raw/README.md
# NASA C-MAPSS Dataset

Raw data files (train_FD001.txt etc.) are not uploaded due to size.

Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Place all 12 .txt files in this folder before running preprocessing.
