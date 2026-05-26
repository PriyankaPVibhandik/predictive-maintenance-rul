"""
Model Evaluation — RUL LSTM
Generates: MAPE, RMSE, MAE, scatter plot, residuals, SHAP summary
Run: python notebooks/evaluate_model.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from src.models.rul_model import RULLSTMModel
from src.data.dataset import CMAPSSDataset

OUT  = Path("notebooks/figures")
OUT.mkdir(parents=True, exist_ok=True)
PROC = Path("data/processed")
CKPT = Path("models/checkpoints")

# ── Load best checkpoint ──────────────────────────────────────────────────────
ckpts = sorted(CKPT.rglob("*.ckpt"))
assert ckpts, "No checkpoint found. Run training first."
best_ckpt = str(ckpts[0])  # lowest val loss
print(f"Loading: {best_ckpt}")
model = RULLSTMModel.load_from_checkpoint(best_ckpt)
model.eval()

feat_cols = pd.read_csv(PROC / "feature_cols.csv", header=None)[0].tolist()
SEQ_LEN = 30

# ── Predict on test set ───────────────────────────────────────────────────────
test_ds = CMAPSSDataset(PROC / "test.parquet", feat_cols, seq_len=SEQ_LEN)
loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

y_true_all, y_pred_all = [], []
with torch.no_grad():
    for x, y in loader:
        preds = model(x)
        y_true_all.extend(y.numpy())
        y_pred_all.extend(preds.numpy())

y_true = np.array(y_true_all)
y_pred = np.array(y_pred_all)

# ── Metrics ───────────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

print("\n" + "="*40)
print("  RUL Model Test Set Metrics")
print("="*40)
print(f"  MAE  : {mae:.2f} cycles")
print(f"  RMSE : {rmse:.2f} cycles")
print(f"  MAPE : {mape:.2f}%")
print(f"  R²   : {r2:.4f}")
print("="*40)

metrics_df = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}])
metrics_df.to_csv(OUT / "metrics.csv", index=False)

# ── 1. Predicted vs Actual scatter ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_true, y_pred, alpha=0.25, s=8, color="#185FA5", label="Predictions")
lim = max(y_true.max(), y_pred.max())
ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title(f"Predicted vs Actual RUL  (RMSE={rmse:.1f}, MAPE={mape:.1f}%)", fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "08_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 08_predicted_vs_actual.png")

# ── 2. Residuals plot ─────────────────────────────────────────────────────────
residuals = y_pred - y_true
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_true, residuals, alpha=0.2, s=8, color="#534AB7")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Actual RUL")
axes[0].set_ylabel("Residual (predicted - actual)")
axes[0].set_title("Residuals vs Actual RUL")
axes[1].hist(residuals, bins=50, color="#534AB7", edgecolor="white", linewidth=0.4)
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Residual distribution  (mean={residuals.mean():.1f})")
plt.tight_layout()
plt.savefig(OUT / "09_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 09_residuals.png")

# ── 3. Critical failure detection (RUL < 30) ──────────────────────────────────
threshold = 30
y_true_bin = (y_true < threshold).astype(int)
y_pred_bin = (y_pred < threshold).astype(int)

from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve)

cm = confusion_matrix(y_true_bin, y_pred_bin)
print("\nClassification Report (RUL < 30 = Critical):")
print(classification_report(y_true_bin, y_pred_bin,
                             target_names=["Normal", "Critical"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Confusion matrix
sns_hm = plt.matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "#1D9E75"])
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap=sns_hm, ax=axes[0],
            xticklabels=["Normal", "Critical"], yticklabels=["Normal", "Critical"])
axes[0].set_title("Confusion matrix\n(Critical = RUL < 30)")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# ROC curve
fpr, tpr, _ = roc_curve(y_true_bin, -y_pred)  # lower pred RUL = higher risk
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color="#185FA5", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curve")
axes[1].legend()

# Precision-Recall
prec, rec, _ = precision_recall_curve(y_true_bin, -y_pred)
pr_auc = auc(rec, prec)
axes[2].plot(rec, prec, color="#534AB7", lw=2, label=f"PR (AUC={pr_auc:.3f})")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].set_title("Precision-Recall curve")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUT / "10_classification_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 10_classification_metrics.png")

# ── 4. SHAP feature importance ────────────────────────────────────────────────
try:
    import shap
    print("\nComputing SHAP values (this may take 1–2 minutes)...")

    # Use last time-step features for SHAP (tabular approximation)
    test_df = pd.read_parquet(PROC / "test.parquet").dropna(subset=["RUL"])
    X_shap  = test_df[feat_cols].values.astype(np.float32)
    X_bg    = X_shap[np.random.choice(len(X_shap), 200, replace=False)]
    X_eval  = X_shap[np.random.choice(len(X_shap), 100, replace=False)]

    def model_predict(X):
        # Repeat each row seq_len times to form a window (simplified)
        t = torch.tensor(
            np.stack([np.tile(x, (SEQ_LEN, 1)) for x in X], axis=0),
            dtype=torch.float32
        )
        with torch.no_grad():
            return model(t).numpy()

    explainer  = shap.KernelExplainer(model_predict, X_bg)
    shap_vals  = explainer.shap_values(X_eval, nsamples=80)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_eval, feature_names=feat_cols,
                      show=False, plot_size=None)
    plt.title("SHAP feature importance — RUL prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "11_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 11_shap_summary.png")

except Exception as e:
    print(f"SHAP skipped: {e}")

print("\n✅ Evaluation complete. All figures saved to notebooks/figures/")
