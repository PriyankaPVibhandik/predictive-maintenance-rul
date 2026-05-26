"""
Generate all missing figures:
- 03_sensor_trends_unit1.png
- 10_classification_metrics.png  (Confusion Matrix + ROC + PR)
- 11_shap_summary.png
- 04_rul_degradation_curves.png  (bonus)

Run: python notebooks/generate_missing_figures.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_recall_curve, mean_absolute_error,
                              mean_squared_error)

sns.set_theme(style="whitegrid", palette="muted")

PROC = Path("data/processed")
CKPT = Path("models/checkpoints")
OUT  = Path("notebooks/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
feat_cols   = pd.read_csv(PROC / "feature_cols.csv", header=None)[0].tolist()
sensor_cols = [c for c in feat_cols if c.startswith("s_")]
train_df    = pd.read_parquet(PROC / "train.parquet")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 03 — Sensor trends for Unit 1
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating 03_sensor_trends_unit1.png ...")
unit_df = train_df[train_df["unit"] == 1].sort_values("cycle")
plot_sensors = sensor_cols[:8]

fig, axes = plt.subplots(len(plot_sensors), 1,
                          figsize=(14, 2.8 * len(plot_sensors)), sharex=True)
colors_list = ["#185FA5", "#1D9E75", "#534AB7", "#E24B4A",
               "#BA7517", "#0F6E56", "#8B1A1A", "#2C5F8A"]

for ax, col, color in zip(axes, plot_sensors, colors_list):
    ax.plot(unit_df["cycle"], unit_df[col], linewidth=1.2,
            color=color, alpha=0.85)
    ax.fill_between(unit_df["cycle"], unit_df[col],
                    alpha=0.08, color=color)
    ax.set_ylabel(col, fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# Add failure zone shading
for ax in axes:
    ax.axvspan(unit_df["cycle"].max() - 30, unit_df["cycle"].max(),
               alpha=0.08, color="#E24B4A", label="Critical zone")

axes[-1].set_xlabel("Engine Cycle", fontsize=10)
axes[0].set_title("Sensor Readings Over Engine Lifetime — Unit 1\n"
                  "(Red shading = final 30 cycles before failure)",
                  fontsize=12, pad=10)

plt.tight_layout()
plt.savefig(OUT / "03_sensor_trends_unit1.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 03_sensor_trends_unit1.png ✅")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 04 — RUL degradation curves (bonus, useful for report)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating 04_rul_degradation_curves.png ...")
sample_units = sorted(train_df["unit"].unique())[:10]

fig, ax = plt.subplots(figsize=(13, 6))
palette = plt.cm.tab10(np.linspace(0, 1, len(sample_units)))

for uid, color in zip(sample_units, palette):
    u = train_df[train_df["unit"] == uid].sort_values("cycle")
    ax.plot(u["cycle"], u["RUL"], linewidth=1.5,
            alpha=0.75, color=color, label=f"Unit {uid}")

ax.axhline(30, color="#E24B4A", linestyle="--", linewidth=2,
           label="Alert threshold (RUL=30)")
ax.axhline(15, color="#8B1A1A", linestyle=":", linewidth=1.5,
           label="Critical threshold (RUL=15)")
ax.fill_between([0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 400],
                0, 30, alpha=0.05, color="#E24B4A")

ax.set_xlabel("Engine Cycle", fontsize=11)
ax.set_ylabel("Remaining Useful Life (cycles)", fontsize=11)
ax.set_title("RUL Degradation Curves — Sample Engine Units\n"
             "(Piece-wise linear RUL labelling, clip=125)", fontsize=12)
ax.legend(fontsize=8, ncol=5, loc="upper right")
ax.set_ylim(0, 130)
plt.tight_layout()
plt.savefig(OUT / "04_rul_degradation_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 04_rul_degradation_curves.png ✅")

# ══════════════════════════════════════════════════════════════════════════════
# Load model for predictions
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading LSTM model...")
from src.models.rul_model import RULLSTMModel
from src.data.dataset import CMAPSSDataset

ckpts = sorted(CKPT.rglob("*.ckpt"))
if not ckpts:
    print("ERROR: No checkpoint found. Cannot generate figures 10 and 11.")
    sys.exit(1)

# Pick checkpoint with lowest loss (first by name sort)
best_ckpt = str(ckpts[0])
print(f"   Using: {best_ckpt}")
model = RULLSTMModel.load_from_checkpoint(best_ckpt)
model.eval()

SEQ_LEN  = 30
test_ds  = CMAPSSDataset(PROC / "test.parquet", feat_cols, seq_len=SEQ_LEN)
loader   = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

y_true_all, y_pred_all = [], []
with torch.no_grad():
    for x, y in loader:
        preds = model(x)
        y_true_all.extend(y.numpy())
        y_pred_all.extend(preds.numpy())

y_true = np.array(y_true_all)
y_pred = np.array(y_pred_all)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Confusion Matrix + ROC + PR curves
# ══════════════════════════════════════════════════════════════════════════════
print("Generating 10_classification_metrics.png ...")
THRESHOLD = 30
y_true_bin = (y_true < THRESHOLD).astype(int)
y_pred_bin = (y_pred < THRESHOLD).astype(int)

cm = confusion_matrix(y_true_bin, y_pred_bin)

fig = plt.figure(figsize=(17, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# ── Confusion matrix ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["#FFFFFF", "#1D9E75"])
sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax0,
            xticklabels=["Normal", "Critical"],
            yticklabels=["Normal", "Critical"],
            linewidths=0.5, linecolor="#CCCCCC",
            annot_kws={"size": 14, "weight": "bold"})
ax0.set_title(f"Confusion Matrix\n(Critical = RUL < {THRESHOLD} cycles)",
              fontsize=11, pad=10)
ax0.set_ylabel("Actual", fontsize=10)
ax0.set_xlabel("Predicted", fontsize=10)

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp + 1e-9)
recall    = tp / (tp + fn + 1e-9)
f1        = 2 * precision * recall / (precision + recall + 1e-9)
ax0.text(0.5, -0.18,
         f"Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}",
         transform=ax0.transAxes, ha="center", fontsize=9,
         color="#2C2C2A", style="italic")

# ── ROC curve ────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
fpr, tpr, _ = roc_curve(y_true_bin, -y_pred)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color="#185FA5", lw=2.5,
         label=f"LSTM (AUC = {roc_auc:.3f})")
ax1.fill_between(fpr, tpr, alpha=0.08, color="#185FA5")
ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
ax1.set_xlabel("False Positive Rate", fontsize=10)
ax1.set_ylabel("True Positive Rate", fontsize=10)
ax1.set_title("ROC Curve — Critical Failure Detection", fontsize=11, pad=10)
ax1.legend(fontsize=9)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.grid(True, alpha=0.3)

# ── Precision-Recall curve ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
prec, rec, _ = precision_recall_curve(y_true_bin, -y_pred)
pr_auc = auc(rec, prec)
ax2.plot(rec, prec, color="#534AB7", lw=2.5,
         label=f"LSTM (AUC = {pr_auc:.3f})")
ax2.fill_between(rec, prec, alpha=0.08, color="#534AB7")
baseline = y_true_bin.mean()
ax2.axhline(baseline, color="#BA7517", linestyle="--", lw=1.5,
            label=f"Baseline ({baseline:.2f})")
ax2.set_xlabel("Recall", fontsize=10)
ax2.set_ylabel("Precision", fontsize=10)
ax2.set_title("Precision-Recall Curve", fontsize=11, pad=10)
ax2.legend(fontsize=9)
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.05])
ax2.grid(True, alpha=0.3)

fig.suptitle(f"Critical Failure Detection Metrics  |  MAE={mae:.1f} cycles  "
             f"MAPE={mape:.1f}%  RMSE={rmse:.1f}",
             fontsize=12, y=1.02, fontweight="bold")

plt.savefig(OUT / "10_classification_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: 10_classification_metrics.png ✅")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — SHAP Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
print("Generating 11_shap_summary.png ...")
print("   (Computing SHAP — takes 1-3 minutes, please wait...)")

try:
    import shap

    test_df_raw = pd.read_parquet(PROC / "test.parquet").fillna(0)
    X_all = test_df_raw[feat_cols].values.astype(np.float32)

    np.random.seed(42)
    X_bg   = X_all[np.random.choice(len(X_all), 150, replace=False)]
    X_eval = X_all[np.random.choice(len(X_all),  80, replace=False)]

    def predict_fn(X):
        t = torch.tensor(
            np.stack([np.tile(x, (SEQ_LEN, 1)) for x in X], axis=0),
            dtype=torch.float32
        )
        with torch.no_grad():
            return model(t).numpy()

    explainer  = shap.KernelExplainer(predict_fn, X_bg)
    shap_vals  = explainer.shap_values(X_eval, nsamples=100)

    # ── SHAP bar chart (mean absolute) ────────────────────────────────────────
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_df   = pd.DataFrame({"Feature": feat_cols,
                               "Mean |SHAP|": mean_shap}
                             ).sort_values("Mean |SHAP|", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart
    bar_colors = plt.cm.RdYlGn(
        np.linspace(0.2, 0.9, len(shap_df)))
    axes[0].barh(shap_df["Feature"], shap_df["Mean |SHAP|"],
                 color=bar_colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Mean |SHAP value|  (impact on RUL prediction)", fontsize=10)
    axes[0].set_title("Feature Importance — Mean |SHAP|", fontsize=11, pad=10)
    axes[0].grid(True, alpha=0.3, axis="x")

    for i, (val, feat) in enumerate(zip(shap_df["Mean |SHAP|"],
                                         shap_df["Feature"])):
        axes[0].text(val + 0.0005, i, f"{val:.4f}",
                     va="center", fontsize=8, color="#2C2C2A")

    # Dot plot (beeswarm approximation)
    top_features = shap_df.nlargest(10, "Mean |SHAP|")["Feature"].tolist()
    top_idx      = [feat_cols.index(f) for f in top_features]
    shap_top     = shap_vals[:, top_idx]
    X_top        = X_eval[:, top_idx]

    # Normalise feature values for colour
    X_norm = (X_top - X_top.min(0)) / (X_top.ptp(0) + 1e-9)

    for j, (feat, fi) in enumerate(zip(reversed(top_features),
                                        range(len(top_features)))):
        sv   = shap_top[:, len(top_features)-1-j]
        xn   = X_norm[:, len(top_features)-1-j]
        jitter = np.random.uniform(-0.25, 0.25, len(sv))
        sc = axes[1].scatter(sv, fi + jitter, c=xn,
                             cmap="RdBu_r", s=18, alpha=0.6,
                             vmin=0, vmax=1)

    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(list(reversed(top_features)), fontsize=9)
    axes[1].axvline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("SHAP value (impact on RUL)", fontsize=10)
    axes[1].set_title("SHAP Beeswarm — Top 10 Features\n"
                       "(Blue=low feature value, Red=high)", fontsize=11, pad=10)
    axes[1].grid(True, alpha=0.3, axis="x")
    plt.colorbar(sc, ax=axes[1], label="Normalised feature value",
                 fraction=0.03, pad=0.04)

    plt.suptitle("SHAP Feature Importance — RUL LSTM Model",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "11_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 11_shap_summary.png ✅")

except Exception as e:
    print(f"   SHAP failed: {e}")
    print("   Generating feature importance from weights instead...")

    # Fallback: gradient-based importance
    model.eval()
    test_ds2 = CMAPSSDataset(PROC / "test.parquet", feat_cols, seq_len=SEQ_LEN)
    sample_x = torch.stack([test_ds2[i][0] for i in range(min(200, len(test_ds2)))])
    sample_x.requires_grad_(True)
    out = model(sample_x)
    out.sum().backward()
    importance = sample_x.grad.abs().mean(dim=(0,1)).detach().numpy()

    imp_df = pd.DataFrame({"Feature": feat_cols,
                            "Importance": importance}
                          ).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp_df)))
    ax.barh(imp_df["Feature"], imp_df["Importance"],
            color=colors_imp, edgecolor="white")
    ax.set_xlabel("Gradient-based Importance (proxy for SHAP)", fontsize=10)
    ax.set_title("Feature Importance — RUL LSTM\n(Gradient magnitude w.r.t. input)",
                 fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT / "11_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 11_shap_summary.png (gradient-based) ✅")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  All figures generated successfully!")
print("="*55)
generated = list(OUT.glob("*.png"))
for f in sorted(generated):
    print(f"  ✅ {f.name}")
print(f"\n  Total: {len(generated)} figures in notebooks/figures/")
