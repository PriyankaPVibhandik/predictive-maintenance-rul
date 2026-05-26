"""
Exploratory Data Analysis — NASA C-MAPSS FD001
Run: jupyter notebook  OR  python notebooks/eda.py
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.fft import fft, fftfreq

sns.set_theme(style="whitegrid", palette="muted")
PROC = Path("data/processed")
RAW  = Path("data/raw")
OUT  = Path("notebooks/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_parquet(PROC / "train.parquet")
val   = pd.read_parquet(PROC / "val.parquet")
test  = pd.read_parquet(PROC / "test.parquet")
feat_cols = pd.read_csv(PROC / "feature_cols.csv", header=None)[0].tolist()
sensor_cols = [c for c in feat_cols if c.startswith("s_")]
op_cols     = [c for c in feat_cols if c.startswith("op_")]

print(f"Train: {train.shape}  Val: {val.shape}  Test: {test.shape}")
print(f"Features: {len(feat_cols)}  Sensors: {len(sensor_cols)}")
print(train.describe().round(3).to_string())

# ── 1. RUL distribution ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (df, name) in zip(axes, [(train, "Train"), (val, "Val"), (test, "Test")]):
    rul = df["RUL"].dropna()
    ax.hist(rul, bins=40, color="#1D9E75", edgecolor="white", linewidth=0.4)
    ax.axvline(30, color="#E24B4A", linestyle="--", linewidth=1.5, label="Alert threshold")
    ax.set_title(f"{name} RUL distribution", fontsize=12)
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "01_rul_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_rul_distribution.png")

# ── 2. Sensor correlation heatmap ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
corr = train[sensor_cols + ["RUL"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.3, ax=ax, annot_kws={"size": 8})
ax.set_title("Sensor & RUL correlation matrix", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(OUT / "02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 02_correlation_heatmap.png")

# ── 3. Sensor trends for one unit ────────────────────────────────────────────
unit_df = train[train["unit"] == 1].sort_values("cycle")
plot_sensors = sensor_cols[:6]
fig, axes = plt.subplots(len(plot_sensors), 1, figsize=(14, 2.5 * len(plot_sensors)), sharex=True)
for ax, col in zip(axes, plot_sensors):
    ax.plot(unit_df["cycle"], unit_df[col], linewidth=1, color="#185FA5")
    ax.set_ylabel(col, fontsize=9)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Cycle")
axes[0].set_title("Sensor readings over lifetime — Unit 1", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "03_sensor_trends_unit1.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 03_sensor_trends_unit1.png")

# ── 4. RUL vs cycle for sample units ────────────────────────────────────────
sample_units = train["unit"].unique()[:8]
fig, ax = plt.subplots(figsize=(12, 5))
for uid in sample_units:
    u = train[train["unit"] == uid].sort_values("cycle")
    ax.plot(u["cycle"], u["RUL"], linewidth=1.2, alpha=0.7, label=f"Unit {uid}")
ax.axhline(30, color="#E24B4A", linestyle="--", linewidth=1.5, label="Alert threshold")
ax.set_xlabel("Cycle")
ax.set_ylabel("RUL (cycles)")
ax.set_title("RUL degradation curves — sample units", fontsize=12)
ax.legend(fontsize=8, ncol=4)
plt.tight_layout()
plt.savefig(OUT / "04_rul_degradation_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 04_rul_degradation_curves.png")

# ── 5. FFT spectrum analysis ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes = axes.flatten()
for ax, col in zip(axes, sensor_cols[:6]):
    signal = unit_df[col].values - unit_df[col].mean()
    N = len(signal)
    freqs = fftfreq(N, d=1)[:N // 2]
    amps   = np.abs(fft(signal))[:N // 2] * 2 / N
    ax.plot(freqs, amps, linewidth=0.8, color="#534AB7")
    ax.set_title(f"FFT — {col}", fontsize=10)
    ax.set_xlabel("Frequency (1/cycle)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.3)
plt.suptitle("Frequency spectrum analysis — Unit 1", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "05_fft_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 05_fft_spectrum.png")

# ── 6. ADF stationarity test ─────────────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller
results = []
for col in sensor_cols:
    sig = unit_df[col].dropna().values
    if len(sig) > 20:
        adf_stat, p_val, _, _, _, _ = adfuller(sig, maxlag=5)
        results.append({"sensor": col, "ADF statistic": round(adf_stat, 3),
                         "p-value": round(p_val, 4), "stationary": p_val < 0.05})
adf_df = pd.DataFrame(results)
print("\nADF Stationarity Test Results:")
print(adf_df.to_string(index=False))
adf_df.to_csv(OUT / "adf_stationarity.csv", index=False)

# ── 7. Missing value & variance analysis ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Variance
var_series = train[feat_cols].var().sort_values(ascending=True)
var_series.plot(kind="barh", ax=axes[0], color="#1D9E75", edgecolor="white")
axes[0].set_title("Feature variance (normalised data)", fontsize=11)
axes[0].set_xlabel("Variance")
# Missing values
miss = train.isnull().sum()
miss = miss[miss > 0] if miss.any() else pd.Series({"No missing values": 0})
miss.plot(kind="bar", ax=axes[1], color="#E24B4A", edgecolor="white")
axes[1].set_title("Missing values per column", fontsize=11)
axes[1].set_xlabel("Column")
axes[1].set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT / "06_variance_missing.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 06_variance_missing.png")

# ── 8. Operating condition clusters ──────────────────────────────────────────
if len(op_cols) >= 2:
    fig, ax = plt.subplots(figsize=(8, 6))
    sample = train.sample(min(3000, len(train)), random_state=42)
    sc = ax.scatter(sample[op_cols[0]], sample[op_cols[1]],
                    c=sample["RUL"], cmap="RdYlGn", s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="RUL")
    ax.set_xlabel(op_cols[0])
    ax.set_ylabel(op_cols[1])
    ax.set_title("Operating conditions coloured by RUL", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "07_operating_conditions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 07_operating_conditions.png")

print("\n✅ EDA complete. Figures saved to notebooks/figures/")
