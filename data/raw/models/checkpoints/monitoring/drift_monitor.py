"""
Data & Model Drift Monitoring using Evidently AI
Run: python src/monitoring/drift_monitor.py
"""

"""
Data Drift Monitoring — KS-test based
Run: python src/monitoring/drift_monitor.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp

sns.set_theme(style="whitegrid")

OUT  = Path("notebooks/figures")
PROC = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
feat_cols = pd.read_csv(PROC / "feature_cols.csv", header=None)[0].tolist()
train_df  = pd.read_parquet(PROC / "train.parquet")[feat_cols]
test_df   = pd.read_parquet(PROC / "test.parquet")[feat_cols].fillna(0)

reference = train_df.sample(min(2000, len(train_df)), random_state=42).reset_index(drop=True)
current   = test_df.sample(min(1000, len(test_df)),  random_state=42).reset_index(drop=True)

print(f"Reference samples: {len(reference)}  |  Current samples: {len(current)}")

# ── KS-test per feature ───────────────────────────────────────────────────────
print("Running KS-tests...")
results = []
for col in feat_cols:
    ref_vals = reference[col].dropna().values
    cur_vals = current[col].dropna().values
    stat, p_val = ks_2samp(ref_vals, cur_vals)
    results.append({
        "Feature":        col,
        "KS Statistic":   round(stat, 4),
        "P-Value":        round(p_val, 4),
        "Drift Detected": p_val < 0.05,
    })

drift_df = pd.DataFrame(results).sort_values("KS Statistic", ascending=True)
drift_df.to_csv(OUT / "drift_ks_results.csv", index=False)

n_drift = drift_df["Drift Detected"].sum()
print(f"\nDrift detected in {n_drift}/{len(drift_df)} features")
print(drift_df.to_string(index=False))

# ── Plot 1: KS statistic bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

colors = ["#E24B4A" if d else "#1D9E75" for d in drift_df["Drift Detected"]]
bars = ax.barh(drift_df["Feature"], drift_df["KS Statistic"],
               color=colors, edgecolor="white", linewidth=0.5, height=0.7)

for bar, val in zip(bars, drift_df["KS Statistic"]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left", fontsize=8,
            color="#2C2C2A", fontweight="bold")

ax.axvline(0.1, color="#BA7517", linestyle="--", linewidth=1.5,
           label="Drift threshold (KS > 0.10, p < 0.05)")
ax.set_xlabel("KS Statistic", fontsize=11)
ax.set_title(f"Feature Drift Detection — KS Test\n"
             f"({n_drift} of {len(drift_df)} features show drift: train vs test distribution)",
             fontsize=12, pad=12)

red_patch   = mpatches.Patch(color="#E24B4A", label=f"Drift detected ({n_drift})")
green_patch = mpatches.Patch(color="#1D9E75", label=f"No drift ({len(drift_df)-n_drift})")
ax.legend(handles=[red_patch, green_patch,
                   plt.Line2D([0],[0], color="#BA7517", linestyle="--", linewidth=1.5,
                               label="Drift threshold")],
          fontsize=9, loc="lower right")

ax.set_xlim(0, drift_df["KS Statistic"].max() * 1.18)
plt.tight_layout()
plt.savefig(OUT / "12_drift_detection.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/figures/12_drift_detection.png")

# ── Plot 2: Distribution comparison for top drifted features ─────────────────
top_drift = drift_df[drift_df["Drift Detected"]].nlargest(4, "KS Statistic")["Feature"].tolist()

if top_drift:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, top_drift[:4]):
        ax.hist(reference[col].dropna(), bins=40, alpha=0.6,
                color="#185FA5", label="Reference (train)", density=True)
        ax.hist(current[col].dropna(),   bins=40, alpha=0.6,
                color="#E24B4A", label="Current (test)",    density=True)
        ks = drift_df[drift_df["Feature"]==col]["KS Statistic"].values[0]
        ax.set_title(f"{col}  (KS={ks:.3f})", fontsize=10)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for i in range(len(top_drift), 4):
        axes[i].set_visible(False)
    plt.suptitle("Distribution Shift — Top Drifted Features", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "12b_drift_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/figures/12b_drift_distributions.png")

print("\nDrift monitoring complete!")

# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pandas as pd
# import numpy as np
# from pathlib import Path

# OUT = Path("notebooks/figures")
# OUT.mkdir(parents=True, exist_ok=True)
# PROC = Path("data/processed")

# feat_cols = pd.read_csv(PROC / "feature_cols.csv", header=None)[0].tolist()
# train_df  = pd.read_parquet(PROC / "train.parquet")[feat_cols + ["RUL"]]
# test_df   = pd.read_parquet(PROC / "test.parquet")[feat_cols].assign(RUL=np.nan)

# # Sample reference (train) and current (test) windows
# reference = train_df.sample(min(2000, len(train_df)), random_state=42).reset_index(drop=True)
# current   = test_df.sample(min(1000, len(test_df)), random_state=42).fillna(0).reset_index(drop=True)

# try:
#     from evidently.report import Report
#     from evidently.metric_preset import DataDriftPreset, DataQualityPreset

#     report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
#     report.run(reference_data=reference, current_data=current)
#     report_path = OUT / "drift_report.html"
#     report.save_html(str(report_path))
#     print(f"✅ Drift report saved → {report_path}")
#     print("Open in browser to view feature drift scores and KS-test results.")

# except ImportError:
#     print("Evidently not installed. Installing now...")
#     os.system("pip install evidently")
#     print("Re-run this script after installation.")

# except Exception as e:
#     # Fallback: manual KS-test drift report
#     print(f"Evidently error: {e}")
#     print("Falling back to manual KS-test drift detection...")

#     from scipy.stats import ks_2samp
#     import matplotlib.pyplot as plt

#     drift_results = []
#     for col in feat_cols:
#         ref_vals = reference[col].dropna().values
#         cur_vals = current[col].dropna().values
#         if len(ref_vals) > 10 and len(cur_vals) > 10:
#             stat, p_val = ks_2samp(ref_vals, cur_vals)
#             drift_results.append({
#                 "feature": col,
#                 "KS statistic": round(stat, 4),
#                 "p-value": round(p_val, 4),
#                 "drift detected": p_val < 0.05
#             })

#     drift_df = pd.DataFrame(drift_results).sort_values("KS statistic", ascending=False)
#     print("\nKS-Test Drift Results:")
#     print(drift_df.to_string(index=False))
#     drift_df.to_csv(OUT / "drift_ks_results.csv", index=False)

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     colors = ["#E24B4A" if d else "#1D9E75" for d in drift_df["drift detected"]]
#     ax.barh(drift_df["feature"], drift_df["KS statistic"], color=colors)
#     ax.axvline(0.1, color="gray", linestyle="--", linewidth=1, label="Drift threshold (0.1)")
#     ax.set_xlabel("KS statistic")
#     ax.set_title("Feature drift detection (red = drift detected)")
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(OUT / "12_drift_detection.png", dpi=150, bbox_inches="tight")
#     plt.show()
#     print("Saved: 12_drift_detection.png")

