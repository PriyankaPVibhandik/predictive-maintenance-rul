import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

"""
Streamlit Predictive Maintenance Dashboard
Run: streamlit run src/dashboard/app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

st.set_page_config(
    page_title="Predictive Maintenance Platform",
    page_icon="🔧",
    layout="wide",
)


@st.cache_data
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data(proc_dir: str):
    proc = Path(proc_dir)
    dfs = {}
    for split in ("train", "val", "test"):
        path = proc / f"{split}.parquet"
        if path.exists():
            dfs[split] = pd.read_parquet(path)
    return dfs


def rul_gauge(rul_value: float, threshold: int = 30) -> go.Figure:
    color = "red" if rul_value < threshold * 0.5 else "orange" if rul_value < threshold else "green"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul_value,
            title={"text": "Predicted RUL (cycles)"},
            gauge={
                "axis": {"range": [0, 130]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 0.5], "color": "rgba(255,0,0,0.2)"},
                    {"range": [threshold * 0.5, threshold], "color": "rgba(255,165,0,0.2)"},
                    {"range": [threshold, 130], "color": "rgba(0,128,0,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=40, b=20))
    return fig


def main():
    cfg = load_config()
    proc_dir = cfg["data"]["processed_dir"]
    rul_threshold = cfg.get("model", {}).get("rul_threshold", 30)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Controls")
    page = st.sidebar.radio(
        "Navigate", ["📊 Overview", "🔍 Unit Deep Dive", "🗓️ Maintenance Schedule", "📈 Model Metrics"]
    )

    st.title("🔧 Predictive Maintenance Platform")
    st.caption(f"Dataset: NASA C-MAPSS {cfg['data']['cmapss_subset']} | RUL Alert Threshold: {rul_threshold} cycles")

    dfs = load_data(proc_dir)
    if not dfs:
        st.warning("No processed data found. Run `python -m src.data.preprocess` first.")
        return

    train_df = dfs.get("train", pd.DataFrame())

    # ── Overview ─────────────────────────────────────────────────────────────
    if page == "📊 Overview":
        st.subheader("Fleet Overview")

        if train_df.empty:
            st.info("Training data not available.")
            return

        last_rul = (
            train_df.sort_values("cycle")
            .groupby("unit")
            .last()
            .reset_index()[["unit", "RUL"]]
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Units", len(last_rul))
        col2.metric("Critical (<15 cycles)", int((last_rul["RUL"] < 15).sum()))
        col3.metric("Warning (<30 cycles)", int((last_rul["RUL"] < 30).sum()))
        col4.metric("Mean RUL", f"{last_rul['RUL'].mean():.1f}")

        fig = px.histogram(
            last_rul, x="RUL", nbins=30,
            title="Distribution of Current RUL Across Fleet",
            color_discrete_sequence=["#1f77b4"],
        )
        fig.add_vline(x=rul_threshold, line_dash="dash", line_color="red",
                      annotation_text="Alert threshold")
        st.plotly_chart(fig, use_container_width=True)

    # ── Unit Deep Dive ────────────────────────────────────────────────────────
    elif page == "🔍 Unit Deep Dive":
        if train_df.empty:
            st.info("Training data not available.")
            return

        units = sorted(train_df["unit"].unique())
        selected_unit = st.sidebar.selectbox("Select Engine Unit", units)

        unit_df = train_df[train_df["unit"] == selected_unit].sort_values("cycle")
        latest_rul = unit_df["RUL"].iloc[-1]

        st.subheader(f"Engine Unit {selected_unit}")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(rul_gauge(latest_rul, rul_threshold), use_container_width=True)

        with col2:
            fig = px.line(unit_df, x="cycle", y="RUL", title="RUL Over Time")
            fig.add_hline(y=rul_threshold, line_dash="dash", line_color="orange")
            st.plotly_chart(fig, use_container_width=True)

        feature_cols_path = Path(proc_dir) / "feature_cols.csv"
        if feature_cols_path.exists():
            feature_cols = pd.read_csv(feature_cols_path, header=None)[0].tolist()
            sensor_cols = [c for c in feature_cols if c.startswith("s_")]
            selected_sensors = st.multiselect("Sensors to plot", sensor_cols, default=sensor_cols[:4])
            if selected_sensors:
                fig = px.line(unit_df, x="cycle", y=selected_sensors, title="Sensor Readings")
                st.plotly_chart(fig, use_container_width=True)

    # ── Maintenance Schedule ──────────────────────────────────────────────────
    elif page == "🗓️ Maintenance Schedule":
        st.subheader("MILP Maintenance Schedule")

        try:
            import numpy as np
            from src.scheduler.milp_scheduler import EngineUnit, schedule_maintenance

            np.random.seed(42)
            demo_units = [
                EngineUnit(f"ENG-{i:03d}", predicted_rul=float(np.random.randint(5, 50)))
                for i in range(1, 11)
            ]
            sched_df = schedule_maintenance(demo_units, planning_horizon=30, max_per_day=3)
            st.dataframe(sched_df, use_container_width=True)

            fig = px.bar(
                sched_df, x="scheduled_day", y="unit_id",
                color="on_time",
                color_discrete_map={True: "green", False: "red"},
                title="Maintenance Schedule (green = on-time, red = overdue risk)",
                orientation="h",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Scheduler error: {e}")

    # ── Model Metrics ─────────────────────────────────────────────────────────
    elif page == "📈 Model Metrics":
        st.subheader("MLflow Experiment Tracker")
        tracking_uri = cfg["mlflow"]["tracking_uri"]
        st.markdown(f"MLflow UI: [http://localhost:5000](http://localhost:5000)")
        st.info("Start MLflow with: `mlflow ui --port 5000`")


if __name__ == "__main__":
    main()