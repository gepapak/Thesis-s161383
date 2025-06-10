import os
import streamlit as st
import wandb
import pandas as pd
import matplotlib.pyplot as plt

# --- WandB API Key ---
os.environ["WANDB_API_KEY"] = "9a27acac954ecccc53a820d856e7a6a088487f04"

# --- WandB Project Info ---
ENTITY = "s161383-technical-university-of-denmark"
PROJECT = "energy-thesis"

st.set_page_config(page_title="Energy Investment RL Dashboard", layout="wide")
st.title("🔋 Energy Investment RL Dashboard")

# --- Session Reset Button ---
if st.button("🔄 Reset App"):
    st.session_state.clear()
    st.experimental_rerun()

# --- Load WandB Runs ---
api = wandb.Api()
try:
    runs = api.runs(path=f"{ENTITY}/{PROJECT}")
except Exception as e:
    st.error(f"Failed to fetch runs: {e}")
    st.stop()

run_names = [f"{run.name} ({run.id})" for run in runs]
selected_run_display = st.sidebar.selectbox("Choose a run", run_names)
selected_run_id = selected_run_display.split("(")[-1].strip(")")
selected_run = next(run for run in runs if run.id == selected_run_id)

# --- Load Full History ---
try:
    history = pd.DataFrame([row for row in selected_run.scan_history()])
except Exception as e:
    st.error(f"Failed to load run history.\nError: {e}")
    st.stop()

# --- Sidebar Diagnostics ---
st.sidebar.markdown("### 📊 Available Metrics")
available_metrics = history.columns.tolist()
st.sidebar.write(available_metrics)

# --- Metric Plot Helper ---
def plot_metric(metric_name, label=None):
    if metric_name in history.columns:
        label = label or metric_name.replace("_", " ").title()
        fig, ax = plt.subplots()
        ax.plot(history["_step"], history[metric_name])
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning(f"Metric '{metric_name}' not found.")

# --- Plot Sections ---
st.subheader("📈 Core Performance Metrics")
for metric in ["reward", "net_profit", "revenue", "risk_penalty"]:
    plot_metric(metric)

st.subheader("🔋 Capacity Trends")
for metric in ["wind_capacity", "solar_capacity", "hydro_capacity", "battery_capacity", "battery_energy"]:
    plot_metric(metric)

st.subheader("🏁 Cumulative Results")
plot_metric("cumulative_reward")

st.subheader("🔍 Raw Data Preview")
st.dataframe(history.head(20))

st.markdown("---")
st.caption("Built for DTU Energy Thesis • Powered by WandB + Streamlit")
