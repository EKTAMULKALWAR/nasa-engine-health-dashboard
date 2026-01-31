import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="NASA Engine Health Dashboard", layout="wide")

st.title("NASA C-MAPSS Engine Health Monitoring")
st.caption("RUL prediction using LSTM (RUL capped). Dataset: NASA C-MAPSS FD001")

# Load data
@st.cache_data
def load_preds():
    df = pd.read_csv("predictions_lstm_fd001.csv")
    # add risk band
    df["risk_band"] = pd.cut(
        df["predicted_rul_capped"],
        bins=[-1, 30, 80, 125],
        labels=["HIGH", "MEDIUM", "LOW"]
    )
    return df

df = load_preds()

# Metrics
rmse = float(np.sqrt(mean_squared_error(df["true_rul_capped"], df["predicted_rul_capped"])))
c1, c2, c3 = st.columns(3)
c1.metric("RMSE (capped RUL)", f"{rmse:.2f}")
c2.metric("Rows", f"{len(df):,}")
c3.metric("High Risk %", f"{(df['risk_band']=='HIGH').mean()*100:.1f}%")

st.divider()

# Filters
risk = st.multiselect("Filter by risk band", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
filtered = df[df["risk_band"].isin(risk)]

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Risk Band Distribution")
    counts = filtered["risk_band"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"])
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Risk band")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with colB:
    st.subheader("Predicted vs True (sample)")
    n = st.slider("Number of points", 100, 2000, 500, step=100)
    sample = filtered.sample(min(n, len(filtered)), random_state=42)

    fig2, ax2 = plt.subplots()
    ax2.scatter(sample["true_rul_capped"], sample["predicted_rul_capped"], alpha=0.25)
    ax2.set_xlabel("True RUL (capped)")
    ax2.set_ylabel("Predicted RUL (capped)")
    ax2.set_title("Prediction Quality")
    st.pyplot(fig2)

st.divider()

st.subheader("Table (filtered)")
st.dataframe(filtered.head(50))
