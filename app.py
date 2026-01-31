import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="NASA Engine Health Dashboard", layout="wide")
st.title("NASA C-MAPSS Engine Health Monitoring Dashboard")
st.caption("LSTM Remaining Useful Life (RUL) Prediction • FD001 • RUL capped at 125")

DATA_FILE = "predictions_lstm_fd001.csv"
RUL_CAP = 125

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize column names
    rename_map = {}
    if "predicted_rul" in df.columns and "predicted_rul_capped" not in df.columns:
        rename_map["predicted_rul"] = "predicted_rul_capped"
    if "true_rul" in df.columns and "true_rul_capped" not in df.columns:
        rename_map["true_rul"] = "true_rul_capped"
    if rename_map:
        df = df.rename(columns=rename_map)

    # clip
    df["predicted_rul_capped"] = df["predicted_rul_capped"].clip(0, RUL_CAP)
    df["true_rul_capped"] = df["true_rul_capped"].clip(0, RUL_CAP)

    # risk bands
    df["risk_band"] = pd.cut(
        df["predicted_rul_capped"],
        bins=[-1, 30, 80, RUL_CAP],
        labels=["HIGH", "MEDIUM", "LOW"]
    )
    return df

def compute_rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(mean_squared_error(df["true_rul_capped"], df["predicted_rul_capped"])))

def summarize(df: pd.DataFrame) -> str:
    rmse = compute_rmse(df)
    high_pct = (df["risk_band"] == "HIGH").mean() * 100
    return (
        f"RMSE (capped RUL): {rmse:.2f}\n"
        f"Rows: {len(df):,}\n"
        f"High risk: {high_pct:.1f}%\n"
        f"Median predicted RUL: {df['predicted_rul_capped'].median():.1f}\n"
        f"Mean predicted RUL: {df['predicted_rul_capped'].mean():.1f}"
    )

def top_examples(df: pd.DataFrame, band: str, k: int = 10) -> pd.DataFrame:
    # “worst” = lowest predicted RUL (highest urgency)
    sub = df[df["risk_band"] == band].copy()
    sub = sub.sort_values("predicted_rul_capped", ascending=True).head(k)
    return sub.reset_index(drop=True)

def parse_k(text: str, default: int = 10) -> int:
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return default
    k = int(m.group(1))
    return max(1, min(100, k))

def analyst_ai_answer(question: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame | None]:
    q = question.strip().lower()

    # quick help
    if q in {"help", "examples", "what can you do", "commands"}:
        return (
            "Try questions like:\n"
            "- summary\n"
            "- how many high risk?\n"
            "- show top 10 high risk\n"
            "- average predicted RUL\n"
            "- distribution of risk\n"
            "- worst 5 predictions\n",
            None
        )

    # summary
    if "summary" in q or "overall" in q:
        return summarize(df), None

    # counts / percentages
    if "how many" in q or "count" in q:
        if "high" in q:
            n = int((df["risk_band"] == "HIGH").sum())
            return f"HIGH risk rows: {n:,}", None
        if "medium" in q:
            n = int((df["risk_band"] == "MEDIUM").sum())
            return f"MEDIUM risk rows: {n:,}", None
        if "low" in q:
            n = int((df["risk_band"] == "LOW").sum())
            return f"LOW risk rows: {n:,}", None
        # total
        return f"Total rows: {len(df):,}", None

    if "percent" in q or "%" in q:
        high = (df["risk_band"] == "HIGH").mean() * 100
        med = (df["risk_band"] == "MEDIUM").mean() * 100
        low = (df["risk_band"] == "LOW").mean() * 100
        return f"HIGH: {high:.1f}% | MEDIUM: {med:.1f}% | LOW: {low:.1f}%", None

    # averages / medians
    if "average" in q or "mean" in q:
        return f"Mean predicted RUL (capped): {df['predicted_rul_capped'].mean():.2f}", None
    if "median" in q:
        return f"Median predicted RUL (capped): {df['predicted_rul_capped'].median():.2f}", None

    # show top K by urgency
    if "show" in q or "top" in q or "worst" in q:
        k = parse_k(q, default=10)
        if "high" in q:
            tbl = top_examples(df, "HIGH", k=k)
            return f"Top {len(tbl)} HIGH risk rows (lowest predicted RUL):", tbl
        if "medium" in q:
            tbl = top_examples(df, "MEDIUM", k=k)
            return f"Top {len(tbl)} MEDIUM risk rows:", tbl
        if "low" in q:
            tbl = top_examples(df, "LOW", k=k)
            return f"Top {len(tbl)} LOW risk rows:", tbl

        # if user didn't specify band, show overall worst
        tmp = df.sort_values("predicted_rul_capped", ascending=True).head(k).reset_index(drop=True)
        return f"Worst {len(tmp)} rows overall (lowest predicted RUL):", tmp

    # distribution
    if "distribution" in q or "breakdown" in q:
        counts = df["risk_band"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0).astype(int)
        msg = "Risk distribution:\n" + "\n".join([f"- {idx}: {val:,}" for idx, val in counts.items()])
        return msg, None

    # RMSE
    if "rmse" in q or "error" in q:
        return f"RMSE (capped RUL): {compute_rmse(df):.3f}", None

    # fallback
    return (
        "I can answer questions about the displayed predictions (counts, risk breakdown, top high-risk rows, averages, RMSE).\n"
        "Type **help** to see examples.",
        None
    )

# -----------------------------
# Load
# -----------------------------
try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"Missing {DATA_FILE}. Upload it to the repo in the same folder as app.py.")
    st.stop()

rmse = compute_rmse(df)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
selected_risk = st.sidebar.multiselect(
    "Risk band",
    ["HIGH", "MEDIUM", "LOW"],
    default=["HIGH", "MEDIUM", "LOW"]
)
n_points = st.sidebar.slider("Scatter sample size", 200, 5000, 1500, step=100)
filtered = df[df["risk_band"].isin(selected_risk)].copy()

# -----------------------------
# Top metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE (capped RUL)", f"{rmse:.2f}")
c2.metric("Rows", f"{len(df):,}")
c3.metric("High Risk %", f"{(df['risk_band']=='HIGH').mean()*100:.1f}%")
c4.metric("Median Predicted RUL", f"{df['predicted_rul_capped'].median():.1f}")

st.divider()

# -----------------------------
# Visuals
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Risk Band Distribution")
    counts = filtered["risk_band"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"])
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Risk band")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with right:
    st.subheader("Predicted vs True (Sample)")
    if len(filtered) == 0:
        st.warning("No rows after filtering.")
    else:
        sample = filtered.sample(min(n_points, len(filtered)), random_state=42)
        fig2, ax2 = plt.subplots()
        ax2.scatter(sample["true_rul_capped"], sample["predicted_rul_capped"], alpha=0.25)
        ax2.set_xlabel("True RUL (capped)")
        ax2.set_ylabel("Predicted RUL (capped)")
        ax2.set_title("Prediction Quality")
        st.pyplot(fig2)

st.divider()

# -----------------------------
# AI Assistant (works with no API key)
# -----------------------------
st.subheader("AI Assistant")
st.caption("Ask questions about the predictions. (Type **help** for examples.)")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("Your question", placeholder="e.g., summary • how many high risk? • show top 10 high risk")
    submitted = st.form_submit_button("Ask")

if submitted and user_q:
    answer, table = analyst_ai_answer(user_q, df)
    st.session_state.chat.append(("You", user_q))
    st.session_state.chat.append(("Assistant", answer))
    if table is not None:
        st.session_state.chat.append(("Table", table))

for role, content in st.session_state.chat[-12:]:
    if role == "You":
        st.markdown(f"**You:** {content}")
    elif role == "Assistant":
        st.markdown(f"**Assistant:** {content}")
    else:
        st.dataframe(content)

st.divider()
st.subheader("Data Preview")
st.dataframe(filtered.head(100))
