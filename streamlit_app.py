"""Premium Streamlit frontend for Churn Prediction API.

Run:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.preprocessing import LabelEncoder

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Northstar Churn Desk",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --bg: #0f1724;
    --bg-soft: #152337;
    --surface: rgba(17, 27, 41, 0.74);
    --surface-strong: rgba(15, 24, 37, 0.9);
    --ink: #f2f6fc;
    --muted: #9eb1c7;
    --accent: #3aa4ff;
    --accent-2: #7ce7ff;
    --danger: #ff6f83;
    --gold: #f5b955;
    --border: rgba(124, 184, 255, 0.24);
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 430px at 6% -5%, rgba(58, 164, 255, 0.28), transparent 62%),
        radial-gradient(760px 360px at 98% 8%, rgba(124, 231, 255, 0.16), transparent 58%),
        radial-gradient(900px 420px at 48% 100%, rgba(40, 72, 115, 0.32), transparent 60%),
        linear-gradient(168deg, #0a111c 0%, var(--bg) 58%, var(--bg-soft) 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1321 0%, #0e1a2c 100%);
    border-right: 1px solid rgba(124, 184, 255, 0.2);
}

[data-testid="stSidebar"] * {
    color: #eef5ff !important;
}

h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif;
    color: #f4f9ff;
    letter-spacing: 0.2px;
}

.hero {
    background:
        linear-gradient(130deg, rgba(58, 164, 255, 0.24), rgba(124, 231, 255, 0.08)),
        var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.45rem 1.55rem;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.35);
    animation: fade-up 0.65s ease-out;
}

.hero .title {
    font-size: 2.08rem;
    color: #f4f9ff;
    margin-bottom: 0.2rem;
}

.hero .subtitle {
    color: var(--muted);
    font-size: 0.97rem;
}

.glass {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1rem 0.85rem 1rem;
    backdrop-filter: blur(11px);
    box-shadow: 0 12px 26px rgba(0, 0, 0, 0.22);
    animation: fade-up 0.65s ease-out;
}

.kpi-card {
    background: linear-gradient(155deg, rgba(58, 164, 255, 0.2), rgba(10, 18, 30, 0.76));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.92rem 1rem;
    min-height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.3);
}

.kpi-label {
    color: #9eb4ca;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}

.kpi-value {
    color: #f1f7ff;
    font-size: 1.2rem;
    font-weight: 700;
}

.section-kicker {
    color: var(--accent-2);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.7rem;
    margin-top: 0.2rem;
}

.section-title {
    margin-top: 0.1rem;
    margin-bottom: 0;
    font-size: 1.56rem;
    color: #f4f9ff;
}

.section-subtitle {
    margin-top: 0.2rem;
    margin-bottom: 0.85rem;
    color: var(--muted);
    font-size: 0.94rem;
}

.risk-good { color: #68e3b5; font-weight: 700; }
.risk-mid { color: var(--gold); font-weight: 700; }
.risk-bad { color: var(--danger); font-weight: 700; }

.status-pill {
    display: inline-flex;
    gap: 0.44rem;
    align-items: center;
    border-radius: 999px;
    padding: 0.32rem 0.72rem;
    border: 1px solid var(--border);
    background: rgba(14, 24, 38, 0.85);
    color: #e6f1ff;
    font-size: 0.8rem;
    margin-top: 0.62rem;
}

.status-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: #68e3b5;
    box-shadow: 0 0 0 6px rgba(104, 227, 181, 0.16);
}

.status-dot.offline {
    background: var(--danger);
    box-shadow: 0 0 0 6px rgba(255, 111, 131, 0.16);
}

[data-testid="stMetric"] {
    background: rgba(20, 31, 47, 0.84);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.66rem;
}

.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(124, 184, 255, 0.54);
    background: linear-gradient(90deg, #2c8bf0, #42b3ff);
    color: #ecf7ff;
    font-weight: 700;
}

.stButton > button:hover {
    border-color: #7ce7ff;
    box-shadow: 0 0 0 3px rgba(58, 164, 255, 0.2);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
}

@media (max-width: 900px) {
    .hero .title {
        font-size: 1.6rem;
    }

    .section-title {
        font-size: 1.23rem;
    }
}

@keyframes fade-up {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""",
        unsafe_allow_html=True,
)


@st.cache_data(ttl=20)
def get_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=20)
def get_features() -> List[str]:
    try:
        response = requests.get(f"{API_URL}/features", timeout=4)
        response.raise_for_status()
        return response.json().get("features", [])
    except requests.RequestException:
        return []


def predict_single(features: List[float]) -> dict | None:
    try:
        response = requests.post(
            f"{API_URL}/predict", json={"features": features}, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
        return None


def predict_batch(instances: List[List[float]]) -> dict | None:
    try:
        response = requests.post(
            f"{API_URL}/predict/batch", json={"instances": instances}, timeout=15
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Batch request failed: {exc}")
        return None


def parse_vector(text: str) -> List[float] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        if cleaned.startswith("["):
            values = json.loads(cleaned)
        else:
            values = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
        return [float(v) for v in values]
    except (ValueError, json.JSONDecodeError):
        return None


def risk_label(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.35:
        return "MEDIUM"
    return "LOW"


def render_section_heading(title: str, subtitle: str, kicker: str = "") -> None:
        kicker_html = f"<div class='section-kicker'>{kicker}</div>" if kicker else ""
        st.markdown(
                f"""
<div>
    {kicker_html}
    <h2 class='section-title'>{title}</h2>
    <p class='section-subtitle'>{subtitle}</p>
</div>
""",
                unsafe_allow_html=True,
        )


def render_kpi_card(label: str, value: str):
        st.markdown(
                f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
</div>
""",
                unsafe_allow_html=True,
        )


def transform_raw_telco_csv(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """Transform raw Telco CSV into model-ready numeric feature matrix."""
    transformed = df.copy()

    if "customerID" in transformed.columns:
        transformed = transformed.drop(columns=["customerID"])

    if "TotalCharges" in transformed.columns:
        transformed["TotalCharges"] = pd.to_numeric(
            transformed["TotalCharges"], errors="coerce"
        )
        transformed["TotalCharges"] = transformed["TotalCharges"].fillna(
            transformed["TotalCharges"].median()
        )

    if "Churn" in transformed.columns:
        transformed["Churn"] = (
            transformed["Churn"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)

    categorical_cols = transformed.select_dtypes(include="object").columns.tolist()
    encoder = LabelEncoder()
    for col in categorical_cols:
        transformed[col] = encoder.fit_transform(transformed[col].astype(str))

    if "Churn" in transformed.columns:
        transformed = transformed.drop(columns=["Churn"])

    for col in expected_features:
        if col not in transformed.columns:
            transformed[col] = 0

    transformed = transformed[expected_features].copy()
    transformed = transformed.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return transformed


with st.sidebar:
    st.markdown("### Northstar Churn Desk")
    st.caption("High-signal retention intelligence")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Single Prediction", "Batch Prediction", "Model Profile"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("API Endpoint")
    st.code(API_URL)

health = get_health()
features = get_features()
feature_count = len(features) if features else 19

status_text = "API ONLINE" if health else "API OFFLINE"
status_class = "status-dot" if health else "status-dot offline"


st.markdown(
    f"""
<div class="hero">
    <div class="title">Northstar Churn Desk</div>
    <div class="subtitle">Signal-driven retention intelligence for production decisioning.</div>
  <div class="status-pill"><span class="{status_class}"></span>{status_text}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

if page == "Dashboard":
    if not health:
        st.error("API is offline. Start FastAPI server on localhost:8000 first.")
    else:
        render_section_heading(
            "Command Overview",
            "Track serving health, model quality, and production readiness at a glance.",
            kicker="Live Operations",
        )
        metrics = health.get("metrics", {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card("Service Status", "Online")
        with c2:
            render_kpi_card("Accuracy", f"{metrics.get('accuracy', 0.0):.2%}")
        with c3:
            render_kpi_card("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.2%}")
        with c4:
            render_kpi_card("F1 Score", f"{metrics.get('f1_score', 0.0):.4f}")

        st.write("")
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Performance Snapshot")
            chart_df = pd.DataFrame(
                {
                    "Metric": ["Accuracy", "ROC-AUC", "F1 Score"],
                    "Score": [
                        metrics.get("accuracy", 0.0),
                        metrics.get("roc_auc", 0.0),
                        metrics.get("f1_score", 0.0),
                    ],
                }
            )
            st.bar_chart(chart_df.set_index("Metric"), color="#00c2a8")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Deployment")
            st.write("Model: XGBoost classifier")
            st.write(f"Feature vector width: {feature_count}")
            st.write("Serving: FastAPI + Uvicorn")
            st.write("Packaging: Docker + ECR")
            st.write("Pipeline gate: Accuracy >= 80%")
            st.markdown("</div>", unsafe_allow_html=True)


elif page == "Single Prediction":
    render_section_heading(
        "Single Prediction",
        "Use profile presets for speed or a custom vector for precision scoring.",
        kicker="Realtime Inference",
    )
    st.caption(f"Expected features: {feature_count}.")

    mode = st.segmented_control(
        "Input mode", ["Preset Profile", "Manual Vector"], default="Preset Profile"
    )

    if mode == "Preset Profile":
        profile = st.selectbox(
            "Profile",
            [
                "Low Risk Baseline",
                "Mid Risk Customer",
                "High Risk Customer",
            ],
        )
        if profile == "Low Risk Baseline":
            vector = [0, 48, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 45.5, 2190.0]
        elif profile == "Mid Risk Customer":
            vector = [1, 18, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 79.9, 1430.0]
        else:
            vector = [1, 6, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 96.8, 612.0]

    else:
        default_vector = ",".join(["0"] * max(feature_count - 2, 0) + ["29.85", "29.85"])
        vector_text = st.text_area(
            "Comma-separated feature vector",
            value=default_vector,
            height=110,
            help="You can also paste JSON list format such as [0,1,2,...].",
        )
        parsed = parse_vector(vector_text)
        if parsed is None:
            st.warning("Invalid vector format.")
            vector = []
        else:
            vector = parsed

    if st.button("Run Prediction", use_container_width=True):
        if len(vector) != feature_count:
            st.error(f"Feature vector must have exactly {feature_count} values.")
        else:
            result = predict_single(vector)
            if result:
                probability = float(result.get("churn_probability", 0.0))
                label = risk_label(probability)
                css_class = (
                    "risk-bad"
                    if label == "HIGH"
                    else "risk-mid"
                    if label == "MEDIUM"
                    else "risk-good"
                )
                a, b = st.columns([1.1, 1])
                with a:
                    st.markdown('<div class="glass">', unsafe_allow_html=True)
                    st.subheader("Decision")
                    st.metric("Predicted Churn", "Yes" if result.get("churn") else "No")
                    st.metric("Probability", f"{probability:.2%}")
                    st.markdown(
                        f"Risk Tier: <span class='{css_class}'>{label}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with b:
                    st.markdown('<div class="glass">', unsafe_allow_html=True)
                    st.subheader("Retention Action")
                    if label == "HIGH":
                        st.write("Offer targeted retention package in first contact.")
                        st.write("Escalate to account specialist within 24 hours.")
                    elif label == "MEDIUM":
                        st.write("Schedule proactive support outreach.")
                        st.write("Promote annual contract incentives.")
                    else:
                        st.write("Maintain service quality and monitor sentiment.")
                        st.write("Keep customer in low-touch loyalty stream.")
                    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Batch Prediction":
    render_section_heading(
        "Batch Prediction",
        "Upload customer cohorts and score churn risk in one premium batch flow.",
        kicker="Portfolio Scoring",
    )
    mode = st.radio(
        "CSV Type",
        ["Raw Telco CSV (with original columns)", "Numeric Feature CSV"],
        horizontal=True,
    )

    if mode == "Raw Telco CSV (with original columns)":
        st.caption(
            "Upload the original Telco file (columns like gender, tenure, Contract, TotalCharges, Churn). "
            "The app will auto-clean, encode, align features, and predict churn for each row."
        )
    else:
        st.caption(
            "Upload CSV with numeric features. "
            f"First {feature_count} columns are used as model features."
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Rows detected: {len(df)}")
        st.dataframe(df.head(8), use_container_width=True)

        if st.button("Run Batch Prediction", use_container_width=True):
            if mode == "Raw Telco CSV (with original columns)":
                if not features:
                    st.error("Could not fetch model feature names from API. Check /features endpoint.")
                else:
                    model_df = transform_raw_telco_csv(df, features)
                    response = predict_batch(model_df.values.tolist())
                    if response:
                        preds = response.get("predictions", [])
                        out = df.copy()
                        out["predicted_churn"] = [p.get("churn", 0) for p in preds]
                        out["churn_probability"] = [
                            p.get("churn_probability", 0.0) for p in preds
                        ]
                        out["risk_tier"] = out["churn_probability"].apply(risk_label)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rows Scored", len(out))
                        c2.metric("Predicted Churners", int(out["predicted_churn"].sum()))
                        c3.metric("Average Risk", f"{out['churn_probability'].mean():.2%}")

                        st.dataframe(out, use_container_width=True)
                        csv_bytes = out.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Scored CSV",
                            data=csv_bytes,
                            file_name="scored_telco_customers.csv",
                            mime="text/csv",
                        )
            else:
                if df.shape[1] < feature_count:
                    st.error(
                        f"Need at least {feature_count} columns. Found {df.shape[1]}."
                    )
                else:
                    instances = df.iloc[:, :feature_count].astype(float).values.tolist()
                    response = predict_batch(instances)
                    if response:
                        preds = response.get("predictions", [])
                        out = df.copy()
                        out["predicted_churn"] = [p.get("churn", 0) for p in preds]
                        out["churn_probability"] = [
                            p.get("churn_probability", 0.0) for p in preds
                        ]
                        out["risk_tier"] = out["churn_probability"].apply(risk_label)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rows Scored", len(out))
                        c2.metric("Predicted Churners", int(out["predicted_churn"].sum()))
                        c3.metric("Average Risk", f"{out['churn_probability'].mean():.2%}")

                        st.dataframe(out, use_container_width=True)
                        csv_bytes = out.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Scored CSV",
                            data=csv_bytes,
                            file_name="scored_customers.csv",
                            mime="text/csv",
                        )

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Quick Synthetic Run")
    sample_rows = st.slider("Rows", 5, 200, 30)
    if st.button("Generate and Score", use_container_width=True):
        synthetic = np.random.normal(loc=0.5, scale=0.2, size=(sample_rows, feature_count))
        synthetic = np.clip(synthetic, 0, None)
        response = predict_batch(synthetic.tolist())
        if response:
            preds = response.get("predictions", [])
            synthetic_df = pd.DataFrame(synthetic, columns=[f"feature_{i}" for i in range(feature_count)])
            synthetic_df["predicted_churn"] = [p.get("churn", 0) for p in preds]
            synthetic_df["churn_probability"] = [p.get("churn_probability", 0.0) for p in preds]
            st.dataframe(synthetic_df.head(25), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Model Profile":
    render_section_heading(
        "Model Profile",
        "Explore the deployed model's operating profile and feature interface.",
        kicker="Model Intelligence",
    )
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.write("Algorithm: Gradient-boosted ensemble")
        st.write("Runtime: FastAPI on Uvicorn")
        st.write("CI/CD: GitHub Actions")
        st.write("Registry: MLflow + ECR")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.write("Data: Telco customer churn")
        st.write(f"Expected feature count: {feature_count}")
        st.write("Decision threshold: model-native probability")
        st.write("Quality gate: Accuracy >= 0.80")
        st.markdown("</div>", unsafe_allow_html=True)

    if features:
        st.write("")
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Feature Map")
        st.dataframe(pd.DataFrame({"feature": features}), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


st.write("")
st.caption("Northstar Churn Desk - premium UI layer for the local churn inference API")
