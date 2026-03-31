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
    page_title="Aurelia Churn Studio",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #06131a;
  --bg-2: #0b1f29;
  --ink: #e8f0f2;
  --muted: #9db2b8;
  --accent: #00c2a8;
  --accent-soft: #7de3d6;
  --danger: #ff6b6b;
  --gold: #d4a017;
  --card: rgba(14, 35, 46, 0.66);
  --border: rgba(125, 227, 214, 0.25);
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(700px 260px at 10% 0%, rgba(0, 194, 168, 0.16), transparent 60%),
    radial-gradient(900px 380px at 100% 0%, rgba(212, 160, 23, 0.10), transparent 58%),
    linear-gradient(165deg, var(--bg) 0%, var(--bg-2) 70%);
}

[data-testid="stSidebar"] {
  background: rgba(6, 19, 26, 0.94);
  border-right: 1px solid var(--border);
}

h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: 0.2px;
}

.hero {
  background: linear-gradient(120deg, rgba(0, 194, 168, 0.28), rgba(212, 160, 23, 0.20));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.25rem 1.4rem;
  backdrop-filter: blur(7px);
  animation: fade-up 0.7s ease-out;
}

.hero .title {
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 0.15rem;
}

.hero .subtitle {
  color: var(--muted);
  font-size: 0.98rem;
}

.glass {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem 1rem 0.8rem 1rem;
  animation: fade-up 0.65s ease-out;
}

.kpi {
  background: linear-gradient(140deg, rgba(0, 194, 168, 0.24), rgba(6, 19, 26, 0.4));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.8rem;
}

.risk-good { color: var(--accent-soft); font-weight: 700; }
.risk-mid { color: var(--gold); font-weight: 700; }
.risk-bad { color: var(--danger); font-weight: 700; }

[data-testid="stMetric"] {
  background: rgba(8, 28, 38, 0.72);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.65rem;
}

.stButton > button {
  border-radius: 999px;
  border: 1px solid rgba(125, 227, 214, 0.45);
  background: linear-gradient(90deg, #00c2a8, #12a48e);
  color: #03241f;
  font-weight: 700;
}

.stButton > button:hover {
  border-color: #7de3d6;
  box-shadow: 0 0 0 3px rgba(125, 227, 214, 0.20);
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
    st.markdown("### Aurelia Churn Studio")
    st.caption("Premium inference cockpit")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Single Prediction", "Batch Prediction", "Model Profile"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("API Endpoint")
    st.code(API_URL)


st.markdown(
    """
<div class="hero">
  <div class="title">Aurelia Churn Studio</div>
  <div class="subtitle">Realtime churn intelligence with production-grade API orchestration.</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

health = get_health()
features = get_features()
feature_count = len(features) if features else 19

if page == "Dashboard":
    if not health:
        st.error("API is offline. Start FastAPI server on localhost:8000 first.")
    else:
        metrics = health.get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Service", "Online", "Healthy")
        c2.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.2%}")
        c3.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.2%}")
        c4.metric("F1 Score", f"{metrics.get('f1_score', 0.0):.4f}")

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
    st.subheader("Single Prediction")
    st.caption(
        "Use quick profile presets or provide an exact numeric vector. "
        f"Expected features: {feature_count}."
    )

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
    st.subheader("Batch Prediction")
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
    st.subheader("Model Profile")
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
st.caption("Aurelia Churn Studio - premium UI layer for the local churn inference API")
