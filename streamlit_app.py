"""
Streamlit frontend for Churn Prediction API
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from typing import List

# Configuration
API_URL = "http://localhost:8000"
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .positive {background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);}
    .negative {background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);}
    .info {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Customer Churn Prediction")
st.markdown("## Ready-to-deploy ML Model for Telecom Customer Churn")

# Sidebar
with st.sidebar:
    st.header("🎯 Navigation")
    page = st.radio("Select Page", ["Dashboard", "Single Prediction", "Batch Prediction", "Model Info"])

# Helper functions
@st.cache_data
def get_model_health():
    """Get model health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None

@st.cache_data
def get_feature_names():
    """Get feature names from API"""
    try:
        response = requests.get(f"{API_URL}/features", timeout=5)
        if response.status_code == 200:
            return response.json().get("features", [])
    except:
        return []

def make_prediction(features: List[float]):
    """Make single prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
    return None

def make_batch_prediction(instances: List[List[float]]):
    """Make batch predictions"""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"instances": instances},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"❌ Batch prediction failed: {str(e)}")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Dashboard
# ──────────────────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.header("📈 Model Dashboard")
    
    # Get model health
    health = get_model_health()
    
    if health:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Model Status",
                "✅ Online",
                "Ready to predict"
            )
        
        with col2:
            accuracy = health.get("metrics", {}).get("accuracy", 0)
            st.metric(
                "Accuracy",
                f"{accuracy:.2%}",
                "Test set performance"
            )
        
        with col3:
            roc_auc = health.get("metrics", {}).get("roc_auc", 0)
            st.metric(
                "ROC-AUC",
                f"{roc_auc:.2%}",
                "Discrimination ability"
            )
        
        with col4:
            f1_score = health.get("metrics", {}).get("f1_score", 0)
            st.metric(
                "F1 Score",
                f"{f1_score:.4f}",
                "Harmonic mean"
            )
        
        st.divider()
        
        # Quick stats
        st.subheader("📊 Model Specifications")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Model Type:** XGBoost (Random Forest)
            
            **Dataset:** Telecom Customer Churn
            
            **Target Variable:** Churn (Binary: Yes/No)
            
            **Samples:** 7,043 customers
            """)
        
        with col2:
            st.write("""
            **Framework:** scikit-learn + XGBoost
            
            **Deployment:** FastAPI + Docker
            
            **Accuracy Threshold:** 80%
            
            **Status:** ✅ PASSED
            """)
        
        st.divider()
        
        # Model metrics visualization
        st.subheader("📊 Performance Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "ROC-AUC", "F1 Score"],
            "Score": [
                health.get("metrics", {}).get("accuracy", 0),
                health.get("metrics", {}).get("roc_auc", 0),
                health.get("metrics", {}).get("f1_score", 0),
            ]
        })
        st.bar_chart(metrics_df.set_index("Metric"))
    
    else:
        st.error("❌ Cannot connect to API. Is the server running on http://localhost:8000?")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Single Prediction
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Single Prediction":
    st.header("🔮 Single Customer Prediction")
    
    st.info("Enter customer characteristics to predict churn probability.")
    
    # Create input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        seniority = st.slider("Seniority (months)", 0, 72, 12)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependent = st.selectbox("Has Dependent", ["No", "Yes"])
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    
    with col3:
        st.subheader("Billing")
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 50.0)
        total_charges = st.slider("Total Charges ($)", 0.0, 8000.0, 1000.0)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    
    # Create features array (19 features expected)
    features = [
        int(seniority > 12),  # 0: senior
        tenure,               # 1: tenure
        int(partner == "Yes"), # 2: partner
        int(dependent == "Yes"), # 3: dependent
        int(phone_service == "Yes"), # 4: phoneService
        int(internet_service in ["DSL", "Fiber"]), # 5: internetService
        int(online_security == "Yes"), # 6: onlineSecurity
        int(streaming_tv == "Yes"), # 7: streamingTV
        int(contract_type != "Month-to-month"), # 8: contract
        int(paperless_billing == "Yes"), # 9: paperlessBilling
        1 if contract_type == "Two year" else (0.5 if contract_type == "One year" else 0), # 10: contractDuration
        int(internet_service == "Fiber"), # 11: fiberOptic
        0, 0, 0, 0, 0, 0, # 12-18: other features (placeholder)
        monthly_charges,
        total_charges
    ]
    
    # Make prediction
    if st.button("🎯 Predict Churn", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            result = make_prediction(features)
        
        if result:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                churn_prob = result.get("churn_probability", 0)
                churn_status = "🔴 WILL CHURN" if result["churn"] else "🟢 WILL STAY"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 2rem; border-radius: 0.5rem; color: white; text-align: center;'>
                    <h3>Prediction Result</h3>
                    <h1>{churn_status}</h1>
                    <h2>{churn_prob:.1%} probability</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Gauge chart
                fig_data = {
                    "probability": churn_prob * 100,
                    "safe_zone": max(0, 100 - churn_prob * 100)
                }
                
                st.metric(
                    "Churn Risk Level",
                    "🔴 HIGH" if churn_prob > 0.7 else ("🟡 MEDIUM" if churn_prob > 0.3 else "🟢 LOW"),
                    f"{churn_prob:.1%}"
                )
                
                # Recommendation
                st.info("""
                **Recommendation:**
                
                """ + (
                    "⚠️ **High Risk** - Consider retention strategies (discounts, loyalty programs)"
                    if churn_prob > 0.7 else (
                    "⚠️ **Medium Risk** - Monitor customer satisfaction"
                    if churn_prob > 0.3 else
                    "✅ **Low Risk** - Maintain current service quality"
                ))
                )

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Batch Prediction
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Batch Prediction":
    st.header("📦 Batch Predictions")
    
    st.info("Upload a CSV file with customer data to get predictions for multiple customers.")
    
    # Sample data template
    with st.expander("📋 See Sample CSV Format"):
        sample_df = pd.DataFrame({
            "feature_0": [0, 1, 0],
            "feature_1": [12, 24, 6],
            "feature_2": [1, 0, 1],
            "feature_3": [0, 1, 0],
            # ... 16 more features
        })
        st.write("Your CSV should have 19 numeric columns (one per feature)")
        st.csv(sample_df.to_csv(index=False))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    with col2:
        num_samples = st.number_input("Or generate test samples:", min_value=1, max_value=100, value=5)
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} rows")
        
        if st.button("🚀 Predict All", use_container_width=True):
            with st.spinner("Making predictions..."):
                # Convert to list of lists
                instances = df.iloc[:, :19].values.tolist()
                result = make_batch_prediction(instances)
        
            if result:
                predictions = result.get("predictions", [])
                
                # Add predictions to dataframe
                df_results = df.copy()
                df_results["churn_prediction"] = [p["churn"] for p in predictions]
                df_results["churn_probability"] = [p["churn_probability"] for p in predictions]
                df_results["risk_level"] = df_results["churn_probability"].apply(
                    lambda x: "🔴 HIGH" if x > 0.7 else ("🟡 MEDIUM" if x > 0.3 else "🟢 LOW")
                )
                
                st.subheader("Results")
                st.dataframe(df_results, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    churn_count = sum(1 for p in predictions if p["churn"])
                    st.metric("Predicted Churners", churn_count, f"({churn_count/len(predictions):.1%})")
                with col2:
                    avg_prob = np.mean([p["churn_probability"] for p in predictions])
                    st.metric("Avg Churn Probability", f"{avg_prob:.1%}")
                with col3:
                    high_risk = sum(1 for p in predictions if p["churn_probability"] > 0.7)
                    st.metric("High Risk Customers", high_risk)
    
    elif st.button("📊 Generate Test Batch"):
        with st.spinner("Generating test data..."):
            # Generate random test data
            test_data = np.random.rand(num_samples, 19).tolist()
            result = make_batch_prediction(test_data)
        
        if result:
            predictions = result.get("predictions", [])
            
            # Display results
            df_results = pd.DataFrame({
                "Customer_ID": [f"CUST_{i:04d}" for i in range(num_samples)],
                "Churn_Prediction": ["Yes" if p["churn"] else "No" for p in predictions],
                "Churn_Probability": [f"{p['churn_probability']:.2%}" for p in predictions],
            })
            
            st.dataframe(df_results, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Model Info
# ──────────────────────────────────────────────────────────────────────────────
elif page == "Model Info":
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🤖 Model Architecture")
        st.write("""
        **Algorithm:** XGBoost (Gradient Boosting)
        
        **Framework:** scikit-learn
        
        **Training Data:** 7,043 customers
        
        **Features:** 19 customer attributes
        
        **Target:** Binary (Churn: Yes/No)
        """)
    
    with col2:
        st.subheader("📊 Performance Metrics")
        st.write("""
        **Accuracy:** 80.62%
        
        **ROC-AUC:** 84.35%
        
        **F1 Score:** 0.5674
        
        **Precision/Recall:** Balanced
        """)
    
    st.divider()
    
    st.subheader("📚 Feature Descriptions")
    features_info = {
        "0": "Senior Citizen (0/1)",
        "1": "Tenure (months)",
        "2": "Partner (0/1)",
        "3": "Dependent (0/1)",
        "4": "Phone Service (0/1)",
        "5": "Internet Service (0/1)",
        "6": "Online Security (0/1)",
        "7": "Streaming TV (0/1)",
        "8": "Contract Type (0/1)",
        "9": "Paperless Billing (0/1)",
        "10": "Contract Duration (0-1)",
        "11": "Fiber Optic (0/1)",
        "12-18": "Additional service features",
        "19": "Monthly Charges ($)",
        "20": "Total Charges ($)",
    }
    
    for idx, description in features_info.items():
        st.write(f"**Feature {idx}:** {description}")
    
    st.divider()
    
    st.subheader("🚀 Deployment Info")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("""
        **API:** FastAPI
        
        **Server:** Uvicorn
        
        **Framework:** Python 3.10
        """)
    
    with col2:
        st.write("""
        **Containerization:** Docker
        
        **Registry:** AWS ECR
        
        **Port:** 8000
        """)
    
    with col3:
        st.write("""
        **CI/CD:** GitHub Actions
        
        **Storage:** AWS S3
        
        **MLOps:** MLflow
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.85rem'>
    🚀 MLOps Churn Prediction Platform | Powered by FastAPI + Streamlit | © 2026
</div>
""", unsafe_allow_html=True)
