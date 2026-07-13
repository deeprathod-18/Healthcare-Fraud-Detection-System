import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

# ─── Load Model & Scaler ───────────────────────────────────
@st.cache_resource
def load_model():
    with open('xgboost_fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.json', 'r') as f:
        columns = json.load(f)
    return model, scaler, columns

model, scaler, feature_columns = load_model()

# ─── Header ────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#e74c3c;'>
        🏥 Healthcare Fraud Detection System
    </h1>
    <p style='text-align:center; color:gray; font-size:18px;'>
        Powered by XGBoost — ROC-AUC: 98.94%
    </p>
    <hr>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Mode",
    ["🏠 Home", "🔍 Single Provider Prediction", "📂 Batch CSV Prediction"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.markdown("- **Model**: XGBoost")
st.sidebar.markdown("- **Accuracy**: 95.21%")
st.sidebar.markdown("- **ROC-AUC**: 98.94%")
st.sidebar.markdown("- **F1-Score**: 95.14%")

# ══════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("## 👋 Welcome to the Healthcare Fraud Detection App")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy",  "95.21%")
    col2.metric("📈 ROC-AUC",   "98.94%")
    col3.metric("🔍 Recall",    "97.05%")
    col4.metric("⚖️ F1-Score",  "95.14%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Single Provider Prediction")
        st.info("""
        Enter provider statistics manually and get
        an instant fraud prediction with probability score.
        
        **Use this when:**
        - Checking a specific provider
        - Manual investigation
        - Real-time fraud screening
        """)
        if st.button("Go to Single Prediction →"):
            st.rerun()

    with col2:
        st.markdown("### 📂 Batch CSV Prediction")
        st.info("""
        Upload a CSV file with multiple providers and
        get fraud predictions for all of them at once.
        
        **Use this when:**
        - Auditing multiple providers
        - Bulk fraud screening
        - Generating fraud reports
        """)

    st.markdown("---")
    st.markdown("### 📌 About This Project")
    st.markdown("""
    This application detects fraudulent Medicare insurance claims using
    Machine Learning. It was trained on **5,410 providers** with
    **517,737+ claims** and achieves a **98.94% ROC-AUC score**.

    | Phase | Description |
    |---|---|
    | Data | Medicare Claims Dataset (Kaggle) |
    | Preprocessing | SMOTE, StandardScaler, Feature Engineering |
    | Models Tried | Logistic Reg, Decision Tree, Random Forest, KNN, SVM, XGBoost, Naive Bayes |
    | Best Model | XGBoost |
    """)

# ══════════════════════════════════════════════════
# PAGE 2 — SINGLE PROVIDER PREDICTION
# ══════════════════════════════════════════════════
elif page == "🔍 Single Provider Prediction":
    st.markdown("## 🔍 Single Provider Fraud Prediction")
    st.markdown("Enter the provider details below to check for fraud:")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📋 Claims Info")
        total_claims      = st.number_input("Total Claims",           min_value=0, value=100)
        total_reimbursed  = st.number_input("Total Reimbursed ($)",   min_value=0.0, value=50000.0)
        avg_reimbursed    = st.number_input("Avg Reimbursed ($)",     min_value=0.0, value=500.0)
        max_reimbursed    = st.number_input("Max Reimbursed ($)",     min_value=0.0, value=5000.0)

    with col2:
        st.markdown("### 💰 Financial Info")
        total_deductible  = st.number_input("Total Deductible ($)",   min_value=0.0, value=10000.0)
        avg_deductible    = st.number_input("Avg Deductible ($)",     min_value=0.0, value=100.0)
        unique_patients   = st.number_input("Unique Patients",        min_value=0, value=80)
        inpatient_claims  = st.number_input("Inpatient Claims",       min_value=0, value=20)

    with col3:
        st.markdown("### 🏥 Provider Info")
        unique_physicians = st.number_input("Unique Physicians",      min_value=0, value=5)
        avg_age           = st.number_input("Avg Patient Age",        min_value=0.0, value=70.0)
        avg_claim_bene    = st.number_input("Avg Claim (Bene)",       min_value=0.0, value=500.0)

    st.markdown("### 🩺 Chronic Conditions (Total Patients)")
    c1, c2, c3, c4 = st.columns(4)
    alzheimer    = c1.number_input("Alzheimer",     min_value=0, value=10)
    heartfailure = c2.number_input("Heart Failure", min_value=0, value=8)
    kidney       = c3.number_input("Kidney Disease",min_value=0, value=5)
    cancer       = c4.number_input("Cancer",        min_value=0, value=3)

    c5, c6, c7 = st.columns(3)
    diabetes     = c5.number_input("Diabetes",      min_value=0, value=20)
    depression   = c6.number_input("Depression",    min_value=0, value=7)
    stroke       = c7.number_input("Stroke",        min_value=0, value=4)

    st.markdown("---")

    if st.button("🔍 Predict Fraud", use_container_width=True):
        # Build input
        input_data = pd.DataFrame([[
            total_claims, total_reimbursed, avg_reimbursed,
            max_reimbursed, total_deductible, avg_deductible,
            unique_patients, inpatient_claims, unique_physicians,
            avg_age, avg_claim_bene,
            alzheimer, heartfailure, kidney, cancer,
            diabetes, depression, stroke
        ]], columns=feature_columns)

        # Scale & Predict
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error(f"🚨 FRAUDULENT PROVIDER DETECTED!")
                st.markdown(f"### Fraud Probability: `{round(probability*100, 2)}%`")
            else:
                st.success(f"✅ LEGITIMATE PROVIDER")
                st.markdown(f"### Fraud Probability: `{round(probability*100, 2)}%`")

        with col2:
            # Gauge-style progress bar
            st.markdown("### Risk Level")
            st.progress(float(probability))
            if probability < 0.3:
                st.markdown("🟢 **Low Risk**")
            elif probability < 0.6:
                st.markdown("🟡 **Medium Risk**")
            else:
                st.markdown("🔴 **High Risk**")

# ══════════════════════════════════════════════════
# PAGE 3 — BATCH CSV PREDICTION
# ══════════════════════════════════════════════════
elif page == "📂 Batch CSV Prediction":
    st.markdown("## 📂 Batch CSV Fraud Prediction")
    st.markdown("Upload a CSV file with provider data to get predictions for all providers.")
    st.markdown("---")

    # Show expected format
    st.markdown("### 📋 Expected CSV Format")
    sample_df = pd.DataFrame([{
        col: 0 for col in feature_columns
    }])
    st.dataframe(sample_df)

    # Download sample CSV
    csv_sample = sample_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Sample CSV Template",
        data=csv_sample,
        file_name="sample_provider_template.csv",
        mime="text/csv"
    )

    st.markdown("---")

    uploaded_file = st.file_uploader("📁 Upload Your CSV File", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"### ✅ File Uploaded — {df.shape[0]} Providers Found")
        st.dataframe(df.head())

        if st.button("🔍 Run Fraud Detection on All Providers",
                     use_container_width=True):
            try:
                # Check columns
                missing_cols = [c for c in feature_columns if c not in df.columns]

                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    # Scale & Predict
                    X_input      = df[feature_columns]
                    X_scaled     = scaler.transform(X_input)
                    predictions  = model.predict(X_scaled)
                    probabilities= model.predict_proba(X_scaled)[:, 1]

                    # Add results
                    df['Fraud_Prediction'] = predictions
                    df['Fraud_Probability']= (probabilities * 100).round(2)
                    df['Risk_Level']       = pd.cut(
                        probabilities,
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['🟢 Low', '🟡 Medium', '🔴 High']
                    )

                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Results")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Providers",    df.shape[0])
                    col2.metric("🚨 Fraud Detected",  int(predictions.sum()))
                    col3.metric("✅ Legitimate",       int((predictions==0).sum()))

                    # Results table
                    st.markdown("### 📊 Detailed Results")
                    st.dataframe(df[['Fraud_Prediction',
                                     'Fraud_Probability',
                                     'Risk_Level']])

                    # Download results
                    result_csv = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=result_csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ─── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:gray;'>
    🎓 Healthcare Fraud Detection | ML Semester Project | Built with Streamlit & XGBoost
</p>
""", unsafe_allow_html=True)