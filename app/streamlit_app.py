import pandas as pd
import requests
import streamlit as st
import os 

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write(
    "This demo sends a real transaction (from a held-out test set) to a live "
    "fraud detection API and shows the model's prediction."
)

RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

@st.cache_data
def load_samples():
    return pd.read_csv("data/processed/api_samples.csv")

samples = load_samples()

st.subheader("1. Pick a sample transaction")
labels = [f"Sample {i+1}" for i in range(len(samples))]
choice = st.selectbox("Choose a transaction to test:", labels)
idx = labels.index(choice)
row = samples.iloc[idx]

col1, col2 = st.columns(2)
col1.metric("Amount", f"${row['Amount']:.2f}")
col2.metric("Time (seconds since start)", f"{row['Time']:.0f}")

if st.button("Check for fraud", type="primary"):
    with st.spinner("Calling the API..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"features": row[RAW_COLS].tolist()},
                timeout=5,
            )
            resp.raise_for_status()
            result = resp.json()

            st.subheader("2. Result")
            proba = result["fraud_probability"]
            is_fraud = result["is_fraud"]

            if is_fraud:
                st.error(f"⚠️ Flagged as FRAUD (probability: {proba:.3f})")
            else:
                st.success(f"✅ Looks legitimate (probability: {proba:.3f})")

            st.caption(f"Decision threshold: {result['threshold_used']:.3f}")
            st.progress(min(proba, 1.0))

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the API. Make sure it's running "
                "(`docker run -p 8000:8000 fraud-api` or `uvicorn src.api:app`)."
            )