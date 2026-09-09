# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 22:31:13 2026

@author: Mohammed Abebe
"""

import streamlit as st
import pandas as pd
import joblib

# Define paths
#PROJECT_DIR = "/content/drive/MyDrive/Colab Notebooks/ACT-Africa2026"
#OUTPUT_DIR = f"{PROJECT_DIR}/TrainedNetworks_hack"
#MODEL_PATH = os.path.join(OUTPUT_DIR, "optimized_rf_model.pkl")
#SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl") # Assuming scaler is saved here

# --- Load Model and Scaler ---
try:
    model = joblib.load("optimized_rf_model.pkl")
    st.success("Model loaded successfully")
except FileNotFoundError:
    st.error("Error: Model file not found. Please ensure the model is trained and saved.")
    st.stop()

try:
    scaler = joblib.load("scaler.pkl")
    st.success("Scaler loaded successfully")
except FileNotFoundError:
    st.error("Error: Scaler file not found. Please ensure the StandardScaler object from data preprocessing is saved to this path.")
    st.stop()

# Define feature names (should match the order used during training)
feature_names = ['Year', ' DOY', ' HourF(UT)', ' Alt(km)', ' Lat(deg)', ' Lon(deg)']

st.set_page_config(page_title="Temperature Prediction App", layout="centered")

st.title("🌡️ Temp. Prediction with RF")
st.markdown("Enter the feature values below to predict the temperature.")

# --- Input Fields ---
with st.form("prediction_form"):
    st.header("Input Features")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2050, value=2023, step=1)
        doy = st.number_input("DOY", min_value=1, max_value=366, value=180, step=1)
        hourf = st.number_input("HourF(UT)", min_value=0.0, max_value=23.99, value=12.0, step=0.01)
    with col2:
        alt = st.number_input("Alt(km)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        lat = st.number_input("Lat(deg)", min_value=-90.0, max_value=90.0, value=0.0, step=0.01)
        lon = st.number_input("Lon(deg)", min_value=-180.0, max_value=180.0, value=0.0, step=0.01)

    submitted = st.form_submit_button("Predict Temperature")

# --- Prediction Logic ---
if submitted:
    input_data = pd.DataFrame([[year, doy, hourf, alt, lat, lon]], columns=feature_names)

    # Scale the input data
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)

    st.subheader("Prediction Result")
    st.success(f"The predicted temperature is: **{prediction[0]:.2f} °C**")

    st.markdown("---")
    st.subheader("Input Data Summary")
    st.write(input_data)