import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
try:
    with open('cement_strength_model.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('cement_scaler.sav', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler not found. Please ensure 'cement_strength_model.sav' and 'cement_scaler.sav' are in the same directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title="StrongCEMENT - Cement Strength Predictor",
    page_icon="🧱",
    layout="wide"
)

# Custom header
st.markdown(
    "<h1 style='text-align: center; color: #F4A261;'>🧱 StrongCEMENT</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>A machine learning app to predict concrete strength from cement mix composition</p>",
    unsafe_allow_html=True
)

# Initialize session state for prediction
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

# Prediction form layout
st.markdown("### 📝 Predict New Cement Mix Strength")
with st.form("prediction_form"):
    cols = st.columns(2)
    input_values = {}
    
    # Define user-friendly labels for the input fields
    feature_labels = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']

    for idx, label in enumerate(feature_labels):
        with cols[idx % 2]:
            # Use the label for the UI, but the key for the dictionary can be the same
            input_values[label] = st.number_input(f"{label}", min_value=0.0, value=0.0, step=0.1)

    submitted = st.form_submit_button("🔍 Predict Strength")

if submitted:
    # Define the exact feature names the model was trained on
    model_feature_names = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'age']

    # Create a DataFrame from the user inputs
    # The values are ordered based on the user-friendly labels
    input_data = [input_values[label] for label in feature_labels]
    input_df = pd.DataFrame([input_data], columns=model_feature_names)

    # Scale the input features
    scaled_features = scaler.transform(input_df)
    
    # Make a prediction
    prediction = model.predict(scaled_features)
    
    st.session_state['prediction'] = prediction[0]

if st.session_state['prediction'] is not None:
    st.success(f"🎯 Predicted Concrete Strength: **{st.session_state['prediction']:.2f} MPa**")

# Use the user-friendly labels for display
with st.expander("📊 Model Details"):
    st.markdown("**Features Used:**")
    st.code(", ".join(feature_labels), language="markdown")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by <strong>Diwakar Mishra</strong></p>", unsafe_allow_html=True)