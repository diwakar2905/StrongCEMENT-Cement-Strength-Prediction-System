import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

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

# Sidebar Upload
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your Cement Mix Data (CSV)", type=["csv"])

# Load and display dataset
if uploaded_file is not None:
    cement_df = pd.read_csv(uploaded_file)
    cement_df.columns = cement_df.columns.str.strip()

    with st.expander("📄 Preview Uploaded Dataset"):
        st.dataframe(cement_df.head(10), use_container_width=True)

    try:
        # Feature and target
        target_col = 'concrete_compressive_strength'
        X = cement_df.drop(target_col, axis=1)
        y = cement_df[target_col]

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)

        st.markdown("### ✅ Model Accuracy")
        st.success(f"R² Score: **{accuracy:.2f}**")

        # Prediction form layout
        st.markdown("### 📝 Predict New Cement Mix Strength")
        with st.form("prediction_form"):
            cols = st.columns(2)
            input_values = {}

            for idx, col in enumerate(X.columns):
                with cols[idx % 2]:
                    input_values[col] = st.number_input(f"{col}", min_value=0.0, value=0.0, step=0.1)

            submitted = st.form_submit_button("🔍 Predict Strength")

        if submitted:
            input_df = pd.DataFrame([input_values])
            prediction = model.predict(input_df)
            st.success(f"🎯 Predicted Concrete Strength: **{prediction[0]:.2f} MPa**")

        # Model Details
        with st.expander("📊 Model Details"):
            st.markdown("**Features Used:**")
            st.code(", ".join(X.columns), language="markdown")
            st.markdown("**Dataset Dimensions:**")
            st.write(f"{cement_df.shape[0]} rows × {cement_df.shape[1]} columns")

    except KeyError:
        st.error("❌ Please ensure your CSV has a column named `concrete_compressive_strength`.")

else:
    st.info("👈 Upload a dataset from the sidebar to begin.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by <strong>Diwakar Mishra</strong></p>", unsafe_allow_html=True)
