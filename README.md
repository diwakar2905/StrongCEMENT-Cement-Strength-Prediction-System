# 🧱 StrongCEMENT - Cement Strength Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://strongcement-cement-strength-prediction-system.streamlit.app/)

A user-friendly web application built with Streamlit that predicts the compressive strength of concrete based on its constituent components. This tool leverages a machine learning model to provide instant estimations, making it valuable for civil engineers, researchers, and students in the field of material science.

## 🚀 Live Demo

You can access and interact with the live application here:

**[https://strongcement-cement-strength-prediction-system.streamlit.app/](https://strongcement-cement-strength-prediction-system.streamlit.app/)**



---

## ✨ Features

- **Interactive UI**: A clean and simple interface for inputting cement mix values.
- **Real-Time Predictions**: Instantly get the predicted compressive strength (in MPa) upon submission.
- **Model Transparency**: An expandable section provides details on the features the model uses for its predictions.
- **Responsive Design**: The app is accessible on both desktop and mobile devices.

---

## ⚙️ How It Works

The application follows a standard machine learning pipeline to deliver predictions:

1.  **Data Collection**: The model was trained on the popular "Concrete Compressive Strength" dataset, which contains 1,030 instances of concrete mixes with varying compositions.

2.  **Model Training**:
    -   The dataset was split into features (the 8 input components) and the target (compressive strength).
    -   A `StandardScaler` was used to normalize the feature values, ensuring that each feature contributes proportionally to the model's decision-making process.
    -   A `DecisionTreeRegressor` model from the scikit-learn library was trained on this preprocessed data to learn the complex relationships between the mix ingredients and the resulting strength.

3.  **Deployment**:
    -   The trained `DecisionTreeRegressor` model and the `StandardScaler` were saved (serialized) into `.sav` files using `pickle`.
    -   The Streamlit application (`app.py`) loads these saved files. When a user enters new values and clicks "Predict," the app scales the inputs using the loaded scaler and feeds them to the loaded model to generate a prediction.

---

## 🛠️ Technology Stack

- **Backend & ML**: Python
- **Web Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Deployment**: Streamlit Cloud

---

## 📋 How to Run Locally

To run this project on your local machine, please follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/diwakar2905/StrongCEMENT-Cement-Strength-Prediction-System.git
    ```

2.  **Navigate to the Project Directory**
    ```bash
    cd StrongCEMENT-Cement-Strength-Prediction-System
    ```

3.  **Create and Activate a Virtual Environment** (Recommended)
    - For Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - For macOS/Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

4.  **Install Dependencies**
    Install all the required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    Your web browser should automatically open to the application's local address.

---

## 📂 Project Structure

```
├── app.py                          # The main Streamlit application script
├── cement_strength_model.sav       # The trained Decision Tree model
├── cement_scaler.sav               # The saved StandardScaler object
├── requirements.txt                # Project dependencies
├── cement_strength_Prediction_model.py # Script for training and evaluating the model
├── concrete_data.csv               # The dataset used for training
└── README.md                       # This file
```

---

## ✍️ Author

**Diwakar Mishra**

---