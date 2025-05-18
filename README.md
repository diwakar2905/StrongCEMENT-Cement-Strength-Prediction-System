# 🏗️ StrongCEMENT-Predict Concrete Strength Instantly Using Machine Learning

Welcome to StrongCEMENT – a user-friendly web app that helps you predict the compressive strength of concrete based on its mix composition. Whether you're a civil engineer, a student, or just curious about how machine learning can help with materials science, this project is for you!

🚀 What is StrongCEMENT?
StrongCEMENT is an interactive web application built with Streamlit and scikit-learn. It lets you:

Upload your own cement mix data (CSV format)

Instantly train a predictive model on your data

Enter new mix values and get real-time strength predictions

See how well the model performs on your dataset

No coding required – just upload, click, and predict!

✨ Features
Easy CSV Upload: Drag and drop your dataset (up to 200 MB)

Automatic Model Training: Uses a Decision Tree Regressor for fast, accurate predictions

Performance Metrics: Instantly see the model’s R² score after training

Interactive Predictions: Enter new mix proportions and get strength predictions on the fly

Data Preview: Glance at the first 10 rows of your uploaded data

Model Insights: See which features are used and the size of your dataset

Helpful Error Messages: Get clear feedback if your data is missing required columns

🖼️ Live Demo
![Screenshot 2025-05-18 090653](https://github.com/user-attachments/assets/60b5b618-941f-489d-9e4f-cdca73c39009)
 Dataset and start predicting!*

![Screenshot 2025-05-18 090833](https://github.com/user-attachments/assets/fb90a834-57b3-484f-a6de-bc8cf61de614)
 Make new predictions instantly.*

🛠️ How to Use
Clone this repo:

bash
git clone https://github.com/yourusername/strongcement.git
cd strongcement
Install dependencies:

bash
pip install -r requirements.txt
Run the app:

bash
streamlit run app.py
Open your browser, upload your dataset, and start predicting!

🔬 How Does It Work?
Data Upload: You provide a CSV with your cement mix data.

Model Training: The app splits your data, scales features, and trains a Decision Tree Regressor.

Evaluation: R² score is displayed to show model accuracy.

Prediction: Enter new mix values and see the predicted compressive strength (in MPa) instantly.

📚 Tech Stack
Frontend: Streamlit

Model: scikit-learn’s DecisionTreeRegressor

Data Handling: pandas

📦 Dataset Source
This app was inspired by the UCI Concrete Compressive Strength Dataset.

👨‍💻 Author
Made by Diwakar Mishra

📄 License
MIT License.
