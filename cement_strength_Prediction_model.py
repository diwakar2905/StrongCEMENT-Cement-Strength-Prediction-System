# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pickle

# Load Data
cement = pd.read_csv("concrete_data.csv")
cement.head()

# Basic Data Info
print(cement.info())
print(cement.describe())
print(cement.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cement.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Define Features and Target
x = cement.drop('concrete_compressive_strength', axis=1)
y = cement['concrete_compressive_strength']

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(x_train, y_train)

# Evaluate Model
y_pred = regressor.predict(x_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Predictive System Function
def predict_strength(Cement, Slag, FlyAsh, Water, SP, CoarseAggregate, FineAggregate, Age):
    features = np.array([[Cement, Slag, FlyAsh, Water, SP, CoarseAggregate, FineAggregate, Age]])
    transformed_features = sc.transform(features)
    prediction = regressor.predict(transformed_features)
    return f"Estimated Cement Strength: {prediction[0]:.2f} MPa"

# Example Prediction
result = predict_strength(540, 0, 0, 162, 2.5, 1040, 676, 28)
print(result)

# Save the model and scaler
pickle.dump(regressor, open('cement_strength_model.sav', 'wb'))
pickle.dump(sc, open('cement_scaler.sav', 'wb'))

# Load the model and make a fresh prediction
loaded_model = pickle.load(open('cement_strength_model.sav', 'rb'))
loaded_scaler = pickle.load(open('cement_scaler.sav', 'rb'))

# New prediction
new_result = predict_strength(540, 0, 0, 162, 2.5, 1040, 676, 28)
print(new_result)
