# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("🛌 Sleep Prediction App")
st.write("This app predicts how many hours you will sleep based on your daily activities.")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('sleep_dataset.csv')  # <-- Make sure your CSV is named 'sleep_data.csv'
    return data

data = load_data()

st.subheader("📊 Raw Data")
st.write(data.head())

# Split Data
X = data.drop("sleep_hours", axis=1)
y = data["sleep_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("🔧 Enter your details to predict sleep hours:")

# Input fields
hours_worked = st.slider('Hours Worked', 0, 16, 8)
stress_level = st.slider('Stress Level (1-10)', 1, 10, 5)
caffeine_intake_mg = st.slider('Caffeine Intake (mg)', 0, 500, 100)
physical_activity_min = st.slider('Physical Activity (minutes)', 0, 180, 30)

# Prediction
input_data = np.array([[hours_worked, stress_level, caffeine_intake_mg, physical_activity_min]])
predicted_sleep = model.predict(input_data)[0]

if st.button("Predict Sleep Hours"):
    st.success(f"😴 You are likely to sleep for **{predicted_sleep:.2f} hours** tonight!")

# Footer
st.write("Made with ❤️ by SABARINATHAN AI DATA SPECIALIST | MACHINE LEARNING PRACTITIONER")
