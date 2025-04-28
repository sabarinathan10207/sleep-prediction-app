import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the dataset (same as earlier)
data = pd.read_csv('sleep_dataset.csv')

# Preprocess the dataset
X = data.drop('sleep_hours', axis=1)
y = data['sleep_hours']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

# Streamlit UI
st.title('Sleep Prediction App')
#st.write("Enter your daily data to predict your sleep hours.")
st.write("Welcome to the Sleep Prediction app!")

# Input fields for new user data
hours_worked = st.number_input('Hours Worked', min_value=0, max_value=24)
stress_level = st.number_input('Stress Level', min_value=0, max_value=10)
caffeine_intake_mg = st.number_input('Caffeine Intake (mg)', min_value=0)
physical_activity_min = st.number_input('Physical Activity (minutes)', min_value=0)

# Make prediction when user clicks button
if st.button('Predict Sleep Hours'):
    new_input = [[hours_worked, stress_level, caffeine_intake_mg, physical_activity_min]]
    new_input_scaled = scaler.transform(new_input)
    predicted_sleep = model.predict(new_input_scaled)
    st.write(f"Predicted Sleep Hours: {predicted_sleep[0]:.2f}")
