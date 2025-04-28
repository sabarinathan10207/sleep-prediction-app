import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assuming you already have your data in 'sleep_dataset.csv'
data = pd.read_csv('sleep_dataset.csv')

# Extract features and target
X = data[['hours_worked', 'stress_level', 'caffeine_intake_mg', 'physical_activity_min']]
y = data['sleep_hours']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
with open('sleep_model.pkl', 'wb') as file:
    pickle.dump(model, file)
