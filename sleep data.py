import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('sleep_dataset.csv')  # (change filename if needed)

# Check for missing values
print(data.isnull().sum())

# Optionally, fill missing values if any
data.fillna(data.mean(), inplace=True)

# Separate features and target
X = data.drop('sleep_hours', axis=1)  # Assuming 'sleep_hours' is your target column
y = data['sleep_hours']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Data ready for model training! ✅")
print(data.head())  # This will show the first few rows of the dataset


