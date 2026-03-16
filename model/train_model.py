import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("data/energy_data.csv")

# Encode Season column
season_mapping = {
    "Winter": 0,
    "Summer": 1,
    "Monsoon": 2,
    "Post-Monsoon": 3
}

df["Season"] = df["Season"].map(season_mapping)

# Define features (X) and target (y)
X = df[["Month", "Units_Consumed", "Previous_Month_Units", "Season"]]
y = df["Estimated_Bill"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Create model folder if not exists
if not os.path.exists("model"):
    os.makedirs("model")

# Save model
joblib.dump(model, "model/bill_prediction_model.pkl")

print("✅ Model trained and saved successfully!")