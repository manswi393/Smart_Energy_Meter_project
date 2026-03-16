import joblib
import numpy as np

# Load saved model
model = joblib.load("model/bill_prediction_model.pkl")

# Example input:
# Month = 3 (March)
# Units_Consumed = 210
# Previous_Month_Units = 200
# Season = 1 (Summer)

month = 3
units = 210
prev_units = 200
season = 1  # 0=Winter,1=Summer,2=Monsoon,3=Post-Monsoon

input_data = np.array([[month, units, prev_units, season]])

prediction = model.predict(input_data)

print("Predicted Electricity Bill: ₹", round(prediction[0], 2))