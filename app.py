from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model/bill_prediction_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    month = data["month"]
    units = data["units"]
    prev_units = data["previous_units"]
    season = data["season"]

    input_data = np.array([[month, units, prev_units, season]])

    prediction = model.predict(input_data)

    return jsonify({
        "predicted_bill": round(float(prediction[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)