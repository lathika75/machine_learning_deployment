import os
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Define paths for the model and scaler
MODEL_PATH = 'ann_time_series_model.keras'
SCALER_PATH = 'scaler.pkl'

# Load the trained Keras model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the StandardScaler object
try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Define the numerical features that were scaled during training
# This list should match the `numerical_features` used in preprocessing
numerical_features = [
    'Battery Capacity (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)',
    'Charging Cost (USD)', 'State of Charge (Start %)', 'State of Charge (End %)',
    'Distance Driven (since last charge) (km)', 'Temperature (°C)', 'Vehicle Age (years)',
    'start_hour', 'start_day_of_week', 'start_month'
]
# ✅ Home route (ADDED)
@app.route('/')
def home():
    return "ML API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly'}), 500

    try:
        # Get JSON data from the request
        json_data = request.get_json(force=True)

        # Convert JSON data to a pandas DataFrame
        # Ensure consistent column order as during training
        input_df = pd.DataFrame([json_data])

        # Check if all required numerical features are present
        missing_features = [f for f in numerical_features if f not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing features in input data: {missing_features}'}), 400

        # Select and scale the numerical features
        input_scaled = scaler.transform(input_df[numerical_features])

        # Make prediction
        prediction = model.predict(input_scaled)

        # Convert prediction to a readable format (e.g., float)
        predicted_energy_consumed = float(prediction[0][0])

        # Return the prediction as a JSON response
        return jsonify({'predicted_energy_consumed_kWh': predicted_energy_consumed})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
