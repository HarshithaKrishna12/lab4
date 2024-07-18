from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Initialize model and scaler variables
model = joblib.load('./fish_weight_model.pkl')
scaler = joblib.load('./scaler.pkl')

# Define routes and functions


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        return jsonify(error="Model or scaler not loaded"), 500

    try:
        data = request.get_json(force=True)
        length = float(data['length'])
        height = float(data['height'])
        width = float(data['width'])

        # Check the number of features expected by the scaler
        num_features_expected = scaler.n_features_in_

        # Prepare input data for prediction
        if num_features_expected == 3:  # Example: Scaler expects 3 features
            input_data = np.array([[length, height, width]])
        elif num_features_expected == 6:  # Example: Scaler expects 6 features
            # Assuming other features should be 0 (or some default value)
            other_features = [0.0] * (num_features_expected - 3)
            input_data = np.array([[length, height, width] + other_features])
        else:
            return jsonify(error=f"Unexpected number of features ({num_features_expected}) in scaler"), 500

        # Make sure to scale the input data correctly
        scaled_data = scaler.transform(input_data)

        # Make prediction using the model
        prediction = model.predict(scaled_data)

        # Return the prediction as JSON response
        return jsonify(weight=prediction[0])

    except KeyError as e:
        return jsonify(error=f"KeyError: {str(e)}"), 400  # Bad request
    except ValueError as e:
        return jsonify(error=f"ValueError: {str(e)}"), 400  # Bad request
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)
