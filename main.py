import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the trained model
model = pickle.load(open('mlmodel.pkl', 'rb'))

@app.route('/')
def index():
    return "Flask API for Liver Disease Prediction"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON input
        data = request.json
        print("Received data:", data)
        
        # Extract features from the input
        input_features = np.array([data['features']])  # Example input: { "features": [age, gender, etc.] }
        
        # Predict using the model
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)
        
        # Prepare the response
        response = {
            'prediction': int(prediction[0]),
            'probability': probability.tolist()
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
