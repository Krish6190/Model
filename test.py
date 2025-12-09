import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your trained model
model = joblib.load('random_forest_regressor_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess your input data as needed for your model
    prediction = model.predict([data['features']]) # Example: assuming features are sent in a list
    return jsonify({'prediction': prediction.tolist()}) # Convert tolist() if prediction is a numpy array

if __name__ == '__main__':
    app.run(debug=True)