# Install necessary libraries if running in Colab or a new environment


# Step 1: Save this as `app.py`
from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer
model = joblib.load('wati_response_model.pkl')
vectorizer = joblib.load('wati_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "🟢 WATI Response Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    # Vectorize and predict
    vec = vectorizer.transform([message])
    prediction = model.predict(vec)[0]

    return jsonify({'message': message, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
