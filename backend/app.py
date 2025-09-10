# Main Flask app for AgroSmart

from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__)

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../data_ml/models/crop_recommendation.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), '../data_ml/models/crop_recommendation_label_encoder.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = None, None
try:
    model, encoder = load_model()
except Exception:
    pass

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    features = [data.get(k) for k in ['N','P','K','temperature','humidity','ph','rainfall']]
    if None in features or model is None or encoder is None:
        return jsonify({'error': 'Invalid input or model not loaded'}), 400
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)
    label = encoder.inverse_transform(pred)[0]
    return jsonify({'recommendation': label})


@app.route('/')
def home():
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/style.css')
def style_css():
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'style.css')

@app.route('/app.js')
def app_js():
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'app.js')

@app.route('/style.css')
def style():
    return send_from_directory('..', 'style.css')

@app.route('/app.js')
def appjs():
    return send_from_directory('..', 'app.js')

if __name__ == "__main__":
    app.run(debug=True)
