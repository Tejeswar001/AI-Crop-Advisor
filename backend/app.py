# Main Flask app for AgroSmart

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os
import requests
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import json

app = Flask(__name__)

def load_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_ml/models/crop_recommendation/crop_recommendation_model.pkl'))
    encoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_ml/models/crop_recommendation/label_encoder.pkl'))
    print(f"Loading model from: {model_path}")
    print(f"Loading encoder from: {encoder_path}")
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        print("Model and encoder loaded successfully.")
        return model, encoder
    except Exception as e:
        print(f"Error loading model or encoder: {e}")
        return None, None

model, encoder = None, None
try:
    model, encoder = load_model()
except Exception:
    pass

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    print(f"Received data: {data}")
    features = [data.get(k) for k in ['N','P','K','temperature','humidity','ph','rainfall']]
    print(f"Features: {features}")
    if None in features or model is None or encoder is None:
        print(f"Error: Invalid input or model not loaded. Model: {model is not None}, Encoder: {encoder is not None}")
        return jsonify({'error': 'Invalid input or model not loaded'}), 400
    arr = np.array(features).reshape(1, -1)
    try:
        pred = model.predict(arr)
        label = encoder.inverse_transform(pred)[0]
        print(f"Predicted crop: {label}")
        return jsonify({'recommendation': label})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city')
    if not city:
        return jsonify({'error': 'City required'}), 400
    # Step 1: Geocode city name to lat/lon
    geo_url = f'https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1'
    geo_resp = requests.get(geo_url, headers={'User-Agent': 'AgroSmart/1.0'})
    geo_data = geo_resp.json()
    if not geo_data:
        return jsonify({'error': 'City not found'}), 404
    lat = geo_data[0]['lat']
    lon = geo_data[0]['lon']
    # Request daily weather data from Open-Meteo
    weather_url = (
        f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
        '&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max,relative_humidity_2m_min'
        '&current_weather=true&timezone=auto'
    )
    weather_resp = requests.get(weather_url)
    weather_data = weather_resp.json()
    daily = weather_data.get('daily', {})
    current = weather_data.get('current_weather', {})
    # Get today's values (first in arrays)
    # Defensive: always return arrays of length 7, fallback to None
    def pad(arr):
        arr = arr if isinstance(arr, list) else [None]*7
        return (arr + [None]*7)[:7]
    temp_max_arr = pad(daily.get('temperature_2m_max'))
    temp_min_arr = pad(daily.get('temperature_2m_min'))
    humidity_max_arr = pad(daily.get('relative_humidity_2m_max'))
    humidity_min_arr = pad(daily.get('relative_humidity_2m_min'))
    result = {
        'city': city,
        'latitude': lat,
        'longitude': lon,
        'weather': {
            'temperature': current.get('temperature'),
            'temperature_max': temp_max_arr[0],
            'temperature_min': temp_min_arr[0],
            'humidity_max': humidity_max_arr[0],
            'humidity_min': humidity_min_arr[0],
            'precipitation': daily.get('precipitation_sum', [None])[0],
            'weathercode': current.get('weathercode'),
            'future_temperature_max': temp_max_arr,
            'future_temperature_min': temp_min_arr,
            'future_humidity_max': humidity_max_arr,
            'future_humidity_min': humidity_min_arr
        }
    }
    print('Weather API arrays:', {
        'future_temperature_max': temp_max_arr,
        'future_temperature_min': temp_min_arr,
        'future_humidity_max': humidity_max_arr,
        'future_humidity_min': humidity_min_arr
    })
    print('Weather API result:', result)
    return jsonify(result)

@app.route('/api/disease', methods=['POST'])
def detect_disease():
    if 'leafImage' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    img_file = request.files['leafImage']
    img = Image.open(img_file.stream).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    model_path = os.path.join(os.path.dirname(__file__), '../data_ml/models/mobilenetv3_plant_disease.pth')
    num_classes = 38  # Use full model output
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
    # Load class labels from file
    labels_path = os.path.join(os.path.dirname(__file__), '../data_ml/models/class_labels.json')
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_map = json.load(f)
        label = class_map.get(str(pred), f'Class {pred}')
    else:
        label = f'Class {pred}'
    return jsonify({'disease': label})

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
