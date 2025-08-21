from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import os
from yield_validator import validate_yield_prediction, get_yield_context, convert_yield_units
from translate_utils import translate_text, get_supported_languages, get_current_language
from disease_detection.model import load_model, predict_disease
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Load the models
crop_recommendation_model = joblib.load('crop_recommendation_model.pkl')
yield_prediction_model = joblib.load('yield_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
yield_scaler = joblib.load('scaler.pkl')
recommendation_scaler = joblib.load('recommendation_scaler.pkl')

# Load disease detection model
disease_model = load_model()

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Crop details for info page
crop_details = {
    "rice": {"N": 90, "P": 40, "K": 40, "temperature": "24-30°C", "humidity": "80-90%", "ph": "5.5-7.0", "rainfall": "100-200mm"},
    "wheat": {"N": 80, "P": 35, "K": 40, "temperature": "15-25°C", "humidity": "60-70%", "ph": "6.0-7.5", "rainfall": "75-100mm"},
    "maize": {"N": 85, "P": 45, "K": 50, "temperature": "18-27°C", "humidity": "65-75%", "ph": "5.5-7.0", "rainfall": "50-75mm"},
    "sugarcane": {"N": 120, "P": 60, "K": 60, "temperature": "25-35°C", "humidity": "70-80%", "ph": "6.0-7.5", "rainfall": "150-250mm"},
    "cotton": {"N": 60, "P": 30, "K": 30, "temperature": "21-30°C", "humidity": "50-60%", "ph": "6.0-7.5", "rainfall": "50-100mm"},
    "apple": {"N": 26, "P": 134, "K": 185, "temperature": "20-25°C", "humidity": "85-95%", "ph": "6.5-7.5", "rainfall": "120-150mm"},
    "banana": {"N": 95, "P": 80, "K": 50, "temperature": "25-35°C", "humidity": "75-90%", "ph": "6.0-7.5", "rainfall": "100-150mm"},
    "blackgram": {"N": 46, "P": 71, "K": 23, "temperature": "25-35°C", "humidity": "60-70%", "ph": "6.0-7.5", "rainfall": "60-80mm"},
    "chickpea": {"N": 47, "P": 67, "K": 79, "temperature": "20-30°C", "humidity": "20-30%", "ph": "6.0-7.5", "rainfall": "70-90mm"},
    "coconut": {"N": 27, "P": 19, "K": 37, "temperature": "27-32°C", "humidity": "80-90%", "ph": "5.5-7.5", "rainfall": "160-200mm"},
    "coffee": {"N": 104, "P": 33, "K": 33, "temperature": "20-30°C", "humidity": "60-70%", "ph": "6.0-7.5", "rainfall": "120-180mm"},
    "grapes": {"N": 26, "P": 152, "K": 187, "temperature": "20-30°C", "humidity": "75-85%", "ph": "6.0-7.5", "rainfall": "80-100mm"},
    "jute": {"N": 84, "P": 48, "K": 50, "temperature": "24-30°C", "humidity": "75-85%", "ph": "6.0-7.5", "rainfall": "150-200mm"},
    "kidneybeans": {"N": 25, "P": 70, "K": 25, "temperature": "18-25°C", "humidity": "30-40%", "ph": "6.0-7.5", "rainfall": "80-110mm"},
    "lentil": {"N": 24, "P": 67, "K": 22, "temperature": "20-30°C", "humidity": "60-70%", "ph": "6.0-7.5", "rainfall": "50-70mm"},
    "mango": {"N": 21, "P": 31, "K": 32, "temperature": "30-40°C", "humidity": "50-60%", "ph": "5.5-7.5", "rainfall": "80-120mm"},
    "mothbeans": {"N": 25, "P": 49, "K": 26, "temperature": "30-35°C", "humidity": "50-65%", "ph": "6.0-7.5", "rainfall": "50-80mm"},
    "mungbean": {"N": 26, "P": 50, "K": 23, "temperature": "25-35°C", "humidity": "75-85%", "ph": "6.0-7.5", "rainfall": "50-80mm"},
    "muskmelon": {"N": 101, "P": 21, "K": 50, "temperature": "25-35°C", "humidity": "85-95%", "ph": "6.0-7.5", "rainfall": "30-60mm"},
    "orange": {"N": 22, "P": 20, "K": 13, "temperature": "20-30°C", "humidity": "80-90%", "ph": "6.0-7.5", "rainfall": "100-120mm"},
    "papaya": {"N": 53, "P": 69, "K": 53, "temperature": "30-35°C", "humidity": "90-100%", "ph": "6.0-7.5", "rainfall": "140-180mm"},
    "pigeonpeas": {"N": 26, "P": 70, "K": 22, "temperature": "25-35°C", "humidity": "50-60%", "ph": "6.0-7.5", "rainfall": "140-170mm"},
    "pomegranate": {"N": 23, "P": 23, "K": 43, "temperature": "20-30°C", "humidity": "85-95%", "ph": "6.0-7.5", "rainfall": "100-130mm"},
    "watermelon": {"N": 99, "P": 25, "K": 53, "temperature": "24-30°C", "humidity": "85-95%", "ph": "5.5-7.5", "rainfall": "50-80mm"}
}

def translate_dict_values(d, target_lang):
    if target_lang == 'en':
        return d
    translated = {}
    for key, value in d.items():
        if isinstance(value, dict):
            translated[key] = translate_dict_values(value, target_lang)
        elif isinstance(value, str):
            translated[key] = translate_text(value, target_lang)
        else:
            translated[key] = value
    return translated

@app.context_processor
def inject_languages():
    return {
        'supported_languages': get_supported_languages(),
        'current_language': get_current_language(),
        'translate_text': translate_text  # Make translate_text available in templates
    }

@app.route('/change-language', methods=['POST'])
def change_language():
    language = request.form.get('language', 'en')
    session['language'] = language
    return redirect(request.referrer or url_for('home'))

@app.route('/')
def home():
    current_lang = get_current_language()
    translated_content = {
        'title': translate_text('AGRI-ADVISOR', current_lang),
        'description': translate_text('Your Smart Farming Companion - Get data-driven insights for optimal crop selection and yield forecasting', current_lang)
    }
    return render_template('index.html', content=translated_content)

@app.route('/crop-recommendation')
def crop_recommendation():
    current_lang = get_current_language()
    translated_content = {
        'title': translate_text('Crop Recommendation', current_lang),
        'description': translate_text('Get personalized crop recommendations based on your soil conditions', current_lang)
    }
    return render_template('crop_recommendation.html', content=translated_content)

@app.route('/ideal-conditions')
def ideal_conditions():
    current_lang = get_current_language()
    translated_content = {
        'title': translate_text('Ideal Growing Conditions', current_lang),
        'description': translate_text('Learn about the optimal conditions for growing different crops', current_lang)
    }
    translated_details = translate_dict_values(crop_details, current_lang)
    return render_template('ideal_conditions.html', content=translated_content, crop_details=translated_details)

@app.route('/yield-prediction')
def yield_prediction():
    current_lang = get_current_language()
    translated_content = {
        'title': translate_text('Yield Prediction', current_lang),
        'description': translate_text('Predict your crop yield based on various parameters', current_lang)
    }
    return render_template('yield_prediction.html', content=translated_content)

@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    try:
        # Get values from the JSON request
        data = request.get_json()
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Create feature array
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Scale the features
        features_scaled = recommendation_scaler.transform(features)
        
        # Make prediction
        prediction = crop_recommendation_model.predict(features_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({
            'success': True,
            'crop': str(predicted_crop)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Get values from the JSON data
        crop = data['crop'].lower()
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Create feature array
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale the features
        features_scaled = yield_scaler.transform(features)

        # Add crop encoding
        crop_encoded = np.zeros((features.shape[0], 1))
        if crop == 'rice':
            crop_encoded[0] = 0
        elif crop == 'wheat':
            crop_encoded[0] = 1
        elif crop == 'maize':
            crop_encoded[0] = 2
        elif crop == 'cotton':
            crop_encoded[0] = 3
        elif crop == 'sugarcane':
            crop_encoded[0] = 4
        else:
            raise ValueError(f"Unsupported crop type: {crop}")

        # Combine crop encoding with scaled features
        features_final = np.hstack((crop_encoded, features_scaled))

        # Make prediction
        prediction = yield_prediction_model.predict(features_final)
        
        # Validate and adjust the prediction
        validated_yield = validate_yield_prediction(crop, float(prediction[0]))
        
        # Get contextual information
        context = get_yield_context(crop)
        
        return jsonify({
            'success': True,
            'yield': validated_yield,
            'period': context['period'],
            'seasons_per_year': context['seasons_per_year']
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/crop-disease', methods=['GET', 'POST'])
def crop_disease():
    current_lang = get_current_language()
    translated_content = {
        'title': translate_text('Crop Disease Detection', current_lang),
        'description': translate_text('Upload a leaf image to detect crop diseases', current_lang)
    }
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': translate_text('No file uploaded', current_lang)})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': translate_text('No file selected', current_lang)})
        
        try:
            # Save the uploaded file
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_disease(disease_model, filepath)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            # Translate the disease name
            translated_disease = translate_text(result['disease'], current_lang)
            
            return jsonify({
                'success': True,
                'disease': translated_disease,
                'confidence': result['confidence'],
                'message': translate_text(
                    f'Detected: {translated_disease} with {result["confidence"]:.2%} confidence',
                    current_lang
                )
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': translate_text(str(e), current_lang)
            })
    
    return render_template('crop_disease.html', content=translated_content)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
