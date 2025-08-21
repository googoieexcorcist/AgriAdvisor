# Smart Agriculture Assistant

A web application that helps farmers make data-driven decisions using machine learning models for crop recommendation and yield prediction.

## Features

- **Crop Recommendation**: Get personalized crop recommendations based on:
  - Soil composition (N, P, K values)
  - Environmental conditions (temperature, humidity, rainfall)
  - Soil pH level

- **Yield Prediction**: Predict crop yield based on:
  - Crop type
  - Area of cultivation
  - Environmental factors
  - Agricultural inputs (pesticides, fertilizers)

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
.
├── app.py                              # Flask application
├── requirements.txt                     # Project dependencies
├── static/                             # Static files (CSS, images)
├── templates/                          # HTML templates
│   ├── index.html                      # Landing page
│   ├── crop_recommendation.html        # Crop recommendation form
│   └── yield_prediction.html           # Yield prediction form
├── models/                             # Trained ML models
│   ├── crop_recommendation_model.pkl   # Crop recommendation model
│   ├── yield_prediction_model.pkl      # Yield prediction model
│   └── label_encoder.pkl               # Label encoder for crop names
└── README.md                           # Project documentation
```

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: scikit-learn, numpy, pandas
- **Model Storage**: joblib

## Model Information

### Crop Recommendation Model
- Input features: N, P, K, temperature, humidity, pH, rainfall
- Output: Recommended crop name
- Type: Classification model

### Yield Prediction Model
- Input features: crop type, area, temperature, rainfall, pesticides, fertilizer
- Output: Predicted yield (tonnes/hectare)
- Type: Regression model

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 