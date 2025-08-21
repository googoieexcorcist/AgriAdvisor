import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Crop_recommendation_raw_medium.csv")

# Step 1: Fix number format (convert commas to dots, then to float)
for col in ['temperature', 'humidity', 'ph', 'rainfall']:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 2: Handle missing values using mean imputation
num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Step 3: Remove duplicate rows
df = df.drop_duplicates()

# Step 4: Fix typos in crop names using a mapping
# Load the original correct labels to map
correct_labels = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean',
    'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon',
    'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
]

# Use fuzzy matching (optional) to correct typos
from fuzzywuzzy import process

def correct_crop_name(name):
    best_match = process.extractOne(name, correct_labels)
    if best_match[1] > 80:  # confidence threshold
        return best_match[0]
    return name  # leave as is if no good match

df['label'] = df['label'].apply(correct_crop_name)

# Step 5: Encode the crop labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Step 6: Final cleaned dataset ready for ML
X = df[num_cols]
y = df['label_encoded']

# Optional: Save cleaned dataset
df.to_csv("Crop_recommendation_cleaned.csv", index=False)

print("Preprocessing complete. Cleaned data saved as 'Crop_recommendation_cleaned.csv'.")
