import pandas as pd
import numpy as np

# Load the messy dataset
df = pd.read_csv("crop_yield_dataset_messy.csv")

# Show initial shape
print("Original dataset shape:", df.shape)

# Remove rows with invalid crop names
valid_crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'groundnut', 'barley', 'millet', 'banana', 'soybean']
df = df[df['Crop'].isin(valid_crops)]

# Replace empty strings and 'N/A' with NaN
df.replace(['', 'N/A', 'unknown'], np.nan, inplace=True)

# Convert numeric columns to float
numeric_columns = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Yield']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with column mean
for col in numeric_columns[:-1]:  # exclude 'Yield' for now
    df[col].fillna(df[col].mean(), inplace=True)

# Drop rows where 'Yield' is still missing or non-numeric
df.dropna(subset=['Yield'], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Show final shape and sample
print("Cleaned dataset shape:", df.shape)
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_crop_yield_dataset.csv", index=False)
print("Cleaned dataset saved as 'cleaned_crop_yield_dataset.csv'")
