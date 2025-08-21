import numpy as np

# Realistic yield ranges (tonnes per hectare)
YIELD_RANGES = {
    'rice': {
        'min': 2.5,
        'max': 4.5,
        'period': 'season',
        'seasons_per_year': '2-3'
    },
    'wheat': {
        'min': 2.0,
        'max': 3.5,
        'period': 'year',
        'seasons_per_year': '1'
    },
    'maize': {
        'min': 2.5,
        'max': 4.0,
        'period': 'season',
        'seasons_per_year': '2'
    },
    'cotton': {
        'min': 1.2,
        'max': 2.0,
        'period': 'year',
        'seasons_per_year': '1'
    },
    'sugarcane': {
        'min': 40.0,
        'max': 60.0,
        'period': 'year',
        'seasons_per_year': '1'
    }
}

def validate_yield_prediction(crop: str, predicted_yield: float) -> float:
    """
    Validates and adjusts yield predictions to ensure they fall within realistic ranges.
    
    Args:
        crop: The crop type (lowercase)
        predicted_yield: The predicted yield in tonnes per hectare
    
    Returns:
        float: Adjusted yield prediction within realistic range
    """
    if crop not in YIELD_RANGES:
        raise ValueError(f"Unknown crop type: {crop}")
    
    crop_range = YIELD_RANGES[crop]
    min_yield = crop_range['min']
    max_yield = crop_range['max']
    
    # If prediction is outside realistic range, clip it
    if predicted_yield < min_yield:
        return min_yield
    elif predicted_yield > max_yield:
        return max_yield
    
    return predicted_yield

def get_yield_context(crop: str) -> dict:
    """
    Get contextual information about the crop's yield.
    
    Args:
        crop: The crop type (lowercase)
    
    Returns:
        dict: Context information including period and seasons per year
    """
    if crop not in YIELD_RANGES:
        raise ValueError(f"Unknown crop type: {crop}")
    
    return {
        'period': YIELD_RANGES[crop]['period'],
        'seasons_per_year': YIELD_RANGES[crop]['seasons_per_year']
    }

def convert_yield_units(yield_value: float, from_unit: str = 'tonnes', to_unit: str = 'kg') -> float:
    """
    Convert yield between different units.
    
    Args:
        yield_value: The yield value to convert
        from_unit: Source unit ('tonnes', 'quintals', or 'kg')
        to_unit: Target unit ('tonnes', 'quintals', or 'kg')
    
    Returns:
        float: Converted yield value
    """
    # Conversion factors
    CONVERSIONS = {
        'tonnes': {'kg': 1000, 'quintals': 10},
        'quintals': {'kg': 100, 'tonnes': 0.1},
        'kg': {'tonnes': 0.001, 'quintals': 0.01}
    }
    
    if from_unit == to_unit:
        return yield_value
    
    if from_unit not in CONVERSIONS or to_unit not in CONVERSIONS[from_unit]:
        raise ValueError(f"Invalid unit conversion from {from_unit} to {to_unit}")
    
    return yield_value * CONVERSIONS[from_unit][to_unit] 