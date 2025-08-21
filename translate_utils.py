from deep_translator import GoogleTranslator
from flask import session
import time

def translate_text(text, target_language):
    if not text:
        return text
        
    if target_language == 'en':
        return text

    # Map Flask language codes to Google Translator codes
    language_map = {
        'hi': 'hi',  # Hindi
        'kn': 'kn',  # Kannada
        'en': 'en'   # English
    }
    
    try:
        # Add a small delay to avoid rate limiting
        time.sleep(0.2)
        translator = GoogleTranslator(source='en', target=language_map.get(target_language, 'en'))
        result = translator.translate(text)
        return result if result else text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def get_supported_languages():
    return {
        'en': 'English',
        'hi': 'हिंदी',
        'kn': 'ಕನ್ನಡ'
    }

def get_current_language():
    return session.get('language', 'en') 