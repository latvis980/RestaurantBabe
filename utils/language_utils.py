# utils/language_utils.py
import requests
import logging

logger = logging.getLogger(__name__)

def detect_language(text):
    """
    Simple language detection based on character set patterns

    Args:
        text (str): Text to detect language

    Returns:
        str: Two-letter language code (en, ru, fr, etc.)
    """
    # Check for Cyrillic characters (Russian, Ukrainian, etc.)
    if any('\u0400' <= c <= '\u04FF' for c in text):
        return 'ru'  # Default to Russian for Cyrillic

    # Check for Latin characters (English, Spanish, French, etc.)
    elif any('a' <= c.lower() <= 'z' for c in text):
        return 'en'  # Default to English for Latin

    # Add more language detection patterns as needed

    # Default to English if nothing specific detected
    return 'en'

def format_address_for_language(address, language_code):
    """
    Format address according to language conventions

    Args:
        address (str): Original address
        language_code (str): Target language code

    Returns:
        str: Formatted address
    """
    if language_code == 'ru':
        # Russian address format typically has street name first, then number
        if ',' in address:
            parts = address.split(',')
            return ', '.join(parts)
        return address

    # Default format (Western style)
    return address