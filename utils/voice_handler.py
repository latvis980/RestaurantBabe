# utils/voice_handler.py
import logging
import os
import tempfile
import time
from typing import Optional, Dict, Any
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

class VoiceMessageHandler:
    """Handles voice message transcription using OpenAI Whisper"""

    def __init__(self):
        """Initialize the voice handler with OpenAI client"""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        logger.info("✅ Voice Message Handler initialized with Whisper")

    def transcribe_voice_message(self, voice_file_path: str) -> Optional[str]:
        """
        Transcribe a voice message using OpenAI Whisper

        Args:
            voice_file_path: Path to the voice file

        Returns:
            Transcribed text or None if transcription failed
        """
        try:
            logger.info(f"🎤 Starting transcription of voice file: {voice_file_path}")

            # Check if file exists and has reasonable size
            if not os.path.exists(voice_file_path):
                logger.error(f"Voice file not found: {voice_file_path}")
                return None

            file_size = os.path.getsize(voice_file_path)
            if file_size == 0:
                logger.error("Voice file is empty")
                return None

            if file_size > 25 * 1024 * 1024:  # 25MB limit for OpenAI
                logger.error(f"Voice file too large: {file_size} bytes")
                return None

            logger.info(f"📊 Voice file size: {file_size} bytes")

            # Transcribe using OpenAI Whisper
            with open(voice_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                    language=None,  # Auto-detect language
                    temperature=0.0  # More deterministic transcription
                )

            # Clean up the transcription
            transcribed_text = transcript.strip()

            if not transcribed_text:
                logger.warning("Transcription resulted in empty text")
                return None

            logger.info(f"✅ Transcription successful: '{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}'")
            return transcribed_text

        except Exception as e:
            logger.error(f"❌ Error transcribing voice message: {e}")
            return None

    def download_and_convert_voice(self, bot, voice_message) -> Optional[str]:
        """
        Download voice message from Telegram and prepare for transcription

        Args:
            bot: Telegram bot instance
            voice_message: Telegram voice message object

        Returns:
            Path to the downloaded voice file or None if failed
        """
        try:
            # Get file info from Telegram
            file_info = bot.get_file(voice_message.file_id)

            # Download the file
            downloaded_file = bot.download_file(file_info.file_path)

            # Create temporary file with .ogg extension (Telegram voice format)
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.ogg', 
                delete=False,
                prefix='voice_'
            )

            # Write downloaded content to temp file
            temp_file.write(downloaded_file)
            temp_file.close()

            logger.info(f"📥 Voice message downloaded to: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"❌ Error downloading voice message: {e}")
            return None

    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary voice file"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"🗑️ Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

    def process_voice_message(self, bot, voice_message) -> Optional[str]:
        """
        Complete pipeline: download, transcribe, and cleanup voice message

        Args:
            bot: Telegram bot instance
            voice_message: Telegram voice message object

        Returns:
            Transcribed text or None if processing failed
        """
        temp_file_path = None

        try:
            # Step 1: Download voice message
            temp_file_path = self.download_and_convert_voice(bot, voice_message)
            if not temp_file_path:
                return None

            # Step 2: Transcribe using Whisper
            transcribed_text = self.transcribe_voice_message(temp_file_path)

            return transcribed_text

        finally:
            # Step 3: Always cleanup temp file
            if temp_file_path:
                self.cleanup_temp_file(temp_file_path)

    def generate_voice_confirmation_message(self, transcribed_text: str) -> str:
        """
        Generate a friendly confirmation message that shows understanding without revealing transcription

        Args:
            transcribed_text: The transcribed voice message

        Returns:
            Confirmation message for the user
        """
        # Extract key elements for confirmation without showing full transcription
        text_lower = transcribed_text.lower()

        # Detect cuisine types
        cuisines = []
        cuisine_keywords = {
            'pizza': '🍕 pizza',
            'sushi': '🍣 sushi', 
            'ramen': '🍜 ramen',
            'pasta': '🍝 pasta',
            'burger': '🍔 burgers',
            'taco': '🌮 tacos',
            'coffee': '☕ coffee',
            'wine': '🍷 wine',
            'cocktail': '🍸 cocktails',
            'brunch': '🥞 brunch',
            'dessert': '🍰 desserts',
            'seafood': '🦐 seafood',
            'steakhouse': '🥩 steakhouse',
            'vegetarian': '🥗 vegetarian',
            'vegan': '🌱 vegan'
        }

        for keyword, emoji_cuisine in cuisine_keywords.items():
            if keyword in text_lower:
                cuisines.append(emoji_cuisine)

        # Detect location mentions (basic)
        location_indicators = ['in ', 'near ', 'around ', 'at ']
        has_location = any(indicator in text_lower for indicator in location_indicators)

        # Detect quality/style indicators
        quality_indicators = {
            'best': 'the best',
            'good': 'good',
            'great': 'great',
            'romantic': 'romantic',
            'fancy': 'upscale',
            'cheap': 'affordable',
            'family': 'family-friendly',
            'quiet': 'quiet',
            'trendy': 'trendy'
        }

        qualities = []
        for keyword, description in quality_indicators.items():
            if keyword in text_lower:
                qualities.append(description)

        # Build confirmation message
        if cuisines and has_location:
            cuisine_text = ', '.join(cuisines[:2])  # Limit to 2 cuisines
            confirmation = f"🎤 <b>Got it! Searching for {cuisine_text} places for you.</b>"
        elif cuisines:
            cuisine_text = ', '.join(cuisines[:2])
            confirmation = f"🎤 <b>Perfect! Looking for {cuisine_text} recommendations.</b>"
        elif has_location:
            confirmation = f"🎤 <b>Understood! Searching for restaurant recommendations in your area.</b>"
        else:
            confirmation = f"🎤 <b>Got your request! Let me find some great dining recommendations for you.</b>"

        # Add quality indicators if found
        if qualities:
            quality_text = ', '.join(qualities[:2])
            confirmation += f"\n\n✨ <i>Looking for {quality_text} options...</i>"

        confirmation += "\n\n⏱ <i>This might take a minute while I consult with my critic friends!</i>"

        return confirmation