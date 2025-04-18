"""
Voice Transcription Module using OpenAI's Whisper API.

This module handles transcription of voice messages from Telegram using the OpenAI API.
"""
import os
import tempfile
import logging
from typing import Optional
import httpx
from openai import OpenAI
import config

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("whisper_transcriber")

class WhisperTranscriber:
    """
    Transcribes voice messages using OpenAI's Whisper API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the WhisperTranscriber with OpenAI API.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)

        if not self.api_key:
            raise ValueError("OpenAI API key is required for voice transcription")

    async def download_voice_file(self, file_url: str, token: str) -> str:
        """
        Download a voice file from Telegram.

        Args:
            file_url: URL to the voice file
            token: Telegram bot token for authentication

        Returns:
            Path to the downloaded temporary file
        """
        # Create a temporary file to store the voice message
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ogg')
        temp_path = temp_file.name
        temp_file.close()

        # Download the file
        try:
            # Prepare the URL if it's a file_id instead of a full URL
            if not file_url.startswith('http'):
                file_url = f"https://api.telegram.org/file/bot{token}/{file_url}"

            logger.info(f"Downloading voice file from: {file_url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(file_url)
                response.raise_for_status()

                # Write the content to the temporary file
                with open(temp_path, 'wb') as f:
                    f.write(response.content)

            logger.info(f"Voice file downloaded successfully to: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Error downloading voice file: {e}")
            # Clean up the temporary file if download failed
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    async def transcribe(self, file_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe a voice file using OpenAI's Whisper API.

        Args:
            file_path: Path to the voice file
            language: Optional language code to assist transcription

        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing voice file: {file_path}")

            # Prepare transcription options
            options = {}
            if language:
                options["language"] = language

            # Open the audio file
            with open(file_path, "rb") as audio_file:
                # Call the OpenAI API for transcription
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    **options
                )

            # Extract the transcribed text
            transcription = response.text
            logger.info(f"Transcription successful: {transcription[:50]}...")

            return transcription
        except Exception as e:
            logger.error(f"Error transcribing voice file: {e}")
            return ""
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Deleted temporary file: {file_path}")

# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_transcription():
        if "OPENAI_API_KEY" not in os.environ:
            print("Warning: OPENAI_API_KEY environment variable not set")
            return

        transcriber = WhisperTranscriber()

        # Example: Download and transcribe a voice file
        # In a real scenario, you would get this URL from Telegram
        test_url = "https://example.com/voice.ogg"
        test_token = "YOUR_TELEGRAM_BOT_TOKEN"

        try:
            file_path = await transcriber.download_voice_file(test_url, test_token)
            transcription = await transcriber.transcribe(file_path)
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"Error during test: {e}")

    asyncio.run(test_transcription())