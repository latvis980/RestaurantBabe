# agents/translator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
import json

class TranslatorAgent:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1
        )

        # Create system prompt that includes HTML formatting for Telegram
        self.system_prompt = """
        You are a professional translator specializing in restaurant recommendations.
        Your task is to translate restaurant recommendations from any language to Russian, 
        and format them properly for display in a Telegram bot.

        GUIDELINES:
        1. Translate all content to natural, fluent Russian
        2. Format the output with HTML tags for Telegram:
           - Use <b>text</b> for bold (restaurant names, section headers)
           - Use <i>text</i> for italics (sources, footer)
        3. Keep restaurant names in their original form, but add Russian transliteration in parentheses if needed
        4. Keep dish names in original language but add Russian translations in parentheses
        5. Convert all prices and measurements to appropriate Russian formats
        6. Create a well-formatted output ready for Telegram display

        FORMAT THE OUTPUT AS:
        <b>🍽️ РЕКОМЕНДУЕМЫЕ РЕСТОРАНЫ:</b>

        <b>1. [Restaurant Name]</b>
        📍 [Address]
        [Description]
        <i>✅ Рекомендуют: [Sources]</i>

        <b>2. [Restaurant Name]</b>
        ...

        <b>💎 ДЛЯ СВОИХ:</b>

        <b>1. [Hidden Gem Name]</b>
        ...

        <i>Рекомендации составлены на основе анализа экспертных источников.</i>

        IMPORTANT: The total output should not exceed 4000 characters for Telegram.
        """

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Format these restaurant recommendations for Telegram in Russian:\n\n{content}")
        ])

        # Create chain
        self.chain = self.prompt | self.model

        self.config = config

    def translate(self, content):
        """
        Translate and format restaurant recommendations for Telegram in Russian

        Args:
            content (dict or str): The content to translate and format

        Returns:
            str: Formatted HTML text for Telegram
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                # If content is a dictionary, convert to JSON string
                if isinstance(content, dict):
                    # Handle both main_list and recommended formats
                    if "main_list" in content and "hidden_gems" in content:
                        content_to_format = content
                    elif "recommended" in content and "hidden_gems" in content:
                        # Convert to main_list format
                        content_to_format = {
                            "main_list": content["recommended"],
                            "hidden_gems": content["hidden_gems"]
                        }
                    else:
                        # Unknown format, just pass it through
                        content_to_format = content

                    content_str = json.dumps(content_to_format, ensure_ascii=False, indent=2)
                else:
                    content_str = content

                # Invoke the chain
                response = self.chain.invoke({"content": content_str})

                # Just return the text content directly
                return response.content

            except Exception as e:
                print(f"Error in translator agent: {e}")
                # Create a basic fallback response
                return self._create_fallback_response(content)

    def _create_fallback_response(self, content):
        """Create a basic fallback response if translation fails"""
        try:
            response = "<b>🍽️ РЕКОМЕНДУЕМЫЕ РЕСТОРАНЫ:</b>\n\n"

            # Get the restaurants list
            restaurants = []
            hidden_gems = []

            if isinstance(content, dict):
                if "main_list" in content:
                    restaurants = content["main_list"]
                elif "recommended" in content:
                    restaurants = content["recommended"]

                if "hidden_gems" in content:
                    hidden_gems = content["hidden_gems"]

            # Format main restaurants
            if restaurants:
                for i, restaurant in enumerate(restaurants, 1):
                    response += f"<b>{i}. {restaurant.get('name', 'Ресторан')}</b>\n"
                    if "address" in restaurant:
                        response += f"📍 {restaurant['address']}\n"
                    if "description" in restaurant:
                        response += f"{restaurant['description']}\n"
                    if "recommended_by" in restaurant and restaurant["recommended_by"]:
                        sources = restaurant["recommended_by"]
                        if isinstance(sources, list):
                            response += f"<i>✅ Рекомендуют: {', '.join(sources[:3])}</i>\n"
                        else:
                            response += f"<i>✅ Рекомендуют: {sources}</i>\n"
                    response += "\n"
            else:
                response += "К сожалению, рекомендуемые рестораны не найдены.\n\n"

            # Format hidden gems
            if hidden_gems:
                response += "<b>💎 ДЛЯ СВОИХ:</b>\n\n"
                for i, restaurant in enumerate(hidden_gems, 1):
                    response += f"<b>{i}. {restaurant.get('name', 'Ресторан')}</b>\n"
                    if "address" in restaurant:
                        response += f"📍 {restaurant['address']}\n"
                    if "description" in restaurant:
                        response += f"{restaurant['description']}\n"
                    if "recommended_by" in restaurant and restaurant["recommended_by"]:
                        sources = restaurant["recommended_by"]
                        if isinstance(sources, list):
                            response += f"<i>✅ Рекомендуют: {', '.join(sources[:3])}</i>\n"
                        else:
                            response += f"<i>✅ Рекомендуют: {sources}</i>\n"
                    response += "\n"

            # Add footer
            response += "<i>Рекомендации составлены на основе анализа экспертных источников.</i>"

            # Ensure we don't exceed Telegram's limit
            if len(response) > 4000:
                response = response[:3997] + "..."

            return response

        except Exception as e:
            print(f"Error creating fallback response: {e}")
            return "Извините, произошла ошибка при обработке результатов."