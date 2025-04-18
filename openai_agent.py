"""
OpenAI Formatting Agent for restaurant recommendations.

This module formats the raw search results into friendly,
engaging restaurant recommendations using OpenAI.
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import config
from langsmith import traceable
from html.parser import HTMLParser
from html import escape


class RestaurantFormattingAgent:
    """
    An agent that uses OpenAI to format raw restaurant search results
    with a friendly, engaging, and somewhat humorous tone.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        temperature: float = None,
        system_prompt: Optional[str] = None
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else config.OPENAI_TEMPERATURE
        self.system_prompt = system_prompt or config.RESTAURANT_TOV_PROMPT

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self._get_human_prompt_template())
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def _get_human_prompt_template(self) -> str:
        return """
        USER QUERY: {query}
        USER LANGUAGE: {language}

        RESTAURANT SEARCH RESULTS:
        {restaurant_results}

        Please format these restaurant recommendations according to the specified tone and format. Be conversational and friendly. Make sure to detect the language of the query and respond in the same language.
        Use HTML formatting (<b>, <i>, <a>) for titles, highlights, and links. Output must be valid Telegram-compatible HTML.
        """

    @traceable(name="format_restaurant_results")
    def format(self, query: str, restaurant_results: List[Dict[str, Any]], language: str = "English") -> str:
        if isinstance(restaurant_results, str):
            results_str = restaurant_results
        else:
            results_str = json.dumps(restaurant_results, indent=2)

        formatted_response = self.chain.invoke({
            "query": query,
            "restaurant_results": results_str,
            "language": language
        })

        return self.sanitize_telegram_html(formatted_response)

    @staticmethod
    def sanitize_telegram_html(text: str) -> str:
        """
        Fully sanitize and auto-close AI-generated HTML for Telegram compatibility.
        Allows only <b>, <i>, <a href="">.
        """
        class TelegramHTMLSanitizer(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = []
                self.allowed_tags = {'b', 'i', 'a'}
                self.open_tags = []

            def handle_starttag(self, tag, attrs):
                tag = tag.lower()
                if tag not in self.allowed_tags:
                    return
                if tag == 'a':
                    href = next((v for k, v in attrs if k == 'href'), None)
                    if href:
                        self.result.append(f'<a href="{escape(href)}">')
                        self.open_tags.append('a')
                else:
                    self.result.append(f'<{tag}>')
                    self.open_tags.append(tag)

            def handle_endtag(self, tag):
                tag = tag.lower()
                if tag in self.allowed_tags and tag in self.open_tags:
                    self.result.append(f'</{tag}>')
                    self.open_tags.remove(tag)  # remove first open match

            def handle_data(self, data):
                self.result.append(escape(data))

            def get_data(self):
                while self.open_tags:
                    tag = self.open_tags.pop()
                    self.result.append(f'</{tag}>')
                return ''.join(self.result)

        parser = TelegramHTMLSanitizer()
        parser.feed(text)
        return parser.get_data()


# Example usage for testing
if __name__ == "__main__":
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set")

    formatter = RestaurantFormattingAgent()

    mock_results = [
        {
            "name": "Sushi Nakazawa",
            "description": "Acclaimed sushi restaurant offering an omakase experience with fish sourced from around the world.",
            "url": "https://guide.michelin.com/us/en/new-york-state/new-york/restaurant/sushi-nakazawa",
            "source": "guide.michelin.com",
            "score": 0.95
        },
        {
            "name": "Le Bernardin",
            "description": "Upscale French seafood restaurant with three Michelin stars, known for exquisite seafood preparations.",
            "url": "https://www.cntraveler.com/restaurants/new-york/le-bernardin",
            "source": "cntraveler.com",
            "score": 0.92
        }
    ]

    formatted = formatter.format("best seafood restaurants in NYC", mock_results)
    print(formatted)
