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

        # Create system prompt
        self.system_prompt = """
        You are a professional translator specializing in restaurant and culinary content.
        Your task is to translate restaurant recommendations from any language to Russian, 
        maintaining all the nuances, specialized culinary terms, and formatting.

        GUIDELINES:
        1. Translate all content to natural, fluent Russian
        2. Preserve all formatting (bold text, bullet points, etc.)
        3. Keep restaurant names and chef names in their original form, but add Russian transliteration in parentheses
        4. Keep dish names in original language but add Russian translations in parentheses
        5. Convert all prices and measurements to appropriate Russian formats
        6. Preserve all JSON structure and field names (do not translate field names)
        7. Make sure the translation sounds natural to Russian speakers

        For JSON content, translate only the text values, NOT the keys or structure.
        """

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Translate this restaurant recommendation content to Russian:\n\n{content}")
        ])

        # Create chain
        self.chain = self.prompt | self.model

        self.config = config

    def translate(self, content):
        """
        Translate restaurant recommendations to Russian

        Args:
            content (dict or str): The content to translate

        Returns:
            dict or str: Translated content in the same format
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # If content is a dictionary, convert to JSON string
            if isinstance(content, dict):
                content_str = json.dumps(content, ensure_ascii=False, indent=2)
            else:
                content_str = content

            # Invoke the chain
            response = self.chain.invoke({"content": content_str})

            try:
                # If original content was a dictionary, parse the response back to a dictionary
                if isinstance(content, dict):
                    translated_content = json.loads(response.content)
                    return translated_content
                else:
                    return response.content
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing translator response: {e}")
                print(f"Response content: {response.content}")

                # Return the original content if parsing fails
                return content