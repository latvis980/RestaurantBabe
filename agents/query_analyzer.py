from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tracers.context import tracing_v2_enabled
from utils.database import save_to_mongodb, find_in_mongodb

import json
import datetime
from typing import List, Optional
from pydantic import BaseModel, ValidationError


class QueryResultSchema(BaseModel):
    destination: str
    is_english_speaking: bool
    local_language: Optional[str]
    primary_search_parameters: List[str]
    secondary_filter_parameters: List[str]
    english_search_query: Optional[str]
    local_language_search_query: Optional[str]
    keywords_for_analysis: List[str]
    user_preferences: Optional[str]
    local_sources: Optional[List[dict]] = []


class QueryAnalyzer:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2
        )

        self.config = config

        self.system_prompt = """
        You are a restaurant recommendation system that analyzes user queries about restaurants.
        Your task is to extract key information and prepare search queries.

        SEARCH STRATEGY:
        1. First, identify PRIMARY search parameters that are likely to have existing curated lists online
           (e.g., "romantic restaurants in Paris", "best brunch in Tokyo")
        2. Then, identify SECONDARY parameters that will be used for filtering and detailed analysis later
           (e.g., "gluten-free options", "has outdoor seating", "serves oysters")

        GUIDELINES:
        1. Extract the destination (city/country) from the query
        2. Determine if the destination is English-speaking or not
        3. For non-English speaking destinations, identify the local language
        4. Create appropriate search queries in English and local language (for non-English destinations)
           - Search queries should focus ONLY on primary parameters
        5. Extract or create keywords for analysis based on user preferences
           - Analysis keywords should include ALL parameters (primary and secondary)

        EXCLUDE from recommendations:
        - Tripadvisor
        - Yelp
        - Google Maps reviews

        OUTPUT FORMAT:
        Respond with a JSON object containing:
        {
          "destination": "extracted city/country", 
          "is_english_speaking": true/false,
          "local_language": "language name (if not English-speaking)",
          "primary_search_parameters": ["param1", "param2", ...],
          "secondary_filter_parameters": ["param1", "param2", ...],
          "english_search_query": "search query in English using only primary parameters",
          "local_language_search_query": "search query in local language (if applicable) using only primary parameters",
          "keywords_for_analysis": ["all keywords including primary and secondary"],
          "user_preferences": "brief summary of what makes this request unique"
        }
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])

        self.chain = self.prompt | self.model

    def analyze(self, query):
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            response = self.chain.invoke({"query": query})

            try:
                result = json.loads(response.content)
                validated = QueryResultSchema(**result)

                location = validated.destination
                is_english_speaking = self._is_english_speaking_city_openai(location)
                validated.is_english_speaking = is_english_speaking

                # Save local sources if city is non-English-speaking and not yet saved
                if location and not is_english_speaking:
                    local_sources = find_in_mongodb(
                        self.config.MONGODB_COLLECTION_SOURCES,
                        {"location": location},
                        self.config
                    )

                    if not local_sources:
                        local_sources = self._compile_local_sources(location, validated.local_language)
                        save_to_mongodb(
                            self.config.MONGODB_COLLECTION_SOURCES,
                            {"location": location, "sources": local_sources},
                            self.config
                        )
                    validated.local_sources = local_sources

                return validated.dict()

            except (json.JSONDecodeError, AttributeError, ValidationError) as e:
                log_failed_query(query, response.content, str(e))
                return {
                    "destination": "Unknown",
                    "is_english_speaking": True,
                    "search_queries": [f"best restaurants {query}"],
                    "primary_search_parameters": [],
                    "secondary_filter_parameters": [],
                    "keywords_for_analysis": [],
                    "local_sources": [],
                    "user_preferences": ""
                }

    def _compile_local_sources(self, location, language):
        """Compile a list of reputable local sources for a non-English speaking location"""

        local_sources_prompt = f"""
        Identify 5-7 reputable local sources for restaurant reviews and food recommendations in {location}.
        Focus on local press, respected food experts, bloggers, and local food guides that publish in {language}.
        Do NOT include international sites like TripAdvisor, Yelp, or Google. Only include sources that locals would trust.

        Return a JSON array with objects containing "name" and "url" (if available).
        """

        local_sources_chain = ChatPromptTemplate.from_template(local_sources_prompt) | self.model
        response = local_sources_chain.invoke({})

        try:
            sources = json.loads(response.content)
            return sources
        except (json.JSONDecodeError, AttributeError):
            return []

    def _is_english_speaking_city_openai(self, city_name):
        """Use OpenAI to determine if a city is English-speaking in local life/media"""
        prompt = f"""Is {city_name} an English-speaking city in daily life and local media? 
Respond with only "yes" or "no"."""

        response = self.model.invoke(prompt)
        content = response.content.strip().lower()
        return content.startswith("yes")


def log_failed_query(query, response_content, error):
    with open("bad_queries.log", "a", encoding="utf-8") as f:
        f.write("\n--- FAILED QUERY ---\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Error: {error}\n")
        f.write("Response content:\n")
        f.write(response_content + "\n")
