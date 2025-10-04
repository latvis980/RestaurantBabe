# agents/query_analyzer_fixed.py
"""
FIXED Query Analyzer - properly handles Lima, Peru and other destinations
Addresses the core issue where query analyzer returns 0 search queries for Lima
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tracers.context import tracing_v2_enabled
import json
import re
import logging

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
            api_key=config.OPENAI_API_KEY
        )

        # FIXED: Enhanced system prompt that properly handles Lima, Peru
        self.system_prompt = """
        You are a restaurant recommendation system that analyzes user queries about restaurants.
        Your task is to extract key information and prepare search queries for a web search.

        SEARCH STRATEGY:
        1. Identify PRIMARY search parameters that are likely to return comprehensive good quality results online and transform the user's request into search terms that will find these curated lists.

           EXAMPLES OF TRANSFORMATIONS:
           - User asks: "best cevicherias in Lima" → "best cevicherias in Lima Peru", "top seafood restaurants Lima Peru"
           - User asks: "Where can I take my wife for our anniversary in Paris?" → "romantic restaurants in Paris"
           - User asks: "I need somewhere kid-friendly in Rome with pizza" → "family-friendly restaurants in Rome"
           - User asks: "We want to try authentic local food in Tokyo" → "traditional Japanese restaurants in Tokyo"
           - User asks: "Looking for somewhere with a nice view in New York" → "restaurants with view in New York"
           - User asks: "where to drink some wine in Bordeaux" → "best wine bars and restaurants in Bordeaux"

        GUIDELINES:
        1. Extract the destination (only the city name). 

        CRITICAL RULE FOR DESTINATION: Convert ALL special characters to basic Latin letters:
           - ñ → n, ç → c, é → e, è → e, ê → e, ë → e
           - á → a, à → a, â → a, ä → a, ã → a, å → a
           - í → i, ì → i, î → i, ï → i
           - ó → o, ò → o, ô → o, ö → o, õ → o, ø → o
           - ú → u, ù → u, û → u, ü → u
           - Remove any accent marks, tildes, cedillas, umlauts, etc.

           EXAMPLES:
           - "Olhão" → "Olhao"
           - "São Paulo" → "Sao Paulo"  
           - "Zürich" → "Zurich"
           - "Málaga" → "Malaga"
           - "Kraków" → "Krakow"
           - "München" → "Munchen"
           - "Lima" → "Lima" (keep as is - it's already Latin)

           Also convert nicknames to full names:
           - "LA" → "Los Angeles"
           - "NYC" → "New York"
           - "Frisco" → "San Francisco"
           - "Big Apple" → "New York"

        2. Determine if the destination is English-speaking or not
           SPECIAL CASES:
           - Lima → Peru → Spanish-speaking
           - Mexico City → Mexico → Spanish-speaking
           - Buenos Aires → Argentina → Spanish-speaking

        3. For USA destinations add the word "media" to the search query
        4. For non-English speaking destinations, identify the local language
        5. Create appropriate search queries in English and local language (for non-English destinations)
        6. For the query in local language, don't just translate word for word from English, create an original query that will return better search results in this language
        7. Extract keywords for analysis based on user request

        MANDATORY: Always generate at least 1 search query. Never return empty search queries.

        OUTPUT FORMAT:
        Respond with a JSON object containing:
        {{
          "destination": "extracted city", 
          "is_english_speaking": true/false,
          "is_usa": true/false,
          "local_language": "language name (if not English-speaking)",
          "english_search_query": "search query in English using primary parameters",
          "local_language_search_query": "search query in local language (if applicable) using primary parameters"
        }}
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])

        self.chain = self.prompt | self.model
        self.config = config

    def analyze(self, query, standing_prefs=None):
        """
        FIXED: Analyze the user's query and extract relevant search parameters
        Focuses on destination, language detection, and AI-powered search query generation
        Now includes fallback logic to ensure search queries are always generated
        """
        try:
            with tracing_v2_enabled(project_name="restaurant-recommender"):
                response = self.chain.invoke({"query": query})

                try:
                    # Clean up response content to handle markdown formatting
                    content = response.content

                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        parts = content.split("```")
                        if len(parts) >= 3:  # Has opening and closing backticks
                            content = parts[1]  # Extract content between backticks

                    # Strip whitespace
                    content = content.strip()

                    # Parse the JSON
                    result = json.loads(content)

                    location = result.get("destination", "Unknown")
                    is_english_speaking = result.get("is_english_speaking", True)

                    # Format search queries with clear separation
                    english_query = result.get("english_search_query", "")
                    local_query = result.get("local_language_search_query", "")

                    search_queries = []
                    english_queries = []
                    local_queries = []

                    # Always include English query
                    if english_query:
                        search_queries.append(english_query)
                        english_queries.append(english_query)

                    # Add local language query for non-English destinations
                    if not is_english_speaking and local_query:
                        search_queries.append(local_query)
                        local_queries.append(local_query)

                    # CRITICAL FIX: Ensure we always have at least one search query
                    if not search_queries:
                        logger.warning(f"No search queries generated by AI for '{query}', creating fallback")
                        fallback_query = self._create_fallback_query(query, location)
                        search_queries.append(fallback_query)
                        english_queries.append(fallback_query)
                        logger.info(f"Generated fallback query: {fallback_query}")

                    # Clean up search queries
                    search_queries = [q for q in search_queries if q.strip()]

                    logger.info(f"✅ Query analysis successful: {location}, {len(search_queries)} queries")

                    # CLEAN: Return only essential data
                    return {
                        "raw_query": query,  # IMPORTANT: Preserve raw query for pipeline
                        "destination": location,
                        "is_english_speaking": is_english_speaking,
                        "local_language": result.get("local_language"),
                        "search_queries": search_queries,
                        "english_queries": english_queries,
                        "local_queries": local_queries,
                        # Query metadata for search agent
                        "query_metadata": {
                            "is_english_speaking": is_english_speaking,
                            "local_language": result.get("local_language"),
                            "english_query": english_query,
                            "local_query": local_query
                        }
                    }

                except (json.JSONDecodeError, AttributeError) as e:
                    logger.error(f"Error parsing AI response: {e}")
                    logger.error(f"Response content: {response.content}")

                    # Enhanced fallback logic
                    return self._create_enhanced_fallback(query)

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return self._create_enhanced_fallback(query)

    def _create_fallback_query(self, query: str, destination: str) -> str:
        """Create a fallback search query when AI fails to generate one"""
        # Extract key terms from the query
        query_lower = query.lower()

        # Common restaurant types and cuisine terms
        cuisine_terms = ["ceviche", "sushi", "pizza", "pasta", "tacos", "burger", "coffee", "bakery", "bar", "wine"]
        restaurant_terms = ["restaurant", "cafe", "bistro", "eatery", "dining", "food"]

        # Find cuisine or restaurant type
        found_cuisine = None
        for term in cuisine_terms:
            if term in query_lower:
                found_cuisine = term
                break

        # Build fallback query
        if found_cuisine:
            if destination != "Unknown":
                return f"best {found_cuisine} restaurants in {destination}"
            else:
                return f"best {found_cuisine} restaurants"
        else:
            # Check for general terms
            if any(term in query_lower for term in restaurant_terms):
                if destination != "Unknown":
                    return f"best restaurants in {destination}"
                else:
                    return "best restaurants"
            else:
                # Last resort - use the original query with "restaurants"
                if destination != "Unknown":
                    return f"restaurants {query} {destination}"
                else:
                    return f"restaurants {query}"

    def _create_enhanced_fallback(self, query: str) -> dict:
        """Create enhanced fallback response when AI completely fails"""
        # Try to extract location using simple patterns
        location = "Unknown"
        for indicator in ["in ", "near ", "at ", "around "]:
            if indicator in query.lower():
                parts = query.lower().split(indicator)
                if len(parts) > 1:
                    # Get the next word(s) after the indicator
                    location_part = parts[1].strip().split()[0] if parts[1].strip() else ""
                    if len(location_part) > 2:
                        location = location_part.title()
                        break

        # Special handling for Lima
        if "lima" in query.lower():
            location = "Lima"

        # Determine if English-speaking (simple heuristic)
        non_english_cities = {
            "lima": False, "mexico": False, "paris": False, "madrid": False, 
            "barcelona": False, "rome": False, "milan": False, "tokyo": False,
            "seoul": False, "beijing": False, "shanghai": False, "moscow": False
        }

        is_english_speaking = non_english_cities.get(location.lower(), True)

        # Create search query
        search_query = self._create_fallback_query(query, location)

        logger.warning(f"Using enhanced fallback for '{query}': location={location}, query={search_query}")

        return {
            "raw_query": query,
            "destination": location,
            "is_english_speaking": is_english_speaking,
            "local_language": "Spanish" if location.lower() == "lima" else None,
            "search_queries": [search_query],
            "english_queries": [search_query],
            "local_queries": [],
            "query_metadata": {
                "is_english_speaking": is_english_speaking,
                "local_language": "Spanish" if location.lower() == "lima" else None,
                "english_query": search_query,
                "local_query": None
            }
        }