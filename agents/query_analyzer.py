# agents/query_analyzer.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tracers.context import tracing_v2_enabled
import json
import re

class QueryAnalyzer:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2
        )

        # Replace the system prompt in your query_analyzer.py with this clean version

        self.system_prompt = """
        You are a restaurant recommendation system that analyzes user queries about restaurants.
        Your task is to extract key information and prepare search queries for a web search.

        SEARCH STRATEGY:
        1. Identify PRIMARY search parameters that are likely to return comprehensive good quality results online and transform the user's request into search terms that will find these curated lists.

           EXAMPLES OF TRANSFORMATIONS:
           - User asks: "Where can I take my wife for our anniversary in Paris?"
             Search query: "romantic restaurants in Paris"
           - User asks: "I need somewhere kid-friendly in Rome with pizza"
             Search query: "family-friendly restaurants in Rome"
           - User asks: "We want to try authentic local food in Tokyo"
             Search query: "traditional Japanese restaurants in Tokyo"
           - User asks: "Looking for somewhere with a nice view in New York"
             Search query: "restaurants with view in New York"
           - User asks: "where to drink some wine in Bordeaux"
             Search query: "best wine bars and restaurants in Bordeaux"
           - User asks: "Any interesting restaurant openings in London?"
             Search query: "new restaurants in London 2025"    

        GUIDELINES:
        1. Extract the destination (only the city name, standartized international form) from the query. If the name has special characters, discard of them. If the user is using a shortened name or a nickname,like LA, Frisco or Big Apple, convert it to the full city name. 
        2. Determine if the destination is English-speaking or not  
        3. For USA destinations add the word "media" to the search query
        4. For non-English speaking destinations, identify the local language
        5. Create appropriate search queries in English and local language (for non-English destinations)
        6. For the query in local language, don't just translate word for word from English, create an original query that will return better search results in this language
        7. Extract keywords for analysis based on user request

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

    # Enhancement to your existing query_analyzer.py
    # Update the analyze method to provide clearer query separation

    def analyze(self, query, standing_prefs=None):
        """
        CLEAN: Analyze the user's query and extract relevant search parameters
        Focuses on destination, language detection, and AI-powered search query generation
        """
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

                location = result.get("destination")
                is_english_speaking = result.get("is_english_speaking", True)

                # Format search queries with clear separation
                english_query = result.get("english_search_query")
                local_query = result.get("local_language_search_query")

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

                # Clean up search queries
                search_queries = [q for q in search_queries if q]

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
                print(f"Error parsing response: {e}")
                print(f"Response content: {response.content}")

                # CLEAN: Minimal fallback - let AI handle this in a retry or return minimal data
                location = "Unknown"
                for indicator in ["in ", "near ", "at "]:
                    if indicator in query.lower():
                        parts = query.lower().split(indicator)
                        if len(parts) > 1:
                            possible_location = parts[1].split()[0]
                            if len(possible_location) > 2:
                                location = possible_location
                                break

                # Try to determine language
                is_english_speaking = True
                local_language = None

                if location != "Unknown":
                    try:
                        language_prompt = f"""
                        Is {location} in a primarily English-speaking country? Answer with only 'yes' or 'no'.
                        If 'no', what is the primary local language? Just provide the language name.
                        """

                        language_chain = ChatPromptTemplate.from_template(language_prompt) | self.model
                        language_response = language_chain.invoke({})
                        response_text = language_response.content.lower()

                        if 'no' in response_text:
                            is_english_speaking = False
                            if '\n' in response_text:
                                language_line = response_text.split('\n')[1].strip()
                                local_language = language_line
                    except Exception as lang_error:
                        print(f"Error determining language: {lang_error}")

                # CLEAN: Minimal fallback - no generic queries, let the system handle it upstream
                return {
                    "raw_query": query,  # IMPORTANT: Always preserve raw query
                    "destination": location,
                    "is_english_speaking": is_english_speaking,
                    "local_language": local_language,
                    "search_queries": [],  # Empty - let upstream handle the failure
                    "english_queries": [],
                    "local_queries": [],
                    # Minimal query metadata
                    "query_metadata": {
                        "is_english_speaking": is_english_speaking,
                        "local_language": local_language,
                        "english_query": None,
                        "local_query": None
                    }
                }

    
    def _compile_local_sources(self, location, language):
        """
        Compile a list of reputable local sources for restaurant recommendations

        Args:
            location (str): City or location name
            language (str): Local language

        Returns:
            list: List of local sources
        """
        local_sources_prompt = f"""
        Identify 5-7 reputable local sources for restaurant reviews and food recommendations in {location}.
        Focus on local press, respected food experts, bloggers, and local food guides that publish in {language}.
        Do NOT include generic content sites like TripAdvisor, Opentable, Yelp, or Google. Only include sources with professionally curated content.

        Each source should be either:
        1) A local newspaper/magazine with a dedicated food section
        2) A respected local food blog with in-depth reviews
        3) A local food award organization
        4) A local guide/publication specifically focused on restaurants and dining
        5) A notable local chef or food personality with authoritative recommendations

        Return a JSON array with objects containing \"name\", \"url\" (if available), and \"type\" (one of the categories above).
        """

        local_sources_chain = ChatPromptTemplate.from_template(local_sources_prompt) | self.model
        response = local_sources_chain.invoke({})

        try:
            # Clean up response content to handle markdown formatting
            content = response.content

            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 3:  # Has opening and closing backticks
                    content = parts[1]  # Extract content between backticks

            # Strip whitespace
            content = content.strip()

            sources = json.loads(content)

            # Add metadata to help with searches
            for source in sources:
                source["city"] = location
                source["language"] = language

            return sources
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing local sources: {e}")
            return []