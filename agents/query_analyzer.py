from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tracers.context import tracing_v2_enabled
import json
from utils.database import save_to_mongodb, find_in_mongodb

class QueryAnalyzer:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2
        )

        # Create prompt template
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
        {{
          "destination": "extracted city/country", 
          "is_english_speaking": true/false,
          "local_language": "language name (if not English-speaking)",
          "primary_search_parameters": ["param1", "param2", ...],
          "secondary_filter_parameters": ["param1", "param2", ...],
          "english_search_query": "search query in English using only primary parameters",
          "local_language_search_query": "search query in local language (if applicable) using only primary parameters",
          "keywords_for_analysis": ["all keywords including primary and secondary"],
          "user_preferences": "brief summary of what makes this request unique"
        }}

        EXAMPLES:
        For "I want to find romantic restaurants in Paris with gluten-free options and oysters":
        - Primary parameters: ["Paris", "romantic", "restaurants"]
        - Secondary parameters: ["gluten-free", "oysters"]
        - English search query: "best romantic restaurants in Paris"
        - French search query: "meilleurs restaurants romantiques à Paris"
        - Analysis keywords: ["Paris", "romantic", "restaurants", "gluten-free", "oysters"]

        For "Looking for seafood restaurants with a terrace in Barcelona where they allow children":
        - Primary parameters: ["Barcelona", "seafood", "restaurants"]
        - Secondary parameters: ["outdoor seating", "kid-friendly"]
        - English search query: "best seafood restaurants in Barcelona"
        - Spanish search query: "mejores restaurantes de mariscos en Barcelona"
        - Analysis keywords: ["Barcelona", "seafood", "restaurants", "outdoor seating", "kid-friendly"]
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])

        # Create chain
        self.chain = self.prompt | self.model

        self.config = config

    def analyze(self, query):
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Invoke the chain
            response = self.chain.invoke({"query": query})

            try:
                # Parse the JSON response
                result = json.loads(response.content)

                # Check if we have local sources for non-English speaking destinations
                location = result.get("destination")
                is_english_speaking = result.get("is_english_speaking", True)

                if location and not is_english_speaking:
                    # Try to find local sources in database
                    local_sources = find_in_mongodb(
                        self.config.MONGODB_COLLECTION_SOURCES,
                        {"location": location},
                        self.config
                    )

                    # If no local sources found, compile and save them
                    if not local_sources:
                        local_sources = self._compile_local_sources(location, result.get("local_language"))
                        # Save to database for future use
                        save_to_mongodb(
                            self.config.MONGODB_COLLECTION_SOURCES,
                            {"location": location, "sources": local_sources},
                            self.config
                        )

                    result["local_sources"] = local_sources

                # Create search queries
                search_queries = [result.get("english_search_query")]

                # Add local language query if available
                if result.get("local_language_search_query"):
                    search_queries.append(result.get("local_language_search_query"))

                # Remove None or empty queries
                search_queries = [q for q in search_queries if q]

                # Ensure keywords_for_analysis is always a list
                keywords = result.get("keywords_for_analysis", [])
                if isinstance(keywords, str):
                    # Split the string by commas if it's a string
                    keywords = [k.strip() for k in keywords.split(",") if k.strip()]

                # Get primary and secondary parameters
                primary_params = result.get("primary_search_parameters", [])
                secondary_params = result.get("secondary_filter_parameters", [])

                # Ensure they're lists
                if isinstance(primary_params, str):
                    primary_params = [p.strip() for p in primary_params.split(",") if p.strip()]
                if isinstance(secondary_params, str):
                    secondary_params = [p.strip() for p in secondary_params.split(",") if p.strip()]

                return {
                    "destination": location,
                    "is_english_speaking": is_english_speaking,
                    "local_language": result.get("local_language"),
                    "search_queries": search_queries,
                    "primary_search_parameters": primary_params,
                    "secondary_filter_parameters": secondary_params,
                    "keywords_for_analysis": keywords,
                    "local_sources": result.get("local_sources", []),
                    "user_preferences": result.get("user_preferences", "")
                }

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing response: {e}")
                print(f"Response content: {response.content}")
                # Return a basic analysis to avoid complete failure
                return {
                    "destination": "Unknown",
                    "is_english_speaking": True,
                    "search_queries": [f"best restaurants {query}"],
                    "primary_search_parameters": ["restaurants"],
                    "secondary_filter_parameters": [],
                    "keywords_for_analysis": query.split(),
                    "local_sources": []
                }

    def _compile_local_sources(self, location, language):
        """Compile a list of reputable local sources for a non-English speaking location"""

        # Create a prompt to find local sources
        local_sources_prompt = f"""
        Identify 5-7 reputable local sources for restaurant reviews and food recommendations in {location}.
        Focus on local press, respected food experts, bloggers, and local food guides that publish in {language}.
        Do NOT include international sites like TripAdvisor, Yelp, or Google. Only include sources that locals would trust.

        Return a JSON array with objects containing "name" and "url" (if available).
        """

        # Create a one-off chain for this purpose
        local_sources_chain = ChatPromptTemplate.from_template(local_sources_prompt) | self.model

        # Invoke the chain
        response = local_sources_chain.invoke({})

        try:
            # Extract JSON array from response
            sources = json.loads(response.content)
            return sources
        except (json.JSONDecodeError, AttributeError):
            # Return empty list if parsing fails
            return []