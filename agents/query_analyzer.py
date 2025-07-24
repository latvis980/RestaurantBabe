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

        self.system_prompt = """
        You are a restaurant recommendation system that analyzes user queries about restaurants.
        Your task is to extract key information and prepare search queries.

        SEARCH STRATEGY:
        1. First, identify PRIMARY search parameters that are likely to have existing curated lists online
           and transform the user's request into search terms that will find these curated lists.

           EXAMPLES OF TRANSFORMATIONS:
           - User asks: "Where can I take my wife for our anniversary in Paris?"
             Search query: "romantic restaurants in Paris"

           - User asks: "I need somewhere kid-friendly in Rome with pizza"
             Search query: "family-friendly restaurants in Rome"

           - User asks: "We want to try authentic local food in Tokyo"
             Search query: "traditional Japanese restaurants in Tokyo"

           - User asks: "Looking for somewhere with a nice view in New York"
             Search query: "restaurants with view in New York"

        2. Then, identify SECONDARY parameters that will be used for filtering and detailed analysis later.
           These are the specific preferences that won't be part of the main search but will be used
           to filter results afterward.

           EXAMPLES OF SECONDARY PARAMETERS:
           - "serves oysters" or "has seafood"
           - "formal dress code" or "elegant atmosphere"
           - "outdoor seating" or "garden"
           - "tasting menu" or "chef's table"

        GUIDELINES:
        1. Extract the destination (city/country) from the query
        2. Determine if the destination is English-speaking or not
        3. For non-English speaking destinations, identify the local language
        4. Create appropriate search queries in English and local language (for non-English destinations)
        5. Extract or create keywords for analysis based on user preferences

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
          "keywords_for_analysis": ["all keywords including primary and secondary"]
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
        Analyze the user's query and extract relevant search parameters

        Args:
            query (str): The user's query about restaurant recommendations
            standing_prefs (list, optional): List of user's standing preferences

        Returns:
            dict: Extracted search parameters
        """
        standing_prefs = standing_prefs or []
        original_query_lower = query.lower()

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

                # Format search queries - always include English query
                search_queries = [result.get("english_search_query")]

                # Add only one local language query for non-English speaking locations
                if not is_english_speaking and result.get("local_language_search_query"):
                    search_queries.append(result.get("local_language_search_query"))

                # Clean up search queries
                search_queries = [q for q in search_queries if q]

                # Format keywords and parameters
                keywords = result.get("keywords_for_analysis", [])
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in keywords.split(",") if k.strip()]

                primary_params = result.get("primary_search_parameters", [])
                secondary_params = result.get("secondary_filter_parameters", [])

                if isinstance(primary_params, str):
                    primary_params = [p.strip() for p in primary_params.split(",") if p.strip()]
                if isinstance(secondary_params, str):
                    secondary_params = [p.strip() for p in secondary_params.split(",") if p.strip()]

                # Handle standing preferences
                NEGATION_PATTERNS = [
                    r"\bnot\s+(?:necessarily\s+)?{p}\b",
                    r"\bбез\s+{p}\b",                    
                    r"\bnon-{p}\b",
                    r"\bexcept\s+{p}\b",
                ]

                for pref in standing_prefs:
                    skip = False
                    for pat in NEGATION_PATTERNS:
                        if re.search(pat.format(p=re.escape(pref)), original_query_lower):
                            skip = True
                            break
                    if skip:
                        continue
                    # add pref only if it isn't already captured
                    if pref not in secondary_params and pref not in primary_params:
                        secondary_params.append(pref)

                return {
                    "destination": location,
                    "is_english_speaking": is_english_speaking,
                    "local_language": result.get("local_language"),
                    "search_queries": search_queries,
                    "primary_search_parameters": primary_params,
                    "secondary_filter_parameters": secondary_params,
                    "keywords_for_analysis": keywords,
                }

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing response: {e}")
                print(f"Response content: {response.content}")

                # Simple fallback - use best restaurants + location if we can extract it
                location = "Unknown"
                for indicator in ["in ", "near ", "at "]:
                    if indicator in query.lower():
                        parts = query.lower().split(indicator)
                        if len(parts) > 1:
                            possible_location = parts[1].split()[0]
                            if len(possible_location) > 2:
                                location = possible_location
                                break

                # Create a basic search query
                search_query = "best restaurants" + (f" in {location}" if location != "Unknown" else "")

                # Try to at least determine if this is a non-English speaking destination
                is_english_speaking = True
                local_language = None

                if location != "Unknown":
                    # Attempt to identify if this is a non-English speaking location
                    language_prompt = f"""
                    Is {location} in a primarily English-speaking country? Answer with only 'yes' or 'no'.
                    If 'no', what is the primary local language? Just provide the language name.
                    """

                    try:
                        language_chain = ChatPromptTemplate.from_template(language_prompt) | self.model
                        language_response = language_chain.invoke({})

                        response_text = language_response.content.lower()

                        if 'no' in response_text:
                            is_english_speaking = False

                            # Try to extract the language name
                            if '\n' in response_text:
                                language_line = response_text.split('\n')[1].strip()
                                local_language = language_line
                    except Exception as lang_error:
                        print(f"Error determining language: {lang_error}")

                # Add standing preferences to secondary parameters
                primary_params = ["best restaurants", f"in {location}"] if location != "Unknown" else ["best restaurants"]
                secondary_params = []

                # Process standing preferences
                for pref in standing_prefs:
                    skip = False
                    for pat in NEGATION_PATTERNS:
                        if re.search(pat.format(p=re.escape(pref)), original_query_lower):
                            skip = True
                            break
                    if skip:
                        continue
                    if pref not in secondary_params and pref not in primary_params:
                        secondary_params.append(pref)

                return {
                    "destination": location,
                    "is_english_speaking": is_english_speaking,
                    "local_language": local_language,
                    "search_queries": [search_query],
                    "primary_search_parameters": primary_params,
                    "secondary_filter_parameters": secondary_params,
                    "keywords_for_analysis": query.split(),
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