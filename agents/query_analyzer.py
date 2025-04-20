from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tracers.context import tracing_v2_enabled
import json
from utils.database import save_data, find_data


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
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])

        self.chain = self.prompt | self.model
        self.config = config

    def analyze(self, query):
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            response = self.chain.invoke({"query": query})

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

                # Parse the JSON
                result = json.loads(content)

                location = result.get("destination")
                is_english_speaking = result.get("is_english_speaking", True)

                if location and not is_english_speaking:
                    local_sources = find_in_mongodb(
                        self.config.DB_TABLE_SOURCES,
                        {"location": location},
                        self.config
                    )

                    if not local_sources:
                        local_sources = self._compile_local_sources(location, result.get("local_language"))
                        save_to_mongodb(
                            self.config.DB_TABLE_SOURCES,
                            {"location": location, "sources": local_sources},
                            self.config
                        )

                    result["local_sources"] = local_sources

                search_queries = [result.get("english_search_query")]
                if result.get("local_language_search_query"):
                    search_queries.append(result.get("local_language_search_query"))
                search_queries = [q for q in search_queries if q]

                keywords = result.get("keywords_for_analysis", [])
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in keywords.split(",") if k.strip()]

                primary_params = result.get("primary_search_parameters", [])
                secondary_params = result.get("secondary_filter_parameters", [])

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
        local_sources_prompt = f"""
        Identify 5-7 reputable local sources for restaurant reviews and food recommendations in {location}.
        Focus on local press, respected food experts, bloggers, and local food guides that publish in {language}.
        Do NOT include international sites like TripAdvisor, Yelp, or Google. Only include sources that locals would trust.

        Return a JSON array with objects containing \"name\" and \"url\" (if available).
        """

        local_sources_chain = ChatPromptTemplate.from_template(local_sources_prompt) | self.model
        response = local_sources_chain.invoke({})

        try:
            sources = json.loads(response.content)
            return sources
        except (json.JSONDecodeError, AttributeError):
            return []
