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

        # Get the prompt from prompt_templates
        from prompts.prompt_templates import QUERY_ANALYZER_PROMPT

        # Create prompt template - Fix the template to only expect 'query'
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_ANALYZER_PROMPT),
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
                        {"location": location}
                    )

                    # If no local sources found, compile and save them
                    if not local_sources:
                        local_sources = self._compile_local_sources(location, result.get("local_language"))
                        # Save to database for future use
                        save_to_mongodb(
                            self.config.MONGODB_COLLECTION_SOURCES,
                            {"location": location, "sources": local_sources}
                        )

                    result["local_sources"] = local_sources

                # Create search queries
                search_queries = [result.get("english_search_query")]

                # Add local language query if available
                if result.get("local_language_search_query"):
                    search_queries.append(result.get("local_language_search_query"))

                # Remove None or empty queries
                search_queries = [q for q in search_queries if q]

                return {
                    "destination": location,
                    "is_english_speaking": is_english_speaking,
                    "local_language": result.get("local_language"),
                    "search_queries": search_queries,
                    "keywords_for_analysis": result.get("keywords_for_analysis", []),
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
                    "keywords_for_analysis": [query.split()],
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