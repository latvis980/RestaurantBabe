from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
import json

class EditorAgent:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3
        )

        # Get the prompt from prompt_templates
        from prompts.prompt_templates import EDITOR_PROMPT
        self.prompt_template = EDITOR_PROMPT

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template),
            ("human", "Here are the restaurant recommendations to format:\n\n{recommendations}")
        ])

        # Create chain
        self.chain = self.prompt | self.model

        self.config = config

    def edit(self, recommendations, original_query):
        """
        Format and polish the restaurant recommendations

        Args:
            recommendations (dict): The restaurant recommendations from the list analyzer
            original_query (str): The original user query

        Returns:
            dict: The formatted recommendations with all required information
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                # Generate follow-up search queries
                follow_up_queries = self._generate_follow_up_queries(recommendations, original_query)

                # Invoke the formatting chain
                response = self.chain.invoke({
                    "recommendations": json.dumps(recommendations, ensure_ascii=False, indent=2),
                    "original_query": original_query
                })

                try:
                    # Parse the JSON response
                    formatted_results = json.loads(response.content)

                    # Add the follow-up queries
                    formatted_results["follow_up_queries"] = follow_up_queries

                    return formatted_results
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error parsing editor response: {e}")
                    print(f"Response content: {response.content}")

                    # Return the original recommendations if parsing fails
                    return {
                        "formatted_recommendations": recommendations,
                        "follow_up_queries": follow_up_queries
                    }
            except Exception as e:
                print(f"Error in editor agent: {e}")
                return {
                    "formatted_recommendations": recommendations,
                    "follow_up_queries": []
                }

    def _generate_follow_up_queries(self, recommendations, original_query):
        """Generate follow-up search queries for each restaurant"""
        follow_up_queries = []

        # Create a prompt for generating follow-up queries
        follow_up_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert at creating targeted search queries for restaurants.
            For each restaurant, create search queries to find:
            1. Missing information (address, hours, price range, etc.)
            2. Specific information about menu items mentioned in the original query
            3. Mentions in global restaurant guides

            Return a JSON array of objects with "restaurant_name" and "queries" (array of strings).
            """),
            ("human", f"""
            Original user query: {original_query}

            Restaurant recommendations: {json.dumps(recommendations, ensure_ascii=False)}

            Create follow-up search queries for each restaurant.
            """)
        ])

        # Create a follow-up chain
        follow_up_chain = follow_up_prompt | self.model

        # Invoke the chain
        response = follow_up_chain.invoke({})

        try:
            # Parse the JSON response
            queries = json.loads(response.content)
            return queries
        except (json.JSONDecodeError, AttributeError):
            # Generate basic queries if parsing fails
            basic_queries = []

            # Process recommended restaurants
            for restaurant in recommendations.get("recommended", []):
                name = restaurant.get("name", "")
                if name:
                    basic_queries.append({
                        "restaurant_name": name,
                        "queries": [
                            f"{name} restaurant address hours",
                            f"{name} restaurant menu specialties",
                            f"{name} restaurant michelin guide"
                        ]
                    })

            # Process hidden gems
            for restaurant in recommendations.get("hidden_gems", []):
                name = restaurant.get("name", "")
                if name:
                    basic_queries.append({
                        "restaurant_name": name,
                        "queries": [
                            f"{name} restaurant address hours",
                            f"{name} restaurant menu specialties",
                            f"{name} restaurant reviews"
                        ]
                    })

            return basic_queries