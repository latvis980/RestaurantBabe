from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
import json
from utils.debug_utils import dump_chain_state, log_function_call

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

    @log_function_call
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
                # Debug log initial input
                dump_chain_state("editor_input", {
                    "recommendations_keys": list(recommendations.keys() if isinstance(recommendations, dict) else []),
                    "original_query": original_query
                })

                # Generate follow-up search queries
                follow_up_queries = self._generate_follow_up_queries(recommendations, original_query)

                # Make sure recommendations has the required structure
                if not recommendations or not isinstance(recommendations, dict):
                    recommendations = {"main_list": [], "hidden_gems": []}

                # Ensure required keys exist - check for both main_list and recommended (for backward compatibility)
                if "main_list" not in recommendations and "recommended" not in recommendations:
                    recommendations["main_list"] = []
                elif "recommended" in recommendations and "main_list" not in recommendations:
                    # Convert old format to new format
                    recommendations["main_list"] = recommendations.pop("recommended")

                if "hidden_gems" not in recommendations:
                    recommendations["hidden_gems"] = []

                # Invoke the formatting chain
                response = self.chain.invoke({
                    "recommendations": json.dumps(recommendations, ensure_ascii=False, indent=2),
                    "original_query": original_query
                })

                try:
                    # Parse the JSON response
                    content = response.content

                    # Handle different response formats
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()

                    # Debug log the raw response
                    dump_chain_state("editor_raw_response", {
                        "raw_response": content[:1000]  # Limit to 1000 chars
                    })

                    # Parse the JSON
                    formatted_results = json.loads(content)

                    # Convert old format to new format if needed
                    if "formatted_recommendations" in formatted_results:
                        recommendations_obj = formatted_results["formatted_recommendations"]
                        if isinstance(recommendations_obj, dict):
                            if "recommended" in recommendations_obj and "main_list" not in recommendations_obj:
                                recommendations_obj["main_list"] = recommendations_obj.pop("recommended")

                    # Add the follow-up queries
                    formatted_results["follow_up_queries"] = follow_up_queries

                    # Debug log the final structured results
                    dump_chain_state("editor_final_results", {
                        "formatted_results_keys": list(formatted_results.keys()),
                        "follow_up_queries_count": len(follow_up_queries)
                    })

                    return formatted_results

                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error parsing editor response: {e}")
                    print(f"Response content: {response.content}")

                    # Debug log the error
                    dump_chain_state("editor_json_error", {
                        "error": str(e),
                        "response_preview": response.content[:500] if hasattr(response, 'content') else "No content"
                    })

                    # Return the original recommendations if parsing fails
                    return {
                        "formatted_recommendations": recommendations,
                        "follow_up_queries": follow_up_queries
                    }
            except Exception as e:
                print(f"Error in editor agent: {e}")

                # Debug log the error
                dump_chain_state("editor_general_error", {
                    "error": str(e),
                    "recommendations_preview": str(recommendations)[:500] if recommendations else "No recommendations"
                }, error=e)

                return {
                    "formatted_recommendations": recommendations,
                    "follow_up_queries": []
                }

    def _generate_follow_up_queries(self, recommendations, original_query):
        """Generate follow-up search queries for each restaurant"""
        follow_up_queries = []

        try:
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
                content = response.content

                # Handle different response formats
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                # Parse the JSON
                queries = json.loads(content)
                return queries
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing follow-up queries: {e}")

                # Generate basic queries if parsing fails
                return self._generate_basic_queries(recommendations)

        except Exception as e:
            print(f"Error generating follow-up queries: {e}")
            return self._generate_basic_queries(recommendations)

    def _generate_basic_queries(self, recommendations):
        """Generate basic follow-up queries if the main generation fails"""
        basic_queries = []

        # Process main_list restaurants (check both new and old formats)
        main_list = []
        if isinstance(recommendations, dict):
            if "main_list" in recommendations:
                main_list = recommendations["main_list"]
            elif "recommended" in recommendations:
                main_list = recommendations["recommended"]

        for restaurant in main_list:
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
        hidden_gems = recommendations.get("hidden_gems", []) if isinstance(recommendations, dict) else []
        for restaurant in hidden_gems:
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