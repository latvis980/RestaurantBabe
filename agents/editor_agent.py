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

        # Editor prompt - updated to generate hidden_gems from main_list
        self.system_prompt = """
        You are a professional editor for a food publication specializing in restaurant recommendations. 
        Your task is to format and polish restaurant recommendations according to strict formatting guidelines.

        INFORMATION REQUIREMENTS:
        Obligatory information for each restaurant:
        - Name (always bold)
        - Street address: street number and street name
        - Informative description 2-40 words
        - Price range
        - Recommended dishes (at least 2-3 signature items)
        - At least two sources of recommendation (e.g., "Recommended by Michelin Guide and Timeout Lisboa")
        - NEVER mention Tripadvisor, Yelp, or Google as sources

        Optional information (include when available):
        - If reservations are highly recommended, clearly state this
        - Instagram handle in format "instagram.com/username"
        - Chef name or background
        - Opening hours
        - Special atmosphere details

        HIDDEN GEMS SELECTION:
        - Select 1-2 lesser-known but interesting places from the main list to feature as "hidden gems"
        - Look for restaurants with unique concepts, local significance, or interesting specialties
        - Move these selected restaurants to the hidden_gems list and remove them from the main list

        FORMATTING INSTRUCTIONS:
        1. Organize into two sections: "Recommended Restaurants" (from main_list) and "Hidden Gems" (selected by you)
        2. For each restaurant, create a structured listing with all required information
        3. Make restaurant names bold
        4. Use consistent formatting across all listings
        5. Ensure descriptions are concise but informative
        6. Verify all information is complete according to requirements
        7. If any required information is missing, note what follow-up information is needed

        TONE GUIDELINES:
        - Professional but engaging
        - Highlight what makes each restaurant special
        - Focus on culinary experience and atmosphere
        - Be specific about menu recommendations
        - Avoid generic praise or marketing language

        OUTPUT FORMAT:
        Provide a structured JSON object with:
        - "formatted_recommendations": Object with "main_list" and "hidden_gems" arrays
        - Each restaurant in the arrays should have all the required fields:
          - "name": Restaurant name
          - "address": Complete street address
          - "description": Concise description
          - "price_range": Number of 🟡 symbols (1-3)
          - "recommended_dishes": Array of dishes
          - "sources": Array of recommendation sources
          - "reservations_required": Boolean (if known)
          - "instagram": Instagram handle (if available)
          - "chef": Chef information (if available)
          - "hours": Opening hours (if available)
          - "atmosphere": Atmosphere details (if available)
          - "missing_info": Array of missing information fields
        """

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Here are the restaurant recommendations to format:\n\n{recommendations}\n\nOriginal query: {original_query}")
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
                    recommendations = {"main_list": []}

                # Ensure main_list exists - handle legacy format if necessary
                if "main_list" not in recommendations:
                    if "recommended" in recommendations:
                        # Convert old format to new format
                        recommendations["main_list"] = recommendations.pop("recommended")
                    else:
                        # Initialize empty main_list
                        recommendations["main_list"] = []

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

                    # Ensure we have the correct structure
                    if "formatted_recommendations" in formatted_results:
                        formatted_rec = formatted_results["formatted_recommendations"]
                        # Convert any "recommended" to "main_list" if present
                        if isinstance(formatted_rec, dict):
                            if "recommended" in formatted_rec and "main_list" not in formatted_rec:
                                formatted_rec["main_list"] = formatted_rec.pop("recommended")
                    else:
                        # If no formatted_recommendations key, wrap the result
                        formatted_results = {
                            "formatted_recommendations": formatted_results
                        }
                        # Check if we need to convert recommendations to main_list in the wrapped result
                        if "recommended" in formatted_results["formatted_recommendations"] and "main_list" not in formatted_results["formatted_recommendations"]:
                            formatted_results["formatted_recommendations"]["main_list"] = formatted_results["formatted_recommendations"].pop("recommended")

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
                        "formatted_recommendations": {
                            "main_list": recommendations.get("main_list", []),
                            "hidden_gems": []
                        },
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
                    "formatted_recommendations": {
                        "main_list": recommendations.get("main_list", []) if isinstance(recommendations, dict) else [],
                        "hidden_gems": []
                    },
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

        # Get restaurants from main_list
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

        return basic_queries