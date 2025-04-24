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

        # Define an editor prompt that includes the "recommended by" field and HTML formatting
        self.system_prompt = """
        You are a professional editor for a food publication specializing in restaurant recommendations. 
        Your task is to format and polish restaurant recommendations.

        For each restaurant, include these fields:
        1. Name (always bold with HTML <b> tags)
        2. Street address: street number and street name (preceded by 📍 emoji)
        3. Description (300-400 characters) that highlights what makes the restaurant special
        4. Recommended by: List at least 2 reputable sources that mentioned this restaurant (NEVER include Tripadvisor, Yelp, or Google)
           (formatted in italics with HTML <i> tags and preceded by ✅ emoji)

        FORMATTING INSTRUCTIONS:
        - Apply HTML formatting for Telegram: use <b>text</b> for bold and <i>text</i> for italics
        - Organize into two sections: "Main List" and "Hidden Gems"
        - Make descriptions engaging but factual
        - Focus on culinary experience and atmosphere
        - Avoid generic praise or marketing language
        - Be specific about what makes each place special
        - ALWAYS include the "recommended by" field with at least 2 sources

        OUTPUT FORMAT:
        Provide a JSON object with:
        - "formatted_recommendations": Object with "main_list" and "hidden_gems" arrays
        - Each restaurant should have:
          - "name": Restaurant name (without HTML tags in the JSON)
          - "address": Street address (without emoji in the JSON)
          - "description": Polished description (300-400 characters)
          - "recommended_by": Array of sources that recommend this restaurant
          - "html_formatted": Complete HTML-formatted text for this restaurant entry
        """

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Here are the restaurant recommendations to format:\n\n{recommendations}\n\nOriginal query: {original_query}")
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
            dict: The formatted recommendations with basic information
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                # Debug information
                print(f"Editor received recommendations structure: {list(recommendations.keys())}")

                # Make sure recommendations has the required structure
                if not recommendations or not isinstance(recommendations, dict):
                    recommendations = {"main_list": [], "hidden_gems": []}

                # Ensure required keys exist (handle both old "recommended" and new "main_list")
                if "main_list" not in recommendations and "recommended" in recommendations:
                    # Handle old structure
                    recommendations["main_list"] = recommendations["recommended"]
                    del recommendations["recommended"]
                elif "main_list" not in recommendations:
                    recommendations["main_list"] = []

                if "hidden_gems" not in recommendations:
                    recommendations["hidden_gems"] = []

                # Print debug info
                print(f"Editor processing: main_list({len(recommendations['main_list'])} items), hidden_gems({len(recommendations['hidden_gems'])} items)")

                # Make sure each restaurant has a sources field
                for restaurant in recommendations["main_list"]:
                    if "sources" not in restaurant:
                        restaurant["sources"] = ["Unknown Source", "Food Expert"]
                    # Ensure city is recorded for follow-up queries
                    if "city" not in restaurant and "location" in restaurant:
                        restaurant["city"] = restaurant["location"]

                for restaurant in recommendations["hidden_gems"]:
                    if "sources" not in restaurant:
                        restaurant["sources"] = ["Local Food Blog", "Culinary Magazine"]
                    # Ensure city is recorded for follow-up queries
                    if "city" not in restaurant and "location" in restaurant:
                        restaurant["city"] = restaurant["location"]

                # Extract city from original query if needed
                query_city = self._extract_city_from_query(original_query)

                # Invoke the formatting chain
                response = self.chain.invoke({
                    "recommendations": json.dumps(recommendations, ensure_ascii=False, indent=2),
                    "original_query": original_query
                })

                try:
                    # Parse the JSON response
                    formatted_results = json.loads(response.content)

                    # Generate follow-up queries for each restaurant including city
                    follow_up_queries = self._generate_follow_up_queries(formatted_results, query_city)

                    # Add the follow-up queries to the result
                    formatted_results["follow_up_queries"] = follow_up_queries

                    # Create HTML formatted version for telegram
                    formatted_results["html_formatted"] = self._create_html_output(formatted_results)

                    return formatted_results
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error parsing editor response: {e}")
                    print(f"Response content: {response.content}")

                    # Return the original recommendations if parsing fails
                    return {
                        "formatted_recommendations": recommendations,
                        "follow_up_queries": self._generate_follow_up_queries(recommendations, query_city),
                        "html_formatted": self._create_fallback_html(recommendations)
                    }
            except Exception as e:
                print(f"Error in editor agent: {e}")
                return {
                    "formatted_recommendations": recommendations,
                    "follow_up_queries": [],
                    "html_formatted": self._create_fallback_html(recommendations)
                }

    def _extract_city_from_query(self, query):
        """Extract city name from the original query"""
        city = ""

        # Check for common patterns that indicate a city
        if "in " in query.lower():
            parts = query.lower().split("in ")
            if len(parts) > 1:
                # Take the word after "in"
                city_part = parts[1].split()
                if city_part:
                    city = city_part[0].strip().capitalize()

                    # If followed by another word that could be part of city name, include it
                    if len(city_part) > 1 and not any(word in city_part[1].lower() for word in 
                                                    ["with", "that", "for", "and", "or", "where", "which"]):
                        city += " " + city_part[1].strip().capitalize()

        return city

    def _generate_follow_up_queries(self, recommendations, query_city=""):
        """Generate follow-up search queries for each restaurant including city"""
        follow_up_queries = []

        # Process restaurants from formatted_recommendations if that key exists
        if "formatted_recommendations" in recommendations:
            recommendations = recommendations["formatted_recommendations"]

        # Process main list restaurants
        for restaurant in recommendations.get("main_list", []):
            name = restaurant.get("name", "")
            # Try to get city either from restaurant object or query
            city = restaurant.get("city", "")
            if not city:
                city = query_city

            location_part = f" {city}" if city else ""

            if name:
                follow_up_queries.append({
                    "restaurant_name": name,
                    "queries": [
                        f"{name} restaurant{location_part} address hours",
                        f"{name} restaurant{location_part} menu",
                        f"{name} restaurant{location_part} chef",
                        # Add source search queries
                        f"{name} restaurant{location_part} michelin guide",
                        f"{name} restaurant{location_part} michelin",
                        f"{name} restaurant{location_part} 50best",
                        f"{name} restaurant{location_part} word of mouth"
                    ]
                })

        # Process hidden gems
        for restaurant in recommendations.get("hidden_gems", []):
            name = restaurant.get("name", "")
            # Try to get city either from restaurant object or query
            city = restaurant.get("city", "")
            if not city:
                city = query_city

            location_part = f" {city}" if city else ""

            if name:
                follow_up_queries.append({
                    "restaurant_name": name,
                    "queries": [
                        f"{name} restaurant{location_part} address hours",
                        f"{name} restaurant{location_part} menu",
                        # Add source search queries
                        f"{name} restaurant{location_part} michelin",
                        f"{name} restaurant{location_part} word of mouth",
                        f"{name} restaurant{location_part} chef interview"
                    ]
                })

        return follow_up_queries

    def _create_html_output(self, formatted_results):
        """Create HTML formatted output for Telegram from the editor results"""
        try:
            html_output = "<b>🍽️ RECOMMENDED RESTAURANTS:</b>\n\n"

            # Check if we need to extract from formatted_recommendations
            if "formatted_recommendations" in formatted_results:
                recommendations = formatted_results["formatted_recommendations"]
            else:
                recommendations = formatted_results

            # Format main list
            main_list = recommendations.get("main_list", [])
            if main_list:
                for i, restaurant in enumerate(main_list, 1):
                    # If html_formatted exists, use it directly
                    if "html_formatted" in restaurant:
                        html_output += f"{i}. {restaurant['html_formatted']}\n\n"
                    else:
                        # Otherwise build the formatted text
                        name = restaurant.get("name", "Restaurant")
                        html_output += f"<b>{i}. {name}</b>\n"

                        if "address" in restaurant:
                            html_output += f"📍 {restaurant['address']}\n"

                        if "description" in restaurant:
                            html_output += f"{restaurant['description']}\n"

                        if "recommended_by" in restaurant and restaurant["recommended_by"]:
                            sources = restaurant["recommended_by"]
                            if isinstance(sources, list):
                                sources_text = ", ".join(sources[:3])
                                html_output += f"<i>✅ Recommended by: {sources_text}</i>\n"
                            else:
                                html_output += f"<i>✅ Recommended by: {sources}</i>\n"

                        html_output += "\n"
            else:
                html_output += "Sorry, no recommended restaurants found.\n\n"

            # Format hidden gems
            hidden_gems = recommendations.get("hidden_gems", [])
            if hidden_gems:
                html_output += "<b>💎 HIDDEN GEMS:</b>\n\n"

                for i, restaurant in enumerate(hidden_gems, 1):
                    # If html_formatted exists, use it directly
                    if "html_formatted" in restaurant:
                        html_output += f"{i}. {restaurant['html_formatted']}\n\n"
                    else:
                        # Otherwise build the formatted text
                        name = restaurant.get("name", "Restaurant")
                        html_output += f"<b>{i}. {name}</b>\n"

                        if "address" in restaurant:
                            html_output += f"📍 {restaurant['address']}\n"

                        if "description" in restaurant:
                            html_output += f"{restaurant['description']}\n"

                        if "recommended_by" in restaurant and restaurant["recommended_by"]:
                            sources = restaurant["recommended_by"]
                            if isinstance(sources, list):
                                sources_text = ", ".join(sources[:3])
                                html_output += f"<i>✅ Recommended by: {sources_text}</i>\n"
                            else:
                                html_output += f"<i>✅ Recommended by: {sources}</i>\n"

                        html_output += "\n"

            # Add footer
            html_output += "<i>Recommendations based on analysis of expert sources.</i>"

            return html_output
        except Exception as e:
            print(f"Error creating HTML output: {e}")
            return self._create_fallback_html(formatted_results)

    def _create_fallback_html(self, recommendations):
        """Create fallback HTML if the main formatter fails"""
        try:
            html_output = "<b>🍽️ RECOMMENDED RESTAURANTS:</b>\n\n"

            # Handle different possible structures
            if "main_list" in recommendations:
                main_list = recommendations["main_list"]
            elif "recommended" in recommendations:
                main_list = recommendations["recommended"]
            else:
                main_list = []

            if "hidden_gems" in recommendations:
                hidden_gems = recommendations["hidden_gems"]
            else:
                hidden_gems = []

            # Format main list
            if main_list:
                for i, restaurant in enumerate(main_list, 1):
                    name = restaurant.get("name", "Restaurant")
                    html_output += f"<b>{i}. {name}</b>\n"

                    if "address" in restaurant:
                        html_output += f"📍 {restaurant['address']}\n"

                    if "description" in restaurant:
                        html_output += f"{restaurant['description']}\n"

                    if "sources" in restaurant and restaurant["sources"]:
                        sources = restaurant["sources"]
                        if isinstance(sources, list):
                            sources_text = ", ".join(sources[:3])
                            html_output += f"<i>✅ Recommended by: {sources_text}</i>\n"
                        else:
                            html_output += f"<i>✅ Recommended by: {sources}</i>\n"

                    html_output += "\n"
            else:
                html_output += "Sorry, no recommended restaurants found.\n\n"

            # Format hidden gems
            if hidden_gems:
                html_output += "<b>💎 HIDDEN GEMS:</b>\n\n"

                for i, restaurant in enumerate(hidden_gems, 1):
                    name = restaurant.get("name", "Restaurant")
                    html_output += f"<b>{i}. {name}</b>\n"

                    if "address" in restaurant:
                        html_output += f"📍 {restaurant['address']}\n"

                    if "description" in restaurant:
                        html_output += f"{restaurant['description']}\n"

                    if "sources" in restaurant and restaurant["sources"]:
                        sources = restaurant["sources"]
                        if isinstance(sources, list):
                            sources_text = ", ".join(sources[:3])
                            html_output += f"<i>✅ Recommended by: {sources_text}</i>\n"
                        else:
                            html_output += f"<i>✅ Recommended by: {sources}</i>\n"

                    html_output += "\n"

            # Add footer
            html_output += "<i>Recommendations based on analysis of expert sources.</i>"

            return html_output
        except Exception as e:
            print(f"Error creating fallback HTML: {e}")
            return "<b>Sorry, couldn't format restaurant recommendations properly.</b>"