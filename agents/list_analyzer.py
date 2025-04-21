# agents/list_analyzer.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_mistralai import ChatMistralAI
import json
import time
from utils.database import save_data

class ListAnalyzer:
    def __init__(self, config):
        # Initialize Mistral model instead of OpenAI
        self.model = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.2,
            mistral_api_key=config.MISTRAL_API_KEY  # You'll need to add this to your config
        )

        # Create updated system prompt
        self.system_prompt = """
        You are a restaurant recommendation expert analyzing search results to identify the best restaurants.

        TASK:
        Analyze the search results and identify promising restaurants that match the search parameters.

        PRIMARY SEARCH PARAMETERS:
        {primary_parameters}

        SECONDARY FILTER PARAMETERS:
        {secondary_parameters}

        KEYWORDS FOR ANALYSIS:
        {keywords_for_analysis}

        GUIDELINES:
        1. Analyze the tone and content of reviews to identify genuinely recommended restaurants
        2. Cross-reference the descriptions against the keywords and search parameters
        3. Look for restaurants mentioned in multiple reputable sources
        4. IGNORE results from Tripadvisor, Yelp
        5. Pay special attention to restaurants featured in food guides, local publications, or by respected critics
        6. When analyzing content, check if restaurants meet the secondary filter parameters

        OUTPUT REQUIREMENTS:
        - Identify at least 10 restaurants (up to 15 for broad searches like "traditional restaurants")
        - Do not separate into "recommended" and "hidden gems" categories
        - For EACH restaurant, extract:
          1. Name (exact as mentioned in sources)
          2. Street address (as complete as possible)
          3. Raw description (40-60 words) including key details, dishes, interior, chef, and atmosphere
          4. ALL sources where mentioned (just the source name, NOT the URL, e.g., "Le Foodling" not "lefooding.com")

        OUTPUT FORMAT:
        Provide a structured JSON object with one array: "restaurants"
        Each restaurant object should include:
        - name
        - address
        - description (40-60 words summary)
        - sources (array of source names where it was mentioned)
        - location (city name from the search)
        """

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Please analyze these search results and extract restaurant recommendations:\n\n{search_results}")
        ])

        # Create chain
        self.chain = self.prompt | self.model

        self.config = config

    def analyze(self, search_results, keywords_for_analysis, primary_parameters=None, secondary_parameters=None):
        """
        Analyze search results to extract and rank restaurant recommendations

        Args:
            search_results (list): List of search results with scraped content
            keywords_for_analysis (list): Keywords for analysis from the query analysis
            primary_parameters (list, optional): Primary search parameters
            secondary_parameters (list, optional): Secondary filter parameters

        Returns:
            dict: Structured recommendations with restaurants
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Extract location/city from the analysis
            city = self._extract_city(primary_parameters)

            # Format the search results for the prompt
            formatted_results = self._format_search_results(search_results)

            # Format the keywords for the prompt
            # Convert list to string if needed
            if isinstance(keywords_for_analysis, list):
                keywords_str = ", ".join(keywords_for_analysis)
            else:
                keywords_str = keywords_for_analysis if keywords_for_analysis else ""

            # Format primary parameters
            if isinstance(primary_parameters, list):
                primary_params_str = ", ".join(primary_parameters)
            else:
                primary_params_str = primary_parameters if primary_parameters else ""

            # Format secondary parameters
            if isinstance(secondary_parameters, list):
                secondary_params_str = ", ".join(secondary_parameters)
            else:
                secondary_params_str = secondary_parameters if secondary_parameters else ""

            # Invoke the chain
            response = self.chain.invoke({
                "search_results": formatted_results,
                "keywords_for_analysis": keywords_str,
                "primary_parameters": primary_params_str,
                "secondary_parameters": secondary_params_str
            })

            try:
                # Parse the JSON response
                results = json.loads(response.content)

                # Add the city to each restaurant
                for restaurant in results.get("restaurants", []):
                    restaurant["city"] = city

                # Save restaurant data to database
                self._save_restaurants_to_db(results.get("restaurants", []), city)

                # Format for backwards compatibility with existing code
                # This creates the structure that the editor_agent and other components expect
                compatible_results = {
                    "recommended": results.get("restaurants", [])[:5],  # Top 5 as recommended
                    "hidden_gems": results.get("restaurants", [])[5:]   # Rest as hidden gems
                }

                return compatible_results
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing ListAnalyzer response: {e}")
                print(f"Response content: {response.content}")

                # Fallback: Return a basic structure
                return {
                    "recommended": [],
                    "hidden_gems": []
                }

    def _extract_city(self, primary_parameters):
        """Extract the city name from the search parameters"""
        # Look for city names in primary parameters first
        if isinstance(primary_parameters, list):
            for param in primary_parameters:
                if "in " in param.lower():
                    city = param.lower().split("in ")[1].strip()
                    return city

        # Default fallback
        return "unknown_location"

    def _format_search_results(self, search_results):
        """Format search results for the prompt"""
        formatted_results = []

        for i, result in enumerate(search_results):
            result_str = f"RESULT {i+1}:\n"
            result_str += f"Title: {result.get('title', 'Unknown')}\n"
            result_str += f"URL: {result.get('url', 'Unknown')}\n"
            result_str += f"Source: {result.get('source_domain', 'Unknown')}\n"

            # Add description
            description = result.get('description', '')
            if description:
                result_str += f"Description: {description}\n"

            # Add scraped content (truncated to avoid very long prompts)
            scraped_content = result.get('scraped_content', '')
            if scraped_content:
                # Truncate to approximately 2000 characters
                if len(scraped_content) > 2000:
                    scraped_content = scraped_content[:2000] + "..."
                result_str += f"Content: {scraped_content}\n"

            # Add restaurant info if available
            restaurant_info = result.get('restaurant_info', {})
            if restaurant_info:
                result_str += "Restaurant Info:\n"
                for key, value in restaurant_info.items():
                    if key != 'source_domain':  # Already included above
                        result_str += f"- {key}: {value}\n"

            formatted_results.append(result_str)

        return "\n\n".join(formatted_results)

    def _save_restaurants_to_db(self, restaurants, city):
        """Save restaurant data to city-specific database table"""
        try:
            # Create a table name based on the city (lowercase, no spaces)
            table_name = f"restaurants_{city.lower().replace(' ', '_')}"

            # Save each restaurant to the database
            for restaurant in restaurants:
                # Add timestamp for database
                restaurant["timestamp"] = time.time()

                # Generate a unique ID based on name and address
                restaurant_id = f"{restaurant['name']}_{restaurant['address']}".lower().replace(' ', '_')
                restaurant["id"] = restaurant_id

                # Save to database
                save_data(
                    table_name,
                    restaurant,
                    self.config
                )

            print(f"Saved {len(restaurants)} restaurants to table {table_name}")
        except Exception as e:
            print(f"Error saving restaurants to database: {e}")