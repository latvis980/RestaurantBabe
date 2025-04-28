# agents/list_analyzer.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_mistralai import ChatMistralAI
import json
import time
from utils.database import save_data
from utils.debug_utils import dump_chain_state, log_function_call

class ListAnalyzer:
    def __init__(self, config):
        # Initialize Mistral model instead of OpenAI
        self.model = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.2,
            mistral_api_key=config.MISTRAL_API_KEY
        )

        # Create updated system prompt - modified to ensure we get restaurant results
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
        7. IMPORTANT: If you can't find perfect matches, still provide at least 3-5 restaurants that are the closest matches

        OUTPUT REQUIREMENTS:
        - ALWAYS identify at least 5 restaurants (even with limited information)
        - Do not separate restaurants into different categories, just provide one main list
        - If search results are limited, create entries based on the available information
        - For EACH restaurant, extract:
          1. Name (exact as mentioned in sources)
          2. Street address (as complete as possible, or "Address unavailable" if not found)
          3. Raw description (40-60 words) including key details, dishes, interior, chef, and atmosphere
          4. ALL sources where mentioned (just the source name, NOT the URL, e.g., "Le Foodling" not "lefooding.com")
          5. Pay special attention to restaurants featured in food guides, local publications, or by respected critics
          6. For each restaurant, collect ALL source names found in the search results (look for "Source Name:" in each result)
          7. When analyzing content, check if restaurants meet the secondary filter parameters

        OUTPUT FORMAT:
        Provide a structured JSON object with one array: "main_list"
        Each restaurant object should include:
        - name (required, never empty)
        - address (required, use "Address unavailable" if not found)
        - description (required, 40-60 words summary, use available information to create one if needed)
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

    @log_function_call
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
            # Debug logging
            dump_chain_state("analyze_start", {
                "search_results_count": len(search_results),
                "keywords": keywords_for_analysis,
                "primary_parameters": primary_parameters,
                "secondary_parameters": secondary_parameters
            })

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
                content = response.content

                # Handle different response formats
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                # Debug log the raw response
                dump_chain_state("analyze_raw_response", {
                    "raw_response": content[:1000]  # Limit to 1000 chars
                })

                # Parse the JSON
                results = json.loads(content)

                # If the output still has "restaurants" key, convert it to "main_list"
                if "restaurants" in results and "main_list" not in results:
                    results["main_list"] = results.pop("restaurants")

                # Ensure we have a main_list key
                if "main_list" not in results:
                    results["main_list"] = []

                # Add the city to each restaurant
                for restaurant in results.get("main_list", []):
                    restaurant["city"] = city

                # Save restaurant data to database
                self._save_restaurants_to_db(results.get("main_list", []), city)

                # Handle the case when we get an empty result
                if not results.get("main_list"):
                    dump_chain_state("analyze_no_results", {
                        "warning": "No restaurants found in response"
                    })
                    # Create a fallback result with a placeholder restaurant
                    results["main_list"] = [{
                        "name": "Поиск не дал результатов",
                        "address": "Адрес недоступен",
                        "description": "К сожалению, мы не смогли найти рестораны, соответствующие вашему запросу. Пожалуйста, попробуйте изменить параметры поиска.",
                        "sources": ["Системное сообщение"],
                        "city": city
                    }]

                # Debug log the final structured results
                dump_chain_state("analyze_final_results", {
                    "main_list_count": len(results["main_list"])
                })

                return results

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing ListAnalyzer response: {e}")
                print(f"Response content: {response.content}")

                # Debug log the error
                dump_chain_state("analyze_json_error", {
                    "error": str(e),
                    "response_preview": response.content[:500] if hasattr(response, 'content') else "No content"
                })

                # Fallback: Return a basic structure with an error message restaurant
                return {
                    "main_list": [{
                        "name": "Ошибка обработки результатов",
                        "address": "Адрес недоступен",
                        "description": "Произошла ошибка при обработке результатов поиска. Пожалуйста, попробуйте позже или измените параметры поиска.",
                        "sources": ["Системное сообщение"],
                        "city": city
                    }]
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
            if result.get('source_name'):
                result_str += f"Source Name: {result.get('source_name')}\n"

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