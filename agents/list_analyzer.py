from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
import json

class ListAnalyzer:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2
        )

        # Create updated system prompt
        self.system_prompt = """
        You are a restaurant recommendation expert analyzing search results to identify the best restaurants.

        TASK:
        Analyze the search results and identify the most promising restaurants that match the user's preferences.

        USER PREFERENCES:
        {user_preferences}

        PRIMARY SEARCH PARAMETERS:
        {primary_parameters}

        SECONDARY FILTER PARAMETERS:
        {secondary_parameters}

        KEYWORDS FOR ANALYSIS:
        {keywords_for_analysis}

        GUIDELINES:
        1. Analyze the tone and content of reviews to identify genuinely recommended restaurants
        2. Cross-reference the descriptions against the keywords and user preferences
        3. Look for restaurants mentioned in multiple reputable sources
        4. IGNORE results from Tripadvisor, Yelp
        5. Pay special attention to restaurants featured in food guides, local publications, or by respected critics
        6. When analyzing content, check if restaurants meet the secondary filter parameters

        CREATE TWO LISTS:
        1. Top Recommended Restaurants (maximum 5):
           - These should be well-established, highly praised restaurants that match the user's preferences
           - They should appear in multiple sources

        2. Hidden Gems (1-2 restaurants):
           - Less frequently mentioned but highly praised restaurants
           - Must still match the user's preferences
           - Look for passionate, detailed reviews from respected sources

        FOR EACH RESTAURANT, EXTRACT:
        - Name
        - Location/Address (if available)
        - Brief description of cuisine and atmosphere
        - What makes it special or unique
        - Price indication (if available)
        - Sources where it was mentioned
        - Note which secondary filter parameters it appears to satisfy based on the content

        OUTPUT FORMAT:
        Provide a structured JSON object with two arrays: "recommended" and "hidden_gems"
        Each restaurant object should include all information you can find, including:
        - name
        - address
        - description
        - special_features (what makes it unique)
        - recommended_dishes (if mentioned)
        - sources (array of source domains where it was mentioned)
        - price_indication (if available)
        - matches_secondary_filters (array of secondary parameters it matches)
        - missing_info (array of information that's missing and needs follow-up)
        """

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Please analyze these search results and extract restaurant recommendations:\n\n{search_results}")
        ])

        # Create chain
        self.chain = self.prompt | self.model

        self.config = config

    def analyze(self, search_results, user_preferences, keywords_for_analysis, primary_parameters=None, secondary_parameters=None):
        """
        Analyze search results to extract and rank restaurant recommendations

        Args:
            search_results (list): List of search results with scraped content
            user_preferences (str): User's preferences from the query analysis
            keywords_for_analysis (list): Keywords for analysis from the query analysis
            primary_parameters (list, optional): Primary search parameters
            secondary_parameters (list, optional): Secondary filter parameters

        Returns:
            dict: Structured recommendations with top restaurants and hidden gems
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
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
                "user_preferences": user_preferences,
                "keywords_for_analysis": keywords_str,
                "primary_parameters": primary_params_str,
                "secondary_parameters": secondary_params_str
            })

            try:
                # Parse the JSON response
                results = json.loads(response.content)
                return results
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing ListAnalyzer response: {e}")
                print(f"Response content: {response.content}")

                # Fallback: Return a basic structure
                return {
                    "recommended": [],
                    "hidden_gems": []
                }

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