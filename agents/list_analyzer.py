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

        # Get the prompt from prompt_templates
        from prompts.prompt_templates import LIST_ANALYZER_PROMPT
        self.prompt_template = LIST_ANALYZER_PROMPT

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template),
            ("human", "Please analyze these search results and extract restaurant recommendations:\n\n{search_results}")
        ])

        # Create chain
        self.chain = self.prompt | self.model

        self.config = config

    def analyze(self, search_results, user_preferences, keywords_for_analysis):
        """
        Analyze search results to extract and rank restaurant recommendations

        Args:
            search_results (list): List of search results with scraped content
            user_preferences (str): User's preferences from the query analysis
            keywords_for_analysis (list): Keywords for analysis from the query analysis

        Returns:
            dict: Structured recommendations with top restaurants and hidden gems
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            # Format the search results for the prompt
            formatted_results = self._format_search_results(search_results)

            # Format the keywords for the prompt
            keywords_str = ", ".join(keywords_for_analysis) if keywords_for_analysis else ""

            # Invoke the chain
            response = self.chain.invoke({
                "search_results": formatted_results,
                "user_preferences": user_preferences,
                "keywords_for_analysis": keywords_str
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