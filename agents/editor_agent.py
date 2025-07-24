# agents/editor_agent.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
import json
import logging
from collections import defaultdict
from utils.debug_utils import dump_chain_state, log_function_call

logger = logging.getLogger(__name__)

class EditorAgent:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2  # Lower temperature for more consistent formatting
        )

        # Simplified prompt focused on consolidation and JSON output
        self.system_prompt = """
You are an AI assistant that processes restaurant recommendations from web scraping results.

YOUR TASK:
1. Analyze all scraped content to identify unique restaurants
2. Group descriptions and addresses for restaurants mentioned multiple times
3. Create comprehensive 15-30 word descriptions combining information from all sources
4. Extract source domains (timeout.com, tastingtable.com, etc.) from URLs

CONSOLIDATION RULES:
- If a restaurant appears in multiple sources, combine all information
- Use the most complete address found across sources
- If addresses conflict or are missing, mark for verification
- Create detailed descriptions that highlight what makes each restaurant special
- Avoid generic phrases like "great food" or "nice atmosphere"

OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure:
{{
  "restaurants": [
    {{
      "name": "Restaurant Name",
      "address": "Complete address OR 'Requires verification'",
      "description": "Detailed 15-30 word description highlighting unique features",
      "sources": ["domain1.com", "domain2.com"]
    }}
  ]
}}

IMPORTANT:
- Never include Tripadvisor, Yelp, Opentable, or Google in sources
- Focus on unique selling points in descriptions (signature dishes, style, atmosphere)
- Be specific about cuisine types, specialties, or notable features
- If multiple locations exist, include only the main/original location
"""

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
Original query: {original_query}
Destination: {destination}

Scraped content from multiple sources:
{scraped_content}

Process this content and return the consolidated restaurant list as JSON.
""")
        ])

        # Create chain
        self.chain = self.prompt | self.model
        self.config = config

    def _prepare_scraped_content(self, search_results):
        """
        Convert search results into a clean format for the AI to process.
        Includes URL sources and content for each article.
        """
        formatted_content = []

        for i, result in enumerate(search_results, 1):
            # Extract domain from URL for source attribution
            url = result.get('url', '')
            domain = self._extract_domain(url)

            # Get content
            content = result.get('scraped_content', result.get('content', ''))
            title = result.get('title', 'Untitled')

            if content and len(content.strip()) > 50:  # Only include substantial content
                formatted_content.append(f"""
ARTICLE {i}:
URL: {url}
Domain: {domain}  
Title: {title}
Content: {content[:3000]}...  
---""")

        return "\n".join(formatted_content)

    def _extract_domain(self, url):
        """Extract clean domain name from URL for source attribution"""
        try:
            if not url:
                return "unknown.com"

            # Remove protocol
            if url.startswith(('http://', 'https://')):
                url = url.split('://', 1)[1]

            # Get domain part
            domain = url.split('/')[0]

            # Remove www.
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain

        except Exception:
            return "unknown.com"

    def _post_process_results(self, ai_output, original_query, destination):
        """
        Post-process AI output to ensure quality and consistency.
        """
        try:
            # Parse JSON response
            if isinstance(ai_output, str):
                # Clean markdown formatting if present
                content = ai_output.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                parsed_data = json.loads(content)
            else:
                parsed_data = ai_output

            restaurants = parsed_data.get("restaurants", [])

            # Validate and clean each restaurant entry
            cleaned_restaurants = []
            seen_names = set()

            for restaurant in restaurants:
                name = restaurant.get("name", "").strip()
                if not name or name.lower() in seen_names:
                    continue

                seen_names.add(name.lower())

                # Validate description length
                description = restaurant.get("description", "").strip()
                word_count = len(description.split())
                if word_count < 10 or word_count > 35:
                    logger.warning(f"Description for {name} has {word_count} words, outside 15-30 range")

                # Clean sources
                sources = restaurant.get("sources", [])
                clean_sources = []
                for source in sources:
                    source = source.lower().strip()
                    if not any(blocked in source for blocked in ['tripadvisor', 'yelp', 'google']):
                        clean_sources.append(source)

                cleaned_restaurant = {
                    "name": name,
                    "address": restaurant.get("address", "Requires verification").strip(),
                    "description": description,
                    "sources": clean_sources
                }

                cleaned_restaurants.append(cleaned_restaurant)

            # Format for compatibility with existing system
            return {
                "edited_results": {
                    "main_list": [{
                        "name": r["name"],
                        "address": r["address"],
                        "description": r["description"],
                        "sources": r["sources"],
                        "missing_info": ["address"] if r["address"] == "Requires verification" else []
                    } for r in cleaned_restaurants]
                },
                "follow_up_queries": self._generate_follow_up_queries(cleaned_restaurants, destination)
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Raw response: {ai_output[:500]}...")
            return self._fallback_response()
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return self._fallback_response()

    def _generate_follow_up_queries(self, restaurants, destination):
        """
        Generate search queries for restaurants that need address verification.
        """
        queries = []
        for restaurant in restaurants:
            if restaurant.get("address") == "Requires verification":
                queries.append(f"{restaurant['name']} {destination} address location")

        return queries[:5]  # Limit to 5 follow-up queries

    def _fallback_response(self):
        """Return fallback response when AI processing fails"""
        return {
            "edited_results": {
                "main_list": []
            },
            "follow_up_queries": []
        }

    @log_function_call
    def edit(self, scraped_results, original_query, destination="Unknown"):
        """
        Main method to process scraped results and return consolidated restaurant recommendations.

        Args:
            scraped_results: List of scraped articles with content  
            original_query: The user's original search query
            destination: The city/location being searched

        Returns:
            Dict with edited_results and follow_up_queries
        """
        try:
            logger.info(f"Processing {len(scraped_results)} articles for {destination}")

            # Prepare content for AI processing
            scraped_content = self._prepare_scraped_content(scraped_results)

            if not scraped_content.strip():
                logger.warning("No substantial content found in scraped results")
                return self._fallback_response()

            # Get AI processing
            response = self.chain.invoke({
                "original_query": original_query,
                "destination": destination,
                "scraped_content": scraped_content
            })

            # Process AI output
            result = self._post_process_results(response.content, original_query, destination)

            logger.info(f"Successfully processed {len(result['edited_results']['main_list'])} restaurants")

            return result

        except Exception as e:
            logger.error(f"Error in editor: {e}")
            dump_chain_state("editor_error", {
                "error": str(e),
                "query": original_query,
                "destination": destination
            })
            return self._fallback_response()