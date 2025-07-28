# agents/editor_agent.py - SIMPLIFIED VERSION with raw query validation in prompts only
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
            model="gpt-4o",  # Use GPT-4o as specified
            temperature=0.2  # Lower temperature for more consistent formatting
        )
        self.config = config

        # SIMPLIFIED: Database restaurant processing with raw query validation built into prompt
        self.database_formatting_prompt = """
You are an AI assistant that formats restaurant recommendations from a database for users.

ORIGINAL USER REQUEST: "{{original_query}}"
DESTINATION: {{destination}}

YOUR TASK:
1. Format these database restaurants for the user
2. ONLY include restaurants that truly match the user's original request
3. Create engaging descriptions highlighting what makes each place special
4. Remove restaurants that don't fit the user's actual needs

FILTERING RULES:
- Check each restaurant against the user's ORIGINAL REQUEST (not just the destination)
- Remove restaurants that don't match the user's style, cuisine, or occasion preferences
- If user mentions specific needs (romantic, family-friendly, budget, etc.), only include matching restaurants
- Focus on restaurants that would genuinely satisfy the user's intent

OUTPUT FORMAT (keep it simple like before):
Return ONLY valid JSON:
{{
  "restaurants": [
    {{
      "name": "Restaurant Name",
      "address": "Address or 'Address verification needed'",
      "description": "Engaging 20-35 word description highlighting unique features",
      "sources": ["domain1.com", "domain2.com"]
    }}
  ]
}}

QUALITY STANDARDS:
- Descriptions should be specific and engaging
- Include signature dishes, atmosphere, or special features when available
- Remove generic phrases like "great food" or "nice atmosphere"
- Focus on what makes each restaurant special for this user's needs
"""

        # SIMPLIFIED: Scraped content processing with raw query validation built into prompt
        self.scraped_content_prompt = """
You are an AI assistant that processes restaurant recommendations from web scraping results.

ORIGINAL USER REQUEST: "{{original_query}}"
DESTINATION: {{destination}}

YOUR TASK:
1. Extract restaurants from scraped content that match the user's original request
2. ONLY include restaurants that truly fit what the user is looking for
3. Create detailed descriptions from the source content
4. Group information if restaurants appear in multiple sources

FILTERING RULES:
- Check each restaurant against the user's ORIGINAL REQUEST
- Remove restaurants that don't match the user's stated preferences
- If user asks for specific cuisine, occasion, or style - only include matching places
- Focus on restaurants that would genuinely satisfy the user's intent

CONSOLIDATION RULES:
- If a restaurant appears in multiple sources, combine all information
- Use the most complete address found across sources
- If addresses conflict or are missing, mark for verification
- Create detailed descriptions that highlight what makes each restaurant special
- Avoid generic phrases like "great food" or "nice atmosphere"

OUTPUT FORMAT (keep it simple):
Return ONLY valid JSON:
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

        # Create prompt templates
        self.database_prompt = ChatPromptTemplate.from_messages([
            ("system", self.database_formatting_prompt),
            ("human", """
Original user request: {{original_query}}
Destination: {{destination}}

Database restaurants to format:
{{database_restaurants}}

Format these restaurants, but ONLY include ones that match the user's original request.
""")
        ])

        self.scraped_prompt = ChatPromptTemplate.from_messages([
            ("system", self.scraped_content_prompt),
            ("human", """
Original user request: {{original_query}}
Destination: {{destination}}

Scraped content from multiple sources:
{{scraped_content}}

Extract restaurants that match the user's original request and return as JSON.
""")
        ])

        # Create chains
        self.database_chain = self.database_prompt | self.model
        self.scraped_chain = self.scraped_prompt | self.model

    @log_function_call
    def edit(self, scraped_results=None, database_restaurants=None, original_query="", destination="Unknown"):
        """
        Main method that handles both database restaurants and scraped content.
        SIMPLIFIED: Raw query validation is built into the prompts, no separate verification step.

        Args:
            scraped_results: List of scraped articles (traditional mode)
            database_restaurants: List of database restaurant objects (new AI mode)
            original_query: The user's original RAW search query (not formatted search query)
            destination: The city/location being searched

        Returns:
            Dict with edited_results and follow_up_queries
        """
        try:
            logger.info(f"✏️ Editor processing for destination: {destination}")
            logger.info(f"📝 Original user query: {original_query}")

            # Determine which mode to use
            if database_restaurants:
                return self._process_database_restaurants(database_restaurants, original_query, destination)
            elif scraped_results:
                return self._process_scraped_content(scraped_results, original_query, destination)
            else:
                logger.warning("No input provided to editor - neither database restaurants nor scraped results")
                return self._fallback_response()

        except Exception as e:
            logger.error(f"Error in editor: {e}")
            dump_chain_state("editor_error", {
                "error": str(e),
                "query": original_query,
                "destination": destination,
                "has_database_restaurants": bool(database_restaurants),
                "has_scraped_results": bool(scraped_results)
            })
            return self._fallback_response()

    def _process_database_restaurants(self, database_restaurants, original_query, destination):
        """Process restaurants from AI-matched database"""
        try:
            logger.info(f"🗃️ Processing {len(database_restaurants)} restaurants from database for {destination}")

            # Prepare database restaurant data for AI formatting
            database_content = self._prepare_database_content(database_restaurants)

            if not database_content.strip():
                logger.warning("No substantial database content to process")
                return self._fallback_response()

            # Get AI formatting (with raw query validation built into prompt)
            response = self.database_chain.invoke({
                "original_query": original_query,
                "destination": destination,
                "database_restaurants": database_content
            })

            # Process AI output (simplified)
            result = self._post_process_results(response.content, "database", destination)

            logger.info(f"✅ Successfully formatted {len(result['edited_results']['main_list'])} database restaurants")

            return result

        except Exception as e:
            logger.error(f"Error processing database restaurants: {e}")
            return self._fallback_response()

    def _process_scraped_content(self, scraped_results, original_query, destination):
        """Process traditional scraped content"""
        try:
            logger.info(f"🌐 Processing {len(scraped_results)} scraped articles for {destination}")

            # Prepare content for AI processing
            scraped_content = self._prepare_scraped_content(scraped_results)

            if not scraped_content.strip():
                logger.warning("No substantial scraped content found")
                return self._fallback_response()

            # Get AI processing (with raw query validation built into prompt)
            response = self.scraped_chain.invoke({
                "original_query": original_query,
                "destination": destination,
                "scraped_content": scraped_content
            })

            # Process AI output (simplified)
            result = self._post_process_results(response.content, "scraped", destination)

            logger.info(f"✅ Successfully processed {len(result['edited_results']['main_list'])} scraped restaurants")

            return result

        except Exception as e:
            logger.error(f"Error processing scraped content: {e}")
            return self._fallback_response()

    def _prepare_database_content(self, database_restaurants):
        """Convert database restaurant objects into format for AI processing"""
        formatted_restaurants = []

        for i, restaurant in enumerate(database_restaurants, 1):
            # Extract key information from database format
            name = restaurant.get("name", "Unknown Restaurant")
            address = restaurant.get("address") or "Address verification needed"
            cuisine_tags = restaurant.get("cuisine_tags", [])
            raw_description = restaurant.get("raw_description", "")
            sources = restaurant.get("sources", [])
            mention_count = restaurant.get("mention_count", 1)

            # Clean sources to domain names
            clean_sources = [self._extract_domain(url) for url in sources]
            clean_sources = [s for s in clean_sources if s != "unknown"]

            formatted_restaurant = f"""
RESTAURANT {i}:
Name: {name}
Address: {address}
Cuisine Tags: {', '.join(cuisine_tags[:5]) if cuisine_tags else 'None'}
Mention Count: {mention_count}
Sources: {', '.join(clean_sources[:3]) if clean_sources else 'None'}
Raw Description Sample: {raw_description[:300] if raw_description else 'No description available'}...
---"""
            formatted_restaurants.append(formatted_restaurant)

        return "\n".join(formatted_restaurants)

    def _prepare_scraped_content(self, search_results):
        """Convert search results into a clean format for AI processing"""
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

    def _post_process_results(self, ai_output, source_type, destination):
        """
        SIMPLIFIED: Single post-processing method for both database and scraped results
        Uses the original simple JSON structure
        """
        try:
            # Parse JSON response
            if isinstance(ai_output, str):
                content = ai_output.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                parsed_data = json.loads(content)
            else:
                parsed_data = ai_output

            restaurants = parsed_data.get("restaurants", [])

            # Simple validation and cleaning
            cleaned_restaurants = []
            seen_names = set()

            for restaurant in restaurants:
                name = restaurant.get("name", "").strip()
                if not name or name.lower() in seen_names:
                    continue

                seen_names.add(name.lower())

                # Keep the simple structure from your original implementation
                cleaned_restaurant = {
                    "name": name,
                    "address": restaurant.get("address", "Requires verification"),
                    "description": restaurant.get("description", "").strip(),
                    "sources": restaurant.get("sources", [])
                }

                # Validate description
                if len(cleaned_restaurant["description"]) < 10:
                    logger.warning(f"Short description for {name}: {cleaned_restaurant['description']}")

                cleaned_restaurants.append(cleaned_restaurant)

            # Generate follow-up queries for address verification
            follow_up_queries = self._generate_follow_up_queries(cleaned_restaurants, destination)

            return {
                "edited_results": {
                    "main_list": cleaned_restaurants
                },
                "follow_up_queries": follow_up_queries
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI output as JSON: {e}")
            logger.error(f"Raw AI output: {ai_output[:500]}...")
            return self._fallback_response()
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return self._fallback_response()

    def _extract_domain(self, url):
        """Extract domain from URL for source attribution"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc

            if domain.startswith('www.'):
                domain = domain[4:]

            return domain.lower()

        except Exception:
            return "unknown"

    def _generate_follow_up_queries(self, restaurants, destination):
        """Generate follow-up queries for restaurants that need address verification"""
        queries = []
        for restaurant in restaurants:
            address = restaurant.get("address", "")
            if "verification" in address.lower() or "requires" in address.lower():
                queries.append(f"{restaurant['name']} {destination} address location contact")

        return queries[:5]  # Limit to 5 follow-up queries

    def _fallback_response(self):
        """Return fallback response when AI processing fails"""
        return {
            "edited_results": {
                "main_list": []
            },
            "follow_up_queries": []
        }

    # BACKWARD COMPATIBILITY: Keep existing method signatures
    def process_scraped_results(self, scraped_results, original_query, destination="Unknown"):
        """Backward compatibility method"""
        return self.edit(scraped_results=scraped_results, original_query=original_query, destination=destination)

    def process_database_restaurants(self, database_restaurants, original_query, destination="Unknown"):
        """Method for database restaurant processing"""
        return self.edit(database_restaurants=database_restaurants, original_query=original_query, destination=destination)

    def get_editor_stats(self):
        """Get statistics about editor performance"""
        return {
            "editor_agent_enabled": True,
            "model_used": "gpt-4o",
            "validation": "built_into_prompts"
        }