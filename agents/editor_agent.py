# agents/editor_agent.py - UPDATED FOR AI DATABASE INTEGRATION
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
        self.config = config

        # AI-enhanced prompt for database restaurant formatting
        self.database_formatting_prompt = """
You are an AI assistant that formats restaurant recommendations from a database for users.

YOUR TASK:
1. Take structured restaurant data from database 
2. Create engaging, user-friendly descriptions
3. Enhance with context from cuisine tags and raw descriptions
4. Format consistently for presentation

FORMATTING RULES:
- Create 20-35 word descriptions that highlight unique features
- Use cuisine tags to provide context (e.g., "modern italian", "neighborhood gem")
- Extract key highlights from raw descriptions without copying verbatim
- Focus on what makes each restaurant special or noteworthy
- Maintain authenticity - don't oversell or add false information

OUTPUT FORMAT:
Return ONLY valid JSON:
{{
  "restaurants": [
    {{
      "name": "Restaurant Name",
      "address": "Address or 'Address verification needed'",
      "description": "Enhanced 20-35 word description with key highlights",
      "sources": ["domain1.com", "domain2.com"],
      "cuisine_context": "Primary cuisine type and style",
      "mention_count": 3
    }}
  ]
}}

QUALITY STANDARDS:
- Descriptions should feel natural and engaging
- Include specific details when available (signature dishes, atmosphere, etc.)
- Use cuisine tags to provide helpful context
- Maintain professional, informative tone
"""

        # Traditional prompt for scraped content (keep existing logic)
        self.scraped_content_prompt = """
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

        # Create prompt templates
        self.database_prompt = ChatPromptTemplate.from_messages([
            ("system", self.database_formatting_prompt),
            ("human", """
Original query: {original_query}
Destination: {destination}

Database restaurants to format:
{database_restaurants}

Format these restaurants for user presentation as JSON.
""")
        ])

        self.scraped_prompt = ChatPromptTemplate.from_messages([
            ("system", self.scraped_content_prompt),
            ("human", """
Original query: {original_query}
Destination: {destination}

Scraped content from multiple sources:
{scraped_content}

Process this content and return the consolidated restaurant list as JSON.
""")
        ])

        # Create chains
        self.database_chain = self.database_prompt | self.model
        self.scraped_chain = self.scraped_prompt | self.model

    @log_function_call
    def edit(self, scraped_results=None, database_restaurants=None, original_query="", destination="Unknown"):
        """
        Main method that handles both database restaurants and scraped content.

        Args:
            scraped_results: List of scraped articles (traditional mode)
            database_restaurants: List of database restaurant objects (new AI mode)
            original_query: The user's original search query
            destination: The city/location being searched

        Returns:
            Dict with edited_results and follow_up_queries
        """
        try:
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
            logger.info(f"🗃️ Processing {len(database_restaurants)} restaurants from AI database for {destination}")

            # Prepare database restaurant data for AI formatting
            database_content = self._prepare_database_content(database_restaurants)

            if not database_content.strip():
                logger.warning("No substantial database content to process")
                return self._fallback_response()

            # Get AI formatting
            response = self.database_chain.invoke({
                "original_query": original_query,
                "destination": destination,
                "database_restaurants": database_content
            })

            # Process AI output
            result = self._post_process_database_results(response.content, database_restaurants, original_query, destination)

            logger.info(f"✅ Successfully formatted {len(result['edited_results']['main_list'])} database restaurants")

            return result

        except Exception as e:
            logger.error(f"Error processing database restaurants: {e}")
            return self._fallback_response()

    def _process_scraped_content(self, scraped_results, original_query, destination):
        """Process traditional scraped content (existing logic)"""
        try:
            logger.info(f"🌐 Processing {len(scraped_results)} scraped articles for {destination}")

            # Prepare content for AI processing
            scraped_content = self._prepare_scraped_content(scraped_results)

            if not scraped_content.strip():
                logger.warning("No substantial scraped content found")
                return self._fallback_response()

            # Get AI processing
            response = self.scraped_chain.invoke({
                "original_query": original_query,
                "destination": destination,
                "scraped_content": scraped_content
            })

            # Process AI output
            result = self._post_process_scraped_results(response.content, original_query, destination)

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
        """Convert search results into a clean format for AI processing (existing method)"""
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

    def _post_process_database_results(self, ai_output, original_restaurants, original_query, destination):
        """Post-process AI output for database restaurants"""
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

            # Validate and enhance each restaurant entry
            cleaned_restaurants = []
            seen_names = set()

            for restaurant in restaurants:
                name = restaurant.get("name", "").strip()
                if not name or name.lower() in seen_names:
                    continue

                seen_names.add(name.lower())

                # Find original restaurant data for additional info
                original = self._find_original_restaurant(name, original_restaurants)

                # Enhance with database information
                enhanced_restaurant = {
                    "name": name,
                    "address": restaurant.get("address", "Address verification needed"),
                    "description": restaurant.get("description", "").strip(),
                    "sources": restaurant.get("sources", []),
                    "cuisine_context": restaurant.get("cuisine_context", ""),
                    "mention_count": restaurant.get("mention_count", 1),
                    "from_database": True,
                    "database_id": original.get("id") if original else None,
                    "cuisine_tags": original.get("cuisine_tags", []) if original else []
                }

                # Validate description length
                description = enhanced_restaurant["description"]
                word_count = len(description.split()) if description else 0
                if word_count < 15 or word_count > 40:
                    logger.warning(f"Database restaurant {name} description has {word_count} words, outside 15-40 range")

                cleaned_restaurants.append(enhanced_restaurant)

            # Format for compatibility with existing system
            return {
                "edited_results": {
                    "main_list": [{
                        "name": r["name"],
                        "address": r["address"],
                        "description": r["description"],
                        "sources": r["sources"],
                        "cuisine_context": r["cuisine_context"],
                        "mention_count": r["mention_count"],
                        "from_database": True,
                        "missing_info": ["address"] if r["address"] == "Address verification needed" else []
                    } for r in cleaned_restaurants]
                },
                "follow_up_queries": self._generate_database_follow_up_queries(cleaned_restaurants, destination)
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI database response as JSON: {e}")
            logger.error(f"Raw response: {ai_output[:500]}...")
            return self._fallback_response()
        except Exception as e:
            logger.error(f"Error in database post-processing: {e}")
            return self._fallback_response()

    def _post_process_scraped_results(self, ai_output, original_query, destination):
        """Post-process AI output for scraped content (existing logic with enhancements)"""
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
                    logger.warning(f"Scraped restaurant {name} description has {word_count} words, outside 10-35 range")

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
                    "sources": clean_sources,
                    "from_database": False
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
                        "from_database": False,
                        "missing_info": ["address"] if r["address"] == "Requires verification" else []
                    } for r in cleaned_restaurants]
                },
                "follow_up_queries": self._generate_scraped_follow_up_queries(cleaned_restaurants, destination)
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI scraped response as JSON: {e}")
            logger.error(f"Raw response: {ai_output[:500]}...")
            return self._fallback_response()
        except Exception as e:
            logger.error(f"Error in scraped post-processing: {e}")
            return self._fallback_response()

    def _find_original_restaurant(self, name, original_restaurants):
        """Find original restaurant data by name"""
        for restaurant in original_restaurants:
            if restaurant.get("name", "").lower() == name.lower():
                return restaurant
        return None

    def _extract_domain(self, url):
        """Extract clean domain name from URL for source attribution"""
        try:
            if not url:
                return "unknown"

            # Remove protocol
            if url.startswith(('http://', 'https://')):
                url = url.split('://', 1)[1]

            # Get domain part
            domain = url.split('/')[0]

            # Remove www.
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain.lower()

        except Exception:
            return "unknown"

    def _generate_database_follow_up_queries(self, restaurants, destination):
        """Generate follow-up queries for database restaurants that need address verification"""
        queries = []
        for restaurant in restaurants:
            if restaurant.get("address") == "Address verification needed":
                queries.append(f"{restaurant['name']} {destination} address location contact")

        # Prioritize restaurants with higher mention counts
        restaurants_needing_verification = [r for r in restaurants if r.get("address") == "Address verification needed"]
        restaurants_needing_verification.sort(key=lambda x: x.get("mention_count", 0), reverse=True)

        return [f"{r['name']} {destination} address location" for r in restaurants_needing_verification[:5]]

    def _generate_scraped_follow_up_queries(self, restaurants, destination):
        """Generate search queries for scraped restaurants that need address verification"""
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

    # BACKWARD COMPATIBILITY: Keep existing method signature for old calls
    def process_scraped_results(self, scraped_results, original_query, destination="Unknown"):
        """Backward compatibility method"""
        return self.edit(scraped_results=scraped_results, original_query=original_query, destination=destination)

    # NEW METHOD: For AI database processing
    def process_database_restaurants(self, database_restaurants, original_query, destination="Unknown"):
        """New method specifically for database restaurant processing"""
        return self.edit(database_restaurants=database_restaurants, original_query=original_query, destination=destination)