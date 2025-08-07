# agents/editor_agent.py - ENHANCED VERSION with intelligent selection and dual API keys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
import json
import logging
from collections import defaultdict
from utils.debug_utils import dump_chain_state, log_function_call
import os

logger = logging.getLogger(__name__)

class EditorAgent:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2
        )
        self.config = config

        # Initialize selection model for restaurant filtering
        self.selection_model = ChatOpenAI(
            model="gpt-4o-mini",  # Faster model for selection
            temperature=0.3  # Slightly higher for diversity
        )

        # Selection prompt for choosing best restaurants BEFORE follow-up
        self.selection_prompt = """
        You are an expert restaurant curator selecting the BEST restaurants for a user's query.

        USER QUERY: {{raw_query}}
        DESTINATION: {{destination}}

        AVAILABLE RESTAURANTS:
        {{restaurants_list}}

        YOUR TASK: Select up to {{max_selections}} restaurants that:
        1. Best match the user's query
        2. Offer variety and diversity
        3. Are highly recommended/praised
        4. If the query doesn't specify, include both traditional and innovative options
        5. Prioritize restaurants mentioned by famous guides and media (Michelin, World's 50 Best, Le Monde, Vodue, Condé Nast Traveller, etc.)

        SELECTION CRITERIA:
        - Quality over quantity - only include excellent matches
        - For broad queries, ensure diversity (cuisines, styles, price points)
        - For specific queries, focus on best matches but still include variety
        - Consider recommendation sources (prefer expert/guide recommendations)
        - If the query doesn't specify, balance between established classics and exciting new places

        Return ONLY a JSON array of selected restaurant names:
        ["Restaurant Name 1", "Restaurant Name 2", ...]
        """

        # Database restaurant processing - diplomatic approach like a hotel concierge
        self.database_formatting_prompt = """
        You are an expert restaurant concierge who formats database restaurant recommendations for users.
        YOU HAVE TWO JOBS:
        1. Format the restaurants in a clean, engaging way
        2. Be diplomatic about matches - like a skilled concierge, explain your choices even if they're not 100% perfect matches

        CONCIERGE APPROACH:
        - If restaurants don't perfectly match ALL user requirements, still include them but explain diplomatically why you chose them
        - Use phrases like "While this may not have X specifically mentioned, it offers Y which makes it worth considering"
        - Be honest about uncertainties: "though I cannot confirm if they have vegan options, their modern approach suggests they likely accommodate dietary preferences"
        - Focus on positive aspects and potential matches rather than strict filtering

        ORIGINAL USER REQUEST: {{raw_query}}
        DESTINATION: {{destination}}

        SOURCE OUTPUT RULE:
        List 1–3 sources for each restaurant, if possible, avoid listing the same source for many restaurants in the list.

        OUTPUT FORMAT (keep it simple):
        Return ONLY valid JSON:
        {{
          "restaurants": [
            {{
              "name": "Restaurant Name",
              "address": "Full Address Here",
              "description": "Engaging 25-45 word description highlighting features, with diplomatic notes about potential matches to user needs",
              "sources": ["domain1.com", "domain2.com", "domain3.com", etc.]
            }}
          ]
        }}

        DESCRIPTION GUIDELINES:
        - Include what makes each restaurant special
        - Diplomatically address user requirements (even if uncertain)
        - Use concierge language: "likely offers", "known for", "specializes in", "worth considering for"
        - Be specific about confirmed features, diplomatic about uncertain ones
        """

        # Scraped content processing - diplomatic approach
        self.scraped_content_prompt = """
        You are an expert restaurant concierge who processes web content to recommend restaurants.
        YOU HAVE TWO JOBS:
        1. Extract restaurant recommendations from the content in all languages present in the file, not just English
        2. List restaurants that match user's request best. If they don't match all requirements, still include those that might be suitable, but explain diplomatically why you chose them

        CONSOLIDATION RULES:
        - Give preference to the restaurants that best match the user's original request, list them first
        - If a restaurant appears in multiple sources, combine all information
        - Use the most complete address found across sources
        - If addresses conflict or are missing, mark for verification
        - Create descriptions that highlight strengths while diplomatically addressing user needs
        - Avoid generic phrases like "great food" or "nice atmosphere"

        CONCIERGE APPROACH:
        - Include restaurants that fully or for most part match user needs, but explain diplomatically why
        - If the restaurant only matches the query partially use phrases like "While not explicitly mentioned as X, their focus on Y suggests they would accommodate Z"
        - Be honest about what you can/cannot confirm from the content
        - Focus on potential and positive aspects rather than strict requirements

        ORIGINAL USER REQUEST: {{raw_query}}
        DESTINATION: {{destination}}

        SOURCE OUTPUT RULE:
        List 1–3 sources for each restaurant, if possible, avoid listing the same source for many restaurants in the list.

        OUTPUT FORMAT:
        Return ONLY valid JSON:
        {{
          "restaurants": [
            {{
              "name": "Restaurant Name", 
              "address": "Full Address Here",
              "description": "Diplomatic 20-30 word description explaining why this restaurant suits the user, even if not a perfect match",
              "sources": ["domain1.com", "domain2.com", "domain3.com", etc.]
            }}
          ]
        }}

        IMPORTANT:
        - NEVER include Tripadvisor, Yelp, Opentable, or Google in sources
        - Be diplomatic but honest in descriptions
        - Include restaurants that partially match rather than having an empty list
        - Use concierge language to explain your reasoning
        - ALL addresses must be formatted as clickable Google Maps links
        """

        # Create prompt templates
        self.database_prompt = ChatPromptTemplate.from_messages([
            ("system", self.database_formatting_prompt),
            ("human", """
Original user request: {raw_query}
Destination: {destination}

Database restaurants to format:
{database_restaurants}

Format these restaurants using the diplomatic concierge approach. Include restaurants even if they don't perfectly match all requirements, but explain your reasoning diplomatically.
""")
        ])

        self.scraped_prompt = ChatPromptTemplate.from_messages([
            ("system", self.scraped_content_prompt),
            ("human", """
Original user request: {raw_query}
Destination: {destination}

Scraped content from multiple sources:
{scraped_content}

Extract restaurants using the diplomatic concierge approach. Include good options even if they don't perfectly match all user requirements, but explain your reasoning.
""")
        ])

        # Create chains
        self.database_chain = self.database_prompt | self.model
        self.scraped_chain = self.scraped_prompt | self.model

    def _select_best_restaurants(self, restaurants, raw_query, destination, max_selections=10):
        """
        Select the best restaurants BEFORE follow-up queries

        Args:
            restaurants: List of restaurant dictionaries
            raw_query: User's original query
            destination: City/location
            max_selections: Maximum number to select (default 10)

        Returns:
            List of selected restaurants
        """
        try:
            # Format restaurants for selection
            restaurants_text = []
            for i, restaurant in enumerate(restaurants):
                name = restaurant.get('name', 'Unknown')
                description = restaurant.get('description', '')
                sources = restaurant.get('sources', [])

                # Check if mentioned by famous guides
                famous_guides = ['michelin', 'worlds 50 best', 'worlds50best', 'asia 50 best']
                has_guide_mention = any(
                    any(guide in str(source).lower() for guide in famous_guides)
                    for source in sources
                )

                restaurant_info = f"{i+1}. {name}"
                if description:
                    restaurant_info += f" - {description[:100]}..."
                if has_guide_mention:
                    restaurant_info += " [GUIDE RECOMMENDED]"

                restaurants_text.append(restaurant_info)

            # Create selection prompt
            prompt = ChatPromptTemplate.from_template(self.selection_prompt)
            selection_chain = prompt | self.selection_model

            # Get selection
            response = selection_chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "restaurants_list": "\n".join(restaurants_text),
                "max_selections": max_selections
            })

            # Parse selected names
            selected_names = json.loads(response.content)

            # Filter restaurants to only selected ones
            selected_restaurants = []
            for restaurant in restaurants:
                if restaurant.get('name') in selected_names:
                    selected_restaurants.append(restaurant)

            logger.info(f"🎯 Selected {len(selected_restaurants)} best restaurants from {len(restaurants)} total")

            return selected_restaurants

        except Exception as e:
            logger.error(f"Error in restaurant selection: {e}")
            # Fallback: return first max_selections restaurants
            return restaurants[:max_selections]

    def _generate_follow_up_queries(self, restaurants, destination):
        """
        Generate follow-up queries for restaurants that need address verification
        Now supports using both API keys to double the capacity
        """
        queries = []

        # Check if we have a second Google Maps API key
        has_second_key = bool(os.environ.get("GOOGLE_MAPS_API_KEY2"))

        # Determine max queries based on available API keys
        if has_second_key:
            max_queries = 10  # Double capacity with 2 keys
            logger.info("🔑 Using dual Google Maps API keys - capacity doubled to 10 queries")
        else:
            max_queries = 5  # Single key capacity
            logger.info("🔑 Using single Google Maps API key - capacity limited to 5 queries")

        for restaurant in restaurants:
            address = restaurant.get("address", "")
            # Generate query for all restaurants to ensure proper address verification
            queries.append({
                "restaurant_name": restaurant['name'],
                "query": f"{restaurant['name']} {destination} restaurant address location",
                "needs_verification": "verification" in address.lower() or "requires" in address.lower() or not address
            })

        # Prioritize restaurants that explicitly need verification
        queries.sort(key=lambda x: not x['needs_verification'])

        return queries[:max_queries]

    @log_function_call
    def edit(self, scraped_results=None, database_restaurants=None, raw_query="", destination="Unknown", 
             content_source=None, processing_mode=None, evaluation_context=None, **kwargs):
        """
        Main editing method with intelligent restaurant selection

        Key improvements:
        1. Selects best restaurants BEFORE follow-up queries
        2. Ensures diversity in selection
        3. Supports dual API keys for more follow-up queries
        """
        try:
            # Use raw_query consistently
            query = raw_query or kwargs.get('original_query', '') or ""

            # Determine processing mode
            if processing_mode:
                mode = processing_mode
            else:
                # Auto-detect mode based on available content
                if database_restaurants and scraped_results:
                    mode = "hybrid"
                elif database_restaurants:
                    mode = "database_only"
                elif scraped_results:
                    mode = "web_only"
                else:
                    mode = "unknown"

            # Use content_source from ContentEvaluationAgent if available
            if content_source:
                source = content_source
                logger.info(f"📋 Using content source from evaluation: {source}")
            else:
                # Auto-detect source based on available content
                if database_restaurants and scraped_results:
                    source = "hybrid"
                elif database_restaurants:
                    source = "database"
                elif scraped_results:
                    source = "web_search"
                else:
                    source = "unknown"

            logger.info(f"📝 Editor processing - Mode: {mode}, Source: {source}")
            logger.info(f"🎯 Query: {query}")
            logger.info(f"📍 Destination: {destination}")

            # Route to appropriate processing method
            if mode == "hybrid":
                result = self._process_hybrid_content(database_restaurants, scraped_results, query, destination)
            elif mode == "database_only":
                result = self._process_database_restaurants(database_restaurants, query, destination)
            elif mode == "web_only":
                result = self._process_scraped_content(scraped_results, query, destination)
            else:
                logger.warning(f"⚠️ Unknown processing mode: {mode}")
                return self._fallback_response()

            # ENHANCED: Select best restaurants before generating follow-up queries
            if result and result.get("edited_results", {}).get("main_list"):
                all_restaurants = result["edited_results"]["main_list"]

                # Determine selection count based on query breadth
                is_broad_query = any(term in query.lower() for term in 
                    ['best restaurants', 'where to eat', 'food scene', 'dining', 'recommendations'])

                max_selections = 10 if is_broad_query else 8

                # Select best restaurants
                selected_restaurants = self._select_best_restaurants(
                    all_restaurants, 
                    query, 
                    destination,
                    max_selections
                )

                # Update result with selected restaurants
                result["edited_results"]["main_list"] = selected_restaurants

                # Generate follow-up queries for selected restaurants
                result["follow_up_queries"] = self._generate_follow_up_queries(
                    selected_restaurants, 
                    destination
                )

            return result

        except Exception as e:
            logger.error(f"❌ Error in editor agent: {e}")
            dump_chain_state("editor_error", locals(), error=e)
            return self._fallback_response()

    def _process_hybrid_content(self, database_restaurants, scraped_results, raw_query, destination):
        """Process both database and scraped content - updated for better selection"""
        logger.info(f"🔄 Processing hybrid content for {destination}")
        logger.info(f"📊 Database restaurants: {len(database_restaurants) if database_restaurants else 0}")
        logger.info(f"📊 Scraped results: {len(scraped_results) if scraped_results else 0}")

        try:
            all_restaurants = []

            # Process database restaurants first (these were preserved by ContentEvaluationAgent)
            if database_restaurants:
                logger.info("🗃️ Processing preserved database restaurants")
                db_result = self._process_database_restaurants(database_restaurants, raw_query, destination)
                db_restaurants = db_result.get('edited_results', {}).get('main_list', [])

                # Mark database restaurants as preserved
                for restaurant in db_restaurants:
                    restaurant['_source_type'] = 'database_preserved'

                all_restaurants.extend(db_restaurants)
                logger.info(f"✅ Added {len(db_restaurants)} database restaurants")

            # Then process scraped content (additional search results)
            if scraped_results:
                logger.info("🌐 Processing additional web search results")
                scraped_result = self._process_scraped_content(scraped_results, raw_query, destination)
                web_restaurants = scraped_result.get('edited_results', {}).get('main_list', [])

                # Mark web restaurants as additional
                for restaurant in web_restaurants:
                    restaurant['_source_type'] = 'web_additional'

                all_restaurants.extend(web_restaurants)
                logger.info(f"✅ Added {len(web_restaurants)} web search restaurants")

            # Remove duplicates based on restaurant name with preference for database
            seen_names = set()
            unique_restaurants = []

            # First pass: Add database restaurants (preserved ones get priority)
            for restaurant in all_restaurants:
                name = restaurant.get('name', '').lower().strip()
                if name and name not in seen_names and restaurant.get('_source_type') == 'database_preserved':
                    seen_names.add(name)
                    unique_restaurants.append(restaurant)

            # Second pass: Add web restaurants if not already present
            for restaurant in all_restaurants:
                name = restaurant.get('name', '').lower().strip()
                if name and name not in seen_names and restaurant.get('_source_type') == 'web_additional':
                    seen_names.add(name)
                    unique_restaurants.append(restaurant)

            # Clean up metadata before returning
            for restaurant in unique_restaurants:
                restaurant.pop('_source_type', None)

            logger.info(f"🎯 Final hybrid result: {len(unique_restaurants)} unique restaurants")

            return {
                "edited_results": {"main_list": unique_restaurants},
                "follow_up_queries": [],  # Will be generated after selection
                "processing_notes": {
                    "mode": "hybrid",
                    "database_count": len(database_restaurants) if database_restaurants else 0,
                    "web_count": len(scraped_results) if scraped_results else 0,
                    "final_count": len(unique_restaurants)
                }
            }

        except Exception as e:
            logger.error(f"Error processing hybrid content: {e}")
            return self._fallback_response()

    def _process_database_restaurants(self, database_restaurants, raw_query, destination):
        """Process restaurants from AI-matched database"""
        try:
            logger.info(f"🗃️ Processing {len(database_restaurants)} restaurants from database for {destination}")

            # Prepare database restaurant data for AI formatting
            database_content = self._prepare_database_content(database_restaurants)

            if not database_content.strip():
                logger.warning("No substantial database content to process")
                return self._fallback_response()

            # Get AI formatting with diplomatic approach
            response = self.database_chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "database_restaurants": database_content
            })

            result = self._post_process_results(response, "database", destination)

            logger.info(f"✅ Successfully formatted {len(result['edited_results']['main_list'])} database restaurants")

            return result

        except Exception as e:
            logger.error(f"❌ Error processing database restaurants: {e}")
            dump_chain_state("database_processing_error", locals(), error=e)
            return self._fallback_response()

    def _process_scraped_content(self, scraped_results, raw_query, destination):
        """Process scraped web content"""
        try:
            logger.info(f"🌐 Processing {len(scraped_results)} scraped articles for {destination}")

            # Prepare scraped content
            scraped_content = self._prepare_scraped_content(scraped_results)

            if not scraped_content.strip():
                logger.warning("No substantial scraped content to process")
                return self._fallback_response()

            # Get AI extraction
            response = self.scraped_chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "scraped_content": scraped_content
            })

            result = self._post_process_results(response, "scraped", destination)

            logger.info(f"✅ Successfully extracted {len(result['edited_results']['main_list'])} restaurants from scraped content")

            return result

        except Exception as e:
            logger.error(f"❌ Error processing scraped content: {e}")
            dump_chain_state("scraped_processing_error", locals(), error=e)
            return self._fallback_response()

    def _prepare_database_content(self, database_restaurants):
        """Prepare database restaurant data for AI processing"""
        content_parts = []

        for restaurant in database_restaurants:
            # Get all available information
            name = restaurant.get('name', 'Unknown Restaurant')
            city = restaurant.get('city', '')
            country = restaurant.get('country', '')
            address = restaurant.get('address', '')
            cuisine_tags = restaurant.get('cuisine_tags', [])
            description = restaurant.get('description', '')

            # Get source information
            sources = []
            article_sources = restaurant.get('article_sources', [])

            if article_sources:
                for source in article_sources:
                    if isinstance(source, dict):
                        url = source.get('url', '')
                        if url:
                            # Extract domain
                            domain = url.split('/')[2] if '/' in url else url
                            sources.append(domain)
                    elif isinstance(source, str):
                        sources.append(source)

            # Build restaurant entry
            entry = f"Restaurant: {name}"

            if city or country:
                location_parts = [part for part in [city, country] if part]
                entry += f" ({', '.join(location_parts)})"

            if address:
                entry += f"\nAddress: {address}"

            if cuisine_tags:
                entry += f"\nCuisine: {', '.join(cuisine_tags)}"

            if description:
                entry += f"\nDescription: {description}"

            if sources:
                # Limit to 3 unique sources
                unique_sources = list(dict.fromkeys(sources))[:3]
                entry += f"\nSources: {', '.join(unique_sources)}"

            content_parts.append(entry)

        return "\n\n".join(content_parts)

    def _prepare_scraped_content(self, scraped_results):
        """Prepare scraped content for AI processing"""
        content_parts = []

        for article in scraped_results:
            # Extract article information
            url = article.get('url', '')
            title = article.get('title', '')
            content = article.get('content', '')
            sections = article.get('sections', [])

            # Skip if no meaningful content
            if not content and not sections:
                continue

            # Get domain for source tracking
            domain = url.split('/')[2] if '/' in url and url.startswith('http') else url

            # Build article entry
            entry = f"SOURCE: {domain}"
            if title:
                entry += f"\nTITLE: {title}"

            # Add content or sections
            if sections:
                # Use sectioned content if available
                for section in sections:
                    if isinstance(section, dict):
                        section_title = section.get('title', '')
                        section_content = section.get('content', '')
                        if section_title:
                            entry += f"\n\n[{section_title}]"
                        if section_content:
                            entry += f"\n{section_content}"
                    elif isinstance(section, str):
                        entry += f"\n{section}"
            elif content:
                # Use raw content if no sections
                entry += f"\n\nCONTENT:\n{content}"

            content_parts.append(entry)

        return "\n\n" + "="*50 + "\n\n".join(content_parts)

    def _post_process_results(self, ai_output, source_type, destination):
        """
        Process AI output for both database and scraped results
        FIXED: Handle AIMessage objects correctly
        """
        try:
            logger.info(f"🔍 Processing AI output from {source_type}")

            # FIX: Handle both AIMessage objects and direct strings
            if hasattr(ai_output, 'content'):
                # This is an AIMessage from LangChain
                content = ai_output.content
            elif isinstance(ai_output, str):
                # This is already a string
                content = ai_output
            else:
                # Try to convert to string
                content = str(ai_output)

            content = content.strip()

            # FIXED: Better handling of markdown code blocks
            # Remove markdown code blocks if present
            if "```json" in content:
                # Extract content between ```json and ```
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    content = content[start:end].strip()
            elif "```" in content:
                # Extract content between first ``` and second ```
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    content = content[start:end].strip()

            # Check if it's actually JSON
            if not content.startswith('{') and not content.startswith('['):
                logger.error(f"AI output doesn't look like JSON: {content[:200]}...")
                return self._fallback_response()

            # Parse the JSON
            parsed_data = json.loads(content)

            # Handle both direct restaurant list and nested structure
            if isinstance(parsed_data, list):
                # Direct list of restaurants
                restaurants = parsed_data
            elif isinstance(parsed_data, dict):
                # Nested structure
                restaurants = parsed_data.get("restaurants", [])
            else:
                logger.error(f"Unexpected AI output format: {type(parsed_data)}")
                return self._fallback_response()

            logger.info(f"📋 Parsed {len(restaurants)} restaurants from AI response")

            # Simple validation and cleaning
            cleaned_restaurants = []
            seen_names = set()

            for restaurant in restaurants:
                # Handle both dict and non-dict restaurant objects
                if not isinstance(restaurant, dict):
                    logger.warning(f"Skipping non-dict restaurant: {restaurant}")
                    continue

                name = restaurant.get("name", "").strip()
                if not name or name.lower() in seen_names:
                    logger.debug(f"Skipping duplicate or empty restaurant: {name}")
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
                description_length = len(cleaned_restaurant["description"])
                if description_length < 10:
                    logger.warning(f"Short description for {name}: {cleaned_restaurant['description']}")
                elif description_length > 200:
                    logger.warning(f"Long description for {name}: {description_length} chars")

                cleaned_restaurants.append(cleaned_restaurant)

            logger.info(f"✅ Cleaned and validated {len(cleaned_restaurants)} restaurants")

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
            logger.error(f"Raw AI output: {content[:500] if 'content' in locals() else 'Unable to extract content'}...")
            return self._fallback_response()
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            logger.error(f"AI output type: {type(ai_output)}")
            logger.error(f"AI output: {str(ai_output)[:200]}...")
            return self._fallback_response()

    def _extract_domain(self, url):
        """Extract clean domain from URL"""
        try:
            # Remove protocol
            if '://' in url:
                url = url.split('://', 1)[1]

            # Get domain part
            domain = url.split('/', 1)[0]

            # Remove www
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain.lower()

        except Exception:
            return "unknown"

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
        return self.edit(scraped_results=scraped_results, raw_query=original_query, destination=destination)

    def process_database_restaurants(self, database_restaurants, original_query, destination="Unknown"):
        """Method for database restaurant processing"""
        return self.edit(database_restaurants=database_restaurants, raw_query=original_query, destination=destination)

    def get_editor_stats(self):
        """Get statistics about editor performance"""
        return {
            "editor_agent_enabled": True,
            "model_used": "gpt-4o",
            "validation": "diplomatic_concierge_approach",
            "selection_enabled": True,
            "dual_api_keys_supported": True
        }