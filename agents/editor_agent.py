# agents/editor_agent.py - ENHANCED VERSION with chunking and AI-driven selection
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
from datetime import datetime
import json
import logging
from collections import defaultdict
from utils.debug_utils import dump_chain_state, log_function_call
import os
import tiktoken
import math

logger = logging.getLogger(__name__)

class EditorAgent:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2
        )
        self.config = config

        # Initialize tokenizer for chunking
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            logger.info("✅ Tokenizer initialized for chunking")
        except Exception as e:
            logger.warning(f"⚠️ Tokenizer initialization failed: {e}")
            self.tokenizer = None

        # Chunking configuration
        self.max_tokens_per_chunk = 15000  # Safe limit for Claude (leaves room for prompt)
        self.max_chars_per_chunk = 45000   # Character-based fallback
        self.chunk_overlap = 500           # Overlap between chunks

        # Database restaurant processing - diplomatic approach like a hotel concierge
        self.database_formatting_prompt = """
        You are an expert restaurant concierge who formats database restaurant recommendations for users.
        YOU HAVE TWO JOBS:
        1. Format the restaurants in a clean, engaging way
        2. Be diplomatic about matches - like a skilled concierge, explain your choices even if they're not 100% perfect matches

        CONCIERGE APPROACH:
        - List the restaurants that best match the user's original request first
        - If some restaurant in the final list are possible duplicates (similar name, same address), list them only once.
        - If restaurants don't perfectly match ALL user requirements and were included due to very limited results and very narrow query, explain diplomatically why you chose them
        - Whenever necessary (limited results), use phrases like "While this may not have X specifically mentioned, it offers Y which makes it worth considering"
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
        2. List restaurants that match user's request best. 
        3. If the results are very lmited due to very specific request, you can include those that might be suitable, but explain diplomatically why you chose them

        CONSOLIDATION RULES:
        - Give preference to the restaurants that best match the user's original request, list them first
        - If a restaurant appears in multiple sources, combine all information
        - Use the most complete address found across sources
        - If addresses conflict or are missing, mark for verification
        - Create descriptions that highlight strengths while diplomatically addressing user needs
        - Avoid generic phrases like "great food" or "nice atmosphere"

        CONCIERGE APPROACH:
        - If the restaurant only matches the query partially and had to be included due to very limited results, use phrases like "While not explicitly mentioned as X, their focus on Y suggests they would accommodate Z"
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

    def _needs_chunking(self, content):
        """Determine if content needs chunking"""
        # Character-based check (fast)
        if len(content) <= self.max_chars_per_chunk:
            return False

        # Token-based check if tokenizer available
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(content)
                return len(tokens) > self.max_tokens_per_chunk
            except Exception as e:
                logger.warning(f"Token counting failed: {e}")

        # Fallback to character count
        return len(content) > self.max_chars_per_chunk

    def _chunk_content(self, content):
        """Break content into manageable chunks with overlap"""
        if not self._needs_chunking(content):
            return [content]

        logger.info(f"📚 Chunking large content ({len(content)} chars)")

        chunks = []

        if self.tokenizer:
            # Token-based chunking (more accurate)
            try:
                tokens = self.tokenizer.encode(content)
                total_tokens = len(tokens)

                logger.info(f"🔢 Total tokens: {total_tokens}")

                chunk_count = math.ceil(total_tokens / self.max_tokens_per_chunk)
                logger.info(f"📊 Creating {chunk_count} chunks")

                for i in range(chunk_count):
                    start_token = i * self.max_tokens_per_chunk
                    end_token = min((i + 1) * self.max_tokens_per_chunk + self.chunk_overlap, total_tokens)

                    chunk_tokens = tokens[start_token:end_token]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    chunks.append(chunk_text)

                return chunks

            except Exception as e:
                logger.warning(f"Token-based chunking failed: {e}, falling back to character-based")

        # Character-based chunking (fallback)
        chunk_size = self.max_chars_per_chunk
        overlap = self.chunk_overlap

        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)

            if i + chunk_size >= len(content):
                break

        logger.info(f"✂️ Created {len(chunks)} character-based chunks")
        return chunks

    def _merge_chunked_results(self, chunk_results):
        """Merge results from multiple chunks, removing duplicates"""
        all_restaurants = []
        seen_names = set()

        for result in chunk_results:
            if not result or not result.get('edited_results', {}).get('main_list'):
                continue

            restaurants = result['edited_results']['main_list']

            for restaurant in restaurants:
                name = restaurant.get('name', '').lower().strip()

                if name and name not in seen_names:
                    seen_names.add(name)
                    all_restaurants.append(restaurant)
                elif name in seen_names:
                    # Merge information if restaurant already exists
                    existing_restaurant = next(
                        (r for r in all_restaurants if r.get('name', '').lower().strip() == name), 
                        None
                    )
                    if existing_restaurant:
                        # Combine sources
                        existing_sources = set(existing_restaurant.get('sources', []))
                        new_sources = set(restaurant.get('sources', []))
                        existing_restaurant['sources'] = list(existing_sources | new_sources)

                        # Use longer description
                        existing_desc = existing_restaurant.get('description', '')
                        new_desc = restaurant.get('description', '')
                        if len(new_desc) > len(existing_desc):
                            existing_restaurant['description'] = new_desc

        logger.info(f"🔗 Merged chunks into {len(all_restaurants)} unique restaurants")
        return all_restaurants

    @log_function_call
    def edit(self, scraped_results=None, database_restaurants=None, raw_query="", destination="Unknown", 
         content_source=None, processing_mode=None, cleaned_file_path=None, **kwargs):
        """
        CORRECTED: Main editing method with chunking support and AI-driven selection

        Flow:
        1. Receive results (database, hybrid, or cleaned)
        2. Chunk large content if needed
        3. Process with AI (which includes selection via prompt)
        4. Generate follow-up searches
        """
        try:
            # Use raw_query consistently
            query = raw_query or kwargs.get('original_query', '') or ""

            # Determine processing mode FIRST (before routing)
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

            # Route to appropriate processing method (CORRECTED - use internal methods)
            if mode == "hybrid":
                result = self._process_hybrid_content(
                    database_restaurants, 
                    scraped_results, 
                    raw_query,
                    destination, 
                    cleaned_file_path  # Now this will work
                )
            elif mode == "database_only":
                result = self._process_database_restaurants(
                    database_restaurants, 
                    raw_query,
                    destination
                    # No cleaned_file_path needed for database-only
                )
            elif mode == "web_only":
                result = self._process_scraped_content(
                    scraped_results, 
                    raw_query,
                    destination, 
                    cleaned_file_path  # This already works
                )
            else:
                logger.warning(f"⚠️ Unknown processing mode: {mode}")
                return self._fallback_response()

            if result and result.get("edited_results", {}).get("main_list"):
                all_restaurants = result["edited_results"]["main_list"]
                logger.info(f"✅ Final result: {len(all_restaurants)} restaurants")

            return result

        except Exception as e:
            logger.error(f"❌ Error in editor agent: {e}")
            dump_chain_state("editor_error", locals(), error=e)
            return self._fallback_response()

    def _process_hybrid_content(self, database_restaurants, scraped_results, raw_query, destination, cleaned_file_path=None):
        """Process both database and scraped content - with chunking support"""
        logger.info(f"🔄 Processing hybrid content for {destination}")
        logger.info(f"📊 Database restaurants: {len(database_restaurants) if database_restaurants else 0}")
        logger.info(f"📊 Scraped results: {len(scraped_results) if scraped_results else 0}")
        logger.info(f"📊 Cleaned file: {cleaned_file_path}")  # NEW LOG

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

            # Then process scraped content (additional search results) - WITH CHUNKING
            if scraped_results:
                logger.info("🌐 Processing additional web search results with chunking support")
                # UPDATED: Pass cleaned_file_path to _process_scraped_content
                scraped_result = self._process_scraped_content(
                    scraped_results, 
                    raw_query, 
                    destination, 
                    cleaned_file_path  # PASS IT ALONG
                )
                web_restaurants = scraped_result.get('edited_results', {}).get('main_list', [])

                # Mark web restaurants as additional
                for restaurant in web_restaurants:
                    restaurant['_source_type'] = 'web_additional'

                all_restaurants.extend(web_restaurants)
                logger.info(f"✅ Added {len(web_restaurants)} web search restaurants")

            # [Rest of the method remains the same...]

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
        """Process restaurants from AI-matched database - FIXED to preserve sources"""
        try:
            logger.info(f"🗃️ Processing {len(database_restaurants)} restaurants from database for {destination}")

            # Prepare database content for AI processing
            database_content = self._prepare_database_content(database_restaurants)

            if not database_content.strip():
                logger.warning("No substantial database content to process")
                return self._fallback_response()

            # Get AI formatting
            response = self.database_chain.invoke({
                "raw_query": raw_query,
                "destination": destination,
                "database_restaurants": database_content
            })

            result = self._post_process_results(response, "database", destination)

            # Preserve original sources for database restaurants
            processed_restaurants = result.get('edited_results', {}).get('main_list', [])

            for processed_restaurant in processed_restaurants:
                # Find matching original restaurant by name
                name = processed_restaurant.get('name', '').lower().strip()

                for original_restaurant in database_restaurants:
                    original_name = original_restaurant.get('name', '').lower().strip()

                    if name == original_name:
                        # Preserve the original sources
                        original_sources = original_restaurant.get('sources', [])
                        processed_restaurant['sources'] = original_sources

                        logger.info(f"🔍 EDITOR OUTPUT - Restaurant: {processed_restaurant['name']}")
                        logger.info(f"🔍 EDITOR OUTPUT - Sources: {processed_restaurant['sources']}")
                        break

            logger.info(f"✅ Successfully preserved sources for {len(processed_restaurants)} database restaurants")

            return {
                "edited_results": {
                    "main_list": processed_restaurants
                },
            }

        except Exception as e:
            logger.error(f"❌ Error processing database restaurants: {e}")
            try:
                from utils.debug_utils import dump_chain_state
                dump_chain_state("database_processing_error", locals(), error=e)
            except ImportError:
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

            return self._fallback_response()

    def _process_cleaned_file_content(self, cleaned_content, raw_query, destination):
        """
        NEW: Process the well-structured cleaned file from TextCleanerAgent

        This handles the high-quality, structured content that TextCleanerAgent produces
        """
        try:
            logger.info(f"🧹 Processing well-structured cleaned file ({len(cleaned_content)} chars)")

            # Add header for context (similar to existing scraped content)
            header = f"""WELL-STRUCTURED RESTAURANT CONTENT
    Generated by: TextCleanerAgent (individual processing + deduplication)
    Query: {raw_query}
    Destination: {destination}
    Content Quality: High (AI-processed and structured)

    Instructions: Extract restaurant information from this well-structured content.
    This content has been pre-processed for quality and duplicates have been removed.

    {'═' * 80}
    """

            full_content = header + cleaned_content

            # Check if content needs chunking (reuse existing chunking logic)
            if self._needs_chunking(full_content):
                logger.info("📚 Large cleaned file detected - using chunking approach")
                return self._process_cleaned_content_with_chunks(full_content, raw_query, destination)
            else:
                logger.info("📄 Small cleaned file - processing as single chunk")

                # Use scraped content chain (works fine for cleaned content too)
                response = self.scraped_chain.invoke({
                    "raw_query": raw_query,
                    "destination": destination,
                    "scraped_content": full_content  # Chain parameter name stays same
                })

                result = self._post_process_results(response, "cleaned_file", destination)
                logger.info(f"✅ Successfully extracted {len(result['edited_results']['main_list'])} restaurants from cleaned file")
                return result

        except Exception as e:
            logger.error(f"❌ Error processing cleaned file: {e}")
            # Fallback to empty response
            return self._fallback_response()

    def _process_cleaned_content_with_chunks(self, cleaned_content, raw_query, destination):
        """
        NEW: Process large cleaned file content by breaking it into chunks

        Similar to existing _process_scraped_content_with_chunks but for cleaned files
        """
        try:
            # Break content into chunks (reuse existing chunking logic)
            chunks = self._chunk_content(cleaned_content)
            logger.info(f"📚 Processing {len(chunks)} chunks from cleaned file")

            # Process each chunk (reuse existing chunk processing logic)
            chunk_results = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"📄 Processing chunk {i}/{len(chunks)}")

                try:
                    response = self.scraped_chain.invoke({
                        "raw_query": raw_query,
                        "destination": destination,
                        "scraped_content": chunk  # Chain parameter name stays same
                    })

                    chunk_result = self._post_process_results(response, f"cleaned_file_chunk_{i}", destination)

                    if chunk_result.get('edited_results', {}).get('main_list'):
                        chunk_results.append(chunk_result)
                        restaurant_count = len(chunk_result['edited_results']['main_list'])
                        logger.info(f"✅ Chunk {i} processed: {restaurant_count} restaurants found")
                    else:
                        logger.info(f"⚠️ Chunk {i} processed: no restaurants found")

                except Exception as e:
                    logger.error(f"❌ Error processing chunk {i}: {e}")
                    continue

            if not chunk_results:
                logger.warning("❌ No successful chunk results from cleaned file")
                return self._fallback_response()

            # Merge results from all chunks (reuse existing merge logic)
            logger.info(f"🔗 Merging results from {len(chunk_results)} successful chunks")
            merged_restaurants = self._merge_chunked_results(chunk_results)

            result = {
                "edited_results": {
                    "main_list": merged_restaurants
                },                
            }

            logger.info(f"✅ Successfully processed cleaned file chunks: {len(merged_restaurants)} restaurants")
            return result

        except Exception as e:
            logger.error(f"❌ Error processing cleaned file chunks: {e}")
            return self._fallback_response()

    def _process_scraped_content(self, scraped_results, raw_query, destination, cleaned_file_path=None):
        """
        CORRECTED: Process scraped web content with priority for one well-structured cleaned file

        New Priority Order:
        1. One combined cleaned file from TextCleanerAgent (highest quality)
        2. Multiple scraped files as fallback (raw content per URL)
        """
        try:
            logger.info(f"🌐 Processing scraped content for {destination}")

            # PRIORITY 1: Use the well-structured cleaned file from TextCleanerAgent
            if cleaned_file_path and os.path.exists(cleaned_file_path):
                logger.info("🧹 Using well-structured cleaned file from TextCleanerAgent")

                with open(cleaned_file_path, 'r', encoding='utf-8') as f:
                    cleaned_file_content = f.read()

                if cleaned_file_content.strip():
                    logger.info(f"✅ Loaded {len(cleaned_file_content)} characters from cleaned file")

                    # Process the cleaned file (with chunking if needed)
                    return self._process_cleaned_file_content(cleaned_file_content, raw_query, destination)
                else:
                    logger.warning("⚠️ Cleaned file is empty, falling back to individual scraped files")
            else:
                logger.info("🔄 No cleaned file available, using individual scraped files as fallback")

            # FALLBACK: Use individual scraped files (current logic)
            logger.info(f"📄 Falling back to processing {len(scraped_results)} individual scraped articles")

            # Prepare content from individual scraped results (existing logic)
            scraped_content = self._prepare_scraped_content_from_individual_files(scraped_results)

            if not scraped_content.strip():
                logger.warning("No substantial scraped content to process")
                return self._fallback_response()

            # Check if content needs chunking (existing logic)
            if self._needs_chunking(scraped_content):
                logger.info("📚 Large scraped content detected - using chunking approach")
                return self._process_scraped_content_with_chunks(scraped_content, raw_query, destination)
            else:
                logger.info("📄 Small scraped content - processing as single chunk")
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
            return self._fallback_response()

    def _process_scraped_content_with_chunks(self, scraped_content, raw_query, destination):
        """Process large scraped content by breaking it into chunks"""
        try:
            # Break content into chunks
            chunks = self._chunk_content(scraped_content)
            logger.info(f"📚 Processing {len(chunks)} chunks")

            # Process each chunk
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.info(f"🔍 Processing chunk {i+1}/{len(chunks)}")

                try:
                    response = self.scraped_chain.invoke({
                        "raw_query": raw_query,
                        "destination": destination,
                        "scraped_content": chunk
                    })

                    result = self._post_process_results(response, f"scraped_chunk_{i+1}", destination)
                    chunk_results.append(result)

                except Exception as e:
                    logger.warning(f"⚠️ Error processing chunk {i+1}: {e}")
                    continue

            if not chunk_results:
                logger.warning("⚠️ No successful chunk processing results")
                return self._fallback_response()

            # Merge results from all chunks
            merged_restaurants = self._merge_chunked_results(chunk_results)

            logger.info(f"✅ Successfully processed {len(chunks)} chunks, extracted {len(merged_restaurants)} restaurants")

            return {
                "edited_results": {"main_list": merged_restaurants},
                "processing_notes": {
                    "mode": "scraped_chunked",
                    "chunk_count": len(chunks),
                    "successful_chunks": len(chunk_results),
                    "final_count": len(merged_restaurants)
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in chunked processing: {e}")
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

    def _prepare_scraped_content_from_individual_files(self, scraped_results):
        """
        RENAMED: Prepare content from individual scraped files (fallback method)

        This is the existing _prepare_scraped_content logic, renamed for clarity.
        Used as fallback when no cleaned file is available.
        """
        try:
            logger.info(f"📝 Preparing content from {len(scraped_results)} individual scraped files (fallback)")

            content_pieces = []
            raw_content_count = 0

            for i, result in enumerate(scraped_results, 1):
                url = result.get('url', f'Unknown URL {i}')
                title = result.get('title', 'No title')

                # Get raw scraped content (no cleaned_content expected in fallback mode)
                content = result.get('scraped_content', '') or result.get('content', '')

                # Skip if no content at all
                if not content or len(content.strip()) < 50:
                    logger.warning(f"⚠️ Skipping {url} - insufficient content")
                    continue

                raw_content_count += 1

                # Format content piece with source information
                content_piece = f"""
    SOURCE {i}: {url}
    TITLE: {title}
    CONTENT_TYPE: RAW_FALLBACK
    DOMAIN: {self._extract_domain(url)}
    ───────────────────────────────────────
    {content.strip()}
    ═══════════════════════════════════════
    """
                content_pieces.append(content_piece)

            # Log content statistics
            logger.info(f"📊 Fallback content stats:")
            logger.info(f"   ⚠️ Raw content sources used: {raw_content_count}")
            logger.info(f"   📄 Total pieces prepared: {len(content_pieces)}")

            if raw_content_count > 0:
                logger.warning("⚠️ WARNING: Using raw scraped content as fallback - no cleaned file available")

            # Combine all content pieces
            combined_content = "\n".join(content_pieces)

            if not combined_content.strip():
                logger.warning("⚠️ No substantial content prepared from scraped results")
                return ""

            # Add header with processing information
            header = f"""RESTAURANT CONTENT FOR PROCESSING (FALLBACK MODE)
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Total Sources: {len(content_pieces)}
    Content Quality: Raw (no TextCleanerAgent processing available)

    Instructions: Extract restaurant information from the following content.
    Focus on restaurant names, locations, cuisines, descriptions, and key details.

    {'═' * 80}
    """

            final_content = header + combined_content

            logger.info(f"✅ Prepared {len(final_content)} characters of fallback content")
            return final_content

        except Exception as e:
            logger.error(f"❌ Error preparing fallback scraped content: {e}")
            return ""

    def _post_process_results(self, ai_output, source_type, destination):
        """
        Process AI output for both database and scraped results
        Handle AIMessage objects correctly
        """
        try:
            logger.info(f"🔍 Processing AI output from {source_type}")

            # Handle both AIMessage objects and direct strings
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

            # Better handling of markdown code blocks
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

            return {
                "edited_results": {
                    "main_list": cleaned_restaurants
                },
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI output as JSON: {e}")
            content_preview = content[:500] if 'content' in locals() and content else 'Unable to extract content'
            logger.error(f"Raw AI output: {content_preview}...")
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
            "chunking_enabled": True,
            "dual_api_keys_supported": True
        }