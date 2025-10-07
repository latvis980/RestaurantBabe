# agents/text_cleaner_agent.py
"""
OPTIMIZED: Individual File Processing Text Cleaner Agent with CONCURRENT PROCESSING
NEW: Processes 2-3 files concurrently for 2-3x speed improvement

WORKFLOW:
1. Process each scraped URL individually with CONCURRENT execution (NEW!)
2. Save each cleaned result as individual file with URL metadata  
3. Combine all individual files into master file
4. Deduplicate restaurants (combine entries for same restaurant)
5. Track ALL sources for each restaurant (comma-separated URLs)
6. Upload final combined file to Supabase

OPTIMIZATION: Uses asyncio.Semaphore to process multiple files concurrently
"""

import re
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime
import os
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class TextCleanerAgent:
    """
    OPTIMIZED: Individual file processing with CONCURRENT PROCESSING + restaurant deduplication
    """

    def __init__(self, config, model_override=None):
        self.config = config

        # Model selection with INCREASED token limits
        self.current_model_type = model_override or config.MODEL_STRATEGY.get('content_cleaning', 'openai')
        self.model = self._initialize_model(self.current_model_type)

        # OPTIMIZATION: Get concurrent processing settings from config
        self.max_concurrent_files = getattr(
            self.config.INDIVIDUAL_FILE_PROCESSING, 
            'max_concurrent_files', 
            3  # Default to 3 concurrent files
        )
        self.concurrent_enabled = getattr(
            self.config.INDIVIDUAL_FILE_PROCESSING,
            'concurrent_processing_enabled',
            True  # Default to enabled
        )

        # Enhanced stats tracking for individual processing
        self.stats = {
            "files_processed": 0,
            "individual_files_processed": 0,
            "rtf_files_processed": 0,
            "restaurants_extracted": 0,
            "restaurants_deduplicated": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "current_model": self.current_model_type,
            "individual_files_saved": 0,
            "combined_files_saved": 0,
            "supabase_uploads": 0,
            "concurrent_batches": 0,  # NEW: Track concurrent batches
            "max_concurrent_files": self.max_concurrent_files  # NEW: Track concurrency setting
        }

        # Create directories for individual and combined files
        self._setup_directories()

        logger.info(f"✅ Text Cleaner initialized with CONCURRENT processing: {self.max_concurrent_files} files at once")

    def _initialize_model(self, model_type: str):
        """Initialize AI model with INCREASED token limits"""
        logger.info(f"🤖 Initializing OPTIMIZED Text Cleaner with {model_type} model")

        if model_type == 'openai':
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                temperature=0.1,
                model="gpt-4o-mini",
                max_tokens=16000,  # INCREASED: was 8000, now 16000
                max_retries=3
            )
        elif model_type == 'deepseek':
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                temperature=0.1,
                openai_api_base="https://api.deepseek.com",
                openai_api_key=self.config.DEEPSEEK_API_KEY,
                model="deepseek-chat",
                max_tokens=16000,  # INCREASED: was 8000, now 16000
                max_retries=3
            )
        elif model_type == 'claude':
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                temperature=0.1,
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000,  # INCREASED: was 4000, now 8000
                max_retries=3
            )
        else:
            logger.warning(f"Unknown model type: {model_type}, defaulting to OpenAI")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                temperature=0.1,
                model="gpt-4o-mini",
                max_tokens=16000
            )

    def _setup_directories(self):
        """Create necessary directories for individual and combined files"""
        try:
            individual_dir = Path(self.config.INDIVIDUAL_FILE_PROCESSING['individual_files_directory'])
            combined_dir = Path(self.config.INDIVIDUAL_FILE_PROCESSING['combined_files_directory'])

            individual_dir.mkdir(parents=True, exist_ok=True)
            combined_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"📁 Created directories: {individual_dir}, {combined_dir}")
        except Exception as e:
            logger.error(f"❌ Error creating directories: {e}")

    # NEW METHOD: Main entry point called by orchestrator
    async def process_scraped_results_individually(self, scraped_results: List[Dict[str, Any]], query: str = "") -> str:
        """
        OPTIMIZED: Process each scraped result individually with CONCURRENT execution

        Returns: path to final combined TXT file
        """
        logger.info(f"🔄 OPTIMIZED: Starting CONCURRENT processing for {len(scraped_results)} URLs...")
        logger.info(f"⚡ Concurrency setting: {self.max_concurrent_files} files at once")

        # Step 1: Process each URL individually with CONCURRENCY (OPTIMIZED!)
        individual_results = await self._process_urls_individually_concurrent(scraped_results, query)

        if not individual_results:
            logger.warning("⚠️ No individual results to combine")
            return ""

        # Step 2: Combine all individual results
        combined_restaurants = self._combine_individual_results(individual_results)

        # Step 3: Deduplicate restaurants and combine sources
        deduplicated_restaurants = self._deduplicate_restaurants(combined_restaurants)

        # Step 4: Create final combined file
        final_file_path = self._create_final_combined_file(deduplicated_restaurants, query, scraped_results)

        # Step 5: Upload final combined file to Supabase
        await self._upload_final_file_to_supabase(final_file_path, deduplicated_restaurants, query)

        # Step 6: Update stats
        self._update_final_stats(individual_results, combined_restaurants, deduplicated_restaurants)

        logger.info(f"✅ OPTIMIZED: CONCURRENT processing complete. Processed {len(individual_results)} files")
        return final_file_path

    async def _process_urls_individually_concurrent(self, scraped_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Process URLs with concurrent execution using asyncio.Semaphore
        This is the KEY optimization - processes multiple files at the same time!
        """
        if not self.concurrent_enabled:
            logger.info("⚠️ Concurrent processing disabled, falling back to sequential")
            return await self._process_urls_individually_sequential(scraped_results, query)

        logger.info(f"⚡ CONCURRENT PROCESSING: Processing {len(scraped_results)} files with max {self.max_concurrent_files} concurrent")

        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.max_concurrent_files)

        # Create tasks for all files
        tasks = []
        for idx, result in enumerate(scraped_results, 1):
            task = self._process_single_file_with_semaphore(semaphore, result, idx, len(scraped_results), query)
            tasks.append(task)

        # Execute all tasks concurrently and gather results
        logger.info(f"🚀 Starting concurrent execution of {len(tasks)} tasks...")
        start_time = asyncio.get_event_loop().time()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = asyncio.get_event_loop().time() - start_time

        # Filter out None results and exceptions
        individual_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Task {i+1} failed with exception: {result}")
            elif result is not None:
                individual_results.append(result)

        self.stats["concurrent_batches"] += 1

        logger.info(f"🏁 CONCURRENT processing complete: {len(individual_results)}/{len(scraped_results)} files processed in {elapsed:.2f}s")
        logger.info(f"⚡ Average time per file: {elapsed/len(scraped_results):.2f}s (with concurrency)")

        return individual_results

    async def _process_single_file_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        result: Dict[str, Any], 
        idx: int, 
        total: int,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single file with semaphore control
        Semaphore ensures we don't exceed max_concurrent_files
        """
        async with semaphore:  # This line controls concurrency!
            try:
                url = result.get('url', f'source_{idx}')
                logger.info(f"🧹 [{idx}/{total}] Processing: {urlparse(url).netloc}")

                # Determine content format 
                content_format = 'rtf' if result.get('scraping_method') == 'human_mimic_rtf' else 'text'

                # Get content
                content = result.get('content', '') or result.get('scraped_content', '') or result.get('text_content', '')

                if not content:
                    logger.warning(f"⚠️ [{idx}/{total}] No content found for {url}")
                    return None

                # Clean individual source with INCREASED token limits
                cleaned_result = await self._clean_individual_source(content, url, content_format)

                if cleaned_result and cleaned_result.get('restaurants'):
                    # Save individual file
                    individual_file_path = self._save_individual_file(cleaned_result, idx, query)
                    cleaned_result['individual_file_path'] = individual_file_path

                    restaurant_count = len(cleaned_result['restaurants'])
                    logger.info(f"✅ [{idx}/{total}] Processed: {restaurant_count} restaurants found")
                    return cleaned_result
                else:
                    logger.warning(f"⚠️ [{idx}/{total}] No valid restaurants found")
                    return None

            except Exception as e:
                logger.error(f"❌ [{idx}/{total}] Error processing file: {e}")
                return None

    async def _process_urls_individually_sequential(self, scraped_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        FALLBACK: Sequential processing (original method)
        Used only if concurrent processing is disabled
        """
        logger.info(f"🔄 SEQUENTIAL processing for {len(scraped_results)} URLs...")
        individual_results = []

        for idx, result in enumerate(scraped_results, 1):
            try:
                url = result.get('url', f'source_{idx}')
                logger.info(f"🧹 Processing individual file {idx}/{len(scraped_results)}: {urlparse(url).netloc}")

                # Determine content format 
                content_format = 'rtf' if result.get('scraping_method') == 'human_mimic_rtf' else 'text'

                # Get content
                content = result.get('content', '') or result.get('scraped_content', '') or result.get('text_content', '')

                if not content:
                    logger.warning(f"⚠️ No content found for {url}")
                    continue

                # Clean individual source with INCREASED token limits
                cleaned_result = await self._clean_individual_source(content, url, content_format)

                if cleaned_result and cleaned_result.get('restaurants'):
                    # Save individual file
                    individual_file_path = self._save_individual_file(cleaned_result, idx, query)
                    cleaned_result['individual_file_path'] = individual_file_path
                    individual_results.append(cleaned_result)

                    restaurant_count = len(cleaned_result['restaurants'])
                    logger.info(f"✅ Individual file {idx} processed: {restaurant_count} restaurants found")
                else:
                    logger.warning(f"⚠️ Individual file {idx} processed: no valid restaurants found")

            except Exception as e:
                logger.error(f"❌ Error processing individual file {idx}: {e}")
                continue

        logger.info(f"🏁 Sequential processing complete: {len(individual_results)}/{len(scraped_results)} files processed successfully")
        return individual_results

    async def _clean_individual_source(self, content: str, source_url: str, content_format: str = 'text') -> Optional[Dict[str, Any]]:
        """Clean individual source content with INCREASED token limits"""
        start_time = asyncio.get_event_loop().time()

        try:
            # Convert RTF to text if needed
            if content_format == 'rtf':
                content = self._rtf_to_text(content)
                self.stats["rtf_files_processed"] += 1

            # Truncate content if too long (prevent token limits)
            max_chars = 50000  # INCREASED: was 30000, now 50000 to handle more content
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[Content truncated due to length...]"
                logger.info(f"⚠️ Content truncated for {urlparse(source_url).netloc}: {len(content)} chars")

            # Enhanced cleaning prompt for individual processing
            enhanced_prompt = f"""
Extract restaurant information from this web content. Focus on finding complete restaurant details.

SOURCE URL: {source_url}
CONTENT FORMAT: {content_format}

CONTENT:
{content}

INSTRUCTIONS:
1. Extract ONLY restaurants (not cafes, bars, or food trucks unless specifically mentioned as restaurants)
2. For each restaurant, provide:
   - Name: Full restaurant name
   - City: City where the restaurant is located
   - Country: Country where the restaurant is located
   - Description: Brief description (max 2-3 sentences)
   - Address: Full address if available, otherwise "Not specified"

3. Return as JSON array with this exact format:
[
  {{
    "Name": "Restaurant Name",
    "City": "City Name",
    "Country": "Country Name", 
    "Description": "Brief description of the restaurant and cuisine",
    "Address": "Full address or 'Not specified'"
  }}
]

4. QUALITY REQUIREMENTS:
   - Only include restaurants with clear names
   - Ensure city/country are properly identified
   - Descriptions should be informative but concise
   - Skip entries that are unclear or incomplete

Return ONLY the JSON array, no other text.
"""

            # Process with AI model
            response = await self.model.ainvoke([{"role": "user", "content": enhanced_prompt}])
            response_text = response.content.strip()

            # Parse JSON response
            try:
                # Clean common JSON formatting issues
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                restaurants = json.loads(response_text)

                if not isinstance(restaurants, list):
                    logger.warning(f"⚠️ Invalid response format for {urlparse(source_url).netloc}")
                    return None

                # Add source URL to each restaurant
                for restaurant in restaurants:
                    restaurant['Source'] = source_url

                processing_time = asyncio.get_event_loop().time() - start_time
                self.stats["total_processing_time"] += processing_time
                self.stats["files_processed"] += 1
                self.stats["restaurants_extracted"] += len(restaurants)

                logger.debug(f"🧹 Individual cleaning: {len(restaurants)} restaurants, {processing_time:.2f}s")

                return {
                    'source_url': source_url,
                    'restaurants': restaurants,
                    'processing_time': processing_time,
                    'content_format': content_format
                }

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing error for {urlparse(source_url).netloc}: {e}")
                logger.debug(f"Response was: {response_text[:200]}...")
                return None

        except Exception as e:
            logger.error(f"❌ Error cleaning individual source {urlparse(source_url).netloc}: {e}")
            return None

    def _save_individual_file(self, cleaned_result: Dict[str, Any], file_index: int, query: str) -> str:
        """Save individual cleaned result to file"""
        try:
            # Create safe filename
            safe_query = re.sub(r'[^\w\s-]', '', query)[:30]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = urlparse(cleaned_result['source_url']).netloc.replace('.', '_')
            filename = f"individual_{file_index:03d}_{safe_query}_{domain}_{timestamp}.txt"

            # File path
            individual_dir = Path(self.config.INDIVIDUAL_FILE_PROCESSING['individual_files_directory'])
            filepath = individual_dir / filename

            # Format content for file
            file_content = f"INDIVIDUAL FILE: {file_index}\n"
            file_content += f"SOURCE URL: {cleaned_result['source_url']}\n"
            file_content += f"PROCESSED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            file_content += f"RESTAURANTS FOUND: {len(cleaned_result['restaurants'])}\n"
            file_content += "=" * 80 + "\n\n"

            # Add restaurant entries
            for restaurant in cleaned_result['restaurants']:
                file_content += f"Name: {restaurant.get('Name', 'N/A')}\n"
                file_content += f"City: {restaurant.get('City', 'N/A')}\n"
                file_content += f"Country: {restaurant.get('Country', 'N/A')}\n"
                file_content += f"Description: {restaurant.get('Description', 'N/A')}\n"
                file_content += f"Address: {restaurant.get('Address', 'Not specified')}\n"
                file_content += f"Source: {restaurant.get('Source', cleaned_result['source_url'])}\n\n"

            # Save file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(file_content)

            self.stats["individual_files_saved"] += 1
            logger.debug(f"💾 Saved individual file: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error saving individual file: {e}")
            return ""

    def _combine_individual_results(self, individual_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine all individual results into single list of restaurants"""
        logger.info(f"🔗 Combining {len(individual_results)} individual results...")

        all_restaurants = []

        for result in individual_results:
            restaurants = result.get('restaurants', [])
            all_restaurants.extend(restaurants)

        logger.info(f"✅ Combined {len(all_restaurants)} total restaurants from {len(individual_results)} sources")
        return all_restaurants

    def _deduplicate_restaurants(self, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate restaurants by combining entries for the same venue
        Preserves ALL sources (comma-separated URLs)
        """
        if not restaurants:
            return []

        logger.info(f"🔍 Deduplicating {len(restaurants)} restaurants...")

        # Get deduplication settings
        name_threshold = self.config.RESTAURANT_DEDUPLICATION['name_similarity_threshold']
        address_threshold = self.config.RESTAURANT_DEDUPLICATION['address_similarity_threshold']

        deduplicated = []
        processed_indices = set()

        for i, restaurant in enumerate(restaurants):
            if i in processed_indices:
                continue

            # Start with this restaurant
            merged_restaurant = restaurant.copy()
            sources = [restaurant.get('Source', '')]
            descriptions = [restaurant.get('Description', '')]

            # Find duplicates
            for j, other_restaurant in enumerate(restaurants[i+1:], start=i+1):
                if j in processed_indices:
                    continue

                # Check name similarity
                name_similarity = self._calculate_similarity(
                    restaurant.get('Name', '').lower(),
                    other_restaurant.get('Name', '').lower()
                )

                # Check address similarity (if both have addresses)
                address1 = restaurant.get('Address', '')
                address2 = other_restaurant.get('Address', '')
                address_similarity = 0.0
                if address1 and address2 and address1 != 'Not specified' and address2 != 'Not specified':
                    address_similarity = self._calculate_similarity(address1.lower(), address2.lower())

                # Consider duplicate if name is very similar OR both name and address are similar
                is_duplicate = (
                    name_similarity >= name_threshold or
                    (name_similarity >= 0.70 and address_similarity >= address_threshold)
                )

                if is_duplicate:
                    # Merge sources
                    other_source = other_restaurant.get('Source', '')
                    if other_source and other_source not in sources:
                        sources.append(other_source)

                    # Merge descriptions
                    other_desc = other_restaurant.get('Description', '')
                    if other_desc and other_desc not in descriptions:
                        descriptions.append(other_desc)

                    # Mark as processed
                    processed_indices.add(j)
                    self.stats["restaurants_deduplicated"] += 1

            # Combine sources (comma-separated, max 5)
            max_sources = self.config.RESTAURANT_DEDUPLICATION.get('max_sources_per_restaurant', 5)
            merged_restaurant['Source'] = ', '.join(sources[:max_sources])

            # Combine descriptions if enabled
            if self.config.RESTAURANT_DEDUPLICATION['combine_descriptions'] and len(descriptions) > 1:
                # Use the longest/most detailed description
                merged_restaurant['Description'] = max(descriptions, key=len)

            deduplicated.append(merged_restaurant)
            processed_indices.add(i)

        logger.info(f"✅ Deduplication complete: {len(restaurants)} → {len(deduplicated)} unique restaurants")
        return deduplicated

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings"""
        return SequenceMatcher(None, str1, str2).ratio()

    def _create_final_combined_file(self, restaurants: List[Dict[str, Any]], query: str, scraped_results: List[Dict[str, Any]]) -> str:
        """Create final combined TXT file with all deduplicated restaurants"""
        try:
            # Create safe filename
            safe_query = re.sub(r'[^\w\s-]', '', query)[:30]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Extract city from query or restaurants
            city = self._extract_city_from_query(query) or self._extract_city_from_restaurants(restaurants)
            country = self._extract_country_from_restaurants(restaurants)

            filename = f"combined_{safe_query}_{city}_{country}_{timestamp}.txt"

            # File path
            combined_dir = Path(self.config.INDIVIDUAL_FILE_PROCESSING['combined_files_directory'])
            filepath = combined_dir / filename

            # Format header
            file_content = f"COMBINED RESTAURANT FILE\n"
            file_content += f"QUERY: {query}\n"
            file_content += f"CITY: {city}\n"
            file_content += f"COUNTRY: {country}\n"
            file_content += f"CREATED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            file_content += f"SOURCES PROCESSED: {len(scraped_results)}\n"
            file_content += f"UNIQUE RESTAURANTS: {len(restaurants)}\n"
            file_content += "=" * 80 + "\n\n"

            # Add restaurant entries
            for i, restaurant in enumerate(restaurants, 1):
                name = restaurant.get('Name', 'N/A')
                city_name = restaurant.get('City', 'N/A')
                country_name = restaurant.get('Country', 'N/A')
                desc = restaurant.get('Description', 'N/A')
                address = restaurant.get('Address', 'Not specified')
                sources = restaurant.get('Source', 'N/A')

                file_content += f"Restaurant {i}: {name}\n"
                file_content += f"Location: {city_name}, {country_name}\n"
                file_content += f"Address: {address}\n"
                file_content += f"Description: {desc}\n"
                file_content += f"Sources: {sources}\n\n"  # ALL sources, comma-separated

            # Save file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(file_content)

            self.stats["combined_files_saved"] += 1
            logger.info(f"💾 Final combined file saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error creating final combined file: {e}")
            return ""

    def _extract_city_from_query(self, query: str) -> str:
        """Extract city from query string"""
        # Simple extraction - look for common patterns
        query_lower = query.lower()

        # Pattern: "restaurants in CITY"
        if ' in ' in query_lower:
            parts = query_lower.split(' in ')
            if len(parts) > 1:
                potential_city = parts[1].split()[0]  # First word after "in"
                return potential_city.capitalize()

        # Pattern: "CITY restaurants"
        if 'restaurants' in query_lower or 'restaurant' in query_lower:
            parts = query_lower.split('restaurants')
            if len(parts) > 0 and parts[0].strip():
                potential_city = parts[0].strip().split()[-1]  # Last word before "restaurants"
                return potential_city.capitalize()

            parts = query_lower.split('restaurant')
            if len(parts) > 0 and parts[0].strip():
                potential_city = parts[0].strip().split()[-1]  # Last word before "restaurant"
                return potential_city.capitalize()

        # Pattern: "best/top places CITY"
        for indicator in ['best ', 'top ', 'places ']:
            if indicator in query_lower:
                parts = query_lower.split(indicator)
                if len(parts) > 1:
                    potential_city = parts[1].split()[0]  # First word after indicator
                    return potential_city.capitalize()

        return 'unknown'

    def _extract_country_from_restaurants(self, restaurants: List[Dict[str, Any]]) -> str:
        """Extract most common country from restaurants"""
        countries = {}
        for restaurant in restaurants:
            country = restaurant.get('Country', '').strip()
            if country and country.lower() not in ['n/a', 'not specified', 'unknown']:
                countries[country] = countries.get(country, 0) + 1

        if countries:
            # Return most common country
            return max(countries.items(), key=lambda x: x[1])[0]

        return 'unknown'

    def _extract_city_from_restaurants(self, restaurants: List[Dict[str, Any]]) -> str:
        """Extract most common city from restaurants"""
        cities = {}
        for restaurant in restaurants:
            city = restaurant.get('City', '').strip()
            if city and city.lower() not in ['n/a', 'not specified', 'unknown']:
                cities[city] = cities.get(city, 0) + 1

        if cities:
            # Return most common city
            return max(cities.items(), key=lambda x: x[1])[0]

        return 'unknown'

    async def _upload_final_file_to_supabase(self, final_file_path: str, restaurants: List[Dict[str, Any]], query: str) -> bool:
        """
        Upload the final combined file to Supabase storage with RB naming convention

        Args:
            final_file_path: Path to the local final combined file
            restaurants: List of deduplicated restaurants
            query: Original search query

        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            if not final_file_path or not os.path.exists(final_file_path):
                logger.warning("⚠️ No final file to upload to Supabase")
                return False

            # Read the final combined file content
            with open(final_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            if not file_content.strip():
                logger.warning("⚠️ Final file is empty, skipping Supabase upload")
                return False

            # Extract city from restaurants for metadata
            cities = set()
            for restaurant in restaurants:
                city = restaurant.get('City', '').strip()
                if city and city.lower() not in ['n/a', 'not specified', 'unknown']:
                    cities.add(city)

            # Use first city found, or extract from query, or default to 'unknown'
            primary_city = next(iter(cities)) if cities else self._extract_city_from_query(query)

            # Prepare metadata for Supabase storage with cleanedRB naming
            metadata = {
                'city': primary_city,
                'country': self._extract_country_from_restaurants(restaurants),
                'query': query,
                'restaurants_count': len(restaurants),
                'upload_source': 'text_cleaner_agent',
                'processing_method': 'individual_file_processing_with_deduplication',
                'timestamp': datetime.now().isoformat()
            }

            # Import and use the Supabase storage utility with RB naming convention
            from utils.supabase_storage import upload_content_to_bucket

            logger.info(f"📤 Uploading final combined file to Supabase with cleanedRB naming...")
            logger.info(f"   📊 Content size: {len(file_content)} characters")
            logger.info(f"   🍽️ Restaurants: {len(restaurants)}")
            logger.info(f"   🏙️ Primary city: {primary_city}")

            # Upload to Supabase with cleanedRB_ prefix naming convention
            success, uploaded_file_path = upload_content_to_bucket(
                content=file_content,
                metadata=metadata,
                file_type="txt"
            )

            if success:
                logger.info(f"✅ Successfully uploaded to Supabase with RB suffix: {uploaded_file_path}")
                self.stats["supabase_uploads"] += 1
                return True
            else:
                logger.error("❌ Failed to upload to Supabase")
                return False

        except Exception as e:
            logger.error(f"❌ Error uploading to Supabase: {e}")
            return False

    def _update_final_stats(self, individual_results: List[Dict], combined_restaurants: List[Dict], deduplicated_restaurants: List[Dict]):
        """Update final processing statistics"""
        self.stats["avg_processing_time"] = (
            self.stats["total_processing_time"] / max(self.stats["files_processed"], 1)
        )

        logger.info("📊 OPTIMIZED PROCESSING STATS:")
        logger.info(f"   ⚡ Concurrent processing: ENABLED ({self.max_concurrent_files} files at once)")
        logger.info(f"   🔢 URLs processed: {len(individual_results)}")
        logger.info(f"   🍽️ Total restaurants extracted: {len(combined_restaurants)}")
        logger.info(f"   🔗 Unique restaurants after deduplication: {len(deduplicated_restaurants)}")
        logger.info(f"   🔄 Restaurants merged: {self.stats['restaurants_deduplicated']}")
        logger.info(f"   ⏱️ Average time per URL: {self.stats['avg_processing_time']:.2f}s")
        logger.info(f"   🤖 Model used: {self.current_model_type}")
        logger.info(f"   📤 Supabase uploads: {self.stats['supabase_uploads']}")
        logger.info(f"   🚀 Concurrent batches: {self.stats['concurrent_batches']}")

    # RTF conversion and utility methods (unchanged)
    def _rtf_to_text(self, rtf_content: str) -> str:
        """Convert RTF content to clean text while preserving important structure"""
        if not rtf_content or not rtf_content.startswith('{\\rtf'):
            return rtf_content

        try:
            text = rtf_content
            text = re.sub(r'^{\s*\\rtf[^{]*?{[^}]*?}[^}]*?}', '', text, flags=re.DOTALL)
            text = re.sub(r'\\b\s*(.*?)\s*\\b0', r'**\1**', text, flags=re.DOTALL)
            text = re.sub(r'\\i\s*(.*?)\s*\\i0', r'*\1*', text, flags=re.DOTALL)
            text = re.sub(r'\\u\d+\s*', '', text)
            text = re.sub(r'\\[a-z]+\d*\s*', ' ', text)
            text = re.sub(r'[{}]', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            return text

        except Exception as e:
            logger.error(f"❌ Error converting RTF to text: {e}")
            return rtf_content

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return self.stats.copy()