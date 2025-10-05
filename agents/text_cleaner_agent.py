# agents/text_cleaner_agent.py
"""
REFACTORED: Individual File Processing Text Cleaner Agent
Processes each scraped URL individually, then combines with deduplication

NEW WORKFLOW:
1. Process each scraped URL individually with increased token limits
2. Save each cleaned result as individual file with URL metadata  
3. Combine all individual files into master file
4. Deduplicate restaurants (combine entries for same restaurant)
5. Track ALL sources for each restaurant (comma-separated URLs)
6. Upload final combined file to Supabase (ADDED!)
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
    REFACTORED: Individual file processing with restaurant deduplication + Supabase upload
    """

    def __init__(self, config, model_override=None):
        self.config = config

        # Model selection with INCREASED token limits
        self.current_model_type = model_override or config.MODEL_STRATEGY.get('content_cleaning', 'openai')
        self.model = self._initialize_model(self.current_model_type)

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
            "supabase_uploads": 0  # NEW: track uploads
        }

        # Create directories for individual and combined files
        self._setup_directories()

    def _initialize_model(self, model_type: str):
        """Initialize AI model with INCREASED token limits"""
        logger.info(f"🤖 Initializing REFACTORED Text Cleaner with {model_type} model")

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
        NEW METHOD: Process each scraped result individually, then combine with deduplication

        Returns: path to final combined TXT file
        """
        logger.info(f"🔄 REFACTORED: Starting individual processing for {len(scraped_results)} URLs...")

        # Step 1: Process each URL individually
        individual_results = await self._process_urls_individually(scraped_results, query)

        if not individual_results:
            logger.warning("⚠️ No individual results to combine")
            return ""

        # Step 2: Combine all individual results
        combined_restaurants = self._combine_individual_results(individual_results)

        # Step 3: Deduplicate restaurants and combine sources
        deduplicated_restaurants = self._deduplicate_restaurants(combined_restaurants)

        # Step 4: Create final combined file
        final_file_path = self._create_final_combined_file(deduplicated_restaurants, query, scraped_results)

        # Step 5: Upload final combined file to Supabase (NEW!)
        await self._upload_final_file_to_supabase(final_file_path, deduplicated_restaurants, query)

        # Step 6: Update stats
        self._update_final_stats(individual_results, combined_restaurants, deduplicated_restaurants)

        logger.info(f"✅ REFACTORED: Individual processing complete. Final file: {final_file_path}")
        return final_file_path

    async def _upload_final_file_to_supabase(self, final_file_path: str, restaurants: List[Dict[str, Any]], query: str) -> bool:
        """
        NEW METHOD: Upload the final combined file to Supabase storage

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

            # Prepare metadata for Supabase storage
            metadata = {
                'city': primary_city,
                'country': self._extract_country_from_restaurants(restaurants),
                'query': query,
                'restaurants_count': len(restaurants),
                'upload_source': 'text_cleaner_agent',
                'processing_method': 'individual_file_processing_with_deduplication',
                'timestamp': datetime.now().isoformat()
            }

            # Import and use the Supabase storage utility
            from utils.supabase_storage import upload_content_to_bucket

            logger.info(f"📤 Uploading final combined file to Supabase...")
            logger.info(f"   📊 Content size: {len(file_content)} characters")
            logger.info(f"   🍽️ Restaurants: {len(restaurants)}")
            logger.info(f"   🏙️ Primary city: {primary_city}")

            # Upload to Supabase with RB naming convention
            success, uploaded_file_path = upload_content_to_bucket(
                content=file_content,
                metadata=metadata,
                file_type="txt"
            )

            if success:
                logger.info(f"✅ Successfully uploaded to Supabase: {uploaded_file_path}")
                self.stats["supabase_uploads"] += 1
                return True
            else:
                logger.error("❌ Failed to upload to Supabase")
                return False

        except Exception as e:
            logger.error(f"❌ Error uploading to Supabase: {e}")
            return False

    def _extract_city_from_query(self, query: str) -> str:
        """Extract city name from the original query as fallback"""
        if not query:
            return 'unknown'

        # Simple city extraction - look for common patterns
        # This is a basic implementation, you could make it more sophisticated
        query_lower = query.lower()

        # Common city patterns in restaurant queries
        city_indicators = ['in ', 'at ', 'near ', 'around ']
        for indicator in city_indicators:
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

    async def _process_urls_individually(self, scraped_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Process each URL individually with increased token limits"""
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

        logger.info(f"🏁 Individual processing complete: {len(individual_results)}/{len(scraped_results)} files processed successfully")
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

        logger.info(f"📊 Combined total: {len(all_restaurants)} restaurants from all sources")
        return all_restaurants

    def _deduplicate_restaurants(self, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate restaurants and combine sources"""
        logger.info(f"🔍 Starting deduplication of {len(restaurants)} restaurants...")

        if not restaurants:
            return []

        deduplicated = []
        processed_names = set()

        for restaurant in restaurants:
            name = restaurant.get('Name', '').strip()
            city = restaurant.get('City', '').strip()

            if not name or not city:
                continue

            # Create unique key for comparison
            unique_key = f"{name.lower()}|{city.lower()}"

            # Check for exact matches first
            if unique_key in processed_names:
                # Find existing restaurant and merge sources
                for existing in deduplicated:
                    if (existing.get('Name', '').lower() == name.lower() and 
                        existing.get('City', '').lower() == city.lower()):

                        # Combine sources
                        existing_sources = existing.get('Source', '').split(', ')
                        new_source = restaurant.get('Source', '')
                        if new_source and new_source not in existing_sources:
                            existing['Source'] = ', '.join(existing_sources + [new_source])

                        # Merge other details if they're better
                        self._merge_restaurant_details(existing, restaurant)
                        self.stats["restaurants_deduplicated"] += 1
                        break
                continue

            # Check for fuzzy matches (similar names)
            similar_found = False
            for existing in deduplicated:
                existing_name = existing.get('Name', '')
                existing_city = existing.get('City', '')

                # Only compare restaurants in same city
                if existing_city.lower() != city.lower():
                    continue

                # Calculate similarity
                similarity = SequenceMatcher(None, name.lower(), existing_name.lower()).ratio()

                if similarity > 0.85:  # 85% similarity threshold
                    # Merge with existing
                    existing_sources = existing.get('Source', '').split(', ')
                    new_source = restaurant.get('Source', '')
                    if new_source and new_source not in existing_sources:
                        existing['Source'] = ', '.join(existing_sources + [new_source])

                    self._merge_restaurant_details(existing, restaurant)
                    self.stats["restaurants_deduplicated"] += 1
                    similar_found = True
                    break

            if not similar_found:
                # Add as new restaurant
                deduplicated.append(restaurant.copy())
                processed_names.add(unique_key)

        logger.info(f"✅ Deduplication complete: {len(deduplicated)} unique restaurants")
        logger.info(f"🔄 Merged {self.stats['restaurants_deduplicated']} duplicate entries")
        return deduplicated

    def _merge_restaurant_details(self, existing: Dict[str, Any], new: Dict[str, Any]):
        """Merge details from new restaurant into existing one, keeping the best information"""
        # Merge descriptions (combine if both are meaningful)
        existing_desc = existing.get('Description', '').strip()
        new_desc = new.get('Description', '').strip()

        if new_desc and new_desc.lower() not in ['n/a', 'not specified']:
            if not existing_desc or existing_desc.lower() in ['n/a', 'not specified']:
                existing['Description'] = new_desc
            elif existing_desc != new_desc and len(new_desc) > len(existing_desc):
                # Use longer description if it's significantly longer
                if len(new_desc) > len(existing_desc) * 1.5:
                    existing['Description'] = new_desc

        # Update other fields if they're more complete in new entry
        for field in ['Address', 'City', 'Country']:
            if not existing.get(field) or existing.get(field) == 'Not specified':
                if new.get(field) and new.get(field) != 'Not specified':
                    existing[field] = new[field]

    def _create_final_combined_file(self, restaurants: List[Dict[str, Any]], query: str, original_results: List[Dict[str, Any]]) -> str:
        """Create final combined file with all deduplicated restaurants"""
        try:
            # Create filename
            safe_query = re.sub(r'[^\w\s-]', '', query)[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"combined_cleaned_{safe_query}_{timestamp}.txt"

            # File path
            combined_dir = Path(self.config.INDIVIDUAL_FILE_PROCESSING['combined_files_directory'])
            filepath = combined_dir / filename

            # Create header
            file_content = f"RESTAURANT RECOMMENDATIONS - COMBINED & DEDUPLICATED\n"
            file_content += f"Query: {query}\n" if query else ""
            file_content += f"Individual URLs Processed: {len(original_results)}\n"
            file_content += f"Total Restaurants Found: {len(restaurants)}\n"
            file_content += f"Duplicates Merged: {self.stats['restaurants_deduplicated']}\n"
            file_content += f"Processing Method: Individual file processing with deduplication\n"
            file_content += f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            file_content += "=" * 80 + "\n\n"

            # Add all restaurants
            for idx, restaurant in enumerate(restaurants, 1):
                file_content += f"RESTAURANT {idx}:\n"
                file_content += f"Name: {restaurant.get('Name', 'N/A')}\n"
                file_content += f"City: {restaurant.get('City', 'N/A')}\n"
                file_content += f"Country: {restaurant.get('Country', 'N/A')}\n"
                file_content += f"Description: {restaurant.get('Description', 'N/A')}\n"
                file_content += f"Address: {restaurant.get('Address', 'Not specified')}\n"
                file_content += f"Sources: {restaurant.get('Source', 'N/A')}\n\n"  # ALL sources, comma-separated

            # Save file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(file_content)

            self.stats["combined_files_saved"] += 1
            logger.info(f"💾 Final combined file saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error creating final combined file: {e}")
            return ""

    def _update_final_stats(self, individual_results: List[Dict], combined_restaurants: List[Dict], deduplicated_restaurants: List[Dict]):
        """Update final processing statistics"""
        self.stats["avg_processing_time"] = (
            self.stats["total_processing_time"] / max(self.stats["files_processed"], 1)
        )

        logger.info("📊 FINAL PROCESSING STATS:")
        logger.info(f"   🔢 URLs processed individually: {len(individual_results)}")
        logger.info(f"   🍽️ Total restaurants extracted: {len(combined_restaurants)}")
        logger.info(f"   🔗 Unique restaurants after deduplication: {len(deduplicated_restaurants)}")
        logger.info(f"   🔄 Restaurants merged: {self.stats['restaurants_deduplicated']}")
        logger.info(f"   ⏱️ Average processing time per URL: {self.stats['avg_processing_time']:.2f}s")
        logger.info(f"   🤖 Model used: {self.current_model_type}")
        logger.info(f"   📤 Supabase uploads: {self.stats['supabase_uploads']}")  # NEW

    # Keep existing RTF conversion and utility methods unchanged
    def _rtf_to_text(self, rtf_content: str) -> str:
        """Convert RTF content to clean text while preserving important structure"""
        if not rtf_content or not rtf_content.startswith('{\\rtf'):
            return rtf_content

        try:
            text = rtf_content
            text = re.sub(r'^{\s*\\rtf[^{]*?{[^}]*?}[^}]*?}', '', text, flags=re.DOTALL)
            text = re.sub(r'\\b\s*(.*?)\s*\\b0', r'**\1**', text, flags=re.DOTALL)
            text = re.sub(r'\\i\s*(.*?)\s*\\i0', r'*\1*', text, flags=re.DOTALL)
            text = re.sub(r'\\[a-z]+\d*\s*', ' ', text)
            text = re.sub(r'[{}]', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            return text

        except Exception as e:
            logger.warning(f"⚠️ RTF conversion error: {e}")
            return rtf_content

    def cleanup_old_files(self, days_to_keep: int = 3):
        """Clean up old individual and combined files"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            cleanup_count = 0

            # Clean individual files
            individual_dir = Path(self.config.INDIVIDUAL_FILE_PROCESSING['individual_files_directory'])
            if individual_dir.exists():
                for txt_file in individual_dir.glob("individual_*.txt"):
                    try:
                        file_mtime = datetime.fromtimestamp(txt_file.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            txt_file.unlink()
                            cleanup_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove {txt_file}: {e}")

            # Clean combined files  
            combined_dir = Path(self.config.INDIVIDUAL_FILE_PROCESSING['combined_files_directory'])
            if combined_dir.exists():
                for txt_file in combined_dir.glob("combined_*.txt"):
                    try:
                        file_mtime = datetime.fromtimestamp(txt_file.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            txt_file.unlink()
                            cleanup_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove {txt_file}: {e}")

            if cleanup_count > 0:
                logger.info(f"🧹 Text cleaner cleaned {cleanup_count} old files")

        except Exception as e:
            logger.error(f"❌ Error in text cleaner cleanup: {e}")