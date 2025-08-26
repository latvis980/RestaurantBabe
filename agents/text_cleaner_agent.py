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
6. Upload final combined file to Supabase
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
    REFACTORED: Individual file processing with restaurant deduplication
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
            "combined_files_saved": 0
        }

        # Create directories for individual and combined files
        self._setup_directories()

    def _initialize_model(self, model_type: str):
        """Initialize AI model with INCREASED token limits"""
        logger.info(f"🤖 Initializing REFACTORED Text Cleaner with {model_type} model")

        if model_type.lower() == 'deepseek':
            try:
                from langchain_deepseek import ChatDeepSeek
                return ChatDeepSeek(
                    model=self.config.DEEPSEEK_CHAT_MODEL,
                    temperature=self.config.DEEPSEEK_TEMPERATURE,
                    max_tokens=self.config.DEEPSEEK_MAX_TOKENS_BY_COMPONENT.get('content_cleaning', 12288),  # INCREASED
                    api_key=self.config.DEEPSEEK_API_KEY
                )
            except ImportError:
                logger.warning("⚠️ DeepSeek not available, falling back to OpenAI")
                model_type = 'openai'

        if model_type.lower() == 'openai':
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.config.OPENAI_MODEL,
                temperature=self.config.OPENAI_TEMPERATURE,
                max_tokens=self.config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('content_cleaning', 12288),  # FIXED: max_tokens parameter
                api_key=self.config.OPENAI_API_KEY
            )

        raise ValueError(f"Unsupported model type: {model_type}")

    def _setup_directories(self):
        """Create directories for individual and combined files"""
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

        # Step 5: Update stats
        self._update_final_stats(individual_results, combined_restaurants, deduplicated_restaurants)

        logger.info(f"✅ REFACTORED: Individual processing complete. Final file: {final_file_path}")
        return final_file_path

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
                    self.stats["individual_files_processed"] += 1

            except Exception as e:
                logger.error(f"❌ Error processing individual URL {idx}: {e}")
                continue

        logger.info(f"✅ Individual processing complete: {len(individual_results)}/{len(scraped_results)} successful")
        return individual_results

    async def _clean_individual_source(self, content: str, url: str, content_format: str = 'text') -> Dict[str, Any]:
        """Clean content from individual source with enhanced extraction"""
        start_time = datetime.now()

        try:
            # Convert RTF to text if needed
            if content_format == 'rtf' or content.startswith('{\\rtf'):
                text_content = self._rtf_to_text(content)
                self.stats["rtf_files_processed"] += 1
                logger.debug(f"📄 Converted RTF to text for {urlparse(url).netloc}")
            else:
                text_content = content

            # Skip if content is too short
            if len(text_content.strip()) < 100:
                logger.warning(f"⚠️ Content too short to clean: {url}")
                return {}

            # Enhanced AI prompt for individual processing
            cleaning_prompt = self._create_enhanced_individual_prompt()
            content_with_metadata = f"SOURCE URL: {url}\n\n{text_content}"

            # Use AI with INCREASED token limits
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=cleaning_prompt.format(content=content_with_metadata))]
            response = await self.model.ainvoke(messages)
            cleaned_content = str(response.content).strip()  

            # Parse AI response into structured data
            parsed_result = self._parse_individual_ai_response(cleaned_content, url)

            # Update processing stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_processing_time"] += processing_time
            self.stats["files_processed"] += 1

            if parsed_result.get('restaurants'):
                restaurant_count = len(parsed_result['restaurants'])
                self.stats["restaurants_extracted"] += restaurant_count
                logger.info(f"✅ Extracted {restaurant_count} restaurants from {urlparse(url).netloc}")

            return parsed_result

        except Exception as e:
            logger.error(f"❌ Error cleaning individual source {url}: {e}")
            return {}

    def _create_enhanced_individual_prompt(self) -> str:
        """Enhanced prompt for individual file processing with source URL tracking"""
        return """You are a restaurant extraction specialist. Extract ALL restaurants from this web content.

**CRITICAL REQUIREMENTS:**
1. Extract EVERY restaurant, cafe, bar, bistro mentioned
2. Add the source URL to EACH restaurant entry  
3. Include comprehensive descriptions with all available details
4. Assign specific city/country to each restaurant
5. Use the increased token limit to provide complete extractions

**Content to process:**
{content}

**REQUIRED OUTPUT FORMAT:**

METADATA:
Original URL: [Extract from content header]
Primary City: [Main city mentioned in content]
Primary Country: [Main country mentioned in content]
Total Restaurants Found: [Number]

RESTAURANTS:
Name: [Restaurant Name]
City: [Specific city for this restaurant]
Country: [Country for this restaurant] 
Description: [Comprehensive description including cuisine, atmosphere, specialties, chef info, etc.]
Address: [Full address if available, or "Not specified"]
Source: [Source URL from content header]

Name: [Next Restaurant Name]
City: [Specific city]
Country: [Country]
Description: [Full description]
Address: [Address or "Not specified"]
Source: [Source URL]

**EXTRACTION RULES:**
- Extract restaurants even if only briefly mentioned
- Translate non-English descriptions to English
- For minimal info sources (like Michelin lists), use "Recommended by [source name]" as description
- Include ALL details: cuisine type, price range, atmosphere, signature dishes, chef names
- If address is partial, include what's available
- NEVER skip restaurants due to minimal information

Return ONLY the formatted data above, no other text."""

    def _parse_individual_ai_response(self, ai_response: str, source_url: str) -> Dict[str, Any]:
        """Parse AI response into structured restaurant data"""
        try:
            result = {
                'source_url': source_url,
                'restaurants': [],
                'metadata': {}
            }

            lines = ai_response.strip().split('\n')
            current_restaurant = {}
            in_restaurants_section = False

            for line in lines:
                line = line.strip()

                if not line:
                    # End of restaurant entry
                    if current_restaurant.get('Name') and current_restaurant.get('Description'):
                        # Ensure source URL is set for each restaurant
                        current_restaurant['Source'] = source_url
                        result['restaurants'].append(current_restaurant.copy())
                        current_restaurant = {}
                    continue

                if line.startswith('RESTAURANTS:'):
                    in_restaurants_section = True
                    continue

                if line.startswith('METADATA:'):
                    in_restaurants_section = False
                    continue

                if not in_restaurants_section:
                    # Parse metadata
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result['metadata'][key.strip()] = value.strip()
                else:
                    # Parse restaurant fields
                    if ':' in line:
                        key, value = line.split(':', 1)
                        current_restaurant[key.strip()] = value.strip()

            # Add final restaurant if exists
            if current_restaurant.get('Name') and current_restaurant.get('Description'):
                current_restaurant['Source'] = source_url
                result['restaurants'].append(current_restaurant)

            return result

        except Exception as e:
            logger.error(f"❌ Error parsing AI response: {e}")
            return {'source_url': source_url, 'restaurants': [], 'metadata': {}}

    def _save_individual_file(self, cleaned_result: Dict[str, Any], file_index: int, query: str) -> str:
        """Save individual cleaned result to file"""
        try:
            # Create filename
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
            for restaurant in result.get('restaurants', []):
                # Ensure source URL is preserved
                if 'Source' not in restaurant:
                    restaurant['Source'] = result['source_url']
                all_restaurants.append(restaurant)

        logger.info(f"✅ Combined into {len(all_restaurants)} total restaurant entries")
        return all_restaurants

    def _deduplicate_restaurants(self, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate restaurants and combine sources (ALL sources, comma-separated)"""
        logger.info(f"🔍 Deduplicating {len(restaurants)} restaurant entries...")

        deduplicated = []
        processed_names = set()

        for restaurant in restaurants:
            name = restaurant.get('Name', '').strip().lower()
            city = restaurant.get('City', '').strip().lower()

            # Create unique identifier
            restaurant_id = f"{name}_{city}"

            if not name:
                continue

            # Check for duplicates using similarity matching
            existing_restaurant = None
            for existing in deduplicated:
                existing_name = existing.get('Name', '').strip().lower()
                existing_city = existing.get('City', '').strip().lower()
                existing_id = f"{existing_name}_{existing_city}"

                # Check name similarity
                name_similarity = self._calculate_similarity(name, existing_name)
                city_match = city == existing_city or not city or not existing_city

                if name_similarity >= self.config.RESTAURANT_DEDUPLICATION['name_similarity_threshold'] and city_match:
                    existing_restaurant = existing
                    break

            if existing_restaurant:
                # Merge with existing restaurant
                self._merge_restaurant_entries(existing_restaurant, restaurant)
                self.stats["restaurants_deduplicated"] += 1
            else:
                # Add as new restaurant
                # Ensure Source is a string (convert list to comma-separated if needed)
                if isinstance(restaurant.get('Source'), list):
                    restaurant['Source'] = ', '.join(restaurant['Source'])
                deduplicated.append(restaurant)

        logger.info(f"✅ Deduplication complete: {len(deduplicated)} unique restaurants ({self.stats['restaurants_deduplicated']} duplicates merged)")
        return deduplicated

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _merge_restaurant_entries(self, existing: Dict[str, Any], new: Dict[str, Any]):
        """Merge new restaurant entry into existing one (combine ALL sources)"""

        # Combine sources (KEY REQUIREMENT: ALL sources, comma-separated)
        existing_sources = existing.get('Source', '')
        new_sources = new.get('Source', '')

        # Convert to lists if needed
        if isinstance(existing_sources, str):
            existing_sources = [s.strip() for s in existing_sources.split(',') if s.strip()]
        if isinstance(new_sources, str):
            new_sources = [s.strip() for s in new_sources.split(',') if s.strip()]

        # Combine unique sources
        all_sources = list(set(existing_sources + new_sources))
        existing['Source'] = ', '.join(all_sources[:self.config.RESTAURANT_DEDUPLICATION['max_sources_per_restaurant']])

        # Combine descriptions if enabled
        if self.config.RESTAURANT_DEDUPLICATION['combine_descriptions']:
            existing_desc = existing.get('Description', '').strip()
            new_desc = new.get('Description', '').strip()

            if new_desc and new_desc not in existing_desc:
                if existing_desc:
                    existing['Description'] = f"{existing_desc} | {new_desc}"
                else:
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
            text = re.sub(r'\\par\s*', '\n\n', text)
            text = re.sub(r'\\line\s*', '\n', text)
            text = re.sub(r'\\f\d+', '', text)
            text = re.sub(r'\\fs\d+', '', text)
            text = re.sub(r'\\cf\d+', '', text)
            text = re.sub(r'\\[a-zA-Z]+\d*\s*', ' ', text)
            text = text.replace(r'\\', '\\')
            text = text.replace(r'\{', '{')
            text = text.replace(r'\}', '}')
            text = re.sub(r'[{}]', '', text)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            text = text.strip()
            return text
        except Exception as e:
            logger.error(f"❌ Error converting RTF to text: {e}")
            return rtf_content

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced cleaning statistics"""
        return {
            **self.stats,
            "rtf_processing_rate": self.stats["rtf_files_processed"] / max(self.stats["files_processed"], 1),
            "avg_restaurants_per_file": self.stats["restaurants_extracted"] / max(self.stats["individual_files_processed"], 1),
            "deduplication_rate": self.stats["restaurants_deduplicated"] / max(self.stats["restaurants_extracted"], 1),
            "model": self.current_model_type,
            "processing_method": "individual_with_deduplication",
            "input_format": "RTF/TEXT",
            "output_format": "COMBINED_TEXT"
        }

    # Keep existing cleanup method unchanged but extend for new directories
    def cleanup_old_files(self, max_age_hours: int = 48):
        """Clean up old files from both individual and combined directories"""
        try:
            from datetime import timedelta
            cleanup_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

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
