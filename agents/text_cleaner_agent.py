# agents/text_cleaner_agent.py
"""
UPDATED: Text Cleaner Agent for Smart Scraper Content
Processes clean text content from Smart Scraper and extracts restaurants

Purpose:
1. Takes clean text content from Smart Scraper (no RTF involved)
2. Uses AI to extract restaurant information from each source
3. Compiles results into clean text file
4. Passes to editor for final processing
"""

import re
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class TextCleanerAgent:
    """
    UPDATED: Clean scraped text content and extract restaurants

    Updated workflow:
    1. Receive clean text content from Smart Scraper (no RTF involved)
    2. Use AI to extract restaurant information from each source
    3. Compile results into clean text file
    4. Pass to editor for final processing
    """

    def __init__(self, config, model_override=None):
        self.config = config

        # Model selection with override capability for testing
        self.current_model_type = model_override or config.MODEL_STRATEGY.get('content_cleaning', 'deepseek')
        self.model = self._initialize_model(self.current_model_type)

        # UPDATED: Stats tracking (removed RTF references)
        self.stats = {
            "files_processed": 0,
            "sources_processed": 0,  # UPDATED: More accurate naming
            "restaurants_extracted": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "current_model": self.current_model_type
        }

    def _initialize_model(self, model_type: str):
        """Initialize AI model based on type"""
        logger.info(f"🤖 Initializing Text Cleaner with {model_type} model")

        if model_type.lower() == 'deepseek':
            try:
                from langchain_deepseek import ChatDeepSeek
                return ChatDeepSeek(
                    model=self.config.DEEPSEEK_CHAT_MODEL,
                    temperature=self.config.DEEPSEEK_TEMPERATURE,
                    max_tokens=self.config.DEEPSEEK_MAX_TOKENS_BY_COMPONENT.get('content_cleaning', 2048),
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
                max_tokens=self.config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('content_cleaning', 2048),
                api_key=self.config.OPENAI_API_KEY
            )

        raise ValueError(f"Unsupported model type: {model_type}")

    def _create_enhanced_cleaning_prompt(self) -> str:
        """
        Enhanced prompt for extracting restaurants from clean scraped text
        """
        return """You are a restaurant content specialist. Extract restaurant information from web content.

**Instructions:**
1. Find ALL restaurants, cafes, bars, bistros, trattorias, and dining establishments mentioned
2. Extract: name, location/address, cuisine type, brief description with key details
3. Focus on establishments that serve food (ignore hotels unless they have notable restaurants)
4. Include food trucks, markets, and specialty food shops if mentioned
5. Preserve original names and addresses exactly as written
6. Add source URL to each restaurant entry

**Output Format:**
For each restaurant found, use this format:
Restaurant: [Name]
Location: [City, Address if available]
Cuisine: [Type of cuisine]
Description: [Brief description with highlights, ambiance, signature dishes, etc.]
Source: {url}

**If no restaurants are found, respond exactly with:**
No restaurants found in this content.

**Important:**
- Only extract establishments that actually serve food
- Don't include just mentions - only places with actual details
- Keep descriptions concise but informative
- Include price ranges, chef names, or special features if mentioned

Return only restaurant entries, nothing else."""

    async def clean_single_source(self, content: str, url: str, content_format: str = 'text') -> str:
        """
        Clean content from smart scraper and extract restaurants
        Works with clean text content (no RTF processing needed)
        """
        try:
            self.stats["sources_processed"] += 1  # UPDATED: More accurate stat name
            start_time = time.time()

            # Content is already clean text from smart scraper
            clean_text = content

            # Use AI to extract and clean restaurant information
            logger.debug(f"🤖 Processing {len(clean_text)} chars with {self.current_model_type}")

            # Enhanced prompt for better restaurant extraction
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self._create_enhanced_cleaning_prompt()),
                ("human", "Extract restaurants from this content from {url}:\n\n{content}")
            ])

            chain = prompt_template | self.model
            response = await chain.ainvoke({
                "url": url,
                "content": clean_text[:8000]  # Limit content to avoid token limits
            })

            # Extract the response content
            if hasattr(response, 'content'):
                ai_response = response.content
            else:
                ai_response = str(response)

            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time

            # Check if AI found restaurants
            if "no restaurants found" in ai_response.lower() or len(ai_response.strip()) < 20:
                logger.info(f"🚫 No restaurants found in {urlparse(url).netloc}")
                return ""

            # Count restaurants for stats
            restaurant_count = ai_response.lower().count("restaurant:") + ai_response.lower().count("name:")
            self.stats["restaurants_extracted"] += restaurant_count

            logger.debug(f"✅ Extracted {restaurant_count} restaurants from {urlparse(url).netloc}")
            return ai_response

        except Exception as e:
            logger.error(f"❌ Error cleaning source {url}: {e}")
            return ""

    async def clean_scraped_results(self, scraped_results: List[Dict[str, Any]], query: str = "") -> str:
        """
        Process scraped results from Smart Scraper and save to TXT file
        Smart Scraper provides clean text content, no RTF conversion needed
        Returns: path to saved TXT file
        """
        logger.info(f"🧹 Starting text cleaning for {len(scraped_results)} scraped sources...")

        all_cleaned_content = []
        successful_cleanings = 0

        for idx, result in enumerate(scraped_results, 1):
            try:
                url = result.get('url', f'source_{idx}')

                # Get content from smart scraper format
                # Smart scraper puts clean text content in 'content' field
                content = result.get('content', '')

                # Skip if no content
                if not content or len(content.strip()) < 50:
                    logger.warning(f"⚠️ No substantial content found for {url}")
                    continue

                logger.info(f"🧹 Cleaning source {idx}/{len(scraped_results)}: {urlparse(url).netloc}")

                # Process as clean text (no RTF conversion needed)
                # Smart scraper already provides clean, extracted text
                cleaned_content = await self.clean_single_source(content, url, 'text')

                if cleaned_content.strip():
                    all_cleaned_content.append(cleaned_content)
                    successful_cleanings += 1
                else:
                    logger.warning(f"⚠️ AI cleaning returned no restaurants for {url}")

            except Exception as e:
                logger.error(f"❌ Error processing source {idx}: {e}")
                continue

        # Check if we found any restaurants at all
        if not all_cleaned_content:
            logger.warning("⚠️ No restaurants found in any source - creating raw content TXT file as fallback")

            # FALLBACK: If AI found no restaurants, create a raw content file for manual review
            raw_content = f"RESTAURANT SEARCH RESULTS (NO RESTAURANTS FOUND BY AI)\n"
            raw_content += f"Query: {query}\n\n"
            raw_content += "RAW SCRAPED CONTENT:\n"
            raw_content += "=" * 80 + "\n\n"

            for idx, result in enumerate(scraped_results, 1):
                content = result.get('content', '')
                url = result.get('url', f'source_{idx}')
                if content:
                    raw_content += f"SOURCE {idx}: {url}\n"
                    raw_content += "-" * 60 + "\n"
                    raw_content += content + "\n\n"

            # Save raw content file
            txt_file_path = self._save_txt_file(raw_content, query, suffix="raw_fallback")
            logger.warning(f"⚠️ Saved raw content to: {txt_file_path}")

            # Update results with raw content for editor fallback
            for result in scraped_results:
                if result.get('url'):
                    result['cleaned_content'] = raw_content
                    result['content_format'] = 'raw_fallback'

            return txt_file_path

        # Create compiled content with restaurants found
        compiled_content = f"RESTAURANT RECOMMENDATIONS\n"
        compiled_content += f"Query: {query}\n" if query else ""
        compiled_content += f"Sources Processed: {successful_cleanings}/{len(scraped_results)}\n"
        compiled_content += f"Cleaned at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        compiled_content += "=" * 80 + "\n\n"

        # Add all cleaned content
        compiled_content += "\n".join(all_cleaned_content)

        # Save to TXT file
        txt_file_path = self._save_txt_file(compiled_content, query)

        logger.info(f"✅ Text cleaning complete: {successful_cleanings}/{len(scraped_results)} sources processed")
        logger.info(f"📊 Total restaurants extracted: {self.stats['restaurants_extracted']}")
        logger.info(f"💾 Saved clean text to: {txt_file_path}")

        # Update the scraped results with cleaned content for editor
        for result in scraped_results:
            if result.get('url'):
                # Set cleaned_content field that editor expects
                result['cleaned_content'] = compiled_content
                result['content_format'] = 'cleaned_text'

        # Return the file path
        return txt_file_path

    def _save_txt_file(self, clean_content: str, query: str, suffix: str = "") -> str:
        """
        Save clean text content to TXT file with optional suffix
        """
        try:
            # Create filename from query and timestamp
            safe_query = re.sub(r'[^\w\s-]', '', query)[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if suffix:
                filename = f"scraped_{safe_query}_{suffix}_{timestamp}.txt"
            else:
                filename = f"scraped_{safe_query}_{timestamp}.txt"

            # Ensure scraped_content directory exists
            output_dir = Path("scraped_content")
            output_dir.mkdir(exist_ok=True)

            filepath = output_dir / filename

            # Save TXT file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(clean_content)

            logger.info(f"💾 Saved text content to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error saving TXT file: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        """UPDATED: Get cleaning statistics (removed RTF references)"""
        return {
            **self.stats,
            "sources_processing_rate": self.stats["sources_processed"] / max(self.stats["sources_processed"], 1),
            "avg_restaurants_per_source": self.stats["restaurants_extracted"] / max(self.stats["sources_processed"], 1),
            "model": self.current_model_type,
            "input_format": "CLEAN_TEXT",  # UPDATED: No longer RTF
            "output_format": "TEXT"
        }

    def cleanup_old_files(self, max_age_hours: int = 48):
        """
        UPDATED: Clean up old TXT files created by this text cleaner
        """
        try:
            from datetime import datetime, timedelta

            cleanup_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            # Clean up files in scraped_content directory
            scraped_content_dir = Path("scraped_content")
            if scraped_content_dir.exists():
                # UPDATED: Look for files created by this cleaner (including raw fallback)
                for txt_file in scraped_content_dir.glob("scraped_*.txt"):
                    try:
                        file_mtime = datetime.fromtimestamp(txt_file.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            txt_file.unlink()
                            cleanup_count += 1
                            logger.debug(f"🗑️ Cleaned old TXT file: {txt_file.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove {txt_file}: {e}")

            if cleanup_count > 0:
                logger.info(f"🧹 Text cleaner cleaned {cleanup_count} old TXT files")

        except Exception as e:
            logger.error(f"❌ Error in text cleaner cleanup: {e}")