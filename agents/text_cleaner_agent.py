# agents/text_cleaner_agent.py
"""
Simple RTF to Text Cleaner Agent 
Converts RTF to clean text for editor processing

Purpose:
1. Takes RTF formatted content from human-mimic scraper
2. Converts RTF to readable text while preserving structure
3. Uses AI to clean and extract restaurant information
4. Outputs clean text for editor (same as before)
5. Supabase gets TXT files (existing workflow)
"""

import re
import logging
import json
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TextCleanerAgent:
    """
    Clean RTF content and output clean text for editor

    Workflow:
    1. Receive RTF formatted results from human_mimic_scraper
    2. Convert RTF to clean text while preserving structure
    3. Use AI to clean and extract restaurant information  
    4. Output clean text file for editor (same as before)
    5. Editor processes normally, Supabase gets TXT files
    """

    def __init__(self, config, model_override=None):
        self.config = config

        # Model selection with override capability for testing
        self.current_model_type = model_override or config.MODEL_STRATEGY.get('content_cleaning', 'deepseek')
        self.model = self._initialize_model(self.current_model_type)

        # Stats tracking
        self.stats = {
            "files_processed": 0,
            "rtf_files_processed": 0,
            "restaurants_extracted": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "current_model": self.current_model_type
        }

    def _initialize_model(self, model_type: str):
        """Initialize AI model based on type"""
        logger.info(f"🤖 Initializing RTF-to-Text Cleaner with {model_type} model")

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

    def _rtf_to_text(self, rtf_content: str) -> str:
        """
        Convert RTF content to clean text while preserving important structure
        """
        if not rtf_content or not rtf_content.startswith('{\\rtf'):
            # Not RTF format, return as-is
            return rtf_content

        try:
            text = rtf_content

            # Remove RTF header
            text = re.sub(r'^{\s*\\rtf[^{]*?{[^}]*?}[^}]*?}', '', text, flags=re.DOTALL)

            # Convert RTF formatting to text markers for AI analysis
            # Bold text (often restaurant names)
            text = re.sub(r'\\b\s*(.*?)\s*\\b0', r'**\1**', text, flags=re.DOTALL)

            # Italic text
            text = re.sub(r'\\i\s*(.*?)\s*\\i0', r'*\1*', text, flags=re.DOTALL)

            # Paragraphs
            text = re.sub(r'\\par\s*', '\n\n', text)

            # Line breaks
            text = re.sub(r'\\line\s*', '\n', text)

            # Remove font specifications
            text = re.sub(r'\\f\d+', '', text)
            text = re.sub(r'\\fs\d+', '', text)
            text = re.sub(r'\\cf\d+', '', text)

            # Remove other RTF control words
            text = re.sub(r'\\[a-zA-Z]+\d*\s*', ' ', text)

            # Clean up RTF escape sequences
            text = text.replace(r'\\', '\\')
            text = text.replace(r'\{', '{')
            text = text.replace(r'\}', '}')

            # Remove remaining RTF brackets
            text = re.sub(r'[{}]', '', text)

            # Clean up whitespace
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            text = text.strip()

            logger.debug(f"📄 Converted RTF to text: {len(rtf_content)} → {len(text)} chars")
            return text

        except Exception as e:
            logger.error(f"❌ Error converting RTF to text: {e}")
            # Fallback: try to extract text content from RTF
            try:
                text = re.sub(r'\\[a-zA-Z]+\d*\s*', ' ', rtf_content)
                text = re.sub(r'[{}]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            except:
                return rtf_content

    def _create_cleaning_prompt(self) -> str:
        """
        Create AI prompt for cleaning restaurant content with source attribution
        FIXED: Now properly handles double curly braces for LangChain compatibility
        """
        return """You are a content cleaning specialist for restaurant recommendation systems. 
Your job is to extract clean, useful restaurant information from scraped web content.

The content comes from a single source. The source URL is shown at the top of the content.

TASK: Extract restaurant names and descriptions from the provided content.

RULES:
1. Focus ONLY on restaurants, cafes, bars, bistros, and similar dining establishments
2. Extract restaurant name and a comprehensive description with all key details (cuisine, atmosphere, chef, concept, signature dishes etc.)
3. Ignore: navigation menus, ads, cookie notices, social media links, unrelated articles
4. Preserve: restaurant names, addresses, descriptions as-is, only translated to English if in a foreign language
5. IMPORTANT: Add the source URL (from the top of the content) to each restaurant entry
6. Format: "Restaurant Name (source_url): Description, address if available"
7. If there are no description, only restaurant names, but the source is very reputable (i.e. Michelin), just include names and url, and set description to "recommended by..." and the name of the source.

CONTENT TYPE: Restaurant guide/listing page
DESIRED OUTPUT: Clean list of restaurants with descriptions, addresses, and source URLs

Content to clean:
{content}

OUTPUT FORMAT:
Restaurant Name 1
source_url
Description in English
address if available

Restaurant Name 2
source_url
Description in English
address if available

...

If no clear restaurants are found, respond with: "No restaurants found in this content."
"""

    async def clean_single_source(self, content: str, url: str, content_format: str = 'text') -> str:
        """
        Clean content from a single source and return clean text
        FIXED: Now properly substitutes content into prompt template
        """
        start_time = datetime.now()

        try:
            # Convert RTF to text if needed
            if content_format == 'rtf' or content.startswith('{\\rtf'):
                text_content = self._rtf_to_text(content)
                self.stats["rtf_files_processed"] += 1
                logger.info(f"📄 Converted RTF to text for {urlparse(url).netloc}")
            else:
                text_content = content

            # Skip if content is too short
            if len(text_content.strip()) < 100:
                logger.warning(f"⚠️ Content too short to clean: {url}")
                return ""

            # Create AI prompt with proper substitution
            cleaning_prompt_template = self._create_cleaning_prompt()

            # FIXED: Properly substitute content into the template
            # Add URL context to help AI understand the source
            content_with_url = f"SOURCE URL: {url}\n\n{text_content}"
            final_prompt = cleaning_prompt_template.format(content=content_with_url)

            # Use AI to clean content
            from langchain.schema import HumanMessage

            messages = [HumanMessage(content=final_prompt)]
            response = await self.model.ainvoke(messages)
            cleaned_content = response.content.strip()

            # Post-process AI response
            if "No restaurants found" in cleaned_content:
                logger.info(f"🚫 No restaurants found in {urlparse(url).netloc}")
                return ""

            # Count extracted restaurants - look for multiple indicators
            restaurant_lines = []
            lines = cleaned_content.split('\n')

            for line in lines:
                line = line.strip()
                # Look for restaurant names (lines that don't look like URLs or descriptions)
                if line and not line.startswith('http') and not line.startswith('SOURCE'):
                    # Check if this looks like a restaurant name line
                    if (len(line) < 100 and  # Not too long to be a description
                        not line.startswith('Description') and 
                        not line.startswith('Address') and
                        not line.startswith('source_url')):
                        restaurant_lines.append(line)

            restaurant_count = len(restaurant_lines)

            if restaurant_count > 0:
                self.stats["restaurants_extracted"] += restaurant_count
                logger.info(f"✅ Extracted {restaurant_count} restaurants from {urlparse(url).netloc}")

                # Add source attribution
                source_header = f"\n=== {urlparse(url).netloc.upper()} ===\n"
                cleaned_content = source_header + cleaned_content + "\n"
            else:
                logger.warning(f"⚠️ AI cleaning produced no valid restaurants for {url}")
                # Log the first part of the AI response for debugging
                logger.debug(f"AI response preview: {cleaned_content[:200]}...")
                return ""

            return cleaned_content

        except Exception as e:
            logger.error(f"❌ Error cleaning content from {url}: {e}")
            return ""

        finally:
            # Update processing time stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_processing_time"] += processing_time
            self.stats["files_processed"] += 1

            if self.stats["files_processed"] > 0:
                self.stats["avg_processing_time"] = self.stats["total_processing_time"] / self.stats["files_processed"]

    async def clean_scraped_results(self, scraped_results: List[Dict[str, Any]], query: str = "") -> str:
        """
        Clean multiple scraped results and save to TXT file
        Returns: path to saved TXT file
        """
        logger.info(f"🧹 Starting RTF-to-Text cleaning for {len(scraped_results)} sources...")

        all_cleaned_content = []
        successful_cleanings = 0

        for idx, result in enumerate(scraped_results, 1):
            try:
                url = result.get('url', f'source_{idx}')

                # Determine content format 
                content_format = 'rtf' if result.get('scraping_method') == 'human_mimic_rtf' else 'text'

                # Get RTF or text content
                content = result.get('content', '') or result.get('scraped_content', '') or result.get('text_content', '')

                if not content:
                    logger.warning(f"⚠️ No content found for {url}")
                    continue

                logger.info(f"🧹 Cleaning source {idx}/{len(scraped_results)}: {urlparse(url).netloc}")

                # Clean the content
                cleaned_content = await self.clean_single_source(content, url, content_format)

                if cleaned_content.strip():
                    all_cleaned_content.append(cleaned_content)
                    successful_cleanings += 1

            except Exception as e:
                logger.error(f"❌ Error processing source {idx}: {e}")
                continue

        # Compile all cleaned content
        if not all_cleaned_content:
            logger.warning("⚠️ No content was successfully cleaned")
            return ""

        # Create header with query and metadata
        compiled_content = f"RESTAURANT RECOMMENDATIONS\n"
        compiled_content += f"Query: {query}\n" if query else ""
        compiled_content += f"Sources Processed: {successful_cleanings}/{len(scraped_results)}\n"
        compiled_content += f"Cleaned at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        compiled_content += "=" * 80 + "\n\n"

        # Add all cleaned content
        compiled_content += "\n".join(all_cleaned_content)

        # Save to TXT file
        txt_file_path = self._save_txt_file(compiled_content, query)

        logger.info(f"✅ RTF-to-Text cleaning complete: {successful_cleanings}/{len(scraped_results)} sources cleaned")
        logger.info(f"📊 Total restaurants extracted: {self.stats['restaurants_extracted']}")
        logger.info(f"💾 Saved clean text to: {txt_file_path}")

        # Update the scraped results with cleaned content for editor
        for result in scraped_results:
            if result.get('url'):
                # Set cleaned_content field that editor expects
                result['cleaned_content'] = compiled_content
                result['content_format'] = 'text'

        # Return the file path (not content)
        return txt_file_path

    def _save_txt_file(self, clean_content: str, query: str) -> str:
        """Save clean text content to TXT file"""
        try:
            # Create filename from query and timestamp
            safe_query = re.sub(r'[^\w\s-]', '', query)[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cleaned_{safe_query}_{timestamp}.txt"

            # Ensure scraped_content directory exists
            output_dir = Path("scraped_content")
            output_dir.mkdir(exist_ok=True)

            filepath = output_dir / filename

            # Save TXT file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(clean_content)

            logger.info(f"💾 Saved clean text to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error saving TXT file: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics"""
        return {
            **self.stats,
            "rtf_processing_rate": self.stats["rtf_files_processed"] / max(self.stats["files_processed"], 1),
            "avg_restaurants_per_file": self.stats["restaurants_extracted"] / max(self.stats["files_processed"], 1),
            "model": self.current_model_type,
            "input_format": "RTF",
            "output_format": "TEXT"
        }

    def cleanup_old_files(self, max_age_hours: int = 48):
        """
        Clean up old TXT files created by this text cleaner
        Called by the file cleanup manager
        """
        try:
            from datetime import datetime, timedelta

            cleanup_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            # Clean up files in scraped_content directory
            scraped_content_dir = Path("scraped_content")
            if scraped_content_dir.exists():
                # Look for files created by this cleaner
                for txt_file in scraped_content_dir.glob("cleaned_*.txt"):
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