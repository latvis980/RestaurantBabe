# agents/text_cleaner_agent.py
"""
Text Cleaner Agent for Human-Mimic Scraped Content
Sits between human_mimic_scraper and editor_agent

Purpose:
1. Takes raw scraped content (like timeout.fr example)
2. Uses AI to distinguish restaurant content from site noise
3. Extracts clean restaurant name + description pairs
4. Preserves source URL information
5. Compiles multiple files into one clean text file for editor

Input: Raw scraped content with lots of site noise
Output: Clean text file with just restaurant info for editor
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
    Clean scraped content and extract restaurant information

    Workflow:
    1. Receive scraped results from human_mimic_scraper
    2. Clean each result individually 
    3. Extract restaurant name + description pairs
    4. Preserve URL source info
    5. Compile into one clean text file
    6. Pass to editor_agent
    """

    def __init__(self, config, model_override=None):
        self.config = config

        # Model selection with override capability for testing
        self.current_model_type = model_override or config.MODEL_STRATEGY.get('content_cleaning', 'deepseek')
        self.model = self._initialize_model(self.current_model_type)

        # Stats tracking with model type
        self.stats = {
            "files_processed": 0,
            "restaurants_extracted": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "cleanup_success_rate": 0.0,
            "content_reduction_ratio": 0.0,
            "current_model": self.current_model_type
        }

    def _initialize_model(self, model_type: str):
        """Initialize AI model based on type"""
        logger.info(f"🤖 Initializing Text Cleaner with {model_type} model")

        if model_type.lower() == 'deepseek':
            try:
                from langchain_deepseek import ChatDeepSeek  # CORRECT IMPORT
                return ChatDeepSeek(
                    model=self.config.DEEPSEEK_CHAT_MODEL,
                    temperature=self.config.DEEPSEEK_TEMPERATURE,
                    max_tokens=self.config.DEEPSEEK_MAX_TOKENS_BY_COMPONENT.get('content_cleaning', 2048),
                    api_key=self.config.DEEPSEEK_API_KEY
                )
            except ImportError:
                logger.warning("⚠️ DeepSeek not available, falling back to OpenAI")
                model_type = 'openai'  # Fallback to OpenAI

        if model_type.lower() == 'openai':
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.config.OPENAI_MODEL,
                temperature=self.config.OPENAI_TEMPERATURE,
                max_tokens=self.config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('content_cleaning', 2048),
                api_key=self.config.OPENAI_API_KEY
            )
        else:
            logger.warning(f"⚠️ Unknown model type {model_type}, defaulting to OpenAI")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.config.OPENAI_MODEL,
                temperature=self.config.OPENAI_TEMPERATURE,
                max_tokens=self.config.OPENAI_MAX_TOKENS_BY_COMPONENT.get('content_cleaning', 2048),
                api_key=self.config.OPENAI_API_KEY
            )

    def switch_model(self, model_type: str):
        """Switch between DeepSeek and OpenAI for testing"""
        logger.info(f"🔄 Switching Text Cleaner model from {self.current_model_type} to {model_type}")
        self.current_model_type = model_type
        self.model = self._initialize_model(model_type)
        self.stats["current_model"] = model_type

    def clean_scraped_results(self, scraped_results: List[Dict[str, Any]]) -> str:
        """
        Main entry point: clean all scraped results and compile into one text file

        Args:
            scraped_results: List of results from human_mimic_scraper

        Returns:
            String: Clean compiled text for editor_agent
        """
        start_time = datetime.now()
        logger.info(f"🧹 Text Cleaner processing {len(scraped_results)} scraped files using {self.current_model_type}")

        cleaned_sections = []
        total_restaurants = 0
        original_length = 0
        cleaned_length = 0

        for i, result in enumerate(scraped_results, 1):
            if not result.get('scraping_success'):
                logger.warning(f"⚠️ Skipping failed scrape: {result.get('url', 'unknown')}")
                continue

            content = result.get('scraped_content', '')
            url = result.get('url', '')

            if not content or len(content.strip()) < 100:
                logger.warning(f"⚠️ Skipping short content from {url}")
                continue

            logger.info(f"🧹 Cleaning content {i}/{len(scraped_results)} from {self._get_domain(url)}")

            # Extract restaurants directly from raw content - let AI decide what's noise
            extracted_restaurants = self._ai_extract_restaurants(content, url)

            if extracted_restaurants:
                # Format for compilation
                domain = self._get_domain(url)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                formatted_content = f"""
SOURCE: {url}
DOMAIN: {domain}
SCRAPED_AT: {timestamp}
MODEL_USED: {self.current_model_type}
CONTENT_LENGTH: {len(content)} chars → {len(extracted_restaurants)} chars
===============================================================================

{extracted_restaurants}

===============================================================================
"""
                cleaned_sections.append(formatted_content)

                # Count restaurants extracted
                restaurant_count = extracted_restaurants.count('Restaurant:')
                total_restaurants += restaurant_count

                # Track content reduction
                original_length += len(content)
                cleaned_length += len(extracted_restaurants)

                logger.info(f"✅ Extracted {restaurant_count} restaurants from {self._get_domain(url)} using {self.current_model_type}")

        # Compile all sections
        if cleaned_sections:
            compiled_content = self._compile_sections(cleaned_sections)

            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(len(scraped_results), total_restaurants, processing_time, original_length, cleaned_length)

            logger.info(f"✅ Text Cleaner complete: {total_restaurants} restaurants extracted from {len(cleaned_sections)} sources using {self.current_model_type}")
            return compiled_content
        else:
            logger.warning("⚠️ No clean content extracted from any sources")
            return ""

    def _ai_extract_restaurants(self, content: str, url: str) -> str:
        """Use AI to extract clean restaurant name + description pairs directly from raw content"""

        extraction_prompt = f"""You are a restaurant content extraction expert analyzing scraped website content.

Your task: Extract ONLY restaurants and their information from this messy web content.

INSTRUCTIONS:
1. Identify what is restaurant content vs. website navigation/ads/clutter
2. Extract restaurant names, locations, and descriptions
3. Ignore: navigation menus, headers, footers, ads, newsletter signups, social media links, copyright text
4. Include: restaurant names, addresses, food descriptions, atmosphere descriptions, specialties
5. Combine all information about each restaurant into one coherent description
6. If content mentions food items, drinks, atmosphere - include it in the description

SOURCE URL: {url}

FORMAT YOUR OUTPUT EXACTLY LIKE THIS:

Restaurant: [Name]
Location: [Address/Area if available]
Description: [Complete description including food, drinks, atmosphere, specialties]

Restaurant: [Name]  
Location: [Address/Area if available]
Description: [Complete description including food, drinks, atmosphere, specialties]

CONTENT TO ANALYZE:
{content[:4000]}"""

        try:
            response = self.model.invoke(extraction_prompt)

            if hasattr(response, 'content'):
                extracted = response.content.strip()
            else:
                extracted = str(response).strip()

            # Validate that we got restaurant data
            if 'Restaurant:' in extracted and len(extracted) > 50:
                return extracted
            else:
                logger.warning(f"⚠️ AI extraction failed for {url} - no valid restaurant format")
                return ""

        except Exception as e:
            logger.error(f"❌ AI extraction error for {url}: {e}")
            return ""

    def _compile_sections(self, cleaned_sections: List[str]) -> str:
        """Compile all cleaned sections into one file for the editor"""

        header = f"""CLEANED RESTAURANT CONTENT FOR EDITOR
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Sources: {len(cleaned_sections)} websites
Cleaned by: Text Cleaner Agent ({self.current_model_type.upper()} model)

This file contains restaurant information extracted from scraped content.
All navigation text, ads, and site clutter has been removed.
Only restaurant names, locations, and descriptions remain.

===============================================================================
"""

        # Combine all sections
        combined_content = header + "\n".join(cleaned_sections)

        # Add summary footer
        restaurant_count = combined_content.count('Restaurant:')
        footer = f"""
===============================================================================
SUMMARY:
- Total restaurants extracted: {restaurant_count}
- Sources processed: {len(cleaned_sections)}
- Model used: {self.current_model_type.upper()}
- Ready for editor processing
===============================================================================
"""

        return combined_content + footer

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"

    def _update_stats(self, files_processed: int, restaurants_extracted: int, 
                     processing_time: float, original_length: int, cleaned_length: int):
        """Update processing statistics"""
        self.stats["files_processed"] += files_processed
        self.stats["restaurants_extracted"] += restaurants_extracted
        self.stats["total_processing_time"] += processing_time

        if self.stats["files_processed"] > 0:
            self.stats["avg_processing_time"] = self.stats["total_processing_time"] / self.stats["files_processed"]
            self.stats["cleanup_success_rate"] = restaurants_extracted / files_processed if files_processed > 0 else 0

        if original_length > 0:
            self.stats["content_reduction_ratio"] = (original_length - cleaned_length) / original_length

    def get_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "files_processed": 0,
            "restaurants_extracted": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "cleanup_success_rate": 0.0,
            "content_reduction_ratio": 0.0,
            "current_model": self.current_model_type
        }

    # Optional: Save to file system for debugging
    def save_cleaned_content(self, cleaned_content: str, query: str) -> str:
        """Save cleaned content to file for debugging/inspection"""
        try:
            # Create filename from query and timestamp
            safe_query = re.sub(r'[^\w\s-]', '', query)[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cleaned_{safe_query}_{timestamp}.txt"

            # Ensure debug directory exists
            debug_dir = Path("debug_logs")
            debug_dir.mkdir(exist_ok=True)

            filepath = debug_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            logger.info(f"💾 Saved cleaned content to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error saving cleaned content: {e}")
            return ""