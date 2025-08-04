# agents/firecrawl_scraper.py - CLEANED VERSION
"""
CLEANED Firecrawl Web Scraper - Focused on Quality Restaurant Extraction

CLEANED UP:
1. ✅ Removed unnecessary pre-filtering (handled by smart scraper)
2. ✅ Removed hardcoded restaurant keywords check
3. ✅ Removed legacy compatibility wrapper (not used)
4. ✅ Focused purely on quality restaurant extraction
5. ✅ Simplified to single responsibility: extract restaurant data

Since the smart scraper handles pre-filtering, this scraper's only job is:
- Extract high-quality restaurant information 
- Format it properly for the editor (no more list_analyzer)
- Provide clean, structured restaurant data
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class RestaurantExtractionResult:
    """Result from restaurant extraction"""
    restaurants: List[Dict[str, Any]]
    source_url: str
    source_name: str
    extraction_method: str
    success: bool
    error_message: Optional[str] = None

class RestaurantSchema(BaseModel):
    """Restaurant data schema for Firecrawl extraction"""
    name: str = Field(description="Restaurant name exactly as written")
    description: str = Field(description="Complete description including cuisine, atmosphere, and specialties")
    address: str = Field(default="", description="Full street address if available")
    city: str = Field(default="", description="City name")
    price_range: str = Field(default="", description="Price indication (€, €€, €€€ or budget/mid-range/expensive)")
    cuisine_type: str = Field(default="", description="Type of cuisine or food style")
    recommended_dishes: List[str] = Field(default_factory=list, description="Notable dishes or specialties mentioned")
    phone: str = Field(default="", description="Phone number if provided")
    website: str = Field(default="", description="Restaurant website URL if mentioned")
    rating: str = Field(default="", description="Rating or review score if available")
    opening_hours: str = Field(default="", description="Operating hours if specified")
    special_features: List[str] = Field(default_factory=list, description="Special features like outdoor seating, live music, etc.")

class RestaurantGuideExtraction(BaseModel):
    """Schema for extracting restaurant guides using Firecrawl extraction API"""
    restaurants: List[RestaurantSchema] = Field(description="All restaurants mentioned in the guide")
    guide_title: str = Field(description="Title of the restaurant guide or article")
    publication: str = Field(description="Name of the publication or website")
    article_date: str = Field(default="", description="Publication date if available")
    city_focus: str = Field(default="", description="Primary city or location covered")
    cuisine_focus: str = Field(default="", description="Specific cuisine type if the guide is focused")
    total_restaurants: int = Field(description="Number of restaurants found")

class FirecrawlWebScraper:
    """
    CLEANED: Focused Firecrawl scraper for quality restaurant extraction

    Single responsibility: Extract restaurant data from JavaScript-heavy sites
    No pre-filtering needed - that's handled by the smart scraper upstream
    """

    def __init__(self, config):
        self.config = config

        # Initialize Firecrawl with API key
        api_key = getattr(config, 'FIRECRAWL_API_KEY', None)
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in config")

        self.firecrawl = FirecrawlApp(api_key=api_key)

        # Rate limiting for expensive API calls
        self.semaphore = asyncio.Semaphore(3)  # Max 3 concurrent calls

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "restaurants_extracted": 0,
            "total_cost_estimate": 0.0,
            "avg_processing_time": 0.0,
            "error_types": {},
            "anti_bot_detected": 0
        }

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Extract restaurant data from a single URL
        SINGLE ATTEMPT POLICY - no retries to control costs
        """
        start_time = time.time()

        logger.info(f"🔥 Firecrawl extracting restaurants from: {url}")

        try:
            # Check for problematic domains that need special handling
            is_problematic = self._is_problematic_domain(url)

            if is_problematic:
                logger.warning(f"⚠️ Problematic domain detected: {url} - using enhanced settings")
                self.stats["anti_bot_detected"] += 1
                await asyncio.sleep(5)  # Extra delay for anti-bot domains

            # Single extraction attempt
            extraction_result = await self._extract_restaurants_single_attempt(url)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(extraction_result, processing_time)

            if extraction_result.success:
                # Format for editor (no more list_analyzer)
                formatted_content = self._format_restaurants_for_editor(extraction_result.restaurants)

                return {
                    "success": True,
                    "content": formatted_content,
                    "restaurants_found": [r.get("name", "") for r in extraction_result.restaurants],
                    "extraction_method": extraction_result.extraction_method,
                    "source_name": extraction_result.source_name,
                    "processing_time": processing_time
                }
            else:
                logger.error(f"❌ Firecrawl extraction failed for {url}: {extraction_result.error_message}")
                return {
                    "success": False,
                    "content": "",
                    "restaurants_found": [],
                    "error": extraction_result.error_message,
                    "extraction_method": "failed",
                    "processing_time": processing_time
                }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Firecrawl exception for {url}: {e}")

            # Track error types for debugging
            error_type = type(e).__name__
            self.stats["error_types"][error_type] = self.stats["error_types"].get(error_type, 0) + 1

            return {
                "success": False,
                "content": "",
                "restaurants_found": [],
                "error": str(e),
                "extraction_method": "failed",
                "processing_time": processing_time
            }

    async def _extract_restaurants_single_attempt(self, url: str) -> RestaurantExtractionResult:
        """
        Single attempt extraction using latest Firecrawl API
        POLICY: One attempt only to control expensive API costs
        """
        async with self.semaphore:
            try:
                logger.info(f"🔥 Firecrawl extraction attempt for: {url}")

                # Use Firecrawl's extraction API with restaurant schema
                extraction_result = self.firecrawl.scrape_url(
                    url=url,
                    params={
                        'formats': ['extract'],
                        'extract': {
                            'schema': RestaurantGuideExtraction.model_json_schema(),
                            'prompt': """Extract comprehensive restaurant information from this content.

                            For each restaurant, gather:
                            - Complete name and detailed description
                            - Full location details (address, city, neighborhood)
                            - Cuisine type and price range indicators
                            - Signature dishes, specialties, and menu highlights  
                            - Contact information (phone, website)
                            - Operating hours and special features
                            - Any ratings, awards, or notable mentions

                            Focus on quality over quantity - ensure each restaurant entry is complete and accurate.
                            Include brief restaurants mentioned in passing, but prioritize detailed entries."""
                        },
                        'timeout': 45000,  # 45 second timeout
                        'waitFor': 3000,   # Wait for dynamic content
                        'headers': {
                            'User-Agent': 'Mozilla/5.0 (compatible; RestaurantBot/1.0)'
                        }
                    }
                )

                # Process successful extraction
                if extraction_result.get('success') and extraction_result.get('extract'):
                    extracted_data = extraction_result['extract']
                    restaurants = extracted_data.get('restaurants', [])

                    if restaurants:
                        # Convert to standard dictionaries
                        restaurant_dicts = []
                        for restaurant in restaurants:
                            if isinstance(restaurant, dict):
                                restaurant_dicts.append(restaurant)
                            else:
                                # Handle Pydantic model conversion
                                restaurant_dicts.append(
                                    restaurant.dict() if hasattr(restaurant, 'dict') else restaurant
                                )

                        logger.info(f"✅ Successfully extracted {len(restaurant_dicts)} restaurants from {url}")

                        return RestaurantExtractionResult(
                            restaurants=restaurant_dicts,
                            source_url=url,
                            source_name=extracted_data.get('publication', 'Unknown Source'),
                            extraction_method="firecrawl_extraction",
                            success=True
                        )
                    else:
                        logger.warning(f"⚠️ No restaurants found in Firecrawl extraction for {url}")
                        return RestaurantExtractionResult(
                            restaurants=[],
                            source_url=url,
                            source_name="Unknown",
                            extraction_method="firecrawl_extraction",
                            success=False,
                            error_message="No restaurants found in extracted content"
                        )
                else:
                    # Try fallback basic scraping
                    logger.warning(f"Extraction API failed for {url}, trying basic scraping")
                    return await self._fallback_basic_scraping(url)

            except Exception as e:
                logger.error(f"Firecrawl extraction error for {url}: {e}")
                return RestaurantExtractionResult(
                    restaurants=[],
                    source_url=url,
                    source_name="Unknown",
                    extraction_method="firecrawl_failed",
                    success=False,
                    error_message=str(e)
                )

    async def _fallback_basic_scraping(self, url: str) -> RestaurantExtractionResult:
        """
        Fallback to basic Firecrawl scraping when extraction fails
        Returns content for processing by content sectioner
        """
        try:
            logger.info(f"🔄 Fallback: Basic Firecrawl scraping for {url}")

            # Basic content scraping
            scrape_result = self.firecrawl.scrape_url(
                url=url,
                params={
                    'formats': ['markdown'],
                    'onlyMainContent': True,
                    'removeBase64Images': True,
                    'timeout': 30000,
                    'waitFor': 2000
                }
            )

            if scrape_result.get('success') and scrape_result.get('markdown'):
                content = scrape_result['markdown']

                # Return raw content - will be processed by content sectioner in smart scraper
                return RestaurantExtractionResult(
                    restaurants=[],  # Empty - content will be processed upstream
                    source_url=url,
                    source_name=scrape_result.get('metadata', {}).get('title', 'Unknown'),
                    extraction_method="firecrawl_basic",
                    success=True
                )
            else:
                return RestaurantExtractionResult(
                    restaurants=[],
                    source_url=url,
                    source_name="Unknown",
                    extraction_method="firecrawl_basic",
                    success=False,
                    error_message="Basic scraping returned no content"
                )

        except Exception as e:
            logger.error(f"Fallback scraping failed for {url}: {e}")
            return RestaurantExtractionResult(
                restaurants=[],
                source_url=url,
                source_name="Unknown",
                extraction_method="firecrawl_failed",
                success=False,
                error_message=f"Fallback failed: {str(e)}"
            )

    def _is_problematic_domain(self, url: str) -> bool:
        """Detect domains known to have anti-bot protection"""
        problematic_patterns = [
            'resy.com',
            'opentable.com', 
            'thrillist.com',
            'seamless.com',
            'grubhub.com',
            'doordash.com',
            'ubereats.com'
        ]

        return any(pattern in url.lower() for pattern in problematic_patterns)

    def _format_restaurants_for_editor(self, restaurants: List[Dict[str, Any]]) -> str:
        """
        Format extracted restaurants for the editor agent (no more list_analyzer)
        Clean, structured format optimized for final editing
        """
        if not restaurants:
            return ""

        formatted_sections = []

        for i, restaurant in enumerate(restaurants, 1):
            name = restaurant.get('name', f'Restaurant {i}')
            description = restaurant.get('description', '')
            address = restaurant.get('address', '')
            city = restaurant.get('city', '')
            cuisine = restaurant.get('cuisine_type', '')
            price_range = restaurant.get('price_range', '')
            dishes = restaurant.get('recommended_dishes', [])
            phone = restaurant.get('phone', '')
            website = restaurant.get('website', '')
            hours = restaurant.get('opening_hours', '')
            features = restaurant.get('special_features', [])

            # Build clean, structured entry for editor
            entry_parts = [f"**{name}**"]

            if description:
                entry_parts.append(description)

            # Location and basic details
            details = []
            if cuisine:
                details.append(f"Cuisine: {cuisine}")
            if price_range:
                details.append(f"Price: {price_range}")
            if address and city:
                details.append(f"Location: {address}, {city}")
            elif city:
                details.append(f"Location: {city}")

            if details:
                entry_parts.append(" | ".join(details))

            # Recommended dishes (limit to avoid bloat)
            if dishes:
                dishes_text = ", ".join(dishes[:3])  # Top 3 dishes
                entry_parts.append(f"Recommended: {dishes_text}")

            # Contact and hours
            contact_info = []
            if phone:
                contact_info.append(f"Phone: {phone}")
            if website:
                contact_info.append(f"Website: {website}")
            if hours:
                contact_info.append(f"Hours: {hours}")

            if contact_info:
                entry_parts.append(" | ".join(contact_info))

            # Special features
            if features:
                features_text = ", ".join(features[:2])  # Top 2 features
                entry_parts.append(f"Features: {features_text}")

            formatted_sections.append("\n".join(entry_parts))

        return "\n\n".join(formatted_sections)

    def _update_stats(self, result: RestaurantExtractionResult, processing_time: float):
        """Update performance statistics"""
        self.stats["total_processed"] += 1

        if result.success:
            self.stats["successful_extractions"] += 1
            self.stats["restaurants_extracted"] += len(result.restaurants)
        else:
            self.stats["failed_extractions"] += 1

        # Update average processing time
        current_avg = self.stats["avg_processing_time"]
        total_processed = self.stats["total_processed"]
        self.stats["avg_processing_time"] = (current_avg * (total_processed - 1) + processing_time) / total_processed

        # Update cost estimate (10 credits per Firecrawl call)
        self.stats["total_cost_estimate"] += 10.0

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics with calculated metrics"""
        stats = self.stats.copy()

        # Calculate additional metrics
        total = stats["total_processed"]
        if total > 0:
            stats["success_rate"] = (stats["successful_extractions"] / total) * 100
            stats["avg_restaurants_per_success"] = stats["restaurants_extracted"] / max(stats["successful_extractions"], 1)
        else:
            stats["success_rate"] = 0
            stats["avg_restaurants_per_success"] = 0

        return stats

    def log_stats(self):
        """Log comprehensive statistics"""
        stats = self.get_stats()

        logger.info("=" * 50)
        logger.info("🔥 FIRECRAWL SCRAPER STATISTICS")
        logger.info("=" * 50)
        logger.info(f"📊 Total Processed: {stats['total_processed']}")
        logger.info(f"✅ Successful: {stats['successful_extractions']}")
        logger.info(f"❌ Failed: {stats['failed_extractions']}")
        logger.info(f"🎯 Success Rate: {stats['success_rate']:.1f}%")
        logger.info(f"🍽️ Restaurants Extracted: {stats['restaurants_extracted']}")
        logger.info(f"📈 Avg Restaurants/Success: {stats['avg_restaurants_per_success']:.1f}")
        logger.info(f"⏱️ Avg Processing Time: {stats['avg_processing_time']:.1f}s")
        logger.info(f"💰 Total Cost Estimate: {stats['total_cost_estimate']:.1f} credits")

        if stats['error_types']:
            logger.info("\n🐛 Error Breakdown:")
            for error_type, count in stats['error_types'].items():
                logger.info(f"   {error_type}: {count}")

        if stats['anti_bot_detected'] > 0:
            logger.info(f"🛡️ Anti-bot Protection Detected: {stats['anti_bot_detected']} times")

        logger.info("=" * 50)

    async def cleanup(self):
        """Cleanup resources and log final stats"""
        try:
            self.log_stats()
            logger.info("🔥 Firecrawl scraper cleanup completed")
        except Exception as e:
            logger.error(f"Error during Firecrawl cleanup: {e}")