# agents/smart_scraper.py - UPDATED VERSION with Human Mimic Integration
"""
Smart Scraper with Human Mimic Integration
Replaces Firecrawl with Human Mimic for content sites without CAPTCHA

Strategy hierarchy:
🆓 SPECIALIZED (RSS/Sitemap) - 0.0 credits
🟢 SIMPLE_HTTP - 0.1 credits  
🟡 ENHANCED_HTTP - 0.5 credits
🎭 HUMAN_MIMIC - 2.0 credits (NEW - replaces most Firecrawl usage)
🔴 FIRECRAWL - 10.0 credits (only for heavily protected sites)
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from enum import Enum
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
from readability import Document

from agents.specialized_scraper import EaterTimeoutSpecializedScraper
from agents.human_mimic_scraper import HumanMimicScrapingStrategy
from agents.content_sectioning_agent import ContentSectioningAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.database import get_database

logger = logging.getLogger(__name__)

class ScrapingStrategy(Enum):
    """Updated scraping strategies with Human Mimic"""
    SPECIALIZED = "specialized"      # FREE - RSS/Sitemap
    SIMPLE_HTTP = "simple_http"      # 0.1 credits
    ENHANCED_HTTP = "enhanced_http"  # 0.5 credits
    HUMAN_MIMIC = "human_mimic"      # 2.0 credits - NEW!

class SmartRestaurantScraper:
    """
    AI-Powered Smart Scraper with Human Mimic Integration

    NEW Strategy Flow:
    1. Specialized handlers (free) 
    2. Simple HTTP for static content
    3. Enhanced HTTP for light JS
    4. Human Mimic for dynamic content (REPLACES most Firecrawl)
    """

    def __init__(self, config):
        self.config = config
        self.database = get_database()

        # AI analyzer for strategy classification
        self.analyzer = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Lazy initialization of scrapers
        self._specialized_scraper = None
        self._human_mimic_scraper = None
        self._content_sectioner = None

        # Initialize Text Cleaner Agent - FIXED: Remove duplicate line
        from agents.text_cleaner_agent import TextCleanerAgent
        self._text_cleaner = TextCleanerAgent(config, model_override='deepseek')  # Start with DeepSeek

        # Updated stats tracking with Human Mimic
        self.stats = {
            "total_processed": 0,
            "strategy_breakdown": {strategy.value: 0 for strategy in ScrapingStrategy},
            "ai_analysis_calls": 0,
            "domain_cache_hits": 0,
            "new_domains_learned": 0,
            "total_cost_estimate": 0.0
        }

        # Updated AI prompt to include Human Mimic strategy
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
    You are an expert web crawling strategist. Recommend the optimal scraping strategy based on technical complexity.

    Available strategies:
    🟢 SIMPLE_HTTP (0.1 credits):
    - Static HTML pages with content in initial server response
    - Minimal JavaScript required
    - News sites, blogs, simple restaurant pages

    🟡 ENHANCED_HTTP (0.5 credits):  
    - Moderate client-side scripting
    - Content mostly in HTML but needs cleaning
    - CMS platforms with some dynamic elements

    🎭 HUMAN_MIMIC (2.0 credits):
    - Dynamic content that requires JavaScript execution
    - Content loads after page render (common for restaurant sites)
    - Modern sites without anti-bot protection
    - SPAs with moderate complexity
    - Perfect for: Timeout, Eater, Michelin, most restaurant guides

    Output in JSON format:
    {{
        "strategy": "SIMPLE_HTTP|ENHANCED_HTTP|HUMAN_MIMIC",
        "confidence": 0.0-1.0,
        "reasoning": "Technical explanation"
    }}
            """),
            ("human", """
    Analyze this URL and recommend scraping strategy:
    URL: {{url}}
    Domain: {{domain}}
    Title: {{title}}
    Content preview: {{content_preview}}
    JavaScript indicators: {{js_indicators}}

    Focus on technical complexity and anti-bot measures, not content topic.
            """)
        ])

        logger.info("✅ SmartRestaurantScraper initialized with Human Mimic strategy")

    @property
    def specialized_scraper(self):
        if self._specialized_scraper is None:
            self._specialized_scraper = EaterTimeoutSpecializedScraper(self.config)
        return self._specialized_scraper

    @property
    def human_mimic_scraper(self):
        """NEW: Human Mimic scraper for content sites"""
        if self._human_mimic_scraper is None:
            self._human_mimic_scraper = HumanMimicScrapingStrategy(self.config)
        return self._human_mimic_scraper


    @property
    def content_sectioner(self):
        if self._content_sectioner is None:
            self._content_sectioner = ContentSectioningAgent(self.config)
        return self._content_sectioner

    async def scrape_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main scraping method with Human Mimic integration
        """
        if not search_results:
            return []

        logger.info(f"🤖 Smart scraper processing {len(search_results)} URLs with Human Mimic strategy")

        # Step 1: Classify all URLs by strategy
        classified_results = await self._classify_all_urls(search_results)

        # Step 2: Group by strategy and process
        enriched_results = []
        strategy_groups = {}

        for result in classified_results:
            strategy = result.get("scrape_strategy", ScrapingStrategy.HUMAN_MIMIC)  # Default to Human Mimic
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(result)

        # Process each strategy group
        for strategy, results in strategy_groups.items():
            logger.info(f"📊 Processing {len(results)} URLs with {strategy.value}")

            if strategy == ScrapingStrategy.SPECIALIZED:
                scraped = await self._scrape_specialized(results)
            elif strategy == ScrapingStrategy.SIMPLE_HTTP:
                scraped = await self._scrape_simple_http(results)
            elif strategy == ScrapingStrategy.ENHANCED_HTTP:
                scraped = await self._scrape_enhanced_http(results)
            elif strategy == ScrapingStrategy.HUMAN_MIMIC:
                scraped = await self._scrape_human_mimic(results)  # NEW!
            else:
                scraped = results

            enriched_results.extend(scraped)

            # Update stats
            self.stats["strategy_breakdown"][strategy.value] += len(results)
            self.stats["total_processed"] += len(results)

            # Track Human Mimic replacements of Firecrawl
            if strategy == ScrapingStrategy.HUMAN_MIMIC:
                self.stats["human_mimic_replacements"] += len(results)

        # Calculate costs with new strategy
        self._calculate_cost_savings()

        successful = sum(1 for r in enriched_results if r.get('scraping_success'))
        logger.info(f"✅ Smart scraping complete: {successful}/{len(enriched_results)} successful")

        return enriched_results

    async def _scrape_human_mimic(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        UPDATED: Scrape using Human Mimic strategy with Text Cleaner integration
        """
        logger.info(f"🎭 Human Mimic scraping {len(results)} URLs")

        try:
            # Step 1: Get raw scraped content
            scraped_results = await self.human_mimic_scraper.scrape_search_results(results)

            # Step 2: NEW - Clean the content with Text Cleaner Agent
            if hasattr(self, '_text_cleaner') and self._text_cleaner:
                logger.info("🧹 Applying Text Cleaner to Human Mimic results")

                # Filter successful scrapes for cleaning
                successful_scrapes = [r for r in scraped_results if r.get('scraping_success')]

                if successful_scrapes:
                    # Clean the content
                    cleaned_content = self._text_cleaner.clean_scraped_results(successful_scrapes)

                    # Add cleaned content to results
                    for result in scraped_results:
                        if result.get('scraping_success'):
                            result['cleaned_content'] = cleaned_content
                            result['content_cleaning_applied'] = True
                        else:
                            result['cleaned_content'] = ""
                            result['content_cleaning_applied'] = False

            # Step 3: Process through content sectioner for restaurant extraction
            final_results = []
            for result in scraped_results:
                if result.get('scraping_success') and result.get('scraped_content'):
                    # Use cleaned content if available, otherwise original
                    content_to_process = result.get('cleaned_content') or result.get('scraped_content')

                    # Extract restaurants from content
                    try:
                        if hasattr(self, '_content_sectioner') and self._content_sectioner:
                            sectioned_content = await self._content_sectioner.section_content(content_to_process)
                            result['restaurants_found'] = sectioned_content.get('restaurants', [])
                        else:
                            # Fallback: mark for editor processing
                            result['restaurants_found'] = []
                            result['needs_editor_processing'] = True

                    except Exception as e:
                        logger.warning(f"Content sectioning failed for {result.get('url')}: {e}")
                        result['restaurants_found'] = []

                final_results.append(result)

            return final_results

        except Exception as e:
            logger.error(f"Human Mimic scraping batch failed: {e}")
            # Return results with error status
            return [{**r, 'scraping_success': False, 'scraping_error': str(e)} for r in results]


    async def _classify_all_urls(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify URLs with updated Human Mimic strategy logic
        """
        classified = []

        for result in search_results:
            url = result.get('url', '')
            domain = urlparse(url).netloc

            # Check specialized first
            if await self._is_specialized_url(url):
                result['scrape_strategy'] = ScrapingStrategy.SPECIALIZED
                classified.append(result)
                continue

            # Check domain cache
            cached_strategy = await self._get_cached_strategy(domain)
            if cached_strategy:
                result['scrape_strategy'] = cached_strategy
                self.stats["domain_cache_hits"] += 1
                classified.append(result)
                continue

            # AI classification with Human Mimic preference
            strategy = await self._ai_classify_url(result)
            result['scrape_strategy'] = strategy

            # Cache the decision
            await self._cache_strategy(domain, strategy)
            self.stats["ai_analysis_calls"] += 1
            self.stats["new_domains_learned"] += 1

            classified.append(result)

        return classified

    async def _ai_classify_url(self, result: Dict[str, Any]) -> ScrapingStrategy:
        """
        AI classification with bias toward Human Mimic for content sites
        """
        try:
            url = result.get('url', '')
            domain = urlparse(url).netlhost
            title = result.get('title', '')
            content_preview = result.get('content_preview', '')

            # Simple JS detection
            js_indicators = self._detect_js_indicators(content_preview)

            # Format prompt
            response = await self.analyzer.ainvoke(
                self.analysis_prompt.format_messages(
                    url=url,
                    domain=domain,
                    title=title,
                    content_preview=content_preview[:500],
                    js_indicators=js_indicators
                )
            )

            # Parse AI response
            try:
                analysis = json.loads(response.content)
                strategy_name = analysis.get('strategy', 'HUMAN_MIMIC')

                # Map string to enum, default to Human Mimic
                strategy_mapping = {
                    'SIMPLE_HTTP': ScrapingStrategy.SIMPLE_HTTP,
                    'ENHANCED_HTTP': ScrapingStrategy.ENHANCED_HTTP,
                    'HUMAN_MIMIC': ScrapingStrategy.HUMAN_MIMIC
                }

                strategy = strategy_mapping.get(strategy_name, ScrapingStrategy.HUMAN_MIMIC)

                # Log AI decision for monitoring
                logger.debug(f"AI classified {domain}: {strategy.value} (confidence: {analysis.get('confidence', 0)})")

                return strategy

            except json.JSONDecodeError:
                logger.warning(f"Could not parse AI response for {url}, defaulting to Human Mimic")
                return ScrapingStrategy.HUMAN_MIMIC

        except Exception as e:
            logger.error(f"AI classification failed for {url}: {e}, defaulting to Human Mimic")
            return ScrapingStrategy.HUMAN_MIMIC

    def _detect_js_indicators(self, content_preview: str) -> str:
        """Detect JavaScript complexity indicators in content"""
        js_patterns = [
            r'<script[^>]*>',
            r'React\.|Vue\.|Angular',
            r'data-react-|data-vue-',
            r'__NEXT_DATA__|__nuxt',
            r'window\.__INITIAL_STATE__',
            r'defer|async',
            r'addEventListener',
            r'document\.ready'
        ]

        indicators = []
        for pattern in js_patterns:
            if re.search(pattern, content_preview, re.IGNORECASE):
                indicators.append(pattern.split('|')[0])  # First part of pattern

        return f"Found {len(indicators)} JS indicators: {', '.join(indicators[:3])}" if indicators else "Minimal JS detected"

    async def _cache_strategy(self, domain: str, strategy: ScrapingStrategy):
        """Cache strategy decision for domain"""
        try:
            self.database.table('domain_intelligence').upsert({
                'domain': domain,
                'strategy': strategy.value,
                'learned_at': time.time(),
                'source': 'smart_scraper_ai'
            }).execute()
        except Exception as e:
            logger.debug(f"Could not cache strategy for {domain}: {e}")

    async def _get_cached_strategy(self, domain: str) -> Optional[ScrapingStrategy]:
        """Get cached strategy for domain"""
        try:
            result = self.database.table('domain_intelligence').select('strategy').eq('domain', domain).execute()
            if result.data and len(result.data) > 0:
                strategy_name = result.data[0].get('strategy')
                strategy_mapping = {
                    'specialized': ScrapingStrategy.SPECIALIZED,
                    'simple_http': ScrapingStrategy.SIMPLE_HTTP,
                    'enhanced_http': ScrapingStrategy.ENHANCED_HTTP,
                    'human_mimic': ScrapingStrategy.HUMAN_MIMIC
                }
                return strategy_mapping.get(strategy_name)
        except Exception as e:
            logger.debug(f"Could not get cached strategy for {domain}: {e}")
        return None

    async def _is_specialized_url(self, url: str) -> bool:
        """Check if URL can be handled by specialized scrapers"""
        try:
            return await self.specialized_scraper.can_handle(url)
        except Exception as e:
            logger.debug(f"Error checking specialized handler for {url}: {e}")
            return False

    async def _scrape_specialized(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scrape using specialized scrapers (RSS/Sitemap)"""
        logger.info(f"🆓 Specialized scraping {len(results)} URLs")

        try:
            return await self.specialized_scraper.scrape_search_results(results)
        except Exception as e:
            logger.error(f"Specialized scraping failed: {e}")
            return [{**r, 'scraping_success': False, 'scraping_error': str(e)} for r in results]

    async def _scrape_simple_http(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scrape using simple HTTP requests"""
        logger.info(f"🟢 Simple HTTP scraping {len(results)} URLs")

        semaphore = asyncio.Semaphore(5)  # Limit concurrency

        async def scrape_single(result):
            async with semaphore:
                return await self._simple_http_scrape(result)

        tasks = [scrape_single(result) for result in results]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _simple_http_scrape(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Simple HTTP scrape with basic text extraction"""
        url = result.get('url', '')

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; RestaurantBot/1.0)'
                })

                if response.status_code == 200:
                    # Extract text using BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Remove script and style elements
                    for element in soup(["script", "style", "nav", "footer"]):
                        element.decompose()

                    text_content = soup.get_text(separator=' ', strip=True)

                    # Limit content length
                    if len(text_content) > 6000:
                        text_content = text_content[:6000] + "..."

                    result.update({
                        'scraped_content': text_content,
                        'scraping_success': True,
                        'scraping_method': 'simple_http',
                        'char_count': len(text_content)
                    })
                else:
                    result.update({
                        'scraped_content': "",
                        'scraping_success': False,
                        'scraping_method': 'simple_http',
                        'scraping_error': f"HTTP {response.status_code}"
                    })

        except Exception as e:
            result.update({
                'scraped_content': "",
                'scraping_success': False,
                'scraping_method': 'simple_http',
                'scraping_error': str(e)
            })

        return result

    async def _scrape_enhanced_http(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced HTTP scraping with readability extraction"""
        logger.info(f"🟡 Enhanced HTTP scraping {len(results)} URLs")

        semaphore = asyncio.Semaphore(3)  # Lower concurrency for enhanced processing

        async def scrape_single(result):
            async with semaphore:
                return await self._enhanced_http_scrape(result)

        tasks = [scrape_single(result) for result in results]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _enhanced_http_scrape(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced HTTP scrape with readability processing"""
        url = result.get('url', '')

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; RestaurantBot/1.0)'
                })

                if response.status_code == 200:
                    # Use readability to extract main content
                    doc = Document(response.text)

                    # Get title and content
                    title = doc.title()
                    content = doc.summary()

                    # Extract text from HTML
                    soup = BeautifulSoup(content, 'html.parser')
                    text_content = soup.get_text(separator=' ', strip=True)

                    # Combine title and content
                    full_content = f"{title}\n\n{text_content}" if title else text_content

                    # Limit content length
                    if len(full_content) > 8000:
                        full_content = full_content[:8000] + "..."

                    result.update({
                        'scraped_content': full_content,
                        'scraping_success': True,
                        'scraping_method': 'enhanced_http',
                        'char_count': len(full_content)
                    })
                else:
                    result.update({
                        'scraped_content': "",
                        'scraping_success': False,
                        'scraping_method': 'enhanced_http',
                        'scraping_error': f"HTTP {response.status_code}"
                    })

        except Exception as e:
            result.update({
                'scraped_content': "",
                'scraping_success': False,
                'scraping_method': 'enhanced_http',
                'scraping_error': str(e)
            })

        return result


    def _calculate_cost_savings(self):
        """Calculate cost savings with Human Mimic strategy"""
        # Cost per URL for each strategy
        cost_map = {
            ScrapingStrategy.SPECIALIZED: 0.0,
            ScrapingStrategy.SIMPLE_HTTP: 0.1,
            ScrapingStrategy.ENHANCED_HTTP: 0.5,
            ScrapingStrategy.HUMAN_MIMIC: 2.0
        }

        # Calculate actual cost
        actual_cost = 0.0
        for strategy, count in self.stats["strategy_breakdown"].items():
            strategy_enum = ScrapingStrategy(strategy)
            actual_cost += count * cost_map[strategy_enum]

        self.stats["total_cost_estimate"] = actual_cost

        # Calculate savings vs all Firecrawl
        total_urls = self.stats["total_processed"]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including Human Mimic metrics"""
        stats = self.stats.copy()

        # Add Human Mimic specific metrics
        if hasattr(self, '_human_mimic_scraper') and self._human_mimic_scraper:
            human_mimic_stats = self._human_mimic_scraper.get_stats()
            stats["human_mimic_details"] = human_mimic_stats

        return stats

    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            "total_processed": 0,
            "strategy_breakdown": {strategy.value: 0 for strategy in ScrapingStrategy},
            "ai_analysis_calls": 0,
            "domain_cache_hits": 0,
            "new_domains_learned": 0,
            "total_cost_estimate": 0.0,
            "human_mimic_replacements": 0
        }

        # Reset Human Mimic scraper stats
        if hasattr(self, '_human_mimic_scraper') and self._human_mimic_scraper:
            self._human_mimic_scraper.reset_stats()