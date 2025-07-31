# agents/search_agent.py
"""
Enhanced Search Agent with Parallel Brave + Tavily Search and AI Filtering

Features:
- Parallel search: Brave Search for English queries, Tavily for local language queries  
- AI-based content evaluation and filtering
- Preview fetching for all results
- Database storage of high-quality sources
- Seamless integration with existing orchestrator pipeline
"""

import requests
import asyncio
import aiohttp
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
import json
import logging
import time
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("restaurant-recommender.search_agent")

class BraveSearchAgent:
    """
    Enhanced search agent with parallel search capabilities and AI filtering
    Maintains compatibility with existing orchestrator while adding new features
    """

    def __init__(self, config):
        self.config = config

        # API Configuration
        self.brave_api_key = config.BRAVE_API_KEY
        self.tavily_api_key = getattr(config, 'TAVILY_API_KEY', None)
        self.search_count = config.BRAVE_SEARCH_COUNT

        # Search URLs
        self.brave_base_url = "https://api.search.brave.com/res/v1/web/search"
        self.tavily_base_url = "https://api.tavily.com/search"

        # Filtering Configuration
        self.excluded_domains = config.EXCLUDED_RESTAURANT_SOURCES

        # Initialize AI for content evaluation (using exact prompt you provided)
        self.eval_model = ChatOpenAI(
            model=config.OPENAI_MODEL,  # Using GPT-4o as requested
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Your exact filtering prompt - DO NOT CHANGE
        self.eval_system_prompt = """
        You are an expert at evaluating web content about restaurants.
        Your task is to analyze a web page's content and rate it using the criteria below.

        PRIORITIZE THESE SOURCES (score 0.8-1.0):
        - Established food and travel publications (e.g., Conde Nast Traveler, Forbes Travel, Food & Wine, Bon Appétit, etc.)
        - Local newspapers and magazines (like Expresso.pt, Le Monde, El Pais, Time Out)
        - Professional food critics and culinary experts
        - Reputable local food blogs (Katie Parla for Italy, 2Foodtrippers, David Leibovitz for Paris, etc.)
        - Local tourism boards and official regional and city guides
        - Restaurant guides and gastronomic awards (Michelin, The World's 50 Best, World of Mouth)
        VALID CONTENT (score 0.6-0.8):
        - Curated lists of multiple restaurants (e.g., "Top 10 restaurants in Paris", "Best artisanal pizza in Rome", etc.)
        - Food critic reviews of a single restaurant, but ONLY in professional media
        - Articles in reputable local media discussing various dining options in an area
        - Food blog articles with restaurant recommendations
        - Travel articles mentioning multiple dining options
        NOT VALID CONTENT (score < 0.3):
        - Official website of a single restaurant
        - ANYTHING on Tripadvisor, Yelp, OpenTable, RestaurantGuru and other UGC sites
        - Collections of restaurants on booking and delivery websites like Uber Eats, The Fork, Glovo, Bolt, Wolt, Mesa24, etc.
        - Social media content on Facebook and Instagram without professional curation
        - ANY Wanderlog content
        - Individual restaurant menus
        - Single restaurant reviews
        - Social media posts about individual dining experiences
        - Forum/Reddit discussions without professional curation
        - Hotel booking sites
        - Video content (YouTube, TikTok, etc.)
        SCORING CRITERIA:
        - Multiple restaurants mentioned (essential, with the only exception of single restaurant reviews in professional media)
        - Professional curation or expertise evident
        - Local expertise and knowledge
        - Detailed professional descriptions of restaurants/cuisine
        FORMAT:
        Respond with a JSON object containing:
        {{
          "is_restaurant_list": true/false,
          "restaurant_count": estimated number of restaurants mentioned,
          "content_quality": 0.0-1.0,
          "passed_filter": true/false,
          "reasoning": "brief explanation of your decision"
        }}
        """

        # Set up evaluation prompt
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.eval_system_prompt),
            ("human", "URL: {url}\nTitle: {title}\nDescription: {description}\nContent Preview: {content_preview}")
        ])

        # Create evaluation chain
        self.eval_chain = self.eval_prompt | self.eval_model

        # Statistics tracking
        self.stats = {
            "brave_searches": 0,
            "tavily_searches": 0,
            "total_results": 0,
            "filtered_results": 0,
            "high_quality_sources": 0,
            "database_saves": 0
        }

        logger.info("✅ Enhanced Search Agent initialized with parallel search and AI filtering")

    # Replace the search method in your agents/search_agent.py with this fixed version

    def search(self, search_queries: List[str], destination: str, query_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute search queries and return filtered results
        FIXED: Handles event loop conflicts properly

        Args:
            search_queries: List of search queries
            destination: Target destination/city 
            query_metadata: Query analyzer metadata for intelligent routing

        Returns:
            List of filtered and evaluated search results
        """
        logger.info(f"🔍 Starting parallel search for destination: {destination}")
        logger.info(f"📋 Search queries: {search_queries}")

        if query_metadata:
            logger.info(f"🧠 Using query analyzer metadata: {query_metadata.get('is_english_speaking', 'unknown')} speaking, local language: {query_metadata.get('local_language', 'none')}")

        try:
            # FIXED: Check if event loop is already running
            try:
                # Try to get the current running loop
                loop = asyncio.get_running_loop()
                # If we get here, a loop is already running - use different approach
                logger.info("🔄 Event loop already running - using concurrent execution")

                # Run in thread pool to avoid loop conflict
                import concurrent.futures
                import threading

                def run_async_search():
                    # Create new loop in this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self._parallel_search_and_filter(search_queries, destination, query_metadata)
                        )
                    finally:
                        new_loop.close()

                # Execute in thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_async_search)
                    results = future.result(timeout=60)  # 60 second timeout

                return results

            except RuntimeError:
                # No event loop is running - create one
                logger.info("🆕 No event loop running - creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        self._parallel_search_and_filter(search_queries, destination, query_metadata)
                    )
                    return results
                finally:
                    loop.close()

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    async def _parallel_search_and_filter(self, search_queries: List[str], destination: str, query_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute parallel searches and filtering pipeline
        """
        logger.info("🚀 Starting parallel search and filtering pipeline")

        # Step 1: Determine query languages using query analyzer metadata
        english_queries, local_queries = self._categorize_queries(search_queries, destination, query_metadata)

        # Step 2: Parallel search execution
        all_results = []

        # Execute searches in parallel
        search_tasks = []

        if english_queries:
            logger.info(f"🇺🇸 Brave Search queries: {english_queries}")
            search_tasks.append(self._brave_search_batch(english_queries))

        if local_queries and self.tavily_api_key:
            logger.info(f"🌍 Tavily Search queries: {local_queries}")
            search_tasks.append(self._tavily_search_batch(local_queries))
        elif local_queries:
            logger.warning("📋 Local language queries found but Tavily API key not configured")
            # Fall back to Brave for local queries
            search_tasks.append(self._brave_search_batch(local_queries))

        # Execute parallel searches
        if search_tasks:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Combine results
            for result_set in search_results:
                if isinstance(result_set, Exception):
                    logger.error(f"❌ Search batch failed: {result_set}")
                    continue
                if isinstance(result_set, list):
                    all_results.extend(result_set)

        logger.info(f"📊 Total search results before filtering: {len(all_results)}")

        # Step 3: Fetch previews for all results
        results_with_previews = await self._fetch_previews_batch(all_results)

        # Step 4: AI-based filtering
        filtered_results = await self._filter_results_batch(results_with_previews, destination)

        # Step 5: Store high-quality sources in database
        await self._store_quality_sources(filtered_results, destination)

        logger.info(f"✅ Final filtered results: {len(filtered_results)}")
        return filtered_results

    def _categorize_queries(self, search_queries: List[str], destination: str, query_metadata: Dict[str, Any] = None) -> Tuple[List[str], List[str]]:
        """
        Categorize queries into English (Brave) and local language (Tavily)
        Uses AI-powered query analyzer metadata for intelligent categorization
        """
        english_queries = []
        local_queries = []

        # Always prefer query analyzer metadata (AI-powered decision)
        if query_metadata:
            is_english_speaking = query_metadata.get('is_english_speaking', True)
            has_local_language = query_metadata.get('local_language') is not None

            logger.info(f"🧠 Using AI-powered language detection: English-speaking={is_english_speaking}, Local language={query_metadata.get('local_language', 'None')}")

            if is_english_speaking:
                # AI determined this is English-speaking: all queries go to Brave
                english_queries = search_queries.copy()
                logger.info(f"🇺🇸 AI-detected English-speaking destination: {len(english_queries)} queries → Brave Search")
            else:
                # AI determined this is non-English: separate by query content
                for query in search_queries:
                    if self._is_likely_english(query):
                        english_queries.append(query)
                    else:
                        local_queries.append(query)

                # If query analyzer provided both English and local queries but all ended up as English
                if has_local_language and len(local_queries) == 0:
                    # Split for better coverage since we know there should be local content
                    mid = len(english_queries) // 2
                    local_queries = english_queries[mid:]
                    english_queries = english_queries[:mid]

                logger.info(f"🌍 AI-detected non-English destination: {len(english_queries)} → Brave, {len(local_queries)} → Tavily")
        else:
            # Fallback: simple content-based categorization only
            logger.warning("⚠️ No AI language metadata available - using content-based fallback")

            for query in search_queries:
                if self._is_likely_english(query):
                    english_queries.append(query)
                else:
                    local_queries.append(query)

            # Without AI guidance, default to English search only
            if len(local_queries) == 0:
                english_queries = search_queries.copy()
                logger.info(f"📝 Fallback: All {len(english_queries)} queries → Brave Search")

        return english_queries, local_queries

    def _is_likely_english(self, query: str) -> bool:
        """Simple heuristic to detect English queries based on character content"""
        # Check for non-ASCII characters (simple indicator of non-English)
        try:
            query.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False

    async def _brave_search_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute Brave Search for multiple queries"""
        logger.info(f"🔍 Executing Brave Search for {len(queries)} queries")

        all_results = []

        async with aiohttp.ClientSession() as session:
            tasks = [self._brave_search_single(session, query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Brave search failed for query {i}: {result}")
                    continue
                if isinstance(result, list):
                    all_results.extend(result)

        self.stats["brave_searches"] += len(queries)
        logger.info(f"✅ Brave Search completed: {len(all_results)} results")
        return all_results

    async def _brave_search_single(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        """Execute single Brave search"""
        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key
            }

            params = {
                "q": query,
                "count": self.search_count,
                "freshness": "month"
            }

            async with session.get(self.brave_base_url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"❌ Brave API error: {response.status}")
                    return []

                data = await response.json()
                return self._parse_brave_results(data, query)

        except Exception as e:
            logger.error(f"❌ Brave search error for '{query}': {e}")
            return []

    def _parse_brave_results(self, data: Dict, query: str) -> List[Dict[str, Any]]:
        """Parse Brave search API response"""
        results = []

        web_results = data.get('web', {}).get('results', [])

        for result in web_results:
            # Basic domain filtering
            url = result.get('url', '')
            if self._should_exclude_domain(url):
                continue

            parsed_result = {
                'url': url,
                'title': result.get('title', ''),
                'description': result.get('description', ''),
                'source_engine': 'brave',
                'search_query': query,
                'favicon': result.get('favicon', ''),
                'language': result.get('language', 'en')
            }
            results.append(parsed_result)

        return results

    async def _tavily_search_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute Tavily Search for multiple queries"""
        logger.info(f"🌍 Executing Tavily Search for {len(queries)} queries")

        all_results = []

        async with aiohttp.ClientSession() as session:
            tasks = [self._tavily_search_single(session, query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Tavily search failed for query {i}: {result}")
                    continue
                if isinstance(result, list):
                    all_results.extend(result)

        self.stats["tavily_searches"] += len(queries)
        logger.info(f"✅ Tavily Search completed: {len(all_results)} results")
        return all_results

    async def _tavily_search_single(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        """Execute single Tavily search"""
        try:
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": self.search_count
            }

            async with session.post(self.tavily_base_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"❌ Tavily API error: {response.status}")
                    return []

                data = await response.json()
                return self._parse_tavily_results(data, query)

        except Exception as e:
            logger.error(f"❌ Tavily search error for '{query}': {e}")
            return []

    def _parse_tavily_results(self, data: Dict, query: str) -> List[Dict[str, Any]]:
        """Parse Tavily search API response"""
        results = []

        tavily_results = data.get('results', [])

        for result in tavily_results:
            # Basic domain filtering
            url = result.get('url', '')
            if self._should_exclude_domain(url):
                continue

            parsed_result = {
                'url': url,
                'title': result.get('title', ''),
                'description': result.get('content', ''),  # Tavily uses 'content' field
                'source_engine': 'tavily',
                'search_query': query,
                'score': result.get('score', 0.0)
            }
            results.append(parsed_result)

        return results

    def _should_exclude_domain(self, url: str) -> bool:
        """Check if URL domain should be excluded"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return any(excluded in domain for excluded in self.excluded_domains)
        except Exception:
            return False

    async def _fetch_previews_batch(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch content previews for all search results"""
        logger.info(f"📖 Fetching previews for {len(results)} results")

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            tasks = [self._fetch_single_preview(session, result) for result in results]
            previews = await asyncio.gather(*tasks, return_exceptions=True)

            results_with_previews = []
            for i, (result, preview) in enumerate(zip(results, previews)):
                if isinstance(preview, Exception):
                    logger.debug(f"Preview fetch failed for {result.get('url', 'unknown')}: {preview}")
                    result['content_preview'] = result.get('description', '')
                else:
                    result['content_preview'] = preview or result.get('description', '')

                results_with_previews.append(result)

        logger.info(f"✅ Preview fetching completed")
        return results_with_previews

    async def _fetch_single_preview(self, session: aiohttp.ClientSession, result: Dict[str, Any]) -> str:
        """Fetch content preview for a single URL"""
        url = result.get('url', '')

        try:
            async with session.get(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; RestaurantBot/1.0)'}) as response:
                if response.status != 200:
                    return ""

                # Only read first part of content for preview
                content = await response.text()

                # Extract text using BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text and limit length
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                # Return first 500 characters as preview
                return text[:500]

        except Exception as e:
            logger.debug(f"Failed to fetch preview for {url}: {e}")
            return ""

    async def _filter_results_batch(self, results: List[Dict[str, Any]], destination: str) -> List[Dict[str, Any]]:
        """Filter results using AI evaluation in batches"""
        logger.info(f"🧠 AI filtering {len(results)} results")

        # Process in smaller batches to avoid overwhelming the API
        batch_size = 5
        filtered_results = []

        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            batch_filtered = await self._evaluate_batch(batch, destination)
            filtered_results.extend(batch_filtered)

            # Small delay between batches
            await asyncio.sleep(0.1)

        self.stats["total_results"] = len(results)
        self.stats["filtered_results"] = len(filtered_results)

        logger.info(f"✅ AI filtering completed: {len(filtered_results)}/{len(results)} passed")
        return filtered_results

    async def _evaluate_batch(self, batch: List[Dict[str, Any]], destination: str) -> List[Dict[str, Any]]:
        """Evaluate a batch of results using AI"""
        filtered_batch = []

        # Create evaluation tasks
        tasks = []
        for result in batch:
            task = self._evaluate_single_result(result, destination)
            tasks.append(task)

        # Execute evaluations in parallel
        evaluations = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result, evaluation in zip(batch, evaluations):
            if isinstance(evaluation, Exception):
                logger.error(f"❌ Evaluation failed for {result.get('url', 'unknown')}: {evaluation}")
                continue

            if evaluation and evaluation.get('passed_filter', False):
                # Add evaluation metadata
                result['ai_evaluation'] = evaluation
                result['quality_score'] = evaluation.get('content_quality', 0.0)
                filtered_batch.append(result)

        return filtered_batch

    async def _evaluate_single_result(self, result: Dict[str, Any], destination: str) -> Optional[Dict[str, Any]]:
        """Evaluate a single result using AI"""
        try:
            # Prepare data for evaluation
            evaluation_data = {
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'description': result.get('description', ''),
                'content_preview': result.get('content_preview', '')
            }

            # Run AI evaluation
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.eval_chain.invoke(evaluation_data)
            )

            # Parse AI response
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Try to parse JSON response
            try:
                # Clean up response text
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]

                evaluation = json.loads(response_text)
                return evaluation

            except json.JSONDecodeError:
                logger.error(f"❌ Failed to parse AI evaluation response: {response_text}")
                return None

        except Exception as e:
            logger.error(f"❌ AI evaluation error: {e}")
            return None

    async def _store_quality_sources(self, filtered_results: List[Dict[str, Any]], destination: str):
        """Store high-quality sources (score 0.7-1.0) in database"""
        try:
            high_quality_sources = [
                result for result in filtered_results 
                if result.get('quality_score', 0) >= 0.7
            ]

            if not high_quality_sources:
                logger.info("📊 No high-quality sources to store")
                return

            logger.info(f"💾 Storing {len(high_quality_sources)} high-quality sources")

            # Import here to avoid circular imports
            from utils.database import get_database

            for result in high_quality_sources:
                # Clean up URL (remove everything after first dash as specified)
                url = result.get('url', '')
                clean_url = self._clean_url_for_storage(url)
                score = result.get('quality_score', 0.0)

                try:
                    get_database().store_source_quality(destination, clean_url, score)
                    self.stats["database_saves"] += 1
                except Exception as e:
                    logger.error(f"❌ Failed to store source {clean_url}: {e}")

            self.stats["high_quality_sources"] = len(high_quality_sources)
            logger.info(f"✅ Stored {self.stats['database_saves']} sources in database")

        except Exception as e:
            logger.error(f"❌ Error storing quality sources: {e}")

    def _clean_url_for_storage(self, url: str) -> str:
        """Clean URL for database storage (until first dash, website's main page)"""
        try:
            parsed = urlparse(url)
            # Return scheme + netloc (main domain)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return url

    def get_stats(self) -> Dict[str, Any]:
        """Get search and filtering statistics"""
        return self.stats.copy()