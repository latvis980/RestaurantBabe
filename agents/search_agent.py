# agents/search_agent.py
"""
Enhanced Search Agent with Parallel Brave + Tavily Search and AI Filtering

FIXED: Proper async client lifecycle management to prevent "Event loop is closed" errors
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
    FIXED: Proper async client lifecycle management
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

        # Initialize AI for content evaluation
        self.eval_model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )

        # Search statistics
        self.stats = {
            "total_searches": 0,
            "brave_searches": 0,
            "tavily_searches": 0,
            "total_results": 0,
            "filtered_results": 0
        }

        # Restored working prompt - DO NOT CHANGE
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
        - Wikipedia
        - Collections of restaurants on booking and delivery websites like Uber Eats, The Fork, Glovo, Bolt, Wolt, Mesa24, etc.
        - Social media content on Facebook and Instagram without professional curation
        - ANY Wanderlog content
        - Single restaurant reviews (with exception of professional media)
        - Social media posts about individual dining experiences
        - Forum/Reddit discussions without professional curation
        - Hotel booking sites like Booking.com, Agoda, Expedia, Jalan, etc.
        - Websites of irrelevant businesses: real estate agencies, rental companies, tour booking sites, etc.
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

        # Initialize evaluation prompt template
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.eval_system_prompt),
            ("user", "Title: {title}\nDescription: {description}\nURL: {url}\nContent Preview: {content_preview}")
        ])

    def search(self, search_queries: List[str], destination: str, query_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        FIXED: Main search method with proper async event loop management
        """
        logger.info(f"🔍 Starting enhanced search for {len(search_queries)} queries in {destination}")
        logger.info(f"🧠 Using query analyzer metadata: {query_metadata.get('is_english_speaking', 'unknown')} speaking, local language: {query_metadata.get('local_language', 'none')}")

        try:
            # FIXED: Proper event loop management
            try:
                # Check if event loop is already running
                loop = asyncio.get_running_loop()
                # If we get here, a loop is already running
                logger.info("🔄 Event loop already running - using thread pool execution")

                # Use thread pool to avoid loop conflict
                def run_async_search():
                    # Create new event loop in this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self._parallel_search_and_filter(search_queries, destination, query_metadata)
                        )
                    finally:
                        # FIXED: Properly close the loop
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
                    # FIXED: Properly close the loop
                    loop.close()

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    async def _parallel_search_and_filter(self, search_queries: List[str], destination: str, query_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        FIXED: Execute parallel searches and filtering pipeline with proper client management
        """
        logger.info("🚀 Starting parallel search and filtering pipeline")

        # Step 1: Determine query languages using query analyzer metadata
        english_queries, local_queries = self._categorize_queries(search_queries, destination, query_metadata)

        # Step 2: Parallel search execution
        all_results = []

        # FIXED: Use single session for all HTTP operations to manage lifecycle properly
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        def _deduplicate_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            Remove duplicate URLs from search results, keeping the best result for each URL
            """
            seen_urls = {}
            deduplicated = []

            for result in results:
                url = result.get('url', '').strip()
                if not url:
                    continue

                # Normalize URL (remove trailing slash, convert to lowercase)
                normalized_url = url.lower().rstrip('/')

                if normalized_url not in seen_urls:
                    seen_urls[normalized_url] = result
                    deduplicated.append(result)
                else:
                    # Keep the result with more complete information
                    existing = seen_urls[normalized_url]
                    current = result

                    # Prefer results with more content (longer description)
                    existing_desc_len = len(existing.get('description', ''))
                    current_desc_len = len(current.get('description', ''))

                    if current_desc_len > existing_desc_len:
                        # Replace existing with current (better description)
                        seen_urls[normalized_url] = current
                        # Find and replace in deduplicated list
                        for i, item in enumerate(deduplicated):
                            if item.get('url', '').lower().rstrip('/') == normalized_url:
                                deduplicated[i] = current
                                break

            removed_count = len(results) - len(deduplicated)
            if removed_count > 0:
                logger.info(f"🔗 Removed {removed_count} duplicate URLs from search results")

            return deduplicated

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Execute searches in parallel
            search_tasks = []

            if english_queries:
                logger.info(f"🇺🇸 Brave Search queries: {english_queries}")
                search_tasks.append(self._brave_search_batch(session, english_queries))

            if local_queries and self.tavily_api_key:
                logger.info(f"🌍 Tavily Search queries: {local_queries}")
                search_tasks.append(self._tavily_search_batch(session, local_queries))
            elif local_queries:
                logger.warning("📋 Local language queries found but Tavily API key not configured")
                # Fall back to Brave for local queries
                search_tasks.append(self._brave_search_batch(session, local_queries))

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

            # NEW: Deduplicate URLs before filtering
            all_results = self._deduplicate_search_results(all_results)

            logger.info(f"📊 Total search results after deduplication: {len(all_results)}")

            # Step 3: Fetch previews for all results
            results_with_previews = await self._fetch_previews_batch(session, all_results)

        # Step 4: AI-based filtering
        filtered_results = await self._filter_results_batch(results_with_previews, destination)

        # Step 5: Store high-quality sources in database
        await self._store_quality_sources(filtered_results, destination)

        logger.info(f"✅ Final filtered results: {len(filtered_results)}")
        return filtered_results

    def _categorize_queries(self, search_queries: List[str], destination: str, query_metadata: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Categorize queries into English and local language based on AI metadata"""
        english_queries = []
        local_queries = []

        # Use AI query analyzer metadata if available
        if query_metadata and query_metadata.get('is_english_speaking') is not None:
            is_english_speaking = query_metadata.get('is_english_speaking', True)
            local_language = query_metadata.get('local_language', 'none')

            if is_english_speaking or local_language == 'none':
                # English-speaking destination or no local language detected
                english_queries = search_queries.copy()
                logger.info(f"📝 AI Guidance: All {len(english_queries)} queries → Brave Search (English-speaking: {is_english_speaking})")
            else:
                # Non-English speaking destination with local language
                # Split queries: broader terms to English, specific terms to local
                for query in search_queries:
                    if self._is_likely_english(query):
                        english_queries.append(query)
                    else:
                        local_queries.append(query)

                # If no local queries detected, fall back to mixed approach
                if len(local_queries) == 0:
                    # Use both engines for better coverage
                    english_queries = search_queries.copy()
                    local_queries = search_queries.copy()
                    logger.info(f"📝 AI Guidance: Mixed approach - {local_language} destination with English queries")

        else:
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

    async def _brave_search_batch(self, session: aiohttp.ClientSession, queries: List[str]) -> List[Dict[str, Any]]:
        """
        FIXED: Execute Brave Search for multiple queries using shared session
        """
        logger.info(f"🔍 Executing Brave Search for {len(queries)} queries")

        all_results = []

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
        """
        FIXED: Execute single Brave search using shared session
        """
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

    async def _tavily_search_batch(self, session: aiohttp.ClientSession, queries: List[str]) -> List[Dict[str, Any]]:
        """
        FIXED: Execute Tavily Search for multiple queries using shared session
        """
        logger.info(f"🌍 Executing Tavily Search for {len(queries)} queries")

        all_results = []

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
        """
        FIXED: Execute single Tavily search using shared session
        """
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

    async def _fetch_previews_batch(self, session: aiohttp.ClientSession, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        FIXED: Fetch content previews for all search results using shared session
        """
        logger.info(f"📖 Fetching previews for {len(results)} results")

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
        """
        FIXED: Fetch content preview for a single URL using shared session
        """
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

            # Check if evaluation is valid and passed the filter
            if evaluation and isinstance(evaluation, dict) and evaluation.get('passed_filter', False):
                result['ai_evaluation'] = evaluation
                result['quality_score'] = evaluation.get('content_quality', 0.0)
                filtered_batch.append(result)
                logger.debug(f"✅ Passed: {result.get('title', 'No title')} (quality: {evaluation.get('content_quality', 0)})")
            else:
                quality = evaluation.get('content_quality') if evaluation and isinstance(evaluation, dict) else 'N/A'
                logger.debug(f"❌ Filtered: {result.get('title', 'No title')} (quality: {quality})")

        return filtered_batch

    async def _evaluate_single_result(self, result: Dict[str, Any], destination: str) -> Optional[Dict[str, Any]]:
        """Evaluate a single result using AI"""
        try:
            prompt = self.eval_prompt.format_messages(
                title=result.get('title', ''),
                description=result.get('description', ''),
                url=result.get('url', ''),
                content_preview=result.get('content_preview', '')
            )

            response = await self.eval_model.ainvoke(prompt)

            # Parse JSON response
            try:
                # Handle both string and list responses from the model
                content = response.content
                if isinstance(content, list):
                    # If content is a list, join it to a string
                    content_str = ''.join(str(item) for item in content)
                else:
                    content_str = str(content)

                # FIXED: Remove markdown code blocks if present
                content_str = content_str.strip()

                # Debug logging to see what we're getting
                logger.debug(f"🔍 Raw AI response: {content_str[:200]}...")

                if content_str.startswith('```json'):
                    # Remove ```json at start and ``` at end
                    content_str = content_str[7:]  # Remove ```json
                    if content_str.endswith('```'):
                        content_str = content_str[:-3]  # Remove trailing ```
                elif content_str.startswith('```'):
                    # Remove generic ``` blocks
                    content_str = content_str[3:]  # Remove ```
                    if content_str.endswith('```'):
                        content_str = content_str[:-3]  # Remove trailing ```

                content_str = content_str.strip()

                # Additional debug logging after cleaning
                logger.debug(f"🧹 Cleaned content for parsing: {content_str}")

                # Validate that we have content to parse
                if not content_str:
                    logger.error("❌ Empty content after cleaning markdown")
                    return None

                # Attempt to parse JSON
                evaluation = json.loads(content_str)

                # Validate the expected structure
                if not isinstance(evaluation, dict):
                    logger.error(f"❌ Expected dict, got {type(evaluation)}: {evaluation}")
                    return None

                # Check for required fields from working format
                required_fields = ['is_restaurant_list', 'content_quality', 'passed_filter', 'reasoning']
                missing_fields = [field for field in required_fields if field not in evaluation]

                if missing_fields:
                    logger.error(f"❌ Missing required fields: {missing_fields}. Got: {list(evaluation.keys())}")
                    return None

                # Ensure content_quality is numeric
                try:
                    quality = float(evaluation['content_quality'])
                    evaluation['content_quality'] = quality
                except (ValueError, TypeError):
                    logger.error(f"❌ Invalid content_quality value: {evaluation.get('content_quality')}")
                    return None

                return evaluation

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing failed: {e}")
                logger.error(f"❌ Content that failed to parse: {content_str}")
                return None

        except Exception as e:
            logger.error(f"❌ AI evaluation error: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None

    async def _store_quality_sources(self, filtered_results: List[Dict[str, Any]], destination: str):
        """Store high-quality sources in database for future reference"""
        try:
            # This would integrate with your database storage system
            high_quality_count = len([r for r in filtered_results if r.get('ai_evaluation', {}).get('score', 0) >= 0.8])
            logger.info(f"📚 Found {high_quality_count} high-quality sources for {destination}")

            # TODO: Implement actual database storage
            # For now, just log the statistics

        except Exception as e:
            logger.error(f"❌ Failed to store quality sources: {e}")

    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset search statistics"""
        self.stats = {
            "total_searches": 0,
            "brave_searches": 0,
            "tavily_searches": 0,
            "total_results": 0,
            "filtered_results": 0
        }