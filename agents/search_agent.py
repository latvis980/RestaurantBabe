# Search agent with AI-based filtering system

import requests
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import time
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger("restaurant-recommender.search_agent")

class BraveSearchAgent:
    def __init__(self, config):
        self.api_key = config.BRAVE_API_KEY
        self.search_count = config.BRAVE_SEARCH_COUNT
        self.excluded_domains = config.EXCLUDED_RESTAURANT_SOURCES
        self.config = config
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

        # Initialize the AI evaluation model
        self.model = ChatOpenAI(
            model=config.SEARCH_EVALUATION_MODEL,
            temperature=config.SEARCH_EVALUATION_TEMPERATURE,  # Use the config value
            api_key=config.OPENAI_API_KEY  # Explicitly set the API key
        )

        # Log the model being used for transparency
        logger.info(f"🔍 Search evaluation using: {config.SEARCH_EVALUATION_MODEL} (cost-optimized)")

        # Enhanced AI evaluation system prompt that prioritizes local sources
        self.eval_system_prompt = """
        You are an expert at evaluating web content about restaurants.
        Your task is to analyze if a web page contains a curated list of restaurants or restaurant recommendations.

        PRIORITIZE THESE SOURCES (score 0.8-1.0):
        - Local newspapers and magazines (like Expresso, Le Monde, El Pais, Time Out, reputable local food blogs)
        - Professional food critics and culinary experts
        - Established food and travel publications (e.g., Conde Nast Traveler, Forbes Travel, Food & Wine, Bon Appétit, etc.)
        - Local tourism boards and official guides
        - Restaurant guides and gastronomic awards (Michelin, The World's 50 Best, World of Mouth)

        VALID CONTENT (score 0.6-0.8):
        - Curated lists of multiple restaurants (e.g., "Top 10 restaurants in Paris")
        - Collections of restaurants in professional restaurant guides
        - Food critic reviews of a single restaurant ONLY in professional media
        - Articles in reputable local media discussing various dining options in an area
        - Food blog articles with restaurant recommendations
        - Travel articles mentioning multiple dining options

        NOT VALID CONTENT (score < 0.3):
        - Official website of a single restaurant
        - Anything on Tripadvisor, Yelp, OpenTable, RestaurantGuru and other review sites and generic restaurant lists, not professionally curated
        - Collections of restaurants in booking and delivery websites like Uber Eats, The Fork, Glovo, Bolt, etc.
        - Wanderlog content
        - Individual restaurant menus
        - Single restaurant reviews
        - Social media posts about individual dining experiences
        - Forum/Reddit discussions without professional curation
        - Hotel booking sites
        - Video content (YouTube, TikTok, etc.)

        SCORING CRITERIA:
        - Multiple restaurants mentioned (essential with the only exception of single restaurant reviews in professional media)
        - Professional curation or expertise evident
        - Local expertise and knowledge
        - Detailed descriptions of restaurants/cuisine
        - Price or quality indications for multiple venues

        FORMAT:
        Respond with a JSON object containing:
        {{
          "is_restaurant_list": true/false,
          "restaurant_count": estimated number of restaurants mentioned,
          "content_quality": 0.0-1.0,
          "passed_filter": true/false,
          "reasoning": "brief explanation emphasizing local expertise and content quality"
        }}
        """

        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.eval_system_prompt),
            ("human", "URL: {{url}}\n\nPage Title: {{title}}\n\nContent Preview:\n{{preview}}")
        ])

        self.eval_chain = self.eval_prompt | self.model

        # Statistics tracking
        self.filtered_urls = []
        self.evaluation_stats = {
            "total_evaluated": 0,
            "passed_filter": 0,
            "failed_filter": 0,
            "evaluation_errors": 0,
            "domain_filtered": 0,
            "model_used": config.SEARCH_EVALUATION_MODEL,  # Track which model we're using
            "estimated_cost_saved": 0.0  # Track cost savings vs GPT-4o
        }

        # Define video/streaming platforms to exclude
        self.video_platforms = {
            'youtube.com',
            'youtu.be', 
            'tiktok.com',
            'instagram.com',
            'facebook.com',
            'twitter.com',
            'x.com',
            'vimeo.com',
            'dailymotion.com',
            'twitch.tv',
            'pinterest.com',
            'snapchat.com'
        }

    def search(self, queries, max_retries=3, retry_delay=2, enable_ai_filtering=True):
        """
        Perform searches with the given queries and optional AI filtering

        Args:
            queries (list): List of search queries
            max_retries (int): Maximum number of retries for failed requests
            retry_delay (int): Delay between retries in seconds
            enable_ai_filtering (bool): Whether to apply AI-based content filtering

        Returns:
            list: Combined search results from all queries
        """
        all_results = []

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            for query in queries:
                retry_count = 0
                success = False

                while not success and retry_count < max_retries:
                    try:
                        logger.info(f"[SearchAgent] Searching for: {query}")
                        results = self._execute_search(query)
                        logger.info(f"[SearchAgent] Raw results count: {len(results.get('web', {}).get('results', []))}")

                        filtered_results = self._filter_results(results)
                        logger.info(f"[SearchAgent] Domain-filtered results count: {len(filtered_results)}")

                        # Apply AI filtering if enabled
                        if enable_ai_filtering and filtered_results:
                            logger.info(f"[SearchAgent] Applying AI content filtering...")
                            # Use the thread-based approach for async execution
                            ai_filtered_results = self._run_async_in_thread(self._apply_ai_filtering(filtered_results))
                            logger.info(f"[SearchAgent] AI-filtered results count: {len(ai_filtered_results)}")
                            all_results.extend(ai_filtered_results)
                        else:
                            all_results.extend(filtered_results)

                        success = True
                    except Exception as e:
                        logger.error(f"Error in search for query '{query}': {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"Max retries reached for query '{query}'")

                    # Respect rate limits
                    time.sleep(1)

        logger.info(f"[SearchAgent] Total search results after all filtering: {len(all_results)}")

        # Log AI filtering statistics
        if enable_ai_filtering:
            logger.info(f"[SearchAgent] AI Filtering Stats: {self.evaluation_stats}")

        # Cache search results using new Supabase system
        if all_results:
            try:
                from utils.database import cache_search_results
                cache_search_results(str(queries), {
                    "queries": queries,
                    "timestamp": time.time(),
                    "results": all_results,
                    "ai_filtering_enabled": enable_ai_filtering,
                    "filtering_stats": self.evaluation_stats.copy()
                })
            except Exception as e:
                logger.warning(f"Failed to cache search results: {e}")

        # CRITICAL FIX: Add consistent deduplication
        deduplicated_results = self._deduplicate_search_results(all_results)
        logger.info(f"[SearchAgent] Final results: {len(all_results)} → {len(deduplicated_results)} after deduplication")

        return deduplicated_results

    def _deduplicate_search_results(self, results: List[Dict]) -> List[Dict]:
        """
        Deduplicate search results while preserving highest quality sources

        Strategy:
        1. Group by URL
        2. For duplicates, keep the one with highest AI evaluation score
        3. If no AI scores, keep the first one
        """
        url_groups = {}

        # Group results by URL
        for result in results:
            url = result.get('url', '')
            if not url:
                continue

            if url not in url_groups:
                url_groups[url] = []
            url_groups[url].append(result)

        # Select best result from each URL group
        deduplicated = []

        for url, duplicates in url_groups.items():
            if len(duplicates) == 1:
                # No duplicates, keep as-is
                deduplicated.append(duplicates[0])
            else:
                # Multiple results for same URL - pick the best one
                best_result = self._select_best_duplicate(duplicates, url)
                deduplicated.append(best_result)

                logger.info(f"[SearchAgent] Deduplicated {len(duplicates)} results for {url}")

        return deduplicated

    def _select_best_duplicate(self, duplicates: List[Dict], url: str) -> Dict:
        """
        Select the best result from duplicates for the same URL

        Priority:
        1. Highest AI evaluation content_quality score
        2. If no AI scores, prefer newer content (based on title/description)
        3. If still tied, take the first one
        """
        # Check if any have AI evaluation scores
        with_ai_scores = [r for r in duplicates if r.get('ai_evaluation', {}).get('content_quality')]

        if with_ai_scores:
            # Pick the one with highest content quality score
            best = max(with_ai_scores, key=lambda r: r.get('ai_evaluation', {}).get('content_quality', 0))
            score = best.get('ai_evaluation', {}).get('content_quality', 0)
            logger.info(f"[SearchAgent] Selected best duplicate for {url}: quality={score}")
            return best

        # No AI scores - check for date indicators in title
        current_year = "2025"
        recent_results = [r for r in duplicates if current_year in r.get('title', '')]

        if recent_results:
            logger.debug(f"[SearchAgent] Selected recent result for {url}")
            return recent_results[0]

        # Default to first result
        logger.debug(f"[SearchAgent] Selected first result for {url}")
        return duplicates[0]

    def _run_async_in_thread(self, coro):
        """
        Run an async coroutine in a new thread with its own event loop
        This avoids the 'asyncio.run() cannot be called from a running event loop' error
        """
        def run_in_new_event_loop():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        # Execute the function in a thread
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(run_in_new_event_loop).result()

    async def _apply_ai_filtering(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply AI-based content filtering to search results with domain pre-filtering

        Args:
            search_results: List of search result dictionaries

        Returns:
            List of filtered search results that pass AI evaluation
        """
        # First, apply domain-based filtering to remove obvious video platforms
        domain_filtered_results = []

        for result in search_results:
            url = result.get('url', '')
            if self._is_video_platform(url):
                logger.info(f"Domain-filtered video platform: {url}")
                self.evaluation_stats["domain_filtered"] += 1
                self.filtered_urls.append(url)
                continue

            domain_filtered_results.append(result)

        logger.info(f"[SearchAgent] After domain filtering: {len(domain_filtered_results)} results (filtered {len(search_results) - len(domain_filtered_results)} video platforms)")

        # Now apply AI filtering to remaining results
        filtered_results = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent AI evaluations

        async def evaluate_single_result(result):
            async with semaphore:
                return await self._evaluate_search_result(result)

        # Create tasks for all evaluations
        tasks = [evaluate_single_result(result) for result in domain_filtered_results]

        # Wait for all evaluations to complete
        evaluation_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result, evaluation in zip(domain_filtered_results, evaluation_results):
            if isinstance(evaluation, Exception):
                logger.error(f"Error evaluating {result.get('url', 'unknown')}: {evaluation}")
                self.evaluation_stats["evaluation_errors"] += 1
                # Include result if evaluation failed (conservative approach)
                filtered_results.append(result)
            elif evaluation and evaluation.get("passed_filter", False):
                # Add evaluation metadata to the result
                result["ai_evaluation"] = evaluation
                filtered_results.append(result)
            else:
                # Result was filtered out
                self.filtered_urls.append(result.get("url", "unknown"))

        return filtered_results

    def _is_video_platform(self, url: str) -> bool:
        """
        Check if URL is from a video/social media platform that should be excluded

        Args:
            url: URL to check

        Returns:
            bool: True if URL is from a video platform
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()

            # Remove www. prefix for comparison
            if domain.startswith('www.'):
                domain = domain[4:]

            # Check if domain matches any video platform
            return domain in self.video_platforms

        except Exception as e:
            logger.warning(f"Error parsing URL for video platform check: {url}, error: {e}")
            return False

    async def _evaluate_search_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single search result using AI

        Args:
            result: Search result dictionary

        Returns:
            Evaluation result dictionary or None if evaluation failed
        """
        url = result.get("url", "")
        title = result.get("title", "")
        description = result.get("description", "")

        self.evaluation_stats["total_evaluated"] += 1

        try:
            # First, do a quick content preview fetch
            content_preview = await self._fetch_content_preview(url)
            if not content_preview:
                # If we can't fetch content, apply basic keyword filtering
                return self._basic_keyword_evaluation(url, title, description)

            # Combine title and description with content preview
            full_preview = f"{title}\n\n{description}\n\n{content_preview}"

            # Basic keyword check to avoid LLM calls for obviously irrelevant content
            restaurant_keywords = ["restaurant", "dining", "food", "eat", "chef", "cuisine", "menu", "dish", "gastronomy", "culinary"]
            if not any(kw in full_preview.lower() for kw in restaurant_keywords):
                logger.info(f"URL filtered by basic keyword check: {url}")
                self.evaluation_stats["failed_filter"] += 1
                return {
                    "passed_filter": False,
                    "is_restaurant_list": False,
                    "restaurant_count": 0,
                    "content_quality": 0.0,
                    "reasoning": "No restaurant-related keywords found"
                }

            # AI evaluation
            response = await self.eval_chain.ainvoke({
                "url": url,
                "title": title,
                "preview": full_preview[:1500]  # Limit to avoid token limits
            })

            # Parse AI response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content.strip())

            # Ensure content_quality is in the response
            if "content_quality" not in evaluation:
                evaluation["content_quality"] = 0.8 if evaluation.get("is_restaurant_list", False) else 0.2

            # Set passed_filter if not explicitly set
            if "passed_filter" not in evaluation:
                threshold = 0.5
                is_restaurant_list = evaluation.get("is_restaurant_list", False)
                content_quality = evaluation.get("content_quality", 0.0)
                evaluation["passed_filter"] = is_restaurant_list and content_quality > threshold

            # Update statistics
            if evaluation.get("passed_filter", False):
                self.evaluation_stats["passed_filter"] += 1
            else:
                self.evaluation_stats["failed_filter"] += 1

            # Log evaluation details
            logger.info(f"AI evaluation for {url}: List={evaluation.get('is_restaurant_list')}, Quality={evaluation.get('content_quality', 0):.2f}, Pass={evaluation.get('passed_filter')}")

            return evaluation

        except Exception as e:
            logger.error(f"Error in AI evaluation for {url}: {str(e)}")
            self.evaluation_stats["evaluation_errors"] += 1
            # Return conservative result (pass the filter) if evaluation fails
            return {
                "passed_filter": True,
                "is_restaurant_list": True,
                "restaurant_count": 0,
                "content_quality": 0.5,
                "reasoning": f"Evaluation error: {str(e)}"
            }

    async def _fetch_content_preview(self, url: str) -> str:
        """
        Fetch a brief content preview from URL for evaluation

        Args:
            url: URL to fetch preview from

        Returns:
            Content preview string
        """
        try:
            # Use aiohttp for async HTTP requests
            import aiohttp

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Connection": "close"
            }

            timeout = aiohttp.ClientTimeout(total=10)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()

                        # Extract text content
                        main_content = (soup.find('main') or 
                                      soup.find('article') or 
                                      soup.find(class_='content') or 
                                      soup.body)

                        if main_content:
                            preview_text = main_content.get_text(separator=' ', strip=True)
                            return preview_text[:1000]  # Return first 1000 characters

                        return soup.get_text(separator=' ', strip=True)[:1000]

                    return ""

        except ImportError:
            # Fallback to requests if aiohttp not available
            try:
                import requests
                response = requests.get(url, timeout=10, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    return soup.get_text(separator=' ', strip=True)[:1000]
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Error fetching content preview for {url}: {str(e)}")

        return ""

    def _basic_keyword_evaluation(self, url: str, title: str, description: str) -> Dict[str, Any]:
        """
        Apply basic keyword-based evaluation when content fetching fails

        Args:
            url: URL being evaluated
            title: Page title
            description: Page description

        Returns:
            Basic evaluation result
        """
        combined_text = f"{title} {description}".lower()

        # Positive keywords that indicate restaurant lists/guides
        positive_keywords = [
            "best restaurants", "top restaurants", "restaurant guide", "food guide",
            "where to eat", "dining guide", "restaurant list", "food critic",
            "restaurant recommendations", "culinary guide", "michelin", "zagat",
            "dining recommendations", "food scene", "restaurants to try"
        ]

        # Negative keywords that indicate single restaurants or unwanted content
        negative_keywords = [
            "menu", "book table", "order online", "delivery", "takeaway",
            "single restaurant", "one restaurant", "hotel", "booking",
            "reservation", "uber eats", "doordash", "grubhub"
        ]

        positive_score = sum(1 for kw in positive_keywords if kw in combined_text)
        negative_score = sum(1 for kw in negative_keywords if kw in combined_text)

        # Simple scoring
        if positive_score > negative_score and positive_score > 0:
            self.evaluation_stats["passed_filter"] += 1
            return {
                "passed_filter": True,
                "is_restaurant_list": True,
                "restaurant_count": 5,  # Estimate
                "content_quality": 0.6,
                "reasoning": f"Basic keyword evaluation: {positive_score} positive, {negative_score} negative keywords"
            }
        else:
            self.evaluation_stats["failed_filter"] += 1
            return {
                "passed_filter": False,
                "is_restaurant_list": False,
                "restaurant_count": 0,
                "content_quality": 0.3,
                "reasoning": f"Basic keyword evaluation: {positive_score} positive, {negative_score} negative keywords"
            }

    def _execute_search(self, query):
        """Execute a single search query against Brave Search API"""
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": self.search_count,
            "freshness": "month"  # Get recent results
        }

        response = requests.get(
            self.base_url,
            headers=headers,
            params=params
        )

        if response.status_code != 200:
            raise Exception(f"Brave Search API error: {response.status_code}, {response.text}")

        return response.json()

    def _filter_results(self, search_results):
        """Filter search results to exclude unwanted domains with enhanced logging"""
        if not search_results or "web" not in search_results or "results" not in search_results["web"]:
            return []

        filtered_results = []
        excluded_count = 0

        for result in search_results["web"]["results"]:
            url = result.get("url", "")

            # Check if URL should be excluded by domain
            if self._should_exclude_domain(url):
                logger.info(f"[SearchAgent] ❌ Domain-filtered: {url}")
                self.evaluation_stats["domain_filtered"] += 1
                excluded_count += 1
                continue

            # Check if it's a video platform
            if self._is_video_platform(url):
                logger.debug(f"[SearchAgent] Video platform filtered: {url}")
                self.evaluation_stats["domain_filtered"] += 1
                excluded_count += 1
                continue

            # Log kept URLs for debugging
            logger.info(f"[SearchAgent] ✅ Keeping: {url}")

            # Clean and extract the relevant information
            filtered_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
                "language": result.get("language", "en"),
                "favicon": result.get("favicon", "")
            }
            filtered_results.append(filtered_result)

        logger.info(f"[SearchAgent] Domain filtering complete: {len(filtered_results)} kept, {excluded_count} excluded")
        return filtered_results

    def _should_exclude_domain(self, url):
        """
        Check if URL domain should be excluded using proper domain matching
        Fixes the bug where TripAdvisor variants weren't being filtered
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()

            # Remove www. prefix for comparison
            if domain.startswith('www.'):
                domain = domain[4:]

            for excluded in self.excluded_domains:
                excluded_clean = excluded.lower()
                if excluded_clean.startswith('www.'):
                    excluded_clean = excluded_clean[4:]

                # Method 1: Exact domain match
                if domain == excluded_clean:
                    logger.info(f"[SearchAgent] ❌ Exact match filtered: {domain}")
                    return True

                # Method 2: Base domain match (handles country variants)
                # e.g., "tripadvisor" matches tripadvisor.com, tripadvisor.co.uk, etc.
                excluded_base = excluded_clean.split('.')[0]
                domain_base = domain.split('.')[0]

                # Only match if it's a known problematic base domain
                problematic_bases = ['tripadvisor', 'yelp', 'opentable', 'booking', 'hotels', 'expedia']
                if excluded_base in problematic_bases and excluded_base == domain_base:
                    logger.info(f"[SearchAgent] ❌ Base domain match filtered: {domain} (base: {domain_base})")
                    return True

                # Method 3: Full subdomain match (for google.com/maps)
                if excluded_clean == "google.com/maps" and ("maps.google" in domain or "google.com" in domain):
                    logger.info(f"[SearchAgent] ❌ Google Maps filtered: {domain}")
                    return True

            logger.debug(f"[SearchAgent] ✅ Domain allowed: {domain}")
            return False

        except Exception as e:
            logger.warning(f"Error parsing domain from {url}: {e}")
            return False

    def follow_up_search(self, restaurant_name, location, additional_context=None):
        """
        Perform a follow-up search for a specific restaurant

        Args:
            restaurant_name (str): Name of the restaurant
            location (str): Location of the restaurant
            additional_context (str, optional): Additional search context

        Returns:
            dict: Search results specifically about this restaurant
        """
        # Create a specific query for this restaurant
        query = f"{restaurant_name} restaurant {location}"
        if additional_context:
            query += f" {additional_context}"

        # Search for this specific restaurant (without AI filtering for specific searches)
        results = self._execute_search(query)
        filtered_results = self._filter_results(results)

        # Also check global guides
        global_guides_results = self._check_global_guides(restaurant_name, location)

        return {
            "direct_search": filtered_results,
            "global_guides": global_guides_results
        }

    def _check_global_guides(self, restaurant_name, location):
        """Check if the restaurant is mentioned in global guides"""
        global_guides = [
            "theworlds50best.com",
            "worldofmouth.app", 
            "guide.michelin.com",
            "culinarybackstreets.com",
            "oadguides.com",
            "laliste.com"
        ]

        results = []

        for guide in global_guides:
            try:
                query = f"site:{guide} {restaurant_name} {location}"
                guide_results = self._execute_search(query)
                filtered_guide_results = self._filter_results(guide_results)

                if filtered_guide_results:
                    for result in filtered_guide_results:
                        result["guide"] = guide
                        results.append(result)

                # Respect rate limits
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error checking guide {guide}: {e}")

        return results

    def get_filtering_stats(self):
        """Get current AI filtering statistics with cost information"""
        # Calculate estimated cost savings
        if self.config.SEARCH_EVALUATION_MODEL == "gpt-4o-mini":
            # Rough calculation: GPT-4o-mini is ~95% cheaper than GPT-4o
            cost_multiplier = 0.05  # GPT-4o-mini costs about 5% of GPT-4o
            estimated_calls = self.evaluation_stats["total_evaluated"]
            estimated_savings_per_call = 0.012  # Rough estimate per evaluation call
            self.evaluation_stats["estimated_cost_saved"] = estimated_calls * estimated_savings_per_call * 0.95

        return {
            "evaluation_stats": self.evaluation_stats.copy(),
            "filtered_urls": self.filtered_urls.copy()
        }

    def reset_stats(self):
        """Reset evaluation statistics"""
        self.evaluation_stats = {
            "total_evaluated": 0,
            "passed_filter": 0,
            "failed_filter": 0,
            "evaluation_errors": 0,
            "domain_filtered": 0,
            "model_used": self.config.SEARCH_EVALUATION_MODEL,
            "estimated_cost_saved": 0.0
        }
        self.filtered_urls = []

    def get_excluded_domains(self):
        """Get list of excluded domains"""
        return self.excluded_domains.copy()

    def add_excluded_domain(self, domain: str):
        """Add a domain to the exclusion list"""
        if domain not in self.excluded_domains:
            self.excluded_domains.append(domain)
            logger.info(f"Added {domain} to excluded domains")

    def remove_excluded_domain(self, domain: str):
        """Remove a domain from the exclusion list"""
        if domain in self.excluded_domains:
            self.excluded_domains.remove(domain)
            logger.info(f"Removed {domain} from excluded domains")

    def get_search_count(self):
        """Get current search count setting"""
        return self.search_count

    def set_search_count(self, count: int):
        """Set search count"""
        if 1 <= count <= 50:  # Reasonable limits
            self.search_count = count
            logger.info(f"Set search count to {count}")
        else:
            logger.warning(f"Invalid search count: {count}. Must be between 1-50")

    def get_video_platforms(self):
        """Get list of video platforms that are filtered"""
        return list(self.video_platforms)

    def add_video_platform(self, platform: str):
        """Add a video platform to the filter list"""
        self.video_platforms.add(platform.lower())
        logger.info(f"Added {platform} to video platform filters")

    def remove_video_platform(self, platform: str):
        """Remove a video platform from the filter list"""
        self.video_platforms.discard(platform.lower())
        logger.info(f"Removed {platform} from video platform filters")

    def search_with_cache(self, queries, cache_key=None, cache_ttl=3600):
        """
        Perform search with optional caching

        Args:
            queries: List of search queries
            cache_key: Optional cache key (defaults to queries hash)
            cache_ttl: Cache time-to-live in seconds

        Returns:
            Search results (from cache or fresh search)
        """
        if not cache_key:
            cache_key = str(hash(str(queries)))

        try:
            from utils.database import get_cached_results
            cached = get_cached_results(cache_key)

            if cached and (time.time() - cached.get('timestamp', 0)) < cache_ttl:
                logger.info(f"[SearchAgent] Using cached results for: {queries}")
                return cached.get('results', [])

        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

        # Perform fresh search
        results = self.search(queries)

        # Cache the results
        try:
            from utils.database import cache_search_results
            cache_search_results(cache_key, {
                "queries": queries,
                "timestamp": time.time(), 
                "results": results
            })
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

        return results

    def search_specific_domains(self, queries, allowed_domains=None, blocked_domains=None):
        """
        Search with custom domain filtering

        Args:
            queries: Search queries
            allowed_domains: List of domains to allow (if specified, only these are allowed)
            blocked_domains: Additional domains to block

        Returns:
            Filtered search results
        """
        # Temporarily modify domain filtering
        original_excluded = self.excluded_domains.copy()

        try:
            if allowed_domains:
                # If allowed_domains specified, block everything else
                logger.info(f"[SearchAgent] Restricting search to domains: {allowed_domains}")
                # We'll filter in post-processing since we can't modify the search API

            if blocked_domains:
                # Add additional blocked domains
                self.excluded_domains.extend(blocked_domains)
                logger.info(f"[SearchAgent] Added temporary blocked domains: {blocked_domains}")

            # Perform search
            results = self.search(queries)

            # Apply allowed_domains filter if specified
            if allowed_domains:
                allowed_set = set(d.lower() for d in allowed_domains)
                filtered_results = []

                for result in results:
                    url = result.get('url', '')
                    try:
                        domain = urlparse(url).netloc.lower()
                        if domain.startswith('www.'):
                            domain = domain[4:]

                        if any(allowed in domain for allowed in allowed_set):
                            filtered_results.append(result)
                        else:
                            logger.debug(f"[SearchAgent] Filtered non-allowed domain: {domain}")

                    except Exception:
                        continue

                results = filtered_results
                logger.info(f"[SearchAgent] After allowed-domain filtering: {len(results)} results")

            return results

        finally:
            # Restore original excluded domains
            self.excluded_domains = original_excluded

    def get_search_performance_stats(self):
        """Get detailed performance statistics"""
        return {
            "evaluation_stats": self.evaluation_stats.copy(),
            "filtered_urls_count": len(self.filtered_urls),
            "video_platforms_count": len(self.video_platforms),
            "excluded_domains_count": len(self.excluded_domains),
            "current_search_count": self.search_count,
            "model_used": self.config.SEARCH_EVALUATION_MODEL
        }

    def export_filtered_urls(self, filename=None):
        """Export filtered URLs to file for analysis"""
        if not filename:
            timestamp = int(time.time())
            filename = f"filtered_urls_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "filtered_urls": self.filtered_urls,
                    "evaluation_stats": self.evaluation_stats,
                    "excluded_domains": self.excluded_domains
                }, f, indent=2)

            logger.info(f"Exported filtered URLs to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Failed to export filtered URLs: {e}")
            return None

    def import_domain_intelligence(self, filename):
        """Import domain intelligence from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if 'excluded_domains' in data:
                self.excluded_domains = data['excluded_domains']
                logger.info(f"Imported {len(self.excluded_domains)} excluded domains")

            if 'video_platforms' in data:
                self.video_platforms = set(data['video_platforms'])
                logger.info(f"Imported {len(self.video_platforms)} video platforms")

            return True

        except Exception as e:
            logger.error(f"Failed to import domain intelligence: {e}")
            return False

    def validate_configuration(self):
        """Validate search agent configuration"""
        issues = []

        if not self.api_key:
            issues.append("Missing Brave Search API key")

        if not self.excluded_domains:
            issues.append("No excluded domains configured")

        if self.search_count < 1 or self.search_count > 50:
            issues.append(f"Invalid search count: {self.search_count}")

        if not hasattr(self.config, 'OPENAI_API_KEY') or not self.config.OPENAI_API_KEY:
            issues.append("Missing OpenAI API key for AI evaluation")

        if issues:
            logger.warning(f"Configuration issues found: {issues}")
            return False, issues
        else:
            logger.info("Search agent configuration validated successfully")
            return True, []

    def health_check(self):
        """Perform health check of search functionality"""
        try:
            # Test basic search
            test_query = "restaurants"
            test_results = self._execute_search(test_query)

            if not test_results or 'web' not in test_results:
                return False, "Search API not responding correctly"

            # Test AI evaluation
            if hasattr(self, 'model'):
                try:
                    test_eval = self.model.invoke("Test message")
                    if not test_eval:
                        return False, "AI evaluation model not responding"
                except Exception as e:
                    return False, f"AI evaluation error: {str(e)}"

            return True, "All systems operational"

        except Exception as e:
            return False, f"Health check failed: {str(e)}"

    def get_debug_info(self):
        """Get comprehensive debug information"""
        return {
            "config": {
                "api_key_configured": bool(self.api_key),
                "search_count": self.search_count,
                "base_url": self.base_url,
                "model": self.config.SEARCH_EVALUATION_MODEL,
                "temperature": self.config.SEARCH_EVALUATION_TEMPERATURE
            },
            "filters": {
                "excluded_domains": self.excluded_domains,
                "video_platforms": list(self.video_platforms),
                "excluded_domains_count": len(self.excluded_domains),
                "video_platforms_count": len(self.video_platforms)
            },
            "statistics": self.evaluation_stats.copy(),
            "filtered_urls": {
                "count": len(self.filtered_urls),
                "recent_urls": self.filtered_urls[-10:] if self.filtered_urls else []
            }
        }

    ## Legacy compatibility methods
    def get_stats(self):
        """Legacy method - alias for get_filtering_stats"""
        return self.get_filtering_stats()

    def clear_cache(self):
        """Legacy method - reset statistics"""
        self.reset_stats()

    def get_domain_intelligence(self):
        """Legacy method - return domain information"""
        return {
            "excluded_domains": self.excluded_domains,
            "video_platforms": list(self.video_platforms),
            "evaluation_stats": self.evaluation_stats
        }