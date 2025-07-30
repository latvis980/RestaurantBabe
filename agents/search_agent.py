# Search agent with AI-based filtering system and destination validation

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
            temperature=config.SEARCH_EVALUATION_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )

        logger.info(f"🔍 Search evaluation using: {config.SEARCH_EVALUATION_MODEL} (cost-optimized)")

        # AI evaluation system prompt with destination validation
        self.eval_system_prompt = """
        You are an expert at evaluating web content about restaurants.
        Your task is to analyze if a web page contains a curated list of restaurants or restaurant recommendations FOR THE SPECIFIC DESTINATION requested.

        CRITICAL: The content must be relevant to the requested destination: {{destination}}

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
        - Content that is NOT about the requested destination {{destination}}
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

        DESTINATION VALIDATION:
        - The content must clearly relate to the requested destination: {{destination}}
        - If the content is about a different city, region, or country, it should be rejected
        - Generic restaurant advice without location specificity should be rejected
        - Content about nearby locations may be acceptable if explicitly relevant

        SCORING CRITERIA:
        - Content relates to the requested destination {{destination}} (ESSENTIAL)
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
          "destination_match": true/false,
          "passed_filter": true/false,
          "reasoning": "brief explanation emphasizing destination relevance, local expertise and content quality"
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
            "failed_destination": 0,  # New stat for destination mismatches
            "evaluation_errors": 0,
            "domain_filtered": 0,
            "model_used": config.SEARCH_EVALUATION_MODEL,
            "estimated_cost_saved": 0.0
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

    def search(self, queries, destination, max_retries=3, retry_delay=2, enable_ai_filtering=True):
        """
        Perform searches with the given queries and optional AI filtering

        Args:
            queries (list): List of search queries
            destination (str): The destination/location for restaurant recommendations
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
                        logger.info(f"[SearchAgent] Searching for: {query} (destination: {destination})")
                        results = self._execute_search(query)
                        logger.info(f"[SearchAgent] Raw results count: {len(results.get('web', {}).get('results', []))}")

                        filtered_results = self._filter_results(results)
                        logger.info(f"[SearchAgent] Domain-filtered results count: {len(filtered_results)}")

                        # Apply AI filtering if enabled
                        if enable_ai_filtering and filtered_results:
                            logger.info(f"[SearchAgent] Applying AI content filtering with destination validation...")
                            # Pass destination to AI filtering
                            ai_filtered_results = self._run_async_in_thread(
                                self._apply_ai_filtering(filtered_results, destination)
                            )
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
            from utils.database import cache_search_results
            cache_search_results(str(queries), {
                "queries": queries,
                "destination": destination,
                "timestamp": time.time(),
                "results": all_results,
                "ai_filtering_enabled": enable_ai_filtering,
                "filtering_stats": self.evaluation_stats.copy()
            })

        return all_results

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

    async def _apply_ai_filtering(self, search_results: List[Dict[str, Any]], destination: str) -> List[Dict[str, Any]]:
        """
        Apply AI-based content filtering to search results with domain pre-filtering

        Args:
            search_results: List of search result dictionaries
            destination: The target destination for restaurant recommendations

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
                return await self._evaluate_search_result(result, destination)

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
                # Track destination filtering separately
                if evaluation and not evaluation.get("destination_match", True):
                    self.evaluation_stats["failed_destination"] += 1

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

    async def _evaluate_search_result(self, result: Dict[str, Any], destination: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single search result using AI

        Args:
            result: Search result dictionary
            destination: Target destination for restaurant recommendations

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
                return self._basic_keyword_evaluation(url, title, description, destination)

            # Combine title and description with content preview
            full_preview = f"{title}\n\n{description}\n\n{content_preview}"

            # Basic keyword check to avoid LLM calls for obviously irrelevant content
            restaurant_keywords = ["restaurant", "dining", "food", "eat", "chef", "cuisine", "menu", "dish"]
            if not any(kw in full_preview.lower() for kw in restaurant_keywords):
                logger.info(f"URL filtered by basic keyword check: {url}")
                self.evaluation_stats["failed_filter"] += 1
                return {
                    "passed_filter": False,
                    "is_restaurant_list": False,
                    "restaurant_count": 0,
                    "content_quality": 0.0,
                    "destination_match": False,
                    "reasoning": "No restaurant-related keywords found"
                }

            # AI evaluation with destination
            response = await self.eval_chain.ainvoke({
                "destination": destination,
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

            # Ensure all required fields are in the response
            if "content_quality" not in evaluation:
                evaluation["content_quality"] = 0.8 if evaluation.get("is_restaurant_list", False) else 0.2
            if "destination_match" not in evaluation:
                evaluation["destination_match"] = True  # Default to true if not specified

            # Apply threshold - now requires both restaurant list AND destination match
            threshold = 0.5
            is_restaurant_list = evaluation.get("is_restaurant_list", False)
            content_quality = evaluation.get("content_quality", 0.0)
            destination_match = evaluation.get("destination_match", False)
            passed_filter = is_restaurant_list and content_quality > threshold and destination_match

            if passed_filter:
                self.evaluation_stats["passed_filter"] += 1
            else:
                self.evaluation_stats["failed_filter"] += 1
                if not destination_match:
                    self.evaluation_stats["failed_destination"] += 1

            # Log evaluation details
            logger.info(f"AI evaluation for {url}: List={is_restaurant_list}, Quality={content_quality:.2f}, Destination={destination_match}, Pass={passed_filter}")

            return {
                "passed_filter": passed_filter,
                "is_restaurant_list": is_restaurant_list,
                "restaurant_count": evaluation.get("restaurant_count", 0),
                "content_quality": content_quality,
                "destination_match": destination_match,
                "reasoning": evaluation.get("reasoning", "")
            }

        except Exception as e:
            logger.error(f"Error in AI evaluation for {url}: {str(e)}")
            self.evaluation_stats["evaluation_errors"] += 1
            # Return conservative result (pass the filter) if evaluation fails
            return {
                "passed_filter": True,
                "is_restaurant_list": True,
                "restaurant_count": 0,
                "content_quality": 0.5,
                "destination_match": True,
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
                    return soup.get_text(separator=' ', strip=True)[:1000]
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Error fetching content preview for {url}: {str(e)}")

        return ""

    def _basic_keyword_evaluation(self, url: str, title: str, description: str, destination: str) -> Dict[str, Any]:
        """
        Apply basic keyword-based evaluation when content fetching fails

        Args:
            url: URL being evaluated
            title: Page title
            description: Page description
            destination: Target destination

        Returns:
            Basic evaluation result
        """
        combined_text = f"{title} {description}".lower()
        destination_lower = destination.lower()

        # Check for destination match
        destination_match = destination_lower in combined_text

        # Positive keywords
        positive_keywords = [
            "best restaurants", "top restaurants", "restaurant guide", "food guide",
            "where to eat", "dining guide", "restaurant list", "food critic",
            "restaurant recommendations", "culinary guide", "michelin", "zagat"
        ]

        # Negative keywords
        negative_keywords = [
            "menu", "book table", "order online", "delivery", "takeaway",
            "single restaurant", "one restaurant", "hotel", "booking"
        ]

        positive_score = sum(1 for kw in positive_keywords if kw in combined_text)
        negative_score = sum(1 for kw in negative_keywords if kw in combined_text)

        # Simple scoring - now requires destination match
        if positive_score > negative_score and positive_score > 0 and destination_match:
            self.evaluation_stats["passed_filter"] += 1
            return {
                "passed_filter": True,
                "is_restaurant_list": True,
                "restaurant_count": 5,  # Estimate
                "content_quality": 0.6,
                "destination_match": True,
                "reasoning": f"Basic keyword evaluation: {positive_score} positive, {negative_score} negative keywords, destination match: {destination_match}"
            }
        else:
            self.evaluation_stats["failed_filter"] += 1
            if not destination_match:
                self.evaluation_stats["failed_destination"] += 1
            return {
                "passed_filter": False,
                "is_restaurant_list": False,
                "restaurant_count": 0,
                "content_quality": 0.3,
                "destination_match": destination_match,
                "reasoning": f"Basic keyword evaluation: {positive_score} positive, {negative_score} negative keywords, destination match: {destination_match}"
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
        """Filter search results to exclude unwanted domains"""
        if not search_results or "web" not in search_results or "results" not in search_results["web"]:
            return []

        filtered_results = []

        for result in search_results["web"]["results"]:
            url = result.get("url", "")

            # Check if URL should be excluded by domain
            if self._should_exclude_domain(url):
                logger.debug(f"[SearchAgent] Domain-filtered: {url}")
                self.evaluation_stats["domain_filtered"] += 1
                continue

            # Check if it's a video platform
            if self._is_video_platform(url):
                logger.debug(f"[SearchAgent] Video platform filtered: {url}")
                self.evaluation_stats["domain_filtered"] += 1
                continue

            # Clean and extract the relevant information
            filtered_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
                "language": result.get("language", "en"),
                "favicon": result.get("favicon", "")
            }
            filtered_results.append(filtered_result)

        return filtered_results

    def _should_exclude_domain(self, url):
        """Check if URL domain should be excluded"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            # Remove www. prefix for comparison
            if domain.startswith('www.'):
                domain = domain[4:]

            return any(excluded in domain for excluded in self.excluded_domains)
        except Exception:
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