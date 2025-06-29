# Search agent with AI-based filtering system and enhanced logging

import requests
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from utils.database import save_data
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
            model=config.OPENAI_MODEL,
            temperature=0.2
        )

        # AI evaluation system prompt
        self.eval_system_prompt = """
        You are an expert at evaluating web content about restaurants.
        Your task is to analyze if a web page contains a curated list of restaurants or restaurant recommendations.

        VALID CONTENT (score > 0.7):
        - Curated lists of multiple restaurants (e.g., "Top 10 restaurants in Paris")
        - Collections of restaurants in professional restaurant guides
        - Food critic reviews covering multiple restaurants
        - Articles in reputable media discussing various dining options in an area

        NOT VALID CONTENT (score < 0.3):
        - Official website of a single restaurant
        - Collections of restaurants in booking and delivery websites like Uber Eats, The Fork, Glovo, etc.
        - Wanderlog content
        - Individual restaurant menus
        - Single restaurant reviews
        - Social media posts about individual dining experiences
        - Forum/Reddit discussions without professional curation
        - Hotel booking sites
        - Video content (YouTube, TikTok, etc.)

        SCORING CRITERIA:
        - Multiple restaurants mentioned (essential)
        - Professional curation or expertise evident
        - Detailed descriptions of restaurants/cuisine
        - Location information for multiple restaurants
        - Price or quality indications for multiple venues

        FORMAT:
        Respond with a JSON object containing:
        {{
          "is_restaurant_list": true/false,
          "restaurant_count": estimated number of restaurants mentioned,
          "content_quality": 0.0-1.0,
          "reasoning": "brief explanation of your evaluation"
        }}
        """

        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.eval_system_prompt),
            ("human", "URL: {url}\n\nPage Title: {title}\n\nContent Preview:\n{preview}")
        ])

        self.eval_chain = self.eval_prompt | self.model

        # Statistics tracking
        self.filtered_urls = []
        self.evaluation_stats = {
            "total_evaluated": 0,
            "passed_filter": 0,
            "failed_filter": 0,
            "evaluation_errors": 0,
            "domain_filtered": 0
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

        logger.info(f"[SearchAgent] Initialized with excluded domains: {self.excluded_domains}")
        logger.info(f"[SearchAgent] Video platforms blocked: {len(self.video_platforms)} platforms")

    def search(self, queries, max_retries=3, retry_delay=2, enable_ai_filtering=True):
        """
        Execute multiple search queries and return combined results with enhanced logging

        Args:
            queries (list): List of search query strings
            max_retries (int): Maximum retry attempts per query
            retry_delay (int): Delay between retries in seconds
            enable_ai_filtering (bool): Whether to apply AI-based content filtering

        Returns:
            list: Combined search results from all queries
        """
        logger.info(f"[SearchAgent] Starting search with {len(queries)} queries: {queries}")
        logger.info(f"[SearchAgent] AI filtering enabled: {enable_ai_filtering}")
        logger.info(f"[SearchAgent] Search count per query: {self.search_count}")

        all_results = []
        query_stats = {}

        # Reset stats for this search session
        self.evaluation_stats = {
            "total_evaluated": 0,
            "passed_filter": 0,
            "failed_filter": 0,
            "evaluation_errors": 0,
            "domain_filtered": 0
        }

        start_time = time.time()

        with tracing_v2_enabled(project_name="restaurant-recommender"):
            for query_idx, query in enumerate(queries, 1):
                query_start = time.time()
                retry_count = 0
                success = False
                query_results = []

                logger.info(f"[SearchAgent] Query {query_idx}/{len(queries)}: '{query}'")

                while not success and retry_count < max_retries:
                    try:
                        # Execute search
                        logger.debug(f"[SearchAgent] Executing API call for: {query}")
                        raw_results = self._execute_search(query)

                        # Log raw API response
                        raw_count = len(raw_results.get('web', {}).get('results', []))
                        logger.info(f"[SearchAgent] Brave API returned {raw_count} raw results for '{query}'")

                        # Apply domain filtering
                        domain_filtered = self._filter_results(raw_results)
                        domain_filtered_count = len(domain_filtered)
                        logger.info(f"[SearchAgent] After domain filtering: {domain_filtered_count} results (removed {raw_count - domain_filtered_count})")

                        if domain_filtered:
                            # Log sample URLs
                            sample_urls = [r.get('url', 'N/A')[:60] + '...' for r in domain_filtered[:3]]
                            logger.info(f"[SearchAgent] Sample URLs: {sample_urls}")

                        # Apply AI/content filtering
                        if enable_ai_filtering and domain_filtered:
                            logger.info(f"[SearchAgent] Applying AI content filtering to {len(domain_filtered)} URLs...")
                            ai_filtered_results = self._apply_ai_filtering_sync(domain_filtered)
                            final_count = len(ai_filtered_results)
                            logger.info(f"[SearchAgent] After AI filtering: {final_count} results (removed {domain_filtered_count - final_count})")
                            query_results = ai_filtered_results
                        else:
                            query_results = domain_filtered
                            logger.info(f"[SearchAgent] Skipping AI filtering, using {len(domain_filtered)} domain-filtered results")

                        all_results.extend(query_results)
                        success = True

                        query_time = round(time.time() - query_start, 2)
                        query_stats[query] = {
                            "raw_results": raw_count,
                            "domain_filtered": domain_filtered_count,
                            "final_results": len(query_results),
                            "processing_time": query_time
                        }
                        logger.info(f"[SearchAgent] Query '{query}' completed in {query_time}s: {len(query_results)} final results")

                    except Exception as e:
                        logger.error(f"[SearchAgent] Error in search for query '{query}': {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"[SearchAgent] Retrying in {retry_delay} seconds... (attempt {retry_count + 1}/{max_retries})")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"[SearchAgent] Max retries reached for query '{query}'")

                    # Respect rate limits
                    time.sleep(1)

        total_time = round(time.time() - start_time, 2)
        logger.info(f"[SearchAgent] Search completed in {total_time}s. Total results: {len(all_results)}")

        # Log detailed statistics
        if enable_ai_filtering:
            logger.info(f"[SearchAgent] AI Filtering Stats: {self.evaluation_stats}")
            success_rate = (self.evaluation_stats['passed_filter'] / max(self.evaluation_stats['total_evaluated'], 1)) * 100
            logger.info(f"[SearchAgent] AI filtering success rate: {success_rate:.1f}%")

        # Log query-by-query breakdown
        logger.info(f"[SearchAgent] Query breakdown:")
        for query, stats in query_stats.items():
            logger.info(f"  '{query}': {stats['raw_results']} → {stats['domain_filtered']} → {stats['final_results']} ({stats['processing_time']}s)")

        # Log recently filtered URLs for debugging
        if self.filtered_urls:
            recent_filtered = self.filtered_urls[-5:] if len(self.filtered_urls) > 5 else self.filtered_urls
            logger.info(f"[SearchAgent] Recently filtered URLs (last 5): {recent_filtered}")

        # Save results to database for future reference
        if all_results:
            save_data(
                self.config.DB_TABLE_SEARCHES,
                {
                    "queries": queries,
                    "timestamp": time.time(),
                    "results": all_results,
                    "ai_filtering_enabled": enable_ai_filtering,
                    "filtering_stats": self.evaluation_stats.copy(),
                    "query_stats": query_stats,
                    "total_processing_time": total_time
                },
                self.config
            )
            logger.info(f"[SearchAgent] Search results saved to database")

        return all_results

    def _apply_ai_filtering_sync(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply basic content filtering without async operations (enhanced with logging)
        """
        logger.debug(f"[SearchAgent] Starting content filtering for {len(search_results)} results")

        # First, apply domain-based filtering to remove video platforms
        domain_filtered_results = []
        video_filtered_count = 0

        for result in search_results:
            url = result.get('url', '')
            if self._is_video_platform(url):
                logger.debug(f"[SearchAgent] Video platform filtered: {url[:50]}...")
                self.evaluation_stats["domain_filtered"] += 1
                self._track_filtering_reason(url, "Video/social platform")
                video_filtered_count += 1
                continue

            domain_filtered_results.append(result)

        if video_filtered_count > 0:
            logger.info(f"[SearchAgent] Filtered {video_filtered_count} video platform URLs")

        # Now apply basic keyword filtering
        filtered_results = []
        keyword_stats = {"passed": 0, "failed": 0}

        for result in domain_filtered_results:
            url = result.get("url", "")
            title = result.get("title", "")
            description = result.get("description", "")

            # Enhanced keyword evaluation with logging
            evaluation = self._basic_keyword_evaluation(url, title, description)
            self.evaluation_stats["total_evaluated"] += 1

            if evaluation.get("passed_filter", False):
                result["ai_evaluation"] = evaluation
                filtered_results.append(result)
                self.evaluation_stats["passed_filter"] += 1
                keyword_stats["passed"] += 1
                logger.debug(f"[SearchAgent] ✅ PASSED: {title[:50]}... (score: {evaluation.get('content_quality', 0):.2f})")
            else:
                self.evaluation_stats["failed_filter"] += 1
                keyword_stats["failed"] += 1
                self._track_filtering_reason(url, f"Low content score: {evaluation.get('content_quality', 0):.2f}")
                logger.debug(f"[SearchAgent] ❌ FAILED: {title[:50]}... (score: {evaluation.get('content_quality', 0):.2f})")

        logger.info(f"[SearchAgent] Keyword filtering: {keyword_stats['passed']} passed, {keyword_stats['failed']} failed")
        return filtered_results

    async def _apply_ai_filtering(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply AI-based content filtering to search results with domain pre-filtering
        """
        logger.info(f"[SearchAgent] Starting AI filtering for {len(search_results)} results")

        # First, apply domain-based filtering to remove obvious video platforms
        domain_filtered_results = []

        for result in search_results:
            url = result.get('url', '')
            if self._is_video_platform(url):
                logger.debug(f"[SearchAgent] Domain-filtered video platform: {url}")
                self.evaluation_stats["domain_filtered"] += 1
                self._track_filtering_reason(url, "Video platform")
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
                logger.error(f"[SearchAgent] Error evaluating {result.get('url', 'unknown')}: {evaluation}")
                self.evaluation_stats["evaluation_errors"] += 1
                # Include result if evaluation failed (conservative approach)
                filtered_results.append(result)
            elif evaluation and evaluation.get("passed_filter", False):
                # Add evaluation metadata to the result
                result["ai_evaluation"] = evaluation
                filtered_results.append(result)
                logger.debug(f"[SearchAgent] AI passed: {result.get('title', 'N/A')[:50]}...")
            else:
                # Result was filtered out
                self._track_filtering_reason(result.get("url", "unknown"), "AI evaluation failed")
                logger.debug(f"[SearchAgent] AI filtered: {result.get('title', 'N/A')[:50]}...")

        logger.info(f"[SearchAgent] AI filtering completed: {len(filtered_results)} results passed")
        return filtered_results

    def _is_video_platform(self, url: str) -> bool:
        """
        Check if URL is from a video/social media platform that should be excluded
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()

            # Remove www. prefix for comparison
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain in self.video_platforms
        except Exception:
            return False

    async def _evaluate_search_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single search result using AI
        """
        try:
            url = result.get("url", "")
            title = result.get("title", "")
            description = result.get("description", "")

            # Create preview text for evaluation
            preview_text = f"Title: {title}\nDescription: {description}"

            # Use AI to evaluate the content
            response = await self.eval_chain.ainvoke({
                "url": url,
                "title": title,
                "preview": preview_text
            })

            # Parse the AI response
            try:
                evaluation_data = json.loads(response.content)

                # Determine if result passes filter
                content_quality = evaluation_data.get("content_quality", 0.0)
                is_restaurant_list = evaluation_data.get("is_restaurant_list", False)

                passed_filter = content_quality > 0.5 and is_restaurant_list

                self.evaluation_stats["total_evaluated"] += 1
                if passed_filter:
                    self.evaluation_stats["passed_filter"] += 1
                else:
                    self.evaluation_stats["failed_filter"] += 1

                return {
                    "passed_filter": passed_filter,
                    "restaurant_count": evaluation_data.get("restaurant_count", 0),
                    "content_quality": content_quality,
                    "reasoning": evaluation_data.get("reasoning", "AI evaluation"),
                    "is_restaurant_list": is_restaurant_list
                }

            except json.JSONDecodeError:
                logger.error(f"[SearchAgent] Failed to parse AI evaluation response for {url}")
                self.evaluation_stats["evaluation_errors"] += 1
                return None

        except Exception as e:
            logger.error(f"[SearchAgent] Error evaluating result: {e}")
            self.evaluation_stats["evaluation_errors"] += 1
            return None

    def _basic_keyword_evaluation(self, url: str, title: str, description: str) -> Dict[str, Any]:
        """
        Enhanced basic keyword evaluation with detailed logging
        """
        # Positive indicators for restaurant lists/guides
        positive_keywords = [
            "best restaurants", "top restaurants", "restaurant guide", "food guide",
            "dining guide", "restaurants in", "places to eat", "where to eat",
            "restaurant recommendations", "food recommendations", "must try",
            "foodie guide", "culinary guide", "eats", "dine", "bistro", "cafe",
            "michelin", "zagat", "conde nast", "time out", "eater", "serious eats"
        ]

        # Negative indicators (single restaurant, booking, delivery sites, etc.)
        negative_keywords = [
            "uber eats", "deliveroo", "grubhub", "doordash", "just eat", "foodpanda",
            "the fork", "opentable", "resy", "booking.com", "expedia", "trivago",
            "wanderlog", "tripadvisor", "yelp", "foursquare", "zomato",
            "menu", "order online", "delivery", "takeaway", "book a table",
            "make reservation", "single restaurant", "one restaurant"
        ]

        combined_text = f"{title} {description}".lower()

        # Count positive and negative indicators
        positive_score = sum(1 for keyword in positive_keywords if keyword in combined_text)
        negative_score = sum(1 for keyword in negative_keywords if keyword in combined_text)

        # Find which keywords matched for logging
        matched_positive = [k for k in positive_keywords if k in combined_text]
        matched_negative = [k for k in negative_keywords if k in combined_text]

        # Calculate content quality score
        content_quality = max(0.0, min(1.0, (positive_score - negative_score * 0.5) / 3.0))
        passed_filter = content_quality > 0.3 and positive_score > 0

        # Enhanced logging
        logger.debug(f"[SearchAgent] Evaluating: {title[:50]}...")
        logger.debug(f"[SearchAgent] URL: {url}")
        logger.debug(f"[SearchAgent] Positive keywords ({positive_score}): {matched_positive}")
        logger.debug(f"[SearchAgent] Negative keywords ({negative_score}): {matched_negative}")
        logger.debug(f"[SearchAgent] Final score: {content_quality:.2f}, Passed: {passed_filter}")

        return {
            "passed_filter": passed_filter,
            "restaurant_count": positive_score,
            "content_quality": content_quality,
            "reasoning": f"Basic keyword evaluation: {positive_score} positive, {negative_score} negative keywords",
            "positive_keywords": matched_positive,
            "negative_keywords": matched_negative
        }

    def _execute_search(self, query):
        """Execute a single search query against Brave Search API with logging"""
        logger.debug(f"[SearchAgent] Executing Brave API call for: {query}")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": self.search_count,
            "freshness": "month"  # Get recent results
        }

        start_time = time.time()
        response = requests.get(
            self.base_url,
            headers=headers,
            params=params
        )
        api_time = round(time.time() - start_time, 2)

        logger.debug(f"[SearchAgent] Brave API call completed in {api_time}s")

        if response.status_code != 200:
            logger.error(f"[SearchAgent] Brave Search API error: {response.status_code}, {response.text}")
            raise Exception(f"Brave Search API error: {response.status_code}, {response.text}")

        result_data = response.json()
        logger.debug(f"[SearchAgent] API response contains {len(result_data.get('web', {}).get('results', []))} results")

        return result_data

    def _filter_results(self, search_results):
        """Filter search results to exclude unwanted domains with enhanced logging"""
        if not search_results or "web" not in search_results or "results" not in search_results["web"]:
            logger.warning(f"[SearchAgent] No valid web results in API response")
            return []

        raw_results = search_results["web"]["results"]
        filtered_results = []
        excluded_count = 0

        logger.debug(f"[SearchAgent] Filtering {len(raw_results)} raw results against {len(self.excluded_domains)} excluded domains")

        for result in raw_results:
            url = result.get("url", "")
            title = result.get("title", "")

            # Check if URL contains any excluded domain
            excluded_domain = None
            for excluded in self.excluded_domains:
                if excluded in url:
                    excluded_domain = excluded
                    break

            if excluded_domain:
                logger.debug(f"[SearchAgent] Excluded '{title[:50]}...' (domain: {excluded_domain})")
                self._track_filtering_reason(url, f"Excluded domain: {excluded_domain}")
                excluded_count += 1
            else:
                # Clean and extract the relevant information
                filtered_result = {
                    "title": title,
                    "url": url,
                    "description": result.get("description", ""),
                    "language": result.get("language", "en"),
                    "favicon": result.get("favicon", "")
                }
                filtered_results.append(filtered_result)

        logger.debug(f"[SearchAgent] Domain filtering completed: {len(filtered_results)} passed, {excluded_count} excluded")
        return filtered_results

    def _track_filtering_reason(self, url: str, reason: str):
        """Track why a URL was filtered (for debugging)"""
        timestamp = time.time()
        filter_entry = {
            "url": url,
            "reason": reason,
            "timestamp": timestamp
        }

        self.filtered_urls.append(filter_entry)

        # Keep only last 50 for memory management
        if len(self.filtered_urls) > 50:
            self.filtered_urls = self.filtered_urls[-50:]

    def get_filtering_stats(self) -> Dict[str, Any]:
        """Get current filtering statistics for debugging"""
        return {
            "evaluation_stats": self.evaluation_stats.copy(),
            "recently_filtered_urls": self.filtered_urls[-10:] if self.filtered_urls else [],
            "total_filtered_urls": len(self.filtered_urls),
            "excluded_domains": self.excluded_domains,
            "video_platforms": list(self.video_platforms)
        }

    def follow_up_search(self, restaurant_name, location, additional_context=None):
        """
        Perform a follow-up search for a specific restaurant with logging

        Args:
            restaurant_name (str): Name of the restaurant
            location (str): Location of the restaurant
            additional_context (str, optional): Additional search context

        Returns:
            list: Search results for the specific restaurant
        """
        logger.info(f"[SearchAgent] Follow-up search for: {restaurant_name} in {location}")

        # Construct focused search queries
        base_query = f'"{restaurant_name}" {location}'
        queries = [base_query]

        if additional_context:
            queries.append(f'"{restaurant_name}" {location} {additional_context}')

        # Add review-focused query
        queries.append(f'"{restaurant_name}" {location} review')

        logger.info(f"[SearchAgent] Follow-up queries: {queries}")

        # Execute search with lower count for follow-up
        original_count = self.search_count
        self.search_count = 5  # Fewer results for follow-up

        try:
            results = self.search(queries, enable_ai_filtering=False)  # Skip AI filtering for follow-up
            logger.info(f"[SearchAgent] Follow-up search returned {len(results)} results")
            return results
        finally:
            self.search_count = original_count  # Restore original count