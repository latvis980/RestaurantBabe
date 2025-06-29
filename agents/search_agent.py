# agents/search_agent.py - SIMPLIFIED VERSION
# Remove keyword filtering, keep only domain + AI filtering

import requests
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
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
        Execute multiple search queries and return combined results with simplified filtering

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

                        # Apply domain filtering only
                        domain_filtered = self._filter_results(raw_results)
                        domain_filtered_count = len(domain_filtered)
                        logger.info(f"[SearchAgent] After domain filtering: {domain_filtered_count} results (removed {raw_count - domain_filtered_count})")

                        if domain_filtered:
                            # Log sample URLs
                            sample_urls = [r.get('url', 'N/A')[:60] + '...' for r in domain_filtered[:3]]
                            logger.info(f"[SearchAgent] Sample URLs: {sample_urls}")

                        # Apply AI filtering if enabled
                        if enable_ai_filtering and domain_filtered:
                            logger.info(f"[SearchAgent] Applying AI content filtering to {len(domain_filtered)} URLs...")
                            ai_filtered_results = asyncio.run(self._apply_ai_filtering(domain_filtered))
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

    def _execute_search(self, query):
        """Execute search using Brave API"""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": self.search_count
        }

        response = requests.get(self.base_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def _filter_results(self, brave_response):
        """Apply domain filtering only (no keyword filtering)"""
        try:
            results = brave_response.get('web', {}).get('results', [])
            filtered_results = []

            for result in results:
                url = result.get('url', '')

                # Check if URL should be excluded by domain
                if self._should_exclude_domain(url):
                    logger.debug(f"[SearchAgent] Domain-filtered: {url}")
                    self.evaluation_stats["domain_filtered"] += 1
                    self._track_filtering_reason(url, "Excluded domain")
                    continue

                # Check if it's a video platform
                if self._is_video_platform(url):
                    logger.debug(f"[SearchAgent] Video platform filtered: {url}")
                    self.evaluation_stats["domain_filtered"] += 1
                    self._track_filtering_reason(url, "Video/social platform")
                    continue

                filtered_results.append(result)

            return filtered_results

        except Exception as e:
            logger.error(f"[SearchAgent] Error filtering results: {e}")
            return []

    def _should_exclude_domain(self, url):
        """Check if URL domain should be excluded"""
        try:
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix for comparison
            if domain.startswith('www.'):
                domain = domain[4:]

            return any(excluded in domain for excluded in self.excluded_domains)
        except Exception:
            return False

    def _is_video_platform(self, url):
        """Check if URL is from a video/social platform"""
        try:
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix for comparison
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain in self.video_platforms
        except Exception:
            return False

    async def _apply_ai_filtering(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply AI-based content filtering to search results
        """
        logger.info(f"[SearchAgent] Starting AI filtering for {len(search_results)} results")

        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(3)  # Allow up to 3 concurrent evaluations

        async def evaluate_single_result(result):
            async with semaphore:
                return await self._evaluate_search_result(result)

        # Process all results concurrently
        evaluations = await asyncio.gather(
            *[evaluate_single_result(result) for result in search_results],
            return_exceptions=True
        )

        # Filter results based on evaluations
        filtered_results = []
        for result, evaluation in zip(search_results, evaluations):
            if isinstance(evaluation, Exception):
                logger.error(f"[SearchAgent] Evaluation error for {result.get('url', 'unknown')}: {evaluation}")
                self.evaluation_stats["evaluation_errors"] += 1
                continue

            if evaluation and evaluation.get("passed_filter", False):
                result["ai_evaluation"] = evaluation
                filtered_results.append(result)
                self.evaluation_stats["passed_filter"] += 1
                logger.debug(f"[SearchAgent] ✅ PASSED: {result.get('title', 'N/A')[:50]}... (score: {evaluation.get('content_quality', 0):.2f})")
            else:
                self.evaluation_stats["failed_filter"] += 1
                reason = evaluation.get('reasoning', 'Failed AI evaluation') if evaluation else 'Evaluation failed'
                self._track_filtering_reason(result.get('url', ''), reason)
                logger.debug(f"[SearchAgent] ❌ FAILED: {result.get('title', 'N/A')[:50]}... (reason: {reason})")

        return filtered_results

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

                # Simplified filtering: pass if it's a restaurant list with decent quality
                passed_filter = content_quality > 0.5 and is_restaurant_list

                self.evaluation_stats["total_evaluated"] += 1

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

    def _track_filtering_reason(self, url: str, reason: str):
        """Track why a URL was filtered out"""
        self.filtered_urls.append({
            "url": url,
            "reason": reason,
            "timestamp": time.time()
        })

        # Keep only the last 50 filtered URLs to prevent memory issues
        if len(self.filtered_urls) > 50:
            self.filtered_urls = self.filtered_urls[-50:]