# Search agent with AI-based filtering system - DESTINATION FIX

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

        # FIXED: Updated system prompt to handle destination context properly
        self.eval_system_prompt = """
        You are an expert at evaluating web content about restaurants.
        Your task is to analyze if a web page contains a curated list of restaurants or restaurant recommendations.

        VALID CONTENT (score > 0.7):
        - Curated lists of multiple restaurants (e.g., "Top 10 restaurants in Paris")
        - Collections of restaurants in professional restaurant guides
        - Food critic reviews covering multiple restaurants
        - Articles in reputable local media discussing various dining options in an area

        NOT VALID CONTENT (score < 0.3):
        - Official website of a single restaurant
        - Collections of restaurants in booking and delivery websites like Uber Eats, The Fork, Glovo, Bolt, etc.
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

        DESTINATION MATCHING:
        If a destination is provided, consider whether the content is relevant to that location.
        However, do NOT reject content solely based on location mismatch if it contains valid restaurant recommendations.
        The primary criteria should be whether it's a curated restaurant list.

        FORMAT:
        Respond with a JSON object containing:
        {
          "is_restaurant_list": true/false,
          "restaurant_count": estimated number of restaurants mentioned,
          "content_quality": 0.0-1.0,
          "destination_match": "relevant"/"somewhat_relevant"/"not_relevant",
          "reasoning": "brief explanation of your evaluation"
        }
        """

        # FIXED: Updated prompt template to include destination parameter
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.eval_system_prompt),
            ("human", "URL: {url}\n\nPage Title: {title}\n\nContent Preview:\n{preview}\n\nTarget Destination: {destination}")
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
            "model_used": config.SEARCH_EVALUATION_MODEL,
            "estimated_cost_saved": 0.0
        }

    # FIXED: Updated search method to accept and pass destination parameter
    def search(self, queries, destination="Unknown", max_retries=3, retry_delay=2, enable_ai_filtering=True):
        """
        Perform searches with the given queries and optional AI filtering

        Args:
            queries (list): List of search queries
            destination (str): Target destination for restaurant recommendations
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

                        # Apply AI filtering if enabled - FIXED: Pass destination parameter
                        if enable_ai_filtering and filtered_results:
                            logger.info(f"[SearchAgent] Applying AI content filtering...")
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

                    time.sleep(1)

        logger.info(f"[SearchAgent] Total search results after all filtering: {len(all_results)}")

        if enable_ai_filtering:
            logger.info(f"[SearchAgent] AI Filtering Stats: {self.evaluation_stats}")

        # Cache search results
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
        """Run an async coroutine in a new thread with its own event loop"""
        def run_in_new_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(run_in_new_event_loop).result()

    # FIXED: Updated to accept destination parameter
    async def _apply_ai_filtering(self, search_results: List[Dict[str, Any]], destination: str = "Unknown") -> List[Dict[str, Any]]:
        """
        Apply AI-based content filtering to search results

        Args:
            search_results: List of search result dictionaries
            destination: Target destination for context

        Returns:
            List of filtered search results that pass AI evaluation
        """
        filtered_results = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent AI evaluations

        async def evaluate_single_result(result):
            async with semaphore:
                return await self._evaluate_search_result(result, destination)

        # Create tasks for all evaluations
        tasks = [evaluate_single_result(result) for result in search_results]

        # Wait for all evaluations to complete
        evaluation_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result, evaluation in zip(search_results, evaluation_results):
            if isinstance(evaluation, Exception):
                logger.error(f"Error evaluating {result.get('url', 'unknown')}: {evaluation}")
                self.evaluation_stats["evaluation_errors"] += 1
                # Include result if evaluation failed (conservative approach)
                filtered_results.append(result)
            elif evaluation and evaluation.get("passed_filter", False):
                # Add evaluation metadata to the result
                result["ai_evaluation"] = evaluation
                filtered_results.append(result)
                logger.info(f"✅ AI approved: {result.get('url')} - {evaluation.get('reasoning', 'No reason')}")
            else:
                # Result was filtered out
                self.filtered_urls.append(result.get("url", "unknown"))
                logger.info(f"❌ AI rejected: {result.get('url')} - {evaluation.get('reasoning', 'No reason')}")

        logger.info(f"[SearchAgent] AI filtering: {len(filtered_results)}/{len(search_results)} URLs passed")
        return filtered_results

    # FIXED: Updated to accept destination parameter
    async def _evaluate_search_result(self, result: Dict[str, Any], destination: str = "Unknown") -> Optional[Dict[str, Any]]:
        """
        Evaluate a single search result using AI

        Args:
            result: Search result dictionary
            destination: Target destination for context

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
            restaurant_keywords = ["restaurant", "dining", "food", "eat", "chef", "cuisine", "menu", "dish", "bar", "cafe", "steakhouse"]
            if not any(kw in full_preview.lower() for kw in restaurant_keywords):
                logger.info(f"URL filtered by basic keyword check: {url}")
                self.evaluation_stats["failed_filter"] += 1
                return {
                    "passed_filter": False,
                    "is_restaurant_list": False,
                    "restaurant_count": 0,
                    "content_quality": 0.0,
                    "destination_match": "not_relevant",
                    "reasoning": "No restaurant-related keywords found"
                }

            # FIXED: AI evaluation with destination parameter
            response = await self.eval_chain.ainvoke({
                "url": url,
                "title": title,
                "preview": full_preview[:1500],
                "destination": destination  # CRITICAL FIX: Now passing destination!
            })

            # Parse AI response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            try:
                evaluation = json.loads(content.strip())
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {url}: {e}")
                logger.error(f"Raw response: {content[:300]}")
                # Return conservative result if JSON parsing fails
                return {
                    "passed_filter": True,
                    "is_restaurant_list": True,
                    "restaurant_count": 0,
                    "content_quality": 0.5,
                    "destination_match": "unknown",
                    "reasoning": f"JSON parsing error, passed conservatively: {str(e)}"
                }

            # Ensure required fields are present
            evaluation.setdefault("content_quality", 0.5)
            evaluation.setdefault("is_restaurant_list", False)
            evaluation.setdefault("restaurant_count", 0)
            evaluation.setdefault("destination_match", "unknown")
            evaluation.setdefault("reasoning", "No reasoning provided")

            # Apply threshold - prioritize restaurant list detection over destination matching
            threshold = 0.5
            is_restaurant_list = evaluation.get("is_restaurant_list", False)
            content_quality = evaluation.get("content_quality", 0.0)

            # FIXED: Don't filter based on destination match, only on restaurant content
            passed_filter = is_restaurant_list and content_quality > threshold

            if passed_filter:
                self.evaluation_stats["passed_filter"] += 1
            else:
                self.evaluation_stats["failed_filter"] += 1

            # Enhanced logging
            dest_match = evaluation.get("destination_match", "unknown")
            logger.info(f"AI evaluation for {url}: List={is_restaurant_list}, Quality={content_quality:.2f}, Dest={dest_match}, Pass={passed_filter}")

            return {
                "passed_filter": passed_filter,
                "is_restaurant_list": is_restaurant_list,
                "restaurant_count": evaluation.get("restaurant_count", 0),
                "content_quality": content_quality,
                "destination_match": dest_match,
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
                "destination_match": "unknown",
                "reasoning": f"Evaluation error: {str(e)}"
            }

    async def _fetch_content_preview(self, url: str) -> str:
        """Fetch a brief content preview from URL for evaluation"""
        try:
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
                            return preview_text[:1000]

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

    def _basic_keyword_evaluation(self, url: str, title: str, description: str) -> Dict[str, Any]:
        """Apply basic keyword-based evaluation when content fetching fails"""
        combined_text = f"{title} {description}".lower()

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

        # Simple scoring
        if positive_score > negative_score and positive_score > 0:
            self.evaluation_stats["passed_filter"] += 1
            return {
                "passed_filter": True,
                "is_restaurant_list": True,
                "restaurant_count": 5,
                "content_quality": 0.6,
                "destination_match": "unknown",
                "reasoning": f"Basic keyword evaluation: {positive_score} positive, {negative_score} negative keywords"
            }
        else:
            self.evaluation_stats["failed_filter"] += 1
            return {
                "passed_filter": False,
                "is_restaurant_list": False,
                "restaurant_count": 0,
                "content_quality": 0.3,
                "destination_match": "unknown",
                "reasoning": f"Basic keyword evaluation: {positive_score} positive, {negative_score} negative keywords"
            }

    # ... (rest of the methods remain the same)

    def _execute_search(self, query):
        """Execute a single search query against Brave Search API"""
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": self.search_count,
            "freshness": "month"
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
            if domain.startswith('www.'):
                domain = domain[4:]
            return any(excluded in domain for excluded in self.excluded_domains)
        except Exception:
            return False

    def get_filtering_stats(self):
        """Get current AI filtering statistics with cost information"""
        if self.config.SEARCH_EVALUATION_MODEL == "gpt-4o-mini":
            cost_multiplier = 0.05
            estimated_calls = self.evaluation_stats["total_evaluated"]
            estimated_savings_per_call = 0.012
            self.evaluation_stats["estimated_cost_saved"] = estimated_calls * estimated_savings_per_call * 0.95

        return {
            "evaluation_stats": self.evaluation_stats.copy(),
            "filtered_urls": self.filtered_urls.copy()
        }