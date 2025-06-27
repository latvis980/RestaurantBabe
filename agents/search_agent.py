# Search agent with AI-based filteing system

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

        # AI evaluation system prompt - extracted from your scraper
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
            "evaluation_errors": 0
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
                            ai_filtered_results = asyncio.run(self._apply_ai_filtering(filtered_results))
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

        # Save results to database for future reference
        if all_results:
            save_data(
                self.config.DB_TABLE_SEARCHES,
                {
                    "queries": queries,
                    "timestamp": time.time(),
                    "results": all_results,
                    "ai_filtering_enabled": enable_ai_filtering,
                    "filtering_stats": self.evaluation_stats.copy()
                },
                self.config
            )

        return all_results

    async def _apply_ai_filtering(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply AI-based content filtering to search results

        Args:
            search_results: List of search result dictionaries

        Returns:
            List of filtered search results that pass AI evaluation
        """
        filtered_results = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent AI evaluations

        async def evaluate_single_result(result):
            async with semaphore:
                return await self._evaluate_search_result(result)

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
            else:
                # Result was filtered out
                self.filtered_urls.append(result.get("url", "unknown"))

        return filtered_results

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
            restaurant_keywords = ["restaurant", "dining", "food", "eat", "chef", "cuisine", "menu", "dish"]
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

            # Apply threshold
            threshold = 0.5
            is_restaurant_list = evaluation.get("is_restaurant_list", False)
            content_quality = evaluation.get("content_quality", 0.0)
            passed_filter = is_restaurant_list and content_quality > threshold

            if passed_filter:
                self.evaluation_stats["passed_filter"] += 1
            else:
                self.evaluation_stats["failed_filter"] += 1

            # Log evaluation details
            logger.info(f"AI evaluation for {url}: List={is_restaurant_list}, Quality={content_quality:.2f}, Pass={passed_filter}")

            return {
                "passed_filter": passed_filter,
                "is_restaurant_list": is_restaurant_list,
                "restaurant_count": evaluation.get("restaurant_count", 0),
                "content_quality": content_quality,
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
        """Filter search results to exclude unwanted domains"""
        if not search_results or "web" not in search_results or "results" not in search_results["web"]:
            return []

        filtered_results = []

        for result in search_results["web"]["results"]:
            # Skip results from excluded domains
            if not any(excluded in result.get("url", "") for excluded in self.excluded_domains):
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
        """Get current AI filtering statistics"""
        return {
            "evaluation_stats": self.evaluation_stats.copy(),
            "filtered_urls": self.filtered_urls.copy()
        }