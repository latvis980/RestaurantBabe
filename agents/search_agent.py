# File: agents/search_agent.py
# Replace your current search method with this fixed version

import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class BraveSearchAgent:
    def __init__(self, config):
        self.config = config
        # ... your existing initialization code ...

        # Initialize stats properly
        self.reset_filtering_stats()

    def reset_filtering_stats(self):
        """Reset filtering statistics for a new search"""
        self.filtered_urls = []
        self.evaluation_stats = {
            "total_evaluated": 0,
            "passed_filter": 0,
            "failed_filter": 0,
            "evaluation_errors": 0,
            "domain_filtered": 0
        }

    def search(self, queries, max_retries=3, retry_delay=2, enable_ai_filtering=True):
        """
        Execute multiple search queries and return combined results

        Args:
            queries (list): List of search query strings
            max_retries (int): Maximum retry attempts per query
            retry_delay (int): Delay between retries in seconds
            enable_ai_filtering (bool): Whether to apply AI-based content filtering

        Returns:
            list: Combined search results from all queries
        """
        all_results = []

        # Reset stats for this search session
        self.reset_filtering_stats()

        for query in queries:
            retry_count = 0
            success = False

            while not success and retry_count < max_retries:
                try:
                    logger.info(f"[SearchAgent] Searching for: {query}")
                    results = self._execute_search(query)
                    raw_count = len(results.get('web', {}).get('results', []))
                    logger.info(f"[SearchAgent] Raw results count: {raw_count}")

                    # Apply domain filtering first
                    filtered_results = self._filter_results(results)
                    logger.info(f"[SearchAgent] After domain filtering: {len(filtered_results)} results")

                    # Apply AI filtering if enabled and we have results
                    if enable_ai_filtering and filtered_results:
                        logger.info(f"[SearchAgent] Applying AI content filtering...")

                        # FIXED: Use proper async handling for AI filtering
                        try:
                            # Create new event loop if none exists
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_closed():
                                    raise RuntimeError("Event loop is closed")
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                            ai_filtered_results = loop.run_until_complete(
                                self._apply_ai_filtering_async(filtered_results)
                            )
                            logger.info(f"[SearchAgent] AI-filtered results count: {len(ai_filtered_results)}")
                            all_results.extend(ai_filtered_results)

                        except Exception as ai_error:
                            logger.warning(f"AI filtering failed, using domain-filtered results: {ai_error}")
                            # Fallback to domain-filtered results if AI filtering fails
                            all_results.extend(filtered_results)
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

        # Log filtering statistics
        if enable_ai_filtering:
            logger.info(f"[SearchAgent] AI Filtering Stats: {self.evaluation_stats}")

        return all_results

    async def _apply_ai_filtering_async(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply AI-based content filtering to search results

        Args:
            search_results: List of search result dictionaries

        Returns:
            List of filtered search results that pass AI evaluation
        """
        if not search_results:
            return []

        # Apply additional domain-based filtering for video platforms
        domain_filtered_results = []

        for result in search_results:
            url = result.get('url', '')
            if self._is_video_platform(url):
                logger.info(f"Domain-filtered video platform: {url}")
                self.evaluation_stats["domain_filtered"] += 1
                self.filtered_urls.append(url)
                continue
            domain_filtered_results.append(result)

        logger.info(f"[SearchAgent] After video platform filtering: {len(domain_filtered_results)} results")

        # If we don't have many results, be more lenient with AI filtering
        if len(domain_filtered_results) <= 3:
            logger.info(f"[SearchAgent] Few results ({len(domain_filtered_results)}), using lenient filtering")
            return await self._lenient_ai_filtering(domain_filtered_results)

        # Apply full AI filtering for larger result sets
        return await self._strict_ai_filtering(domain_filtered_results)

    async def _lenient_ai_filtering(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply lenient AI filtering when we have few results
        """
        filtered_results = []

        for result in search_results:
            try:
                # Use basic keyword evaluation for lenient filtering
                evaluation = self._basic_keyword_evaluation(
                    result.get("url", ""),
                    result.get("title", ""),
                    result.get("description", "")
                )

                # More lenient criteria
                if (evaluation.get("passed_filter", False) or 
                    any(keyword in result.get("title", "").lower() for keyword in 
                        ["restaurant", "food", "dining", "ceviche", "best", "guide"])):

                    result["ai_evaluation"] = evaluation
                    filtered_results.append(result)
                    self.evaluation_stats["passed_filter"] += 1
                else:
                    self.evaluation_stats["failed_filter"] += 1
                    self.filtered_urls.append(result.get("url", "unknown"))

            except Exception as e:
                logger.warning(f"Error in lenient filtering for {result.get('url', 'unknown')}: {e}")
                # Include result if evaluation failed (conservative approach)
                filtered_results.append(result)
                self.evaluation_stats["evaluation_errors"] += 1

        return filtered_results

    async def _strict_ai_filtering(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply strict AI filtering for larger result sets
        """
        filtered_results = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent AI evaluations

        async def evaluate_single_result(result):
            async with semaphore:
                return await self._evaluate_search_result(result)

        # Create tasks for all evaluations
        tasks = [evaluate_single_result(result) for result in search_results]

        # Wait for all evaluations to complete with timeout
        try:
            evaluation_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning("AI filtering timed out, falling back to basic filtering")
            return await self._lenient_ai_filtering(search_results)

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
                self.evaluation_stats["passed_filter"] += 1
            else:
                # Result was filtered out
                self.evaluation_stats["failed_filter"] += 1
                self.filtered_urls.append(result.get("url", "unknown"))

        return filtered_results

    def _basic_keyword_evaluation(self, url: str, title: str, description: str) -> Dict[str, Any]:
        """
        Basic keyword-based evaluation fallback

        Args:
            url: URL to evaluate
            title: Page title
            description: Page description

        Returns:
            Evaluation result dictionary
        """
        # Combine all text for analysis
        combined_text = f"{title} {description}".lower()

        # Restaurant-related keywords
        restaurant_keywords = [
            "restaurant", "dining", "food", "eat", "chef", "cuisine", "menu", "dish",
            "ceviche", "cevicheria", "seafood", "peruvian", "best", "guide", "review"
        ]

        # Quality indicators
        quality_indicators = [
            "best", "guide", "review", "recommendation", "top", "guide", "where to eat"
        ]

        # Count keyword matches
        keyword_matches = sum(1 for keyword in restaurant_keywords if keyword in combined_text)
        quality_matches = sum(1 for indicator in quality_indicators if indicator in combined_text)

        # Simple scoring
        keyword_score = min(keyword_matches / 3.0, 1.0)  # Normalize to 0-1
        quality_score = min(quality_matches / 2.0, 1.0)   # Normalize to 0-1

        overall_score = (keyword_score + quality_score) / 2.0
        passed_filter = overall_score >= 0.3  # Lower threshold for basic evaluation

        return {
            "passed_filter": passed_filter,
            "is_restaurant_list": keyword_matches >= 2,
            "restaurant_count": max(1, keyword_matches),
            "content_quality": overall_score,
            "reasoning": f"Basic keyword evaluation: {keyword_matches} restaurant keywords, {quality_matches} quality indicators"
        }

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
            # First, do a basic keyword check to avoid unnecessary LLM calls
            basic_eval = self._basic_keyword_evaluation(url, title, description)

            # If basic evaluation fails completely, don't waste LLM calls
            if basic_eval["content_quality"] < 0.1:
                logger.info(f"URL filtered by basic keyword check: {url}")
                return basic_eval

            # Try AI evaluation with timeout
            try:
                response = await asyncio.wait_for(
                    self.eval_chain.ainvoke({
                        "url": url,
                        "title": title,
                        "preview": f"{title}\n\n{description}"
                    }),
                    timeout=10.0  # 10 second timeout per evaluation
                )

                # FIXED: Proper JSON parsing
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)

                # Clean up the content and parse JSON
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                try:
                    evaluation = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse AI response as JSON for {url}: {e}")
                    logger.debug(f"Raw response: {content}")
                    # Fallback to basic evaluation
                    return basic_eval

                # Validate the evaluation structure
                if not isinstance(evaluation, dict):
                    logger.warning(f"AI evaluation is not a dict for {url}")
                    return basic_eval

                # Ensure required fields exist
                evaluation.setdefault("passed_filter", False)
                evaluation.setdefault("is_restaurant_list", False)
                evaluation.setdefault("restaurant_count", 0)
                evaluation.setdefault("content_quality", 0.0)
                evaluation.setdefault("reasoning", "AI evaluation")

                return evaluation

            except asyncio.TimeoutError:
                logger.warning(f"AI evaluation timed out for {url}, using basic evaluation")
                return basic_eval
            except Exception as ai_error:
                logger.warning(f"AI evaluation failed for {url}: {ai_error}, using basic evaluation")
                return basic_eval

        except Exception as e:
            logger.error(f"Error evaluating {url}: {e}")
            # Return a conservative evaluation that includes the result
            return {
                "passed_filter": True,
                "is_restaurant_list": True,
                "restaurant_count": 1,
                "content_quality": 0.5,
                "reasoning": f"Evaluation error, included conservatively: {str(e)}"
            }

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

            # Video platforms to exclude
            video_platforms = {
                'youtube.com', 'youtu.be', 'tiktok.com', 'instagram.com',
                'facebook.com', 'twitter.com', 'x.com', 'vimeo.com',
                'dailymotion.com', 'twitch.tv', 'pinterest.com', 'snapchat.com'
            }

            return domain in video_platforms

        except Exception as e:
            logger.warning(f"Error parsing URL for video platform check: {url}, error: {e}")
            return False

    def get_filtering_stats(self):
        """Get current AI filtering statistics"""
        return {
            "evaluation_stats": self.evaluation_stats.copy(),
            "filtered_urls": self.filtered_urls.copy()
        }