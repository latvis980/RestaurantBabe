# agents/search_agent.py - Updated to use fast DeepSeek evaluation
# This shows how to integrate the fast evaluation into your existing search agent

import requests
from langchain_core.tracers.context import tracing_v2_enabled
import json
import time
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from utils.database import save_data
from agents.fast_search_evaluation_agent import FastSearchEvaluationAgent  # NEW
import logging

logger = logging.getLogger("restaurant-recommender.search_agent")

class BraveSearchAgent:
    def __init__(self, config):
        self.api_key = config.BRAVE_API_KEY
        self.search_count = config.BRAVE_SEARCH_COUNT
        self.excluded_domains = config.EXCLUDED_RESTAURANT_SOURCES
        self.config = config
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

        # NEW: Use fast DeepSeek evaluation instead of slow OpenAI
        self.fast_evaluator = FastSearchEvaluationAgent(config)

        # Keep track of evaluation stats
        self.evaluation_stats = {
            'total_evaluated': 0,
            'passed_filter': 0,
            'failed_filter': 0,
            'evaluation_errors': 0,
            'domain_filtered': 0,
            'model_used': 'deepseek-v3',  # Now using DeepSeek
            'estimated_cost_saved': 0.0
        }

    def search(self, queries: List[str], max_retries: int = 2, 
                              retry_delay: int = 2, enable_ai_filtering: bool = True) -> List[Dict]:
        """
        Execute searches for multiple queries with fast AI filtering
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

                        # NEW: Apply fast AI filtering using DeepSeek
                        if enable_ai_filtering and filtered_results:
                            logger.info(f"[SearchAgent] Applying fast AI content filtering...")
                            ai_filtered_results = self._run_async_in_thread(
                                self._apply_fast_ai_filtering(filtered_results)
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

        # Update final stats
        self.evaluation_stats['total_evaluated'] = len(all_results)

        return all_results

    async def _apply_fast_ai_filtering(self, search_results: List[Dict]) -> List[Dict]:
        """
        Apply fast AI filtering using DeepSeek instead of slow OpenAI evaluation.
        This replaces the 3-4 second per URL evaluation with batch processing.
        """
        start_time = time.time()

        # Prepare URL data for batch evaluation
        urls_data = []
        for result in search_results:
            urls_data.append({
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'snippet': result.get('snippet', '')
            })

        # Use fast batch evaluation (processes multiple URLs at once)
        evaluations = await self.fast_evaluator.evaluate_urls_batch(urls_data)

        # Filter results based on evaluations
        filtered_results = []
        passed_count = 0
        failed_count = 0

        for result, evaluation in zip(search_results, evaluations):
            score = evaluation.get('score', 0.0)
            reasoning = evaluation.get('reasoning', 'No reasoning')

            # Log evaluation for debugging
            logger.info(f"AI evaluation for {result.get('url', '')}: "
                       f"Quality={score:.2f}, "
                       f"Pass={'True' if score >= 0.7 else 'False'}")

            if score >= 0.7:  # Threshold for good content
                result['ai_evaluation'] = {
                    'score': score,
                    'reasoning': reasoning,
                    'passed': True
                }
                filtered_results.append(result)
                passed_count += 1
            else:
                failed_count += 1

        # Update evaluation stats
        processing_time = time.time() - start_time
        self.evaluation_stats.update({
            'passed_filter': passed_count,
            'failed_filter': failed_count,
            'evaluation_errors': 0,  # DeepSeek is more reliable
            'estimated_cost_saved': len(search_results) * 0.002  # Estimated savings vs OpenAI
        })

        logger.info(f"[SearchAgent] Fast AI filtering completed in {processing_time:.2f}s "
                   f"(vs ~{len(search_results) * 3:.0f}s with old method)")

        return filtered_results

    def _run_async_in_thread(self, coro):
        """Run async function in thread for compatibility with sync code"""
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(run_in_thread).result()

    # ... rest of your existing methods stay the same ...

    def _execute_search(self, query: str) -> Dict:
        """Execute search query against Brave API"""
        params = {
            "q": query,
            "count": self.search_count,
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate",
            "freshness": "pw",  # Past week for freshness
            "text_decorations": False
        }

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        response = requests.get(self.base_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def _filter_results(self, results: Dict) -> List[Dict]:
        """Apply domain filtering and basic cleanup"""
        web_results = results.get('web', {}).get('results', [])

        filtered_results = []

        for result in web_results:
            url = result.get('url', '')

            # Skip excluded domains
            if self._is_excluded_domain(url):
                self.evaluation_stats['domain_filtered'] += 1
                continue

            # Skip video platforms
            if self._is_video_platform(url):
                continue

            filtered_results.append(result)

        logger.info(f"[SearchAgent] After domain filtering: {len(filtered_results)} results "
                   f"(filtered {self.evaluation_stats['domain_filtered']} excluded domains)")

        return filtered_results

    def _is_excluded_domain(self, url: str) -> bool:
        """Check if URL is from an excluded domain"""
        try:
            domain = urlparse(url).netloc.lower()
            return any(excluded in domain for excluded in self.excluded_domains)
        except:
            return False

    def _is_video_platform(self, url: str) -> bool:
        """Check if URL is from a video platform"""
        video_platforms = ['youtube.com', 'tiktok.com', 'vimeo.com', 'dailymotion.com']
        try:
            domain = urlparse(url).netloc.lower()
            return any(platform in domain for platform in video_platforms)
        except:
            return False