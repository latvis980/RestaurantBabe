# agents/fast_search_evaluation_agent.py
"""
Ultra-fast search URL evaluation using DeepSeek.
Replaces the 3-4 second per URL evaluation with 1-2 second processing.

This handles the search agent's AI filtering that was causing delays.
"""

import logging
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from utils.unified_model_manager import get_unified_model_manager

logger = logging.getLogger(__name__)

class FastSearchEvaluationAgent:
    """
    Ultra-fast URL evaluation for restaurant search results using DeepSeek.

    Reduces URL evaluation time from 3-4 seconds to 1-2 seconds per URL,
    with batch processing capabilities for even faster results.
    """

    def __init__(self, config):
        self.config = config
        self.model_manager = get_unified_model_manager(config)

        # Domain cache for known good/bad domains
        self._domain_cache = {}

        # Statistics
        self.stats = {
            "total_evaluated": 0,
            "cache_hits": 0,
            "batch_processed": 0,
            "average_eval_time": 0.0
        }

        # Fast evaluation prompt optimized for DeepSeek speed
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a fast URL evaluator for restaurant content. Rate if URLs contain curated restaurant lists/guides.

GOOD (score 0.7+): Restaurant guides, curated lists, professional reviews
BAD (score 0.3-): Single restaurant sites, booking platforms, forums, videos

Rate each URL quickly and accurately.
            """),
            ("human", """
Evaluate these URLs for restaurant guide content:

{urls_data}

Return JSON with quick evaluations:
{{
    "evaluations": [
        {{"url": "...", "score": 0.8, "reasoning": "Brief reason"}}
    ]
}}

Focus on speed - use URL patterns and titles to make quick decisions.
            """)
        ])

    async def evaluate_urls_batch(self, urls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple URLs in batches for maximum speed.

        Args:
            urls_data: List of dicts with 'url', 'title', 'snippet'

        Returns:
            List of evaluation results with scores
        """
        if not urls_data:
            return []

        # Check cache first and separate cached vs uncached
        cached_results = []
        uncached_urls = []

        for url_data in urls_data:
            url = url_data.get('url', '')
            domain = self._extract_domain(url)

            if domain in self._domain_cache:
                # Use cached domain evaluation
                cached_score = self._domain_cache[domain]
                cached_results.append({
                    'url': url,
                    'score': cached_score,
                    'reasoning': f'Cached domain score for {domain}',
                    'evaluation_time': 0.0
                })
                self.stats["cache_hits"] += 1
            else:
                uncached_urls.append(url_data)

        # Process uncached URLs
        if uncached_urls:
            # Process in batches for speed
            batch_size = 5  # DeepSeek can handle multiple URLs efficiently
            uncached_results = []

            for i in range(0, len(uncached_urls), batch_size):
                batch = uncached_urls[i:i + batch_size]
                batch_results = await self._evaluate_batch_with_deepseek(batch)
                uncached_results.extend(batch_results)

                # Update domain cache with results
                for result in batch_results:
                    domain = self._extract_domain(result['url'])
                    self._domain_cache[domain] = result['score']
        else:
            uncached_results = []

        # Combine cached and uncached results
        all_results = cached_results + uncached_results

        # Update statistics
        self.stats["total_evaluated"] += len(urls_data)
        self.stats["batch_processed"] += 1

        return all_results

    async def _evaluate_batch_with_deepseek(self, urls_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of URLs using DeepSeek for speed"""
        start_time = time.time()

        try:
            # Format URLs data for the prompt
            urls_text = "\n".join([
                f"URL: {item.get('url', '')}\nTitle: {item.get('title', 'No title')}\nSnippet: {item.get('snippet', 'No snippet')[:200]}...\n"
                for item in urls_batch
            ])

            # Make fast DeepSeek API call
            formatted_prompt = self.evaluation_prompt.format(urls_data=urls_text)

            response = await self.model_manager.rate_limited_call(
                'search_evaluation',  # Routes to DeepSeek automatically
                formatted_prompt
            )

            # Parse response
            response_content = response.content.strip()
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0].strip()

            result_data = json.loads(response_content)
            evaluations = result_data.get("evaluations", [])

            # Add timing information
            eval_time = time.time() - start_time
            for evaluation in evaluations:
                evaluation["evaluation_time"] = eval_time / len(evaluations)

            logger.debug(f"⚡ Evaluated {len(urls_batch)} URLs in {eval_time:.2f}s using DeepSeek")
            return evaluations

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse DeepSeek evaluation response: {e}")
            return self._fallback_evaluation(urls_batch, start_time)
        except Exception as e:
            logger.error(f"DeepSeek batch evaluation failed: {e}")
            return self._fallback_evaluation(urls_batch, start_time)

    def _fallback_evaluation(self, urls_batch: List[Dict[str, Any]], start_time: float) -> List[Dict[str, Any]]:
        """Fallback heuristic evaluation when AI fails"""
        results = []
        eval_time = time.time() - start_time

        for url_data in urls_batch:
            url = url_data.get('url', '')
            title = url_data.get('title', '').lower()

            # Simple heuristic scoring based on URL patterns
            score = self._heuristic_score(url, title)

            results.append({
                'url': url,
                'score': score,
                'reasoning': 'Heuristic evaluation (AI fallback)',
                'evaluation_time': eval_time / len(urls_batch)
            })

        return results

    def _heuristic_score(self, url: str, title: str) -> float:
        """Fast heuristic scoring for URLs"""
        url_lower = url.lower()
        title_lower = title.lower()

        # Bad patterns (single restaurants, booking sites, etc.)
        bad_patterns = [
            'tripadvisor.com', 'yelp.com', 'google.com/maps', 
            'opentable.com', 'resy.com', 'bookatable.com',
            'facebook.com', 'instagram.com', 'twitter.com',
            'youtube.com', 'tiktok.com', 'reddit.com'
        ]

        # Good patterns (restaurant guides, professional sites)
        good_patterns = [
            'timeout.com', 'eater.com', 'michelin.', 'cntraveler.com',
            'foodandwine.com', 'bonappetit.com', 'saveur.com',
            'theinfatuation.com', 'zagat.com', 'guide.', 'best-restaurants'
        ]

        # Check for bad patterns
        if any(pattern in url_lower for pattern in bad_patterns):
            return 0.2

        # Check for good patterns
        if any(pattern in url_lower for pattern in good_patterns):
            return 0.9

        # Check title for restaurant list indicators
        list_indicators = ['best', 'top', 'guide', 'restaurants', 'where to eat', 'dining']
        if any(indicator in title_lower for indicator in list_indicators):
            return 0.7

        # Default score for unknown patterns
        return 0.5

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for caching"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url

    async def evaluate_single_url(self, url: str, title: str = "", snippet: str = "") -> Dict[str, Any]:
        """
        Evaluate a single URL (for backward compatibility).

        Args:
            url: URL to evaluate
            title: Page title
            snippet: Page snippet

        Returns:
            Evaluation result
        """
        url_data = {
            'url': url,
            'title': title,
            'snippet': snippet
        }

        results = await self.evaluate_urls_batch([url_data])
        return results[0] if results else {
            'url': url,
            'score': 0.5,
            'reasoning': 'Evaluation failed',
            'evaluation_time': 0.0
        }

    def get_stats(self) -> Dict:
        """Get evaluation statistics"""
        return {
            **self.stats,
            "domain_cache_size": len(self._domain_cache)
        }