# agents/sync_scraper.py
"""
Synchronous wrapper for the WebScraper that works seamlessly with LangChain tracing.

This module provides a synchronous interface to the asynchronous WebScraper,
which makes it compatible with LangChain's RunnableLambda and ensures all
operations appear as part of the same trace in LangSmith.

Usage:
    from agents.sync_scraper import scrape_search_results

    scrape_step = RunnableLambda(
        lambda x: {
            **x,
            "enriched_results": scrape_search_results(
                x.get("search_results", []), 
                scraper
            )
        },
        name="scrape"
    )
"""

import asyncio
import concurrent.futures
import logging
from typing import List, Dict, Any
from langchain_core.tracers.context import tracing_v2_enabled

logger = logging.getLogger("restaurant-recommender.sync_scraper")

def scrape_search_results(search_results: List[Dict[str, Any]], scraper) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for scraper.filter_and_scrape_results

    Args:
        search_results: The list of search results to scrape
        scraper: The WebScraper instance

    Returns:
        The enriched results with scraped content
    """
    logger.info(f"Starting synchronous scrape for {len(search_results)} results")

    try:
        # Use the current tracing context
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            loop = asyncio.new_event_loop()

            try:
                asyncio.set_event_loop(loop)
                # This runs in the current thread with proper tracing context
                enriched_results = loop.run_until_complete(
                    scraper.filter_and_scrape_results(search_results)
                )
                logger.info(f"Scrape completed with {len(enriched_results)} enriched results")
                return enriched_results
            finally:
                loop.close()

    except Exception as e:
        logger.error(f"Error in synchronous scrape: {e}")
        # Return an empty list as fallback
        return []

def scrape_single_url(url: str, scraper) -> Dict[str, Any]:
    """
    Synchronously fetch and process a single URL

    Args:
        url: The URL to fetch
        scraper: The WebScraper instance

    Returns:
        The fetched and processed result
    """
    try:
        # Use the current tracing context
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            loop = asyncio.new_event_loop()

            try:
                asyncio.set_event_loop(loop)
                # This runs in the current thread with proper tracing context
                if hasattr(scraper, "fetch_url"):
                    result = loop.run_until_complete(scraper.fetch_url(url))
                    logger.info(f"URL fetch completed for {url}")
                    return result
                else:
                    logger.error("Scraper doesn't have fetch_url method")
                    return {"error": "Method not available"}
            finally:
                loop.close()

    except Exception as e:
        logger.error(f"Error in synchronous URL fetch: {e}")
        # Return an error dict as fallback
        return {"error": str(e)}