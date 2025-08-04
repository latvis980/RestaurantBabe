# utils/database_domain_intelligence.py - CLEAN VERSION: Domain intelligence operations  
"""
Domain Intelligence Database Module - Compatible with Updated Smart Scraper

Purpose: Handle scraper learning and optimization through domain intelligence tracking.
Strategy: Save intelligence AFTER successful scrapes (post-scrape learning).

COMPATIBILITY NOTES:
- Works with new ScrapingStrategy enum: SPECIALIZED, SIMPLE_HTTP, ENHANCED_HTTP, FIRECRAWL
- Integrates with _save_domain_intelligence() method in smart scraper
- Supports _get_cached_strategy() pattern used by smart scraper
- Compatible with scraping_method field used in results

Key Flow:
1. Smart scraper checks get_cached_strategy() for known domains
2. If no cache, uses AI to classify strategy
3. After successful scrape, calls save_scrape_result() to record learning
4. Domain intelligence builds up over time to optimize future scrapes
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class DomainIntelligenceManager:
    """
    Manages domain intelligence for smart scraping optimization.
    Focus: Learning from scraping results to improve future performance.
    Compatible with updated smart scraper architecture.
    """

    def __init__(self, config):
        """Initialize domain intelligence manager"""
        self.config = config

        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("✅ Domain Intelligence database initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize domain intelligence database: {e}")
            raise

        # In-memory cache for frequently accessed domains
        self.domain_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL

    # ============ POST-SCRAPE LEARNING METHODS ============

    def save_scrape_result(self, url: str, scraping_method: str, scraping_success: bool, **kwargs) -> bool:
        """
        Record scraping results for domain learning (called AFTER scraping)
        Compatible with smart scraper result format

        Args:
            url: The URL that was scraped
            scraping_method: The method used ('specialized', 'simple_http', 'enhanced_http', 'firecrawl')
            scraping_success: Whether the scraping was successful
            **kwargs: Additional data like restaurants_found, content_length, etc.

        Returns:
            bool: Success status
        """
        try:
            domain = self._extract_domain(url)
            current_time = datetime.now(timezone.utc).isoformat()

            # Get existing domain data
            existing_data = self.get_domain_intelligence(domain)

            if existing_data:
                # Update existing record
                total_attempts = existing_data.get('total_attempts', 0) + 1
                success_count = existing_data.get('success_count', 0) + (1 if scraping_success else 0)

                # Calculate new confidence and cost
                confidence = success_count / total_attempts if total_attempts > 0 else 0
                cost_per_scrape = self._get_strategy_cost(scraping_method)

                update_data = {
                    'strategy': scraping_method,
                    'total_attempts': total_attempts,
                    'success_count': success_count,
                    'confidence': confidence,
                    'cost_per_scrape': cost_per_scrape,
                    'last_scraped': current_time,
                    'restaurants_found': kwargs.get('restaurants_found', existing_data.get('restaurants_found', 0)),
                    'avg_content_length': kwargs.get('content_length', existing_data.get('avg_content_length', 0))
                }

                result = self.supabase.table('domain_intelligence')\
                    .update(update_data)\
                    .eq('domain', domain)\
                    .execute()

                # Invalidate cache
                self._invalidate_domain_cache(domain)

                logger.debug(f"🔄 Updated domain intelligence for {domain}: {scraping_method} (confidence: {confidence:.2f})")

            else:
                # Create new record
                confidence = 1.0 if scraping_success else 0.0
                cost_per_scrape = self._get_strategy_cost(scraping_method)

                insert_data = {
                    'domain': domain,
                    'strategy': scraping_method,
                    'total_attempts': 1,
                    'success_count': 1 if scraping_success else 0,
                    'confidence': confidence,
                    'cost_per_scrape': cost_per_scrape,
                    'first_seen': current_time,
                    'last_scraped': current_time,
                    'restaurants_found': kwargs.get('restaurants_found', 0),
                    'avg_content_length': kwargs.get('content_length', 0)
                }

                result = self.supabase.table('domain_intelligence')\
                    .insert(insert_data)\
                    .execute()

                logger.info(f"🆕 Created domain intelligence for {domain}: {scraping_method} (confidence: {confidence:.2f})")

            return True

        except Exception as e:
            logger.error(f"❌ Error saving scrape result for {url}: {e}")
            return False

    def batch_save_scrape_results(self, scrape_results: List[Dict[str, Any]]) -> int:
        """
        Batch save multiple scrape results for efficiency

        Args:
            scrape_results: List of scrape result dictionaries

        Returns:
            int: Number of successfully saved results
        """
        saved_count = 0

        for result in scrape_results:
            url = result.get('url', '')
            scraping_method = result.get('scraping_method', 'enhanced_http')
            scraping_success = result.get('scraping_success', False)

            if self.save_scrape_result(url, scraping_method, scraping_success, **result):
                saved_count += 1

        logger.info(f"💾 Batch saved {saved_count}/{len(scrape_results)} domain intelligence records")
        return saved_count

    # ============ PRE-SCRAPE LOOKUP METHODS ============

    def get_cached_strategy(self, url: str) -> Optional[str]:
        """
        Get cached scraping strategy for a URL (used BEFORE scraping)
        Compatible with smart scraper _get_cached_strategy() pattern

        Args:
            url: The URL to get strategy for

        Returns:
            str or None: Strategy name if cached and reliable, None otherwise
        """
        try:
            domain = self._extract_domain(url)

            # Check cache first
            cached_data = self._get_from_domain_cache(domain)
            if cached_data:
                confidence = cached_data.get('confidence', 0)
                total_attempts = cached_data.get('total_attempts', 0)

                # Only return strategy if we have enough data and good confidence
                # Uses 0.6 threshold to match smart scraper logic
                if total_attempts >= 2 and confidence > 0.6:
                    return cached_data.get('strategy')

                return None

            # Query database
            result = self.supabase.table('domain_intelligence')\
                .select('*')\
                .eq('domain', domain)\
                .execute()

            if result.data:
                data = result.data[0]
                confidence = data.get('confidence', 0)
                total_attempts = data.get('total_attempts', 0)

                # Update cache
                self._update_domain_cache(domain, data)

                # Only return strategy if reliable enough (matches smart scraper logic)
                if total_attempts >= 2 and confidence > 0.6:
                    strategy = data.get('strategy')
                    logger.debug(f"🧠 Cached strategy for {domain}: {strategy} (confidence: {confidence:.2f})")
                    return strategy

            return None

        except Exception as e:
            logger.error(f"❌ Error getting cached strategy for {url}: {e}")
            return None

    def get_domain_intelligence(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get complete domain intelligence data for a domain
        Compatible with database.py interface

        Args:
            domain: Domain name to query

        Returns:
            Dict containing domain intelligence data or None
        """
        try:
            # Check cache first
            cached_data = self._get_from_domain_cache(domain)
            if cached_data:
                return cached_data

            # Query database
            result = self.supabase.table('domain_intelligence')\
                .select('*')\
                .eq('domain', domain)\
                .execute()

            if result.data:
                data = result.data[0]

                # Update cache
                self._update_domain_cache(domain, data)

                logger.debug(f"📊 Retrieved domain intelligence for {domain}: {data.get('strategy')}")
                return data

            return None

        except Exception as e:
            logger.error(f"❌ Error getting domain intelligence for {domain}: {e}")
            return None

    def get_trusted_domains(self, min_confidence: float = 0.7) -> List[str]:
        """
        Get list of trusted domains with high confidence scores

        Args:
            min_confidence: Minimum confidence threshold (default 0.7)

        Returns:
            List of trusted domain names
        """
        try:
            result = self.supabase.table('domain_intelligence')\
                .select('domain')\
                .gte('confidence', min_confidence)\
                .gte('total_attempts', 2)\
                .order('confidence', desc=True)\
                .execute()

            trusted_domains = [row['domain'] for row in result.data]
            logger.info(f"🎯 Found {len(trusted_domains)} trusted domains (min confidence: {min_confidence})")

            return trusted_domains

        except Exception as e:
            logger.error(f"❌ Error getting trusted domains: {e}")
            return []

    def load_all_domain_intelligence(self) -> List[Dict[str, Any]]:
        """
        Load all domain intelligence records for analysis

        Returns:
            List of all domain intelligence records
        """
        try:
            result = self.supabase.table('domain_intelligence')\
                .select('*')\
                .order('confidence', desc=True)\
                .limit(1000)\
                .execute()

            logger.info(f"📚 Loaded {len(result.data)} domain intelligence records")
            return result.data

        except Exception as e:
            logger.error(f"❌ Error loading domain intelligence: {e}")
            return []

    # ============ STATISTICS AND ANALYTICS ============

    def get_domain_intelligence_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive domain intelligence statistics
        Compatible with smart scraper cost analysis

        Returns:
            Dictionary with detailed statistics
        """
        try:
            all_domains = self.load_all_domain_intelligence()

            if not all_domains:
                return {
                    'total_domains': 0,
                    'strategy_breakdown': {},
                    'average_confidence': 0,
                    'high_confidence_domains': 0,
                    'cost_analysis': {}
                }

            # Calculate statistics
            strategy_counts = {}
            confidence_sum = 0
            high_confidence_count = 0
            total_cost_estimate = 0

            for domain in all_domains:
                strategy = domain.get('strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                confidence = domain.get('confidence', 0)
                confidence_sum += confidence

                if confidence >= 0.8:
                    high_confidence_count += 1

                # Estimate cost savings
                cost_per_scrape = domain.get('cost_per_scrape', 0.5)
                total_attempts = domain.get('total_attempts', 1)
                total_cost_estimate += cost_per_scrape * total_attempts

            avg_confidence = confidence_sum / len(all_domains) if all_domains else 0

            # Calculate potential cost savings vs all-Firecrawl
            all_firecrawl_cost = len(all_domains) * 10.0  # If all domains used Firecrawl
            cost_saved = all_firecrawl_cost - total_cost_estimate

            return {
                'total_domains': len(all_domains),
                'strategy_breakdown': strategy_counts,
                'average_confidence': avg_confidence,
                'high_confidence_domains': high_confidence_count,
                'cost_analysis': {
                    'actual_cost_estimate': total_cost_estimate,
                    'all_firecrawl_cost': all_firecrawl_cost,
                    'cost_saved': cost_saved,
                    'efficiency_percentage': (cost_saved / all_firecrawl_cost) * 100 if all_firecrawl_cost > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"❌ Error calculating domain intelligence stats: {e}")
            return {}

    # ============ CACHE MANAGEMENT ============

    def _get_from_domain_cache(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain data from cache if not expired"""
        if domain in self.domain_cache:
            cached_data, timestamp = self.domain_cache[domain]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                # Remove expired entry
                del self.domain_cache[domain]
        return None

    def _update_domain_cache(self, domain: str, data: Dict[str, Any]):
        """Update domain cache with new data"""
        self.domain_cache[domain] = (data.copy(), time.time())

        # Simple cache size management
        if len(self.domain_cache) > 1000:
            # Remove oldest 100 entries
            oldest_domains = sorted(
                self.domain_cache.keys(),
                key=lambda d: self.domain_cache[d][1]
            )[:100]

            for old_domain in oldest_domains:
                del self.domain_cache[old_domain]

    def _invalidate_domain_cache(self, domain: str):
        """Invalidate cache entry for a domain"""
        if domain in self.domain_cache:
            del self.domain_cache[domain]

    def clear_domain_cache(self):
        """Clear the domain intelligence cache"""
        self.domain_cache.clear()
        logger.info("🧹 Cleared domain intelligence cache")

    # ============ UTILITY METHODS ============

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return ""

    def _get_strategy_cost(self, strategy: str) -> float:
        """Get cost estimate for strategy (compatible with smart scraper cost mapping)"""
        cost_map = {
            "specialized": 0.0,      # Free
            "simple_http": 0.1,      # Low cost
            "enhanced_http": 0.5,    # Medium cost
            "firecrawl": 10.0        # High cost
        }
        return cost_map.get(strategy, 0.5)

    def is_domain_trusted(self, domain: str, min_confidence: float = 0.7) -> bool:
        """Check if a domain is trusted based on confidence score"""
        intelligence = self.get_domain_intelligence(domain)
        if intelligence:
            return intelligence.get('confidence', 0) >= min_confidence
        return False

    def log_domain_intelligence_summary(self):
        """Log a summary of current domain intelligence"""
        try:
            stats = self.get_domain_intelligence_stats()

            logger.info("=" * 50)
            logger.info("🧠 DOMAIN INTELLIGENCE SUMMARY")
            logger.info("=" * 50)
            logger.info(f"📊 Total Domains: {stats['total_domains']}")
            logger.info(f"🎯 High Confidence: {stats['high_confidence_domains']}")
            logger.info(f"📈 Average Confidence: {stats['average_confidence']:.2f}")

            logger.info("\n📋 Strategy Breakdown:")
            strategy_emojis = {"specialized": "🆓", "simple_http": "🟢", "enhanced_http": "🟡", "firecrawl": "🔴"}
            for strategy, count in stats['strategy_breakdown'].items():
                emoji = strategy_emojis.get(strategy, "📌")
                logger.info(f"   {emoji} {strategy.upper()}: {count} domains")

            cost_analysis = stats['cost_analysis']
            logger.info(f"\n💰 Cost Analysis:")
            logger.info(f"   Estimated Actual Cost: {cost_analysis.get('actual_cost_estimate', 0):.1f} credits")
            logger.info(f"   All-Firecrawl Cost: {cost_analysis.get('all_firecrawl_cost', 0):.1f} credits")
            logger.info(f"   Cost Saved: {cost_analysis.get('cost_saved', 0):.1f} credits")
            logger.info(f"   Efficiency: {cost_analysis.get('efficiency_percentage', 0):.1f}%")

            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"❌ Error logging domain intelligence summary: {e}")

# ============ GLOBAL DOMAIN INTELLIGENCE INSTANCE ============

_domain_intelligence = None

def initialize_domain_intelligence(config):
    """Initialize the global domain intelligence instance"""
    global _domain_intelligence
    _domain_intelligence = DomainIntelligenceManager(config)

def get_domain_intelligence_manager() -> DomainIntelligenceManager:
    """Get the global domain intelligence instance"""
    if _domain_intelligence is None:
        raise RuntimeError("Domain intelligence not initialized. Call initialize_domain_intelligence() first.")
    return _domain_intelligence

# ============ CONVENIENCE FUNCTIONS FOR INTEGRATION ============
# These integrate with the existing database.py interface

def save_domain_intelligence_from_scraper(domain: str, intelligence_data: Dict[str, Any]) -> bool:
    """
    Save domain intelligence data from smart scraper
    Compatible with existing database.py interface
    """
    try:
        manager = get_domain_intelligence_manager()

        # Extract data from intelligence_data dict
        strategy = intelligence_data.get('strategy', 'enhanced_http')
        success_count = intelligence_data.get('success_count', 1)
        total_attempts = intelligence_data.get('total_attempts', 1)

        # Calculate if this represents a success
        success = success_count > 0

        # Use the new save method (we don't have URL, so construct one)
        fake_url = f"https://{domain}/"

        return manager.save_scrape_result(fake_url, strategy, success)

    except Exception as e:
        logger.error(f"❌ Error saving domain intelligence from scraper: {e}")
        return False

def get_cached_strategy_for_url(url: str) -> Optional[str]:
    """Get cached strategy for a URL (used by smart scraper)"""
    try:
        manager = get_domain_intelligence_manager()
        return manager.get_cached_strategy(url)
    except Exception as e:
        logger.error(f"❌ Error getting cached strategy: {e}")
        return None

def batch_update_from_scraper_results(scrape_results: List[Dict[str, Any]]) -> int:
    """Batch update domain intelligence from scraper results"""
    try:
        manager = get_domain_intelligence_manager()
        return manager.batch_save_scrape_results(scrape_results)
    except Exception as e:
        logger.error(f"❌ Error batch updating from scraper results: {e}")
        return 0