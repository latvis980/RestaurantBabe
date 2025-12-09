# utils/api_usage_tracker.py
"""
Google Maps API Usage Tracker

Tracks API calls per key per month with PostgreSQL persistence (Railway).
Supports multiple API keys with automatic monthly reset detection.

Usage:
    from utils.api_usage_tracker import APIUsageTracker
    
    tracker = APIUsageTracker()
    
    # Before making an API call:
    if tracker.can_make_call("text_search", "GOOGLE_MAPS_API_KEY"):
        # make your API call
        tracker.record_call("text_search", "GOOGLE_MAPS_API_KEY")
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import psycopg2, fall back gracefully
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    logger.warning("⚠️ psycopg2 not installed - API tracking will use in-memory fallback")


@dataclass
class SKUConfig:
    """Configuration for a Google Maps API SKU"""
    name: str
    monthly_free_limit: int
    cost_per_1000: float  # USD after free tier


# SKU configurations based on Google's pricing (as of 2025)
# https://developers.google.com/maps/billing-and-pricing/pricing
SKU_CONFIGS = {
    # Text Search - depends on fields requested
    "text_search_essentials": SKUConfig("Text Search Essentials", 10000, 5.00),
    "text_search_pro": SKUConfig("Text Search Pro", 5000, 32.00),
    "text_search_enterprise": SKUConfig("Text Search Enterprise", 1000, 35.00),
    "text_search_enterprise_atmosphere": SKUConfig("Text Search Enterprise + Atmosphere", 1000, 40.00),
    
    # Place Details - depends on fields requested  
    "place_details_essentials": SKUConfig("Place Details Essentials", 10000, 5.00),
    "place_details_pro": SKUConfig("Place Details Pro", 5000, 17.00),
    "place_details_enterprise": SKUConfig("Place Details Enterprise", 1000, 20.00),
    "place_details_enterprise_atmosphere": SKUConfig("Place Details Enterprise + Atmosphere", 1000, 25.00),
    
    # Nearby Search
    "nearby_search_pro": SKUConfig("Nearby Search Pro", 5000, 32.00),
    "nearby_search_enterprise": SKUConfig("Nearby Search Enterprise", 1000, 35.00),
    "nearby_search_enterprise_atmosphere": SKUConfig("Nearby Search Enterprise + Atmosphere", 1000, 40.00),
    
    # Geocoding
    "geocoding": SKUConfig("Geocoding", 10000, 5.00),
    
    # Legacy Places API (if using old googlemaps library)
    "places_text_search_legacy": SKUConfig("Places Text Search (Legacy)", 5000, 32.00),
    "places_details_legacy": SKUConfig("Places Details (Legacy)", 5000, 17.00),
    "places_nearby_legacy": SKUConfig("Places Nearby (Legacy)", 5000, 32.00),
}

# Default SKU mapping for your current implementation
# Maps your operation names to the appropriate SKU
DEFAULT_SKU_MAPPING = {
    "text_search": "text_search_pro",  # Assumes Pro tier (basic fields + rating)
    "text_search_with_reviews": "text_search_enterprise_atmosphere",  # If fetching reviews
    "place_details": "place_details_pro",  # Basic details
    "place_details_with_reviews": "place_details_enterprise_atmosphere",  # With reviews
    "nearby_search": "nearby_search_pro",
    "geocoding": "geocoding",
}


class APIUsageTracker:
    """
    Tracks Google Maps API usage per key per month.
    
    Uses Railway PostgreSQL for persistence across deployments.
    Falls back to in-memory tracking if database unavailable.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the tracker.
        
        Args:
            database_url: PostgreSQL connection string. 
                         Defaults to RAILWAY_PG_URL or DATABASE_URL env var.
        """
        self.database_url = database_url or os.environ.get("RAILWAY_PG_URL") or os.environ.get("DATABASE_URL")
        self._in_memory_counts: Dict[str, Dict[str, int]] = {}  # Fallback
        self._db_available = False
        
        if self.database_url and HAS_PSYCOPG2:
            self._init_database()
        else:
            logger.warning("⚠️ No database URL configured - using in-memory tracking (won't persist)")
    
    def _init_database(self):
        """Initialize database connection and create tables if needed"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS google_maps_api_usage (
                            id SERIAL PRIMARY KEY,
                            api_key_name VARCHAR(100) NOT NULL,
                            sku_type VARCHAR(100) NOT NULL,
                            year_month VARCHAR(7) NOT NULL,  -- Format: YYYY-MM
                            call_count INTEGER DEFAULT 0,
                            last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            UNIQUE(api_key_name, sku_type, year_month)
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_api_usage_key_month 
                        ON google_maps_api_usage(api_key_name, year_month);
                    """)
                    conn.commit()
            self._db_available = True
            logger.info("✅ API usage tracker database initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize API usage database: {e}")
            self._db_available = False
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup"""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        finally:
            if conn:
                conn.close()
    
    def _get_current_month(self) -> str:
        """Get current month in YYYY-MM format (UTC)"""
        return datetime.now(timezone.utc).strftime("%Y-%m")
    
    def _resolve_sku(self, operation: str) -> str:
        """Resolve operation name to SKU type"""
        return DEFAULT_SKU_MAPPING.get(operation, operation)
    
    def record_call(self, operation: str, api_key_name: str) -> bool:
        """
        Record an API call.
        
        Args:
            operation: Type of operation (e.g., "text_search", "place_details")
            api_key_name: Identifier for the API key (e.g., "GOOGLE_MAPS_API_KEY", "KEY_2")
        
        Returns:
            True if recorded successfully
        """
        sku_type = self._resolve_sku(operation)
        year_month = self._get_current_month()
        
        if self._db_available:
            return self._record_call_db(api_key_name, sku_type, year_month)
        else:
            return self._record_call_memory(api_key_name, sku_type, year_month)
    
    def _record_call_db(self, api_key_name: str, sku_type: str, year_month: str) -> bool:
        """Record call in PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO google_maps_api_usage (api_key_name, sku_type, year_month, call_count)
                        VALUES (%s, %s, %s, 1)
                        ON CONFLICT (api_key_name, sku_type, year_month)
                        DO UPDATE SET 
                            call_count = google_maps_api_usage.call_count + 1,
                            last_updated = NOW()
                        RETURNING call_count;
                    """, (api_key_name, sku_type, year_month))
                    result = cur.fetchone()
                    conn.commit()
                    
                    new_count = result[0] if result else 0
                    logger.debug(f"📊 API call recorded: {api_key_name}/{sku_type} = {new_count} ({year_month})")
                    return True
        except Exception as e:
            logger.error(f"❌ Failed to record API call: {e}")
            return False
    
    def _record_call_memory(self, api_key_name: str, sku_type: str, year_month: str) -> bool:
        """Fallback: record call in memory"""
        key = f"{api_key_name}:{sku_type}:{year_month}"
        self._in_memory_counts[key] = self._in_memory_counts.get(key, 0) + 1
        logger.debug(f"📊 API call recorded (memory): {key} = {self._in_memory_counts[key]}")
        return True
    
    def get_usage(self, api_key_name: str, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current month's usage for an API key.
        
        Args:
            api_key_name: Identifier for the API key
            operation: Optional specific operation to check
        
        Returns:
            Dict with usage stats
        """
        year_month = self._get_current_month()
        sku_type = self._resolve_sku(operation) if operation else None
        
        if self._db_available:
            return self._get_usage_db(api_key_name, sku_type, year_month)
        else:
            return self._get_usage_memory(api_key_name, sku_type, year_month)
    
    def _get_usage_db(self, api_key_name: str, sku_type: Optional[str], year_month: str) -> Dict[str, Any]:
        """Get usage from PostgreSQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if sku_type:
                        cur.execute("""
                            SELECT sku_type, call_count, last_updated
                            FROM google_maps_api_usage
                            WHERE api_key_name = %s AND sku_type = %s AND year_month = %s
                        """, (api_key_name, sku_type, year_month))
                    else:
                        cur.execute("""
                            SELECT sku_type, call_count, last_updated
                            FROM google_maps_api_usage
                            WHERE api_key_name = %s AND year_month = %s
                            ORDER BY call_count DESC
                        """, (api_key_name, year_month))
                    
                    rows = cur.fetchall()
                    
                    usage = {
                        "api_key": api_key_name,
                        "month": year_month,
                        "operations": {},
                        "total_calls": 0
                    }
                    
                    for row in rows:
                        sku = row["sku_type"]
                        count = row["call_count"]
                        config = SKU_CONFIGS.get(sku)
                        
                        usage["operations"][sku] = {
                            "count": count,
                            "free_limit": config.monthly_free_limit if config else 1000,
                            "remaining": max(0, (config.monthly_free_limit if config else 1000) - count),
                            "over_limit": count > (config.monthly_free_limit if config else 1000)
                        }
                        usage["total_calls"] += count
                    
                    return usage
        except Exception as e:
            logger.error(f"❌ Failed to get usage: {e}")
            return {"error": str(e)}
    
    def _get_usage_memory(self, api_key_name: str, sku_type: Optional[str], year_month: str) -> Dict[str, Any]:
        """Fallback: get usage from memory"""
        usage = {
            "api_key": api_key_name,
            "month": year_month,
            "operations": {},
            "total_calls": 0,
            "note": "In-memory only - won't persist across restarts"
        }
        
        for key, count in self._in_memory_counts.items():
            parts = key.split(":")
            if len(parts) == 3 and parts[0] == api_key_name and parts[2] == year_month:
                sku = parts[1]
                if sku_type and sku != sku_type:
                    continue
                    
                config = SKU_CONFIGS.get(sku)
                usage["operations"][sku] = {
                    "count": count,
                    "free_limit": config.monthly_free_limit if config else 1000,
                    "remaining": max(0, (config.monthly_free_limit if config else 1000) - count)
                }
                usage["total_calls"] += count
        
        return usage
    
    def can_make_call(self, operation: str, api_key_name: str, warn_threshold: float = 0.9) -> bool:
        """
        Check if we can make an API call without exceeding free tier.
        
        Args:
            operation: Type of operation
            api_key_name: API key identifier
            warn_threshold: Warn when usage exceeds this percentage (default 90%)
        
        Returns:
            True if within free tier limits
        """
        sku_type = self._resolve_sku(operation)
        config = SKU_CONFIGS.get(sku_type)
        
        if not config:
            logger.warning(f"⚠️ Unknown SKU type: {sku_type}, allowing call")
            return True
        
        usage = self.get_usage(api_key_name, operation)
        
        if "error" in usage:
            # If we can't check, allow but log
            logger.warning(f"⚠️ Can't verify usage, allowing call: {usage.get('error')}")
            return True
        
        ops = usage.get("operations", {})
        if sku_type not in ops:
            # No usage yet
            return True
        
        current = ops[sku_type]["count"]
        limit = config.monthly_free_limit
        
        # Check if approaching limit
        if current >= limit * warn_threshold:
            remaining = limit - current
            logger.warning(f"⚠️ {api_key_name}/{sku_type}: {current}/{limit} calls ({remaining} remaining)")
        
        if current >= limit:
            logger.error(f"🚫 {api_key_name}/{sku_type}: FREE TIER EXCEEDED ({current}/{limit})")
            return False
        
        return True
    
    def get_all_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for all API keys this month"""
        year_month = self._get_current_month()
        
        if not self._db_available:
            return {
                "month": year_month,
                "keys": {},
                "note": "In-memory tracking only"
            }
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT api_key_name, sku_type, call_count
                        FROM google_maps_api_usage
                        WHERE year_month = %s
                        ORDER BY api_key_name, call_count DESC
                    """, (year_month,))
                    
                    rows = cur.fetchall()
                    
                    summary = {
                        "month": year_month,
                        "keys": {},
                        "total_all_keys": 0
                    }
                    
                    for row in rows:
                        key = row["api_key_name"]
                        sku = row["sku_type"]
                        count = row["call_count"]
                        config = SKU_CONFIGS.get(sku)
                        
                        if key not in summary["keys"]:
                            summary["keys"][key] = {"operations": {}, "total": 0}
                        
                        summary["keys"][key]["operations"][sku] = {
                            "count": count,
                            "limit": config.monthly_free_limit if config else 1000,
                            "over_limit": count > (config.monthly_free_limit if config else 1000)
                        }
                        summary["keys"][key]["total"] += count
                        summary["total_all_keys"] += count
                    
                    return summary
        except Exception as e:
            logger.error(f"❌ Failed to get summary: {e}")
            return {"error": str(e)}
    
    def log_usage_report(self):
        """Log a formatted usage report"""
        summary = self.get_all_usage_summary()
        
        logger.info("=" * 60)
        logger.info(f"📊 GOOGLE MAPS API USAGE REPORT - {summary.get('month', 'Unknown')}")
        logger.info("=" * 60)
        
        for key_name, key_data in summary.get("keys", {}).items():
            logger.info(f"\n🔑 {key_name}:")
            for sku, stats in key_data.get("operations", {}).items():
                status = "🚫 OVER LIMIT" if stats.get("over_limit") else "✅"
                logger.info(f"   {sku}: {stats['count']}/{stats['limit']} {status}")
            logger.info(f"   Total: {key_data.get('total', 0)}")
        
        logger.info("=" * 60)


# Singleton instance for easy import
_tracker_instance: Optional[APIUsageTracker] = None

def get_tracker() -> APIUsageTracker:
    """Get the singleton tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = APIUsageTracker()
    return _tracker_instance


# Convenience functions
def record_api_call(operation: str, api_key_name: str = "primary") -> bool:
    """Record an API call (convenience function)"""
    return get_tracker().record_call(operation, api_key_name)

def check_api_limit(operation: str, api_key_name: str = "primary") -> bool:
    """Check if API call is within limits (convenience function)"""
    return get_tracker().can_make_call(operation, api_key_name)

def get_api_usage(api_key_name: str = "primary") -> Dict[str, Any]:
    """Get API usage for a key (convenience function)"""
    return get_tracker().get_usage(api_key_name)
