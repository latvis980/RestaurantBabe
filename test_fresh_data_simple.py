#!/usr/bin/env python3

"""
Simple test script to verify Fresh Data optimization methods work correctly.
Tests the individual methods without complex database setup.
"""

import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from unittest.mock import MagicMock, patch

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_follow_up_agent():
    """Create a follow-up agent with mocked external dependencies"""
    with patch('googlemaps.Client') as mock_client:
        # Mock the Google Maps client to avoid API key issues
        mock_client.return_value = MagicMock()
        
        from agents.follow_up_search_agent import FollowUpSearchAgent
        
        # Create a minimal config object
        mock_config = MagicMock()
        mock_config.GOOGLE_MAPS_API_KEY = "test_key"
        mock_config.GOOGLE_MAPS_API_KEY2 = "test_key_2"
        mock_config.MIN_ACCEPTABLE_RATING = 4.1
        
        # Create the agent with mocked Google Maps client
        agent = FollowUpSearchAgent(mock_config)
        
        return agent

def test_freshness_check():
    """Test the _is_restaurant_data_fresh method"""
    logger.info("🧪 Testing freshness check method")
    
    agent = create_mock_follow_up_agent()
    now = datetime.now(timezone.utc)
    
    # Test cases
    test_cases = [
        {
            "name": "Fresh restaurant (30 days old)",
            "last_updated": (now - timedelta(days=30)).isoformat(),
            "expected": True
        },
        {
            "name": "Stale restaurant (120 days old)", 
            "last_updated": (now - timedelta(days=120)).isoformat(),
            "expected": False
        },
        {
            "name": "No timestamp",
            "last_updated": None,
            "expected": False
        },
        {
            "name": "Empty timestamp",
            "last_updated": "",
            "expected": False
        },
        {
            "name": "Exactly 90 days old (boundary)",
            "last_updated": (now - timedelta(days=90)).isoformat(),
            "expected": False  # Should be stale at exactly 90 days
        },
        {
            "name": "89 days old (just fresh)",
            "last_updated": (now - timedelta(days=89)).isoformat(), 
            "expected": True
        }
    ]
    
    results = []
    for test_case in test_cases:
        restaurant_data = {
            "name": "Test Restaurant",
            "last_updated": test_case["last_updated"]
        }
        
        try:
            result = agent._is_restaurant_data_fresh(restaurant_data)
            success = result == test_case["expected"]
            
            if success:
                logger.info(f"✅ {test_case['name']}: {result} (expected {test_case['expected']})")
            else:
                logger.error(f"❌ {test_case['name']}: {result} (expected {test_case['expected']})")
                
            results.append(success)
            
        except Exception as e:
            logger.error(f"❌ {test_case['name']}: Error - {e}")
            results.append(False)
    
    return all(results)

def test_cached_maps_info():
    """Test the _build_cached_maps_info method"""
    logger.info("🧪 Testing cached maps info builder")
    
    agent = create_mock_follow_up_agent()
    
    # Test restaurant data with all fields
    restaurant_data = {
        "name": "Test Restaurant",
        "address": "123 Test Street, Test City",
        "rating": 4.5,
        "user_ratings_total": 100,
        "business_status": "OPERATIONAL",
        "place_id": "ChIJ_test_place_id",
        "google_maps_url": "https://maps.google.com/test",
        "latitude": 48.8566,
        "longitude": 2.3522,
        "address_components": [{"types": ["country"], "long_name": "France"}]
    }
    
    try:
        cached_info = agent._build_cached_maps_info(restaurant_data)
        
        # Verify all expected fields are present
        expected_fields = [
            "formatted_address", "rating", "user_ratings_total", 
            "business_status", "place_id", "url", "geometry", "address_components"
        ]
        
        success = True
        for field in expected_fields:
            if field not in cached_info:
                logger.error(f"❌ Missing field: {field}")
                success = False
            else:
                logger.debug(f"✅ Field present: {field} = {cached_info[field]}")
        
        # Check geometry structure
        if cached_info.get("geometry", {}).get("location", {}).get("lat") == 48.8566:
            logger.info("✅ Geometry coordinates correctly built")
        else:
            logger.error("❌ Geometry coordinates not built correctly")
            success = False
            
        if success:
            logger.info("✅ Cached maps info builder working correctly")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error testing cached maps info: {e}")
        return False

def test_integration_logic():
    """Test the integration of freshness check with conditional logic"""
    logger.info("🧪 Testing integration logic")
    
    agent = create_mock_follow_up_agent()
    now = datetime.now(timezone.utc)
    
    # Fresh restaurant test
    fresh_restaurant = {
        "name": "Fresh Restaurant",
        "last_updated": (now - timedelta(days=30)).isoformat(),
        "address": "123 Fresh Street",
        "rating": 4.5,
        "place_id": "ChIJ_fresh_test"
    }
    
    # Stale restaurant test  
    stale_restaurant = {
        "name": "Stale Restaurant",
        "last_updated": (now - timedelta(days=120)).isoformat(),
        "address": "456 Stale Avenue", 
        "rating": 4.2,
        "place_id": "ChIJ_stale_test"
    }
    
    results = []
    
    # Test fresh restaurant logic
    try:
        is_fresh = agent._is_restaurant_data_fresh(fresh_restaurant)
        if is_fresh:
            cached_info = agent._build_cached_maps_info(fresh_restaurant)
            if cached_info and cached_info.get("formatted_address") == "123 Fresh Street":
                logger.info("✅ Fresh restaurant: Uses cached data correctly")
                results.append(True)
            else:
                logger.error("❌ Fresh restaurant: Cached data not built correctly")
                results.append(False)
        else:
            logger.error("❌ Fresh restaurant not detected as fresh")
            results.append(False)
    except Exception as e:
        logger.error(f"❌ Error testing fresh restaurant logic: {e}")
        results.append(False)
    
    # Test stale restaurant logic
    try:
        is_fresh = agent._is_restaurant_data_fresh(stale_restaurant)
        if not is_fresh:
            logger.info("✅ Stale restaurant: Correctly detected as needing API call")
            results.append(True)
        else:
            logger.error("❌ Stale restaurant incorrectly detected as fresh")
            results.append(False)
    except Exception as e:
        logger.error(f"❌ Error testing stale restaurant logic: {e}")
        results.append(False)
    
    return all(results)

def main():
    """Run all fresh data optimization tests"""
    logger.info("🚀 Starting Fresh Data Optimization Unit Tests")
    
    test_results = []
    
    try:
        # Test individual methods
        logger.info("=" * 60)
        freshness_result = test_freshness_check()
        test_results.append(("Freshness Check", freshness_result))
        
        logger.info("=" * 60)
        cached_info_result = test_cached_maps_info()
        test_results.append(("Cached Maps Info", cached_info_result))
        
        logger.info("=" * 60) 
        integration_result = test_integration_logic()
        test_results.append(("Integration Logic", integration_result))
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY:")
        all_passed = True
        
        for test_name, result in test_results:
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                all_passed = False
        
        if all_passed:
            logger.info("🎉 ALL FRESH DATA OPTIMIZATION TESTS PASSED!")
            logger.info("💰 Fresh data optimization methods working correctly")
            logger.info("📅 Timestamp validation logic functioning properly") 
            logger.info("🏠 Cached data construction working as expected")
        else:
            logger.error("❌ SOME TESTS FAILED - Review implementation")
            
        logger.info("🏁 Test completed")
        return all_passed
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)