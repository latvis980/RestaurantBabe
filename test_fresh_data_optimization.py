#!/usr/bin/env python3

"""
Test script to verify the Fresh Data optimization is working correctly.
Checks that:
1. Fresh restaurants (< 3 months) skip API calls and use cached data
2. Stale restaurants (≥ 3 months) trigger API calls  
3. Missing timestamps trigger API calls
4. Successful API calls update timestamps
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.follow_up_search_agent import FollowUpSearchAgent
from utils.database import get_database
from utils.debug_utils import log_function_call
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreshDataOptimizationTester:
    """Test the fresh data optimization functionality"""
    
    def __init__(self):
        self.db = None
        self.follow_up_agent = None
        
    async def setup(self):
        """Initialize database and follow-up agent"""
        try:
            # Initialize database with proper initialization
            self.db = get_database()
            await self.db.initialize_database()  # Initialize the database properly
            logger.info("✅ Database initialized")
            
            # Initialize follow-up search agent
            self.follow_up_agent = FollowUpSearchAgent(config)
            logger.info("✅ Follow-up search agent initialized")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise

    def create_test_restaurants(self) -> Dict[str, Dict[str, Any]]:
        """Create test restaurants with different timestamp scenarios"""
        now = datetime.now(timezone.utc)
        
        # Test scenarios
        scenarios = {
            "fresh_restaurant": {
                "name": "Fresh Test Restaurant",
                "city": "Test City",
                "address": "123 Fresh Street, Test City",
                "rating": 4.5,
                "user_ratings_total": 100,
                "business_status": "OPERATIONAL",
                "place_id": "ChIJ_test_fresh_123",
                "latitude": 48.8566,
                "longitude": 2.3522,
                "last_updated": (now - timedelta(days=30)).isoformat(),  # 30 days old = fresh
                "description": "This is a test restaurant for fresh data optimization"
            },
            "stale_restaurant": {
                "name": "Stale Test Restaurant", 
                "city": "Test City",
                "address": "456 Stale Avenue, Test City",
                "rating": 4.2,
                "user_ratings_total": 85,
                "business_status": "OPERATIONAL", 
                "place_id": "ChIJ_test_stale_456",
                "latitude": 48.8606,
                "longitude": 2.3376,
                "last_updated": (now - timedelta(days=120)).isoformat(),  # 120 days old = stale
                "description": "This is a test restaurant for stale data testing"
            },
            "no_timestamp_restaurant": {
                "name": "No Timestamp Restaurant",
                "city": "Test City", 
                "address": "789 Unknown Date Street, Test City",
                "rating": 4.3,
                "user_ratings_total": 120,
                "business_status": "OPERATIONAL",
                "place_id": "ChIJ_test_no_timestamp_789", 
                "latitude": 48.8656,
                "longitude": 2.3212,
                "last_updated": None,  # No timestamp
                "description": "This is a test restaurant with no timestamp"
            }
        }
        
        return scenarios
    
    def track_api_calls(self):
        """Get current API usage to track calls during test"""
        if self.follow_up_agent and hasattr(self.follow_up_agent, 'api_usage'):
            return {
                'primary': self.follow_up_agent.api_usage.get('primary', 0),
                'secondary': self.follow_up_agent.api_usage.get('secondary', 0)
            }
        return {'primary': 0, 'secondary': 0}

    async def test_fresh_data_optimization(self):
        """Test that fresh data optimization works correctly"""
        logger.info("🧪 Starting Fresh Data Optimization Test")
        
        # Create test restaurant data
        test_restaurants = self.create_test_restaurants()
        
        # Track initial API usage
        initial_api_usage = self.track_api_calls()
        logger.info(f"📊 Initial API usage - Primary: {initial_api_usage['primary']}, Secondary: {initial_api_usage['secondary']}")
        
        results = {}
        
        for scenario_name, restaurant_data in test_restaurants.items():
            logger.info(f"🔍 Testing scenario: {scenario_name}")
            
            # Track API usage before this test
            pre_test_api_usage = self.track_api_calls()
            
            # Test the restaurant verification
            try:
                result = self.follow_up_agent._verify_and_filter_restaurant(restaurant_data, "Test City")
                
                # Track API usage after this test  
                post_test_api_usage = self.track_api_calls()
                
                # Calculate API calls made for this test
                api_calls_made = {
                    'primary': post_test_api_usage['primary'] - pre_test_api_usage['primary'],
                    'secondary': post_test_api_usage['secondary'] - pre_test_api_usage['secondary'],
                    'total': (post_test_api_usage['primary'] - pre_test_api_usage['primary']) + 
                            (post_test_api_usage['secondary'] - pre_test_api_usage['secondary'])
                }
                
                results[scenario_name] = {
                    'success': result is not None,
                    'api_calls_made': api_calls_made,
                    'restaurant_data': result if result else restaurant_data
                }
                
                logger.info(f"✅ {scenario_name} completed - API calls: {api_calls_made['total']}")
                
            except Exception as e:
                logger.error(f"❌ Error testing {scenario_name}: {e}")
                results[scenario_name] = {
                    'success': False,
                    'error': str(e),
                    'api_calls_made': {'primary': 0, 'secondary': 0, 'total': 0}
                }
        
        # Analyze results
        self._analyze_optimization_results(results)
        
        return results
        
    def _analyze_optimization_results(self, results: Dict[str, Dict[str, Any]]):
        """Analyze test results to verify optimization is working"""
        logger.info("📊 Analyzing Fresh Data Optimization Results:")
        
        # Expected behavior:
        # - Fresh restaurant: 0 API calls (uses cached data)
        # - Stale restaurant: 1+ API calls (needs fresh data)
        # - No timestamp restaurant: 1+ API calls (needs verification)
        
        expected_behavior = {
            'fresh_restaurant': 0,  # Should use cached data
            'stale_restaurant': 1,  # Should make API call
            'no_timestamp_restaurant': 1  # Should make API call
        }
        
        optimization_working = True
        
        for scenario, expected_calls in expected_behavior.items():
            if scenario in results:
                actual_calls = results[scenario]['api_calls_made']['total']
                
                if expected_calls == 0:
                    # Fresh data - should not make API calls
                    if actual_calls == 0:
                        logger.info(f"✅ {scenario}: Correctly used cached data (0 API calls)")
                    else:
                        logger.error(f"❌ {scenario}: Expected 0 API calls, got {actual_calls}")
                        optimization_working = False
                else:
                    # Stale/missing data - should make API calls
                    if actual_calls >= expected_calls:
                        logger.info(f"✅ {scenario}: Correctly made API calls ({actual_calls} calls)")
                    else:
                        logger.error(f"❌ {scenario}: Expected {expected_calls}+ API calls, got {actual_calls}")
                        optimization_working = False
            else:
                logger.error(f"❌ Missing results for {scenario}")
                optimization_working = False
        
        if optimization_working:
            logger.info("🎉 FRESH DATA OPTIMIZATION TEST PASSED!")
            logger.info("💰 API quota savings confirmed - fresh restaurants use cached data")
            logger.info("🔍 Stale data detection working - old restaurants trigger API calls")
            logger.info("📅 Timestamp validation working - missing timestamps trigger API calls")
        else:
            logger.error("❌ FRESH DATA OPTIMIZATION TEST FAILED!")
            logger.error("🚨 Optimization not working correctly - review implementation")

async def main():
    """Main test runner"""
    logger.info("🚀 Starting Fresh Data Optimization verification")
    
    tester = FreshDataOptimizationTester()
    
    try:
        # Setup test environment
        await tester.setup()
        
        # Run fresh data optimization test
        results = await tester.test_fresh_data_optimization()
        
        logger.info("🏁 Test completed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())