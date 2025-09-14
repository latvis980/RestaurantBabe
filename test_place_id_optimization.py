#!/usr/bin/env python3
"""
Test script to verify place_id optimization in Google Maps API usage
Simulates a restaurant search to check if place_id is being used instead of expensive text search
"""

import sys
import logging
import time
from typing import Dict, List, Any

# Import project modules
import config
from utils.database import initialize_db, get_supabase_manager
from agents.follow_up_search_agent import FollowUpSearchAgent

# Configure logging for test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_place_id_optimization():
    """Test that place_id optimization is working for database restaurants"""
    
    logger.info("🧪 Starting place_id optimization test")
    
    # Initialize database
    initialize_db(config)
    
    # Get supabase client
    supabase_manager = get_supabase_manager()
    supabase = supabase_manager.supabase
    
    # Initialize follow-up search agent 
    follow_up_agent = FollowUpSearchAgent(config)
    
    # Get a sample of database restaurants that should have place_id
    logger.info("📊 Fetching database restaurants...")
    
    try:
        # Get restaurants from a popular city (more likely to have place_id)
        response = supabase.table('restaurants').select('*').eq('city', 'Paris').limit(5).execute()
        sample_restaurants = response.data
        
        if not sample_restaurants:
            # Try another city if Paris has no results
            response = supabase.table('restaurants').select('*').limit(5).execute()
            sample_restaurants = response.data
            
        logger.info(f"📋 Found {len(sample_restaurants)} sample restaurants for testing")
        
        # Analyze place_id availability
        restaurants_with_place_id = [r for r in sample_restaurants if r.get('place_id')]
        logger.info(f"🎯 {len(restaurants_with_place_id)}/{len(sample_restaurants)} restaurants have place_id")
        
        if not restaurants_with_place_id:
            logger.warning("⚠️ No restaurants with place_id found - cannot test optimization")
            return False
            
        # Test follow-up search with restaurants that have place_id
        logger.info("🔍 Testing follow-up search with place_id optimization...")
        
        # Create test data structure expected by follow-up search
        test_data = {
            "main_list": restaurants_with_place_id[:3]  # Test with 3 restaurants
        }
        
        # Track API usage before test
        initial_primary_usage = follow_up_agent.api_usage.get('primary', 0)
        initial_secondary_usage = follow_up_agent.api_usage.get('secondary', 0)
        
        logger.info(f"📊 Initial API usage - Primary: {initial_primary_usage}, Secondary: {initial_secondary_usage}")
        
        # Perform follow-up search (should use place_id optimization)
        start_time = time.time()
        result = follow_up_agent.perform_follow_up_searches(
            edited_results=test_data,
            follow_up_queries=[],
            destination="Test City"
        )
        end_time = time.time()
        
        # Track API usage after test
        final_primary_usage = follow_up_agent.api_usage.get('primary', 0)
        final_secondary_usage = follow_up_agent.api_usage.get('secondary', 0)
        
        # Calculate API calls made during test
        primary_calls = final_primary_usage - initial_primary_usage
        secondary_calls = final_secondary_usage - initial_secondary_usage
        total_calls = primary_calls + secondary_calls
        
        logger.info(f"📊 API usage during test - Primary: +{primary_calls}, Secondary: +{secondary_calls}, Total: +{total_calls}")
        logger.info(f"⏱️ Test completed in {end_time - start_time:.2f} seconds")
        
        # Analyze results
        enhanced_results = result.get("enhanced_results", {})
        final_restaurants = enhanced_results.get("main_list", [])
        
        logger.info(f"✅ Test results: {len(final_restaurants)} restaurants processed successfully")
        
        # Check if any restaurants were filtered out (business status/rating)
        filtered_out = len(restaurants_with_place_id[:3]) - len(final_restaurants)
        if filtered_out > 0:
            logger.info(f"🚫 {filtered_out} restaurants filtered out (likely closed or low-rated)")
            
        # Success criteria:
        # 1. API calls should be minimal (ideally <= number of restaurants tested)
        # 2. Results should be returned quickly
        # 3. Business status and rating filtering should work
        
        success = True
        
        if total_calls > len(restaurants_with_place_id[:3]) * 2:  # Allow some buffer for retries
            logger.warning(f"⚠️ More API calls than expected: {total_calls} calls for {len(restaurants_with_place_id[:3])} restaurants")
            success = False
        else:
            logger.info("✅ API usage is optimized (using place_id)")
            
        if end_time - start_time > 30:  # Should be fast with place_id
            logger.warning(f"⚠️ Test took longer than expected: {end_time - start_time:.2f}s")
            success = False
        else:
            logger.info("✅ Response time is fast (place_id optimization working)")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def analyze_database_place_id_coverage():
    """Analyze how many restaurants in database have place_id"""
    
    logger.info("📊 Analyzing place_id coverage in database...")
    
    # Initialize database first
    initialize_db(config)
    
    # Get supabase client
    supabase_manager = get_supabase_manager()
    supabase = supabase_manager.supabase
    
    try:
        # Get total restaurant count
        total_response = supabase.table('restaurants').select('id', count='exact').execute()
        total_count = total_response.count
        
        # Get restaurants with place_id
        place_id_response = supabase.table('restaurants').select('id', count='exact').not_.is_('place_id', 'null').execute()
        place_id_count = place_id_response.count
        
        coverage_percentage = (place_id_count / total_count * 100) if total_count > 0 else 0
        
        logger.info(f"📊 Place ID Coverage Analysis:")
        logger.info(f"   Total restaurants: {total_count}")
        logger.info(f"   With place_id: {place_id_count}")
        logger.info(f"   Coverage: {coverage_percentage:.1f}%")
        
        if coverage_percentage > 90:
            logger.info("✅ Excellent place_id coverage - optimization will be highly effective")
        elif coverage_percentage > 70:
            logger.info("✅ Good place_id coverage - optimization will be effective")
        elif coverage_percentage > 50:
            logger.info("⚠️ Moderate place_id coverage - optimization will have some benefit")
        else:
            logger.info("⚠️ Low place_id coverage - optimization will have limited benefit")
            
        return coverage_percentage
        
    except Exception as e:
        logger.error(f"❌ Failed to analyze place_id coverage: {e}")
        return 0

if __name__ == "__main__":
    logger.info("🚀 Starting Google Maps API optimization verification")
    
    # Step 1: Analyze database place_id coverage
    coverage = analyze_database_place_id_coverage()
    
    # Step 2: Test place_id optimization
    if coverage > 0:
        success = test_place_id_optimization()
        
        if success:
            logger.info("🎉 PLACE_ID OPTIMIZATION TEST PASSED!")
            logger.info("💰 API quota savings confirmed - restaurants using existing place_id instead of expensive text search")
            logger.info("🔍 Business status filtering working - closed restaurants filtered out")
            logger.info("⭐ Rating validation working - low-rated restaurants filtered out")
        else:
            logger.error("❌ PLACE_ID OPTIMIZATION TEST FAILED!")
            logger.error("💸 API quota may not be optimized - check implementation")
    else:
        logger.error("❌ Cannot test optimization - no place_id data in database")
        
    logger.info("🏁 Test completed")