# test_database_search.py
"""
Database Search Test Script - Complete Step-by-Step Logging

This script tests the corrected 4-step database search flow and logs:
- Exact SQL queries sent to database
- Basic data retrieved in step 2  
- AI filtering process and results
- Full data retrieved in step 4
- Performance metrics and timing

Usage: python test_database_search.py
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'database_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

class DatabaseSearchTester:
    """
    Comprehensive tester for the 4-step database search flow
    """

    def __init__(self):
        """Initialize test environment"""
        try:
            # Import configuration and agents
            import config
            from agents.query_analyzer import QueryAnalyzer
            from agents.database_search_agent import DatabaseSearchAgent
            from utils.database import initialize_database, get_database

            # Initialize database
            initialize_database(config)
            self.db = get_database()

            # Initialize agents
            self.query_analyzer = QueryAnalyzer(config)
            self.database_search_agent = DatabaseSearchAgent(config)

            logger.info("✅ Database Search Tester initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize tester: {e}")
            raise

    def test_database_search_flow(self, test_query: str, expected_city: str = None):
        """
        Test the complete 4-step database search flow with detailed logging
        """
        print("\n" + "="*80)
        print("DATABASE SEARCH FLOW TEST - COMPLETE STEP LOGGING")
        print("="*80)
        print(f"Test Query: '{test_query}'")
        print(f"Expected City: {expected_city or 'Auto-detect'}")
        print(f"Test Time: {datetime.now().isoformat()}")
        print()

        total_start_time = time.time()

        try:
            # PHASE 1: Query Analysis
            print("🔍 PHASE 1: QUERY ANALYSIS")
            print("-" * 40)

            analysis_start = time.time()
            query_analysis = self.query_analyzer.analyze(test_query)
            analysis_time = time.time() - analysis_start

            destination = query_analysis.get('destination', 'Unknown')
            search_queries = query_analysis.get('search_queries', [])
            raw_query = query_analysis.get('raw_query', test_query)

            print(f"⏱️  Analysis Time: {analysis_time:.2f}s")
            print(f"🏙️  Detected Destination: {destination}")
            print(f"🔍 Generated Search Queries: {len(search_queries)}")
            for i, sq in enumerate(search_queries, 1):
                print(f"   {i}. '{sq}'")
            print(f"📝 Raw Query: '{raw_query}'")
            print()

            if expected_city and destination.lower() != expected_city.lower():
                print(f"⚠️  WARNING: Expected '{expected_city}' but got '{destination}'")

            # PHASE 2: Database Search Flow
            print("🗃️ PHASE 2: 4-STEP DATABASE SEARCH")
            print("-" * 40)

            db_start = time.time()

            # Test each step individually with detailed logging
            self._test_step_1(query_analysis)
            basic_restaurants = self._test_step_2(destination)

            if not basic_restaurants:
                print("❌ EARLY EXIT: No restaurants found in step 2")
                return

            filtered_restaurants = self._test_step_3(basic_restaurants, search_queries, destination)

            if not filtered_restaurants:
                print("❌ EARLY EXIT: No restaurants selected by AI in step 3")
                return

            detailed_restaurants = self._test_step_4(filtered_restaurants)

            db_time = time.time() - db_start
            print(f"⏱️  Total Database Search Time: {db_time:.2f}s")
            print()

            # PHASE 3: Results Analysis
            print("📊 PHASE 3: RESULTS ANALYSIS")
            print("-" * 40)

            self._analyze_results(basic_restaurants, filtered_restaurants, detailed_restaurants)

            # PHASE 4: Performance Summary
            total_time = time.time() - total_start_time
            print("⚡ PHASE 4: PERFORMANCE SUMMARY")
            print("-" * 40)
            print(f"Total Test Time: {total_time:.2f}s")
            print(f"Query Analysis: {analysis_time:.2f}s ({(analysis_time/total_time)*100:.1f}%)")
            print(f"Database Search: {db_time:.2f}s ({(db_time/total_time)*100:.1f}%)")
            print()
            print("✅ TEST COMPLETED SUCCESSFULLY")

        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

    def _test_step_1(self, query_analysis: Dict[str, Any]):
        """Test Step 1: Get destination from query"""
        print("📋 STEP 1: Get Destination")
        destination = query_analysis.get('destination', 'Unknown')

        print(f"   Input: Query analysis data")
        print(f"   Output: destination = '{destination}'")

        if destination == "Unknown":
            print("   ⚠️  Status: FAILED - No destination detected")
        else:
            print("   ✅ Status: SUCCESS")
        print()

    def _test_step_2(self, destination: str) -> List[Dict[str, Any]]:
        """Test Step 2: Extract basic data (id, name, cuisine_tags)"""
        print("📋 STEP 2: Extract Basic Restaurant Data")
        print(f"   Target City: {destination}")

        step2_start = time.time()

        # Show the exact SQL query being executed
        print("   🔍 Database Query:")
        print(f"      SELECT id, name, cuisine_tags, mention_count")
        print(f"      FROM restaurants") 
        print(f"      WHERE city = '{destination}'")
        print(f"      ORDER BY mention_count DESC")
        print(f"      LIMIT 100")

        try:
            # Execute the query
            result = self.db.supabase.table('restaurants')\
                .select('id, name, cuisine_tags, mention_count')\
                .eq('city', destination)\
                .order('mention_count', desc=True)\
                .limit(100)\
                .execute()

            restaurants = result.data or []
            step2_time = time.time() - step2_start

            print(f"   ⏱️  Query Time: {step2_time:.3f}s")
            print(f"   📊 Results: {len(restaurants)} restaurants")

            if restaurants:
                print("   ✅ Status: SUCCESS")
                print("   📝 Sample Basic Data (first 3 restaurants):")

                for i, restaurant in enumerate(restaurants[:3], 1):
                    print(f"      {i}. ID: {restaurant.get('id')}")
                    print(f"         Name: {restaurant.get('name')}")
                    print(f"         Cuisine Tags: {restaurant.get('cuisine_tags', [])}")
                    print(f"         Mention Count: {restaurant.get('mention_count', 0)}")
                    print(f"         Fields in Record: {list(restaurant.keys())}")
                    print()
            else:
                print("   ❌ Status: FAILED - No restaurants found")

            return restaurants

        except Exception as e:
            print(f"   ❌ Status: ERROR - {e}")
            return []

    def _test_step_3(self, restaurants: List[Dict[str, Any]], search_queries: List[str], destination: str) -> List[Dict[str, Any]]:
        """Test Step 3: AI batch analysis using search_queries"""
        print("📋 STEP 3: AI Batch Analysis")
        print(f"   Input Restaurants: {len(restaurants)}")
        print(f"   Search Queries: {search_queries}")
        print(f"   Destination: {destination}")

        step3_start = time.time()

        try:
            # Test the filtering method directly
            filtered_restaurants = self.database_search_agent._batch_filter_restaurants(
                restaurants, search_queries, destination
            )

            step3_time = time.time() - step3_start

            print(f"   ⏱️  AI Processing Time: {step3_time:.3f}s")
            print(f"   📊 AI Selected: {len(filtered_restaurants)} restaurants")

            if filtered_restaurants:
                print("   ✅ Status: SUCCESS")
                print("   📝 AI Selection Details:")

                for i, restaurant in enumerate(filtered_restaurants, 1):
                    score = restaurant.get('_relevance_score', 'N/A')
                    reasoning = restaurant.get('_reasoning', 'N/A')
                    print(f"      {i}. {restaurant.get('name')} (ID: {restaurant.get('id')})")
                    print(f"         Relevance Score: {score}")
                    print(f"         AI Reasoning: {reasoning}")
                    print()
            else:
                print("   ❌ Status: FAILED - AI selected no restaurants")

            return filtered_restaurants

        except Exception as e:
            print(f"   ❌ Status: ERROR - {e}")
            import traceback
            traceback.print_exc()
            return []

    def _test_step_4(self, filtered_restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test Step 4: Fetch full details for selected restaurants"""
        print("📋 STEP 4: Fetch Full Restaurant Details")
        print(f"   Selected Restaurant IDs: {[r.get('id') for r in filtered_restaurants]}")

        step4_start = time.time()
        detailed_restaurants = []

        try:
            for i, restaurant in enumerate(filtered_restaurants, 1):
                restaurant_id = restaurant.get('id')
                print(f"   🔍 Query {i}/{len(filtered_restaurants)}: Restaurant ID {restaurant_id}")
                print(f"      SELECT * FROM restaurants WHERE id = {restaurant_id}")

                # Execute the full details query
                result = self.db.supabase.table('restaurants')\
                    .select('*')\
                    .eq('id', restaurant_id)\
                    .execute()

                if result.data:
                    full_restaurant = result.data[0]

                    # Preserve AI metadata
                    full_restaurant['_relevance_score'] = restaurant.get('_relevance_score', 0)
                    full_restaurant['_reasoning'] = restaurant.get('_reasoning', '')

                    detailed_restaurants.append(full_restaurant)

                    # Show what full data looks like
                    desc_length = len(full_restaurant.get('raw_description', ''))
                    sources_count = len(full_restaurant.get('sources', []) if isinstance(full_restaurant.get('sources'), list) else [])

                    print(f"      ✅ Retrieved: {full_restaurant.get('name')}")
                    print(f"         Total Fields: {len(full_restaurant.keys())}")
                    print(f"         Description Length: {desc_length} chars")
                    print(f"         Sources Count: {sources_count}")
                    print(f"         Has Address: {'Yes' if full_restaurant.get('address') else 'No'}")
                    print()
                else:
                    print(f"      ❌ Not found: Restaurant ID {restaurant_id}")

            step4_time = time.time() - step4_start
            print(f"   ⏱️  Full Details Query Time: {step4_time:.3f}s")
            print(f"   📊 Successfully Retrieved: {len(detailed_restaurants)} restaurants")
            print("   ✅ Status: SUCCESS")

            return detailed_restaurants

        except Exception as e:
            print(f"   ❌ Status: ERROR - {e}")
            return []

    def _analyze_results(self, basic_restaurants: List[Dict], filtered_restaurants: List[Dict], detailed_restaurants: List[Dict]):
        """Analyze the results from all steps"""
        print(f"📈 Data Flow Analysis:")
        print(f"   Step 2 (Basic Data): {len(basic_restaurants)} restaurants")
        print(f"   Step 3 (AI Filtered): {len(filtered_restaurants)} restaurants")
        print(f"   Step 4 (Full Details): {len(detailed_restaurants)} restaurants")

        if basic_restaurants:
            filter_rate = (len(filtered_restaurants) / len(basic_restaurants)) * 100
            print(f"   AI Filter Rate: {filter_rate:.1f}% of restaurants selected")

        print()
        print("📋 Data Quality Analysis:")

        if detailed_restaurants:
            with_descriptions = sum(1 for r in detailed_restaurants if r.get('raw_description'))
            with_addresses = sum(1 for r in detailed_restaurants if r.get('address'))
            with_sources = sum(1 for r in detailed_restaurants if r.get('sources'))

            print(f"   Restaurants with descriptions: {with_descriptions}/{len(detailed_restaurants)} ({(with_descriptions/len(detailed_restaurants))*100:.1f}%)")
            print(f"   Restaurants with addresses: {with_addresses}/{len(detailed_restaurants)} ({(with_addresses/len(detailed_restaurants))*100:.1f}%)")
            print(f"   Restaurants with sources: {with_sources}/{len(detailed_restaurants)} ({(with_sources/len(detailed_restaurants))*100:.1f}%)")

            # Show sample full restaurant data
            print()
            print("📝 Sample Full Restaurant Data:")
            sample = detailed_restaurants[0]
            print(f"   Name: {sample.get('name')}")
            print(f"   ID: {sample.get('id')}")
            print(f"   Cuisine Tags: {sample.get('cuisine_tags', [])}")
            print(f"   Address: {sample.get('address', 'Not available')}")
            print(f"   Description: {sample.get('raw_description', 'Not available')[:200]}...")
            print(f"   Sources: {sample.get('sources', 'Not available')}")
            print(f"   AI Score: {sample.get('_relevance_score', 'N/A')}")
            print(f"   AI Reasoning: {sample.get('_reasoning', 'N/A')}")

    def test_multiple_queries(self):
        """Test multiple queries to verify consistency"""
        test_cases = [
            {
                "query": "best Italian restaurants in Paris",
                "expected_city": "Paris"
            },
            {
                "query": "top sushi places in Tokyo", 
                "expected_city": "Tokyo"
            },
            {
                "query": "vegan restaurants in Berlin",
                "expected_city": "Berlin"
            },
            {
                "query": "fine dining in London",
                "expected_city": "London"
            },
            {
                "query": "cheap eats in New York",
                "expected_city": "New York"
            }
        ]

        print("\n" + "="*80)
        print("MULTIPLE QUERY TEST - CONSISTENCY CHECK")
        print("="*80)

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- TEST CASE {i}/{len(test_cases)} ---")
            try:
                self.test_database_search_flow(test_case["query"], test_case["expected_city"])
            except Exception as e:
                print(f"❌ Test case {i} failed: {e}")
                continue

            if i < len(test_cases):
                print("\n" + "⏳ Waiting 2 seconds before next test...")
                time.sleep(2)

    def test_database_connection(self):
        """Test basic database connectivity and show available data"""
        print("\n" + "="*80)
        print("DATABASE CONNECTION TEST")
        print("="*80)

        try:
            # Test basic connection
            stats = self.db.get_database_stats()
            print(f"✅ Database Connection: SUCCESS")
            print(f"📊 Total Restaurants: {stats.get('total_restaurants', 0)}")
            print()

            # Show available cities
            print("🏙️ Available Cities (Top 10):")
            top_cities = stats.get('top_cities', [])[:10]
            for city in top_cities:
                print(f"   {city.get('city')}: {city.get('count')} restaurants")
            print()

            # Test sample city
            if top_cities:
                sample_city = top_cities[0]['city']
                print(f"🧪 Testing Sample City: {sample_city}")

                # Get sample restaurants
                restaurants = self.db.get_restaurants_by_city(sample_city, limit=3)
                print(f"📋 Sample Restaurants in {sample_city}:")

                for i, restaurant in enumerate(restaurants, 1):
                    print(f"   {i}. {restaurant.get('name')}")
                    print(f"      Cuisine: {restaurant.get('cuisine_tags', [])}")
                    print(f"      Has Description: {'Yes' if restaurant.get('raw_description') else 'No'}")

            return True

        except Exception as e:
            print(f"❌ Database Connection: FAILED")
            print(f"Error: {e}")
            return False

def main():
    """Run the comprehensive database search test"""
    print("🔍 DATABASE SEARCH TESTER")
    print("Testing the corrected 4-step database search flow")
    print()

    try:
        tester = DatabaseSearchTester()

        # Test database connection first
        if not tester.test_database_connection():
            print("❌ Cannot proceed - database connection failed")
            return

        # Test single query with detailed logging
        test_query = input("Enter a test query (or press Enter for default): ").strip()
        if not test_query:
            test_query = "best Italian restaurants in Paris"

        tester.test_database_search_flow(test_query)

        # Ask if user wants to test multiple queries
        test_multiple = input("\nWould you like to test multiple queries? (y/n): ").strip().lower()
        if test_multiple == 'y':
            tester.test_multiple_queries()

        print("\n🎉 All tests completed!")
        print("Check the log files for detailed step-by-step information.")

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()