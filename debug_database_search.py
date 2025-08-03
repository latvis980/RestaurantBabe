# debug_database_search.py
"""
Debug script to examine the exact format of restaurant data and AI response
to find the mismatch causing 0 results in Step 3 filtering.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_database_search():
    """Debug the database search step by step"""

    print("🔍 DEBUGGING DATABASE SEARCH AGENT - STEP 3 ISSUE")
    print("=" * 80)

    try:
        # Import required modules
        import config
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent
        from utils.database import initialize_database, get_database

        # Initialize
        initialize_database(config)
        db = get_database()
        query_analyzer = QueryAnalyzer(config)
        database_search_agent = DatabaseSearchAgent(config)

        # Test query that should return 4 restaurants
        test_query = "best brunch restaurants in Berlin"

        print(f"\n📋 TEST QUERY: {test_query}")
        print("-" * 40)

        # Step 1: Query Analysis
        print("\n🔍 STEP 1: Query Analysis")
        query_analysis = query_analyzer.analyze(test_query)

        destination = query_analysis.get("destination", "Unknown")
        search_queries = query_analysis.get("search_queries", [])

        print(f"   Destination: {destination}")
        print(f"   Search Queries: {search_queries}")

        # Step 2: Get basic restaurant data directly from database
        print(f"\n📊 STEP 2: Database Query for {destination}")

        basic_restaurants = database_search_agent._get_basic_restaurant_data(destination)

        print(f"   Found {len(basic_restaurants)} restaurants")
        if basic_restaurants:
            print(f"   Sample restaurant data structure:")
            sample = basic_restaurants[0]
            print(f"      ID: {sample.get('id')} (type: {type(sample.get('id'))})")
            print(f"      Name: {sample.get('name')}")
            print(f"      Cuisine Tags: {sample.get('cuisine_tags')}")
            print(f"      Fields: {list(sample.keys())}")

            # Show first few restaurants
            print(f"\n   First 3 restaurants:")
            for i, restaurant in enumerate(basic_restaurants[:3], 1):
                print(f"      {i}. ID: {restaurant.get('id')} | Name: {restaurant.get('name')}")

        # Step 3: Format for AI analysis
        print(f"\n🤖 STEP 3: AI Input Formatting")

        restaurants_text = database_search_agent._compile_basic_restaurants_for_analysis(basic_restaurants)
        search_intent = " | ".join(search_queries)

        print(f"   Search Intent: {search_intent}")
        print(f"   Compiled Text Length: {len(restaurants_text)} characters")
        print(f"   Compiled Text Sample (first 500 chars):")
        print(f"   {restaurants_text[:500]}...")

        # Step 4: AI Analysis - Manual call to see exact response
        print(f"\n🧠 STEP 4: AI Analysis (Manual)")

        # Use the ACTUAL prompt from the database search agent
        prompt = database_search_agent.batch_analysis_prompt

        chain = prompt | database_search_agent.llm

        response = chain.invoke({
            "search_queries": search_intent,
            "destination": destination,
            "restaurants_text": restaurants_text
        })

        print(f"   AI Raw Response:")
        print(f"   {response.content}")

        # Parse the response
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        print(f"\n   AI Cleaned Response:")
        print(f"   {content}")

        try:
            analysis_result = json.loads(content)
            print(f"\n   AI Selected Restaurants:")

            selected_restaurants = analysis_result.get("selected_restaurants", [])
            print(f"   Count: {len(selected_restaurants)}")

            if selected_restaurants:
                for i, selection in enumerate(selected_restaurants, 1):
                    ai_id = selection.get('id', 'unknown')
                    score = selection.get('relevance_score', 'N/A')
                    reasoning = selection.get('reasoning', 'N/A')
                    print(f"      {i}. AI ID: '{ai_id}' (type: {type(ai_id)}) | Score: {score}")
                    print(f"         Reasoning: {reasoning}")

            # Step 5: Check mapping
            print(f"\n🔗 STEP 5: ID Mapping Check")

            # Create lookup dict like the agent does
            restaurant_lookup = {str(r.get('id')): r for r in basic_restaurants}

            print(f"   Database IDs in lookup:")
            for db_id in list(restaurant_lookup.keys())[:5]:  # Show first 5
                print(f"      Database ID: '{db_id}' (type: {type(db_id)})")

            print(f"\n   Mapping Results:")
            for selection in selected_restaurants:
                ai_id = str(selection.get('id', ''))
                found = ai_id in restaurant_lookup
                print(f"      AI ID: '{ai_id}' → Found in DB: {found}")

                if not found:
                    # Try to find similar IDs
                    similar_ids = [db_id for db_id in restaurant_lookup.keys() if ai_id in db_id or db_id in ai_id]
                    if similar_ids:
                        print(f"         Similar DB IDs: {similar_ids}")
                    else:
                        print(f"         No similar IDs found")

        except json.JSONDecodeError as e:
            print(f"   ❌ JSON parsing failed: {e}")

        # Step 6: Run the actual agent method
        print(f"\n⚙️ STEP 6: Agent Method Test")

        filtered_restaurants = database_search_agent._batch_filter_restaurants(
            basic_restaurants, search_queries, destination
        )

        print(f"   Agent returned: {len(filtered_restaurants)} restaurants")

        if filtered_restaurants:
            for i, restaurant in enumerate(filtered_restaurants, 1):
                print(f"      {i}. {restaurant.get('name')} (ID: {restaurant.get('id')})")
        else:
            print(f"   ❌ No restaurants returned by agent")

    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_database_search()