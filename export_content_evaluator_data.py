"""
Updated test script to export the actual file that goes to the content evaluator.
Uses the actual files and pipeline to generate authentic data with all required Supabase fields.
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Any

def export_content_evaluator_input_data():
    """
    Export the exact data format that goes to the content evaluator (DBContentEvaluationAgent).

    This follows the actual pipeline:
    1. Query Analysis → destination, search_queries
    2. Database Search → basic + full restaurant data with all Supabase fields
    3. Format pipeline_data exactly as content evaluator receives it
    """
    try:
        # Import the actual pipeline components
        import config
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent
        from utils.database import initialize_database, get_database

        print("🔧 Initializing actual pipeline components...")

        # Initialize database and agents (same as production)
        initialize_database(config)
        db = get_database()
        query_analyzer = QueryAnalyzer(config)
        database_search_agent = DatabaseSearchAgent(config)

        # Test query (you can modify this)
        test_query = "best brunch restaurants in Berlin"

        print(f"🔍 Testing query: '{test_query}'")

        # STEP 1: Run actual query analysis
        print("📋 Step 1: Running query analysis...")
        query_analysis = query_analyzer.analyze(test_query)

        destination = query_analysis.get("destination", "Unknown")
        search_queries = query_analysis.get("search_queries", [])

        print(f"   Destination: {destination}")
        print(f"   Search queries: {search_queries}")

        # STEP 2: Run actual database search to get full pipeline data
        print("🗃️ Step 2: Running database search...")
        database_result = database_search_agent.search_and_evaluate(query_analysis)

        database_restaurants = database_result.get("database_restaurants", [])

        print(f"   Found: {len(database_restaurants)} restaurants")

        # STEP 3: Create the exact pipeline_data structure that content evaluator receives
        print("📦 Step 3: Formatting pipeline data for content evaluator...")

        pipeline_data = {
            "raw_query": test_query,
            "destination": destination,
            "search_queries": search_queries,
            "database_restaurants": database_restaurants,
            "has_database_content": len(database_restaurants) > 0,
            "restaurant_count": len(database_restaurants),
            "is_english_speaking": query_analysis.get("is_english_speaking", True),
            "local_language": query_analysis.get("local_language", None)
        }

        # STEP 4: Export to debug_logs folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"content_evaluator_input_{timestamp}.txt"

        # Create debug_logs folder if it doesn't exist
        debug_logs_dir = os.path.join(os.getcwd(), "debug_logs")
        os.makedirs(debug_logs_dir, exist_ok=True)

        filepath = os.path.join(debug_logs_dir, filename)

        print(f"📄 Step 4: Writing to debug_logs/{filename}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CONTENT EVALUATOR INPUT DATA - ACTUAL PIPELINE\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Script: Uses actual QueryAnalyzer and DatabaseSearchAgent\n")
            f.write(f"Query: {test_query}\n")
            f.write(f"Pipeline Data Keys: {list(pipeline_data.keys())}\n\n")

            # PIPELINE METADATA
            f.write("PIPELINE METADATA\n")
            f.write("-" * 40 + "\n")
            f.write(f"Raw Query: {pipeline_data['raw_query']}\n")
            f.write(f"Destination: {pipeline_data['destination']}\n")
            f.write(f"Search Queries: {pipeline_data['search_queries']}\n")
            f.write(f"Has Database Content: {pipeline_data['has_database_content']}\n")
            f.write(f"Restaurant Count: {pipeline_data['restaurant_count']}\n")
            f.write(f"Is English Speaking: {pipeline_data['is_english_speaking']}\n")
            f.write(f"Local Language: {pipeline_data['local_language']}\n\n")

            # RESTAURANT DATA ANALYSIS
            f.write("RESTAURANT DATA ANALYSIS\n")
            f.write("-" * 40 + "\n")

            if database_restaurants:
                # Show available fields from first restaurant
                sample_restaurant = database_restaurants[0]
                f.write(f"Available Fields: {list(sample_restaurant.keys())}\n")

                # Check data completeness
                with_descriptions = sum(1 for r in database_restaurants if r.get('raw_description'))
                with_addresses = sum(1 for r in database_restaurants if r.get('address'))
                with_coordinates = sum(1 for r in database_restaurants if r.get('latitude') and r.get('longitude'))
                with_sources = sum(1 for r in database_restaurants if r.get('sources'))

                f.write(f"Data Completeness:\n")
                f.write(f"  Descriptions: {with_descriptions}/{len(database_restaurants)}\n")
                f.write(f"  Addresses: {with_addresses}/{len(database_restaurants)}\n")
                f.write(f"  Coordinates: {with_coordinates}/{len(database_restaurants)}\n")
                f.write(f"  Sources: {with_sources}/{len(database_restaurants)}\n\n")

                # Show first 5 restaurants in detail
                f.write("RESTAURANT DETAILS (First 5)\n")
                f.write("-" * 40 + "\n")

                for i, restaurant in enumerate(database_restaurants[:5], 1):
                    f.write(f"{i}. RESTAURANT ID: {restaurant.get('id')}\n")
                    f.write(f"   Name: {restaurant.get('name', 'Unknown')}\n")
                    f.write(f"   Cuisine Tags: {restaurant.get('cuisine_tags', [])}\n")
                    f.write(f"   Address: {restaurant.get('address', 'Not available')}\n")
                    f.write(f"   City: {restaurant.get('city', 'Unknown')}\n")
                    f.write(f"   Country: {restaurant.get('country', 'Unknown')}\n")
                    f.write(f"   Mention Count: {restaurant.get('mention_count', 0)}\n")

                    # Coordinates
                    lat = restaurant.get('latitude')
                    lng = restaurant.get('longitude')
                    if lat and lng:
                        f.write(f"   Coordinates: ({lat}, {lng})\n")
                    else:
                        f.write(f"   Coordinates: Not available\n")

                    # Description (truncated)
                    description = restaurant.get('raw_description', '')
                    if description:
                        preview = description[:200] + "..." if len(description) > 200 else description
                        f.write(f"   Description: {preview}\n")
                    else:
                        f.write(f"   Description: Not available\n")

                    # Sources
                    sources = restaurant.get('sources', [])
                    if sources:
                        f.write(f"   Sources ({len(sources)}): {sources[:3]}{'...' if len(sources) > 3 else ''}\n")
                    else:
                        f.write(f"   Sources: Not available\n")

                    # Timestamps
                    f.write(f"   First Added: {restaurant.get('first_added', 'Unknown')}\n")
                    f.write(f"   Last Updated: {restaurant.get('last_updated', 'Unknown')}\n")
                    f.write("\n")

                if len(database_restaurants) > 5:
                    f.write(f"... and {len(database_restaurants) - 5} more restaurants\n\n")

            else:
                f.write("No restaurants found in database for this query.\n\n")

            # CONTENT EVALUATOR FORMAT
            f.write("CONTENT EVALUATOR INPUT FORMAT\n")
            f.write("-" * 40 + "\n")
            f.write("This is the exact structure that DBContentEvaluationAgent.evaluate_and_route() receives:\n\n")

            # Show the restaurant summary format that the AI evaluator sees
            if database_restaurants:
                f.write("AI EVALUATION SUMMARY FORMAT:\n")
                restaurants_summary = _create_restaurants_summary_for_ai(database_restaurants)
                f.write(restaurants_summary[:1000])  # Show first 1000 chars
                if len(restaurants_summary) > 1000:
                    f.write("...\n[Truncated for display]\n")

            f.write("\n\nRAW PIPELINE DATA (JSON)\n")
            f.write("-" * 40 + "\n")
            f.write(json.dumps(pipeline_data, indent=2, default=str))

        print(f"✅ Content evaluator input data exported successfully!")
        print(f"📁 File location: debug_logs/{filename}")
        print(f"📁 Full path: {filepath}")
        print(f"📊 Summary:")
        print(f"   - Query: {test_query}")
        print(f"   - Destination: {destination}")
        print(f"   - Restaurants found: {len(database_restaurants)}")
        print(f"   - Search queries: {len(search_queries)}")

        return filepath

    except Exception as e:
        print(f"❌ Error exporting content evaluator data: {e}")
        import traceback
        traceback.print_exc()
        return None

def _create_restaurants_summary_for_ai(restaurants: List[Dict[str, Any]]) -> str:
    """
    Create the restaurant summary in the exact format that the content evaluator AI sees.
    This mimics the format from DBContentEvaluationAgent._create_restaurants_summary()
    """
    summary_lines = []

    for i, restaurant in enumerate(restaurants, 1):
        name = restaurant.get('name', 'Unknown')
        cuisine_tags = ', '.join(restaurant.get('cuisine_tags', []))
        mention_count = restaurant.get('mention_count', 0)

        # Get description preview (similar to what AI evaluator sees)
        description = restaurant.get('raw_description', '')
        desc_preview = description[:150] + "..." if len(description) > 150 else description

        # Format similar to content evaluator
        line = f"{i}. {name}"
        if cuisine_tags:
            line += f" | Cuisine: {cuisine_tags}"
        line += f" | Mentions: {mention_count}"
        if desc_preview:
            line += f" | Preview: {desc_preview}"

        summary_lines.append(line)

    return "\n".join(summary_lines)

if __name__ == "__main__":
    print("🚀 Exporting Content Evaluator Input Data")
    print("=" * 50)
    export_content_evaluator_input_data()