# export_ai_input_data.py
"""
Simple script to export the exact data format sent to AI for filtering.
Creates a text file showing exactly what the AI sees.
"""

import os
import json
from datetime import datetime

def export_ai_input():
    """Export the exact AI input data to a file"""

    try:
        import config
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent
        from utils.database import initialize_database

        # Initialize
        initialize_database(config)
        query_analyzer = QueryAnalyzer(config)
        database_search_agent = DatabaseSearchAgent(config)

        # Test query
        test_query = "best brunch restaurants in Berlin"

        # Get query analysis
        query_analysis = query_analyzer.analyze(test_query)
        destination = query_analysis.get("destination", "Unknown")
        search_queries = query_analysis.get("search_queries", [])

        # Get basic restaurant data
        basic_restaurants = database_search_agent._get_basic_restaurant_data(destination)

        # Format the data exactly as sent to AI
        restaurants_text = database_search_agent._compile_basic_restaurants_for_analysis(basic_restaurants)
        search_intent = " | ".join(search_queries)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_input_data_{timestamp}.txt"

        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("AI INPUT DATA EXPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Original Query: {test_query}\n")
            f.write(f"Destination: {destination}\n")
            f.write(f"Search Intent: {search_intent}\n")
            f.write(f"Restaurant Count: {len(basic_restaurants)}\n\n")

            f.write("RESTAURANT DATA (as sent to AI):\n")
            f.write("-" * 30 + "\n")
            f.write(restaurants_text)
            f.write("\n\n")

            f.write("RAW RESTAURANT OBJECTS:\n")
            f.write("-" * 30 + "\n")
            for i, restaurant in enumerate(basic_restaurants, 1):
                f.write(f"{i}. {json.dumps(restaurant, indent=2)}\n")

            f.write("\n\nAI PROMPT TEMPLATE:\n")
            f.write("-" * 30 + "\n")
            # Get the template string from the prompt
            try:
                template_str = str(database_search_agent.batch_analysis_prompt)
                f.write(template_str)
            except:
                f.write("Could not extract template string")

        print(f"✅ AI input data exported to: {filename}")
        print(f"📊 Found {len(basic_restaurants)} restaurants in {destination}")
        print(f"🔍 Search intent: {search_intent}")

        return filename

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    export_ai_input()