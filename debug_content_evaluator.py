"""
Debug script to see exactly what the Enhanced Content Evaluator AI receives and how it selects restaurants.

This will:
1. Show the exact prompt sent to AI (enhanced version with selection)
2. Show the exact restaurant data format with descriptions
3. Test with actual data from database
4. Capture the AI's raw response including restaurant selection
5. Analyze selection logic and hybrid mode decisions
6. Show restaurant splitting into final vs hybrid categories
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any

def debug_enhanced_content_evaluator():
    """
    Debug the enhanced content evaluator to see restaurant selection and splitting logic
    """
    try:
        # Import the actual components
        import config
        from agents.query_analyzer import QueryAnalyzer
        from agents.database_search_agent import DatabaseSearchAgent
        from agents.dbcontent_evaluation_agent import ContentEvaluationAgent
        from utils.database import initialize_database

        print("🔧 Initializing debug environment...")

        # Initialize
        initialize_database(config)
        query_analyzer = QueryAnalyzer(config)
        database_search_agent = DatabaseSearchAgent(config)
        content_evaluator = ContentEvaluationAgent(config)

        # Use test query
        test_query = "famous loca chefs' restaurants in Olhao"

        print(f"🔍 Debug query: '{test_query}'")

        # STEP 1: Get the pipeline data (same as production)
        print("📋 Step 1: Getting pipeline data...")
        query_analysis = query_analyzer.analyze(test_query)
        database_result = database_search_agent.search_and_evaluate(query_analysis)

        database_restaurants = database_result.get("database_restaurants", [])
        destination = query_analysis.get("destination", "Unknown")

        print(f"   Found: {len(database_restaurants)} restaurants")
        print(f"   Destination: {destination}")

        # STEP 2: Show the exact restaurant data that AI sees for selection
        print("\n📄 Step 2: Analyzing enhanced restaurant data format...")

        if database_restaurants:
            sample_restaurant = database_restaurants[0]
            print("SAMPLE RESTAURANT STRUCTURE (Full Database Object):")
            print("-" * 60)
            print(f"ID: {sample_restaurant.get('id')}")
            print(f"Name: {sample_restaurant.get('name')}")
            print(f"Cuisine Tags: {sample_restaurant.get('cuisine_tags')}")
            print(f"Raw Description Length: {len(sample_restaurant.get('raw_description', ''))}")
            print(f"Raw Description Preview: {sample_restaurant.get('raw_description', '')[:200]}...")
            print(f"Sources: {len(sample_restaurant.get('sources', []))}")
            print(f"Mention Count: {sample_restaurant.get('mention_count')}")
            print("-" * 60)

        # Show formatted data for AI (enhanced format with descriptions)
        restaurants_data = content_evaluator._format_restaurants_for_selection(database_restaurants)

        print("\nENHANCED RESTAURANT DATA (as AI sees it for selection):")
        print("-" * 60)
        print(restaurants_data[:2000] + "..." if len(restaurants_data) > 2000 else restaurants_data)
        print("-" * 60)

        # STEP 3: Show the exact enhanced prompt that gets sent to AI
        print("\n🤖 Step 3: Showing enhanced AI prompt with selection...")

        # This is the actual enhanced prompt used in _evaluate_and_select_with_ai
        evaluation_prompt = f"""You are evaluating database restaurant results for a user query AND selecting the best matches.

USER QUERY: "{test_query}"
DESTINATION: {destination}
DATABASE RESULTS: {len(database_restaurants)} restaurants found

{restaurants_data}

**STAGE 1: EVALUATE**: Are these database results sufficient, or do we need web search?

CRITERIA:
1. Query Match: Do restaurants match what user wants? (cuisine, style, price range, etc.)
2. Quantity: Enough variety? (4+ = usually sufficient)
3. Quality: Meaningful details and relevance?

**STAGE 2: SELECT**: Choose the BEST matching restaurants from the list above.
- Analyze descriptions carefully, not just names and cuisine tags
- Only select restaurants that truly match the user's query
- Include relevance scores and reasoning for each selection

**STAGE 3: ROUTE** 
FOLLOW THIS LOGIC:
• Perfect matches + sufficient quantity (4+) → USE DATABASE
• Poor matches → TRIGGER WEB SEARCH (discard results)  
• Good matches but need variety → TRIGGER WEB SEARCH (preserve matching results for hybrid)
• Partial matches → TRIGGER WEB SEARCH (preserve matching results for hybrid)
• No results → TRIGGER WEB SEARCH (discard)

HYBRID MODE (preserve + supplement):
In reasoning, use phrases like: "good matches but need more variety", "limited options supplement", "relevant results but too few"

DISCARD MODE (start fresh):
In reasoning, use phrases like: "poor matches for the query", "doesn't match requirements", "completely irrelevant"

Return ONLY JSON:
{{
    "database_sufficient": true/false,
    "trigger_web_search": true/false, 
    "reasoning": "brief explanation",
    "quality_score": 0.8,
    "selected_restaurants": [
        {{
            "id": "restaurant_id",
            "relevance_score": 0.9,
            "match_reasoning": "why this restaurant matches the query"
        }}
    ]
}}

IMPORTANT: Use exact restaurant IDs from the data above. Only select restaurants that match the user's query."""

        print("ENHANCED AI PROMPT:")
        print("-" * 60)
        print(evaluation_prompt)
        print("-" * 60)

        # STEP 4: Make the actual AI call and capture enhanced response
        print("\n🧠 Step 4: Making enhanced AI evaluation + selection call...")

        try:
            # Call the enhanced AI evaluation method directly
            evaluation_result = content_evaluator._evaluate_and_select_with_ai(
                database_restaurants, 
                test_query, 
                destination
            )

            print("ENHANCED AI EVALUATION + SELECTION RESULT:")
            print("-" * 60)
            print(json.dumps(evaluation_result, indent=2))
            print("-" * 60)

            # STEP 5: Test restaurant mapping and splitting
            print("\n🎯 Step 5: Testing restaurant mapping and splitting...")

            selected_restaurants_data = evaluation_result.get('selected_restaurants', [])
            selected_restaurants = content_evaluator._map_selected_restaurants(
                selected_restaurants_data, 
                database_restaurants
            )

            print(f"Selected Restaurants: {len(selected_restaurants)}")
            for i, restaurant in enumerate(selected_restaurants, 1):
                print(f"  {i}. {restaurant.get('name')} (ID: {restaurant.get('id')})")
                print(f"     Relevance Score: {restaurant.get('_relevance_score')}")
                print(f"     AI Reasoning: {restaurant.get('_match_reasoning')}")
                print(f"     Description Length: {len(restaurant.get('raw_description', ''))}")

            # Test hybrid mode decision
            database_sufficient = evaluation_result.get('database_sufficient', False)
            reasoning = evaluation_result.get('reasoning', '')

            if not database_sufficient:
                use_hybrid = content_evaluator._should_use_hybrid_mode(selected_restaurants, reasoning)
                print(f"\nHybrid Mode Decision: {use_hybrid}")
                print(f"Reasoning: {reasoning}")

                if use_hybrid:
                    print("🔄 Would preserve restaurants for hybrid mode")
                else:
                    print("🗑️ Would discard restaurants and start fresh")

            # STEP 6: Enhanced analysis
            print("\n📊 Step 6: Enhanced Decision Analysis...")

            database_sufficient = evaluation_result.get('database_sufficient', False)
            trigger_web_search = evaluation_result.get('trigger_web_search', True)
            reasoning = evaluation_result.get('reasoning', 'No reasoning')
            quality_score = evaluation_result.get('quality_score', 0.0)
            selected_count = len(selected_restaurants_data)

            print(f"Database Sufficient: {database_sufficient}")
            print(f"Trigger Web Search: {trigger_web_search}")
            print(f"Quality Score: {quality_score}")
            print(f"Selected Count: {selected_count}/{len(database_restaurants)}")
            print(f"AI Reasoning: {reasoning}")

            # STEP 7: Manual analysis vs AI selection
            print("\n🔍 Step 7: Manual Analysis vs AI Selection...")

            brunch_matches = []
            for restaurant in database_restaurants:
                name = restaurant.get('name', '')
                cuisine_tags = restaurant.get('cuisine_tags', [])
                description = restaurant.get('raw_description', '')

                # Check for brunch indicators
                brunch_indicators = []
                if any('brunch' in tag.lower() for tag in cuisine_tags):
                    brunch_indicators.append("has 'brunch' cuisine tag")
                if 'brunch' in description.lower():
                    brunch_indicators.append("mentions 'brunch' in description")
                if any('breakfast' in tag.lower() for tag in cuisine_tags):
                    brunch_indicators.append("has 'breakfast' cuisine tag")

                if brunch_indicators:
                    brunch_matches.append({
                        'id': restaurant.get('id'),
                        'name': name,
                        'indicators': brunch_indicators,
                        'cuisine_tags': cuisine_tags,
                        'ai_selected': restaurant.get('id') in [str(s.get('id')) for s in selected_restaurants_data]
                    })

            print(f"MANUAL BRUNCH ANALYSIS:")
            print(f"Clear brunch matches found: {len(brunch_matches)}")

            for i, match in enumerate(brunch_matches, 1):
                ai_selected = "✅ AI Selected" if match['ai_selected'] else "❌ AI Missed"
                print(f"  {i}. {match['name']} - {ai_selected}")
                print(f"     ID: {match['id']}")
                print(f"     Indicators: {', '.join(match['indicators'])}")
                print(f"     Tags: {match['cuisine_tags']}")
                print()

            # STEP 8: Problem identification with enhanced analysis
            print("🎯 Step 8: Enhanced Problem Identification...")

            manual_matches = len(brunch_matches)
            ai_selected = len(selected_restaurants)
            ai_missed = len([m for m in brunch_matches if not m['ai_selected']])

            if manual_matches >= 3 and not database_sufficient:
                print("❌ POTENTIAL ISSUES IDENTIFIED:")
                print(f"   - Manual analysis found {manual_matches} clear brunch matches")
                print(f"   - AI selected only {ai_selected} restaurants")
                print(f"   - AI missed {ai_missed} obvious matches")
                print(f"   - Query '{test_query}' clearly matches available data")
                print(f"   - AI reasoning: '{reasoning}'")
                print("\n💡 POSSIBLE CAUSES:")
                print("   1. AI prompt may be too strict about quality requirements")
                print("   2. Restaurant description format may be confusing AI")
                print("   3. AI may be over-weighting description content vs cuisine tags")
                print("   4. Selection criteria may be too narrow")
            elif ai_selected > 0 and not database_sufficient:
                print("⚠️  AI FOUND MATCHES BUT WANTS WEB SEARCH:")
                print(f"   - AI selected {ai_selected} restaurants as good matches")
                print(f"   - But still triggered web search for more variety")
                print(f"   - This could be correct behavior for insufficient quantity")
            elif ai_selected == 0:
                print("❌ AI FOUND NO MATCHES:")
                print(f"   - Manual analysis found {manual_matches} matches")
                print(f"   - AI selection criteria may be too strict")
            else:
                print("✅ AI decision appears reasonable")

        except Exception as ai_error:
            print(f"❌ Enhanced AI evaluation failed: {ai_error}")
            import traceback
            traceback.print_exc()

        # STEP 9: Save enhanced debug output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_content_evaluator_debug_{timestamp}.txt"

        # Create debug_logs folder if it doesn't exist
        debug_logs_dir = os.path.join(os.getcwd(), "debug_logs")
        os.makedirs(debug_logs_dir, exist_ok=True)
        filepath = os.path.join(debug_logs_dir, filename)

        print(f"\n💾 Step 9: Saving enhanced debug output to debug_logs/{filename}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ENHANCED CONTENT EVALUATOR DEBUG ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Query: {test_query}\n")
            f.write(f"Restaurants Found: {len(database_restaurants)}\n")
            f.write(f"Destination: {destination}\n\n")

            f.write("ENHANCED RESTAURANT DATA (AI Input for Selection):\n")
            f.write("-" * 60 + "\n")
            f.write(restaurants_data)
            f.write("\n" + "-" * 60 + "\n\n")

            f.write("ENHANCED AI PROMPT (Evaluation + Selection):\n")
            f.write("-" * 60 + "\n")
            f.write(evaluation_prompt)
            f.write("\n" + "-" * 60 + "\n\n")

            if 'evaluation_result' in locals():
                f.write("ENHANCED AI EVALUATION + SELECTION RESULT:\n")
                f.write("-" * 60 + "\n")
                f.write(json.dumps(evaluation_result, indent=2))
                f.write("\n" + "-" * 60 + "\n\n")

                f.write("SELECTED RESTAURANTS (Full Objects):\n")
                f.write("-" * 60 + "\n")
                for i, restaurant in enumerate(selected_restaurants, 1):
                    f.write(f"{i}. {restaurant.get('name')} (ID: {restaurant.get('id')})\n")
                    f.write(f"   Relevance Score: {restaurant.get('_relevance_score')}\n")
                    f.write(f"   AI Reasoning: {restaurant.get('_match_reasoning')}\n")
                    f.write(f"   Cuisine Tags: {restaurant.get('cuisine_tags')}\n")
                    f.write(f"   Description Length: {len(restaurant.get('raw_description', ''))}\n")
                    f.write(f"   Sources: {len(restaurant.get('sources', []))}\n\n")

            f.write("MANUAL VS AI ANALYSIS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Manual brunch matches: {len(brunch_matches) if 'brunch_matches' in locals() else 'N/A'}\n")
            f.write(f"AI selected: {len(selected_restaurants) if 'selected_restaurants' in locals() else 'N/A'}\n")

            if 'brunch_matches' in locals():
                f.write("\nDetailed Comparison:\n")
                for i, match in enumerate(brunch_matches, 1):
                    ai_selected = "✅ AI Selected" if match['ai_selected'] else "❌ AI Missed"
                    f.write(f"{i}. {match['name']} - {ai_selected}\n")
                    f.write(f"   ID: {match['id']}\n")
                    f.write(f"   Indicators: {', '.join(match['indicators'])}\n")
                    f.write(f"   Tags: {match['cuisine_tags']}\n\n")

            # Enhanced splitting analysis
            if 'selected_restaurants' in locals() and 'evaluation_result' in locals():
                f.write("RESTAURANT SPLITTING ANALYSIS:\n")
                f.write("-" * 60 + "\n")

                database_sufficient = evaluation_result.get('database_sufficient', False)
                reasoning = evaluation_result.get('reasoning', '')

                if database_sufficient:
                    f.write("ROUTE: DATABASE ONLY\n")
                    f.write(f"database_restaurants_final: {len(selected_restaurants)} restaurants\n")
                    f.write(f"database_restaurants_hybrid: 0 restaurants\n")
                else:
                    use_hybrid = content_evaluator._should_use_hybrid_mode(selected_restaurants, reasoning)
                    f.write("ROUTE: WEB SEARCH\n")
                    if use_hybrid:
                        f.write(f"database_restaurants_final: 0 restaurants\n")
                        f.write(f"database_restaurants_hybrid: {len(selected_restaurants)} restaurants (PRESERVED)\n")
                        f.write(f"Hybrid Mode: ✅ ENABLED\n")
                    else:
                        f.write(f"database_restaurants_final: 0 restaurants\n")
                        f.write(f"database_restaurants_hybrid: 0 restaurants (DISCARDED)\n")
                        f.write(f"Hybrid Mode: ❌ DISABLED\n")

                    f.write(f"Hybrid Decision Reasoning: {reasoning}\n")

        return filepath

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🐛 ENHANCED CONTENT EVALUATOR DEBUG TOOL")
    print("=" * 50)
    debug_enhanced_content_evaluator()