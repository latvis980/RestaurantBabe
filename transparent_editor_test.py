# transparent_editor_test.py
"""
Transparent Editor Test - Uses existing production pipeline without modification

This test:
1. Uses your existing orchestrator pipeline unchanged
2. Shows complete pipeline results 
3. Analyzes what the real production system produces
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TransparentEditorTest:
    """
    Test production pipeline with transparency
    """

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.admin_chat_id = getattr(config, 'ADMIN_CHAT_ID', None)

        logger.info("✅ Transparent Editor Test initialized with production orchestrator")

    async def test_editor_pipeline(self, restaurant_query: str, bot=None) -> str:
        """
        Test complete production pipeline 
        """
        total_start_time = time.time()

        logger.info(f"🧪 Testing Production Pipeline: {restaurant_query}")

        try:
            # =================================================================
            # USE ACTUAL ORCHESTRATOR PIPELINE
            # =================================================================
            logger.info("🔄 Running actual orchestrator pipeline")

            # Run the real production pipeline
            result = self.orchestrator.process_query(restaurant_query)

            total_time = time.time() - total_start_time

            # Extract data from the real pipeline result
            restaurants = result.get('restaurants', [])

            timing_data = {
                'total_time': total_time,
                'final_restaurants_count': len(restaurants)
            }

            # Send analysis to Telegram
            if bot and self.admin_chat_id:
                try:
                    # Create analysis file with real pipeline data
                    analysis_file = self._create_pipeline_analysis_file(restaurant_query, result, timing_data)

                    # Send file using working pattern from other tests
                    with open(analysis_file, 'rb') as f:
                        bot.send_document(self.admin_chat_id, f, caption=f"🔍 Pipeline analysis: {restaurant_query}")

                    # Clean up
                    os.remove(analysis_file)

                    logger.info("✅ Successfully sent pipeline analysis to Telegram")
                except Exception as e:
                    logger.error(f"❌ Failed to send files: {e}")

            return "Analysis sent to Telegram"

        except Exception as e:
            logger.error(f"❌ Error in pipeline test: {e}")

            # Send error to Telegram if possible
            if bot and self.admin_chat_id:
                error_msg = (
                    f"❌ <b>Pipeline Test Failed</b>\n\n"
                    f"📝 Query: <code>{restaurant_query}</code>\n"
                    f"🔥 Error: <code>{str(e)}</code>\n"
                    f"⏱ Time: {datetime.now().isoformat()}"
                )
                try:
                    bot.send_message(self.admin_chat_id, error_msg, parse_mode='HTML')
                except:
                    pass

            return f"Error: {str(e)}"

    def _create_pipeline_analysis_file(self, query: str, pipeline_result: Dict[str, Any], timing_data: Dict[str, Any]) -> str:
        """Create analysis file from real pipeline result"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"/tmp/pipeline_analysis_{timestamp}.txt"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PRODUCTION PIPELINE ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Pipeline: ACTUAL ORCHESTRATOR (unchanged)\n")
            f.write("=" * 80 + "\n\n")

            # Extract restaurants from result
            restaurants = pipeline_result.get('restaurants', [])
            content_source = pipeline_result.get('content_source', 'unknown')

            f.write("PIPELINE RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Restaurants Found: {len(restaurants)}\n")
            f.write(f"Content Source: {content_source}\n")
            f.write(f"Processing Time: {timing_data['total_time']:.2f}s\n")
            f.write(f"Success: {pipeline_result.get('success', False)}\n\n")

            # Full restaurant details
            if restaurants:
                f.write("EXTRACTED RESTAURANTS (FULL DETAILS)\n")
                f.write("=" * 50 + "\n")
                for i, restaurant in enumerate(restaurants, 1):
                    f.write(f"\n{i}. {restaurant.get('name', 'Unknown')}\n")
                    f.write("-" * 30 + "\n")
                    for key, value in restaurant.items():
                        if key != 'name':
                            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                    f.write("\n")
            else:
                f.write("NO RESTAURANTS FOUND\n")
                f.write("=" * 30 + "\n")
                f.write("The pipeline did not return any restaurants.\n\n")

            # Full pipeline result
            f.write("COMPLETE PIPELINE OUTPUT\n")
            f.write("=" * 40 + "\n")
            f.write(json.dumps(pipeline_result, indent=2, ensure_ascii=False, default=str))
            f.write("\n\n")

            f.write("=" * 80 + "\n")
            f.write("ANALYSIS COMPLETE\n")
            f.write("=" * 80 + "\n")

        return filepath


# =================================================================
# TELEGRAM BOT INTEGRATION
# =================================================================

def create_transparent_editor_test(config, orchestrator):
    """Factory function for creating the test"""
    return TransparentEditorTest(config, orchestrator)