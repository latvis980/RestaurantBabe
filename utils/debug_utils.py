# utils/debug_utils.py
import json
import inspect
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Directory for debug logs
DEBUG_DIR = "debug_logs"

def ensure_debug_dir():
    """Ensure the debug directory exists"""
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)
    return DEBUG_DIR

def dump_chain_state(stage_name: str, state: Dict[str, Any], error: Optional[Exception] = None) -> str:
    """
    Dump the current state of the chain to a file for debugging

    Updated for current pipeline structure with new state variables

    Args:
        stage_name (str): Name of the current stage
        state (dict): Current state of the chain
        error (Exception, optional): Exception if there was an error

    Returns:
        str: Path to the created debug file
    """
    ensure_debug_dir()

    # Create a timestamp for the filename
    timestamp = int(time.time())
    filename = f"{DEBUG_DIR}/{timestamp}_{stage_name}.json"

    # Make a copy of the state to avoid modifying the original
    state_copy = {}

    # Current pipeline state variables we track
    important_keys = [
        "query", "raw_query", "destination", "query_type", "search_terms", 
        "database_results", "content_evaluation_result", "search_results", 
        "scraped_results", "edited_results", "enhanced_results", "follow_up_queries",
        "final_recommendations", "pipeline_stages", "content_source", "scraper_stats"
    ]

    # Objects to skip (large/unserializable)
    skip_objects = [
        "scraper", "model", "chain", "prompt", "ai", "editor_agent", 
        "query_analyzer", "database_agent", "evaluation_agent", "model_manager",
        "orchestrator", "search_agent", "content_sectioner"
    ]

    # Try to extract the important parts of the state
    for key, value in state.items():
        try:
            # Skip large objects that can't be easily serialized
            if key in skip_objects:
                state_copy[key] = f"SKIPPED_{key.upper()}_OBJECT"
                continue

            # Handle different data types
            if isinstance(value, list):
                state_copy[key] = _serialize_list(key, value)
            elif isinstance(value, dict):
                state_copy[key] = _serialize_dict(key, value)
            elif isinstance(value, str):
                # Truncate very long strings but keep important ones full
                if key in important_keys or len(value) <= 500:
                    state_copy[key] = value
                else:
                    state_copy[key] = {
                        "type": "truncated_string",
                        "length": len(value),
                        "preview": value[:500] + "..." if len(value) > 500 else value
                    }
            else:
                # For primitive types, include directly
                state_copy[key] = value

        except Exception as e:
            state_copy[key] = f"ERROR_SERIALIZING_{key}: {str(e)}"

    # Add metadata
    state_copy["_debug_metadata"] = {
        "stage_name": stage_name,
        "timestamp": datetime.now().isoformat(),
        "caller_info": _get_caller_info(),
        "total_state_keys": len(state.keys()),
        "serialized_keys": len(state_copy.keys()) - 1  # -1 for metadata
    }

    # Add error information if provided
    if error:
        state_copy["_error_info"] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }

    # Write to file with proper encoding
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state_copy, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"🐛 Debug state dumped: {stage_name} → {filename}")
        return filename

    except Exception as e:
        logger.error(f"❌ Failed to dump debug state for {stage_name}: {e}")
        return ""

def _serialize_list(key: str, value: List[Any]) -> Dict[str, Any]:
    """Serialize list values for debugging"""
    if len(value) == 0:
        return {"type": "empty_list", "length": 0}

    # For important pipeline results, include more detail
    important_list_keys = [
        "search_results", "scraped_results", "database_results", 
        "follow_up_queries", "search_terms"
    ]

    if key in important_list_keys and len(value) <= 20:
        # Include full content for important small lists
        return {
            "type": "full_list",
            "length": len(value),
            "items": value
        }

    # For other lists, include sample and structure
    sample = value[0]
    result = {
        "type": "sampled_list",
        "length": len(value),
        "sample_type": str(type(sample)),
    }

    if isinstance(sample, dict):
        result["sample_keys"] = list(sample.keys())
        result["first_item"] = sample
    else:
        result["first_item"] = str(sample)[:200]

    return result

def _serialize_dict(key: str, value: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize dictionary values for debugging"""
    if len(value) == 0:
        return {"type": "empty_dict", "length": 0}

    # For important pipeline results, include full structure
    important_dict_keys = [
        "edited_results", "enhanced_results", "final_recommendations",
        "content_evaluation_result", "scraper_stats", "pipeline_stages"
    ]

    if key in important_dict_keys:
        try:
            # Include full structure for important dicts
            return value
        except:
            # Fallback if serialization fails
            pass

    # For other dicts, include structure info
    result = {
        "type": "dict_summary",
        "keys": list(value.keys()),
        "length": len(value)
    }

    # Include values for simple types
    simple_values = {}
    for k, v in value.items():
        if isinstance(v, (str, int, float, bool)) and k not in ["content", "text", "description"]:
            simple_values[k] = v
        elif isinstance(v, str) and len(v) <= 100:
            simple_values[k] = v

    if simple_values:
        result["simple_values"] = simple_values

    return result

def _get_caller_info() -> Dict[str, Any]:
    """Get information about the caller"""
    frame = inspect.currentframe().f_back.f_back  # Go up 2 frames
    return {
        "file": frame.f_code.co_filename,
        "function": frame.f_code.co_name,
        "line": frame.f_lineno
    }

def log_function_call(func):
    """Decorator to log function calls with args and return values - updated for current pipeline"""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"🔄 CALLING {func_name} with {len(args)} args and {len(kwargs)} kwargs")

        # Log important pipeline functions with more detail
        pipeline_functions = [
            "process_query", "_check_database_coverage", "_evaluate_content_routing",
            "_search_step", "_scrape_step", "_edit_step", "_follow_up_step", "_format_step"
        ]

        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            logger.info(f"✅ SUCCESS {func_name} completed in {duration:.2f}s, returned {type(result)}")

            # For pipeline functions, dump state for debugging
            if func_name in pipeline_functions and isinstance(result, dict):
                dump_chain_state(f"{func_name}_return", {"result": result})

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ ERROR in {func_name} after {duration:.2f}s: {e}")

            # Dump error state with input context
            error_state = {
                "function": func_name,
                "args_summary": [str(type(a)) for a in args],
                "kwargs_keys": list(kwargs.keys()),
                "duration": duration
            }

            # Try to include first arg if it looks like pipeline state
            if args and isinstance(args[0], dict):
                error_state["input_state_keys"] = list(args[0].keys())

            dump_chain_state(f"{func_name}_error", error_state, error=e)
            raise

    return wrapper

def log_pipeline_timing(stage_name: str):
    """Decorator to log timing for pipeline stages"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"⏱️  Starting {stage_name}")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"⏱️  Completed {stage_name} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"⏱️  Failed {stage_name} after {duration:.2f}s: {e}")
                raise

        return wrapper
    return decorator

def create_debug_summary(query: str, final_result: Dict[str, Any], processing_time: float) -> str:
    """Create a comprehensive debug summary for a complete query processing"""
    ensure_debug_dir()

    timestamp = int(time.time())
    filename = f"{DEBUG_DIR}/{timestamp}_query_summary.md"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Query Debug Summary\n\n")
            f.write(f"**Query:** {query}\n")
            f.write(f"**Processing Time:** {processing_time:.2f}s\n")
            f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")

            # Pipeline results summary
            f.write("## Pipeline Results\n\n")

            if "content_source" in final_result:
                f.write(f"**Content Source:** {final_result['content_source']}\n")

            if "enhanced_results" in final_result:
                results = final_result["enhanced_results"]
                if isinstance(results, dict) and "main_list" in results:
                    f.write(f"**Restaurants Found:** {len(results['main_list'])}\n")

            # Scraper statistics
            if "scraper_stats" in final_result:
                stats = final_result["scraper_stats"]
                f.write(f"\n### Scraper Statistics\n")
                f.write(f"- Total URLs Processed: {stats.get('total_processed', 0)}\n")
                f.write(f"- Cost Savings vs All Firecrawl: ${stats.get('cost_saved_vs_all_firecrawl', 0):.2f}\n")

                if "strategy_breakdown" in stats:
                    f.write(f"\n**Strategy Breakdown:**\n")
                    for strategy, count in stats["strategy_breakdown"].items():
                        if count > 0:
                            f.write(f"- {strategy}: {count} URLs\n")

            # Error information
            if "error" in final_result:
                f.write(f"\n### Error Information\n")
                f.write(f"**Error:** {final_result['error']}\n")

        logger.info(f"📊 Debug summary created: {filename}")
        return filename

    except Exception as e:
        logger.error(f"❌ Failed to create debug summary: {e}")
        return ""

def cleanup_old_debug_files(max_age_hours: int = 24):
    """Clean up old debug files to prevent disk space issues"""
    try:
        debug_path = Path(DEBUG_DIR)
        if not debug_path.exists():
            return

        cutoff_time = time.time() - (max_age_hours * 3600)
        deleted_count = 0

        for file_path in debug_path.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"🗑️  Cleaned up {deleted_count} old debug files")

    except Exception as e:
        logger.error(f"❌ Error cleaning up debug files: {e}")

# Initialize debug directory on import
ensure_debug_dir()