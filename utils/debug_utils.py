# utils/debug_utils.py
import json
import inspect
import os
import time
from datetime import datetime

# Directory for debug logs
DEBUG_DIR = "debug_logs"

def ensure_debug_dir():
    """Ensure the debug directory exists"""
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)

def dump_chain_state(stage_name, state, error=None):
    """
    Dump the current state of the chain to a file for debugging

    Args:
        stage_name (str): Name of the current stage
        state (dict): Current state of the chain
        error (Exception, optional): Exception if there was an error
    """
    ensure_debug_dir()

    # Create a timestamp for the filename
    timestamp = int(time.time())
    filename = f"{DEBUG_DIR}/{timestamp}_{stage_name}.json"

    # Make a copy of the state to avoid modifying the original
    state_copy = {}

    # Try to extract the important parts of the state
    for key, value in state.items():
        try:
            # Skip large objects that can't be easily serialized
            if key in ["scraper", "model", "chain", "prompt"]:
                state_copy[key] = "SKIPPED_OBJECT"
                continue

            # For lists and dicts, just capture length and structure
            if isinstance(value, list):
                if len(value) > 0:
                    # Include first item as sample and length
                    sample = value[0]
                    if isinstance(sample, dict):
                        state_copy[key] = {
                            "length": len(value),
                            "sample_keys": list(sample.keys()),
                            "first_item": sample
                        }
                    else:
                        state_copy[key] = {
                            "length": len(value),
                            "type": str(type(sample)),
                            "first_item": str(sample)[:100]  # Truncate long strings
                        }
                else:
                    state_copy[key] = {"length": 0, "note": "Empty list"}
            elif isinstance(value, dict):
                if len(value) > 0:
                    # Include keys and selective values
                    state_copy[key] = {
                        "keys": list(value.keys()),
                        "values": {k: str(v)[:100] if isinstance(v, str) else v 
                                  for k, v in value.items() 
                                  if not isinstance(v, (dict, list)) and k not in ["scraper", "model", "chain", "prompt"]}
                    }

                    # For specific important keys, include their full structure
                    if key in ["recommendations", "formatted_recommendations", "enhanced_recommendations"]:
                        state_copy[key] = value
                else:
                    state_copy[key] = {"length": 0, "note": "Empty dict"}
            else:
                # For primitive types, include directly
                state_copy[key] = str(value)[:100] if isinstance(value, str) else value
        except Exception as e:
            state_copy[key] = f"ERROR_SERIALIZING: {str(e)}"

    # Add caller information
    frame = inspect.currentframe().f_back
    caller_info = {
        "file": frame.f_code.co_filename,
        "function": frame.f_code.co_name,
        "line": frame.f_lineno,
        "timestamp": datetime.now().isoformat()
    }
    state_copy["_caller_info"] = caller_info

    # Add error information if provided
    if error:
        state_copy["_error"] = {
            "type": type(error).__name__,
            "message": str(error)
        }

    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(state_copy, f, ensure_ascii=False, indent=2, default=str)

    print(f"Dumped debug state for {stage_name} to {filename}")
    return filename

def log_function_call(func):
    """Decorator to log function calls with args and return values"""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print(f"CALLING {func_name} with {len(args)} args and {len(kwargs)} kwargs")

        try:
            result = func(*args, **kwargs)
            print(f"SUCCESS {func_name} returned {type(result)}")

            # For objects we care about, dump them
            if isinstance(result, dict) and any(k in result for k in ["recommendations", "formatted_recommendations", "enhanced_recommendations"]):
                dump_chain_state(f"{func_name}_return", {"result": result})

            return result
        except Exception as e:
            print(f"ERROR in {func_name}: {e}")
            dump_chain_state(f"{func_name}_error", {
                "args": [str(a)[:100] for a in args],
                "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}
            }, error=e)
            raise

    return wrapper