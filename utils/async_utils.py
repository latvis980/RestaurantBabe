# utils/async_utils.py
import asyncio
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# For storing pending tasks
_PENDING_TASKS = set()

def track_async_task(coro):
    """Add a task to the tracked set"""
    try:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(coro)
        _PENDING_TASKS.add(task)
        task.add_done_callback(_PENDING_TASKS.discard)
        return task
    except RuntimeError:
        # No running event loop, run synchronously
        return asyncio.run(coro)

def sync_to_async(func):
    """Decorator to run async functions from sync code safely"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, create a new task
            if loop.is_running():
                fut = asyncio.ensure_future(func(*args, **kwargs), loop=loop)
                # Return the future, don't wait for it to complete here
                return fut
            else:
                # Use the existing loop
                return loop.run_until_complete(func(*args, **kwargs))
        except RuntimeError:
            # No running event loop, create a new one
            try:
                return asyncio.run(func(*args, **kwargs))
            except Exception as e:
                logger.error(f"Error running async function: {e}")
                # Fall back to synchronous execution if needed
                return None
    return wrapper

async def wait_for_pending_tasks():
    """Wait for all tracked tasks to complete"""
    if _PENDING_TASKS:
        try:
            await asyncio.gather(*_PENDING_TASKS, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error waiting for pending tasks: {e}")