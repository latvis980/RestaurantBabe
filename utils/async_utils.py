# utils/async_utils.py
import asyncio
from functools import wraps

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
            # If we're already in an event loop, we need to avoid nested run_until_complete calls
            if loop.is_running():
                # Create a future in the current loop
                fut = asyncio.ensure_future(func(*args, **kwargs), loop=loop)
                # We can't await here, so we'll return the future
                return fut
            else:
                # Use the existing loop
                return loop.run_until_complete(func(*args, **kwargs))
        except RuntimeError:
            # No running event loop, create a new one
            return asyncio.run(func(*args, **kwargs))
    return wrapper

async def wait_for_pending_tasks():
    """Wait for all tracked tasks to complete"""
    if _PENDING_TASKS:
        await asyncio.gather(*_PENDING_TASKS, return_exceptions=True)