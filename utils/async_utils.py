# utils/async_utils.py  – universal helper for sync + async callables
import asyncio
import inspect
from functools import wraps, partial
import logging

logger = logging.getLogger(__name__)

_PENDING_TASKS: set[asyncio.Task] = set()

def track_async_task(coro):
    """
    Schedule *coro* in the current event‑loop and remember it so the
    app can await every outstanding task on shutdown.
    """
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:                       # no loop running
        return asyncio.run(coro)
    _PENDING_TASKS.add(task)
    task.add_done_callback(_PENDING_TASKS.discard)
    return task


async def wait_for_pending_tasks():
    """Await everything registered via `track_async_task()`."""
    if _PENDING_TASKS:
        try:
            await asyncio.gather(*_PENDING_TASKS, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error waiting for pending tasks: {e}")

# ------------------------------------------------------------------#
#  sync_to_async – the real fix                                      #
# ------------------------------------------------------------------#
def sync_to_async(func):
    """
    Turn **any** callable into an awaitable.

    • If *func* is already `async def` → just `await` it.  
    • Otherwise run it in the default ThreadPool so the event‑loop
      never blocks on CPU‑bound or I/O‑heavy work.

    Usage (no change needed in your code):

        result = await sync_to_async(my_func)(*args, **kw)
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Fast path for proper coroutine functions
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)

        # Everything else: run in executor
        loop = asyncio.get_running_loop()
        bound = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound)

    return async_wrapper
