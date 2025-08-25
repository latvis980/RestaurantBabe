# gunicorn.conf.py - ASYNC WORKERS WITH PROPER CONCURRENCY
import os
import multiprocessing

# Railway port binding
bind = f"0.0.0.0:{os.environ.get('PORT', 8080)}"

# ASYNC WORKERS - Much better for async operations
worker_class = "uvicorn.workers.UvicornWorker"

# Dynamic worker calculation based on CPU cores
# For Railway: Usually 1-2 vCPUs, so 2-4 workers
cpu_count = multiprocessing.cpu_count()
workers = max(2, min(cpu_count * 2, 4))  # 2-4 workers optimal for Railway

# Remove threads - async workers handle concurrency internally
# threads = removed - uvicorn worker doesn't use threads

# TIMEOUT CONFIGURATION
# Reduced from 600s to prevent long-running operations from stalling workers
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 120))  # 2 minutes (was 10!)
graceful_timeout = 60  # 1 minute graceful shutdown

# WORKER LIFECYCLE - Prevent memory leaks
max_requests = 500  # Restart workers after 500 requests
max_requests_jitter = 50  # Add randomness to prevent simultaneous restarts

# CONNECTION SETTINGS
keepalive = 5  # Shorter keepalive for better resource management

# PRELOADING - DISABLED for cleaner async initialization
preload_app = False  # This was causing service conflicts!

# LOGGING
accesslog = "-"
errorlog = "-"
loglevel = "info"

# RAILWAY OPTIMIZATIONS
max_header_size = 8192  # Reduced from 16384
worker_tmp_dir = "/dev/shm"

# ASYNC WORKER SPECIFIC SETTINGS
worker_connections = 100  # Concurrent connections per worker

# STARTUP HOOK - for proper async initialization
def on_starting(server):
    """Called when Gunicorn starts"""
    print(f"🚀 Starting RestaurantBabe with {workers} async workers")
    print(f"⚙️ Worker class: {worker_class}")
    print(f"⏱️ Timeout: {timeout}s (reduced for better reliability)")
    print(f"🔄 Max requests per worker: {max_requests}")
    print(f"📦 Preload app: {preload_app} (disabled for async compatibility)")

def on_worker_init(worker):
    """Called when each worker starts"""
    print(f"🔧 Worker {worker.pid} initialized")

# GRACEFUL SHUTDOWN
def on_exit(server):
    """Called when Gunicorn shuts down"""
    print("🛑 RestaurantBabe shutting down gracefully")

print(f"🚀 Gunicorn Config: {workers} async workers, {timeout}s timeout, no preloading")