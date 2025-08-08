# utils/file_cleanup.py
"""
Automatic cleanup system for temporary files
Runs every 24 hours to prevent file accumulation
"""

import os
import time
import logging
import threading
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

class FileCleanupManager:
    """
    Manages automatic cleanup of temporary files

    Cleans up:
    - RTF files from human-mimic scraper
    - TXT files from text cleaner
    - Debug logs
    - Old scraped content
    """

    def __init__(self, config=None):
        self.config = config

        # Cleanup configuration
        self.cleanup_directories = [
            "scraped_content/",          # Main scraped content
            "debug_logs/",               # Debug logs
            "temp/",                     # Temporary files (if exists)
        ]

        # File age thresholds (in hours)
        self.file_age_thresholds = {
            "scraped_content/": 48,      # Keep scraped content for 2 days
            "debug_logs/": 72,           # Keep debug logs for 3 days  
            "temp/": 24,                 # Clean temp files after 1 day
        }

        # File patterns to clean
        self.cleanup_patterns = [
            "*.rtf",                     # RTF files from human-mimic
            "*.txt",                     # TXT files from text cleaner
            "cleaned_*.txt",             # Specific cleaned files
            "scraped_*.rtf",             # Specific scraped files
            "debug_*.txt",               # Debug output files
        ]

        # Stats tracking
        self.stats = {
            "last_cleanup": None,
            "files_cleaned": 0,
            "total_space_freed": 0,
            "cleanup_count": 0
        }

        # Threading
        self.cleanup_thread = None
        self.is_running = False

        logger.info("✅ File Cleanup Manager initialized")

    def start_cleanup_schedule(self):
        """Start the automatic cleanup schedule"""
        if self.is_running:
            logger.warning("⚠️ Cleanup schedule already running")
            return

        # Schedule cleanup every 24 hours
        schedule.every(24).hours.do(self._run_cleanup)

        # Also schedule a lighter cleanup every 6 hours for temp files only
        schedule.every(6).hours.do(self._run_light_cleanup)

        # Start the schedule runner in a separate thread
        self.is_running = True
        self.cleanup_thread = threading.Thread(target=self._schedule_runner, daemon=True)
        self.cleanup_thread.start()

        # Run initial cleanup
        self._run_light_cleanup()

        logger.info("🧹 Automatic file cleanup started - runs every 24 hours")
        logger.info("🧹 Light cleanup (temp files) - runs every 6 hours")

    def stop_cleanup_schedule(self):
        """Stop the automatic cleanup schedule"""
        self.is_running = False
        schedule.clear()
        logger.info("🛑 File cleanup schedule stopped")

    def _schedule_runner(self):
        """Background thread that runs the scheduled cleanups"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ Error in cleanup schedule runner: {e}")
                time.sleep(60)

    def _run_cleanup(self):
        """Run full cleanup of all directories"""
        try:
            logger.info("🧹 Starting scheduled file cleanup...")

            total_files_cleaned = 0
            total_space_freed = 0

            for directory in self.cleanup_directories:
                if os.path.exists(directory):
                    files_cleaned, space_freed = self._clean_directory(
                        directory, 
                        self.file_age_thresholds.get(directory, 24)
                    )
                    total_files_cleaned += files_cleaned
                    total_space_freed += space_freed

            # Update stats
            self.stats["last_cleanup"] = datetime.now().isoformat()
            self.stats["files_cleaned"] += total_files_cleaned
            self.stats["total_space_freed"] += total_space_freed
            self.stats["cleanup_count"] += 1

            if total_files_cleaned > 0:
                logger.info(f"✅ Cleanup complete: {total_files_cleaned} files removed, {total_space_freed / 1024 / 1024:.1f} MB freed")
            else:
                logger.info("✅ Cleanup complete: No old files to remove")

        except Exception as e:
            logger.error(f"❌ Error during scheduled cleanup: {e}")

    def _run_light_cleanup(self):
        """Run light cleanup - only temp files and very old files"""
        try:
            logger.info("🧹 Running light cleanup (temp files only)...")

            # Only clean temp directory and very old files (>72 hours)
            temp_directories = ["temp/", "scraped_content/"]
            total_files_cleaned = 0

            for directory in temp_directories:
                if os.path.exists(directory):
                    # Use longer threshold for light cleanup
                    age_threshold = 72 if directory == "scraped_content/" else 6
                    files_cleaned, _ = self._clean_directory(directory, age_threshold)
                    total_files_cleaned += files_cleaned

            if total_files_cleaned > 0:
                logger.info(f"✅ Light cleanup complete: {total_files_cleaned} files removed")

        except Exception as e:
            logger.error(f"❌ Error during light cleanup: {e}")

    def _clean_directory(self, directory: str, age_threshold_hours: int) -> tuple:
        """Clean files in a specific directory older than threshold"""
        files_cleaned = 0
        space_freed = 0

        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return 0, 0

            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=age_threshold_hours)

            # Find old files matching cleanup patterns
            old_files = []

            for pattern in self.cleanup_patterns:
                for file_path in directory_path.rglob(pattern):
                    if file_path.is_file():
                        # Check file age
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            old_files.append(file_path)

            # Remove old files
            for file_path in old_files:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    files_cleaned += 1
                    space_freed += file_size
                    logger.debug(f"🗑️ Removed: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not remove {file_path}: {e}")

            if files_cleaned > 0:
                logger.info(f"🧹 {directory}: Removed {files_cleaned} files ({space_freed / 1024:.1f} KB)")

        except Exception as e:
            logger.error(f"❌ Error cleaning directory {directory}: {e}")

        return files_cleaned, space_freed

    def manual_cleanup(self, force: bool = False):
        """Manually trigger cleanup"""
        if force:
            logger.info("🧹 Manual cleanup (FORCE) - removing all temp files regardless of age")
            # Force cleanup with very short threshold
            for directory in self.cleanup_directories:
                if os.path.exists(directory):
                    self._clean_directory(directory, 0)  # Remove all files
        else:
            logger.info("🧹 Manual cleanup triggered")
            self._run_cleanup()

    def get_cleanup_stats(self) -> dict:
        """Get cleanup statistics"""
        return {
            **self.stats,
            "is_running": self.is_running,
            "directories_monitored": self.cleanup_directories,
            "patterns_cleaned": self.cleanup_patterns
        }

# Global cleanup manager instance
_cleanup_manager = None

def initialize_cleanup_manager(config=None):
    """Initialize the global cleanup manager"""
    global _cleanup_manager
    _cleanup_manager = FileCleanupManager(config)
    return _cleanup_manager

def get_cleanup_manager():
    """Get the global cleanup manager"""
    return _cleanup_manager

def start_automatic_cleanup(config=None):
    """Start automatic cleanup system"""
    cleanup_manager = initialize_cleanup_manager(config)
    cleanup_manager.start_cleanup_schedule()
    return cleanup_manager

# Add this to main.py initialization
def add_to_main_initialization():
    """
    Add this to your main.py after other initializations:

    # Initialize automatic file cleanup
    try:
        from utils.file_cleanup import start_automatic_cleanup
        cleanup_manager = start_automatic_cleanup(config)
        logger.info("✅ Automatic file cleanup initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize file cleanup: {e}")
        # Don't fail the entire app if cleanup fails
        pass
    """
    pass