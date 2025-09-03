# utils/run_logger.py - Individual Run Logging for Testing
"""
Individual run logging system that saves each user query and its complete log to a separate txt file.
This creates a log file for each run (1 user's 1 query) in the temp folder.
"""

import os
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from io import StringIO

logger = logging.getLogger(__name__)

class RunLogger:
    """Captures logs for individual runs and saves them as txt files"""

    def __init__(self):
        self.temp_dir = "temp"
        self.ensure_temp_dir()
        self.current_run_data = {}
        self.log_capture = None
        self.original_handlers = []

    def ensure_temp_dir(self):
        """Create temp directory if it doesn't exist"""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            logger.info(f"✅ Temp directory ensured: {self.temp_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to create temp directory: {e}")
            raise

    def start_run_logging(self, user_query: str, user_id: Optional[str] = None, chat_id: Optional[str] = None) -> str:
        """
        Start logging for a new run

        Args:
            user_query: The user's query
            user_id: Telegram user ID (optional)
            chat_id: Telegram chat ID (optional)

        Returns:
            run_id: Unique identifier for this run
        """
        # Generate unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        run_id = f"run_{timestamp}"
        if user_id:
            run_id += f"_user{user_id}"

        # Store run metadata
        self.current_run_data = {
            "run_id": run_id,
            "user_query": user_query,
            "user_id": user_id,
            "chat_id": chat_id,
            "start_time": datetime.now().isoformat(),
            "log_entries": []
        }

        # Set up log capture
        self.setup_log_capture()

        logger.info(f"🚀 Started run logging for: {user_query[:50]}...")
        logger.info(f"📝 Run ID: {run_id}")

        return run_id

    def setup_log_capture(self):
        """Set up string buffer to capture log messages"""
        # Create string buffer for capturing logs
        self.log_capture = StringIO()

        # Create handler that writes to our buffer
        capture_handler = logging.StreamHandler(self.log_capture)
        capture_handler.setLevel(logging.INFO)

        # Use same format as main logging
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        capture_handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(capture_handler)

        # Store reference to remove later
        self.capture_handler = capture_handler

    def add_log_entry(self, level: str, message: str, logger_name: str = "manual"):
        """
        Manually add a log entry (for important events)

        Args:
            level: Log level (INFO, ERROR, WARNING, etc.)
            message: Log message
            logger_name: Name of the logger
        """
        if hasattr(self, 'current_run_data') and self.current_run_data:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            log_entry = f"{timestamp} - {logger_name} - {level} - {message}"
            self.current_run_data["log_entries"].append(log_entry)

    def finish_run_logging(self, success: bool = True, error_message: Optional[str] = None) -> Optional[str]:
        """
        Finish logging and save to file

        Args:
            success: Whether the run completed successfully
            error_message: Error message if run failed

        Returns:
            file_path: Path to the saved log file, or None if failed
        """
        if not hasattr(self, 'current_run_data') or not self.current_run_data:
            logger.warning("⚠️ No active run to finish logging")
            return None

        try:
            # Get captured logs
            captured_logs = ""
            if self.log_capture:
                captured_logs = self.log_capture.getvalue()

            # Remove the capture handler
            if hasattr(self, 'capture_handler'):
                root_logger = logging.getLogger()
                root_logger.removeHandler(self.capture_handler)

            # Update run data
            self.current_run_data.update({
                "end_time": datetime.now().isoformat(),
                "success": success,
                "error_message": error_message,
                "captured_logs": captured_logs
            })

            # Generate filename
            filename = f"{self.current_run_data['run_id']}.txt"
            file_path = os.path.join(self.temp_dir, filename)

            # Write to file
            self.write_log_file(file_path)

            logger.info(f"✅ Run log saved: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"❌ Failed to finish run logging: {e}")
            return None
        finally:
            # Clean up
            self.cleanup_capture()

    def write_log_file(self, file_path: str):
        """Write the complete log file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            # Header with query
            f.write("=" * 80 + "\n")
            f.write("RESTAURANT BABE - RUN LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Query: {self.current_run_data['user_query']}\n")
            f.write(f"Run ID: {self.current_run_data['run_id']}\n")
            f.write(f"Start Time: {self.current_run_data['start_time']}\n")
            f.write(f"End Time: {self.current_run_data['end_time']}\n")
            f.write(f"Success: {self.current_run_data['success']}\n")

            if self.current_run_data.get('user_id'):
                f.write(f"User ID: {self.current_run_data['user_id']}\n")
            if self.current_run_data.get('chat_id'):
                f.write(f"Chat ID: {self.current_run_data['chat_id']}\n")

            if self.current_run_data.get('error_message'):
                f.write(f"Error: {self.current_run_data['error_message']}\n")

            f.write("=" * 80 + "\n")
            f.write("FULL LOG\n")
            f.write("=" * 80 + "\n")

            # Write captured logs
            if self.current_run_data.get('captured_logs'):
                f.write(self.current_run_data['captured_logs'])

            # Write manual log entries if any
            if self.current_run_data.get('log_entries'):
                f.write("\n" + "-" * 40 + "\n")
                f.write("MANUAL LOG ENTRIES\n")
                f.write("-" * 40 + "\n")
                for entry in self.current_run_data['log_entries']:
                    f.write(entry + "\n")

    def cleanup_capture(self):
        """Clean up logging capture"""
        if hasattr(self, 'capture_handler'):
            try:
                root_logger = logging.getLogger()
                root_logger.removeHandler(self.capture_handler)
                delattr(self, 'capture_handler')
            except:
                pass

        if self.log_capture:
            try:
                self.log_capture.close()
                self.log_capture = None
            except:
                pass

        # Clear run data
        self.current_run_data = {}

# Global instance
_run_logger = None

def get_run_logger() -> RunLogger:
    """Get the global run logger instance"""
    global _run_logger
    if _run_logger is None:
        _run_logger = RunLogger()
    return _run_logger

def start_run_log(user_query: str, user_id: Optional[str] = None, chat_id: Optional[str] = None) -> str:
    """Convenience function to start run logging"""
    return get_run_logger().start_run_logging(user_query, user_id, chat_id)

def add_run_log(level: str, message: str, logger_name: str = "manual"):
    """Convenience function to add manual log entry"""
    get_run_logger().add_log_entry(level, message, logger_name)

def finish_run_log(success: bool = True, error_message: Optional[str] = None) -> Optional[str]:
    """Convenience function to finish run logging"""
    return get_run_logger().finish_run_logging(success, error_message)