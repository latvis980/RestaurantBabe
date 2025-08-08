# utils/supabase_manager_client.py - NEW: Client for sending content via buckets
"""
Client for integrating with the Supabase Manager via bucket storage.

This handles:
1. Uploading content to Supabase Storage bucket
2. Creating signed download URLs
3. Sending bucket URLs to Supabase Manager for processing
4. Managing file lifecycle in bucket
"""

import logging
import requests
from typing import Dict, Any, Tuple, Optional
from utils.supabase_storage import get_storage_manager

logger = logging.getLogger(__name__)

class SupabaseManagerClient:
    """
    Client for integrating main app with Supabase Manager via bucket storage
    """

    def __init__(self, manager_url: str):
        """
        Initialize client

        Args:
            manager_url: URL of the Supabase Manager service (e.g., Railway deployment)
        """
        self.manager_url = manager_url.rstrip('/')
        self.storage_manager = get_storage_manager()

        if not self.storage_manager:
            raise ValueError("Supabase Storage Manager not initialized. Call initialize_storage_manager() first.")

        logger.info(f"✅ Supabase Manager Client initialized for {manager_url}")

    def process_content_via_bucket(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        file_type: str = "txt"
    ) -> Dict[str, Any]:
        """
        Process scraped content via bucket storage

        Args:
            content: Scraped text content
            metadata: Content metadata (city, country, sources, etc.)
            file_type: File type (txt or json)

        Returns:
            Processing result from Supabase Manager
        """
        try:
            logger.info(f"📤 Processing content via bucket: {metadata.get('city', 'Unknown')}")

            # Step 1: Upload content to bucket
            success, bucket_file_path = self.storage_manager.upload_scraped_content(
                content=content,
                metadata=metadata,
                file_type=file_type
            )

            if not success or not bucket_file_path:
                return {"success": False, "error": "Failed to upload content to bucket"}

            logger.info(f"✅ Uploaded to bucket: {bucket_file_path}")

            # Step 2: Create signed download URL (valid for 1 hour)
            download_url = self.storage_manager.get_download_url(
                file_path=bucket_file_path,
                expires_in=3600  # 1 hour
            )

            if not download_url:
                return {"success": False, "error": "Failed to create download URL"}

            logger.info("🔗 Created signed download URL")

            # Step 3: Send to Supabase Manager for processing
            enhanced_metadata = {
                **metadata,
                "bucket_file_path": bucket_file_path,
                "upload_timestamp": bucket_file_path.split('_')[1] if '_' in bucket_file_path else None
            }

            payload = {
                "download_url": download_url,
                "metadata": enhanced_metadata
            }

            response = requests.post(
                f"{self.manager_url}/process_bucket_content",
                json=payload,
                timeout=300  # 5 minute timeout for processing
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Manager processing completed: {result.get('restaurants_processed', 0)} restaurants")

                # Step 4: Move file to completed folder in bucket
                self._mark_file_as_processed(bucket_file_path, success=True)

                return result
            else:
                logger.error(f"❌ Manager processing failed: {response.status_code} - {response.text}")

                # Mark file as failed
                self._mark_file_as_processed(bucket_file_path, success=False)

                return {
                    "success": False, 
                    "error": f"Manager service error: {response.status_code}",
                    "details": response.text
                }

        except Exception as e:
            logger.error(f"❌ Error in bucket processing: {e}")

            # Try to mark file as failed if we have the path
            if 'bucket_file_path' in locals():
                try:
                    self._mark_file_as_processed(bucket_file_path, success=False)
                except:
                    pass

            return {"success": False, "error": str(e)}

    def process_content_direct(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process content directly (backward compatibility - no bucket)

        Args:
            content: Scraped text content
            metadata: Content metadata

        Returns:
            Processing result from Supabase Manager
        """
        try:
            logger.info(f"📤 Processing content directly: {metadata.get('city', 'Unknown')}")

            payload = {
                "content": content,
                "metadata": metadata
            }

            response = requests.post(
                f"{self.manager_url}/process_scraped_content",
                json=payload,
                timeout=300  # 5 minute timeout
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Direct processing completed: {result.get('restaurants_processed', 0)} restaurants")
                return result
            else:
                logger.error(f"❌ Direct processing failed: {response.status_code} - {response.text}")
                return {
                    "success": False, 
                    "error": f"Manager service error: {response.status_code}",
                    "details": response.text
                }

        except Exception as e:
            logger.error(f"❌ Error in direct processing: {e}")
            return {"success": False, "error": str(e)}

    def _mark_file_as_processed(self, bucket_file_path: str, success: bool = True) -> bool:
        """
        Mark a bucket file as processed by moving it to completed/failed folder

        Args:
            bucket_file_path: Path to file in bucket
            success: Whether processing was successful

        Returns:
            True if successfully moved, False otherwise
        """
        try:
            # Determine target folder
            target_folder = "completed" if success else "failed"
            status = "completed" if success else "failed"

            # Build new path
            path_parts = bucket_file_path.split('/')
            filename = path_parts[-1]

            # Create new path with status folder
            if len(path_parts) >= 3:
                # Organized structure: YYYY/MM/city/status/filename
                new_path = '/'.join(path_parts[:-1]) + f'/{target_folder}/{filename}'
            else:
                # Simple structure: status/filename
                new_path = f'{target_folder}/{filename}'

            # Move file in bucket
            move_success = self.storage_manager.move_file(bucket_file_path, new_path)

            if move_success:
                logger.info(f"📦 Moved to {status}: {bucket_file_path} → {new_path}")
            else:
                logger.warning(f"⚠️ Failed to move file to {status}: {bucket_file_path}")

            return move_success

        except Exception as e:
            logger.error(f"❌ Error marking file as processed: {e}")
            return False

    def get_bucket_stats(self) -> Dict[str, Any]:
        """
        Get statistics about files in the bucket

        Returns:
            Dictionary with file counts and stats
        """
        try:
            return self.storage_manager.get_bucket_stats()
        except Exception as e:
            logger.error(f"❌ Error getting bucket stats: {e}")
            return {"error": str(e)}

    def cleanup_old_files(self, max_age_days: int = 7) -> Dict[str, Any]:
        """
        Clean up old processed files in bucket

        Args:
            max_age_days: Files older than this will be deleted

        Returns:
            Cleanup statistics
        """
        try:
            # Get completed and failed files
            all_files = self.storage_manager.list_files(limit=1000)

            old_files = []
            cutoff_time = datetime.now() - timedelta(days=max_age_days)

            for file_info in all_files:
                file_path = file_info.get('name', '')

                # Only clean up files in completed/ or failed/ folders
                if '/completed/' in file_path or '/failed/' in file_path:
                    try:
                        if 'updated_at' in file_info:
                            file_time = datetime.fromisoformat(file_info['updated_at'].replace('Z', '+00:00'))
                            if file_time < cutoff_time.astimezone():
                                old_files.append(file_path)
                    except Exception:
                        continue

            # Delete old files
            if old_files:
                delete_result = self.storage_manager.delete_files(old_files)
                logger.info(f"🗑️ Cleaned up {delete_result.get('deleted_count', 0)} old files")
                return delete_result
            else:
                logger.info("📭 No old files to clean up")
                return {"deleted_count": 0, "message": "No old files found"}

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the Supabase Manager service is healthy

        Returns:
            Health status
        """
        try:
            response = requests.get(f"{self.manager_url}/", timeout=10)

            if response.status_code == 200:
                return {"healthy": True, "manager_response": response.json()}
            else:
                return {"healthy": False, "error": f"Status {response.status_code}"}

        except Exception as e:
            return {"healthy": False, "error": str(e)}


# Convenience functions for easy integration

def send_content_to_manager(
    content: str,
    metadata: Dict[str, Any],
    manager_url: str,
    use_bucket: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to send content to Supabase Manager

    Args:
        content: Scraped content
        metadata: Content metadata
        manager_url: URL of Supabase Manager service
        use_bucket: Whether to use bucket storage (recommended) or direct send

    Returns:
        Processing result
    """
    client = SupabaseManagerClient(manager_url)

    if use_bucket:
        return client.process_content_via_bucket(content, metadata)
    else:
        return client.process_content_direct(content, metadata)


# Integration with existing scraping code
def integrate_with_scraper(scraper_function, manager_url: str):
    """
    Decorator to automatically send scraped content to Supabase Manager

    Usage:
        @integrate_with_scraper("https://your-manager-url.railway.app")
        def scrape_restaurants(city, country):
            # Your scraping logic here
            return {"content": scraped_text, "metadata": {...}}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Run the original scraper
            result = func(*args, **kwargs)

            if isinstance(result, dict) and 'content' in result:
                # Send to manager
                processing_result = send_content_to_manager(
                    content=result['content'],
                    metadata=result.get('metadata', {}),
                    manager_url=manager_url,
                    use_bucket=True
                )

                # Return combined result
                return {
                    **result,
                    "processing_result": processing_result
                }

            return result
        return wrapper
    return decorator