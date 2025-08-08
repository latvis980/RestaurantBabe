# utils/supabase_manager_client.py - UPDATED: Simplified bucket integration
"""
Client for integrating with the Supabase Manager via bucket storage.

UPDATED: Works with simplified bucket structure (unprocessed/ and processed/ folders)

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
    Client for integrating main app with Supabase Manager via simplified bucket storage
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
        Process scraped content via simplified bucket storage

        Args:
            content: Scraped text content
            metadata: Content metadata (city, country, sources, etc.)
            file_type: File type (txt or json)

        Returns:
            Processing result from Supabase Manager
        """
        try:
            logger.info(f"📤 Processing content via bucket: {metadata.get('city', 'Unknown')}")

            # Step 1: Upload content to bucket (goes to unprocessed/ folder)
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
                "upload_timestamp": self._extract_timestamp_from_path(bucket_file_path)
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

                # Note: File will be moved to processed/ folder by the Supabase Manager
                return result
            else:
                logger.error(f"❌ Manager processing failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Manager returned status {response.status_code}",
                    "details": response.text
                }

        except Exception as e:
            logger.error(f"❌ Error in bucket processing: {e}")
            return {"success": False, "error": str(e)}

    def process_content_direct(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process content directly without bucket storage (legacy endpoint)

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
                f"{self.manager_url}/process_content",
                json=payload,
                timeout=300  # 5 minute timeout
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Direct processing completed: {result.get('restaurants_processed', 0)} restaurants")
                return result
            else:
                logger.error(f"❌ Direct processing failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Manager returned status {response.status_code}",
                    "details": response.text
                }

        except Exception as e:
            logger.error(f"❌ Error in direct processing: {e}")
            return {"success": False, "error": str(e)}

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

    def check_manager_health(self) -> Dict[str, Any]:
        """
        Check if the Supabase Manager service is healthy

        Returns:
            Health check result
        """
        try:
            response = requests.get(f"{self.manager_url}/", timeout=10)

            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Manager service is healthy")
                return {"healthy": True, "status": result}
            else:
                logger.warning(f"⚠️ Manager health check failed: {response.status_code}")
                return {"healthy": False, "status_code": response.status_code}

        except Exception as e:
            logger.error(f"❌ Error checking manager health: {e}")
            return {"healthy": False, "error": str(e)}

    def _extract_timestamp_from_path(self, file_path: str) -> Optional[str]:
        """
        Extract timestamp from simplified file path

        Args:
            file_path: Path like "unprocessed/scraped_20250808_143022_lisbon_abc123.txt"

        Returns:
            Timestamp string or None
        """
        try:
            filename = file_path.split('/')[-1]  # Get just filename

            if filename.startswith('scraped_') and filename.count('_') >= 3:
                parts = filename.split('_')
                if len(parts) >= 3:
                    # scraped_20250808_143022_city_hash.txt
                    date_part = parts[1]  # 20250808
                    time_part = parts[2]  # 143022
                    return f"{date_part}_{time_part}"
        except Exception:
            pass

        return None