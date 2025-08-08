# utils/supabase_storage.py - CORRECTED: Remove bucket creation attempt
"""
Enhanced Supabase Storage functionality for uploading scraped content files.
This handles uploading scraped content to Supabase Storage bucket and includes
new methods for the Supabase Manager integration.

CORRECTED: Removed bucket creation that was causing RLS policy violations
"""

import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class SupabaseStorageManager:
    """Enhanced Supabase Storage Manager with additional methods for bucket file handling"""

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase client for storage operations"""
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            self.bucket_name = "scraped-content"  # Default bucket name
            self._check_bucket_access()  # CHANGED: Just check access, don't try to create
            logger.info("✅ Enhanced Supabase Storage Manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase Storage: {e}")
            raise

    def _check_bucket_access(self):
        """
        CORRECTED: Just check if we can access the bucket (don't try to create it)
        The bucket should already exist
        """
        try:
            # Try to list files in the bucket to verify access
            test_files = self.client.storage.from_(self.bucket_name).list()
            logger.info(f"✅ Successfully accessed existing bucket: {self.bucket_name}")
            logger.info(f"Files in bucket: {len(test_files)}")
        except Exception as access_error:
            logger.error(f"❌ Cannot access bucket {self.bucket_name}: {access_error}")
            logger.error("🔧 Check your storage policies in Supabase Dashboard")
            # Still raise error so initialization fails if bucket not accessible
            raise

    def upload_scraped_content(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        file_type: str = "txt"
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload scraped content to Supabase Storage

        Args:
            content: The scraped text content
            metadata: Contains city, country, sources, query, etc.
            file_type: File extension (txt or json)

        Returns:
            Tuple of (success: bool, file_path: Optional[str])
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            city = metadata.get('city', 'unknown').replace(' ', '_').lower()

            # Organize by year/month/city for easy management
            year_month = datetime.now().strftime("%Y/%m")
            file_path = f"{year_month}/{city}/scraped_{timestamp}_{content_hash}.{file_type}"

            # Prepare content for upload
            if file_type == "json":
                upload_content = content if isinstance(content, str) else str(content)
            else:
                upload_content = content

            # Upload to Supabase Storage
            logger.info(f"📤 Uploading content to: {file_path}")

            response = self.client.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=upload_content.encode('utf-8'),
                file_options={
                    "cache-control": "3600",
                    "upsert": "true"  # Allow overwriting if file exists
                }
            )

            logger.info(f"✅ Successfully uploaded: {file_path}")
            return True, file_path

        except Exception as e:
            logger.error(f"❌ Failed to upload content: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return False, None

    def get_download_url(self, file_path: str, expires_in: int = 3600) -> Optional[str]:
        """
        Create a signed download URL for a file in the bucket

        Args:
            file_path: Path to file in bucket
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            Signed download URL or None if failed
        """
        try:
            response = self.client.storage.from_(self.bucket_name).create_signed_urls(
                [file_path],  # Pass as list
                expires_in
            )

            if response and len(response) > 0 and 'signedURL' in response[0]:
                download_url = response[0]['signedURL']
                logger.info(f"🔗 Created signed URL for: {file_path}")
                return download_url
            else:
                logger.error(f"❌ Failed to create signed URL: {response}")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to create signed URL for {file_path}: {e}")
            return None

    def download_file_content(self, file_path: str) -> Optional[str]:
        """
        Download content directly from bucket

        Args:
            file_path: Path to file in bucket

        Returns:
            File content as string or None if failed
        """
        try:
            response = self.client.storage.from_(self.bucket_name).download(file_path)

            if response:
                content = response.decode('utf-8')
                logger.info(f"⬇️ Downloaded content from: {file_path} ({len(content)} chars)")
                return content
            else:
                logger.error(f"❌ No content received for: {file_path}")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to download {file_path}: {e}")
            return None

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from the bucket

        Args:
            file_path: Path to file in bucket

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.storage.from_(self.bucket_name).remove([file_path])
            logger.info(f"🗑️ Deleted file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete {file_path}: {e}")
            return False

    def move_file(self, old_path: str, new_path: str) -> bool:
        """
        Move/rename a file within the bucket

        Args:
            old_path: Current file path
            new_path: New file path

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.storage.from_(self.bucket_name).move(old_path, new_path)
            logger.info(f"📁 Moved file: {old_path} → {new_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to move {old_path} to {new_path}: {e}")
            return False

    def list_files(self, prefix: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """
        List files in the storage bucket

        Args:
            prefix: Filter files by prefix (e.g., "2024/01/paris/")
            limit: Maximum number of files to return

        Returns:
            List of file objects
        """
        try:
            files = self.client.storage.from_(self.bucket_name).list(
                path=prefix,
                options={
                    "limit": limit,
                    "offset": 0,
                    "sortBy": {
                        "column": "updated_at",
                        "order": "desc"
                    }
                }
            )
            return files
        except Exception as e:
            logger.error(f"❌ Failed to list files: {e}")
            return []

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific file

        Args:
            file_path: Path to file in bucket

        Returns:
            File info dictionary or None if not found
        """
        try:
            # Get parent directory
            path_parts = file_path.split('/')
            if len(path_parts) > 1:
                parent_path = '/'.join(path_parts[:-1])
                filename = path_parts[-1]
            else:
                parent_path = ""
                filename = file_path

            # List files in parent directory
            files = self.client.storage.from_(self.bucket_name).list(
                path=parent_path,
                options={"limit": 1000}
            )

            # Find the specific file
            for file_info in files:
                if file_info.get('name') == filename:
                    return file_info

            return None

        except Exception as e:
            logger.error(f"❌ Failed to get file info for {file_path}: {e}")
            return None

    def get_bucket_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the bucket

        Returns:
            Dictionary with bucket statistics
        """
        try:
            # Get all files (limited sample)
            files = self.list_files(limit=1000)

            stats = {
                "total_files": len(files),
                "bucket_name": self.bucket_name,
                "file_types": {},
                "folders": {},
                "total_size_estimate": 0
            }

            for file_info in files:
                name = file_info.get('name', '')
                size = file_info.get('metadata', {}).get('size', 0)

                # Count file types
                if '.' in name:
                    ext = name.split('.')[-1].lower()
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1

                # Count folders
                if '/' in name:
                    folder = name.split('/')[0]
                    stats["folders"][folder] = stats["folders"].get(folder, 0) + 1

                # Estimate total size
                stats["total_size_estimate"] += size

            logger.info(f"📊 Bucket stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"❌ Failed to get bucket stats: {e}")
            return {"error": str(e)}


# Singleton instance management
_storage_manager = None

def initialize_storage_manager(supabase_url: str, supabase_key: str):
    """Initialize the global storage manager instance"""
    global _storage_manager
    _storage_manager = SupabaseStorageManager(supabase_url, supabase_key)
    return _storage_manager

def get_storage_manager() -> Optional[SupabaseStorageManager]:
    """Get the global storage manager instance"""
    return _storage_manager

# Utility functions for easy access
def upload_content_to_bucket(content: str, metadata: Dict[str, Any], file_type: str = "txt") -> Tuple[bool, Optional[str]]:
    """Upload content to bucket using global storage manager"""
    storage_manager = get_storage_manager()
    if storage_manager:
        return storage_manager.upload_scraped_content(content, metadata, file_type)
    return False, None

def download_content_from_bucket(file_path: str) -> Optional[str]:
    """Download content from bucket using global storage manager"""
    storage_manager = get_storage_manager()
    if storage_manager:
        return storage_manager.download_file_content(file_path)
    return None