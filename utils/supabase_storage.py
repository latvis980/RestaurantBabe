# utils/supabase_storage.py - UPDATED: Simplified storage structure
"""
Enhanced Supabase Storage functionality for uploading cleaned content files.
This handles uploading cleaned content to Supabase Storage bucket.

UPDATED: Simplified bucket structure - unprocessed/ and processed/ folders only
"""

import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class SupabaseStorageManager:
    """Enhanced Supabase Storage Manager with simplified bucket structure"""

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase client for storage operations"""
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            self.bucket_name = "scraped-content"  # Default bucket name
            self._check_bucket_access()
            logger.info("✅ Enhanced Supabase Storage Manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase Storage: {e}")
            raise

    def _check_bucket_access(self):
        """Check if we can access the bucket"""
        try:
            # Try to list files in the bucket to verify access
            test_files = self.client.storage.from_(self.bucket_name).list()
            logger.info(f"✅ Successfully accessed existing bucket: {self.bucket_name}")
            logger.info(f"Files in bucket: {len(test_files)}")
        except Exception as access_error:
            logger.error(f"❌ Cannot access bucket {self.bucket_name}: {access_error}")
            logger.error("🔧 Check your storage policies in Supabase Dashboard")
            raise

    def upload_scraped_content(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        file_type: str = "txt"
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload scraped content to Supabase Storage with simplified structure

        Args:
            content: The scraped text content
            metadata: Contains city, country, sources, query, etc.
            file_type: File extension (txt or json)

        Returns:
            Tuple of (success: bool, file_path: Optional[str])
        """
        try:
            # Generate unique filename with city and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            city = metadata.get('city', 'unknown').replace(' ', '_').lower()

            # SIMPLIFIED: Direct to unprocessed folder
            file_path = f"unprocessed/cleanedRB_{timestamp}_{city}_{content_hash}.{file_type}"

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
                logger.debug(f"🔗 Created download URL for: {file_path}")
                return download_url
            else:
                logger.error(f"❌ Failed to create download URL for: {file_path}")
                return None

        except Exception as e:
            logger.error(f"❌ Error creating download URL: {e}")
            return None

    def download_file_content(self, file_path: str) -> Optional[str]:
        """
        Download and return the content of a file from bucket

        Args:
            file_path: Path to file in bucket

        Returns:
            File content as string, or None if failed
        """
        try:
            response = self.client.storage.from_(self.bucket_name).download(file_path)

            if response:
                content = response.decode('utf-8')
                logger.debug(f"📥 Downloaded content from: {file_path}")
                return content
            else:
                logger.error(f"❌ Failed to download content from: {file_path}")
                return None

        except Exception as e:
            logger.error(f"❌ Error downloading file content: {e}")
            return None

    def move_file(self, old_path: str, new_path: str) -> bool:
        """
        Move a file within the bucket

        Args:
            old_path: Current file path
            new_path: New file path

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.storage.from_(self.bucket_name).move(old_path, new_path)

            if response:
                logger.info(f"📦 Moved file: {old_path} → {new_path}")
                return True
            else:
                logger.error(f"❌ Failed to move file: {old_path} → {new_path}")
                return False

        except Exception as e:
            logger.error(f"❌ Error moving file: {e}")
            return False

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

            if response:
                logger.info(f"🗑️ Deleted file: {file_path}")
                return True
            else:
                logger.error(f"❌ Failed to delete file: {file_path}")
                return False

        except Exception as e:
            logger.error(f"❌ Error deleting file: {e}")
            return False

    def list_files(self, folder_path: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """
        List files in a specific folder

        Args:
            folder_path: Folder to list (empty for root)
            limit: Maximum number of files to return

        Returns:
            List of file information dictionaries
        """
        try:
            files = self.client.storage.from_(self.bucket_name).list(
                path=folder_path,
                options={"limit": limit}
            )

            # Filter out folders (items with no 'metadata' are folders)
            file_list = [file for file in files if file.get('metadata') is not None]

            logger.debug(f"📂 Listed {len(file_list)} files in: {folder_path or 'root'}")
            return file_list

        except Exception as e:
            logger.error(f"❌ Error listing files: {e}")
            return []

    def get_bucket_stats(self) -> Dict[str, Any]:
        """
        Get simplified statistics about the bucket

        Returns:
            Dictionary with bucket statistics
        """
        try:
            # Get files from both main folders
            unprocessed_files = self.list_files("unprocessed")
            processed_files = self.list_files("processed")

            stats = {
                "total_files": len(unprocessed_files) + len(processed_files),
                "bucket_name": self.bucket_name,
                "unprocessed_count": len(unprocessed_files),
                "processed_count": len(processed_files),
                "file_types": {},
                "cities": {},
                "latest_unprocessed": [],
                "timestamp": datetime.now().isoformat()
            }

            # Analyze all files
            all_files = unprocessed_files + processed_files

            for file_info in all_files:
                file_path = file_info.get('name', '')
                size = file_info.get('metadata', {}).get('size', 0)

                # Count file types
                if '.' in file_path:
                    ext = file_path.split('.')[-1].lower()
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1

                # Extract city from filename (cleanedRB_timestamp_city_hash.ext)
                filename = file_path.split('/')[-1]  # Get just filename
                if filename.startswith('cleanedRB_') and filename.count('_') >= 3:
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        city = parts[2].replace('_', ' ').title()  # timestamp_city_hash
                        stats["cities"][city] = stats["cities"].get(city, 0) + 1

            # Get latest unprocessed files (sorted by name which includes timestamp)
            unprocessed_files.sort(key=lambda f: f.get('name', ''), reverse=True)
            stats["latest_unprocessed"] = [
                {
                    "name": f.get('name', ''),
                    "size": f.get('metadata', {}).get('size', 0),
                    "uploaded_at": f.get('metadata', {}).get('updated_at', '')
                }
                for f in unprocessed_files[:5]  # Latest 5
            ]

            logger.info(f"📊 Bucket stats: {stats['unprocessed_count']} unprocessed, {stats['processed_count']} processed")
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