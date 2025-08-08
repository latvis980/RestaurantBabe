# utils/supabase_storage.py - Enhanced for Supabase Manager Integration
"""
Enhanced Supabase Storage functionality for uploading scraped content files.
This handles uploading scraped content to Supabase Storage bucket and includes
new methods for the Supabase Manager integration.
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
            self._ensure_bucket_exists()
            logger.info("✅ Enhanced Supabase Storage Manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase Storage: {e}")
            raise

    def _ensure_bucket_exists(self):
        """Ensure the scraped-content bucket exists"""
        try:
            # Try to get bucket info
            buckets = self.client.storage.list_buckets()
            bucket_exists = any(b['name'] == self.bucket_name for b in buckets)

            if not bucket_exists:
                # Create bucket with correct Supabase Python client format
                response = self.client.storage.create_bucket(
                    self.bucket_name,  # Bucket name as first positional argument
                    options={
                        "public": False,  # Keep scraped content private
                        "file_size_limit": 10485760,  # 10MB limit
                        "allowed_mime_types": ["text/plain", "application/json", "text/csv"]
                    }
                )
                logger.info(f"✅ Created bucket: {self.bucket_name}")
                logger.info(f"Create bucket response: {response}")
            else:
                logger.info(f"✅ Bucket exists: {self.bucket_name}")

        except Exception as e:
            logger.error(f"❌ Failed to ensure bucket exists: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            # Since bucket might exist manually, let's just check if we can access it
            try:
                # Try to list files in the bucket to verify access
                test_files = self.client.storage.from_(self.bucket_name).list()
                logger.info(f"✅ Successfully accessed existing bucket: {self.bucket_name}")
            except Exception as access_error:
                logger.error(f"❌ Cannot access bucket {self.bucket_name}: {access_error}")

            # Don't fail initialization, just log the error
            logger.warning("⚠️ Continuing without bucket creation - bucket should exist manually")

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

            # Prepare file content based on type
            if file_type == "json":
                # Store as structured JSON with metadata
                file_data = {
                    "content": content,
                    "metadata": metadata,
                    "uploaded_at": datetime.now().isoformat(),
                    "content_hash": content_hash
                }
                file_content = json.dumps(file_data, indent=2, ensure_ascii=False)
                content_type = "application/json"
            else:
                # Store as plain text
                file_content = content
                content_type = "text/plain"

            # Convert to bytes
            content_bytes = file_content.encode('utf-8')

            # Upload to Supabase Storage
            response = self.client.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=content_bytes,
                file_options={
                    "content-type": content_type,
                    "x-upsert": "true"  # Allow overwrite if exists
                }
            )

            # Log metadata
            file_metadata = {
                "uploaded_at": datetime.now().isoformat(),
                "city": metadata.get('city'),
                "country": metadata.get('country'),
                "query": metadata.get('query'),
                "sources": metadata.get('sources', []),
                "content_length": len(content),
                "content_hash": content_hash
            }

            logger.info(f"✅ Uploaded to Supabase Storage: {file_path}")
            logger.info(f"📊 Metadata: {file_metadata}")

            return True, file_path

        except Exception as e:
            logger.error(f"❌ Failed to upload to Supabase Storage: {e}")
            return False, None

    def download_file_content(self, file_path: str) -> Optional[str]:
        """
        Download and return content of a file from the bucket

        Args:
            file_path: Path to file in bucket

        Returns:
            File content as string, or None if failed
        """
        try:
            response = self.client.storage.from_(self.bucket_name).download(file_path)

            if response:
                content = response.decode('utf-8')
                logger.info(f"📥 Downloaded file: {file_path} ({len(content)} chars)")
                return content
            else:
                logger.warning(f"⚠️ No content found for file: {file_path}")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to download file {file_path}: {e}")
            return None

    def copy_file(self, source_path: str, target_path: str) -> bool:
        """
        Copy a file to a new location in the bucket

        Args:
            source_path: Current file path in bucket
            target_path: New file path in bucket

        Returns:
            True if successful, False otherwise
        """
        try:
            # Download content from source
            content = self.download_file_content(source_path)
            if not content:
                return False

            # Upload to target location
            content_bytes = content.encode('utf-8')

            response = self.client.storage.from_(self.bucket_name).upload(
                path=target_path,
                file=content_bytes,
                file_options={
                    "content-type": "text/plain",
                    "x-upsert": "true"
                }
            )

            logger.info(f"📋 Copied file: {source_path} → {target_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to copy file: {e}")
            return False

    def move_file(self, source_path: str, target_path: str) -> bool:
        """
        Move a file to a new location in the bucket

        Args:
            source_path: Current file path in bucket
            target_path: New file path in bucket

        Returns:
            True if successful, False otherwise
        """
        try:
            # Copy file to new location
            if self.copy_file(source_path, target_path):
                # Delete original file
                return self.delete_file(source_path)
            return False

        except Exception as e:
            logger.error(f"❌ Failed to move file: {e}")
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
            logger.info(f"🗑️ Deleted file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete file {file_path}: {e}")
            return False

    def delete_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Delete multiple files from the bucket

        Args:
            file_paths: List of file paths to delete

        Returns:
            Dictionary with deletion results
        """
        try:
            response = self.client.storage.from_(self.bucket_name).remove(file_paths)

            result = {
                "success": True,
                "deleted_count": len(file_paths),
                "files": file_paths
            }

            logger.info(f"🗑️ Deleted {len(file_paths)} files from bucket")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to delete files: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_count": 0
            }

    def upload_file(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload an existing local file to Supabase Storage

        Args:
            file_path: Path to the local file
            metadata: File metadata

        Returns:
            Tuple of (success: bool, storage_path: Optional[str])
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Determine file type from extension
            file_type = file_path.split('.')[-1] if '.' in file_path else 'txt'

            # Upload using the content method
            return self.upload_scraped_content(content, metadata, file_type)

        except Exception as e:
            logger.error(f"❌ Failed to upload file {file_path}: {e}")
            return False, None

    def get_download_url(self, file_path: str, expires_in: int = 3600) -> Optional[str]:
        """
        Get a temporary download URL for a file

        Args:
            file_path: Path in the storage bucket
            expires_in: URL expiration time in seconds (default 1 hour)

        Returns:
            Signed URL or None if failed
        """
        try:
            response = self.client.storage.from_(self.bucket_name).create_signed_url(
                path=file_path,
                expires_in=expires_in
            )
            return response.get('signedURL')
        except Exception as e:
            logger.error(f"❌ Failed to create download URL: {e}")
            return None

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
            # List files with the specific path
            files = self.client.storage.from_(self.bucket_name).list(
                path=file_path,
                options={"limit": 1}
            )

            if files:
                return files[0]
            else:
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