# utils/supabase_storage.py
"""
Supabase Storage functionality for uploading scraped content files.
This handles uploading scraped content to Supabase Storage bucket.
"""

import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class SupabaseStorageManager:
    """Handles uploading files to Supabase Storage"""

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase client for storage operations"""
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            self.bucket_name = "scraped-content"  # Default bucket name
            self._ensure_bucket_exists()
            logger.info("✅ Supabase Storage Manager initialized")
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
                # FIX: Create bucket with correct Supabase Python client format
                # The create_bucket method expects: bucket_id, options=None
                response = self.client.storage.create_bucket(
                    bucket_id=self.bucket_name,  # Use bucket_id parameter name
                    options={
                        "public": False,  # Keep scraped content private
                        "file_size_limit": 10485760,  # Use snake_case instead of camelCase
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
            # Since bucket exists manually, let's just check if we can access it
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

            # Organize by year/month/city
            year_month = datetime.now().strftime("%Y/%m")
            file_path = f"{year_month}/{city}/scraped_{timestamp}_{content_hash}.{file_type}"

            # Convert content to bytes
            content_bytes = content.encode('utf-8')

            # Upload to Supabase Storage
            response = self.client.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=content_bytes,
                file_options={
                    "content-type": "text/plain" if file_type == "txt" else "application/json",
                    "x-upsert": "true"  # Allow overwrite if exists
                }
            )

            # Log metadata in the file description
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

    def upload_file(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload an existing file to Supabase Storage

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

    def list_files(self, prefix: str = "", limit: int = 100) -> list:
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
                    "offset": 0
                }
            )
            return files
        except Exception as e:
            logger.error(f"❌ Failed to list files: {e}")
            return []


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