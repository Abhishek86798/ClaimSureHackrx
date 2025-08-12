"""
Utility functions for file handling and management.
"""

import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from config import SUPPORTED_FILE_TYPES, MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)

def validate_file_type(filename: str) -> bool:
    """
    Validate if file type is supported.
    
    Args:
        filename: Name of the file
        
    Returns:
        bool: True if file type is supported
    """
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in SUPPORTED_FILE_TYPES

def validate_file_size(file_size: int) -> bool:
    """
    Validate if file size is within limits.
    
    Args:
        file_size: Size of the file in bytes
        
    Returns:
        bool: True if file size is acceptable
    """
    return file_size <= MAX_FILE_SIZE_BYTES

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict[str, Any]: File information
    """
    try:
        stat = os.stat(file_path)
        
        return {
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "file_size": stat.st_size,
            "file_type": os.path.splitext(file_path)[1].lower(),
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "mime_type": mimetypes.guess_type(file_path)[0] or "unknown"
        }
    
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        raise

def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate file hash for integrity checking.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)
        
    Returns:
        str: File hash
    """
    try:
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    except Exception as e:
        logger.error(f"Error calculating file hash for {file_path}: {str(e)}")
        raise

def create_safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing or replacing unsafe characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Safe filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    safe_filename = filename
    
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    safe_filename = safe_filename.strip('. ')
    
    # Ensure filename is not empty
    if not safe_filename:
        safe_filename = "unnamed_file"
    
    return safe_filename

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        bool: True if directory exists or was created
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            logger.info(f"Created directory: {directory_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {str(e)}")
        return False

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        str: File extension (with dot)
    """
    return os.path.splitext(filename)[1].lower()

def is_text_file(filename: str) -> bool:
    """
    Check if file is a text file based on extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        bool: True if file is a text file
    """
    text_extensions = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.css', '.js'}
    return get_file_extension(filename) in text_extensions

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {str(e)}")
        return 0.0

def list_files_in_directory(directory_path: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    List files in a directory with optional extension filtering.
    
    Args:
        directory_path: Path to the directory
        extensions: List of file extensions to include (optional)
        
    Returns:
        List[str]: List of file paths
    """
    try:
        if not os.path.exists(directory_path):
            return []
        
        files = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                if extensions is None or get_file_extension(filename) in extensions:
                    files.append(file_path)
        
        return files
    
    except Exception as e:
        logger.error(f"Error listing files in {directory_path}: {str(e)}")
        return []

def cleanup_temp_files(temp_files: List[str]) -> None:
    """
    Clean up temporary files.
    
    Args:
        temp_files: List of temporary file paths to delete
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file {temp_file}: {str(e)}")
