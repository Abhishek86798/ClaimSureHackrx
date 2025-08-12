"""
File utility functions for Claimsure.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: str) -> bool:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def get_file_extension(file_path: str) -> str:
    """
    Get file extension from path.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (lowercase)
    """
    return Path(file_path).suffix.lower()


def is_supported_file_type(file_path: str, supported_types: List[str]) -> bool:
    """
    Check if file type is supported.
    
    Args:
        file_path: Path to file
        supported_types: List of supported file extensions
        
    Returns:
        True if file type is supported
    """
    extension = get_file_extension(file_path)
    return extension in supported_types


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return 0.0


def copy_file_to_directory(source_path: str, dest_dir: str, new_name: Optional[str] = None) -> str:
    """
    Copy a file to a directory with optional new name.
    
    Args:
        source_path: Source file path
        dest_dir: Destination directory
        new_name: Optional new filename
        
    Returns:
        Path to copied file
    """
    try:
        ensure_directory(dest_dir)
        
        if new_name:
            dest_path = Path(dest_dir) / new_name
        else:
            dest_path = Path(dest_dir) / Path(source_path).name
        
        shutil.copy2(source_path, dest_path)
        return str(dest_path)
    except Exception as e:
        logger.error(f"Failed to copy file {source_path} to {dest_dir}: {e}")
        raise


def list_files_in_directory(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    List files in directory with optional extension filter.
    
    Args:
        directory: Directory path
        extensions: Optional list of file extensions to include
        
    Returns:
        List of file paths
    """
    try:
        files = []
        for file_path in Path(directory).iterdir():
            if file_path.is_file():
                if extensions is None or get_file_extension(str(file_path)) in extensions:
                    files.append(str(file_path))
        return files
    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {e}")
        return []


def safe_filename(filename: str) -> str:
    """
    Convert filename to safe version by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')
    return safe_name or "unnamed_file"
