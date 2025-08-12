"""
Intelligent chunking utilities for Claimsure.

Provides semantic-aware text chunking that preserves context and meaning.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def semantic_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Split text into semantic chunks while preserving context.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not text or not text.strip():
        return []
    
    # Clean and normalize text
    text = text.strip()
    
    # Split text into paragraphs first
    paragraphs = split_into_paragraphs(text)
    
    # If we have very few paragraphs, split by sentences
    if len(paragraphs) <= 2:
        paragraphs = split_into_sentences(text)
    
    # If still too few segments, split by words
    if len(paragraphs) <= 2:
        paragraphs = split_into_words(text, chunk_size // 4)
    
    # Create chunks from paragraphs
    chunks = create_chunks_from_segments(paragraphs, chunk_size, overlap)
    
    # Add metadata to chunks
    chunk_dicts = []
    for i, chunk_text in enumerate(chunks):
        chunk_dict = {
            "text": chunk_text,
            "metadata": {
                "chunk_id": i,
                "chunk_size": len(chunk_text),
                "total_chunks": len(chunks),
                "overlap_size": overlap
            }
        }
        chunk_dicts.append(chunk_dict)
    
    logger.info(f"Created {len(chunk_dicts)} semantic chunks from text of length {len(text)}")
    return chunk_dicts


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs based on double newlines.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraph strings
    """
    # Split by double newlines (common paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean up paragraphs
    cleaned_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if para:
            # Remove excessive whitespace within paragraphs
            para = re.sub(r'\s+', ' ', para)
            cleaned_paragraphs.append(para)
    
    return cleaned_paragraphs


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using intelligent sentence boundary detection.
    
    Args:
        text: Input text
        
    Returns:
        List of sentence strings
    """
    # Normalize text first
    text = re.sub(r'\s+', ' ', text)
    
    # Split by sentence endings (., !, ?) followed by space and capital letter
    # This handles common cases like "Mr. Smith" or "U.S.A."
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean up sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Minimum sentence length
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def split_into_words(text: str, max_words: int) -> List[str]:
    """
    Split text into word-based segments.
    
    Args:
        text: Input text
        max_words: Maximum words per segment
        
    Returns:
        List of word segments
    """
    # Normalize text
    text = re.sub(r'\s+', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Create segments
    segments = []
    for i in range(0, len(words), max_words):
        segment_words = words[i:i + max_words]
        segment = ' '.join(segment_words)
        if segment.strip():
            segments.append(segment)
    
    return segments


def create_chunks_from_segments(segments: List[str], chunk_size: int, overlap: int) -> List[str]:
    """
    Create chunks from segments while respecting chunk size and overlap.
    
    Args:
        segments: List of text segments (paragraphs, sentences, etc.)
        chunk_size: Maximum chunk size
        overlap: Overlap size between chunks
        
    Returns:
        List of chunk strings
    """
    if not segments:
        return []
    
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for segment in segments:
        segment_size = len(segment)
        
        # If adding this segment would exceed chunk size
        if current_size + segment_size > chunk_size and current_chunk:
            # Finalize current chunk
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if overlap > 0 and chunks:
                # Get the last part of the previous chunk for overlap
                last_chunk = chunks[-1]
                overlap_text = last_chunk[-overlap:] if len(last_chunk) > overlap else last_chunk
                current_chunk = overlap_text + " " + segment
                current_size = len(current_chunk)
            else:
                current_chunk = segment
                current_size = segment_size
        else:
            # Add segment to current chunk
            if current_chunk:
                current_chunk += " " + segment
            else:
                current_chunk = segment
            current_size = len(current_chunk)
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Ensure we have at least 2 chunks for short texts (prevent single giant chunk)
    if len(chunks) == 1 and len(chunks[0]) > chunk_size // 2:
        # Split the single chunk into two
        mid_point = len(chunks[0]) // 2
        # Try to find a good split point (space or punctuation)
        split_point = find_good_split_point(chunks[0], mid_point)
        
        first_chunk = chunks[0][:split_point].strip()
        second_chunk = chunks[0][split_point:].strip()
        
        if first_chunk and second_chunk:
            chunks = [first_chunk, second_chunk]
    
    return chunks


def find_good_split_point(text: str, target_position: int) -> int:
    """
    Find a good split point near the target position.
    
    Args:
        text: Text to split
        target_position: Target position for split
        
    Returns:
        Actual split position
    """
    # Look for spaces near the target position
    search_range = min(100, len(text) // 4)  # Search within this range
    
    start = max(0, target_position - search_range)
    end = min(len(text), target_position + search_range)
    
    # Look for spaces first
    for i in range(target_position, end):
        if text[i] == ' ':
            return i
    
    for i in range(target_position - 1, start - 1, -1):
        if text[i] == ' ':
            return i
    
    # If no spaces, look for punctuation
    for i in range(target_position, end):
        if text[i] in '.!?':
            return i + 1
    
    for i in range(target_position - 1, start - 1, -1):
        if text[i] in '.!?':
            return i + 1
    
    # If all else fails, use the target position
    return target_position


def validate_chunk_parameters(chunk_size: int, overlap: int) -> bool:
    """
    Validate chunking parameters.
    
    Args:
        chunk_size: Maximum chunk size
        overlap: Overlap size
        
    Returns:
        True if parameters are valid
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    return True


def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about the chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "average_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }
    
    chunk_sizes = [len(chunk["text"]) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_characters": sum(chunk_sizes),
        "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes)
    }


def merge_small_chunks(chunks: List[Dict[str, Any]], min_size: int = 100) -> List[Dict[str, Any]]:
    """
    Merge very small chunks with adjacent chunks.
    
    Args:
        chunks: List of chunk dictionaries
        min_size: Minimum chunk size to avoid merging
        
    Returns:
        List of merged chunk dictionaries
    """
    if len(chunks) <= 1:
        return chunks
    
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        current_size = len(current_chunk["text"])
        
        # If current chunk is too small and not the last chunk
        if current_size < min_size and i < len(chunks) - 1:
            # Try to merge with next chunk
            next_chunk = chunks[i + 1]
            combined_size = current_size + len(next_chunk["text"])
            
            # Only merge if combined size is reasonable
            if combined_size <= current_size * 3:  # Don't create huge chunks
                merged_text = current_chunk["text"] + " " + next_chunk["text"]
                merged_chunk = {
                    "text": merged_text,
                    "metadata": {
                        "chunk_id": current_chunk["metadata"]["chunk_id"],
                        "chunk_size": len(merged_text),
                        "total_chunks": len(chunks) - 1,  # Will be updated
                        "overlap_size": current_chunk["metadata"]["overlap_size"],
                        "merged_from": [current_chunk["metadata"]["chunk_id"], next_chunk["metadata"]["chunk_id"]]
                    }
                }
                merged_chunks.append(merged_chunk)
                i += 2  # Skip the next chunk since we merged it
            else:
                merged_chunks.append(current_chunk)
                i += 1
        else:
            merged_chunks.append(current_chunk)
            i += 1
    
    # Update total_chunks in metadata
    for chunk in merged_chunks:
        chunk["metadata"]["total_chunks"] = len(merged_chunks)
    
    return merged_chunks
