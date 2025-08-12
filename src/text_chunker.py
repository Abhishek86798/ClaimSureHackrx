"""
Text chunking module for Claimsure.

Handles splitting documents into smaller chunks for better processing and retrieval.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOCUMENT

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles splitting text documents into smaller chunks for processing."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = MAX_CHUNKS_PER_DOCUMENT
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks.
        
        Args:
            document: Document dictionary containing content and metadata
            
        Returns:
            List of chunk dictionaries
        """
        content = document.get("content", "")
        if not content.strip():
            logger.warning(f"Empty content in document: {document.get('file_path', 'unknown')}")
            return []
        
        chunks = self._split_text(content)
        
        # Limit number of chunks
        if len(chunks) > self.max_chunks:
            logger.warning(f"Document has {len(chunks)} chunks, limiting to {self.max_chunks}")
            chunks = chunks[:self.max_chunks]
        
        # Create chunk dictionaries with metadata
        chunk_dicts = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                "chunk_id": f"{document.get('file_path', 'unknown')}_chunk_{i}",
                "content": chunk_text,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "document_metadata": document.get("metadata", {}),
                "file_path": document.get("file_path", ""),
                "file_type": document.get("file_type", ""),
                "chunk_size": len(chunk_text)
            }
            chunk_dicts.append(chunk_dict)
        
        logger.info(f"Created {len(chunk_dicts)} chunks from document: {document.get('file_path', 'unknown')}")
        return chunk_dicts
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                search_end = min(end + 100, len(text))
                search_text = text[search_start:search_end]
                
                # Find the best sentence boundary
                sentence_end = self._find_sentence_boundary(search_text, end - search_start)
                if sentence_end > 0:
                    end = search_start + sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, preferred_position: int) -> int:
        """
        Find the best sentence boundary near the preferred position.
        
        Args:
            text: Text to search in
            preferred_position: Preferred position for the boundary
            
        Returns:
            Position of the best sentence boundary, or 0 if not found
        """
        # Common sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        # Search backwards from preferred position
        search_start = max(0, preferred_position - 200)
        search_end = min(len(text), preferred_position + 200)
        search_text = text[search_start:search_end]
        
        best_position = 0
        best_score = float('inf')
        
        for ending in sentence_endings:
            pos = search_text.rfind(ending, 0, preferred_position - search_start + 100)
            if pos != -1:
                actual_pos = search_start + pos + len(ending)
                distance = abs(actual_pos - preferred_position)
                if distance < best_score:
                    best_score = distance
                    best_position = actual_pos
        
        return best_position if best_position > 0 else 0
    
    def chunk_multiple_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunk dictionaries
        """
        all_chunks = []
        
        for document in documents:
            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {document.get('file_path', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary containing chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_characters": 0
            }
        
        chunk_sizes = [chunk.get("chunk_size", 0) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes)
        }
