#!/usr/bin/env python3
"""
Lightweight Embedding System for Claimsure
Uses simple text matching instead of heavy ML libraries
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class LightweightEmbeddingSystem:
    """Lightweight document search using text matching"""
    
    def __init__(self, cache_dir: str = "/tmp/embeddings_cache"):
        """
        Initialize the lightweight embedding system
        
        Args:
            cache_dir: Directory for caching (not used in lightweight version)
        """
        self.cache_dir = cache_dir
        self.documents = []
        self.chunks = []
        
        logger.info("Initialized lightweight embedding system")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks to the system
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        try:
            if not chunks:
                logger.warning("No chunks provided to add")
                return
            
            logger.info(f"Adding {len(chunks)} chunks to lightweight system")
            
            # Store chunks with simple indexing
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': i,
                    'text': chunk.get('text', ''),
                    'metadata': chunk.get('metadata', {}),
                    'keywords': self._extract_keywords(chunk.get('text', ''))
                }
                self.chunks.append(chunk_data)
            
            logger.info(f"Successfully added {len(self.chunks)} chunks")
                
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using text similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with similarity scores
        """
        try:
            if not self.chunks:
                logger.warning("No chunks available for search")
                return []
            
            query_keywords = self._extract_keywords(query.lower())
            results = []
            
            for chunk in self.chunks:
                # Calculate similarity using multiple methods
                text_similarity = self._calculate_text_similarity(query, chunk['text'])
                keyword_similarity = self._calculate_keyword_similarity(query_keywords, chunk['keywords'])
                
                # Combined similarity score
                combined_score = (text_similarity * 0.7) + (keyword_similarity * 0.3)
                
                if combined_score > 0.1:  # Only include relevant results
                    results.append({
                        'document': {
                            'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                            'metadata': chunk['metadata']
                        },
                        'similarity_score': combined_score,
                        'id': chunk['id']
                    })
            
            # Sort by similarity score and return top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Calculate text similarity using sequence matching"""
        try:
            return SequenceMatcher(None, query.lower(), text.lower()).ratio()
        except:
            return 0.0
    
    def _calculate_keyword_similarity(self, query_keywords: List[str], text_keywords: List[str]) -> float:
        """Calculate similarity based on keyword overlap"""
        if not query_keywords or not text_keywords:
            return 0.0
        
        # Count overlapping keywords
        overlap = len(set(query_keywords) & set(text_keywords))
        total = len(set(query_keywords) | set(text_keywords))
        
        return overlap / total if total > 0 else 0.0
    
    def clear(self):
        """Clear all stored data"""
        self.documents = []
        self.chunks = []
        logger.info("Cleared all data from lightweight embedding system")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(self.documents),
            "system_type": "lightweight_text_matching"
        }

# Alias for backward compatibility
EmbeddingSystem = LightweightEmbeddingSystem
