"""
Clause retrieval system for Claimsure.

This module provides functionality for retrieving relevant clauses and text chunks
from the vector store based on semantic similarity to user queries.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .embeddings import EmbeddingSystem

logger = logging.getLogger(__name__)


class ClauseRetrieval:
    """
    Clause retrieval system that uses semantic search to find relevant clauses
    and text chunks from the vector store.
    """
    
    def __init__(self, embedding_system: EmbeddingSystem):
        """
        Initialize the clause retrieval system.
        
        Args:
            embedding_system: Pre-initialized embedding system
        """
        self.embedding_system = embedding_system
        logger.info("Initialized ClauseRetrieval system")
    
    def retrieve_clauses(self, 
                        query: str, 
                        top_k: int = 5,
                        similarity_threshold: float = 0.5,
                        include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant clauses based on semantic similarity to the query.
        
        Args:
            query: User query string
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of clause dictionaries sorted by similarity score
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for clause retrieval")
            return []
        
        try:
            logger.info(f"Retrieving clauses for query: {query[:50]}...")
            
            # Use the embedding system's search functionality
            results = self.embedding_system.search(query, top_k=top_k)
            
            if not results:
                logger.info("No results found for query")
                return []
            
            # Filter results by similarity threshold and format
            filtered_results = []
            for result in results:
                similarity_score = result.get("similarity_score", 0.0)
                
                if similarity_score >= similarity_threshold:
                    # Format the result as a clause
                    clause = self._format_clause_result(result, include_metadata)
                    filtered_results.append(clause)
            
            # Sort by similarity score (descending)
            filtered_results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            
            logger.info(f"Retrieved {len(filtered_results)} clauses above threshold {similarity_threshold}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error retrieving clauses for query '{query}': {e}")
            return []
    
    def _format_clause_result(self, result: Dict[str, Any], include_metadata: bool) -> Dict[str, Any]:
        """
        Format a search result as a clause dictionary.
        
        Args:
            result: Raw search result from embedding system
            include_metadata: Whether to include metadata
            
        Returns:
            Formatted clause dictionary
        """
        clause = {
            "clause_id": result.get("id", result.get("chunk_id", "unknown")),
            "text": result.get("text", result.get("content", "")),
            "similarity_score": result.get("similarity_score", 0.0),
            "rank": result.get("rank", 0),
            "source": result.get("source", "unknown"),
            "chunk_id": result.get("chunk_id", 0)
        }
        
        if include_metadata:
            # Include additional metadata if available
            metadata = result.get("metadata", {})
            if metadata:
                clause["metadata"] = metadata
            
            # Include version information
            if "version" in result:
                clause["version"] = result["version"]
        
        return clause
    
    def retrieve_clauses_batch(self, 
                              queries: List[str], 
                              top_k: int = 5,
                              similarity_threshold: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve clauses for multiple queries in batch.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary mapping queries to their clause results
        """
        results = {}
        
        for query in queries:
            try:
                clauses = self.retrieve_clauses(query, top_k, similarity_threshold)
                results[query] = clauses
            except Exception as e:
                logger.error(f"Error processing batch query '{query}': {e}")
                results[query] = []
        
        return results
    
    def retrieve_relevant_chunks(self, 
                                query: str, 
                                top_k: int = 5,
                                similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant text chunks based on semantic similarity to the query.
        This is an alias for retrieve_clauses for compatibility with the enhanced hybrid processor.
        
        Args:
            query: User query string
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of chunk dictionaries sorted by similarity score
        """
        return self.retrieve_clauses(query, top_k, similarity_threshold)
    
    def get_clause_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the clause retrieval system.
        
        Returns:
            Dictionary with system statistics
        """
        stats = self.embedding_system.get_statistics()
        
        # Add retrieval-specific statistics
        retrieval_stats = {
            "total_clauses": stats.get("vector_count", 0),
            "embedding_model": stats.get("model_name", "unknown"),
            "vector_store_type": stats.get("vector_store_type", "unknown"),
            "embedding_dimension": stats.get("dimension", 0)
        }
        
        return retrieval_stats


def retrieve_clauses(query: str, 
                    embedding_system: EmbeddingSystem,
                    top_k: int = 5,
                    similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve clauses for a single query.
    
    Args:
        query: User query string
        embedding_system: Pre-initialized embedding system
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score
        
    Returns:
        List of clause dictionaries sorted by similarity score
    """
    retrieval_system = ClauseRetrieval(embedding_system)
    return retrieval_system.retrieve_clauses(query, top_k, similarity_threshold)


def retrieve_clauses_batch(queries: List[str],
                          embedding_system: EmbeddingSystem,
                          top_k: int = 5,
                          similarity_threshold: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to retrieve clauses for multiple queries.
    
    Args:
        queries: List of query strings
        embedding_system: Pre-initialized embedding system
        top_k: Number of top results per query
        similarity_threshold: Minimum similarity score
        
    Returns:
        Dictionary mapping queries to their clause results
    """
    retrieval_system = ClauseRetrieval(embedding_system)
    return retrieval_system.retrieve_clauses_batch(queries, top_k, similarity_threshold)


# Example usage and testing
if __name__ == "__main__":
    # Test the clause retrieval system
    print("Testing Clause Retrieval System")
    print("=" * 50)
    
    # Initialize embedding system (this would normally be done elsewhere)
    try:
        embedding_system = EmbeddingSystem()
        print("✅ EmbeddingSystem initialized successfully")
        
        # Initialize clause retrieval
        retrieval_system = ClauseRetrieval(embedding_system)
        print("✅ ClauseRetrieval initialized successfully")
        
        # Test queries
        test_queries = [
            "What is covered under health insurance?",
            "What are the coverage limits?",
            "Define deductible",
            "How much does a doctor visit cost?"
        ]
        
        print(f"\n📝 Testing {len(test_queries)} queries:")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            try:
                clauses = retrieval_system.retrieve_clauses(query, top_k=3)
                print(f"\n{i}. Query: {query}")
                print(f"   Found {len(clauses)} clauses:")
                
                for j, clause in enumerate(clauses, 1):
                    print(f"     {j}. Score: {clause['similarity_score']:.3f}")
                    print(f"        Text: {clause['text'][:60]}...")
                    print(f"        Source: {clause['source']}")
            
            except Exception as e:
                print(f"\n{i}. Query: {query}")
                print(f"   Error: {e}")
        
        # Show statistics
        print(f"\n📊 Clause Retrieval Statistics:")
        print("-" * 30)
        stats = retrieval_system.get_clause_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    except Exception as e:
        print(f"❌ Failed to initialize systems: {e}")
