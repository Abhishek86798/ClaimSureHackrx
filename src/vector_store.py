"""
Vector store module for Claimsure.

Handles storage and retrieval of vector embeddings using Pinecone or other vector databases.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import pinecone
from config import (
    VECTOR_DB_TYPE, PINECONE_INDEX_NAME, PINECONE_METRIC,
    PINECONE_API_KEY, PINECONE_ENVIRONMENT, TOP_K_RESULTS, SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Handles storage and retrieval of vector embeddings."""
    
    def __init__(self, vector_db_type: str = VECTOR_DB_TYPE):
        self.vector_db_type = vector_db_type
        self.index = None
        self.initialized = False
        
        if vector_db_type == "pinecone":
            self._initialize_pinecone()
        else:
            raise ValueError(f"Unsupported vector database type: {vector_db_type}")
    
    def _initialize_pinecone(self):
        """Initialize Pinecone vector database."""
        try:
            # Check if API key is available
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY environment variable is required")
            
            if not PINECONE_ENVIRONMENT:
                raise ValueError("PINECONE_ENVIRONMENT environment variable is required")
            
            # Initialize Pinecone
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            
            # Get or create index
            if PINECONE_INDEX_NAME not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                pinecone.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric=PINECONE_METRIC
                )
            
            # Connect to index
            self.index = pinecone.Index(PINECONE_INDEX_NAME)
            self.initialized = True
            logger.info(f"Successfully initialized Pinecone index: {PINECONE_INDEX_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return False
        
        if not chunks:
            logger.warning("No chunks provided for storage")
            return True
        
        try:
            if self.vector_db_type == "pinecone":
                return self._add_chunks_to_pinecone(chunks)
            else:
                logger.error(f"Unsupported vector database type: {self.vector_db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            return False
    
    def _add_chunks_to_pinecone(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add chunks to Pinecone index."""
        try:
            vectors = []
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", "")
                embedding = chunk.get("embedding", [])
                metadata = {
                    "content": chunk.get("content", ""),
                    "file_path": chunk.get("file_path", ""),
                    "file_type": chunk.get("file_type", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 0),
                    "chunk_size": chunk.get("chunk_size", 0)
                }
                
                # Add document metadata
                doc_metadata = chunk.get("document_metadata", {})
                metadata.update(doc_metadata)
                
                vectors.append((chunk_id, embedding, metadata))
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}")
            
            logger.info(f"Successfully added {len(chunks)} chunks to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to Pinecone: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = TOP_K_RESULTS, 
               similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar chunk dictionaries
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        
        try:
            if self.vector_db_type == "pinecone":
                return self._search_pinecone(query_embedding, top_k, similarity_threshold)
            else:
                logger.error(f"Unsupported vector database type: {self.vector_db_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def _search_pinecone(self, query_embedding: List[float], top_k: int, 
                        similarity_threshold: float) -> List[Dict[str, Any]]:
        """Search Pinecone index for similar chunks."""
        try:
            # Query the index
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process results
            similar_chunks = []
            for match in results.matches:
                if match.score >= similarity_threshold:
                    chunk_data = {
                        "chunk_id": match.id,
                        "similarity_score": match.score,
                        "content": match.metadata.get("content", ""),
                        "file_path": match.metadata.get("file_path", ""),
                        "file_type": match.metadata.get("file_type", ""),
                        "chunk_index": match.metadata.get("chunk_index", 0),
                        "total_chunks": match.metadata.get("total_chunks", 0),
                        "chunk_size": match.metadata.get("chunk_size", 0),
                        "document_metadata": {
                            k: v for k, v in match.metadata.items()
                            if k not in ["content", "file_path", "file_type", "chunk_index", "total_chunks", "chunk_size"]
                        }
                    }
                    similar_chunks.append(chunk_data)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks above threshold {similarity_threshold}")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Delete chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            if self.vector_db_type == "pinecone":
                self.index.delete(ids=chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunks from Pinecone")
                return True
            else:
                logger.error(f"Unsupported vector database type: {self.vector_db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting chunks from vector store: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store index.
        
        Returns:
            Dictionary containing index statistics
        """
        if not self.initialized:
            return {"error": "Vector store not initialized"}
        
        try:
            if self.vector_db_type == "pinecone":
                stats = self.index.describe_index_stats()
                return {
                    "total_vector_count": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "index_fullness": stats.index_fullness,
                    "namespaces": stats.namespaces
                }
            else:
                return {"error": f"Unsupported vector database type: {self.vector_db_type}"}
                
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}
    
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized."""
        return self.initialized
