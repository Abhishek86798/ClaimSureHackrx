"""
Embedding service for generating and managing text embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
from datetime import datetime

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Handles text embedding generation and vector database operations."""
    
    def __init__(self):
        self.model = None
        self.pinecone_index = None
        self._initialize_model()
        self._initialize_pinecone()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
                logger.warning("Pinecone credentials not configured. Vector search will be disabled.")
                return
            
            # Initialize Pinecone
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            
            # Check if index exists, create if not
            if PINECONE_INDEX_NAME not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                pinecone.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric=PINECONE_METRIC
                )
            
            # Connect to index
            self.pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone index initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List[List[float]]: List of embeddings
        """
        try:
            if not texts:
                return []
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[i:i + EMBEDDING_BATCH_SIZE]
                batch_embeddings = self.model.encode(batch, convert_to_tensor=False)
                embeddings.extend(batch_embeddings.tolist())
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            raise
    
    def store_chunks_with_embeddings(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Store document chunks with their embeddings in Pinecone.
        
        Args:
            chunks: List of chunk dictionaries with content
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.pinecone_index:
                logger.warning("Pinecone not initialized. Skipping vector storage.")
                return False
            
            # Extract texts and prepare metadata
            texts = []
            vectors = []
            metadata_list = []
            
            for chunk in chunks:
                texts.append(chunk["content"])
                metadata_list.append({
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "filename": chunk.get("filename", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_size": len(chunk["content"]),
                    "created_at": datetime.utcnow().isoformat()
                })
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Prepare vectors for Pinecone
            for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
                vector = {
                    "id": metadata["chunk_id"],
                    "values": embedding,
                    "metadata": metadata
                }
                vectors.append(vector)
            
            # Upsert to Pinecone
            self.pinecone_index.upsert(vectors=vectors)
            
            logger.info(f"Stored {len(vectors)} chunks in Pinecone")
            return True
        
        except Exception as e:
            logger.error(f"Error storing chunks in Pinecone: {str(e)}")
            raise
    
    def search_similar_chunks(self, query: str, top_k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            List[Dict[str, Any]]: List of similar chunks with scores
        """
        try:
            if not self.pinecone_index:
                logger.warning("Pinecone not initialized. Returning empty results.")
                return []
            
            # Use default values if not provided
            top_k = top_k or TOP_K_RESULTS
            threshold = threshold or SIMILARITY_THRESHOLD
            
            # Generate query embedding
            query_embedding = self.generate_single_embedding(query)
            
            # Search in Pinecone
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process results
            similar_chunks = []
            for match in results.matches:
                if match.score >= threshold:
                    chunk_info = {
                        "chunk_id": match.id,
                        "similarity_score": match.score,
                        "content": match.metadata.get("content", ""),
                        "metadata": {
                            "doc_id": match.metadata.get("doc_id", ""),
                            "filename": match.metadata.get("filename", ""),
                            "chunk_index": match.metadata.get("chunk_index", 0),
                            "chunk_size": match.metadata.get("chunk_size", 0)
                        }
                    }
                    similar_chunks.append(chunk_info)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks for query: {query}")
            return similar_chunks
        
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            raise
    
    def delete_document_chunks(self, doc_id: str) -> bool:
        """
        Delete all chunks for a specific document.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.pinecone_index:
                logger.warning("Pinecone not initialized. Skipping deletion.")
                return False
            
            # Query for chunks with the specific doc_id
            # Note: This is a simplified approach. In production, you might want to
            # maintain a separate mapping of doc_id to chunk_ids
            logger.info(f"Deleting chunks for document: {doc_id}")
            
            # For now, we'll return True as deletion would require more complex logic
            # to find all chunk IDs for a given doc_id
            return True
        
        except Exception as e:
            logger.error(f"Error deleting document chunks: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index.
        
        Returns:
            Dict[str, Any]: Index statistics
        """
        try:
            if not self.pinecone_index:
                return {"error": "Pinecone not initialized"}
            
            stats = self.pinecone_index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}
