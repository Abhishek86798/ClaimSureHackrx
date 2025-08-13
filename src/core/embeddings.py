#!/usr/bin/env python3
"""
Embedding System for Claimsure
Handles vector embeddings and similarity search
"""

import os
import logging
import gc
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import psutil

logger = logging.getLogger(__name__)

class EmbeddingSystem:
    """Handles document embeddings and similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "/tmp/embeddings_cache"):
        """
        Initialize the embedding system with memory optimization
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        # Memory optimization settings
        self.max_memory_mb = 512  # Limit memory usage
        self.chunk_size = 100     # Process chunks in batches
        
        # Initialize the model with memory optimization
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model with memory optimization"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            
            # Log memory usage before loading
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage for Model loading start: Process: {memory_before:.1f}MB, System: {psutil.virtual_memory().percent:.1f}% ({psutil.virtual_memory().total / 1024 / 1024 / 1024:.0f}GB total) - model={self.model_name}")
            
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load model with memory optimization
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device='cpu'  # Force CPU to save memory
            )
            
            # Log memory usage after loading
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            logger.info(f"Model loaded successfully. Memory used: {memory_used:.1f}MB, Total: {memory_after:.1f}MB")
            
            # Force garbage collection
            gc.collect()
                
            except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks to the embedding system with memory optimization
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        try:
            if not chunks:
                logger.warning("No chunks provided to add")
                return
            
            logger.info(f"Adding {len(chunks)} chunks to embedding system")
            
            # Process chunks in batches to manage memory
            batch_size = min(self.chunk_size, len(chunks))
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Extract text from chunks
                texts = [chunk.get('text', '') for chunk in batch]
                
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=8  # Small batch size to save memory
                )
                
                # Add to documents and chunks
                for j, chunk in enumerate(batch):
                    self.documents.append({
                        'id': chunk.get('id', f'doc_{len(self.documents)}'),
                        'text': chunk.get('text', ''),
                        'source': chunk.get('source', 'unknown'),
                        'metadata': chunk.get('metadata', {})
                    })
                    self.chunks.append(chunk)
                
                # Store embeddings
                if self.embeddings is None:
                    self.embeddings = batch_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, batch_embeddings])
                
                # Check memory usage
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                
                if memory_usage > self.max_memory_mb:
                    logger.warning(f"Memory usage high: {memory_usage:.1f}MB. Forcing garbage collection.")
                    gc.collect()
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            # Build FAISS index
            self._build_index()
            
            logger.info(f"Successfully added {len(chunks)} chunks. Total documents: {len(self.documents)}")
                
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            raise
    
    def _build_index(self):
        """Build FAISS index for similarity search"""
        try:
            if self.embeddings is None or len(self.embeddings) == 0:
                logger.warning("No embeddings to build index")
            return
        
            # Get embedding dimension
            dimension = self.embeddings.shape[1]
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            
            # Add embeddings to index
            self.index.add(self.embeddings.astype('float32'))
            
            logger.info(f"Built FAISS index with {len(self.embeddings)} vectors of dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            if self.index is None or len(self.documents) == 0:
                logger.warning("No documents indexed for search")
            return []
        
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
        
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.documents)))
        
            # Format results
        results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'document': doc,
                        'score': float(score),
                        'rank': i + 1
                    })
            
            logger.info(f"Search completed. Found {len(results)} results for query: {query[:50]}...")
        return results
    
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def get_total_documents(self) -> int:
        """Get total number of documents"""
        return len(self.documents)
    
    def get_total_chunks(self) -> int:
        """Get total number of chunks"""
        return len(self.chunks)
    
    def clear(self):
        """Clear all documents and free memory"""
        try:
            self.documents = []
            self.chunks = []
            self.embeddings = None
            self.index = None
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Embedding system cleared")
            
        except Exception as e:
            logger.error(f"Error clearing embedding system: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'system_memory_percent': psutil.virtual_memory().percent,
                'total_documents': len(self.documents),
                'total_chunks': len(self.chunks),
                'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
