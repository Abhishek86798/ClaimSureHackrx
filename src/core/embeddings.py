"""
Embedding and vector storage system for Claimsure.

Provides a modular embedding system that can use different vector databases.
Currently supports FAISS with extensibility for Pinecone.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import json
import os
import threading
from pathlib import Path
from joblib import dump, load
import hashlib
import psutil
import time

logger = logging.getLogger(__name__)

# Version for metadata compatibility
METADATA_VERSION = "1.0.0"


def log_memory_usage(operation: str, additional_info: str = ""):
    """Log memory usage for monitoring large-scale operations."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_memory_mb = system_memory.total / 1024 / 1024
        system_memory_percent = system_memory.percent
        
        logger.info(
            f"Memory usage for {operation}: "
            f"Process: {memory_mb:.1f}MB, "
            f"System: {system_memory_percent:.1f}% ({system_memory_mb:.0f}MB total)"
            f"{' - ' + additional_info if additional_info else ''}"
        )
    except Exception as e:
        logger.warning(f"Failed to log memory usage: {e}")


class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors to the store."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """Get the number of vectors in the store."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector storage implementation with thread safety."""
    
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimension of the vectors
            index_type: Type of FAISS index to use
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metadata = []
        self._lock = threading.Lock()  # Thread safety for concurrent access
        
        # Initialize FAISS index
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Initialized FAISS vector store with dimension {dimension}")
        log_memory_usage("FAISS initialization", f"dimension={dimension}, index_type={index_type}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add vectors to the FAISS index with thread safety.
        
        Args:
            vectors: Numpy array of vectors
            metadata: List of metadata dictionaries
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        start_time = time.time()
        log_memory_usage("FAISS add_vectors start", f"vectors={len(vectors)}, dimension={vectors.shape[1]}")
        
        with self._lock:
            try:
                # Normalize vectors for cosine similarity (IMPORTANT for IndexFlatIP)
                if self.index_type == "IndexFlatIP":
                    faiss.normalize_L2(vectors)
                
                # Add vectors to index
                self.index.add(vectors)
                
                # Store metadata with aligned schema for Pinecone compatibility
                aligned_metadata = []
                for i, meta in enumerate(metadata):
                    # Create unique ID for each chunk
                    chunk_id = meta.get("chunk_id", i)
                    source = meta.get("source", "unknown")
                    
                    # Align metadata schema with Pinecone format
                    aligned_meta = {
                        "id": f"chunk_{chunk_id}_{source}",
                        "text": meta.get("text", ""),
                        "source": source,
                        "chunk_id": chunk_id,
                        "version": METADATA_VERSION,  # Add version for future migrations
                        "metadata": meta  # Keep original metadata
                    }
                    aligned_metadata.append(aligned_meta)
                
                self.metadata.extend(aligned_metadata)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Added {len(vectors)} vectors to FAISS index in {elapsed_time:.2f}s")
                log_memory_usage("FAISS add_vectors end", f"total_vectors={self.index.ntotal}")
                
            except Exception as e:
                logger.error(f"Failed to add vectors to FAISS index: {e}")
                raise
    
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors with thread safety.
        
        Args:
            query_vector: Query vector
            top_k: Number of top results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        with self._lock:
            # Normalize query vector for cosine similarity (IMPORTANT for IndexFlatIP)
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(query_vector.reshape(1, -1))
            
            # Search
            distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
            
            return distances[0], indices[0]
    
    def save(self, path: str) -> None:
        """
        Save the FAISS index and metadata to disk with error handling.
        
        Args:
            path: Directory path to save to
        """
        try:
            os.makedirs(path, exist_ok=True)
            log_memory_usage("FAISS save start", f"path={path}, vectors={self.index.ntotal}")
            
            with self._lock:
                # Save FAISS index with error handling
                index_path = os.path.join(path, "faiss_index.bin")
                try:
                    faiss.write_index(self.index, index_path)
                    logger.info(f"Successfully saved FAISS index to {index_path}")
                except Exception as e:
                    logger.error(f"Failed to save FAISS index to {index_path}: {e}")
                    raise
                
                # Save metadata as JSON for better portability
                metadata_path = os.path.join(path, "metadata.json")
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                    logger.info(f"Successfully saved metadata to {metadata_path}")
                except Exception as e:
                    logger.error(f"Failed to save metadata to {metadata_path}: {e}")
                    # Try to clean up the index file if metadata save failed
                    if os.path.exists(index_path):
                        try:
                            os.remove(index_path)
                            logger.info("Cleaned up index file after metadata save failure")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up index file: {cleanup_error}")
                    raise
                
                log_memory_usage("FAISS save end", f"path={path}")
                logger.info(f"Successfully saved FAISS index and metadata to {path}")
                
        except Exception as e:
            logger.error(f"Failed to save FAISS vector store to {path}: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load the FAISS index and metadata from disk with error handling.
        
        Args:
            path: Directory path to load from
        """
        try:
            log_memory_usage("FAISS load start", f"path={path}")
            
            with self._lock:
                # Load FAISS index with error handling
                index_path = os.path.join(path, "faiss_index.bin")
                if not os.path.exists(index_path):
                    raise FileNotFoundError(f"FAISS index file not found: {index_path}")
                
                try:
                    self.index = faiss.read_index(index_path)
                    logger.info(f"Successfully loaded FAISS index from {index_path}")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index from {index_path}: {e}")
                    raise
                
                # Load metadata from JSON with fallback to pickle
                metadata_path = os.path.join(path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            self.metadata = json.load(f)
                        logger.info(f"Successfully loaded metadata from {metadata_path}")
                    except Exception as e:
                        logger.error(f"Failed to load JSON metadata from {metadata_path}: {e}")
                        # Try pickle fallback
                        pickle_path = os.path.join(path, "metadata.pkl")
                        if os.path.exists(pickle_path):
                            try:
                                with open(pickle_path, 'rb') as f:
                                    self.metadata = pickle.load(f)
                                logger.info(f"Successfully loaded metadata from pickle fallback: {pickle_path}")
                            except Exception as pickle_error:
                                logger.error(f"Failed to load pickle metadata from {pickle_path}: {pickle_error}")
                                self.metadata = []
                        else:
                            logger.warning("No metadata file found, using empty metadata")
                            self.metadata = []
                else:
                    # Fallback to pickle for backward compatibility
                    pickle_path = os.path.join(path, "metadata.pkl")
                    if os.path.exists(pickle_path):
                        try:
                            with open(pickle_path, 'rb') as f:
                                self.metadata = pickle.load(f)
                            logger.info(f"Successfully loaded metadata from pickle: {pickle_path}")
                        except Exception as e:
                            logger.error(f"Failed to load pickle metadata from {pickle_path}: {e}")
                            self.metadata = []
                    else:
                        logger.warning("No metadata file found, using empty metadata")
                        self.metadata = []
                
                # Validate metadata version if present
                if self.metadata:
                    for meta in self.metadata:
                        if "version" in meta:
                            version = meta.get("version")
                            if version != METADATA_VERSION:
                                logger.warning(f"Metadata version mismatch: expected {METADATA_VERSION}, got {version}")
                            break
                
                log_memory_usage("FAISS load end", f"path={path}, vectors={self.index.ntotal}")
                logger.info(f"Successfully loaded FAISS index and metadata from {path}")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS vector store from {path}: {e}")
            raise
    
    def get_vector_count(self) -> int:
        """Get the number of vectors in the store."""
        with self._lock:
            return self.index.ntotal


class EmbeddingSystem:
    """Main embedding system that handles text embedding and vector storage with caching."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 vector_store_type: str = "faiss",
                 dimension: int = 384,
                 index_type: str = "IndexFlatIP",
                 storage_path: str = "data/vector_store",
                 enable_caching: bool = True,
                 cache_dir: str = "data/embedding_cache"):
        """
        Initialize the embedding system.
        
        Args:
            model_name: Sentence transformer model name
            vector_store_type: Type of vector store ("faiss" or "pinecone")
            dimension: Dimension of embeddings
            index_type: FAISS index type
            storage_path: Path for storing vectors
            enable_caching: Whether to enable embedding caching
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.vector_store_type = vector_store_type
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.enable_caching = enable_caching
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        log_memory_usage("Model loading start", f"model={model_name}")
        self.model = SentenceTransformer(model_name)
        log_memory_usage("Model loading end", f"model={model_name}")
        
        # Initialize vector store
        if vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore(dimension, index_type)
        elif vector_store_type == "pinecone":
            # TODO: Implement PineconeVectorStore
            raise NotImplementedError("Pinecone support not yet implemented")
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
        
        log_memory_usage("EmbeddingSystem initialization complete", 
                        f"model={model_name}, vector_store={vector_store_type}")
        logger.info(f"Initialized embedding system with {vector_store_type} vector store")
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for a list of texts."""
        # Create hash of texts and model name
        text_hash = hashlib.md5(''.join(texts).encode()).hexdigest()
        model_hash = hashlib.md5(self.model_name.encode()).hexdigest()
        return f"{model_hash}_{text_hash}"
    
    def _load_cached_embeddings(self, cache_key: str) -> Optional[np.ndarray]:
        """Load cached embeddings if available."""
        if not self.enable_caching:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.joblib"
        if cache_file.exists():
            try:
                embeddings = load(cache_file)
                logger.info(f"Loaded cached embeddings for {len(embeddings)} texts")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
                return None
        return None
    
    def _save_cached_embeddings(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Save embeddings to cache."""
        if not self.enable_caching:
            return
        
        try:
            cache_file = self.cache_dir / f"{cache_key}.joblib"
            dump(embeddings, cache_file)
            logger.info(f"Cached embeddings for {len(embeddings)} texts")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for text chunks with caching support.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            Numpy array of embeddings
        """
        if not chunks:
            return np.array([])
        
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Check cache first
        cache_key = self._get_cache_key(texts)
        cached_embeddings = self._load_cached_embeddings(cache_key)
        
        if cached_embeddings is not None:
            return cached_embeddings
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Cache the embeddings
        self._save_cached_embeddings(cache_key, embeddings)
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, chunks)
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of result dictionaries with chunks and similarity scores
        """
        # Check if vector store is empty
        if self.vector_store.get_vector_count() == 0:
            logger.info("Vector store is empty, returning no results")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search vector store
        distances, indices = self.vector_store.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (distance, index) in enumerate(zip(distances, indices)):
            if index < len(self.vector_store.metadata):
                chunk = self.vector_store.metadata[index].copy()
                chunk["similarity_score"] = float(distance)
                chunk["rank"] = i + 1
                results.append(chunk)
        
        logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results
    
    def save(self) -> None:
        """Save the vector store to disk."""
        self.vector_store.save(str(self.storage_path))
        logger.info(f"Saved vector store to {self.storage_path}")
    
    def load(self) -> None:
        """Load the vector store from disk."""
        if (self.storage_path / "faiss_index.bin").exists():
            self.vector_store.load(str(self.storage_path))
            logger.info(f"Loaded vector store from {self.storage_path}")
        else:
            logger.info("No existing vector store found")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "vector_count": self.vector_store.get_vector_count(),
            "model_name": self.model_name,
            "vector_store_type": self.vector_store_type,
            "dimension": self.dimension,
            "storage_path": str(self.storage_path),
            "cache_enabled": self.enable_caching,
            "cache_dir": str(self.cache_dir)
        }
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        # Reinitialize vector store
        if self.vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore(self.dimension)
        else:
            raise NotImplementedError("Clear not implemented for this vector store type")
        
        logger.info("Cleared vector store")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.enable_caching and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared embedding cache")


class PineconeVectorStore(VectorStore):
    """Pinecone-based vector storage implementation (placeholder for future)."""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Pinecone index name
        """
        # TODO: Implement Pinecone integration
        raise NotImplementedError("Pinecone integration not yet implemented")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors to Pinecone."""
        raise NotImplementedError("Pinecone integration not yet implemented")
    
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search Pinecone index."""
        raise NotImplementedError("Pinecone integration not yet implemented")
    
    def save(self, path: str) -> None:
        """Save Pinecone index (not applicable)."""
        logger.warning("Pinecone does not support local saving")
    
    def load(self, path: str) -> None:
        """Load Pinecone index (not applicable)."""
        logger.warning("Pinecone does not support local loading")
    
    def get_vector_count(self) -> int:
        """Get vector count from Pinecone."""
        raise NotImplementedError("Pinecone integration not yet implemented")
