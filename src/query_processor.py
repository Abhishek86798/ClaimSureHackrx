"""
Query processor module for Claimsure.

Handles query processing, retrieval, and response generation.
"""

import logging
from typing import List, Dict, Any, Optional
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm import LLMInterface
from config import TOP_K_RESULTS, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handles query processing and retrieval."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, 
                 vector_store: VectorStore, llm_interface: LLMInterface):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.llm_interface = llm_interface
    
    def process_query(self, query: str, top_k: int = TOP_K_RESULTS, 
                     similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Any]:
        """
        Process a query and return relevant results.
        
        Args:
            query: User query string
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score for results
            
        Returns:
            Dictionary containing query results and response
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return {
                    "success": False,
                    "error": "Failed to generate query embedding",
                    "query": query,
                    "results": [],
                    "response": ""
                }
            
            # Search for similar chunks
            similar_chunks = self.vector_store.search(
                query_embedding, top_k, similarity_threshold
            )
            
            if not similar_chunks:
                logger.warning(f"No similar chunks found for query: {query}")
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "response": "I couldn't find any relevant information for your query.",
                    "similarity_threshold": similarity_threshold
                }
            
            # Generate response using LLM
            response = self._generate_response(query, similar_chunks)
            
            return {
                "success": True,
                "query": query,
                "results": similar_chunks,
                "response": response,
                "num_results": len(similar_chunks),
                "similarity_threshold": similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
                "response": ""
            }
    
    def _generate_response(self, query: str, similar_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM based on retrieved chunks.
        
        Args:
            query: Original user query
            similar_chunks: List of similar chunks
            
        Returns:
            Generated response string
        """
        try:
            # Prepare context from similar chunks
            context = self._prepare_context(similar_chunks)
            
            # Create prompt for LLM
            prompt = self._create_prompt(query, context)
            
            # Generate response
            response = self.llm_interface.generate_response(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."
    
    def _prepare_context(self, similar_chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from similar chunks.
        
        Args:
            similar_chunks: List of similar chunk dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(similar_chunks, 1):
            content = chunk.get("content", "")
            file_path = chunk.get("file_path", "")
            similarity_score = chunk.get("similarity_score", 0.0)
            
            # Format chunk information
            chunk_info = f"Document {i} (from {file_path}, similarity: {similarity_score:.3f}):\n{content}\n"
            context_parts.append(chunk_info)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on the following context, please answer the user's question. 
If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def batch_process_queries(self, queries: List[str], top_k: int = TOP_K_RESULTS,
                             similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score for results
            
        Returns:
            List of query results
        """
        results = []
        
        for query in queries:
            try:
                result = self.process_query(query, top_k, similarity_threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "query": query,
                    "results": [],
                    "response": ""
                })
        
        return results
    
    def get_query_statistics(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about query processing results.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Dictionary containing query statistics
        """
        if not query_results:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "avg_results_per_query": 0,
                "avg_similarity_score": 0.0
            }
        
        total_queries = len(query_results)
        successful_queries = sum(1 for result in query_results if result.get("success", False))
        failed_queries = total_queries - successful_queries
        
        # Calculate average results per query
        total_results = sum(len(result.get("results", [])) for result in query_results)
        avg_results_per_query = total_results / total_queries if total_queries > 0 else 0
        
        # Calculate average similarity score
        similarity_scores = []
        for result in query_results:
            for chunk in result.get("results", []):
                similarity_scores.append(chunk.get("similarity_score", 0.0))
        
        avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "avg_results_per_query": avg_results_per_query,
            "avg_similarity_score": avg_similarity_score
        }
