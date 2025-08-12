"""
Free model alternatives for query processing and classification.

This module provides free, open-source alternatives to OpenAI's GPT models
for query processing, classification, and response generation.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Free model configurations
FREE_MODELS = {
    "sentence_transformers": {
        "all-MiniLM-L6-v2": {
            "description": "Fast and efficient sentence transformer for semantic search",
            "dimension": 384,
            "max_length": 256,
            "use_case": "semantic_search"
        },
        "multi-qa-MiniLM-L6-cos-v1": {
            "description": "Optimized for question-answering and semantic search",
            "dimension": 384,
            "max_length": 256,
            "use_case": "qa_search"
        },
        "all-mpnet-base-v2": {
            "description": "High-quality embeddings for semantic similarity",
            "dimension": 768,
            "max_length": 384,
            "use_case": "semantic_similarity"
        }
    }
}

# Query classification patterns for rule-based classification
QUERY_PATTERNS = {
    "coverage": [
        r"what.*cover", r"what.*included", r"what.*benefit", r"coverage.*include",
        r"what.*policy.*cover", r"what.*insurance.*cover", r"covered.*under"
    ],
    "limit": [
        r"maximum.*payout", r"limit.*amount", r"deductible", r"cap.*amount",
        r"up.*to.*amount", r"maximum.*benefit", r"limit.*coverage"
    ],
    "definition": [
        r"what.*mean", r"define", r"definition", r"explain.*what",
        r"what.*is.*copay", r"what.*is.*deductible", r"term.*mean"
    ],
    "general_info": [
        r"tell.*about", r"information.*policy", r"how.*work", r"what.*policy"
    ]
}

# Entity extraction patterns
ENTITY_PATTERNS = {
    "conditions": [
        r"condition", r"diagnosis", r"illness", r"disease", r"health", r"mental",
        r"pre-existing", r"chronic", r"acute"
    ],
    "amounts": [
        r"\$[\d,]+", r"dollar", r"amount", r"percent", r"cost", r"price", r"fee",
        r"deductible", r"copay", r"premium"
    ],
    "benefits": [
        r"benefit", r"service", r"treatment", r"therapy", r"medication", r"procedure",
        r"doctor.*visit", r"hospital", r"emergency"
    ]
}


class FreeQueryProcessor:
    """
    Free alternative to OpenAI-based query processing using sentence transformers
    and rule-based classification.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 use_semantic_search: bool = True,
                 similarity_threshold: float = 0.7):
        """
        Initialize the free query processor.
        
        Args:
            model_name: Sentence transformer model name
            use_semantic_search: Whether to use semantic search for classification
            similarity_threshold: Threshold for semantic similarity
        """
        self.model_name = model_name
        self.use_semantic_search = use_semantic_search
        self.similarity_threshold = similarity_threshold
        
        # Initialize sentence transformer
        try:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer model: {e}")
            raise
        
        # Pre-computed embeddings for common query patterns
        self.pattern_embeddings = self._initialize_pattern_embeddings()
        
        logger.info(f"Initialized FreeQueryProcessor with {model_name}")
    
    def _initialize_pattern_embeddings(self) -> Dict[str, np.ndarray]:
        """Initialize embeddings for common query patterns."""
        pattern_embeddings = {}
        
        # Create example queries for each intent
        example_queries = {
            "coverage": [
                "What is covered under my health insurance?",
                "What benefits are included in my policy?",
                "What does my insurance cover?",
                "What services are covered?"
            ],
            "limit": [
                "What is the maximum payout?",
                "What is my deductible?",
                "What are the coverage limits?",
                "How much is the maximum benefit?"
            ],
            "definition": [
                "What is a copay?",
                "Define pre-existing condition",
                "What does deductible mean?",
                "Explain what a premium is"
            ],
            "general_info": [
                "Tell me about my insurance policy",
                "How does my insurance work?",
                "What information do you have about my policy?",
                "General information about coverage"
            ]
        }
        
        for intent, queries in example_queries.items():
            try:
                embeddings = self.model.encode(queries)
                pattern_embeddings[intent] = embeddings
                logger.debug(f"Initialized embeddings for {intent} intent")
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings for {intent}: {e}")
        
        return pattern_embeddings
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse and classify an insurance query using free models.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with intent, entities, and raw_query
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return {
                "intent": "general_info",
                "entities": [],
                "raw_query": query,
                "method": "fallback"
            }
        
        try:
            # Try semantic classification first
            if self.use_semantic_search:
                intent = self._semantic_classify(query)
                if intent:
                    entities = self._extract_entities(query)
                    return {
                        "intent": intent,
                        "entities": entities,
                        "raw_query": query,
                        "method": "semantic"
                    }
            
            # Fall back to rule-based classification
            intent = self._rule_based_classify(query)
            entities = self._extract_entities(query)
            
            return {
                "intent": intent,
                "entities": entities,
                "raw_query": query,
                "method": "rule_based"
            }
            
        except Exception as e:
            logger.error(f"Failed to parse query '{query}': {e}")
            return self._fallback_classification(query)
    
    def _semantic_classify(self, query: str) -> Optional[str]:
        """
        Classify query using semantic similarity with pre-computed patterns.
        
        Args:
            query: User query string
            
        Returns:
            Intent classification or None if below threshold
        """
        try:
            # Encode the query
            query_embedding = self.model.encode([query])
            
            best_intent = None
            best_similarity = 0.0
            
            # Compare with pattern embeddings
            for intent, pattern_embeddings in self.pattern_embeddings.items():
                if len(pattern_embeddings) == 0:
                    continue
                
                # Calculate similarities using numpy
                similarities = util.pytorch_cos_sim(query_embedding, pattern_embeddings)
                similarities_np = similarities.cpu().numpy()
                max_similarity = float(np.max(similarities_np))
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_intent = intent
            
            # Return intent if similarity is above threshold
            if best_similarity >= self.similarity_threshold:
                logger.debug(f"Semantic classification: {query[:30]}... -> {best_intent} (similarity: {best_similarity:.3f})")
                return best_intent
            
            return None
            
        except Exception as e:
            logger.warning(f"Semantic classification failed: {e}")
            return None
    
    def _rule_based_classify(self, query: str) -> str:
        """
        Classify query using rule-based pattern matching.
        
        Args:
            query: User query string
            
        Returns:
            Intent classification
        """
        query_lower = query.lower()
        
        # Check patterns for each intent
        for intent, patterns in QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    logger.debug(f"Rule-based classification: {query[:30]}... -> {intent}")
                    return intent
        
        # Default to general_info
        return "general_info"
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from query using pattern matching.
        
        Args:
            query: User query string
            
        Returns:
            List of extracted entities
        """
        query_lower = query.lower()
        entities = []
        
        for entity_type, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    if entity_type not in entities:
                        entities.append(entity_type)
                    break
        
        return entities
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """
        Provide fallback classification when all methods fail.
        
        Args:
            query: Original query string
            
        Returns:
            Fallback classification dictionary
        """
        return {
            "intent": "general_info",
            "entities": [],
            "raw_query": query,
            "method": "fallback"
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the free query processor."""
        return {
            "model_name": self.model_name,
            "use_semantic_search": self.use_semantic_search,
            "similarity_threshold": self.similarity_threshold,
            "available_intents": list(QUERY_PATTERNS.keys()),
            "available_entities": list(ENTITY_PATTERNS.keys()),
            "pattern_embeddings_loaded": len(self.pattern_embeddings) > 0
        }


class SemanticSearchProcessor:
    """
    Semantic search processor using sentence transformers for finding similar content.
    """
    
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        """
        Initialize semantic search processor.
        
        Args:
            model_name: Sentence transformer model optimized for QA
        """
        self.model_name = model_name
        
        try:
            logger.info(f"Loading semantic search model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load semantic search model: {e}")
            raise
    
    def find_similar_content(self, 
                           query: str, 
                           documents: List[str], 
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar content using semantic search.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            # Encode query and documents
            query_embedding = self.model.encode([query])
            document_embeddings = self.model.encode(documents)
            
            # Calculate similarities
            similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)
            similarities_np = similarities[0].cpu().numpy()
            
            # Get top-k results
            top_indices = np.argsort(similarities_np)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities_np[idx] > 0:  # Only include positive similarities
                    results.append({
                        "document": documents[idx],
                        "similarity_score": float(similarities_np[idx]),
                        "rank": len(results) + 1
                    })
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the semantic search model."""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None
        }


# Convenience functions
def parse_query_free(query: str) -> Dict[str, Any]:
    """
    Parse and classify an insurance query using free models.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with intent, entities, and raw_query
    """
    processor = FreeQueryProcessor()
    return processor.parse_query(query)


def semantic_search(query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search using free models.
    
    Args:
        query: Search query
        documents: List of document texts
        top_k: Number of top results to return
        
    Returns:
        List of similar documents with similarity scores
    """
    processor = SemanticSearchProcessor()
    return processor.find_similar_content(query, documents, top_k)


# Example usage and testing
if __name__ == "__main__":
    # Test the free query processor
    processor = FreeQueryProcessor()
    
    test_queries = [
        "What is covered under my health insurance?",
        "What is the maximum payout for dental procedures?",
        "Define pre-existing condition",
        "How much does a doctor visit cost?",
        "What benefits are included for mental health?"
    ]
    
    print("Testing Free Query Processor:")
    print("=" * 50)
    
    for query in test_queries:
        result = processor.parse_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']} (method: {result['method']})")
        print(f"Entities: {result['entities']}")
    
    print(f"\nProcessor Statistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
