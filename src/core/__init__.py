"""
Core components for Claimsure LLM-powered query-retrieval system.

This module provides the core functionality for document processing,
embedding generation, vector storage, and query processing.
"""

__version__ = "1.0.0"

# Import main components
from .embeddings import EmbeddingSystem, VectorStore, FAISSVectorStore, PineconeVectorStore
from .query_processing import QueryProcessor, parse_query
from .free_models import FreeQueryProcessor, SemanticSearchProcessor, parse_query_free, semantic_search
from .retrieval import ClauseRetrieval, retrieve_clauses, retrieve_clauses_batch
from .logic_evaluator import LogicEvaluator, evaluate_decision
from .hybrid_processor import HybridProcessor, ProcessingStrategy, ProcessingResult, process_query_hybrid

__all__ = [
    # Embedding system
    "EmbeddingSystem",
    "VectorStore",
    "FAISSVectorStore",
    "PineconeVectorStore",

    # Query processing
    "QueryProcessor",
    "parse_query",

    # Free models
    "FreeQueryProcessor",
    "SemanticSearchProcessor",
    "parse_query_free",
    "semantic_search",

    # Clause retrieval
    "ClauseRetrieval",
    "retrieve_clauses",
    "retrieve_clauses_batch",

    # Logic evaluation
    "LogicEvaluator",
    "evaluate_decision",

    # Hybrid processing
    "HybridProcessor",
    "ProcessingStrategy",
    "ProcessingResult",
    "process_query_hybrid",
]
