"""
Hybrid Query Processor - Intelligently routes between local and API processing
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .query_processing import QueryProcessor
from .free_models import FreeQueryProcessor
from .retrieval import ClauseRetrieval
from .logic_evaluator import LogicEvaluator
from .gpt35_service import GPT35Service, GPT35Status

logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    """Processing strategy for different query types"""
    LOCAL_ONLY = "local_only"
    FREE_API = "free_api"
    HYBRID = "hybrid"
    GPT35 = "gpt35"
    HF_ENHANCED = "hf_enhanced"
    FALLBACK = "fallback"

@dataclass
class ProcessingResult:
    """Result from hybrid processing"""
    answer: str
    confidence: float
    strategy_used: ProcessingStrategy
    processing_time: float
    source_clauses: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class HybridProcessor:
    """
    Intelligently routes queries between local processing and free API models
    based on complexity, confidence, and available resources.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 complexity_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.complexity_threshold = complexity_threshold
        
        # Initialize all processors
        self.local_processor = FreeQueryProcessor()
        self.api_processor = QueryProcessor()
        # Initialize retrieval with a placeholder - will be set when processing
        self.retrieval = None
        self.evaluator = LogicEvaluator()
        
        # Initialize enhanced HF service
        try:
            from .hf_enhanced_service import HFEnhancedService
            self.hf_enhanced = HFEnhancedService()
            logger.info("✅ Enhanced HF service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced HF service: {e}")
            self.hf_enhanced = None
        
        # Initialize GPT-3.5 service
        try:
            self.gpt35_service = GPT35Service()
            logger.info("✅ GPT-3.5 service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize GPT-3.5 service: {e}")
            self.gpt35_service = None
        
        # Strategy selection weights
        self.strategy_weights = {
            ProcessingStrategy.HF_ENHANCED: 1.0,  # Highest priority
            ProcessingStrategy.LOCAL_ONLY: 0.9,
            ProcessingStrategy.FREE_API: 0.8,
            ProcessingStrategy.HYBRID: 0.7,
            ProcessingStrategy.GPT35: 0.6,
            ProcessingStrategy.FALLBACK: 0.5
        }
    
    def process_query(self, 
                     query: str, 
                     embedding_system=None,
                     force_strategy: Optional[ProcessingStrategy] = None) -> ProcessingResult:
        """
        Process a query using the best available strategy
        """
        start_time = time.time()
        
        try:
            # Determine processing strategy
            if force_strategy:
                strategy = force_strategy
            else:
                strategy = self._select_strategy(query)
            
            logger.info(f"Processing query with strategy: {strategy.value}")
            
            # Execute processing based on strategy
            if strategy == ProcessingStrategy.LOCAL_ONLY:
                result = self._process_local_only(query, embedding_system)
            elif strategy == ProcessingStrategy.FREE_API:
                result = self._process_free_api(query, embedding_system)
            elif strategy == ProcessingStrategy.HYBRID:
                result = self._process_hybrid(query, embedding_system)
            elif strategy == ProcessingStrategy.GPT35:
                result = self._process_gpt35(query, embedding_system)
            elif strategy == ProcessingStrategy.HF_ENHANCED:
                result = self._process_hf_enhanced(query, embedding_system)
            else:  # FALLBACK
                result = self._process_fallback(query, embedding_system)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                answer=result["answer"],
                confidence=result["confidence"],
                strategy_used=strategy,
                processing_time=processing_time,
                source_clauses=result.get("source_clauses", []),
                metadata={
                    "strategy": strategy.value,
                    "processing_time": processing_time,
                    "model_used": result.get("model", "local"),
                    "clauses_retrieved": len(result.get("source_clauses", [])),
                    "query_complexity": self._assess_complexity(query)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            return self._process_fallback(query, embedding_system, start_time)
    
    def _select_strategy(self, query: str) -> ProcessingStrategy:
        """
        Intelligently select the best processing strategy
        """
        complexity = self._assess_complexity(query)
        
        # Check if enhanced HF service is available (highest priority)
        if (self.hf_enhanced and 
            self.hf_enhanced.is_available()):
            return ProcessingStrategy.HF_ENHANCED
        
        # Check if GPT-3.5 is available for complex queries
        if (self.gpt35_service and 
            self.gpt35_service.status == GPT35Status.AVAILABLE and
            complexity > 0.7):
            return ProcessingStrategy.GPT35
        
        # Simple queries: use local processing
        if complexity < 0.3:
            return ProcessingStrategy.LOCAL_ONLY
        
        # Complex queries: try free API first
        if complexity > self.complexity_threshold:
            return ProcessingStrategy.FREE_API
        
        # Medium complexity: use hybrid approach
        return ProcessingStrategy.HYBRID
    
    def _assess_complexity(self, query: str) -> float:
        """
        Assess query complexity (0.0 = simple, 1.0 = complex)
        """
        # Simple heuristics for complexity assessment
        complexity_indicators = [
            len(query.split()) > 20,  # Long queries
            any(word in query.lower() for word in ["because", "why", "how", "explain", "analyze"]),
            query.count("?") > 1,  # Multiple questions
            any(word in query.lower() for word in ["policy", "clause", "exclusion", "condition"]),
            query.count("and") > 2 or query.count("or") > 2  # Complex logic
        ]
        
        return sum(complexity_indicators) / len(complexity_indicators)
    
    def _process_local_only(self, query: str, embedding_system) -> Dict[str, Any]:
        """Process query using only local resources"""
        logger.info("Processing with local-only strategy")
        
        # Use free models for query processing
        query_info = self.local_processor.parse_query(query)
        
        # Retrieve relevant clauses
        if embedding_system and self.retrieval:
            clauses = self.retrieval.retrieve_clauses(query, embedding_system)
        elif embedding_system and self.retrieval is None:
            # Initialize retrieval if needed
            from .retrieval import ClauseRetrieval
            self.retrieval = ClauseRetrieval(embedding_system)
            clauses = self.retrieval.retrieve_clauses(query, embedding_system)
        else:
            clauses = []
        
        # Use local evaluation
        result = self.evaluator._fallback_evaluation(query, clauses)
        result["model"] = "local"
        
        return result
    
    def _process_free_api(self, query: str, embedding_system) -> Dict[str, Any]:
        """Process query using free API models"""
        logger.info("Processing with free API strategy")
        
        try:
            # Try to use OpenAI API if available
            if self.api_processor.client:
                query_info = self.api_processor.parse_query(query)
                logger.info(f"API processing successful: {query_info}")
            else:
                query_info = self.local_processor.parse_query(query)
                logger.info("API unavailable, using local processing")
        except Exception as e:
            logger.warning(f"API processing failed: {e}, falling back to local")
            query_info = self.local_processor.parse_query(query)
        
        # Retrieve clauses
        if embedding_system and self.retrieval:
            clauses = self.retrieval.retrieve_clauses(query, embedding_system)
        elif embedding_system and self.retrieval is None:
            # Initialize retrieval if needed
            from .retrieval import ClauseRetrieval
            self.retrieval = ClauseRetrieval(embedding_system)
            clauses = self.retrieval.retrieve_clauses(query, embedding_system)
        else:
            clauses = []
        
        # Try API evaluation first, fallback to local
        try:
            if self.evaluator.client and self.evaluator.available_models:
                result = self.evaluator.evaluate_decision(query, clauses)
            else:
                result = self.evaluator._fallback_evaluation(query, clauses)
        except Exception as e:
            logger.warning(f"API evaluation failed: {e}, using local fallback")
            result = self.evaluator._fallback_evaluation(query, clauses)
        
        return result
    
    def _process_gpt35(self, query: str, embedding_system) -> Dict[str, Any]:
        """
        Process query using GPT-3.5 service with intelligent fallback.
        """
        try:
            if not self.gpt35_service or self.gpt35_service.status != GPT35Status.AVAILABLE:
                logger.warning("GPT-3.5 service not available, falling back to local processing")
                return self._process_local_only(query, embedding_system)
            
            # Initialize retrieval if needed
            if self.retrieval is None and embedding_system:
                from .retrieval import ClauseRetrieval
                self.retrieval = ClauseRetrieval(embedding_system)
            
            # Retrieve relevant clauses first
            if self.retrieval:
                retrieved_chunks = self.retrieval.retrieve_clauses(query, embedding_system)
            else:
                retrieved_chunks = []
            
            if not retrieved_chunks:
                logger.warning("No relevant clauses found for GPT-3.5 processing")
                return {
                    "answer": "I couldn't find any relevant information to answer your query.",
                    "confidence": 0.0,
                    "source_clauses": [],
                    "metadata": {"strategy": "gpt35", "error": "no_clauses_found"}
                }
            
            # Determine query type for better prompting
            query_type = self._classify_query_type(query)
            
            # Prepare context from retrieved chunks
            context = self._prepare_context_for_gpt35(retrieved_chunks)
            
            # Process with GPT-3.5
            gpt35_result = self.gpt35_service.process_insurance_query(
                query, context, query_type
            )
            
            # If GPT-3.5 fails, fall back to local processing
            if gpt35_result.status != GPT35Status.AVAILABLE:
                logger.warning(f"GPT-3.5 processing failed: {gpt35_result.status.value}")
                return self._process_local_only(query, embedding_system)
            
            # Extract source clauses with relevance scores
            source_clauses = []
            for chunk in retrieved_chunks:
                source_clauses.append({
                    "clause_id": chunk.get("clause_id", chunk.get("id", "unknown")),
                    "text": chunk.get("text", chunk.get("content", ""))[:200] + "...",
                    "relevance_score": chunk.get("similarity_score", 0.0),
                    "source": chunk.get("source", "unknown")
                })
            
            return {
                "answer": gpt35_result.content,
                "confidence": gpt35_result.confidence,
                "source_clauses": source_clauses,
                "metadata": {
                    "strategy": "gpt35",
                    "model_used": gpt35_result.model_used,
                    "tokens_used": gpt35_result.tokens_used,
                    "processing_time": gpt35_result.processing_time,
                    "gpt35_status": gpt35_result.status.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error in GPT-3.5 processing: {e}")
            # Fall back to local processing
            return self._process_local_only(query, embedding_system)
    
    def _process_hf_enhanced(self, query: str, embedding_system) -> Dict[str, Any]:
        """
        Process query using enhanced HF models (Mistral, FLAN-T5, Falcon)
        """
        logger.info("Processing with HF enhanced strategy")
        
        if not self.hf_enhanced or not self.hf_enhanced.is_available():
            logger.warning("Enhanced HF service not available, falling back to basic fallback")
            return self._process_fallback(query, embedding_system)
        
        try:
            # Retrieve relevant chunks if embedding system is available
            retrieved_chunks = []
            if embedding_system:
                if not self.retrieval:
                    self.retrieval = ClauseRetrieval(embedding_system)
                retrieved_chunks = self.retrieval.retrieve_clauses(query, top_k=5)
            
            # Prepare context from retrieved chunks
            context = ""
            if retrieved_chunks:
                context_parts = []
                for chunk in retrieved_chunks[:3]:  # Use top 3 chunks
                    context_parts.append(chunk.get("text", ""))
                context = "\n\n".join(context_parts)
            
            # Process with enhanced HF service
            result = self.hf_enhanced.process_insurance_query(query, context)
            
            # Extract source clauses with relevance scores
            source_clauses = []
            for chunk in retrieved_chunks:
                source_clauses.append({
                    "clause_id": chunk.get("clause_id", chunk.get("id", "unknown")),
                    "text": chunk.get("text", chunk.get("content", ""))[:200] + "...",
                    "relevance_score": chunk.get("similarity_score", 0.0),
                    "source": chunk.get("source", "unknown")
                })
            
            return {
                "answer": result["answer"],
                "confidence": result["confidence"],
                "source_clauses": source_clauses,
                "metadata": {
                    "strategy": "hf_enhanced",
                    "model_used": result["model"],
                    "processing_time": time.time(),
                    "hf_enhanced_status": "success"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in HF enhanced processing: {e}")
            return self._process_fallback(query, embedding_system)
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify the type of insurance query for better GPT-3.5 prompting.
        """
        query_lower = query.lower()
        
        # Check for limits first (more specific than coverage)
        if any(word in query_lower for word in ["limit", "maximum", "deductible", "cap", "amount"]):
            return "limits"
        elif any(word in query_lower for word in ["cover", "coverage", "covered", "include", "exclude"]):
            return "coverage"
        elif any(word in query_lower for word in ["claim", "file", "submit", "process", "procedure"]):
            return "claims"
        elif any(word in query_lower for word in ["cost", "price", "fee", "payment", "bill"]):
            return "costs"
        else:
            return "general"
    
    def _prepare_context_for_gpt35(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare context from retrieved chunks for GPT-3.5 processing.
        """
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", chunk.get("content", ""))
            source = chunk.get("source", "unknown")
            similarity = chunk.get("similarity_score", 0.0)
            
            context_parts.append(f"Document {i} (Source: {source}, Relevance: {similarity:.3f}):\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _process_hybrid(self, query: str, embedding_system) -> Dict[str, Any]:
        """Process query using hybrid approach"""
        logger.info("Processing with hybrid strategy")
        
        # Start with local processing
        local_result = self._process_local_only(query, embedding_system)
        
        # If confidence is low, try to enhance with API
        if local_result["confidence"] < self.confidence_threshold:
            try:
                if self.evaluator.client and self.evaluator.available_models:
                    api_result = self.evaluator.evaluate_decision(query, local_result.get("source_clauses", []))
                    
                    # Combine results intelligently
                    if api_result["confidence"] > local_result["confidence"]:
                        api_result["model"] = "hybrid_api_enhanced"
                        return api_result
                    else:
                        local_result["model"] = "hybrid_local_enhanced"
                        return local_result
            except Exception as e:
                logger.warning(f"Hybrid API enhancement failed: {e}")
        
        local_result["model"] = "hybrid_local"
        return local_result
    
    def _process_fallback(self, query: str, embedding_system, start_time=None) -> ProcessingResult:
        """Process query using fallback strategy"""
        logger.info("Processing with fallback strategy")
        
        if start_time is None:
            start_time = time.time()
        
        try:
            # Use local processing as fallback
            result = self._process_local_only(query, embedding_system)
            result["model"] = "fallback_local"
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                answer=result["answer"],
                confidence=result["confidence"],
                strategy_used=ProcessingStrategy.FALLBACK,
                processing_time=processing_time,
                source_clauses=result.get("source_clauses", []),
                metadata={
                    "strategy": "fallback",
                    "processing_time": processing_time,
                    "model_used": "fallback_local",
                    "clauses_retrieved": len(result.get("source_clauses", [])),
                    "query_complexity": self._assess_complexity(query),
                    "error": "Fallback processing used due to system issues"
                }
            )
            
        except Exception as e:
            logger.error(f"Fallback processing also failed: {e}")
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                answer="I apologize, but I'm unable to process your query at the moment. Please try again later.",
                confidence=0.0,
                strategy_used=ProcessingStrategy.FALLBACK,
                processing_time=processing_time,
                source_clauses=[],
                metadata={
                    "strategy": "fallback",
                    "processing_time": processing_time,
                    "model_used": "none",
                    "clauses_retrieved": 0,
                    "query_complexity": self._assess_complexity(query),
                    "error": f"Complete processing failure: {str(e)}"
                }
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about processing performance"""
        return {
            "confidence_threshold": self.confidence_threshold,
            "complexity_threshold": self.complexity_threshold,
            "strategy_weights": {k.value: v for k, v in self.strategy_weights.items()},
            "available_processors": {
                "local": True,
                "api": self.api_processor.client is not None,
                "evaluator": self.evaluator.client is not None and len(self.evaluator.available_models) > 0,
                "gpt35": self.gpt35_service is not None and self.gpt35_service.status == GPT35Status.AVAILABLE,
                "retrieval": self.retrieval is not None
            }
        }

# Convenience function
def process_query_hybrid(query: str, 
                        embedding_system=None,
                        force_strategy: Optional[ProcessingStrategy] = None) -> ProcessingResult:
    """Convenience function for hybrid query processing"""
    processor = HybridProcessor()
    return processor.process_query(query, embedding_system, force_strategy)
