"""
Hybrid Query Processor - Intelligently routes between local and API processing
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .query_processing import QueryProcessor
from .retrieval import ClauseRetrieval
from .logic_evaluator import LogicEvaluator

logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    """Processing strategy for different query types"""
    LOCAL_ONLY = "local_only"
    FREE_API = "free_api"
    HYBRID = "hybrid"
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
        
        # Initialize processors
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
        
        # Strategy selection weights
        self.strategy_weights = {
            ProcessingStrategy.HF_ENHANCED: 1.0,  # Highest priority
            ProcessingStrategy.LOCAL_ONLY: 0.9,
            ProcessingStrategy.FREE_API: 0.8,
            ProcessingStrategy.HYBRID: 0.7,
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
            
            # Process based on strategy
            if strategy == ProcessingStrategy.HF_ENHANCED:
                result = self._process_hf_enhanced(query, embedding_system)
            elif strategy == ProcessingStrategy.LOCAL_ONLY:
                result = self._process_local_only(query, embedding_system)
            elif strategy == ProcessingStrategy.FREE_API:
                result = self._process_free_api(query, embedding_system)
            elif strategy == ProcessingStrategy.HYBRID:
                result = self._process_hybrid(query, embedding_system)
            else:
                result = self._process_fallback(query, embedding_system)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                answer=result.get("answer", "Unable to process query"),
                confidence=result.get("confidence", 0.0),
                strategy_used=strategy,
                processing_time=processing_time,
                source_clauses=result.get("source_clauses", []),
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                answer=f"Error processing query: {str(e)}",
                confidence=0.0,
                strategy_used=ProcessingStrategy.FALLBACK,
                processing_time=processing_time,
                source_clauses=[],
                metadata={"error": str(e)}
            )
    
    def _select_strategy(self, query: str) -> ProcessingStrategy:
        """
        Select the best processing strategy based on query characteristics
        """
        # Check if enhanced HF service is available
        if (self.hf_enhanced and self.hf_enhanced.is_available()):
            return ProcessingStrategy.HF_ENHANCED
        
        # Check if free API services are available
        if self.api_processor.is_available():
            return ProcessingStrategy.FREE_API
        
        # Fallback to local processing
        return ProcessingStrategy.LOCAL_ONLY
    
    def _process_hf_enhanced(self, query: str, embedding_system=None) -> Dict[str, Any]:
        """Process using enhanced Hugging Face service"""
        try:
            if not self.hf_enhanced or not self.hf_enhanced.is_available():
                raise Exception("Enhanced HF service not available")
            
            # Get relevant chunks if embedding system is provided
            source_clauses = []
            if embedding_system:
                search_results = embedding_system.search(query, top_k=5)
                source_clauses = [result['document']['text'] for result in search_results]
            
            # Process with enhanced HF service
            result = self.hf_enhanced.process_insurance_query(query, source_clauses)
            
            return {
                "answer": result.get("answer", "No answer generated"),
                "confidence": result.get("confidence", 0.8),
                "source_clauses": source_clauses,
                "metadata": {
                    "model": "hf_enhanced",
                    "strategy": "enhanced_hf"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced HF processing: {e}")
            raise
    
    def _process_local_only(self, query: str, embedding_system=None) -> Dict[str, Any]:
        """Process using local models only"""
        try:
            # Get relevant chunks if embedding system is provided
            source_clauses = []
            if embedding_system:
                search_results = embedding_system.search(query, top_k=5)
                source_clauses = [result['document']['text'] for result in search_results]
            
            # Use basic query processing
            query_analysis = self.api_processor.parse_query(query)
            
            # Simple answer generation
            answer = f"Based on the query analysis ({query_analysis['intent']}), this appears to be a {query_analysis['intent']} type question."
            
            return {
                "answer": answer,
                "confidence": 0.6,
                "source_clauses": source_clauses,
                "metadata": {
                    "model": "local_only",
                    "strategy": "local_only",
                    "query_analysis": query_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error in local processing: {e}")
            raise
    
    def _process_free_api(self, query: str, embedding_system=None) -> Dict[str, Any]:
        """Process using free API services"""
        try:
            # Get relevant chunks if embedding system is provided
            source_clauses = []
            if embedding_system:
                search_results = embedding_system.search(query, top_k=5)
                source_clauses = [result['document']['text'] for result in search_results]
            
            # Use API processor
            query_analysis = self.api_processor.parse_query(query)
            
            # Evaluate decision using logic evaluator
            if source_clauses:
                decision_result = self.evaluator.evaluate_decision(query, source_clauses)
                answer = decision_result.get("answer", "Unable to evaluate decision")
                confidence = decision_result.get("confidence", 0.7)
            else:
                answer = f"Query analyzed as {query_analysis['intent']} type"
                confidence = 0.5
            
            return {
                "answer": answer,
                "confidence": confidence,
                "source_clauses": source_clauses,
                "metadata": {
                    "model": "free_api",
                    "strategy": "free_api",
                    "query_analysis": query_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error in free API processing: {e}")
            raise
    
    def _process_hybrid(self, query: str, embedding_system=None) -> Dict[str, Any]:
        """Process using hybrid approach"""
        try:
            # Try enhanced HF first
            try:
                return self._process_hf_enhanced(query, embedding_system)
            except Exception as e:
                logger.warning(f"Enhanced HF failed, trying free API: {e}")
            
            # Fallback to free API
            return self._process_free_api(query, embedding_system)
            
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            raise
    
    def _process_fallback(self, query: str, embedding_system=None) -> Dict[str, Any]:
        """Fallback processing when all else fails"""
        try:
            return {
                "answer": f"Unable to process query: {query}. Please try again later.",
                "confidence": 0.0,
                "source_clauses": [],
                "metadata": {
                    "model": "fallback",
                    "strategy": "fallback",
                    "error": "All processing strategies failed"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fallback processing: {e}")
            return {
                "answer": "System error occurred during processing",
                "confidence": 0.0,
                "source_clauses": [],
                "metadata": {
                    "model": "fallback",
                    "strategy": "fallback",
                    "error": str(e)
                }
            }
    
    def get_available_strategies(self) -> List[ProcessingStrategy]:
        """Get list of available processing strategies"""
        available = []
        
        if self.hf_enhanced and self.hf_enhanced.is_available():
            available.append(ProcessingStrategy.HF_ENHANCED)
        
        if self.api_processor.is_available():
            available.append(ProcessingStrategy.FREE_API)
        
        available.append(ProcessingStrategy.LOCAL_ONLY)
        available.append(ProcessingStrategy.FALLBACK)
        
        return available
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "available_strategies": [s.value for s in self.get_available_strategies()],
            "enhanced_hf_available": self.hf_enhanced and self.hf_enhanced.is_available(),
            "api_processor_available": self.api_processor.is_available(),
            "strategy_weights": {k.value: v for k, v in self.strategy_weights.items()}
        }
