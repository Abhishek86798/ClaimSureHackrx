import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .claude_service import ClaudeService, ClaudeStatus
from .gemini_service import GeminiService, GeminiStatus
from .huggingface_service import HuggingFaceService, HuggingFaceStatus
from .local_mistral_service import LocalMistralService, LocalMistralStatus
from .gpt35_service import GPT35Service, GPT35Status
from .retrieval import ClauseRetrieval
from .logic_evaluator import LogicEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    """Enhanced processing strategies"""
    CLAUDE = "claude"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    LOCAL_MISTRAL = "local_mistral"
    GPT35 = "gpt35"
    HYBRID = "hybrid"
    FALLBACK = "fallback"

@dataclass
class ProcessingResult:
    """Result of processing a query"""
    response: str
    confidence: float
    strategy_used: ProcessingStrategy
    service_name: str
    processing_time: float
    fallback_used: bool = False

class EnhancedHybridProcessor:
    """Enhanced hybrid processor with all LLM services"""
    
    def __init__(self, embedding_system=None):
        self.embedding_system = embedding_system
        self.retrieval = None
        
        # Initialize all LLM services
        self.claude_service = ClaudeService()
        self.gemini_service = GeminiService()
        self.huggingface_service = HuggingFaceService()
        self.local_mistral_service = LocalMistralService()
        self.gpt35_service = GPT35Service()
        
        # Initialize other components
        self.logic_evaluator = LogicEvaluator()
        
        # Get available services
        self.available_services = self._get_available_services()
        
        logger.info(f"Enhanced Hybrid Processor initialized with services: {self.available_services}")
    
    def _get_available_services(self) -> List[str]:
        """Get list of available services"""
        services = []
        if self.claude_service.is_available():
            services.append("claude")
        if self.gemini_service.is_available():
            services.append("gemini")
        if self.huggingface_service.is_available():
            services.append("huggingface")
        if self.local_mistral_service.is_available():
            services.append("local_mistral")
        if self.gpt35_service.is_available():
            services.append("gpt35")
        return services
    
    def process_query(self, query: str, claim_details: Optional[Dict] = None) -> ProcessingResult:
        """Process a query using the enhanced hybrid approach"""
        start_time = time.time()
        
        try:
            # Initialize retrieval if needed
            if self.retrieval is None and self.embedding_system:
                self.retrieval = ClauseRetrieval(self.embedding_system)
            
            # Retrieve relevant clauses
            context = self._retrieve_context(query) if self.retrieval else None
            
            # Select and execute strategy
            strategy = self._select_strategy(query)
            result = self._process_with_strategy(strategy, query, context, claim_details)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                response=result[0],
                confidence=result[1],
                strategy_used=strategy,
                service_name=strategy.value,
                processing_time=processing_time,
                fallback_used=False
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = time.time() - start_time
            
            # Use fallback
            fallback_result = self._process_fallback(query, claim_details)
            
            return ProcessingResult(
                response=fallback_result[0],
                confidence=fallback_result[1],
                strategy_used=ProcessingStrategy.FALLBACK,
                service_name="fallback",
                processing_time=processing_time,
                fallback_used=True
            )
    
    def _retrieve_context(self, query: str) -> Optional[str]:
        """Retrieve relevant document context"""
        if not self.retrieval:
            return None
        
        try:
            chunks = self.retrieval.retrieve_relevant_chunks(query, top_k=5)
            if chunks:
                context_parts = []
                for chunk in chunks:
                    source = chunk.get("source", "unknown")
                    content = chunk.get("content", "")
                    context_parts.append(f"Source: {source}\nContent: {content}")
                return "\n\n".join(context_parts)
            return None
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return None
    
    def _select_strategy(self, query: str) -> ProcessingStrategy:
        """Select processing strategy"""
        if not self.available_services:
            return ProcessingStrategy.FALLBACK
        
        # Prioritize Claude for complex queries
        query_type = self._classify_query_type(query)
        if query_type in ["claims", "limits", "coverage"] and "claude" in self.available_services:
            return ProcessingStrategy.CLAUDE
        
        # Use first available service
        if "claude" in self.available_services:
            return ProcessingStrategy.CLAUDE
        elif "gemini" in self.available_services:
            return ProcessingStrategy.GEMINI
        elif "huggingface" in self.available_services:
            return ProcessingStrategy.HUGGINGFACE
        elif "local_mistral" in self.available_services:
            return ProcessingStrategy.LOCAL_MISTRAL
        elif "gpt35" in self.available_services:
            return ProcessingStrategy.GPT35
        
        return ProcessingStrategy.FALLBACK
    
    def _process_with_strategy(self, strategy: ProcessingStrategy, query: str, context: Optional[str], claim_details: Optional[Dict]) -> Tuple[str, float]:
        """Process using selected strategy"""
        if strategy == ProcessingStrategy.CLAUDE:
            return self.claude_service.process_insurance_query(query, context, claim_details)[:2]
        elif strategy == ProcessingStrategy.GEMINI:
            return self.gemini_service.process_insurance_query(query, context, claim_details)[:2]
        elif strategy == ProcessingStrategy.HUGGINGFACE:
            return self.huggingface_service.process_insurance_query(query, context, claim_details)[:2]
        elif strategy == ProcessingStrategy.LOCAL_MISTRAL:
            return self.local_mistral_service.process_insurance_query(query, context, claim_details)[:2]
        elif strategy == ProcessingStrategy.GPT35:
            return self.gpt35_service.process_insurance_query(query, context, claim_details)[:2]
        else:
            return self._process_fallback(query, claim_details)
    
    def _process_fallback(self, query: str, claim_details: Optional[Dict]) -> Tuple[str, float]:
        """Process using fallback logic"""
        try:
            response = self.logic_evaluator.evaluate_query(query, claim_details or {})
            return response, 0.5
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return "I'm sorry, I'm unable to process your request at the moment. Please try again later.", 0.0
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["limit", "maximum", "cap", "ceiling"]):
            return "limits"
        elif any(word in query_lower for word in ["claim", "file", "submit", "process"]):
            return "claims"
        elif any(word in query_lower for word in ["cover", "coverage", "policy", "protection"]):
            return "coverage"
        elif any(word in query_lower for word in ["cost", "price", "fee", "premium", "deductible"]):
            return "costs"
        else:
            return "general"
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "available_services": self.available_services,
            "service_status": {
                "claude": self.claude_service.get_status().value,
                "gemini": self.gemini_service.get_status().value,
                "huggingface": self.huggingface_service.get_status().value,
                "local_mistral": self.local_mistral_service.get_status().value,
                "gpt35": self.gpt35_service.get_status().value
            },
            "retrieval_available": self.retrieval is not None,
            "embedding_system_available": self.embedding_system is not None
        }
