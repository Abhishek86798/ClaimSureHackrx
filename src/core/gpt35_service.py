"""
Enhanced GPT-3.5 Service for Claimsure.

Provides robust GPT-3.5-turbo integration with free tier management,
rate limiting, intelligent fallbacks, and seamless integration with
the existing hybrid processor.
"""

import logging
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GPT35Status(Enum):
    """Status of GPT-3.5 service"""
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

@dataclass
class GPT35Response:
    """Structured response from GPT-3.5"""
    content: str
    confidence: float
    model_used: str
    tokens_used: int
    processing_time: float
    status: GPT35Status
    error_message: Optional[str] = None

class GPT35Service:
    """
    Enhanced GPT-3.5 service with free tier management and intelligent fallbacks.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 1000,
                 temperature: float = 0.1,
                 max_retries: int = 3,
                 rate_limit_delay: float = 1.0):
        """
        Initialize GPT-3.5 service.
        
        Args:
            api_key: OpenAI API key (will use env var if not provided)
            model: Model to use (default: gpt-3.5-turbo for free tier)
            max_tokens: Maximum tokens for response
            temperature: Response randomness
            max_retries: Maximum retry attempts
            rate_limit_delay: Delay between retries in seconds
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        
        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found - GPT-3.5 service will be unavailable")
            self.client = None
            self.status = GPT35Status.UNAVAILABLE
            return
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.status = GPT35Status.AVAILABLE
            logger.info(f"✅ GPT-3.5 service initialized with model: {model}")
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize GPT-3.5 service: {e}")
            self.client = None
            self.status = GPT35Status.ERROR
    
    def _test_connection(self) -> bool:
        """Test connection to OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            self.status = GPT35Status.AVAILABLE
            logger.info("✅ GPT-3.5 connection test successful")
            return True
        except Exception as e:
            logger.warning(f"GPT-3.5 connection test failed: {e}")
            self.status = GPT35Status.ERROR
            return False
    
    def process_insurance_query(self, 
                               query: str, 
                               context: str,
                               query_type: str = "general") -> GPT35Response:
        """
        Process insurance-related query with GPT-3.5.
        
        Args:
            query: User query
            context: Relevant context/clauses
            query_type: Type of insurance query
            
        Returns:
            GPT35Response with structured results
        """
        start_time = time.time()
        
        if not self.client or self.status == GPT35Status.UNAVAILABLE:
            return GPT35Response(
                content="GPT-3.5 service is not available.",
                confidence=0.0,
                model_used="none",
                tokens_used=0,
                processing_time=time.time() - start_time,
                status=GPT35Status.UNAVAILABLE,
                error_message="Service not initialized"
            )
        
        # Create insurance-specific prompt
        prompt = self._create_insurance_prompt(query, context, query_type)
        
        # Process with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert insurance claims analyst. Provide accurate, helpful information based on the given context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                # Extract response
                content = response.choices[0].message.content.strip()
                tokens_used = response.usage.total_tokens if response.usage else 0
                
                # Calculate confidence based on response quality
                confidence = self._calculate_confidence(content, context, query)
                
                processing_time = time.time() - start_time
                
                # Reset status on success
                self.status = GPT35Status.AVAILABLE
                
                logger.info(f"✅ GPT-3.5 query processed successfully in {processing_time:.2f}s")
                
                return GPT35Response(
                    content=content,
                    confidence=confidence,
                    model_used=self.model,
                    tokens_used=tokens_used,
                    processing_time=processing_time,
                    status=GPT35Status.AVAILABLE
                )
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"GPT-3.5 attempt {attempt + 1} failed: {error_msg}")
                
                # Handle specific error types
                if "rate_limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    self.status = GPT35Status.RATE_LIMITED
                    if attempt < self.max_retries - 1:
                        time.sleep(self.rate_limit_delay * (attempt + 1))  # Exponential backoff
                        continue
                elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                    self.status = GPT35Status.QUOTA_EXCEEDED
                    break
                else:
                    self.status = GPT35Status.ERROR
                    if attempt < self.max_retries - 1:
                        time.sleep(self.rate_limit_delay)
                        continue
        
        # All retries failed
        processing_time = time.time() - start_time
        return GPT35Response(
            content="I'm unable to process your query at the moment. Please try again later.",
            confidence=0.0,
            model_used=self.model,
            tokens_used=0,
            processing_time=processing_time,
            status=self.status,
            error_message=f"Failed after {self.max_retries} attempts"
        )
    
    def _create_insurance_prompt(self, query: str, context: str, query_type: str) -> str:
        """Create insurance-specific prompt for GPT-3.5"""
        
        base_prompt = f"""Context: {context}

User Query: {query}

Please provide a clear, accurate answer based on the context provided. Focus on:
1. Direct answers to the user's question
2. Specific information from the context
3. Clear explanations of coverage, limits, or requirements
4. Professional and helpful tone

Answer:"""
        
        # Add query-type specific instructions
        if query_type == "coverage":
            base_prompt += "\n\nFocus on what is and is not covered under the policy."
        elif query_type == "claims":
            base_prompt += "\n\nFocus on the claims process, requirements, and procedures."
        elif query_type == "limits":
            base_prompt += "\n\nFocus on coverage limits, deductibles, and maximum payouts."
        
        return base_prompt
    
    def _calculate_confidence(self, response: str, context: str, query: str) -> float:
        """Calculate confidence score based on response quality"""
        confidence = 0.5  # Base confidence
        
        # Check response length (too short = low confidence)
        if len(response) < 50:
            confidence -= 0.2
        elif len(response) > 200:
            confidence += 0.1
        
        # Check if response directly addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        word_overlap = len(query_words.intersection(response_words))
        if word_overlap > 0:
            confidence += min(0.2, word_overlap * 0.05)
        
        # Check if response references context
        if any(word in response.lower() for word in context.lower().split()[:10]):
            confidence += 0.1
        
        # Check for uncertainty indicators
        uncertainty_words = ["maybe", "possibly", "might", "could", "uncertain", "unclear"]
        if any(word in response.lower() for word in uncertainty_words):
            confidence -= 0.1
        
        # Ensure confidence is between 0.0 and 1.0
        return max(0.0, min(1.0, confidence))
    
    def batch_process_queries(self, 
                             queries: List[Tuple[str, str, str]]) -> List[GPT35Response]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of (query, context, query_type) tuples
            
        Returns:
            List of GPT35Response objects
        """
        responses = []
        
        for i, (query, context, query_type) in enumerate(queries):
            try:
                logger.info(f"Processing batch query {i+1}/{len(queries)}")
                response = self.process_insurance_query(query, context, query_type)
                responses.append(response)
                
                # Add delay between requests to respect rate limits
                if i < len(queries) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Failed to process batch query {i+1}: {e}")
                responses.append(GPT35Response(
                    content="Processing failed for this query.",
                    confidence=0.0,
                    model_used=self.model,
                    tokens_used=0,
                    processing_time=0.0,
                    status=GPT35Status.ERROR,
                    error_message=str(e)
                ))
        
        return responses
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and statistics"""
        return {
            "status": self.status.value,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key_configured": bool(self.api_key),
            "client_available": self.client is not None,
            "max_retries": self.max_retries,
            "rate_limit_delay": self.rate_limit_delay
        }
    
    def reset_status(self) -> bool:
        """Reset service status and test connection"""
        try:
            if self.client:
                self.status = GPT35Status.AVAILABLE
                return self._test_connection()
            return False
        except Exception as e:
            logger.error(f"Failed to reset status: {e}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if the GPT-3.5 service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        return self.status == GPT35Status.AVAILABLE and self.client is not None

# Convenience function for direct query processing
def process_with_gpt35(query: str, context: str, query_type: str = "general") -> GPT35Response:
    """
    Convenience function to process a query with GPT-3.5.
    
    Args:
        query: User query
        context: Relevant context
        query_type: Type of insurance query
        
    Returns:
        GPT35Response with results
    """
    service = GPT35Service()
    return service.process_insurance_query(query, context, query_type)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Enhanced GPT-3.5 Service")
    print("=" * 50)
    
    # Initialize service
    service = GPT35Service()
    
    print(f"Service Status: {service.get_service_status()}")
    
    if service.status == GPT35Status.AVAILABLE:
        # Test with sample insurance query
        test_query = "What is covered under my health insurance policy?"
        test_context = """
        Your health insurance policy covers:
        - Doctor visits and consultations
        - Hospital stays and surgeries
        - Prescription medications
        - Emergency room visits
        - Preventive care and screenings
        
        Exclusions:
        - Cosmetic procedures
        - Experimental treatments
        - Dental and vision (separate coverage)
        """
        
        print(f"\n📝 Testing with query: {test_query}")
        print(f"   Context length: {len(test_context)} characters")
        
        # Process query
        result = service.process_insurance_query(test_query, test_context, "coverage")
        
        print(f"\n📊 Results:")
        print(f"   Status: {result.status.value}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Tokens Used: {result.tokens_used}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        print(f"   Model: {result.model_used}")
        
        if result.error_message:
            print(f"   Error: {result.error_message}")
        else:
            print(f"\n💬 Response:")
            print(f"   {result.content}")
    
    else:
        print(f"❌ Service not available: {service.status.value}")
        if service.status == GPT35Status.UNAVAILABLE:
            print("   Please check your OPENAI_API_KEY environment variable")
        elif service.status == GPT35Status.RATE_LIMITED:
            print("   Service is currently rate limited. Please wait and try again.")
        elif service.status == GPT35Status.QUOTA_EXCEEDED:
            print("   API quota exceeded. Please check your OpenAI account.")

