import os
import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiStatus(Enum):
    """Status of Gemini service"""
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

class GeminiService:
    """
    Manages interactions with the Google Gemini API.
    Handles API key loading, rate limiting, retries, and intelligent fallbacks.
    """
    
    def __init__(self):
        """Initialize Gemini service with API key and configuration"""
        self.api_key = os.getenv('GOOGLE_AI_API_KEY')
        self.model = None
        self.status = GeminiStatus.UNAVAILABLE
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # seconds between requests
        self.max_retries = 3
        self.total_requests = 0
        self.total_tokens_used = 0
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                # Test the connection
                self._test_connection()
            except Exception as e:
                logging.error(f"Failed to initialize Gemini client: {e}")
                self.status = GeminiStatus.ERROR
        else:
            logging.warning("No Google AI API key found. Gemini service unavailable.")
    
    def _test_connection(self):
        """Test the API connection with a simple request"""
        try:
            # Simple test message
            response = self.model.generate_content("Hello")
            self.status = GeminiStatus.AVAILABLE
            logging.info("Gemini service initialized successfully")
        except Exception as e:
            logging.error(f"Gemini connection test failed: {e}")
            self.status = GeminiStatus.ERROR
    
    def get_status(self) -> GeminiStatus:
        """Get current service status"""
        return self.status
    
    def is_available(self) -> bool:
        """Check if service is available for use"""
        return self.status == GeminiStatus.AVAILABLE
    
    def process_insurance_query(self, 
                              query: str, 
                              context: Optional[str] = None,
                              claim_details: Optional[Dict] = None) -> Tuple[str, float, GeminiStatus]:
        """
        Process an insurance-related query using Gemini
        
        Args:
            query: User's insurance question
            context: Relevant document context/clauses
            claim_details: Additional claim information
            
        Returns:
            Tuple of (response, confidence_score, status)
        """
        if not self.is_available():
            return "", 0.0, self.status
        
        # Prepare the prompt
        prompt = self._prepare_insurance_prompt(query, context, claim_details)
        
        try:
            # Rate limiting
            self._enforce_rate_limit()
            
            # Make API call
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000
                )
            )
            
            # Update statistics
            self.total_requests += 1
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_tokens_used += response.usage_metadata.total_token_count
            
            # Extract response content
            response_text = response.text if response.text else ""
            
            # Calculate confidence (simplified - could be enhanced)
            confidence = self._calculate_confidence(response_text, query)
            
            self.status = GeminiStatus.AVAILABLE
            return response_text, confidence, self.status
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "quota" in error_str:
                self.status = GeminiStatus.RATE_LIMITED
                logging.warning("Gemini API rate limit or quota exceeded")
            else:
                self.status = GeminiStatus.ERROR
                logging.error(f"Gemini API error: {e}")
            return "", 0.0, self.status
    
    def batch_process_queries(self, 
                            queries: List[str], 
                            contexts: Optional[List[str]] = None) -> List[Tuple[str, float, GeminiStatus]]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of insurance queries
            contexts: Optional list of contexts for each query
            
        Returns:
            List of (response, confidence, status) tuples
        """
        if not self.is_available():
            return [("", 0.0, self.status)] * len(queries)
        
        results = []
        for i, query in enumerate(queries):
            context = contexts[i] if contexts and i < len(contexts) else None
            result = self.process_insurance_query(query, context)
            results.append(result)
            
            # Add delay between requests to respect rate limits
            if i < len(queries) - 1:
                time.sleep(self.rate_limit_delay)
        
        return results
    
    def _prepare_insurance_prompt(self, 
                                 query: str, 
                                 context: Optional[str] = None,
                                 claim_details: Optional[Dict] = None) -> str:
        """Prepare a comprehensive prompt for insurance queries"""
        
        prompt = f"""You are an expert insurance claims analyst. Analyze the following query and provide a clear, accurate response based on the provided context.

USER QUERY: {query}

"""
        
        if context:
            prompt += f"RELEVANT DOCUMENT CONTEXT:\n{context}\n\n"
        
        if claim_details:
            prompt += f"CLAIM DETAILS:\n"
            for key, value in claim_details.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        prompt += """INSTRUCTIONS:
1. Analyze the query carefully and provide a comprehensive answer
2. If the context contains relevant information, reference it specifically
3. If information is missing or unclear, state what additional information is needed
4. Provide actionable advice when possible
5. Use clear, professional language suitable for insurance professionals
6. If the query involves calculations, show your work
7. Always prioritize accuracy over speculation

RESPONSE:"""
        
        return prompt
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _calculate_confidence(self, response: str, query: str) -> float:
        """
        Calculate confidence score for the response
        This is a simplified implementation - could be enhanced with more sophisticated analysis
        """
        if not response or len(response.strip()) < 10:
            return 0.0
        
        # Basic confidence indicators
        confidence_indicators = [
            "based on the context",
            "according to the document",
            "the policy states",
            "clearly indicates",
            "specifically mentions"
        ]
        
        response_lower = response.lower()
        confidence_score = 0.5  # Base confidence
        
        # Boost confidence for specific references
        for indicator in confidence_indicators:
            if indicator in response_lower:
                confidence_score += 0.1
        
        # Boost confidence for longer, detailed responses
        if len(response) > 100:
            confidence_score += 0.1
        
        # Boost confidence for structured responses
        if any(char in response for char in ['•', '-', '1.', '2.', '3.']):
            confidence_score += 0.1
        
        return min(confidence_score, 1.0)
    
    def get_service_statistics(self) -> Dict:
        """Get service usage statistics"""
        return {
            "status": self.status.value,
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "rate_limit_delay": self.rate_limit_delay,
            "max_retries": self.max_retries
        }
    
    def reset_statistics(self):
        """Reset service statistics"""
        self.total_requests = 0
        self.total_tokens_used = 0

