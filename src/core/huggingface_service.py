import os
import time
import logging
import requests
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HuggingFaceStatus(Enum):
    """Status of Hugging Face service"""
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

class HuggingFaceService:
    """
    Manages interactions with the Hugging Face Inference API.
    Handles API key loading, rate limiting, retries, and intelligent fallbacks.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize Hugging Face service with API key and configuration"""
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.status = HuggingFaceStatus.UNAVAILABLE
        self.last_request_time = 0
        self.rate_limit_delay = 2.0  # seconds between requests (HF has stricter limits)
        self.max_retries = 3
        self.total_requests = 0
        self.total_tokens_used = 0
        
        if self.api_key:
            try:
                # Test the connection
                self._test_connection()
            except Exception as e:
                logging.error(f"Failed to initialize Hugging Face client: {e}")
                self.status = HuggingFaceStatus.ERROR
        else:
            logging.warning("No Hugging Face API key found. Service unavailable.")
    
    def _test_connection(self):
        """Test the API connection with a simple request"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"inputs": "Hello"}
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                self.status = HuggingFaceStatus.AVAILABLE
                logging.info(f"Hugging Face service initialized successfully with model: {self.model_name}")
            else:
                self.status = HuggingFaceStatus.ERROR
                logging.error(f"Hugging Face connection test failed with status: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Hugging Face connection test failed: {e}")
            self.status = HuggingFaceStatus.ERROR
    
    def get_status(self) -> HuggingFaceStatus:
        """Get current service status"""
        return self.status
    
    def is_available(self) -> bool:
        """Check if service is available for use"""
        return self.status == HuggingFaceStatus.AVAILABLE
    
    def process_insurance_query(self, 
                              query: str, 
                              context: Optional[str] = None,
                              claim_details: Optional[Dict] = None) -> Tuple[str, float, HuggingFaceStatus]:
        """
        Process an insurance-related query using Hugging Face
        
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
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"inputs": prompt}
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                # Update statistics
                self.total_requests += 1
                
                # Extract response content (handle different response formats)
                response_data = response.json()
                response_text = self._extract_response_text(response_data)
                
                # Calculate confidence
                confidence = self._calculate_confidence(response_text, query)
                
                self.status = HuggingFaceStatus.AVAILABLE
                return response_text, confidence, self.status
                
            elif response.status_code == 429:  # Rate limited
                self.status = HuggingFaceStatus.RATE_LIMITED
                logging.warning("Hugging Face API rate limit exceeded")
                return "", 0.0, self.status
            else:
                self.status = HuggingFaceStatus.ERROR
                logging.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                return "", 0.0, self.status
                
        except requests.exceptions.Timeout:
            self.status = HuggingFaceStatus.ERROR
            logging.error("Hugging Face API request timed out")
            return "", 0.0, self.status
        except Exception as e:
            self.status = HuggingFaceStatus.ERROR
            logging.error(f"Unexpected error in Hugging Face service: {e}")
            return "", 0.0, self.status
    
    def batch_process_queries(self, 
                            queries: List[str], 
                            contexts: Optional[List[str]] = None) -> List[Tuple[str, float, HuggingFaceStatus]]:
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
    
    def _extract_response_text(self, response_data) -> str:
        """Extract text from Hugging Face API response (handles different formats)"""
        try:
            if isinstance(response_data, list) and len(response_data) > 0:
                # Handle list format
                if isinstance(response_data[0], dict) and 'generated_text' in response_data[0]:
                    return response_data[0]['generated_text']
                elif isinstance(response_data[0], str):
                    return response_data[0]
            elif isinstance(response_data, dict):
                # Handle dict format
                if 'generated_text' in response_data:
                    return response_data['generated_text']
                elif 'text' in response_data:
                    return response_data['text']
            elif isinstance(response_data, str):
                return response_data
            
            # Fallback: try to extract any text content
            return str(response_data)
            
        except Exception as e:
            logging.error(f"Failed to extract response text: {e}")
            return str(response_data)
    
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
        confidence_score = 0.4  # Lower base confidence for HF models
        
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
    
    def change_model(self, new_model_name: str) -> bool:
        """Change the model being used"""
        try:
            old_model = self.model_name
            self.model_name = new_model_name
            self.api_url = f"https://api-inference.huggingface.co/models/{new_model_name}"
            
            # Test the new model
            self._test_connection()
            
            if self.status == HuggingFaceStatus.AVAILABLE:
                logging.info(f"Successfully changed model from {old_model} to {new_model_name}")
                return True
            else:
                # Revert on failure
                self.model_name = old_model
                self.api_url = f"https://api-inference.huggingface.co/models/{old_model}"
                self._test_connection()
                return False
                
        except Exception as e:
            logging.error(f"Failed to change model: {e}")
            return False
    
    def get_service_statistics(self) -> Dict:
        """Get service usage statistics"""
        return {
            "status": self.status.value,
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "rate_limit_delay": self.rate_limit_delay,
            "max_retries": self.max_retries
        }
    
    def reset_statistics(self):
        """Reset service statistics"""
        self.total_requests = 0
        self.total_tokens_used = 0

