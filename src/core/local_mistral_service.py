import os
import time
import logging
import subprocess
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LocalMistralStatus(Enum):
    """Status of local Mistral service"""
    AVAILABLE = "available"
    MODEL_LOADING = "model_loading"
    MODEL_NOT_FOUND = "model_not_found"
    PROCESSING_ERROR = "processing_error"
    UNAVAILABLE = "unavailable"

class LocalMistralService:
    """
    Manages local Mistral-7B-Instruct model using llama.cpp.
    Provides offline backup when external APIs are unavailable.
    """
    
    def __init__(self, model_path: str = None, llama_cpp_path: str = None):
        """Initialize local Mistral service"""
        self.model_path = model_path or os.getenv('MISTRAL_MODEL_PATH')
        self.llama_cpp_path = llama_cpp_path or os.getenv('LLAMA_CPP_PATH', 'llama.cpp')
        self.status = LocalMistralStatus.UNAVAILABLE
        self.model_loaded = False
        self.total_requests = 0
        self.total_tokens_used = 0
        self.last_request_time = 0
        self.rate_limit_delay = 0.5  # Local processing is faster
        
        # Check if llama.cpp is available
        if self._check_llama_cpp():
            if self.model_path and Path(self.model_path).exists():
                self._load_model()
            else:
                logging.warning("Mistral model path not found. Local service unavailable.")
        else:
            logging.warning("llama.cpp not found. Local service unavailable.")
    
    def _check_llama_cpp(self) -> bool:
        """Check if llama.cpp is available in the system"""
        try:
            result = subprocess.run([self.llama_cpp_path, '--help'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _load_model(self):
        """Load the Mistral model into memory"""
        if not self.model_path or not Path(self.model_path).exists():
            self.status = LocalMistralStatus.MODEL_NOT_FOUND
            return
        
        try:
            self.status = LocalMistralStatus.MODEL_LOADING
            logging.info("Loading Mistral model...")
            
            # Test model loading with a simple inference
            test_prompt = "Hello"
            result = self._run_inference(test_prompt)
            
            if result and result.strip():
                self.model_loaded = True
                self.status = LocalMistralStatus.AVAILABLE
                logging.info("Mistral model loaded successfully")
            else:
                self.status = LocalMistralStatus.PROCESSING_ERROR
                logging.error("Failed to get response from loaded model")
                
        except Exception as e:
            self.status = LocalMistralStatus.PROCESSING_ERROR
            logging.error(f"Failed to load Mistral model: {e}")
    
    def get_status(self) -> LocalMistralStatus:
        """Get current service status"""
        return self.status
    
    def is_available(self) -> bool:
        """Check if service is available for use"""
        return self.status == LocalMistralStatus.AVAILABLE and self.model_loaded
    
    def process_insurance_query(self, 
                              query: str, 
                              context: Optional[str] = None,
                              claim_details: Optional[Dict] = None) -> Tuple[str, float, LocalMistralStatus]:
        """
        Process an insurance-related query using local Mistral model
        
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
            
            # Run local inference
            response_text = self._run_inference(prompt)
            
            if response_text:
                # Update statistics
                self.total_requests += 1
                self.total_tokens_used += len(response_text.split())  # Approximate token count
                
                # Calculate confidence
                confidence = self._calculate_confidence(response_text, query)
                
                return response_text, confidence, self.status
            else:
                self.status = LocalMistralStatus.PROCESSING_ERROR
                return "", 0.0, self.status
                
        except Exception as e:
            self.status = LocalMistralStatus.PROCESSING_ERROR
            logging.error(f"Error in local Mistral processing: {e}")
            return "", 0.0, self.status
    
    def batch_process_queries(self, 
                            queries: List[str], 
                            contexts: Optional[List[str]] = None) -> List[Tuple[str, float, LocalMistralStatus]]:
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
            
            # Add small delay between requests
            if i < len(queries) - 1:
                time.sleep(self.rate_limit_delay)
        
        return results
    
    def _run_inference(self, prompt: str) -> str:
        """Run inference using llama.cpp"""
        try:
            # Prepare the command for llama.cpp
            cmd = [
                self.llama_cpp_path,
                '-m', self.model_path,
                '-n', '512',  # Max tokens to generate
                '-p', prompt,
                '--temp', '0.1',  # Low temperature for consistent responses
                '--repeat_penalty', '1.1'  # Prevent repetition
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Extract the generated text (remove the input prompt)
                output = result.stdout.strip()
                if output.startswith(prompt):
                    generated_text = output[len(prompt):].strip()
                else:
                    generated_text = output
                
                return generated_text
            else:
                logging.error(f"llama.cpp error: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            logging.error("llama.cpp inference timed out")
            return ""
        except Exception as e:
            logging.error(f"Error running llama.cpp inference: {e}")
            return ""
    
    def _prepare_insurance_prompt(self, 
                                 query: str, 
                                 context: Optional[str] = None,
                                 claim_details: Optional[Dict] = None) -> str:
        """Prepare a comprehensive prompt for insurance queries"""
        
        prompt = f"""<s>[INST] You are an expert insurance claims analyst. Analyze the following query and provide a clear, accurate response based on the provided context.

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

RESPONSE: [/INST]"""
        
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
        confidence_score = 0.3  # Lower base confidence for local models
        
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
    
    def reload_model(self) -> bool:
        """Reload the model (useful if there are issues)"""
        try:
            self.model_loaded = False
            self._load_model()
            return self.model_loaded
        except Exception as e:
            logging.error(f"Failed to reload model: {e}")
            return False
    
    def get_service_statistics(self) -> Dict:
        """Get service usage statistics"""
        return {
            "status": self.status.value,
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "llama_cpp_path": self.llama_cpp_path,
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "rate_limit_delay": self.rate_limit_delay
        }
    
    def reset_statistics(self):
        """Reset service statistics"""
        self.total_requests = 0
        self.total_tokens_used = 0

