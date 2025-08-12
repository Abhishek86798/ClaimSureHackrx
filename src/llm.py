"""
LLM interface module for Claimsure.

Handles communication with OpenAI and other LLM providers for text generation.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import openai
from config import (
    OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE,
    OPENAI_API_KEY
)

logger = logging.getLogger(__name__)


class LLMInterface:
    """Handles communication with LLM providers."""
    
    def __init__(self, model: str = OPENAI_MODEL, max_tokens: int = OPENAI_MAX_TOKENS,
                 temperature: float = OPENAI_TEMPERATURE):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Configure OpenAI client
        openai.api_key = self.api_key
        logger.info(f"Initialized LLM interface with model: {model}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            Generated response string
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate and relevant information based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Generated response with {len(response_text)} characters")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    def generate_response_with_context(self, context: str, query: str) -> str:
        """
        Generate a response with explicit context and query.
        
        Args:
            context: Context information
            query: User query
            
        Returns:
            Generated response string
        """
        prompt = f"""Context: {context}

Question: {query}

Please provide a helpful and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""
        
        return self.generate_response(prompt)
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Generated summary string
        """
        prompt = f"""Please provide a concise summary of the following text in {max_length} words or less:

{text}

Summary:"""
        
        return self.generate_response(prompt)
    
    def generate_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        Generate keywords from the given text.
        
        Args:
            text: Text to extract keywords from
            num_keywords: Number of keywords to generate
            
        Returns:
            List of keywords
        """
        prompt = f"""Extract {num_keywords} key terms or phrases from the following text. Return only the keywords, one per line:

{text}"""
        
        response = self.generate_response(prompt)
        
        # Parse keywords from response
        keywords = [line.strip() for line in response.split('\n') if line.strip()]
        return keywords[:num_keywords]
    
    def classify_text(self, text: str, categories: List[str]) -> str:
        """
        Classify text into one of the given categories.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            
        Returns:
            Selected category
        """
        categories_str = ", ".join(categories)
        prompt = f"""Classify the following text into one of these categories: {categories_str}

Text: {text}

Category:"""
        
        response = self.generate_response(prompt)
        
        # Find the best matching category
        response_lower = response.lower().strip()
        for category in categories:
            if category.lower() in response_lower:
                return category
        
        # Return the first category if no match found
        return categories[0] if categories else "unknown"
    
    def batch_generate_responses(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for i, prompt in enumerate(prompts):
            try:
                response = self.generate_response(prompt)
                responses.append(response)
                logger.debug(f"Generated response {i+1}/{len(prompts)}")
            except Exception as e:
                logger.error(f"Error generating response for prompt {i+1}: {str(e)}")
                responses.append("")
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key_configured": bool(self.api_key)
        }
    
    def test_connection(self) -> bool:
        """
        Test the connection to the LLM provider.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Simple test prompt
            response = self.generate_response("Hello, this is a test message.")
            return bool(response and len(response) > 0)
        except Exception as e:
            logger.error(f"LLM connection test failed: {str(e)}")
            return False
