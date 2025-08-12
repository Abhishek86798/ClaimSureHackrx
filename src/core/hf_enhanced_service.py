"""
Enhanced Hugging Face Service for Claimsure

Integrates multiple Hugging Face models for different purposes:
1. sentence-transformers/all-MiniLM-L6-v2 - Embeddings
2. google/flan-t5-base - Reasoning + Q&A
3. mistralai/Mistral-7B-Instruct-v0.2 - Advanced reasoning
4. tiiuae/falcon-7b-instruct - Backup model
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class HFEnhancedService:
    """Enhanced Hugging Face service with multiple models."""
    
    def __init__(self):
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.embedding_model = None
        self.flan_t5_model = None
        self.mistral_model = None
        self.falcon_model = None
        self.tokenizer = None
        self.is_available_flag = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models."""
        try:
            # Initialize embedding model
            self._init_embedding_model()
            
            # Initialize reasoning models
            self._init_reasoning_models()
            
            self.is_available_flag = True
            logger.info("✅ HF Enhanced Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize HF Enhanced Service: {e}")
            self.is_available_flag = False
    
    def _init_embedding_model(self):
        """Initialize sentence transformer for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("✅ Sentence transformer model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
    
    def _init_reasoning_models(self):
        """Initialize reasoning models."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
            
            # Initialize FLAN-T5 for Q&A
            try:
                self.flan_t5_model = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    device=-1  # CPU
                )
                logger.info("✅ FLAN-T5 model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load FLAN-T5: {e}")
            
            # Initialize Mistral for advanced reasoning
            try:
                self.mistral_model = pipeline(
                    "text-generation",
                    model="mistralai/Mistral-7B-Instruct-v0.2",
                    device=-1  # CPU
                )
                logger.info("✅ Mistral model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Mistral: {e}")
            
            # Initialize Falcon as backup
            try:
                self.falcon_model = pipeline(
                    "text-generation",
                    model="tiiuae/falcon-7b-instruct",
                    trust_remote_code=True,
                    device=-1  # CPU
                )
                logger.info("✅ Falcon model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Falcon: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize reasoning models: {e}")
    
    def is_available(self) -> bool:
        """Check if the service is available."""
        return self.is_available_flag and self.embedding_model is not None
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence transformer."""
        if not self.embedding_model:
            raise ValueError("Embedding model not available")
        
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            raise
    
    def calculate_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix between embeddings."""
        if not self.embedding_model:
            raise ValueError("Embedding model not available")
        
        try:
            similarities = self.embedding_model.similarity(embeddings, embeddings)
            return similarities
        except Exception as e:
            logger.error(f"❌ Error calculating similarity: {e}")
            raise
    
    def process_insurance_query(self, query: str, context: str = "") -> Dict[str, Any]:
        """Process insurance query using the best available model."""
        try:
            # Try models in order of preference
            if self.mistral_model:
                return self._process_with_mistral(query, context)
            elif self.flan_t5_model:
                return self._process_with_flan_t5(query, context)
            elif self.falcon_model:
                return self._process_with_falcon(query, context)
            else:
                raise ValueError("No reasoning models available")
                
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "confidence": 0.0,
                "model": "hf_enhanced_error"
            }
    
    def _process_with_mistral(self, query: str, context: str = "") -> Dict[str, Any]:
        """Process query using Mistral model."""
        try:
            prompt = self._create_insurance_prompt(query, context)
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.mistral_model(messages, max_new_tokens=200, temperature=0.1)
            
            if response and len(response) > 0:
                answer = response[0]['generated_text']
                # Extract the generated part (remove the prompt)
                if len(answer) > len(prompt):
                    answer = answer[len(prompt):].strip()
                
                return {
                    "answer": answer,
                    "confidence": 0.85,
                    "model": "mistral-7b-instruct"
                }
            else:
                raise ValueError("Empty response from Mistral")
                
        except Exception as e:
            logger.error(f"❌ Mistral processing error: {e}")
            raise
    
    def _process_with_flan_t5(self, query: str, context: str = "") -> Dict[str, Any]:
        """Process query using FLAN-T5 model."""
        try:
            prompt = self._create_flan_t5_prompt(query, context)
            
            response = self.flan_t5_model(prompt, max_length=200, temperature=0.1)
            
            if response and len(response) > 0:
                answer = response[0]['generated_text']
                
                return {
                    "answer": answer,
                    "confidence": 0.80,
                    "model": "flan-t5-base"
                }
            else:
                raise ValueError("Empty response from FLAN-T5")
                
        except Exception as e:
            logger.error(f"❌ FLAN-T5 processing error: {e}")
            raise
    
    def _process_with_falcon(self, query: str, context: str = "") -> Dict[str, Any]:
        """Process query using Falcon model."""
        try:
            prompt = self._create_insurance_prompt(query, context)
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.falcon_model(messages, max_new_tokens=200, temperature=0.1)
            
            if response and len(response) > 0:
                answer = response[0]['generated_text']
                # Extract the generated part
                if len(answer) > len(prompt):
                    answer = answer[len(prompt):].strip()
                
                return {
                    "answer": answer,
                    "confidence": 0.75,
                    "model": "falcon-7b-instruct"
                }
            else:
                raise ValueError("Empty response from Falcon")
                
        except Exception as e:
            logger.error(f"❌ Falcon processing error: {e}")
            raise
    
    def _create_insurance_prompt(self, query: str, context: str = "") -> str:
        """Create insurance-specific prompt."""
        if context:
            prompt = f"""You are an insurance document expert. Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""You are an insurance document expert. Answer the following question about insurance policies accurately and concisely.

Question: {query}

Answer:"""
        
        return prompt
    
    def _create_flan_t5_prompt(self, query: str, context: str = "") -> str:
        """Create FLAN-T5 specific prompt."""
        if context:
            return f"Based on this context: {context}. Answer this question: {query}"
        else:
            return f"Answer this insurance question: {query}"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        return {
            "embedding_model": self.embedding_model is not None,
            "flan_t5_model": self.flan_t5_model is not None,
            "mistral_model": self.mistral_model is not None,
            "falcon_model": self.falcon_model is not None,
            "overall_available": self.is_available()
        }
