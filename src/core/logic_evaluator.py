"""
Decision logic evaluator for Claimsure.

This module provides functionality for evaluating retrieved clauses and generating
final answers using hybrid LLM processing (Gemini, Claude, Hugging Face, Local Mistral),
with confidence scores and supporting text highlighting.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Decision evaluation prompt template
DECISION_EVALUATION_PROMPT = """You are an expert insurance claims analyst. Based on the provided clauses and the user's query, provide a comprehensive answer with confidence assessment.

Query: "{query}"

Retrieved Clauses:
{clauses_text}

Instructions:
1. Analyze the retrieved clauses for relevance to the query
2. Provide a clear, accurate answer based on the clauses
3. Highlight the exact supporting text from the clauses
4. Assess your confidence level (0.0 to 1.0) based on:
   - Relevance of clauses to the query
   - Completeness of information in clauses
   - Clarity and specificity of clause content
   - Consistency across multiple clauses

Return your response in the following JSON format:
{{
  "answer": "Your comprehensive answer based on the clauses",
  "confidence": 0.85,
  "source_clauses": [
    {{
      "clause_id": "chunk_1_health_policy.pdf",
      "text": "Exact supporting text from this clause",
      "relevance_score": 0.9,
      "reasoning": "Why this clause is relevant"
    }}
  ],
  "reasoning": "Brief explanation of your decision process"
}}

Important:
- Only use information from the provided clauses
- If clauses don't contain enough information, state this clearly
- Be specific about what is and isn't covered
- Include exact quotes from clauses when relevant
- Confidence should reflect how well the clauses answer the query"""

# Available LLM services in order of preference (Gemini first as requested)
AVAILABLE_SERVICES = [
    "gemini",      # Primary choice as requested
    "claude",      # Secondary choice
    "huggingface", # Alternative
    "local_mistral", # Offline backup
    "gpt35"        # Legacy support
]


class LogicEvaluator:
    """
    Decision logic evaluator that uses hybrid LLM processing to analyze retrieved clauses
    and provide final answers with confidence scores and supporting text.
    """
    
    def __init__(self, 
                 max_tokens: int = 2000,
                 temperature: float = 0.1):
        """
        Initialize the logic evaluator with hybrid LLM services.
        
        Args:
            max_tokens: Maximum tokens for response
            temperature: Response randomness (0.0 = deterministic)
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize hybrid LLM services
        self.available_services = []
        
        # Try to initialize Gemini service (primary choice)
        try:
            from .gemini_service import GeminiService
            self.gemini_service = GeminiService()
            if self.gemini_service.is_available():
                self.available_services.append("gemini")
                logger.info("✅ Gemini service available for decision evaluation")
        except Exception as e:
            logger.warning(f"Gemini service not available: {e}")
            self.gemini_service = None
        
        # Try to initialize Claude service
        try:
            from .claude_service import ClaudeService
            self.claude_service = ClaudeService()
            if self.claude_service.is_available():
                self.available_services.append("claude")
                logger.info("✅ Claude service available for decision evaluation")
        except Exception as e:
            logger.warning(f"Claude service not available: {e}")
            self.claude_service = None
        
        # Try to initialize Hugging Face service
        try:
            from .huggingface_service import HuggingFaceService
            self.huggingface_service = HuggingFaceService()
            if self.huggingface_service.is_available():
                self.available_services.append("huggingface")
                logger.info("✅ Hugging Face service available for decision evaluation")
        except Exception as e:
            logger.warning(f"Hugging Face service not available: {e}")
            self.huggingface_service = None
        
        # Try to initialize Local Mistral service
        try:
            from .local_mistral_service import LocalMistralService
            self.local_mistral_service = LocalMistralService()
            if self.local_mistral_service.is_available():
                self.available_services.append("local_mistral")
                logger.info("✅ Local Mistral service available for decision evaluation")
        except Exception as e:
            logger.warning(f"Local Mistral service not available: {e}")
            self.local_mistral_service = None
        
        # Try to initialize GPT-3.5 service as fallback
        try:
            from .gpt35_service import GPT35Service
            self.gpt35_service = GPT35Service()
            if self.gpt35_service.is_available():
                self.available_services.append("gpt35")
                logger.info("✅ GPT-3.5 service available for decision evaluation")
        except Exception as e:
            logger.warning(f"GPT-3.5 service not available: {e}")
            self.gpt35_service = None
        
        logger.info(f"Initialized LogicEvaluator with {len(self.available_services)} LLM services: {self.available_services}")
        
        if not self.available_services:
            logger.warning("No LLM services available - will use fallback evaluation only")
    
    def _get_best_available_service(self) -> Optional[str]:
        """
        Get the best available LLM service for evaluation.
        
        Returns:
            Best available service name or None
        """
        if not self.available_services:
            return None
        
        # Return the first available service (highest preference)
        return self.available_services[0]
    
    def evaluate_decision(self, 
                         query: str, 
                         retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate decision logic based on retrieved clauses and query.
        
        Args:
            query: User query string
            retrieved_chunks: List of retrieved clause dictionaries
            
        Returns:
            Dictionary with answer, confidence, and source clauses
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for decision evaluation")
            return {
                "answer": "No query provided for evaluation.",
                "confidence": 0.0,
                "source_clauses": [],
                "error": "Empty query"
            }
        
        if not retrieved_chunks:
            logger.warning("No retrieved chunks provided for decision evaluation")
            return {
                "answer": "No relevant information found to answer your query.",
                "confidence": 0.0,
                "source_clauses": [],
                "error": "No retrieved chunks"
            }
        
        try:
            logger.info(f"Evaluating decision for query: {query[:50]}... with {len(retrieved_chunks)} chunks")
            
            # Try LLM evaluation first if services are available
            if self.available_services:
                try:
                    # Get the best available service
                    best_service = self._get_best_available_service()
                    if not best_service:
                        logger.warning("No available services found - using fallback evaluation")
                        return self._fallback_evaluation(query, retrieved_chunks)
                    
                    # Prepare clauses text
                    clauses_text = self._prepare_clauses_text(retrieved_chunks)
                    
                    # Create evaluation prompt
                    prompt = DECISION_EVALUATION_PROMPT.format(
                        query=query,
                        clauses_text=clauses_text
                    )
                    
                    # Try services in priority order
                    for service_name in self.available_services:
                        try:
                            response_text = self._get_llm_evaluation(service_name, prompt)
                            if response_text:
                                evaluation = self._parse_evaluation_response(response_text, query, retrieved_chunks)
                                evaluation["model"] = service_name  # Add the actual service used
                                
                                logger.info(f"Successfully evaluated decision with confidence: {evaluation.get('confidence', 0.0)} using service: {service_name}")
                                return evaluation
                        except Exception as e:
                            logger.warning(f"{service_name} evaluation failed for query '{query}': {e}")
                            continue
                    
                    # If all LLM services failed, use fallback
                    logger.warning("All LLM services failed, using fallback evaluation")
                    return self._fallback_evaluation(query, retrieved_chunks)
                    
                except Exception as e:
                    logger.warning(f"LLM evaluation failed for query '{query}': {e}")
                    return self._fallback_evaluation(query, retrieved_chunks)
            else:
                # Use fallback evaluation if no LLM services available
                logger.info(f"Using fallback evaluation for query: {query[:50]}...")
                return self._fallback_evaluation(query, retrieved_chunks)
                
        except Exception as e:
            logger.error(f"Error evaluating decision for query '{query}': {e}")
            return {
                "answer": "I encountered an error while evaluating your query.",
                "confidence": 0.0,
                "source_clauses": [],
                "error": str(e)
            }
    
    def _prepare_clauses_text(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare formatted text from retrieved chunks.
        
        Args:
            retrieved_chunks: List of retrieved clause dictionaries
            
        Returns:
            Formatted clauses text
        """
        clauses_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", chunk.get("content", ""))
            source = chunk.get("source", "unknown")
            similarity_score = chunk.get("similarity_score", 0.0)
            clause_id = chunk.get("clause_id", chunk.get("id", f"chunk_{i}"))
            
            # Format clause information
            clause_info = f"Clause {i} (ID: {clause_id}, Source: {source}, Similarity: {similarity_score:.3f}):\n{text}\n"
            clauses_parts.append(clause_info)
        
        return "\n".join(clauses_parts)
    
    def _parse_evaluation_response(self, 
                                  response_text: str, 
                                  original_query: str,
                                  retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse the GPT evaluation response into structured data.
        
        Args:
            response_text: Raw response from GPT
            original_query: Original user query
            retrieved_chunks: Original retrieved chunks
            
        Returns:
            Parsed evaluation dictionary
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate and extract fields
                answer = parsed.get("answer", "No answer provided.")
                confidence = self._validate_confidence(parsed.get("confidence", 0.0))
                source_clauses = parsed.get("source_clauses", [])
                reasoning = parsed.get("reasoning", "")
                
                # Validate source clauses
                validated_source_clauses = self._validate_source_clauses(source_clauses, retrieved_chunks)
                
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "source_clauses": validated_source_clauses,
                    "reasoning": reasoning
                }
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse GPT evaluation response: {e}")
            return self._fallback_evaluation(original_query, retrieved_chunks)
    
    def _validate_confidence(self, confidence: Any) -> float:
        """
        Validate and normalize confidence score.
        
        Args:
            confidence: Raw confidence value
            
        Returns:
            Normalized confidence score (0.0 to 1.0)
        """
        try:
            confidence_float = float(confidence)
            # Ensure confidence is between 0.0 and 1.0
            return max(0.0, min(1.0, confidence_float))
        except (ValueError, TypeError):
            return 0.0
    
    def _validate_source_clauses(self, 
                                source_clauses: List[Dict[str, Any]], 
                                retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and normalize source clauses.
        
        Args:
            source_clauses: Raw source clauses from GPT
            retrieved_chunks: Original retrieved chunks
            
        Returns:
            Validated source clauses
        """
        validated_clauses = []
        
        for clause in source_clauses:
            try:
                validated_clause = {
                    "clause_id": clause.get("clause_id", "unknown"),
                    "text": clause.get("text", ""),
                    "relevance_score": self._validate_confidence(clause.get("relevance_score", 0.0)),
                    "reasoning": clause.get("reasoning", "")
                }
                validated_clauses.append(validated_clause)
            except Exception as e:
                logger.warning(f"Failed to validate source clause: {e}")
                continue
        
        return validated_clauses
    
    def _get_llm_evaluation(self, service_name: str, prompt: str) -> str:
        """
        Get evaluation from a specific LLM service.
        
        Args:
            service_name: Name of the LLM service to use
            prompt: Evaluation prompt
            
        Returns:
            Response text from the LLM service
        """
        try:
            if service_name == "gemini" and self.gemini_service:
                response, confidence, status = self.gemini_service.process_insurance_query(prompt)
                return response
            elif service_name == "claude" and self.claude_service:
                response, confidence, status = self.claude_service.process_insurance_query(prompt)
                return response
            elif service_name == "huggingface" and self.huggingface_service:
                response, confidence, status = self.huggingface_service.process_insurance_query(prompt)
                return response
            elif service_name == "local_mistral" and self.local_mistral_service:
                response, confidence, status = self.local_mistral_service.process_insurance_query(prompt)
                return response
            elif service_name == "gpt35" and self.gpt35_service:
                response, confidence, status = self.gpt35_service.process_insurance_query(prompt)
                return response
            else:
                return ""
        except Exception as e:
            logger.error(f"Error getting evaluation from {service_name}: {e}")
            return ""
    
    def _fallback_evaluation(self, 
                           query: str, 
                           retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback evaluation when GPT models are not available.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved clauses
            
        Returns:
            Fallback evaluation result
        """
        try:
            # Simple rule-based evaluation
            if not retrieved_chunks:
                return {
                    "answer": "No relevant information found to answer your query.",
                    "confidence": 0.0,
                    "source_clauses": [],
                    "reasoning": "No chunks retrieved"
                }
            
            # Calculate average similarity score as confidence
            similarity_scores = [chunk.get("similarity_score", 0.0) for chunk in retrieved_chunks]
            avg_confidence = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
            # Create simple answer from top chunk
            top_chunk = max(retrieved_chunks, key=lambda x: x.get("similarity_score", 0.0))
            answer = f"Based on the retrieved information: {top_chunk.get('text', '')[:200]}..."
            
            # Create source clauses
            source_clauses = []
            for chunk in retrieved_chunks[:3]:  # Top 3 chunks
                source_clauses.append({
                    "clause_id": chunk.get("clause_id", chunk.get("id", "unknown")),
                    "text": chunk.get("text", chunk.get("content", ""))[:100] + "...",
                    "relevance_score": chunk.get("similarity_score", 0.0),
                    "reasoning": "Retrieved based on semantic similarity"
                })
            
            return {
                "answer": answer,
                "confidence": avg_confidence,
                "source_clauses": source_clauses,
                "reasoning": "Fallback evaluation using similarity scores",
                "model": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback evaluation failed: {e}")
            return {
                "answer": "Unable to evaluate your query at this time.",
                "confidence": 0.0,
                "source_clauses": [],
                "reasoning": f"Fallback evaluation error: {str(e)}",
                "model": "fallback"
            }
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the logic evaluator.
        
        Returns:
            Dictionary with evaluator statistics
        """
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "available_llm_services": self.available_services,
            "total_services": len(self.available_services)
        }


def evaluate_decision(query: str, 
                     retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to evaluate decision logic.
    
    Args:
        query: User query string
        retrieved_chunks: List of retrieved clause dictionaries
        
    Returns:
        Dictionary with answer, confidence, and source clauses
    """
    evaluator = LogicEvaluator()
    return evaluator.evaluate_decision(query, retrieved_chunks)


# Example usage and testing
if __name__ == "__main__":
    # Test the logic evaluator
    print("Testing Logic Evaluator")
    print("=" * 50)
    
    # Initialize evaluator
    try:
        evaluator = LogicEvaluator()
        print("✅ LogicEvaluator initialized successfully")
        
        # Show available services
        if evaluator.available_services:
            print(f"✅ Available LLM services: {evaluator.available_services}")
        else:
            print("⚠️ No LLM services available - will use fallback evaluation")
        
        # Test with sample data
        test_query = "What is covered under health insurance?"
        test_chunks = [
            {
                "clause_id": "chunk_1_health_policy.pdf",
                "text": "Health insurance covers doctor visits, hospital stays, and prescription drugs.",
                "source": "health_policy.pdf",
                "similarity_score": 0.85
            },
            {
                "clause_id": "chunk_2_emergency_policy.pdf", 
                "text": "Emergency room visits are covered under most health insurance plans.",
                "source": "emergency_policy.pdf",
                "similarity_score": 0.72
            }
        ]
        
        print(f"\n📝 Testing evaluation with query: {test_query}")
        print(f"   Retrieved chunks: {len(test_chunks)}")
        
        # Evaluate decision
        result = evaluator.evaluate_decision(test_query, test_chunks)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Source clauses: {len(result['source_clauses'])}")
        print(f"   Model: {result.get('model', 'unknown')}")
        
        if result['source_clauses']:
            print(f"\n🔍 Source Clauses:")
            for i, clause in enumerate(result['source_clauses'], 1):
                print(f"   {i}. {clause['clause_id']} (relevance: {clause['relevance_score']:.3f})")
                print(f"      Text: {clause['text'][:60]}...")
        
    except Exception as e:
        print(f"❌ Failed to test logic evaluator: {e}")
