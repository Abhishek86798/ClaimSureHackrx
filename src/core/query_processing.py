"""
Query processing and classification system for Claimsure.

Uses hybrid LLM processing (Claude, Gemini, Hugging Face, Local Mistral) to classify 
insurance-related queries into intents and entities for improved search and response generation.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Query classification intents
QUERY_INTENTS = {
    "coverage": "Questions about what is covered under the policy",
    "limit": "Questions about coverage limits, deductibles, or maximum payouts",
    "definition": "Questions about insurance terms, definitions, or concepts",
    "general_info": "General questions about policies, procedures, or information"
}

# Query entities for extraction
QUERY_ENTITIES = {
    "conditions": "Medical conditions, pre-existing conditions, or health status",
    "amounts": "Dollar amounts, percentages, limits, deductibles, or financial figures",
    "benefits": "Specific benefits, services, or coverage details"
}

# Optimized prompt for low token usage
QUERY_CLASSIFICATION_PROMPT = """Classify this insurance query into intent and entities. Return JSON only.

Intent options: coverage, limit, definition, general_info
Entity options: conditions, amounts, benefits

Query: "{query}"

Return format:
{{
  "intent": "intent_type",
  "entities": ["entity1", "entity2"],
  "raw_query": "original_query"
}}"""


class QueryProcessor:
    """Process and classify insurance-related queries using hybrid LLM processing."""
    
    def __init__(self, 
                 max_tokens: int = 150,
                 temperature: float = 0.1):
        """
        Initialize the query processor with hybrid LLM services.
        
        Args:
            max_tokens: Maximum tokens for response
            temperature: Response randomness (0.0 = deterministic)
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize hybrid LLM services
        self.available_services = []
        
        # Try to initialize Claude service
        try:
            from .claude_service import ClaudeService
            self.claude_service = ClaudeService()
            if self.claude_service.is_available():
                self.available_services.append("claude")
                logger.info("✅ Claude service available for query classification")
        except Exception as e:
            logger.warning(f"Claude service not available: {e}")
            self.claude_service = None
        
        # Try to initialize Gemini service
        try:
            from .gemini_service import GeminiService
            self.gemini_service = GeminiService()
            if self.gemini_service.is_available():
                self.available_services.append("gemini")
                logger.info("✅ Gemini service available for query classification")
        except Exception as e:
            logger.warning(f"Gemini service not available: {e}")
            self.gemini_service = None
        
        # Try to initialize Hugging Face service
        try:
            from .huggingface_service import HuggingFaceService
            self.huggingface_service = HuggingFaceService()
            if self.huggingface_service.is_available():
                self.available_services.append("huggingface")
                logger.info("✅ Hugging Face service available for query classification")
        except Exception as e:
            logger.warning(f"Hugging Face service not available: {e}")
            self.huggingface_service = None
        
        # Try to initialize Local Mistral service
        try:
            from .local_mistral_service import LocalMistralService
            self.local_mistral_service = LocalMistralService()
            if self.local_mistral_service.is_available():
                self.available_services.append("local_mistral")
                logger.info("✅ Local Mistral service available for query classification")
        except Exception as e:
            logger.warning(f"Local Mistral service not available: {e}")
            self.local_mistral_service = None
        
        # Try to initialize GPT-3.5 service as fallback
        try:
            from .gpt35_service import GPT35Service
            self.gpt35_service = GPT35Service()
            if self.gpt35_service.is_available():
                self.available_services.append("gpt35")
                logger.info("✅ GPT-3.5 service available for query classification")
        except Exception as e:
            logger.warning(f"GPT-3.5 service not available: {e}")
            self.gpt35_service = None
        
        logger.info(f"Initialized QueryProcessor with {len(self.available_services)} LLM services: {self.available_services}")
        
        if not self.available_services:
            logger.warning("No LLM services available - will use fallback classification only")
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse and classify an insurance query using hybrid LLM processing.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with intent, entities, and raw_query
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return {
                "intent": "general_info",
                "entities": [],
                "raw_query": query
            }
        
        # Try LLM classification first if services are available
        if self.available_services:
            try:
                # Clean and prepare query
                cleaned_query = self._clean_query(query)
                
                # Create prompt
                prompt = QUERY_CLASSIFICATION_PROMPT.format(query=cleaned_query)
                
                # Try services in priority order
                for service_name in self.available_services:
                    try:
                        response_text = self._get_llm_classification(service_name, prompt)
                        if response_text:
                            classification = self._parse_response(response_text, query)
                            logger.info(f"Classified query using {service_name}: {query[:50]}... -> Intent: {classification['intent']}, Entities: {classification['entities']}")
                            return classification
                    except Exception as e:
                        logger.warning(f"{service_name} classification failed for query '{query}': {e}")
                        continue
                
                # If all LLM services failed, use fallback
                logger.warning("All LLM services failed, using fallback classification")
                return self._fallback_classification(query)
                
            except Exception as e:
                logger.warning(f"LLM classification failed for query '{query}': {e}")
                return self._fallback_classification(query)
        else:
            # Use fallback classification if no LLM services available
            logger.info(f"Using fallback classification for query: {query[:50]}...")
            return self._fallback_classification(query)
    
    def _get_llm_classification(self, service_name: str, prompt: str) -> str:
        """
        Get classification from a specific LLM service.
        
        Args:
            service_name: Name of the LLM service to use
            prompt: Classification prompt
            
        Returns:
            Response text from the LLM service
        """
        try:
            if service_name == "claude" and self.claude_service:
                response, confidence, status = self.claude_service.process_insurance_query(prompt)
                return response
            elif service_name == "gemini" and self.gemini_service:
                response, confidence, status = self.gemini_service.process_insurance_query(prompt)
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
            logger.error(f"Error getting classification from {service_name}: {e}")
            return ""
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize the query text.
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters that might confuse the model
        cleaned = re.sub(r'[^\w\s\-.,?!]', '', cleaned)
        
        return cleaned
    
    def _parse_response(self, response_text: str, original_query: str) -> Dict[str, Any]:
        """
        Parse the OpenAI response into structured data.
        
        Args:
            response_text: Raw response from OpenAI
            original_query: Original user query
            
        Returns:
            Parsed classification dictionary
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate intent
                intent = parsed.get("intent", "general_info")
                if intent not in QUERY_INTENTS:
                    intent = "general_info"
                
                # Validate entities
                entities = parsed.get("entities", [])
                if not isinstance(entities, list):
                    entities = []
                
                # Filter valid entities
                valid_entities = [e for e in entities if e in QUERY_ENTITIES]
                
                return {
                    "intent": intent,
                    "entities": valid_entities,
                    "raw_query": original_query
                }
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse OpenAI response: {e}")
            return self._fallback_classification(original_query)
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """
        Provide fallback classification when OpenAI is unavailable.
        
        Args:
            query: Original query string
            
        Returns:
            Fallback classification dictionary
        """
        query_lower = query.lower()
        
        # Simple keyword-based intent classification
        intent = "general_info"
        entities = []
        
        # Intent classification
        if any(word in query_lower for word in ["cover", "coverage", "covered", "what's included", "what is included"]):
            intent = "coverage"
        elif any(word in query_lower for word in ["limit", "maximum", "deductible", "cap", "up to", "payout"]):
            intent = "limit"
        elif any(word in query_lower for word in ["what is", "define", "definition", "meaning", "term", "explain"]):
            intent = "definition"
        
        # Entity extraction
        if any(word in query_lower for word in ["condition", "diagnosis", "illness", "disease", "health", "mental"]):
            entities.append("conditions")
        
        if any(word in query_lower for word in ["$", "dollar", "amount", "percent", "cost", "price", "fee", "deductible"]):
            entities.append("amounts")
        
        if any(word in query_lower for word in ["benefit", "service", "treatment", "therapy", "medication", "procedure"]):
            entities.append("benefits")
        
        return {
            "intent": intent,
            "entities": entities,
            "raw_query": query
        }
    
    def get_intent_description(self, intent: str) -> str:
        """
        Get description for a query intent.
        
        Args:
            intent: Intent type
            
        Returns:
            Description of the intent
        """
        return QUERY_INTENTS.get(intent, "Unknown intent")
    
    def get_entity_description(self, entity: str) -> str:
        """
        Get description for a query entity.
        
        Args:
            entity: Entity type
            
        Returns:
            Description of the entity
        """
        return QUERY_ENTITIES.get(entity, "Unknown entity")
    
    def classify_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of classification dictionaries
        """
        results = []
        for query in queries:
            try:
                classification = self.parse_query(query)
                results.append(classification)
            except Exception as e:
                logger.error(f"Failed to classify query '{query}': {e}")
                results.append({
                    "intent": "general_info",
                    "entities": [],
                    "raw_query": query,
                    "error": str(e)
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the query processor."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "available_intents": list(QUERY_INTENTS.keys()),
            "available_entities": list(QUERY_ENTITIES.keys()),
            "available_llm_services": self.available_services,
            "total_services": len(self.available_services)
        }


# Convenience function for direct query parsing
def parse_query(query: str) -> Dict[str, Any]:
    """
    Parse and classify an insurance query.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with intent, entities, and raw_query
    """
    processor = QueryProcessor()
    return processor.parse_query(query)


# Example usage and testing
if __name__ == "__main__":
    # Test the query processor
    processor = QueryProcessor()
    
    test_queries = [
        "What is covered under my health insurance?",
        "What is the maximum payout for dental procedures?",
        "Define pre-existing condition",
        "How much does a doctor visit cost?",
        "What benefits are included for mental health?"
    ]
    
    print("Testing Query Processor:")
    print("=" * 50)
    
    for query in test_queries:
        result = processor.parse_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']} ({processor.get_intent_description(result['intent'])})")
        print(f"Entities: {result['entities']}")
        for entity in result['entities']:
            print(f"  - {entity}: {processor.get_entity_description(entity)}")
    
    print(f"\nProcessor Statistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
