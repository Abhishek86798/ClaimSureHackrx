"""
LLM service for generating responses using OpenAI GPT models.
"""

import logging
from typing import List, Dict, Any, Optional
import openai
from datetime import datetime

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE
)

logger = logging.getLogger(__name__)

class LLMService:
    """Handles LLM interactions for generating responses."""
    
    def __init__(self):
        self.client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            if not OPENAI_API_KEY:
                logger.warning("OpenAI API key not configured. LLM responses will be disabled.")
                return
            
            openai.api_key = OPENAI_API_KEY
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise
    
    def generate_response(
        self, 
        query: str, 
        context_chunks: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a response based on query and context chunks.
        
        Args:
            query: User's query
            context_chunks: List of relevant document chunks
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            
        Returns:
            Dict[str, Any]: Generated response with metadata
        """
        try:
            if not self.client:
                return {
                    "response": "LLM service not available. Please configure OpenAI API key.",
                    "model": "none",
                    "tokens_used": 0,
                    "error": "OpenAI not configured"
                }
            
            # Use default values if not provided
            max_tokens = max_tokens or OPENAI_MAX_TOKENS
            temperature = temperature or OPENAI_TEMPERATURE
            
            # Prepare context from chunks
            context = self._prepare_context(context_chunks)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt
            user_prompt = self._create_user_prompt(query, context)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response
            generated_response = response.choices[0].message.content
            
            # Prepare result
            result = {
                "response": generated_response,
                "model": OPENAI_MODEL,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "context_chunks_used": len(context_chunks),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated response using {result['tokens_used']} tokens")
            return result
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "response": f"Error generating response: {str(e)}",
                "model": OPENAI_MODEL,
                "tokens_used": 0,
                "error": str(e)
            }
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            str: Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            score = chunk.get("similarity_score", 0)
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            
            context_part = f"Document {i} ({filename}, similarity: {score:.3f}):\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """
        Create system prompt for the LLM.
        
        Returns:
            str: System prompt
        """
        return """You are an AI assistant that helps users find information from their documents. 
Your task is to provide accurate, helpful responses based on the context provided from the user's documents.

Guidelines:
1. Only use information from the provided context to answer questions
2. If the context doesn't contain enough information to answer the question, say so
3. Be concise but thorough in your responses
4. Cite the source documents when possible
5. If you're unsure about something, acknowledge the uncertainty
6. Format your responses clearly and professionally

Remember: You can only answer based on the information provided in the context."""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """
        Create user prompt with query and context.
        
        Args:
            query: User's question
            context: Document context
            
        Returns:
            str: User prompt
        """
        return f"""Based on the following context from the user's documents, please answer their question.

Context:
{context}

Question: {query}

Please provide a helpful and accurate response based on the context above."""
    
    def generate_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dict[str, Any]: Summary with metadata
        """
        try:
            if not self.client:
                return {
                    "summary": "LLM service not available for summarization.",
                    "model": "none",
                    "tokens_used": 0,
                    "error": "OpenAI not configured"
                }
            
            # Prepare context
            context = self._prepare_context(chunks)
            
            # Create summary prompt
            summary_prompt = f"""Please provide a concise summary of the following document content:

{context}

Summary:"""
            
            # Generate summary
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries of document content."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=OPENAI_MAX_TOKENS // 2,  # Shorter for summaries
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            summary = response.choices[0].message.content
            
            result = {
                "summary": summary,
                "model": OPENAI_MODEL,
                "tokens_used": response.usage.total_tokens,
                "chunks_summarized": len(chunks),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated summary using {result['tokens_used']} tokens")
            return result
        
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "model": OPENAI_MODEL,
                "tokens_used": 0,
                "error": str(e)
            }
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using LLM.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List[str]: List of keywords
        """
        try:
            if not self.client:
                return []
            
            prompt = f"""Extract the most important keywords from the following text. 
Return only the keywords separated by commas, without explanations:

{text}

Keywords:"""
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a keyword extraction tool. Return only keywords separated by commas."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
            
            logger.info(f"Extracted {len(keywords)} keywords")
            return keywords
        
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if LLM service is available.
        
        Returns:
            bool: True if service is available
        """
        return self.client is not None and OPENAI_API_KEY is not None
