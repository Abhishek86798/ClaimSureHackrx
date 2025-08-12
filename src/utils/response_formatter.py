"""
Response formatter utility for Claimsure.

This module provides functionality for formatting answers into structured JSON output
with metadata including confidence scores, sources, and proper truncation.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
import re

logger = logging.getLogger(__name__)

def format_json_output(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format a list of answers into structured JSON output.
    
    Args:
        answers: List of answer dictionaries with keys like:
                - 'answer': str (the actual answer text)
                - 'confidence': float (confidence score 0.0-1.0)
                - 'source_clauses': List[Dict] (source information)
                - 'model': str (LLM model used)
                - 'error': str (error message if any)
                
    Returns:
        Dictionary with formatted JSON structure:
        {
            "answers": ["..."],
            "metadata": [{"confidence": 0.9, "source": "Doc name, page"}]
        }
    """
    try:
        if not answers:
            logger.warning("No answers provided for formatting")
            return {
                "answers": [],
                "metadata": []
            }
        
        formatted_answers = []
        formatted_metadata = []
        
        for i, answer_data in enumerate(answers):
            try:
                # Extract answer text
                answer_text = answer_data.get('answer', '')
                if not answer_text:
                    logger.warning(f"Empty answer text for answer {i}")
                    continue
                
                # Clean and truncate answer text
                cleaned_answer = _clean_and_truncate_answer(answer_text)
                formatted_answers.append(cleaned_answer)
                
                # Extract metadata
                metadata = _extract_metadata(answer_data, i)
                formatted_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"Error formatting answer {i}: {e}")
                # Add fallback entry
                formatted_answers.append("Error processing this answer")
                formatted_metadata.append({
                    "confidence": 0.0,
                    "source": "Error",
                    "model": "unknown",
                    "error": str(e)
                })
        
        result = {
            "answers": formatted_answers,
            "metadata": formatted_metadata
        }
        
        logger.info(f"Successfully formatted {len(formatted_answers)} answers into JSON")
        return result
        
    except Exception as e:
        logger.error(f"Error in format_json_output: {e}")
        return {
            "answers": ["Error formatting responses"],
            "metadata": [{
                "confidence": 0.0,
                "source": "Error",
                "model": "unknown",
                "error": str(e)
            }]
        }

def _clean_and_truncate_answer(answer_text: str, max_length: int = 500) -> str:
    """
    Clean and truncate answer text.
    
    Args:
        answer_text: Raw answer text
        max_length: Maximum length before truncation
        
    Returns:
        Cleaned and truncated answer text
    """
    if not answer_text:
        return ""
    
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', answer_text.strip())
    
    # Remove common noise patterns
    noise_patterns = [
        r'^\s*(Based on|According to|As per|Per)\s+the\s+(provided|given|retrieved)\s+(information|data|clauses?|text)\s*[:\-]?\s*',
        r'^\s*(Here|This)\s+is\s+(what|the)\s+(I found|information|answer)\s*[:\-]?\s*',
        r'^\s*(The|This)\s+(answer|response|information)\s+(is|shows)\s*[:\-]?\s*',
        r'\s*\[End of response\]\s*$',
        r'\s*---\s*$',
        r'\s*\.{3,}\s*$'
    ]
    
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up any remaining artifacts
    cleaned = re.sub(r'\s+', ' ', cleaned.strip())
    
    # Truncate if too long
    if len(cleaned) > max_length:
        # Try to truncate at sentence boundary
        truncated = cleaned[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        
        # Find the last sentence boundary
        last_boundary = max(last_period, last_question, last_exclamation)
        
        if last_boundary > max_length * 0.7:  # Only use boundary if it's not too early
            cleaned = truncated[:last_boundary + 1]
        else:
            cleaned = truncated + "..."
    
    return cleaned

def _extract_metadata(answer_data: Dict[str, Any], answer_index: int) -> Dict[str, Any]:
    """
    Extract metadata from answer data.
    
    Args:
        answer_data: Answer dictionary with metadata
        answer_index: Index of the answer for fallback
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "confidence": 0.0,
        "source": f"Answer {answer_index + 1}",
        "model": "unknown"
    }
    
    # Extract confidence
    confidence = answer_data.get('confidence', 0.0)
    try:
        confidence_float = float(confidence)
        metadata["confidence"] = max(0.0, min(1.0, confidence_float))
    except (ValueError, TypeError):
        metadata["confidence"] = 0.0
    
    # Extract source information
    source_clauses = answer_data.get('source_clauses', [])
    if source_clauses:
        # Get the first source clause
        first_clause = source_clauses[0]
        clause_id = first_clause.get('clause_id', '')
        source = first_clause.get('source', '')
        
        if clause_id and source:
            metadata["source"] = f"{source}, {clause_id}"
        elif clause_id:
            metadata["source"] = clause_id
        elif source:
            metadata["source"] = source
        else:
            metadata["source"] = f"Answer {answer_index + 1}"
    else:
        # Try to get source from other fields
        source = answer_data.get('source', '')
        if source:
            metadata["source"] = source
    
    # Extract model information
    model = answer_data.get('model', '')
    if model and model != 'unknown':
        metadata["model"] = model
    
    # Add additional metadata if available
    if 'reasoning' in answer_data:
        metadata["reasoning"] = answer_data['reasoning'][:100] + "..." if len(answer_data['reasoning']) > 100 else answer_data['reasoning']
    
    if 'error' in answer_data:
        metadata["error"] = answer_data['error']
    
    return metadata

def format_single_answer(answer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a single answer into JSON structure.
    
    Args:
        answer_data: Single answer dictionary
        
    Returns:
        Formatted JSON structure for single answer
    """
    return format_json_output([answer_data])

def validate_json_output(formatted_output: Dict[str, Any]) -> bool:
    """
    Validate the formatted JSON output structure.
    
    Args:
        formatted_output: Formatted output dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required keys
        if 'answers' not in formatted_output or 'metadata' not in formatted_output:
            logger.error("Missing required keys in formatted output")
            return False
        
        answers = formatted_output['answers']
        metadata = formatted_output['metadata']
        
        # Check that answers and metadata have same length
        if len(answers) != len(metadata):
            logger.error(f"Length mismatch: {len(answers)} answers vs {len(metadata)} metadata")
            return False
        
        # Validate each answer and metadata entry
        for i, (answer, meta) in enumerate(zip(answers, metadata)):
            if not isinstance(answer, str):
                logger.error(f"Answer {i} is not a string: {type(answer)}")
                return False
            
            if not isinstance(meta, dict):
                logger.error(f"Metadata {i} is not a dictionary: {type(meta)}")
                return False
            
            # Check required metadata fields
            if 'confidence' not in meta:
                logger.error(f"Metadata {i} missing confidence field")
                return False
            
            if 'source' not in meta:
                logger.error(f"Metadata {i} missing source field")
                return False
        
        logger.info("JSON output validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Error validating JSON output: {e}")
        return False

def to_json_string(formatted_output: Dict[str, Any], indent: int = 2) -> str:
    """
    Convert formatted output to JSON string.
    
    Args:
        formatted_output: Formatted output dictionary
        indent: JSON indentation level
        
    Returns:
        JSON string representation
    """
    try:
        return json.dumps(formatted_output, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error converting to JSON string: {e}")
        return json.dumps({
            "answers": ["Error serializing response"],
            "metadata": [{"confidence": 0.0, "source": "Error", "error": str(e)}]
        }, indent=indent)

# Example usage and testing
if __name__ == "__main__":
    # Test the response formatter
    print("Testing Response Formatter")
    print("=" * 50)
    
    # Sample answer data
    test_answers = [
        {
            "answer": "Based on the provided policy information, health insurance covers doctor visits, hospital stays, and prescription drugs. The policy specifically states that preventive care is included at 100% coverage, while emergency room visits are covered under most health insurance plans with co-pays that may apply for non-emergency visits.",
            "confidence": 0.85,
            "source_clauses": [
                {
                    "clause_id": "chunk_1_health_policy.pdf",
                    "text": "Health insurance covers doctor visits, hospital stays, and prescription drugs.",
                    "source": "health_policy.pdf",
                    "relevance_score": 0.9
                }
            ],
            "model": "gemini",
            "reasoning": "The answer is based on the retrieved health policy clauses that clearly state coverage details."
        },
        {
            "answer": "According to the retrieved information, the coverage limits for prescription drugs are not explicitly mentioned in the provided clauses. However, the policy does indicate that prescription medications are covered under the health insurance plan.",
            "confidence": 0.65,
            "source_clauses": [
                {
                    "clause_id": "chunk_2_prescription_policy.pdf",
                    "text": "Prescription medications are covered under the health insurance plan.",
                    "source": "prescription_policy.pdf",
                    "relevance_score": 0.7
                }
            ],
            "model": "gemini",
            "reasoning": "Limited information available about specific coverage limits."
        }
    ]
    
    # Format the answers
    formatted_result = format_json_output(test_answers)
    
    print("✅ Formatted JSON Output:")
    print(json.dumps(formatted_result, indent=2))
    
    # Validate the output
    is_valid = validate_json_output(formatted_result)
    print(f"\n✅ Validation Result: {'PASSED' if is_valid else 'FAILED'}")
    
    # Test with empty answers
    empty_result = format_json_output([])
    print(f"\n✅ Empty Answers Test: {empty_result}")
    
    # Test with single answer
    single_result = format_single_answer(test_answers[0])
    print(f"\n✅ Single Answer Test: {len(single_result['answers'])} answers")
    
    print("\n🎉 Response Formatter Testing Complete!")
