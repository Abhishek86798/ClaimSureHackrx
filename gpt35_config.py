"""
GPT-3.5 Configuration for Claimsure.

This file contains all the configuration settings for the GPT-3.5 service,
including model parameters, rate limiting, and fallback behavior.
"""

import os
from typing import Dict, Any

# GPT-3.5 Model Configuration
GPT35_MODEL = "gpt-3.5-turbo"  # Best balance of performance and cost for free tier
GPT35_MAX_TOKENS = 1000  # Conservative limit for free tier usage
GPT35_TEMPERATURE = 0.1  # Low temperature for consistent, factual responses
GPT35_TOP_P = 0.9  # Nucleus sampling for focused responses

# Rate Limiting and Retry Configuration
GPT35_MAX_RETRIES = 3  # Maximum retry attempts for failed requests
GPT35_RATE_LIMIT_DELAY = 1.0  # Base delay between retries in seconds
GPT35_BATCH_DELAY = 0.5  # Delay between batch requests to respect rate limits
GPT35_EXPONENTIAL_BACKOFF = True  # Use exponential backoff for retries

# Free Tier Management
GPT35_FREE_TIER_ENABLED = True  # Enable free tier optimizations
GPT35_TOKEN_BUDGET = 1000  # Daily token budget for free tier (approximate)
GPT35_COST_PER_1K_TOKENS = 0.002  # Cost per 1K tokens for gpt-3.5-turbo

# Fallback Configuration
GPT35_FALLBACK_ENABLED = True  # Enable automatic fallback to local processing
GPT35_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to use GPT-3.5 response
GPT35_COMPLEXITY_THRESHOLD = 0.7  # Query complexity threshold for GPT-3.5 usage

# Insurance-Specific Prompting
GPT35_INSURANCE_PROMPTS = {
    "coverage": {
        "focus": "Focus on what is and is not covered under the policy.",
        "keywords": ["cover", "coverage", "covered", "include", "exclude", "benefits"]
    },
    "claims": {
        "focus": "Focus on the claims process, requirements, and procedures.",
        "keywords": ["claim", "file", "submit", "process", "procedure", "documentation"]
    },
    "limits": {
        "focus": "Focus on coverage limits, deductibles, and maximum payouts.",
        "keywords": ["limit", "maximum", "deductible", "cap", "amount", "coverage"]
    },
    "costs": {
        "focus": "Focus on costs, pricing, fees, and payment information.",
        "keywords": ["cost", "price", "fee", "payment", "bill", "premium"]
    },
    "general": {
        "focus": "Provide general information and guidance based on the context.",
        "keywords": ["what", "how", "when", "where", "why", "explain"]
    }
}

# Error Handling Configuration
GPT35_ERROR_MESSAGES = {
    "rate_limit": "Service is temporarily busy. Please try again in a moment.",
    "quota_exceeded": "Service quota exceeded. Please try again later.",
    "invalid_request": "Invalid request format. Please rephrase your question.",
    "service_unavailable": "Service is temporarily unavailable. Using fallback processing.",
    "timeout": "Request timed out. Using fallback processing.",
    "unknown": "An unexpected error occurred. Using fallback processing."
}

# Performance Monitoring
GPT35_MONITORING_ENABLED = True  # Enable performance monitoring
GPT35_LOG_REQUESTS = True  # Log all GPT-3.5 requests
GPT35_LOG_RESPONSES = False  # Log response content (disable for privacy)
GPT35_LOG_PERFORMANCE = True  # Log performance metrics

# Environment-Specific Overrides
def get_gpt35_config() -> Dict[str, Any]:
    """
    Get GPT-3.5 configuration with environment-specific overrides.
    
    Returns:
        Dictionary containing all configuration settings
    """
    config = {
        "model": os.getenv("GPT35_MODEL", GPT35_MODEL),
        "max_tokens": int(os.getenv("GPT35_MAX_TOKENS", GPT35_MAX_TOKENS)),
        "temperature": float(os.getenv("GPT35_TEMPERATURE", GPT35_TEMPERATURE)),
        "top_p": float(os.getenv("GPT35_TOP_P", GPT35_TOP_P)),
        "max_retries": int(os.getenv("GPT35_MAX_RETRIES", GPT35_MAX_RETRIES)),
        "rate_limit_delay": float(os.getenv("GPT35_RATE_LIMIT_DELAY", GPT35_RATE_LIMIT_DELAY)),
        "batch_delay": float(os.getenv("GPT35_BATCH_DELAY", GPT35_BATCH_DELAY)),
        "exponential_backoff": os.getenv("GPT35_EXPONENTIAL_BACKOFF", str(GPT35_EXPONENTIAL_BACKOFF)).lower() == "true",
        "free_tier_enabled": os.getenv("GPT35_FREE_TIER_ENABLED", str(GPT35_FREE_TIER_ENABLED)).lower() == "true",
        "token_budget": int(os.getenv("GPT35_TOKEN_BUDGET", GPT35_TOKEN_BUDGET)),
        "cost_per_1k_tokens": float(os.getenv("GPT35_COST_PER_1K_TOKENS", GPT35_COST_PER_1K_TOKENS)),
        "fallback_enabled": os.getenv("GPT35_FALLBACK_ENABLED", str(GPT35_FALLBACK_ENABLED)).lower() == "true",
        "confidence_threshold": float(os.getenv("GPT35_CONFIDENCE_THRESHOLD", GPT35_CONFIDENCE_THRESHOLD)),
        "complexity_threshold": float(os.getenv("GPT35_COMPLEXITY_THRESHOLD", GPT35_COMPLEXITY_THRESHOLD)),
        "insurance_prompts": GPT35_INSURANCE_PROMPTS,
        "error_messages": GPT35_ERROR_MESSAGES,
        "monitoring_enabled": os.getenv("GPT35_MONITORING_ENABLED", str(GPT35_MONITORING_ENABLED)).lower() == "true",
        "log_requests": os.getenv("GPT35_LOG_REQUESTS", str(GPT35_LOG_REQUESTS)).lower() == "true",
        "log_responses": os.getenv("GPT35_LOG_RESPONSES", str(GPT35_LOG_RESPONSES)).lower() == "true",
        "log_performance": os.getenv("GPT35_LOG_PERFORMANCE", str(GPT35_LOG_PERFORMANCE)).lower() == "true"
    }
    
    return config

# Configuration validation
def validate_gpt35_config(config: Dict[str, Any]) -> bool:
    """
    Validate GPT-3.5 configuration settings.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Validate numeric ranges
        if not (0 <= config["temperature"] <= 2):
            print("❌ Temperature must be between 0 and 2")
            return False
        
        if not (0 <= config["top_p"] <= 1):
            print("❌ Top_p must be between 0 and 1")
            return False
        
        if config["max_tokens"] < 1:
            print("❌ Max tokens must be at least 1")
            return False
        
        if config["max_retries"] < 0:
            print("❌ Max retries must be non-negative")
            return False
        
        if config["rate_limit_delay"] < 0:
            print("❌ Rate limit delay must be non-negative")
            return False
        
        if not (0 <= config["confidence_threshold"] <= 1):
            print("❌ Confidence threshold must be between 0 and 1")
            return False
        
        if not (0 <= config["complexity_threshold"] <= 1):
            print("❌ Complexity threshold must be between 0 and 1")
            return False
        
        print("✅ GPT-3.5 configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    print("🔧 GPT-3.5 Configuration")
    print("=" * 40)
    
    # Get configuration
    config = get_gpt35_config()
    
    # Display key settings
    print(f"Model: {config['model']}")
    print(f"Max Tokens: {config['max_tokens']}")
    print(f"Temperature: {config['temperature']}")
    print(f"Max Retries: {config['max_retries']}")
    print(f"Rate Limit Delay: {config['rate_limit_delay']}s")
    print(f"Free Tier Enabled: {config['free_tier_enabled']}")
    print(f"Fallback Enabled: {config['fallback_enabled']}")
    print(f"Confidence Threshold: {config['confidence_threshold']}")
    print(f"Complexity Threshold: {config['complexity_threshold']}")
    
    # Validate configuration
    print(f"\nValidation: {'✅ Passed' if validate_gpt35_config(config) else '❌ Failed'}")
    
    # Show insurance prompt types
    print(f"\nInsurance Prompt Types: {list(config['insurance_prompts'].keys())}")
    
    # Show error message types
    print(f"Error Message Types: {list(config['error_messages'].keys())}")

