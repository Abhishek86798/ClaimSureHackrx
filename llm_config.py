"""
Configuration file for Claimsure LLM services.
Manages settings for Claude 3.5 Sonnet, Gemini, Hugging Face, and Local Mistral services.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CLAUDE 3.5 SONNET CONFIGURATION (Primary LLM)
# =============================================================================

# API Configuration
CLAUDE_API_KEY = os.getenv('ANTHROPIC_API_KEY')
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
CLAUDE_MAX_TOKENS = 1000
CLAUDE_TEMPERATURE = 0.1

# Rate Limiting and Retry Configuration
CLAUDE_MAX_RETRIES = 3
CLAUDE_RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Service Priority
CLAUDE_PRIORITY = 1  # Highest priority (primary LLM)

# =============================================================================
# GEMINI CONFIGURATION (Primary Fallback)
# =============================================================================

# API Configuration
GEMINI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_MAX_TOKENS = 1000
GEMINI_TEMPERATURE = 0.1

# Rate Limiting and Retry Configuration
GEMINI_MAX_RETRIES = 3
GEMINI_RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Service Priority
GEMINI_PRIORITY = 2  # Second priority (primary fallback)

# =============================================================================
# HUGGING FACE CONFIGURATION (Alternative Fallback)
# =============================================================================

# API Configuration
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_DEFAULT_MODEL = "microsoft/DialoGPT-medium"
HUGGINGFACE_ALTERNATIVE_MODELS = [
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "microsoft/DialoGPT-small"
]

# Rate Limiting and Retry Configuration
HUGGINGFACE_MAX_RETRIES = 3
HUGGINGFACE_RATE_LIMIT_DELAY = 2.0  # seconds between requests (stricter limits)

# Service Priority
HUGGINGFACE_PRIORITY = 3  # Third priority (alternative fallback)

# =============================================================================
# LOCAL MISTRAL CONFIGURATION (Offline Backup)
# =============================================================================

# Model and Binary Paths
MISTRAL_MODEL_PATH = os.getenv('MISTRAL_MODEL_PATH')
LLAMA_CPP_PATH = os.getenv('LLAMA_CPP_PATH', 'llama.cpp')

# Model Configuration
MISTRAL_MAX_TOKENS = 512
MISTRAL_TEMPERATURE = 0.1
MISTRAL_REPEAT_PENALTY = 1.1

# Processing Configuration
MISTRAL_RATE_LIMIT_DELAY = 0.5  # Local processing is faster
MISTRAL_TIMEOUT = 120  # seconds

# Service Priority
MISTRAL_PRIORITY = 4  # Lowest priority (offline backup)

# =============================================================================
# HYBRID PROCESSOR CONFIGURATION
# =============================================================================

# LLM Service Selection Strategy
LLM_SELECTION_STRATEGY = "priority_based"  # Options: priority_based, confidence_based, round_robin

# Confidence Thresholds
MIN_CONFIDENCE_THRESHOLD = 0.6
CONFIDENCE_BOOST_THRESHOLD = 0.8

# Fallback Configuration
ENABLE_FALLBACKS = True
MAX_FALLBACK_ATTEMPTS = 3

# Query Classification Weights
QUERY_CLASSIFICATION_WEIGHTS = {
    "coverage": 1.0,
    "claims": 1.2,
    "limits": 1.3,
    "costs": 1.1,
    "general": 0.8
}

# =============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# =============================================================================

# FAISS Configuration
FAISS_INDEX_TYPE = "Flat"  # Options: Flat, IVF, HNSW
FAISS_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 embeddings
FAISS_NPROBE = 10  # Number of clusters to visit during search

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_CHUNK_SIZE = 512
EMBEDDING_CHUNK_OVERLAP = 50

# Retrieval Configuration
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.7

# =============================================================================
# LOGGING AND MONITORING CONFIGURATION
# =============================================================================

# Logging Levels
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING = True
PERFORMANCE_METRICS_INTERVAL = 60  # seconds

# =============================================================================
# ERROR HANDLING CONFIGURATION
# =============================================================================

# Retry Configuration
GLOBAL_MAX_RETRIES = 3
GLOBAL_RETRY_DELAY = 1.0  # seconds

# Error Recovery
ENABLE_ERROR_RECOVERY = True
ERROR_RECOVERY_TIMEOUT = 30  # seconds

# =============================================================================
# COST MANAGEMENT CONFIGURATION
# =============================================================================

# Token Budget Management
ENABLE_TOKEN_BUDGET_MANAGEMENT = True
DAILY_TOKEN_BUDGET = 10000
MONTHLY_TOKEN_BUDGET = 300000

# Cost Tracking
ENABLE_COST_TRACKING = True
COST_PER_1K_TOKENS = {
    "claude": 0.003,  # $0.003 per 1K input tokens
    "gemini": 0.0005,  # $0.0005 per 1K characters
    "huggingface": 0.0,  # Free tier
    "local": 0.0  # No cost for local processing
}

# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

# Development Environment
if os.getenv('ENVIRONMENT') == 'development':
    LOG_LEVEL = "DEBUG"
    ENABLE_PERFORMANCE_MONITORING = True
    ENABLE_COST_TRACKING = False

# Production Environment
elif os.getenv('ENVIRONMENT') == 'production':
    LOG_LEVEL = "WARNING"
    ENABLE_PERFORMANCE_MONITORING = True
    ENABLE_COST_TRACKING = True
    ENABLE_ERROR_RECOVERY = True

# Test Environment
elif os.getenv('ENVIRONMENT') == 'test':
    LOG_LEVEL = "INFO"
    ENABLE_PERFORMANCE_MONITORING = False
    ENABLE_COST_TRACKING = False
    ENABLE_ERROR_RECOVERY = False

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_configuration():
    """Validate the configuration and return any issues"""
    issues = []
    
    # Check required API keys
    if not CLAUDE_API_KEY:
        issues.append("Missing Claude API key (ANTHROPIC_API_KEY)")
    
    if not GEMINI_API_KEY:
        issues.append("Missing Gemini API key (GOOGLE_AI_API_KEY)")
    
    if not HUGGINGFACE_API_KEY:
        issues.append("Missing Hugging Face API key (HUGGINGFACE_API_KEY)")
    
    # Check model paths for local service
    if MISTRAL_MODEL_PATH and not os.path.exists(MISTRAL_MODEL_PATH):
        issues.append(f"Mistral model path does not exist: {MISTRAL_MODEL_PATH}")
    
    # Validate priorities
    priorities = [CLAUDE_PRIORITY, GEMINI_PRIORITY, HUGGINGFACE_PRIORITY, MISTRAL_PRIORITY]
    if len(set(priorities)) != len(priorities):
        issues.append("Duplicate priority values found in LLM services")
    
    return issues

def get_available_services():
    """Get list of available LLM services based on configuration"""
    available_services = []
    
    if CLAUDE_API_KEY:
        available_services.append("claude")
    
    if GEMINI_API_KEY:
        available_services.append("gemini")
    
    if HUGGINGFACE_API_KEY:
        available_services.append("huggingface")
    
    if MISTRAL_MODEL_PATH and os.path.exists(MISTRAL_MODEL_PATH):
        available_services.append("local_mistral")
    
    return available_services

def get_service_priority(service_name: str) -> int:
    """Get priority for a specific service"""
    priority_map = {
        "claude": CLAUDE_PRIORITY,
        "gemini": GEMINI_PRIORITY,
        "huggingface": HUGGINGFACE_PRIORITY,
        "local_mistral": MISTRAL_PRIORITY
    }
    return priority_map.get(service_name, 999)

def get_service_config(service_name: str) -> dict:
    """Get configuration for a specific service"""
    configs = {
        "claude": {
            "api_key": CLAUDE_API_KEY,
            "model": CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "temperature": CLAUDE_TEMPERATURE,
            "rate_limit_delay": CLAUDE_RATE_LIMIT_DELAY,
            "priority": CLAUDE_PRIORITY
        },
        "gemini": {
            "api_key": GEMINI_API_KEY,
            "model": GEMINI_MODEL,
            "max_tokens": GEMINI_MAX_TOKENS,
            "temperature": GEMINI_TEMPERATURE,
            "rate_limit_delay": GEMINI_RATE_LIMIT_DELAY,
            "priority": GEMINI_PRIORITY
        },
        "huggingface": {
            "api_key": HUGGINGFACE_API_KEY,
            "default_model": HUGGINGFACE_DEFAULT_MODEL,
            "alternative_models": HUGGINGFACE_ALTERNATIVE_MODELS,
            "rate_limit_delay": HUGGINGFACE_RATE_LIMIT_DELAY,
            "priority": HUGGINGFACE_PRIORITY
        },
        "local_mistral": {
            "model_path": MISTRAL_MODEL_PATH,
            "llama_cpp_path": LLAMA_CPP_PATH,
            "max_tokens": MISTRAL_MAX_TOKENS,
            "temperature": MISTRAL_TEMPERATURE,
            "rate_limit_delay": MISTRAL_RATE_LIMIT_DELAY,
            "priority": MISTRAL_PRIORITY
        }
    }
    return configs.get(service_name, {})

