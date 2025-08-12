"""
Configuration Manager for Claimsure

Handles secure loading and validation of environment variables
and application configuration.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration and environment variables."""
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialize configuration manager.
        
        Args:
            env_file: Path to the .env file
        """
        self.env_file = env_file
        self.config = {}
        self.load_environment()
        self.validate_configuration()
    
    def load_environment(self) -> None:
        """Load environment variables from .env file and system."""
        try:
            # Load from .env file if it exists
            if os.path.exists(self.env_file):
                load_dotenv(self.env_file)
                logger.info(f"Loaded environment from {self.env_file}")
            else:
                logger.warning(f"Environment file {self.env_file} not found. Using system environment variables.")
            
            # Load all configuration
            self.config = self._load_configuration()
            
        except Exception as e:
            logger.error(f"Error loading environment: {e}")
            raise
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load all configuration values from environment."""
        return {
            # Python Version
            "python_version": os.getenv("PYTHON_VERSION", "3.13.5"),
            
            # API Keys
            "google_ai_api_key": os.getenv("GOOGLE_AI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            
            # Server Configuration
            "api_host": os.getenv("API_HOST", "0.0.0.0"),
            "api_port": int(os.getenv("API_PORT", "8000")),
            "api_reload": os.getenv("API_RELOAD", "true").lower() == "true",
            
            # Logging Configuration
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            
            # Document Processing
            "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
            "max_chunks_per_document": int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "100")),
            
            # Embedding Configuration
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "vector_store_type": os.getenv("VECTOR_STORE_TYPE", "faiss"),
            "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "384")),
            
            # LLM Configuration
            "max_tokens": int(os.getenv("MAX_TOKENS", "150")),
            "temperature": float(os.getenv("TEMPERATURE", "0.1")),
            "top_k": int(os.getenv("TOP_K", "5")),
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.5")),
            
            # File Storage
            "storage_path": os.getenv("STORAGE_PATH", "./data"),
            "cache_dir": os.getenv("CACHE_DIR", "./data/cache"),
            "vector_store_path": os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
            
            # Security Configuration
            "cors_origins": self._parse_cors_origins(os.getenv("CORS_ORIGINS", '["*"]')),
            "cors_allow_credentials": os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
            
            # Rate Limiting
            "rate_limit_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            "rate_limit_per_hour": int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")),
            
            # Environment Mode
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "true").lower() == "true",
            
            # Model Configuration
            "llm_service_priority": self._parse_llm_priority(os.getenv("LLM_SERVICE_PRIORITY", "gemini,claude,huggingface,gpt35,local")),
            "gemini_model": os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            "claude_model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            "huggingface_model": os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
            "gpt35_model": os.getenv("GPT35_MODEL", "gpt-3.5-turbo"),
            
            # Local Model Configuration
            "local_model_path": os.getenv("LOCAL_MODEL_PATH", "./models/mistral-7b-instruct"),
            "llama_cpp_path": os.getenv("LLAMA_CPP_PATH", "./llama.cpp"),
            
            # Feature Flags
            "enable_hybrid_processing": os.getenv("ENABLE_HYBRID_PROCESSING", "true").lower() == "true",
            "enable_fallback_logic": os.getenv("ENABLE_FALLBACK_LOGIC", "true").lower() == "true",
            "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
            "enable_monitoring": os.getenv("ENABLE_MONITORING", "true").lower() == "true",
            "enable_auto_scaling": os.getenv("ENABLE_AUTO_SCALING", "false").lower() == "true",
            
            # Testing Configuration
            "test_mode": os.getenv("TEST_MODE", "false").lower() == "true",
            "mock_llm_responses": os.getenv("MOCK_LLM_RESPONSES", "false").lower() == "true",
            "test_api_keys": os.getenv("TEST_API_KEYS", "false").lower() == "true",
        }
    
    def _parse_cors_origins(self, cors_origins: str) -> List[str]:
        """Parse CORS origins from string."""
        try:
            import json
            return json.loads(cors_origins)
        except (json.JSONDecodeError, ImportError):
            # Fallback to simple parsing
            return [origin.strip() for origin in cors_origins.strip('[]').split(',') if origin.strip()]
    
    def _parse_llm_priority(self, priority_string: str) -> List[str]:
        """Parse LLM service priority from comma-separated string."""
        return [service.strip().lower() for service in priority_string.split(',') if service.strip()]
    
    def validate_configuration(self) -> None:
        """Validate the loaded configuration."""
        logger.info("Validating configuration...")
        
        # Validate required API keys
        self._validate_api_keys()
        
        # Validate numeric values
        self._validate_numeric_values()
        
        # Validate file paths
        self._validate_paths()
        
        # Validate model configurations
        self._validate_model_configs()
        
        logger.info("✅ Configuration validation completed")
    
    def _validate_api_keys(self) -> None:
        """Validate API keys are properly formatted."""
        api_keys = {
            "GOOGLE_AI_API_KEY": self.config.get("google_ai_api_key"),
            "ANTHROPIC_API_KEY": self.config.get("anthropic_api_key"),
            "HUGGINGFACE_API_KEY": self.config.get("huggingface_api_key"),
            "OPENAI_API_KEY": self.config.get("openai_api_key"),
        }
        
        for key_name, key_value in api_keys.items():
            if key_value:
                # Check for common formatting issues
                if key_value.strip() != key_value:
                    logger.warning(f"{key_name} contains leading/trailing whitespace")
                
                if "\n" in key_value or "\r" in key_value:
                    logger.warning(f"{key_name} contains newline characters")
                
                # Validate key formats
                if key_name == "ANTHROPIC_API_KEY" and not key_value.startswith("sk-ant-"):
                    logger.warning(f"{key_name} doesn't match expected Claude API key format")
                
                if key_name == "HUGGINGFACE_API_KEY" and not key_value.startswith("hf_"):
                    logger.warning(f"{key_name} doesn't match expected Hugging Face API key format")
                
                if key_name == "OPENAI_API_KEY" and not key_value.startswith("sk-"):
                    logger.warning(f"{key_name} doesn't match expected OpenAI API key format")
                
                logger.info(f"✅ {key_name} is configured")
            else:
                logger.warning(f"⚠️ {key_name} is not configured")
    
    def _validate_numeric_values(self) -> None:
        """Validate numeric configuration values."""
        numeric_configs = {
            "chunk_size": (100, 10000),
            "chunk_overlap": (0, 1000),
            "max_chunks_per_document": (1, 10000),
            "embedding_dimension": (128, 4096),
            "max_tokens": (1, 10000),
            "top_k": (1, 100),
            "api_port": (1024, 65535),
            "rate_limit_per_minute": (1, 10000),
            "rate_limit_per_hour": (1, 100000),
        }
        
        for config_name, (min_val, max_val) in numeric_configs.items():
            value = self.config.get(config_name)
            if value is not None:
                if not (min_val <= value <= max_val):
                    logger.warning(f"{config_name} value {value} is outside recommended range [{min_val}, {max_val}]")
    
    def _validate_paths(self) -> None:
        """Validate file paths and create directories if needed."""
        paths_to_create = [
            self.config.get("storage_path"),
            self.config.get("cache_dir"),
            self.config.get("vector_store_path"),
        ]
        
        for path in paths_to_create:
            if path:
                try:
                    Path(path).mkdir(parents=True, exist_ok=True)
                    logger.info(f"✅ Directory ensured: {path}")
                except Exception as e:
                    logger.error(f"❌ Failed to create directory {path}: {e}")
    
    def _validate_model_configs(self) -> None:
        """Validate model configurations."""
        # Check if local model path exists if specified
        local_model_path = self.config.get("local_model_path")
        if local_model_path and not os.path.exists(local_model_path):
            logger.warning(f"Local model path does not exist: {local_model_path}")
        
        # Check if llama.cpp path exists if specified
        llama_cpp_path = self.config.get("llama_cpp_path")
        if llama_cpp_path and not os.path.exists(llama_cpp_path):
            logger.warning(f"llama.cpp path does not exist: {llama_cpp_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service."""
        key_mapping = {
            "gemini": "google_ai_api_key",
            "claude": "anthropic_api_key",
            "huggingface": "huggingface_api_key",
            "openai": "openai_api_key",
            "gpt35": "openai_api_key",
        }
        
        config_key = key_mapping.get(service.lower())
        if config_key:
            return self.config.get(config_key)
        return None
    
    def get_llm_services(self) -> List[str]:
        """Get list of available LLM services based on configured API keys."""
        available_services = []
        
        if self.config.get("google_ai_api_key"):
            available_services.append("gemini")
        
        if self.config.get("anthropic_api_key"):
            available_services.append("claude")
        
        if self.config.get("huggingface_api_key"):
            available_services.append("huggingface")
        
        if self.config.get("openai_api_key"):
            available_services.append("gpt35")
        
        # Check for local model availability
        if os.path.exists(self.config.get("local_model_path", "")):
            available_services.append("local")
        
        return available_services
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.config.get("environment", "development").lower() == "production"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config.get("debug", False)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration for logging."""
        return {
            "environment": self.config.get("environment"),
            "debug": self.config.get("debug"),
            "api_host": self.config.get("api_host"),
            "api_port": self.config.get("api_port"),
            "available_llm_services": self.get_llm_services(),
            "embedding_model": self.config.get("embedding_model"),
            "vector_store_type": self.config.get("vector_store_type"),
            "chunk_size": self.config.get("chunk_size"),
            "chunk_overlap": self.config.get("chunk_overlap"),
            "enable_hybrid_processing": self.config.get("enable_hybrid_processing"),
            "enable_caching": self.config.get("enable_caching"),
        }

# Global configuration instance
config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager

def init_config(env_file: str = ".env") -> ConfigManager:
    """Initialize the global configuration manager."""
    global config_manager
    config_manager = ConfigManager(env_file)
    return config_manager
