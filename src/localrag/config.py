"""
Configuration settings for LocalRAG system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class LocalRAGConfig:
    """Configuration manager for LocalRAG system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "device": None,  # Auto-detect
                "normalize_embeddings": True,
                "batch_size": 32
            },
            "chunking": {
                "chunk_size": 512,
                "overlap": 50,
                "min_chunk_size": 50,
                "use_tokens": True,
                "separators": ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            },
            "vector_store": {
                "index_type": "flat",  # flat, ivf, hnsw
                "metric_type": "cosine",  # cosine, euclidean, inner_product
                "save_path": "./data/vector_store"
            },
            "retrieval": {
                "k": 5,
                "score_threshold": None,
                "use_reranking": False,
                "diversity_threshold": 0.8
            },
            "toxicity_filter": {
                "enabled": True,
                "model_name": "martin-ha/toxic-comment-model",
                "threshold": 0.7,
                "use_simple_filter": True
            },
            "pdf_extraction": {
                "extract_metadata": True,
                "clean_text": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "performance": {
                "enable_caching": True,
                "cache_size": 1000,
                "parallel_processing": True,
                "max_workers": 4
            }
        }
    
    def load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge with default config
            self.config = self._merge_configs(self.config, user_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    
    def save_config(self, config_path: str):
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge user config with default config.
        
        Args:
            default: Default configuration
            user: User configuration
            
        Returns:
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'embedding.model_name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'embedding.model_name')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.get_section('logging')
        
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=level, format=format_str)
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
        """
        required_sections = ['embedding', 'chunking', 'vector_store', 'retrieval']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate embedding config
        embedding_config = self.get_section('embedding')
        if not embedding_config.get('model_name'):
            logger.error("Embedding model name is required")
            return False
        
        # Validate chunking config
        chunking_config = self.get_section('chunking')
        if chunking_config.get('chunk_size', 0) <= 0:
            logger.error("Chunk size must be positive")
            return False
        
        # Validate vector store config
        vector_config = self.get_section('vector_store')
        valid_index_types = ['flat', 'ivf', 'hnsw']
        if vector_config.get('index_type') not in valid_index_types:
            logger.error(f"Invalid index type. Must be one of: {valid_index_types}")
            return False
        
        valid_metrics = ['cosine', 'euclidean', 'inner_product']
        if vector_config.get('metric_type') not in valid_metrics:
            logger.error(f"Invalid metric type. Must be one of: {valid_metrics}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def create_sample_config(self, output_path: str):
        """
        Create a sample configuration file.
        
        Args:
            output_path: Path to save sample configuration
        """
        sample_config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu",
                "normalize_embeddings": True,
                "batch_size": 16
            },
            "chunking": {
                "chunk_size": 256,
                "overlap": 25,
                "min_chunk_size": 25
            },
            "vector_store": {
                "index_type": "flat",
                "metric_type": "cosine",
                "save_path": "./my_vector_store"
            },
            "retrieval": {
                "k": 3,
                "score_threshold": 0.2,
                "use_reranking": true,
                "diversity_threshold": 0.7
            },
            "toxicity_filter": {
                "enabled": true,
                "threshold": 0.8
            },
            "logging": {
                "level": "DEBUG"
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        logger.info(f"Created sample configuration: {output_path}")


# Global configuration instance
config = LocalRAGConfig()


def get_config() -> LocalRAGConfig:
    """Get the global configuration instance."""
    return config


def load_config(config_path: str):
    """Load configuration from file into global instance."""
    global config
    config.load_config(config_path)


def setup_logging():
    """Setup logging using global configuration."""
    config.setup_logging()