"""Configuration management for LocalRAG."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class LocalRAGConfig(BaseModel):
    """Configuration settings for LocalRAG system."""
    
    # Model settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face embedding model to use"
    )
    
    # Text processing settings
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # Vector store settings
    vector_store_path: str = Field(
        default="./data/vector_store", 
        description="Path to FAISS vector store"
    )
    
    # Document processing
    max_documents: Optional[int] = Field(
        default=None, 
        description="Maximum number of documents to process"
    )
    batch_size: int = Field(default=100, description="Batch size for processing")
    
    # Toxicity filtering
    toxicity_threshold: float = Field(
        default=0.7, 
        description="Threshold for toxicity filtering (0-1)"
    )
    enable_toxicity_filter: bool = Field(
        default=True, 
        description="Enable toxicity filtering"
    )
    
    # Translation settings
    google_credentials_path: Optional[str] = Field(
        default=None, 
        description="Path to Google Cloud credentials JSON"
    )
    default_language: str = Field(default="en", description="Default language code")
    
    # Model optimization
    use_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    use_4bit: bool = Field(default=False, description="Use 4-bit quantization")
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")
    
    # UI settings
    ui_host: str = Field(default="0.0.0.0", description="UI host address")
    ui_port: int = Field(default=8501, description="UI port")
    
    @classmethod
    def from_env(cls) -> "LocalRAGConfig":
        """Create configuration from environment variables."""
        config_dict = {}
        
        # Map environment variables to config fields
        env_mapping = {
            "EMBEDDING_MODEL": "embedding_model",
            "CHUNK_SIZE": "chunk_size",
            "CHUNK_OVERLAP": "chunk_overlap",
            "VECTOR_STORE_PATH": "vector_store_path",
            "MAX_DOCUMENTS": "max_documents",
            "BATCH_SIZE": "batch_size",
            "TOXICITY_THRESHOLD": "toxicity_threshold",
            "ENABLE_TOXICITY_FILTER": "enable_toxicity_filter",
            "GOOGLE_APPLICATION_CREDENTIALS": "google_credentials_path",
            "DEFAULT_LANGUAGE": "default_language",
            "USE_8BIT": "use_8bit",
            "USE_4BIT": "use_4bit",
            "DEVICE": "device",
            "UI_HOST": "ui_host",
            "UI_PORT": "ui_port",
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ["chunk_size", "chunk_overlap", "max_documents", 
                                 "batch_size", "ui_port"]:
                    config_dict[config_key] = int(value)
                elif config_key in ["toxicity_threshold"]:
                    config_dict[config_key] = float(value)
                elif config_key in ["enable_toxicity_filter", "use_8bit", "use_4bit"]:
                    config_dict[config_key] = value.lower() in ["true", "1", "yes"]
                else:
                    config_dict[config_key] = value
        
        return cls(**config_dict)
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        vector_store_dir = Path(self.vector_store_path).parent
        vector_store_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = LocalRAGConfig.from_env()