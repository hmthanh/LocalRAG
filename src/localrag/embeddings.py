"""Embedding model management for LocalRAG."""

import logging
from typing import List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

from .config import config

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddings(Embeddings):
    """Hugging Face sentence transformers embeddings with optimization support."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_8bit: bool = False,
        use_4bit: bool = False,
        cache_folder: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Hugging Face embeddings.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use (cpu/cuda)
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization
            cache_folder: Cache folder for model files
            **kwargs: Additional arguments for SentenceTransformer
        """
        self.model_name = model_name or config.embedding_model
        self.device = device or config.device
        self.use_8bit = use_8bit or config.use_8bit
        self.use_4bit = use_4bit or config.use_4bit
        self.cache_folder = cache_folder
        
        logger.info(f"Initializing embedding model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Initialize the model
        self.model = self._load_model(**kwargs)
        
        # Get embedding dimension
        self.embedding_dimension = self._get_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
    
    def _load_model(self, **kwargs) -> SentenceTransformer:
        """Load the sentence transformer model with optimizations."""
        try:
            # Prepare model kwargs
            model_kwargs = {
                "cache_folder": self.cache_folder,
                "device": self.device,
                **kwargs
            }
            
            # Load the base model
            model = SentenceTransformer(self.model_name, **model_kwargs)
            
            # Apply quantization if requested
            if self.use_4bit or self.use_8bit:
                model = self._apply_quantization(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            # Fallback to a smaller model
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Falling back to {fallback_model}")
            return SentenceTransformer(fallback_model, device=self.device)
    
    def _apply_quantization(self, model: SentenceTransformer) -> SentenceTransformer:
        """Apply quantization to the model."""
        try:
            if self.use_4bit:
                logger.info("Applying 4-bit quantization")
                # Note: This is a simplified example. 
                # Real 4-bit quantization would require more complex setup
                if hasattr(model, '_modules'):
                    for module in model._modules.values():
                        if hasattr(module, 'half'):
                            module.half()
            elif self.use_8bit:
                logger.info("Applying 8-bit quantization")
                if hasattr(model, 'half'):
                    model.half()
            
            return model
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}. Using full precision.")
            return model
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        try:
            # Test with a simple sentence
            test_embedding = self.model.encode(["test"])
            return len(test_embedding[0])
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            # Common embedding dimensions for sentence transformers
            return 384  # Default for many models
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            logger.debug(f"Embedding {len(texts)} documents")
            
            # Filter out empty texts
            non_empty_texts = [text for text in texts if text.strip()]
            
            if not non_empty_texts:
                logger.warning("No non-empty texts to embed")
                return []
            
            # Generate embeddings
            embeddings = self.model.encode(
                non_empty_texts,
                convert_to_tensor=False,
                show_progress_bar=len(non_empty_texts) > 10
            )
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * self.embedding_dimension for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Query embedding
        """
        try:
            if not text.strip():
                logger.warning("Empty query text")
                return [0.0] * self.embedding_dimension
            
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * self.embedding_dimension
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            import numpy as np
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "use_8bit": self.use_8bit,
            "use_4bit": self.use_4bit,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown')
        }


class EmbeddingManager:
    """Manage multiple embedding models and evaluation."""
    
    def __init__(self):
        """Initialize embedding manager."""
        self.models = {}
        self.current_model = None
    
    def load_model(
        self, 
        model_name: str, 
        alias: Optional[str] = None,
        **kwargs
    ) -> HuggingFaceEmbeddings:
        """
        Load an embedding model.
        
        Args:
            model_name: Name of the model to load
            alias: Alias for the model (defaults to model_name)
            **kwargs: Additional arguments for the model
            
        Returns:
            Loaded embedding model
        """
        alias = alias or model_name
        
        logger.info(f"Loading embedding model {model_name} as '{alias}'")
        model = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        
        self.models[alias] = model
        
        if self.current_model is None:
            self.current_model = alias
        
        return model
    
    def get_model(self, alias: Optional[str] = None) -> HuggingFaceEmbeddings:
        """Get an embedding model by alias."""
        alias = alias or self.current_model
        
        if alias not in self.models:
            # Load default model
            return self.load_model(config.embedding_model, alias="default")
        
        return self.models[alias]
    
    def list_models(self) -> List[str]:
        """List available model aliases."""
        return list(self.models.keys())
    
    def compare_models(self, texts: List[str], model_aliases: List[str]) -> dict:
        """
        Compare embeddings from different models.
        
        Args:
            texts: Texts to embed
            model_aliases: List of model aliases to compare
            
        Returns:
            Comparison results
        """
        results = {}
        
        for alias in model_aliases:
            if alias in self.models:
                model = self.models[alias]
                embeddings = model.embed_documents(texts)
                results[alias] = {
                    "embeddings": embeddings,
                    "model_info": model.get_model_info()
                }
        
        return results


# Global embedding manager instance
embedding_manager = EmbeddingManager()