"""
Embedding model functionality using Hugging Face sentence-transformers.
"""

from typing import List, Union, Optional
import logging
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required. Install it with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for Hugging Face embedding models."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
            normalize_embeddings: Whether to normalize embeddings to unit vectors
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            logger.info(f"Successfully loaded model on device: {self.model.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Embeddings as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=self.normalize_embeddings
            )
            
            logger.debug(f"Encoded {len(texts)} texts to embeddings of shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        return self.encode([text])[0]
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        embeddings = self.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def get_similarities(self, query: str, texts: List[str]) -> List[float]:
        """
        Calculate similarities between a query and multiple texts.
        
        Args:
            query: Query text
            texts: List of texts to compare against
            
        Returns:
            List of similarity scores
        """
        all_texts = [query] + texts
        embeddings = self.encode(all_texts)
        
        query_embedding = embeddings[0]
        text_embeddings = embeddings[1:]
        
        similarities = np.dot(text_embeddings, query_embedding)
        return similarities.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            return self.model.get_sentence_embedding_dimension()
        else:
            # Fallback: encode a sample text to get dimension
            sample_embedding = self.encode_single("sample text")
            return len(sample_embedding)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "device": str(self.model.device) if self.model else None,
            "embedding_dimension": self.get_embedding_dimension(),
            "normalize_embeddings": self.normalize_embeddings,
        }
        
        if hasattr(self.model, 'max_seq_length'):
            info["max_sequence_length"] = self.model.max_seq_length
        
        return info


class MultilingualEmbeddingModel(EmbeddingModel):
    """Embedding model optimized for multilingual text."""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize multilingual embedding model.
        
        Args:
            model_name: Name of multilingual model
            device: Device to run the model on
            normalize_embeddings: Whether to normalize embeddings
        """
        super().__init__(model_name, device, normalize_embeddings)


class CodeEmbeddingModel(EmbeddingModel):
    """Embedding model optimized for code."""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize code embedding model.
        
        Args:
            model_name: Name of code-specific model
            device: Device to run the model on
            normalize_embeddings: Whether to normalize embeddings
        """
        super().__init__(model_name, device, normalize_embeddings)