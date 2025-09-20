"""
Vector storage functionality using FAISS for similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import pickle
import logging
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required. Install it with: pip install faiss-cpu")

from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index_type: str = "flat",
        metric_type: str = "cosine"
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_model: Embedding model to use
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            metric_type: Distance metric ('cosine', 'euclidean', 'inner_product')
        """
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.metric_type = metric_type
        
        self.dimension = embedding_model.get_embedding_dimension()
        self.index = None
        self.documents = []  # Store original documents
        self.metadata = []   # Store metadata for each document
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on configuration."""
        logger.info(f"Creating {self.index_type} index with {self.metric_type} metric")
        
        if self.metric_type == "cosine":
            # For cosine similarity, we use inner product with normalized vectors
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 40
        elif self.metric_type == "euclidean":
            if self.index_type == "flat":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        elif self.metric_type == "inner_product":
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        
        logger.info(f"Created index: {type(self.index).__name__}")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32
    ):
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional metadata for each text
            batch_size: Batch size for embedding generation
        """
        if not texts:
            return
        
        logger.info(f"Adding {len(texts)} texts to vector store")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        
        # Train index if needed (for IVF indices)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in texts])
        
        logger.info(f"Vector store now contains {self.index.ntotal} documents")
    
    def add_chunk_data(self, chunks: List[Dict[str, Any]], batch_size: int = 32):
        """
        Add chunk data from text chunker to vector store.
        
        Args:
            chunks: List of chunk dictionaries from TextChunker
            batch_size: Batch size for embedding generation
        """
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk["id"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "size": chunk["size"]
            }
            for chunk in chunks
        ]
        
        self.add_texts(texts, metadatas, batch_size)
    
    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with documents and scores
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            if score_threshold and score < score_threshold:
                continue
            
            result = {
                "content": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": float(score),
                "index": int(idx)
            }
            results.append(result)
        
        logger.debug(f"Found {len(results)} results for query")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar documents and return with scores.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        results = self.search(query, k)
        return [(result["content"], result["score"]) for result in results]
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        return self.index.ntotal
    
    def delete_by_ids(self, ids: List[int]):
        """
        Delete documents by their indices.
        Note: This is expensive and requires rebuilding the index.
        
        Args:
            ids: List of document indices to delete
        """
        if not ids:
            return
        
        logger.warning("Deleting documents requires rebuilding the index")
        
        # Remove documents and metadata
        ids_set = set(ids)
        new_documents = []
        new_metadata = []
        
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            if i not in ids_set:
                new_documents.append(doc)
                new_metadata.append(meta)
        
        # Rebuild index
        self.documents = new_documents
        self.metadata = new_metadata
        self._create_index()
        
        if self.documents:
            self.add_texts(self.documents, self.metadata)
    
    def save(self, path: str):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the store
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # Save configuration
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "embedding_model_name": self.embedding_model.model_name,
            "normalize_embeddings": self.embedding_model.normalize_embeddings,
        }
        
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved vector store to {path}")
    
    def load(self, path: str):
        """
        Load the vector store from disk.
        
        Args:
            path: Directory path to load the store from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store path not found: {path}")
        
        # Load configuration
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        # Verify configuration matches
        if config["dimension"] != self.dimension:
            raise ValueError(f"Dimension mismatch: {config['dimension']} vs {self.dimension}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents and metadata
        with open(path / "documents.json", "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded vector store from {path} with {len(self.documents)} documents")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "total_documents": self.get_document_count(),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "embedding_model": self.embedding_model.model_name,
        }