"""FAISS vector store implementation for LocalRAG."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import faiss
import numpy as np
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

from .embeddings import HuggingFaceEmbeddings, embedding_manager
from .config import config

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(
        self,
        embedding_function: Optional[HuggingFaceEmbeddings] = None,
        index: Optional[faiss.Index] = None,
        docstore: Optional[Dict[int, Document]] = None,
        index_to_docstore_id: Optional[Dict[int, int]] = None,
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_function: Embedding function to use
            index: FAISS index
            docstore: Document store
            index_to_docstore_id: Mapping from index to docstore IDs
        """
        self.embedding_function = embedding_function or embedding_manager.get_model()
        self.index = index
        self.docstore = docstore or {}
        self.index_to_docstore_id = index_to_docstore_id or {}
        self._next_id = 0
        
        logger.info(f"Initialized FAISS vector store with {len(self.docstore)} documents")
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[HuggingFaceEmbeddings] = None,
        **kwargs
    ) -> "FAISSVectorStore":
        """
        Create vector store from documents.
        
        Args:
            documents: List of documents to add
            embedding: Embedding function to use
            **kwargs: Additional arguments
            
        Returns:
            Initialized vector store
        """
        embedding = embedding or embedding_manager.get_model()
        vector_store = cls(embedding_function=embedding, **kwargs)
        vector_store.add_documents(documents)
        return vector_store
    
    @classmethod
    def load_local(
        cls,
        folder_path: Union[str, Path],
        embeddings: Optional[HuggingFaceEmbeddings] = None
    ) -> "FAISSVectorStore":
        """
        Load vector store from local directory.
        
        Args:
            folder_path: Path to folder containing vector store files
            embeddings: Embedding function to use
            
        Returns:
            Loaded vector store
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Vector store folder not found: {folder_path}")
        
        # Load FAISS index
        index_path = folder_path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        index = faiss.read_index(str(index_path))
        
        # Load docstore and metadata
        docstore_path = folder_path / "docstore.pkl"
        metadata_path = folder_path / "metadata.pkl"
        
        docstore = {}
        index_to_docstore_id = {}
        
        if docstore_path.exists():
            with open(docstore_path, 'rb') as f:
                docstore = pickle.load(f)
        
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                index_to_docstore_id = metadata.get('index_to_docstore_id', {})
        
        embeddings = embeddings or embedding_manager.get_model()
        
        vector_store = cls(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        # Set next ID
        if docstore:
            vector_store._next_id = max(docstore.keys()) + 1
        
        logger.info(f"Loaded vector store with {len(docstore)} documents")
        return vector_store
    
    def save_local(self, folder_path: Union[str, Path]) -> None:
        """
        Save vector store to local directory.
        
        Args:
            folder_path: Path to folder to save vector store
        """
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Save FAISS index
        index_path = folder_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save docstore
        docstore_path = folder_path / "docstore.pkl"
        with open(docstore_path, 'wb') as f:
            pickle.dump(self.docstore, f)
        
        # Save metadata
        metadata_path = folder_path / "metadata.pkl"
        metadata = {
            'index_to_docstore_id': self.index_to_docstore_id,
            'embedding_model': self.embedding_function.model_name,
            'embedding_dimension': self.embedding_function.embedding_dimension
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved vector store to {folder_path}")
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index for given dimension."""
        # Use IndexFlatIP for cosine similarity (after normalization)
        index = faiss.IndexFlatIP(dimension)
        logger.info(f"Created FAISS index with dimension {dimension}")
        return index
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: Documents to add
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embedding_function.embed_documents(texts)
        
        if not embeddings:
            logger.error("Failed to generate embeddings")
            return []
        
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Create index if it doesn't exist
        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = self._create_index(dimension)
        
        # Add embeddings to index
        start_idx = self.index.ntotal
        self.index.add(embeddings_array)
        
        # Add documents to docstore
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_id = self._next_id
            self.docstore[doc_id] = doc
            self.index_to_docstore_id[start_idx + i] = doc_id
            doc_ids.append(str(doc_id))
            self._next_id += 1
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return doc_ids
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: Texts to add
            metadatas: Optional metadata for each text
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        # Convert texts to documents
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        
        return self.add_documents(documents, **kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        if self.index is None or len(self.docstore) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search in index
        scores, indices = self.index.search(query_array, k)
        
        # Get documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.index_to_docstore_id:
                doc_id = self.index_to_docstore_id[idx]
                if doc_id in self.docstore:
                    doc = self.docstore[doc_id]
                    
                    # Apply filter if provided
                    if filter and not self._matches_filter(doc.metadata, filter):
                        continue
                    
                    # Add similarity score to metadata
                    doc_copy = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "similarity_score": float(scores[0][i])}
                    )
                    results.append(doc_copy)
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples
        """
        if self.index is None or len(self.docstore) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search in index
        scores, indices = self.index.search(query_array, k)
        
        # Get documents with scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.index_to_docstore_id:
                doc_id = self.index_to_docstore_id[idx]
                if doc_id in self.docstore:
                    doc = self.docstore[doc_id]
                    
                    # Apply filter if provided
                    if filter and not self._matches_filter(doc.metadata, filter):
                        continue
                    
                    results.append((doc, float(scores[0][i])))
        
        return results
    
    def _matches_filter(self, metadata: dict, filter_dict: dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.docstore),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_function.embedding_dimension,
            "embedding_model": self.embedding_function.model_name
        }
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        Note: This creates a new index without the deleted documents.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        # Convert string IDs to integers
        doc_ids_int = [int(doc_id) for doc_id in doc_ids]
        
        # Remove from docstore
        for doc_id in doc_ids_int:
            if doc_id in self.docstore:
                del self.docstore[doc_id]
        
        # Rebuild index and mappings
        if self.docstore:
            documents = list(self.docstore.values())
            self._rebuild_index(documents)
        else:
            self.index = None
            self.index_to_docstore_id = {}
        
        logger.info(f"Deleted {len(doc_ids)} documents from vector store")
    
    def _rebuild_index(self, documents: List[Document]) -> None:
        """Rebuild the FAISS index with given documents."""
        # Reset mappings
        self.index_to_docstore_id = {}
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function.embed_documents(texts)
        
        if embeddings:
            # Create new index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            dimension = embeddings_array.shape[1]
            self.index = self._create_index(dimension)
            self.index.add(embeddings_array)
            
            # Rebuild mappings
            for i, doc_id in enumerate(self.docstore.keys()):
                self.index_to_docstore_id[i] = doc_id


def create_vector_store(
    documents: Optional[List[Document]] = None,
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None
) -> FAISSVectorStore:
    """
    Create or load a FAISS vector store.
    
    Args:
        documents: Documents to add to new vector store
        persist_directory: Directory to load/save vector store
        embedding_model: Embedding model to use
        
    Returns:
        FAISS vector store
    """
    persist_directory = persist_directory or config.vector_store_path
    
    # Load embedding model
    if embedding_model and embedding_model != config.embedding_model:
        embeddings = embedding_manager.load_model(embedding_model)
    else:
        embeddings = embedding_manager.get_model()
    
    # Try to load existing vector store
    if persist_directory and Path(persist_directory).exists():
        try:
            vector_store = FAISSVectorStore.load_local(persist_directory, embeddings)
            
            # Add new documents if provided
            if documents:
                vector_store.add_documents(documents)
                vector_store.save_local(persist_directory)
            
            return vector_store
            
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}")
    
    # Create new vector store
    if documents:
        vector_store = FAISSVectorStore.from_documents(documents, embeddings)
    else:
        vector_store = FAISSVectorStore(embedding_function=embeddings)
    
    # Save if persist directory is provided
    if persist_directory:
        vector_store.save_local(persist_directory)
    
    return vector_store