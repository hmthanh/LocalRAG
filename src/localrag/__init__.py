"""
LocalRAG: A local RAG system with PDF processing, vector storage, and toxicity filtering.
"""

__version__ = "0.1.0"

from .pdf_extractor import PDFExtractor
from .text_chunker import TextChunker
from .vector_store import VectorStore
from .retrieval import RAGRetriever
from .toxicity_filter import ToxicityFilter
from .embeddings import EmbeddingModel

__all__ = [
    "PDFExtractor",
    "TextChunker", 
    "VectorStore",
    "RAGRetriever",
    "ToxicityFilter",
    "EmbeddingModel",
]