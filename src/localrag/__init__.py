"""LocalRAG: A local RAG system with FAISS, LangChain, and Hugging Face."""

from .config import config, LocalRAGConfig
from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .embeddings import HuggingFaceEmbeddings, EmbeddingManager, embedding_manager
from .vector_store import FAISSVectorStore, create_vector_store
from .toxicity_filter import ToxicityFilter, SafetyWrapper, global_toxicity_filter
from .translator import Translator, DocumentTranslator, global_translator
from .retrieval import LocalRAGRetriever, create_retriever

__version__ = "0.1.0"
__author__ = "Thanh Hoang-Minh"
__email__ = "hmthanh@example.com"

# Main components
__all__ = [
    # Configuration
    "config",
    "LocalRAGConfig",
    
    # Document processing
    "DocumentLoader",
    "TextProcessor",
    
    # Embeddings
    "HuggingFaceEmbeddings",
    "EmbeddingManager",
    "embedding_manager",
    
    # Vector storage
    "FAISSVectorStore",
    "create_vector_store",
    
    # Safety and filtering
    "ToxicityFilter",
    "SafetyWrapper",
    "global_toxicity_filter",
    
    # Translation
    "Translator",
    "DocumentTranslator",
    "global_translator",
    
    # Retrieval
    "LocalRAGRetriever",
    "create_retriever",
]


def main() -> None:
    """Main entry point for LocalRAG."""
    print("Welcome to LocalRAG!")
    print("This is a local RAG system using FAISS, LangChain, and Hugging Face.")
    print("Use the CLI commands:")
    print("  localrag-ingest  - Ingest documents")
    print("  localrag-query   - Query the system")
    print("  localrag-ui      - Launch web UI")


if __name__ == "__main__":
    main()
