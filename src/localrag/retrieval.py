"""RAG retrieval implementation for LocalRAG."""

import logging
from typing import List, Optional, Dict, Any, Tuple

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from .vector_store import FAISSVectorStore, create_vector_store
from .toxicity_filter import global_safety_wrapper, SafetyWrapper
from .translator import global_document_translator, DocumentTranslator
from .config import config

logger = logging.getLogger(__name__)


class LocalRAGRetriever:
    """Local RAG retrieval system with safety and translation features."""
    
    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        safety_wrapper: Optional[SafetyWrapper] = None,
        document_translator: Optional[DocumentTranslator] = None,
        retrieval_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Vector store for document retrieval
            safety_wrapper: Safety wrapper for toxicity filtering
            document_translator: Document translator for multilingual support
            retrieval_kwargs: Additional retrieval parameters
        """
        self.vector_store = vector_store or create_vector_store()
        self.safety_wrapper = safety_wrapper or global_safety_wrapper
        self.document_translator = document_translator or global_document_translator
        self.retrieval_kwargs = retrieval_kwargs or {"k": 4}
        
        logger.info("LocalRAG retriever initialized")
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        enable_safety: bool = True,
        target_language: Optional[str] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            filter_metadata: Metadata filter for document selection
            enable_safety: Whether to apply safety filtering
            target_language: Target language for translation
            
        Returns:
            Tuple of (retrieved_documents, metadata)
        """
        metadata = {
            "original_query": query,
            "safety_applied": False,
            "translation_applied": False,
            "toxicity_scores": {},
            "query_safe": True
        }
        
        # Apply safety filtering to query
        processed_query = query
        if enable_safety:
            processed_query, is_safe, toxicity_scores = self.safety_wrapper.safe_query(query)
            metadata.update({
                "safety_applied": True,
                "query_safe": is_safe,
                "toxicity_scores": toxicity_scores
            })
            
            if not is_safe:
                logger.warning("Unsafe query detected, proceeding with filtered version")
        
        # Apply translation if needed
        if target_language and target_language != config.default_language:
            translation_result = self.document_translator.translate_query(
                processed_query, config.default_language
            )
            
            if translation_result["translation_performed"]:
                processed_query = translation_result["translated_query"]
                metadata.update({
                    "translation_applied": True,
                    "detected_language": translation_result.get("detected_language"),
                    "translated_query": processed_query
                })
        
        # Retrieve documents
        k = k or self.retrieval_kwargs.get("k", 4)
        
        try:
            if self.vector_store.index is None or len(self.vector_store.docstore) == 0:
                logger.warning("Vector store is empty")
                return [], metadata
            
            documents = self.vector_store.similarity_search(
                processed_query,
                k=k,
                filter=filter_metadata
            )
            
            metadata.update({
                "retrieved_count": len(documents),
                "requested_count": k
            })
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return [], metadata
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        enable_safety: bool = True,
        target_language: Optional[str] = None
    ) -> Tuple[List[Tuple[Document, float]], Dict[str, Any]]:
        """
        Retrieve relevant documents with similarity scores.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            filter_metadata: Metadata filter for document selection
            enable_safety: Whether to apply safety filtering
            target_language: Target language for translation
            
        Returns:
            Tuple of (documents_with_scores, metadata)
        """
        metadata = {
            "original_query": query,
            "safety_applied": False,
            "translation_applied": False,
            "toxicity_scores": {},
            "query_safe": True
        }
        
        # Apply safety filtering to query
        processed_query = query
        if enable_safety:
            processed_query, is_safe, toxicity_scores = self.safety_wrapper.safe_query(query)
            metadata.update({
                "safety_applied": True,
                "query_safe": is_safe,
                "toxicity_scores": toxicity_scores
            })
        
        # Apply translation if needed
        if target_language and target_language != config.default_language:
            translation_result = self.document_translator.translate_query(
                processed_query, config.default_language
            )
            
            if translation_result["translation_performed"]:
                processed_query = translation_result["translated_query"]
                metadata.update({
                    "translation_applied": True,
                    "detected_language": translation_result.get("detected_language"),
                    "translated_query": processed_query
                })
        
        # Retrieve documents with scores
        k = k or self.retrieval_kwargs.get("k", 4)
        
        try:
            if self.vector_store.index is None or len(self.vector_store.docstore) == 0:
                logger.warning("Vector store is empty")
                return [], metadata
            
            documents_with_scores = self.vector_store.similarity_search_with_score(
                processed_query,
                k=k,
                filter=filter_metadata
            )
            
            metadata.update({
                "retrieved_count": len(documents_with_scores),
                "requested_count": k,
                "scores": [score for _, score in documents_with_scores]
            })
            
            logger.info(f"Retrieved {len(documents_with_scores)} documents with scores")
            return documents_with_scores, metadata
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {e}")
            return [], metadata
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            
            context_parts.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        documents: List[Document],
        template: Optional[str] = None
    ) -> str:
        """
        Generate response based on query and retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents
            template: Optional custom prompt template
            
        Returns:
            Generated response
        """
        if not documents:
            return "I don't have enough information to answer your query. Please try adding more documents to the system."
        
        # Format context
        context = self.format_context(documents)
        
        # Use default template if none provided
        if template is None:
            template = """Based on the following context, please answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        
        # Simple template-based response (in a real implementation, this would use an LLM)
        response = f"""Based on the provided documents, here's what I found relevant to your query:

Query: {query}

Retrieved Information:
{context}

Please note: This is a template-based response. In a full implementation, this would be processed by a language model to provide a more coherent and specific answer."""
        
        return response
    
    def search_and_respond(
        self,
        query: str,
        k: Optional[int] = None,
        enable_safety: bool = True,
        target_language: Optional[str] = None,
        response_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: search documents and generate response.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            enable_safety: Whether to apply safety filtering
            target_language: Target language for translation
            response_template: Optional custom response template
            
        Returns:
            Complete response with metadata
        """
        # Retrieve documents
        documents, retrieval_metadata = self.retrieve_documents(
            query=query,
            k=k,
            enable_safety=enable_safety,
            target_language=target_language
        )
        
        # Generate response
        response = self.generate_response(query, documents, response_template)
        
        # Apply safety filtering to response
        final_response = response
        response_metadata = {}
        
        if enable_safety:
            final_response, is_safe, toxicity_scores = self.safety_wrapper.safe_response(response)
            response_metadata = {
                "response_safe": is_safe,
                "response_toxicity_scores": toxicity_scores
            }
        
        # Apply translation to response if needed
        if target_language and target_language != config.default_language:
            translation_result = self.document_translator.translate_response(
                final_response,
                target_language=target_language,
                source_language=config.default_language
            )
            
            if translation_result["translation_performed"]:
                final_response = translation_result["translated_response"]
                response_metadata.update({
                    "response_translated": True,
                    "original_response": translation_result["original_response"]
                })
        
        return {
            "query": query,
            "response": final_response,
            "documents": documents,
            "retrieval_metadata": retrieval_metadata,
            "response_metadata": response_metadata,
            "document_count": len(documents)
        }
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "vector_store_stats": vector_stats,
            "safety_enabled": self.safety_wrapper.toxicity_filter.enabled,
            "translation_enabled": self.document_translator.translator.enabled,
            "retrieval_config": self.retrieval_kwargs
        }


class SimpleRetrieverChain:
    """Simple retrieval chain for basic RAG functionality."""
    
    def __init__(self, retriever: LocalRAGRetriever):
        """
        Initialize simple retriever chain.
        
        Args:
            retriever: LocalRAG retriever instance
        """
        self.retriever = retriever
    
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Invoke the retrieval chain.
        
        Args:
            query: User query
            **kwargs: Additional arguments
            
        Returns:
            Chain response
        """
        return self.retriever.search_and_respond(query, **kwargs)
    
    def batch(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            **kwargs: Additional arguments
            
        Returns:
            List of responses
        """
        return [self.invoke(query, **kwargs) for query in queries]


def create_retriever(
    vector_store_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
    **kwargs
) -> LocalRAGRetriever:
    """
    Create a LocalRAG retriever.
    
    Args:
        vector_store_path: Path to vector store
        embedding_model: Embedding model to use
        **kwargs: Additional arguments for retriever
        
    Returns:
        LocalRAG retriever instance
    """
    vector_store = create_vector_store(
        persist_directory=vector_store_path,
        embedding_model=embedding_model
    )
    
    return LocalRAGRetriever(vector_store=vector_store, **kwargs)