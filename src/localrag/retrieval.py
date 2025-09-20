"""
RAG retrieval functionality using LangChain.
"""

from typing import List, Dict, Any, Optional
import logging

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
except ImportError:
    try:
        # Fallback for older versions
        from langchain.schema import Document
        from langchain.retrievers.base import BaseRetriever
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun
    except ImportError:
        raise ImportError("langchain-core is required. Install it with: pip install langchain-core")

from .vector_store import VectorStore
from .toxicity_filter import ToxicityFilter

logger = logging.getLogger(__name__)


class RAGRetriever(BaseRetriever):
    """LangChain-compatible retriever for the RAG system."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 5,
        score_threshold: Optional[float] = None,
        use_toxicity_filter: bool = True,
        toxicity_filter: Optional[ToxicityFilter] = None
    ):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Vector store for similarity search
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            use_toxicity_filter: Whether to filter toxic queries
            toxicity_filter: Custom toxicity filter (optional)
        """
        super().__init__()
        self.vector_store = vector_store
        self.k = k
        self.score_threshold = score_threshold
        self.use_toxicity_filter = use_toxicity_filter
        
        if use_toxicity_filter and toxicity_filter is None:
            self.toxicity_filter = ToxicityFilter()
        else:
            self.toxicity_filter = toxicity_filter
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Filter toxic queries
        if self.use_toxicity_filter and self.toxicity_filter:
            if self.toxicity_filter.is_toxic(query):
                logger.warning("Toxic query detected, filtering results")
                return []
            
            # Clean the query
            query = self.toxicity_filter.clean_text(query)
        
        # Retrieve documents from vector store
        results = self.vector_store.search(
            query=query,
            k=self.k,
            score_threshold=self.score_threshold
        )
        
        # Convert to LangChain Document format
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    **result["metadata"],
                    "similarity_score": result["score"],
                    "retrieval_index": result["index"]
                }
            )
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} documents for query")
        return documents
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents (non-LangChain interface).
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of document dictionaries
        """
        original_k = self.k
        if k is not None:
            self.k = k
        
        try:
            documents = self._get_relevant_documents(query, run_manager=None)
            results = []
            
            for doc in documents:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("similarity_score", 0.0)
                })
            
            return results
            
        finally:
            self.k = original_k


class AdvancedRAGRetriever(RAGRetriever):
    """Advanced RAG retriever with additional features."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 5,
        score_threshold: Optional[float] = None,
        use_toxicity_filter: bool = True,
        toxicity_filter: Optional[ToxicityFilter] = None,
        use_reranking: bool = False,
        diversity_threshold: float = 0.8
    ):
        """
        Initialize advanced RAG retriever.
        
        Args:
            vector_store: Vector store for similarity search
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            use_toxicity_filter: Whether to filter toxic queries
            toxicity_filter: Custom toxicity filter
            use_reranking: Whether to use result reranking
            diversity_threshold: Threshold for diversity filtering
        """
        super().__init__(
            vector_store=vector_store,
            k=k,
            score_threshold=score_threshold,
            use_toxicity_filter=use_toxicity_filter,
            toxicity_filter=toxicity_filter
        )
        self.use_reranking = use_reranking
        self.diversity_threshold = diversity_threshold
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Enhanced document retrieval with additional processing."""
        # Get base results
        documents = super()._get_relevant_documents(query, run_manager=run_manager)
        
        if not documents:
            return documents
        
        # Apply diversity filtering
        if self.diversity_threshold < 1.0:
            documents = self._apply_diversity_filter(documents)
        
        # Apply reranking if enabled
        if self.use_reranking:
            documents = self._rerank_documents(query, documents)
        
        return documents
    
    def _apply_diversity_filter(self, documents: List[Document]) -> List[Document]:
        """
        Apply diversity filtering to reduce redundant results.
        
        Args:
            documents: List of documents
            
        Returns:
            Filtered list of documents
        """
        if len(documents) <= 1:
            return documents
        
        filtered_docs = [documents[0]]  # Always include the top result
        
        for doc in documents[1:]:
            # Check similarity with already selected documents
            is_diverse = True
            
            for selected_doc in filtered_docs:
                similarity = self.vector_store.embedding_model.get_similarity(
                    doc.page_content,
                    selected_doc.page_content
                )
                
                if similarity > self.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                filtered_docs.append(doc)
        
        logger.debug(f"Diversity filter: {len(documents)} -> {len(filtered_docs)} documents")
        return filtered_docs
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents based on additional criteria.
        
        Args:
            query: Original query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        # Simple reranking based on content length and metadata
        def rerank_score(doc: Document) -> float:
            base_score = doc.metadata.get("similarity_score", 0.0)
            
            # Bonus for appropriate content length
            content_length = len(doc.page_content)
            length_bonus = 0.0
            if 100 <= content_length <= 1000:
                length_bonus = 0.1
            elif content_length > 1000:
                length_bonus = -0.05  # Slight penalty for very long content
            
            # Bonus for having good metadata
            metadata_bonus = 0.05 if doc.metadata.get("chunk_id") is not None else 0.0
            
            return base_score + length_bonus + metadata_bonus
        
        # Sort by rerank score
        documents.sort(key=rerank_score, reverse=True)
        
        logger.debug("Applied reranking to documents")
        return documents


class MultiQueryRetriever(AdvancedRAGRetriever):
    """Retriever that generates multiple query variations for better results."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 5,
        query_variations: int = 3,
        **kwargs
    ):
        """
        Initialize multi-query retriever.
        
        Args:
            vector_store: Vector store for similarity search
            k: Number of documents to retrieve per query
            query_variations: Number of query variations to generate
            **kwargs: Additional arguments for parent class
        """
        super().__init__(vector_store=vector_store, k=k, **kwargs)
        self.query_variations = query_variations
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of the input query.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        variations = [query]  # Include original query
        
        # Simple query variation strategies
        words = query.split()
        
        if len(words) > 1:
            # Rearrange words
            variations.append(" ".join(words[::-1]))
            
            # Add question words if not present
            if not any(qw in query.lower() for qw in ["what", "how", "why", "when", "where", "who"]):
                variations.append(f"What is {query}?")
                variations.append(f"How to {query}?")
            
            # Truncate to key terms
            if len(words) > 3:
                variations.append(" ".join(words[:3]))
                variations.append(" ".join(words[-3:]))
        
        # Remove duplicates and limit to desired number
        unique_variations = list(dict.fromkeys(variations))
        return unique_variations[:self.query_variations]
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve documents using multiple query variations.
        
        Args:
            query: Original query
            run_manager: Callback manager
            
        Returns:
            Combined and deduplicated results
        """
        # Generate query variations
        variations = self._generate_query_variations(query)
        logger.debug(f"Generated {len(variations)} query variations")
        
        all_documents = []
        seen_content = set()
        
        for variation in variations:
            # Get documents for this variation
            docs = super()._get_relevant_documents(variation, run_manager=run_manager)
            
            # Deduplicate based on content
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    # Add variation info to metadata
                    doc.metadata["query_variation"] = variation
                    all_documents.append(doc)
        
        # Sort by similarity score and limit to k results
        all_documents.sort(
            key=lambda x: x.metadata.get("similarity_score", 0.0),
            reverse=True
        )
        
        final_docs = all_documents[:self.k]
        logger.info(f"Multi-query retrieval found {len(final_docs)} unique documents")
        
        return final_docs


class ContextualRetriever(AdvancedRAGRetriever):
    """Retriever that maintains conversation context."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 5,
        context_window: int = 3,
        **kwargs
    ):
        """
        Initialize contextual retriever.
        
        Args:
            vector_store: Vector store for similarity search
            k: Number of documents to retrieve
            context_window: Number of previous queries to consider
            **kwargs: Additional arguments for parent class
        """
        super().__init__(vector_store=vector_store, k=k, **kwargs)
        self.context_window = context_window
        self.query_history = []
    
    def _build_contextual_query(self, query: str) -> str:
        """
        Build a contextual query using previous queries.
        
        Args:
            query: Current query
            
        Returns:
            Enhanced query with context
        """
        if not self.query_history:
            return query
        
        # Use recent queries as context
        recent_queries = self.query_history[-self.context_window:]
        context = " ".join(recent_queries)
        
        # Combine context with current query
        contextual_query = f"{context} {query}"
        
        logger.debug(f"Built contextual query with {len(recent_queries)} previous queries")
        return contextual_query
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve documents with conversational context.
        
        Args:
            query: Current query
            run_manager: Callback manager
            
        Returns:
            Context-aware retrieved documents
        """
        # Build contextual query
        contextual_query = self._build_contextual_query(query)
        
        # Get documents using contextual query
        documents = super()._get_relevant_documents(contextual_query, run_manager=run_manager)
        
        # Add current query to history
        self.query_history.append(query)
        
        # Limit history size
        if len(self.query_history) > self.context_window * 2:
            self.query_history = self.query_history[-self.context_window:]
        
        return documents
    
    def clear_context(self):
        """Clear the query history."""
        self.query_history.clear()
        logger.info("Cleared query context")