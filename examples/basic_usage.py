#!/usr/bin/env python3
"""
Basic usage example for LocalRAG system.

This script demonstrates how to:
1. Extract text from a PDF
2. Chunk the text
3. Create embeddings and vector store
4. Perform retrieval queries
5. Use toxicity filtering
"""

import os
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from localrag import (
    PDFExtractor,
    TextChunker,
    EmbeddingModel,
    VectorStore,
    RAGRetriever,
    ToxicityFilter
)


def main():
    """Demonstrate basic LocalRAG usage."""
    print("🤖 LocalRAG Basic Usage Example")
    print("=" * 50)
    
    # Create data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample text (simulate PDF content)
    sample_text = """
    Artificial Intelligence (AI) is a rapidly growing field that focuses on creating 
    intelligent machines capable of performing tasks that typically require human intelligence.
    
    Machine Learning is a subset of AI that enables computers to learn and improve from 
    experience without being explicitly programmed. It uses algorithms to analyze data, 
    identify patterns, and make predictions or decisions.
    
    Deep Learning is a specialized area of machine learning that uses neural networks 
    with multiple layers to model and understand complex patterns in data. It has been 
    particularly successful in areas like image recognition, natural language processing, 
    and speech recognition.
    
    Natural Language Processing (NLP) is another important area of AI that focuses on 
    enabling computers to understand, interpret, and generate human language. NLP 
    techniques are used in applications like chatbots, translation services, and 
    sentiment analysis.
    
    Computer Vision enables machines to interpret and understand visual information 
    from the world. It combines techniques from AI, machine learning, and image 
    processing to analyze and extract meaningful information from images and videos.
    """
    
    print("📝 Step 1: Text Chunking")
    print("-" * 30)
    
    # Initialize text chunker
    chunker = TextChunker(chunk_size=200, overlap=20)
    chunks = chunker.chunk_text(sample_text)
    
    print(f"Created {len(chunks)} chunks from sample text")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk['content'])} characters")
    
    print("\n🧠 Step 2: Creating Embeddings")
    print("-" * 30)
    
    # Initialize embedding model
    embedding_model = EmbeddingModel()
    print(f"Loaded embedding model: {embedding_model.model_name}")
    print(f"Embedding dimension: {embedding_model.get_embedding_dimension()}")
    
    print("\n📊 Step 3: Creating Vector Store")
    print("-" * 30)
    
    # Create vector store
    vector_store = VectorStore(embedding_model)
    vector_store.add_chunk_data(chunks)
    
    print(f"Vector store contains {vector_store.get_document_count()} documents")
    
    # Save vector store
    store_path = data_dir / "sample_vector_store"
    vector_store.save(str(store_path))
    print(f"Saved vector store to: {store_path}")
    
    print("\n🔍 Step 4: Setting up Retrieval")
    print("-" * 30)
    
    # Initialize retriever
    retriever = RAGRetriever(vector_store, k=3)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Tell me about computer vision",
        "What are the applications of NLP?"
    ]
    
    print("Testing retrieval with sample queries:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve(query)
        
        for i, result in enumerate(results, 1):
            print(f"  Result {i} (score: {result['score']:.3f}):")
            print(f"    {result['content'][:100]}...")
    
    print("\n🛡️ Step 5: Toxicity Filtering")
    print("-" * 30)
    
    # Initialize toxicity filter
    toxicity_filter = ToxicityFilter()
    
    # Test queries (including potentially problematic ones)
    test_queries_toxicity = [
        "How to build AI systems?",
        "I hate this technology",  # Potentially toxic
        "What are the benefits of machine learning?",
        "Ways to harm people with AI"  # Potentially toxic
    ]
    
    print("Testing toxicity filtering:")
    for query in test_queries_toxicity:
        is_toxic = toxicity_filter.is_toxic(query)
        toxicity_score = toxicity_filter.get_toxicity_score(query)
        cleaned = toxicity_filter.clean_text(query)
        
        print(f"\nQuery: '{query}'")
        print(f"  Toxic: {is_toxic} (score: {toxicity_score:.3f})")
        if is_toxic:
            print(f"  Cleaned: '{cleaned}'")
    
    print("\n🎯 Step 6: Safe RAG Query")
    print("-" * 30)
    
    # Create retriever with toxicity filtering
    safe_retriever = RAGRetriever(
        vector_store, 
        k=2, 
        use_toxicity_filter=True,
        toxicity_filter=toxicity_filter
    )
    
    safe_queries = [
        "Explain artificial intelligence",
        "I want to harm people with AI"  # This should be filtered
    ]
    
    for query in safe_queries:
        print(f"\nSafe query: '{query}'")
        results = safe_retriever.retrieve(query)
        
        if results:
            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result['content'][:80]}...")
        else:
            print("  No results (query may have been filtered)")
    
    print("\n✅ Basic usage example completed!")
    print("Check the 'data' directory for saved vector store files.")


if __name__ == "__main__":
    main()