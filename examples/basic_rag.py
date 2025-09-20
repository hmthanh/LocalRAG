#!/usr/bin/env python3
"""Basic RAG example for LocalRAG."""

import logging
from pathlib import Path

from localrag import (
    DocumentLoader,
    TextProcessor,
    create_vector_store,
    create_retriever
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run basic RAG example."""
    print("🔍 LocalRAG Basic RAG Example")
    print("=" * 40)
    
    # Configuration
    documents_dir = Path("./documents")
    vector_store_path = "./data/example_vector_store"
    
    # Check if documents directory exists
    if not documents_dir.exists():
        print(f"❌ Documents directory not found: {documents_dir}")
        print("Please create the directory and add some PDF files")
        return
    
    # Initialize components
    print("📚 Initializing components...")
    document_loader = DocumentLoader()
    text_processor = TextProcessor(chunk_size=800, chunk_overlap=100)
    
    # Load documents
    print(f"📄 Loading documents from {documents_dir}...")
    documents = list(document_loader.load_mixed_directory(
        documents_dir,
        max_files=5,  # Limit for example
        recursive=True
    ))
    
    if not documents:
        print("❌ No documents found!")
        return
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Process documents
    print("🔪 Processing and chunking documents...")
    chunks = text_processor.chunk_documents(documents)
    
    # Get statistics
    stats = text_processor.get_chunk_statistics(chunks)
    print(f"📊 Created {stats['total_chunks']} chunks")
    print(f"   Average chunk size: {stats['average_chunk_size']:.1f} characters")
    
    # Create vector store
    print("🗂️ Creating vector store...")
    vector_store = create_vector_store(
        documents=chunks,
        persist_directory=vector_store_path
    )
    
    print(f"✅ Vector store created with {len(chunks)} chunks")
    
    # Create retriever
    print("🔍 Creating retriever...")
    retriever = create_retriever(vector_store_path=vector_store_path)
    
    # Example queries
    example_queries = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
        "What are the important details mentioned?",
    ]
    
    print("\n💬 Testing queries...")
    print("=" * 40)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        
        try:
            result = retriever.search_and_respond(
                query=query,
                k=3,
                enable_safety=True
            )
            
            print(f"📋 Response:")
            print(result["response"])
            
            print(f"\n📚 Found {len(result['documents'])} relevant documents:")
            for j, doc in enumerate(result["documents"], 1):
                source = doc.metadata.get("file_name", "Unknown")
                print(f"  {j}. {source}")
                if "similarity_score" in doc.metadata:
                    print(f"     Similarity: {doc.metadata['similarity_score']:.3f}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Example completed!")
    print(f"Vector store saved to: {vector_store_path}")
    print("\nTry running more queries with:")
    print(f"  uv run localrag-query --vector-store {vector_store_path}")


if __name__ == "__main__":
    main()