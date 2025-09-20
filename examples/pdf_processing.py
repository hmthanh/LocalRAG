#!/usr/bin/env python3
"""
PDF processing example for LocalRAG system.

This script demonstrates how to:
1. Process multiple PDF files
2. Extract and clean text
3. Create optimized chunks
4. Build a comprehensive vector store
5. Perform advanced retrieval
"""

import os
import sys
from pathlib import Path
import logging

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from localrag import (
    PDFExtractor,
    TextChunker,
    EmbeddingModel,
    VectorStore,
    AdvancedRAGRetriever,
    ToxicityFilter
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_pdf_content():
    """Create sample content that would come from PDFs."""
    return {
        "research_paper.pdf": """
        Deep Learning for Natural Language Processing
        
        Abstract: This paper presents a comprehensive survey of deep learning techniques 
        applied to natural language processing tasks. We discuss various neural network 
        architectures including recurrent neural networks, convolutional neural networks, 
        and transformer models.
        
        Introduction: Natural language processing has seen remarkable progress with the 
        advent of deep learning. Traditional methods relied heavily on feature engineering 
        and statistical models, but modern approaches use neural networks to learn 
        representations directly from data.
        
        Methodology: We implemented several state-of-the-art models including BERT, GPT, 
        and T5. Each model was trained on large-scale text corpora and evaluated on 
        standard benchmarks.
        
        Results: The transformer-based models achieved state-of-the-art performance on 
        most tasks, with BERT excelling at understanding tasks and GPT performing well 
        on generation tasks.
        
        Conclusion: Deep learning has revolutionized NLP, enabling more sophisticated 
        and context-aware applications. Future work should focus on improving efficiency 
        and reducing computational requirements.
        """,
        
        "ai_ethics.pdf": """
        Ethical Considerations in Artificial Intelligence
        
        Executive Summary: As AI systems become more prevalent in society, it is crucial 
        to address ethical concerns surrounding their development and deployment. This 
        document outlines key ethical principles and guidelines.
        
        Fairness and Bias: AI systems must be designed to treat all individuals fairly, 
        regardless of race, gender, age, or other protected characteristics. Bias in 
        training data can lead to discriminatory outcomes.
        
        Transparency and Explainability: Users should understand how AI systems make 
        decisions, especially in high-stakes applications like healthcare, finance, 
        and criminal justice.
        
        Privacy and Data Protection: AI systems often require large amounts of personal 
        data. It is essential to protect user privacy and ensure data is used responsibly.
        
        Accountability: Clear lines of responsibility must be established for AI system 
        outcomes. Organizations deploying AI must be accountable for their systems' 
        behavior.
        
        Human Oversight: While AI can automate many tasks, human oversight remains 
        crucial for critical decisions and system monitoring.
        """,
        
        "machine_learning_guide.pdf": """
        A Practical Guide to Machine Learning
        
        Chapter 1: Introduction to Machine Learning
        Machine learning is a method of data analysis that automates analytical model 
        building. It uses algorithms that iteratively learn from data, allowing computers 
        to find insights without being explicitly programmed.
        
        Chapter 2: Types of Machine Learning
        Supervised Learning: Uses labeled training data to learn a mapping function from 
        input variables to output variables. Examples include regression and classification.
        
        Unsupervised Learning: Finds hidden patterns in data without labeled examples. 
        Common techniques include clustering and dimensionality reduction.
        
        Reinforcement Learning: Learns through interaction with an environment, receiving 
        rewards or penalties for actions taken.
        
        Chapter 3: Common Algorithms
        Linear Regression: Predicts a continuous target variable using linear relationships.
        Decision Trees: Creates a model that predicts target values by learning simple 
        decision rules inferred from data features.
        Neural Networks: Mimics the human brain's structure to recognize patterns and 
        make predictions.
        
        Chapter 4: Model Evaluation
        It's crucial to evaluate model performance using appropriate metrics and validation 
        techniques to ensure reliability and generalizability.
        """
    }


def main():
    """Demonstrate PDF processing with LocalRAG."""
    print("📄 LocalRAG PDF Processing Example")
    print("=" * 50)
    
    # Create directories
    data_dir = Path("./data")
    pdf_dir = data_dir / "pdfs"
    data_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)
    
    # Get sample PDF content
    pdf_contents = create_sample_pdf_content()
    
    print("📥 Step 1: Simulating PDF Processing")
    print("-" * 40)
    
    # Initialize PDF extractor (we'll simulate with sample content)
    extractor = PDFExtractor()
    
    all_documents = []
    all_metadata = []
    
    for filename, content in pdf_contents.items():
        print(f"Processing: {filename}")
        
        # In real usage, you would use: content = extractor.extract_text(pdf_path)
        # Here we use simulated content
        
        print(f"  Extracted {len(content)} characters")
        
        # Store document info
        all_documents.append({
            "filename": filename,
            "content": content,
            "length": len(content)
        })
    
    print(f"\nProcessed {len(all_documents)} documents")
    
    print("\n✂️ Step 2: Advanced Text Chunking")
    print("-" * 40)
    
    # Initialize chunker with optimized settings
    chunker = TextChunker(
        chunk_size=300,
        overlap=50,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    
    all_chunks = []
    
    for doc in all_documents:
        print(f"Chunking: {doc['filename']}")
        chunks = chunker.chunk_text(doc["content"])
        
        # Add source document metadata to each chunk
        for chunk in chunks:
            chunk["source_file"] = doc["filename"]
            chunk["source_length"] = doc["length"]
        
        all_chunks.extend(chunks)
        print(f"  Created {len(chunks)} chunks")
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    
    # Merge small chunks for better quality
    merged_chunks = chunker.merge_small_chunks(all_chunks, min_size=50)
    print(f"After merging small chunks: {len(merged_chunks)}")
    
    print("\n🧠 Step 3: Creating Embeddings with Multilingual Support")
    print("-" * 40)
    
    # Use a more capable embedding model
    embedding_model = EmbeddingModel(
        model_name="all-MiniLM-L6-v2",  # Good balance of speed and quality
        normalize_embeddings=True
    )
    
    print(f"Model: {embedding_model.model_name}")
    print(f"Dimension: {embedding_model.get_embedding_dimension()}")
    print(f"Device: {embedding_model.model.device}")
    
    print("\n📊 Step 4: Building Advanced Vector Store")
    print("-" * 40)
    
    # Create vector store with IVF index for better performance
    vector_store = VectorStore(
        embedding_model=embedding_model,
        index_type="flat",  # Use flat for small datasets, IVF for larger ones
        metric_type="cosine"
    )
    
    # Add chunks with progress tracking
    print("Adding chunks to vector store...")
    vector_store.add_chunk_data(merged_chunks, batch_size=16)
    
    stats = vector_store.get_stats()
    print(f"Vector store statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save the vector store
    store_path = data_dir / "pdf_vector_store"
    vector_store.save(str(store_path))
    print(f"Saved vector store to: {store_path}")
    
    print("\n🔍 Step 5: Advanced Retrieval Testing")
    print("-" * 40)
    
    # Create advanced retriever with multiple features
    retriever = AdvancedRAGRetriever(
        vector_store=vector_store,
        k=5,
        score_threshold=0.1,
        use_toxicity_filter=True,
        use_reranking=True,
        diversity_threshold=0.7
    )
    
    # Test with various query types
    test_queries = [
        "What is deep learning?",
        "How to evaluate machine learning models?",
        "What are the ethical concerns in AI?",
        "Explain supervised vs unsupervised learning",
        "What is the role of human oversight in AI?",
        "How do transformer models work?",
        "What are common machine learning algorithms?"
    ]
    
    print("Testing advanced retrieval:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        results = retriever.retrieve(query, k=3)
        
        if results:
            for j, result in enumerate(results, 1):
                metadata = result["metadata"]
                source = metadata.get("source_file", "unknown")
                score = result["score"]
                
                print(f"   {j}. [{source}] (score: {score:.3f})")
                print(f"      {result['content'][:120]}...")
        else:
            print("   No results found")
    
    print("\n🎯 Step 6: Source-Based Filtering")
    print("-" * 40)
    
    # Demonstrate filtering by source
    def search_by_source(query, source_filter=None):
        results = retriever.retrieve(query, k=10)
        
        if source_filter:
            filtered_results = [
                r for r in results 
                if r["metadata"].get("source_file", "").startswith(source_filter)
            ]
            return filtered_results[:3]
        
        return results[:3]
    
    # Search specific sources
    sources_to_test = [
        ("ethics", "ai_ethics"),
        ("learning algorithms", "machine_learning"),
        ("neural networks", "research_paper")
    ]
    
    for query, source_prefix in sources_to_test:
        print(f"\nSearching '{query}' in sources starting with '{source_prefix}':")
        results = search_by_source(query, source_prefix)
        
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source_file", "unknown")
            print(f"  {i}. [{source}] {result['content'][:80]}...")
    
    print("\n📈 Step 7: Performance Analysis")
    print("-" * 40)
    
    # Analyze chunk distribution
    source_stats = {}
    for chunk in merged_chunks:
        source = chunk.get("source_file", "unknown")
        if source not in source_stats:
            source_stats[source] = {"count": 0, "total_size": 0}
        source_stats[source]["count"] += 1
        source_stats[source]["total_size"] += chunk["size"]
    
    print("Chunk distribution by source:")
    for source, stats in source_stats.items():
        avg_size = stats["total_size"] / stats["count"]
        print(f"  {source}: {stats['count']} chunks, avg size: {avg_size:.1f}")
    
    print("\n✅ PDF processing example completed!")
    print(f"Vector store saved to: {store_path}")
    print("You can now use this vector store for retrieval in other applications.")


if __name__ == "__main__":
    main()