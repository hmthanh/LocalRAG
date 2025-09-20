#!/usr/bin/env python3
"""
Advanced retrieval examples for LocalRAG system.

This script demonstrates:
1. Multi-query retrieval
2. Contextual retrieval with conversation history
3. Ensemble toxicity filtering
4. Custom reranking strategies
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from localrag import EmbeddingModel, VectorStore
from localrag.retrieval import MultiQueryRetriever, ContextualRetriever
from localrag.toxicity_filter import AdvancedToxicityFilter


def create_sample_knowledge_base():
    """Create a sample knowledge base for testing."""
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Data science combines statistics, programming, and domain knowledge to extract insights.",
        "Artificial intelligence aims to create machines that can perform human-like tasks.",
        "Neural networks are inspired by the structure and function of the human brain.",
        "Supervised learning uses labeled data to train predictive models.",
        "Unsupervised learning finds patterns in data without labeled examples.",
        "Reinforcement learning learns through interaction with an environment.",
        "Feature engineering is the process of selecting and transforming input variables.",
        "Cross-validation is a technique to assess how well a model generalizes.",
        "Overfitting occurs when a model performs well on training data but poorly on new data.",
        "Regularization techniques help prevent overfitting in machine learning models."
    ]
    
    return documents


def demo_multi_query_retrieval():
    """Demonstrate multi-query retrieval capabilities."""
    print("🔍 Multi-Query Retrieval Demo")
    print("-" * 40)
    
    # Set up knowledge base
    documents = create_sample_knowledge_base()
    
    # Create vector store
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    vector_store.add_texts(documents)
    
    # Create multi-query retriever
    retriever = MultiQueryRetriever(
        vector_store=vector_store,
        k=3,
        query_variations=4,
        use_reranking=True
    )
    
    test_queries = [
        "machine learning algorithms",
        "neural network training",
        "prevent model overfitting"
    ]
    
    for query in test_queries:
        print(f"\nOriginal query: '{query}'")
        results = retriever.retrieve(query)
        
        print("Results:")
        for i, result in enumerate(results, 1):
            score = result['score']
            content = result['content']
            variation = result['metadata'].get('query_variation', 'original')
            print(f"  {i}. [{variation[:20]}...] (score: {score:.3f})")
            print(f"      {content[:80]}...")


def demo_contextual_retrieval():
    """Demonstrate contextual retrieval with conversation history."""
    print("\n💬 Contextual Retrieval Demo")
    print("-" * 40)
    
    # Set up knowledge base
    documents = create_sample_knowledge_base()
    
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    vector_store.add_texts(documents)
    
    # Create contextual retriever
    retriever = ContextualRetriever(
        vector_store=vector_store,
        k=2,
        context_window=3,
        diversity_threshold=0.6
    )
    
    # Simulate a conversation
    conversation = [
        "What is machine learning?",
        "How does it relate to AI?",
        "What about deep learning?",
        "Can you explain neural networks?",
        "How do you prevent overfitting?"
    ]
    
    print("Simulating conversation with context:")
    for i, query in enumerate(conversation, 1):
        print(f"\nTurn {i}: '{query}'")
        results = retriever.retrieve(query)
        
        print("Contextual results:")
        for j, result in enumerate(results, 1):
            print(f"  {j}. {result['content'][:70]}... (score: {result['score']:.3f})")
    
    # Show conversation history
    print(f"\nConversation history: {retriever.query_history}")


def demo_advanced_toxicity_filtering():
    """Demonstrate advanced toxicity filtering with ensemble methods."""
    print("\n🛡️ Advanced Toxicity Filtering Demo")
    print("-" * 40)
    
    # Create advanced toxicity filter (note: this might be slow on first run)
    print("Loading toxicity detection models...")
    try:
        toxicity_filter = AdvancedToxicityFilter(
            models=["martin-ha/toxic-comment-model"],  # Using single model for speed
            threshold=0.6,
            ensemble_method="average"
        )
        print("✅ Loaded toxicity filter")
    except Exception as e:
        print(f"⚠️ Could not load advanced filter, using simple filter: {e}")
        from localrag import ToxicityFilter
        toxicity_filter = ToxicityFilter(threshold=0.6)
    
    # Test various types of content
    test_texts = [
        "How do I build a machine learning model?",
        "I hate this stupid technology",  # Potentially toxic
        "What are the benefits of AI in healthcare?",
        "Ways to attack network security",  # Potentially problematic
        "Explain neural network architectures",
        "This is offensive and inappropriate content"  # Potentially toxic
    ]
    
    print("\nTesting toxicity detection:")
    for text in test_texts:
        is_toxic = toxicity_filter.is_toxic(text)
        score = toxicity_filter.get_toxicity_score(text)
        cleaned = toxicity_filter.clean_text(text)
        
        status = "🚫 TOXIC" if is_toxic else "✅ SAFE"
        print(f"\n{status} (score: {score:.3f})")
        print(f"  Original: '{text}'")
        if is_toxic and cleaned != text:
            print(f"  Cleaned:  '{cleaned}'")


def demo_custom_reranking():
    """Demonstrate custom reranking strategies."""
    print("\n📊 Custom Reranking Demo")
    print("-" * 40)
    
    documents = create_sample_knowledge_base()
    
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    
    # Add documents with custom metadata
    metadatas = [
        {"topic": "programming", "difficulty": "beginner"},
        {"topic": "ml", "difficulty": "intermediate"},
        {"topic": "ml", "difficulty": "advanced"},
        {"topic": "nlp", "difficulty": "intermediate"},
        {"topic": "cv", "difficulty": "intermediate"},
        {"topic": "data_science", "difficulty": "beginner"},
        {"topic": "ai", "difficulty": "beginner"},
        {"topic": "ml", "difficulty": "advanced"},
        {"topic": "ml", "difficulty": "beginner"},
        {"topic": "ml", "difficulty": "intermediate"},
        {"topic": "ml", "difficulty": "advanced"},
        {"topic": "ml", "difficulty": "intermediate"},
        {"topic": "ml", "difficulty": "intermediate"},
        {"topic": "ml", "difficulty": "advanced"},
        {"topic": "ml", "difficulty": "intermediate"}
    ]
    
    vector_store.add_texts(documents, metadatas)
    
    # Custom reranking based on topic preference
    def topic_rerank(results, preferred_topic="ml"):
        """Rerank results to prefer specific topics."""
        def score_boost(result):
            base_score = result['score']
            topic = result['metadata'].get('topic', '')
            difficulty = result['metadata'].get('difficulty', '')
            
            # Boost for preferred topic
            topic_boost = 0.1 if topic == preferred_topic else 0.0
            
            # Slight boost for intermediate difficulty
            difficulty_boost = 0.05 if difficulty == "intermediate" else 0.0
            
            return base_score + topic_boost + difficulty_boost
        
        results.sort(key=score_boost, reverse=True)
        return results
    
    # Test query
    query = "learning algorithms"
    results = vector_store.search(query, k=5)
    
    print(f"Query: '{query}'")
    print("\nOriginal ranking:")
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        print(f"  {i}. {result['content'][:50]}... "
              f"(topic: {meta.get('topic', 'N/A')}, "
              f"difficulty: {meta.get('difficulty', 'N/A')}, "
              f"score: {result['score']:.3f})")
    
    # Apply custom reranking
    reranked_results = topic_rerank(results.copy(), preferred_topic="ml")
    
    print("\nAfter custom reranking (ML preference):")
    for i, result in enumerate(reranked_results, 1):
        meta = result['metadata']
        print(f"  {i}. {result['content'][:50]}... "
              f"(topic: {meta.get('topic', 'N/A')}, "
              f"difficulty: {meta.get('difficulty', 'N/A')}, "
              f"score: {result['score']:.3f})")


def main():
    """Run all advanced retrieval demonstrations."""
    print("🚀 LocalRAG Advanced Retrieval Examples")
    print("=" * 50)
    
    try:
        demo_multi_query_retrieval()
        demo_contextual_retrieval()
        demo_advanced_toxicity_filtering()
        demo_custom_reranking()
        
        print("\n✅ All advanced retrieval demos completed!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()