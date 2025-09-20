#!/usr/bin/env python3
"""Toxicity filtering demonstration for LocalRAG."""

import logging

from localrag.toxicity_filter import ToxicityFilter, SafetyWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate toxicity filtering."""
    print("🛡️ LocalRAG Toxicity Filter Demo")
    print("=" * 40)
    
    # Initialize toxicity filter
    print("🔧 Initializing toxicity filter...")
    toxicity_filter = ToxicityFilter()
    safety_wrapper = SafetyWrapper(toxicity_filter)
    
    if not toxicity_filter.enabled:
        print("❌ Toxicity filter is not enabled")
        print("This might be because detoxify is not installed or there was an error")
        print("Install with: uv add detoxify")
        return
    
    print("✅ Toxicity filter initialized")
    
    # Test queries
    test_queries = [
        "What is machine learning?",  # Safe query
        "How does natural language processing work?",  # Safe query
        "Tell me about data science",  # Safe query
        "This is a test query",  # Safe query
        # Note: We won't include actually toxic examples in code
        # The filter would catch genuinely problematic content
    ]
    
    print("\n💬 Testing queries for toxicity...")
    print("=" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        
        # Get toxicity scores
        scores = toxicity_filter.get_toxicity_scores(query)
        is_toxic = toxicity_filter.is_toxic(query)
        
        print(f"Toxic: {'❌ Yes' if is_toxic else '✅ No'}")
        
        if scores:
            print("Detailed scores:")
            for category, score in scores.items():
                status = "⚠️" if score > toxicity_filter.threshold else "✅"
                print(f"  {category}: {score:.3f} {status}")
        
        # Test safety wrapper
        filtered_query, is_safe, wrapper_scores = safety_wrapper.safe_query(query)
        
        if not is_safe:
            print(f"🔄 Filtered query: {filtered_query}")
        
        print()
    
    # Batch analysis
    print("\n📊 Batch Analysis")
    print("=" * 40)
    
    analysis_results = toxicity_filter.analyze_batch(test_queries)
    
    for i, result in enumerate(analysis_results, 1):
        query_preview = result["text_preview"]
        is_toxic = result["is_toxic"]
        scores = result["scores"]
        
        print(f"Query {i}: {query_preview}")
        print(f"  Toxic: {'❌ Yes' if is_toxic else '✅ No'}")
        
        if scores:
            max_score = max(scores.values()) if scores else 0
            print(f"  Max score: {max_score:.3f}")
    
    # Statistics
    print("\n📈 Statistics")
    print("=" * 40)
    
    stats = toxicity_filter.get_statistics(test_queries)
    
    print(f"Total queries analyzed: {stats['total_texts']}")
    print(f"Toxic queries found: {stats['toxic_count']}")
    print(f"Toxicity percentage: {stats['toxic_percentage']:.1f}%")
    print(f"Threshold used: {stats['threshold']}")
    
    if stats.get('average_scores'):
        print("\nAverage scores by category:")
        for category, avg_score in stats['average_scores'].items():
            print(f"  {category}: {avg_score:.3f}")
    
    if stats.get('category_violations'):
        print("\nViolations by category:")
        for category, count in stats['category_violations'].items():
            print(f"  {category}: {count}")
    
    # Demonstrate response filtering
    print("\n🔄 Response Filtering Demo")
    print("=" * 40)
    
    test_responses = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Here's a helpful explanation of the topic you asked about.",
        "I can provide information on various technical subjects."
    ]
    
    for i, response in enumerate(test_responses, 1):
        print(f"\nResponse {i}: {response}")
        
        filtered_response, is_safe, scores = safety_wrapper.safe_response(response)
        
        print(f"Safe: {'✅ Yes' if is_safe else '❌ No'}")
        
        if not is_safe:
            print(f"Filtered: {filtered_response}")
    
    print("\n✅ Toxicity filter demo completed!")
    print("\nThe toxicity filter helps ensure safe interactions by:")
    print("- Analyzing queries before processing")
    print("- Filtering responses before display")
    print("- Providing detailed toxicity scores")
    print("- Supporting batch analysis")


if __name__ == "__main__":
    main()