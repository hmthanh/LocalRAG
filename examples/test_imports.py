#!/usr/bin/env python3
"""
Test script to verify all LocalRAG components can be imported successfully.
This test doesn't require network access or model downloads.
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported without errors."""
    
    print("🔍 Testing LocalRAG imports...")
    
    try:
        # Test individual module imports
        print("  ✓ Testing config module...")
        from localrag.config import LocalRAGConfig
        config = LocalRAGConfig()
        assert config.get('embedding.model_name') == 'all-MiniLM-L6-v2'
        
        print("  ✓ Testing PDF extractor...")
        from localrag.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()
        
        print("  ✓ Testing text chunker...")
        from localrag.text_chunker import TextChunker
        chunker = TextChunker()
        
        # Test basic chunking without tiktoken
        sample_text = "This is a test. This is another sentence. And one more for good measure."
        chunks = chunker.chunk_text(sample_text, use_tokens=False)  # Use character-based chunking
        assert len(chunks) > 0
        print(f"    - Created {len(chunks)} chunks from sample text")
        
        print("  ✓ Testing embeddings (class only, no model loading)...")
        from localrag.embeddings import EmbeddingModel
        # Don't initialize - would require network access
        
        print("  ✓ Testing vector store (class only)...")
        from localrag.vector_store import VectorStore
        # Don't initialize - would require embedding model
        
        print("  ✓ Testing retrieval...")
        from localrag.retrieval import RAGRetriever
        # Don't initialize - would require vector store
        
        print("  ✓ Testing toxicity filter (class only)...")
        from localrag.toxicity_filter import ToxicityFilter
        # Don't initialize - would require network access for model
        
        print("  ✓ Testing main package import...")
        import localrag
        print(f"    - LocalRAG version: {localrag.__version__}")
        
        print("\n✅ All imports successful!")
        print("📝 Note: Full functionality requires internet access to download models.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_imports():
    """Test CLI module imports."""
    try:
        print("\n🖥️  Testing CLI imports...")
        from localrag.main import cli
        print("  ✓ CLI module imported successfully")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False


def test_configurations():
    """Test configuration functionality."""
    try:
        print("\n⚙️  Testing configuration system...")
        from localrag.config import LocalRAGConfig
        
        config = LocalRAGConfig()
        
        # Test getting values
        model_name = config.get('embedding.model_name')
        assert model_name == 'all-MiniLM-L6-v2'
        print(f"  ✓ Default embedding model: {model_name}")
        
        # Test setting values
        config.set('embedding.model_name', 'test-model')
        assert config.get('embedding.model_name') == 'test-model'
        print("  ✓ Configuration set/get works")
        
        # Test validation
        is_valid = config.validate_config()
        print(f"  ✓ Configuration validation: {'passed' if is_valid else 'failed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 LocalRAG Component Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_cli_imports,
        test_configurations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LocalRAG is ready to use.")
        print("\n📋 Next steps:")
        print("  1. Set up internet access to download models")
        print("  2. Run: python examples/basic_usage.py")
        print("  3. Try: localrag --help")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)