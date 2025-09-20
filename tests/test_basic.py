"""Basic tests for LocalRAG."""

import pytest
from pathlib import Path
import tempfile
import os

from localrag import (
    config,
    DocumentLoader,
    TextProcessor,
    HuggingFaceEmbeddings,
    FAISSVectorStore,
    ToxicityFilter,
    Translator
)


def test_config():
    """Test configuration loading."""
    assert config.chunk_size > 0
    assert config.chunk_overlap >= 0
    assert config.embedding_model is not None


def test_document_loader():
    """Test document loader initialization."""
    loader = DocumentLoader()
    assert loader is not None
    assert hasattr(loader, 'load_pdf')
    assert hasattr(loader, 'load_text_file')


def test_text_processor():
    """Test text processor initialization."""
    processor = TextProcessor()
    assert processor is not None
    assert processor.chunk_size > 0
    assert processor.chunk_overlap >= 0


def test_text_processor_clean_text():
    """Test text cleaning functionality."""
    processor = TextProcessor()
    
    # Test basic cleaning
    dirty_text = "  This is   a test  \n\n\n  with multiple spaces  "
    clean_text = processor.clean_text(dirty_text)
    
    assert "This is a test" in clean_text
    assert clean_text.strip() == clean_text
    

def test_embeddings_initialization():
    """Test embeddings initialization."""
    try:
        embeddings = HuggingFaceEmbeddings()
        assert embeddings is not None
        assert embeddings.embedding_dimension > 0
    except Exception as e:
        # Skip if model loading fails (common in CI)
        pytest.skip(f"Embeddings test skipped due to: {e}")


def test_vector_store_initialization():
    """Test vector store initialization."""
    try:
        from localrag.embeddings import embedding_manager
        embeddings = embedding_manager.get_model() 
        vector_store = FAISSVectorStore(embedding_function=embeddings)
        assert vector_store is not None
    except Exception as e:
        pytest.skip(f"Vector store test skipped due to: {e}")


def test_toxicity_filter_initialization():
    """Test toxicity filter initialization."""
    toxicity_filter = ToxicityFilter()
    assert toxicity_filter is not None
    # Filter might be disabled if detoxify is not available
    assert hasattr(toxicity_filter, 'enabled')


def test_translator_initialization():
    """Test translator initialization."""
    translator = Translator()
    assert translator is not None
    # Translator might be disabled if credentials are not available
    assert hasattr(translator, 'enabled')


def test_sample_document_processing():
    """Test processing a sample document."""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a sample document for testing. " * 10)
        temp_file = f.name
    
    try:
        # Load document
        loader = DocumentLoader()
        doc = loader.load_text_file(temp_file)
        
        assert doc is not None
        assert len(doc.page_content) > 0
        assert 'source' in doc.metadata
        
        # Process document
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.chunk_document(doc)
        
        assert len(chunks) > 0
        assert all(len(chunk.page_content) <= 120 for chunk in chunks)  # Some overlap allowed
        
    finally:
        # Clean up
        os.unlink(temp_file)


def test_config_from_env():
    """Test configuration from environment variables."""
    # Set some environment variables
    os.environ['CHUNK_SIZE'] = '500'
    os.environ['ENABLE_TOXICITY_FILTER'] = 'false'
    
    try:
        from localrag.config import LocalRAGConfig
        test_config = LocalRAGConfig.from_env()
        
        assert test_config.chunk_size == 500
        assert test_config.enable_toxicity_filter == False
        
    finally:
        # Clean up environment
        os.environ.pop('CHUNK_SIZE', None)
        os.environ.pop('ENABLE_TOXICITY_FILTER', None)


if __name__ == "__main__":
    pytest.main([__file__])