# LocalRAG Examples

This directory contains example scripts demonstrating various features of the LocalRAG system.

## Available Examples

### 1. `test_imports.py`
**Quick validation script**
- Tests that all components can be imported
- Validates configuration system
- Verifies CLI functionality
- **Run first** to ensure everything is set up correctly

```bash
python test_imports.py
```

### 2. `basic_usage.py`
**Complete basic workflow**
- Text chunking demonstration
- Embedding model setup
- Vector store creation
- Basic retrieval
- Toxicity filtering basics

```bash
python basic_usage.py
```

### 3. `pdf_processing.py`
**Advanced PDF processing**
- Multiple document processing
- Advanced chunking strategies
- Optimized vector store creation
- Source-based filtering
- Performance analysis

```bash
python pdf_processing.py
```

### 4. `advanced_retrieval.py`
**Advanced retrieval features**
- Multi-query retrieval
- Contextual retrieval with conversation history
- Ensemble toxicity filtering
- Custom reranking strategies

```bash
python advanced_retrieval.py
```

### 5. `config_example.json`
**Sample configuration file**
- Shows all configurable options
- Optimized for different use cases
- Load with: `load_config("config_example.json")`

## Running Examples

### Prerequisites
Make sure you have:
1. Installed LocalRAG: `uv sync` or `pip install -e .`
2. Activated the virtual environment: `source .venv/bin/activate`
3. Internet connection (for downloading models)

### Execution Order

1. **Start here**: `python test_imports.py`
2. **Basic features**: `python basic_usage.py`
3. **PDF processing**: `python pdf_processing.py`
4. **Advanced features**: `python advanced_retrieval.py`

## Example Output

### test_imports.py
```
🧪 LocalRAG Component Tests
========================================
🔍 Testing LocalRAG imports...
  ✓ Testing config module...
  ✓ Testing PDF extractor...
  ...
📊 Test Results: 3/3 tests passed
🎉 All tests passed! LocalRAG is ready to use.
```

### basic_usage.py
```
🤖 LocalRAG Basic Usage Example
==================================================
📝 Step 1: Text Chunking
------------------------------
Created 15 chunks from sample text
...
✅ Basic usage example completed!
```

## Customization

Each example can be modified to test your specific use case:

1. **Change models**: Update model names in the scripts
2. **Adjust parameters**: Modify chunk sizes, k values, thresholds
3. **Use your data**: Replace sample text with your own content
4. **Add features**: Combine examples to create custom workflows

## Common Modifications

### Using Different Models

```python
# In any example, change:
embedding_model = EmbeddingModel()

# To:
embedding_model = EmbeddingModel(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
)
```

### Using Your Own PDFs

```python
# Replace the sample content in pdf_processing.py
pdf_path = "your_document.pdf"
extractor = PDFExtractor()
text = extractor.extract_text(pdf_path)
```

### Custom Configuration

```python
# Load your configuration
from localrag.config import load_config
load_config("your_config.json")
```

## Troubleshooting Examples

### Network Issues
If you see connection errors:
1. Check internet connectivity
2. Models will be cached after first download
3. Use offline mode once models are cached

### Memory Issues
If examples run out of memory:
1. Reduce batch sizes in configuration
2. Use smaller models
3. Process fewer documents at once

### Performance Issues
If examples run slowly:
1. Enable GPU if available
2. Use larger batch sizes
3. Consider using IVF or HNSW indices for large datasets

## Creating New Examples

To create your own example:

1. Copy `basic_usage.py` as a template
2. Add the import path fix at the top
3. Modify the main function
4. Test with `python your_example.py`

```python
#!/usr/bin/env python3
"""
Your custom LocalRAG example.
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from localrag import *  # Import what you need

def main():
    """Your custom demonstration."""
    print("🚀 My Custom LocalRAG Example")
    # Your code here

if __name__ == "__main__":
    main()
```

## Getting Help

If you encounter issues with any example:
1. Make sure all prerequisites are met
2. Check the main README.md for setup instructions
3. Look at the error messages - they often point to the solution
4. Try the test_imports.py script first
5. Open an issue on GitHub with the full error message