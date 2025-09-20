# LocalRAG Quick Start Guide

This guide will help you get started with LocalRAG in just a few minutes.

## Prerequisites

- Python 3.9 or higher
- uv package manager (or pip)
- Internet connection (for downloading models)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hmthanh/LocalRAG.git
   cd LocalRAG
   ```

2. **Install with uv (recommended):**
   ```bash
   uv sync
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

   **Or install with pip:**
   ```bash
   pip install -e .
   ```

## Quick Test

Verify everything is working:
```bash
python examples/test_imports.py
localrag --help
```

## Basic Usage

### 1. Simple Example

Create a file called `my_rag.py`:

```python
from localrag import (
    TextChunker, EmbeddingModel, VectorStore, 
    RAGRetriever, ToxicityFilter
)

# Sample text (in real usage, extract from PDFs)
text = """
Machine learning is a method of data analysis that automates 
analytical model building. It uses algorithms that iteratively 
learn from data, allowing computers to find insights without 
being explicitly programmed.
"""

# 1. Chunk the text
chunker = TextChunker(chunk_size=200, overlap=20)
chunks = chunker.chunk_text(text)

# 2. Create embeddings and vector store
embedding_model = EmbeddingModel()
vector_store = VectorStore(embedding_model)
vector_store.add_chunk_data(chunks)

# 3. Set up retrieval
retriever = RAGRetriever(vector_store, k=3)

# 4. Query the system
results = retriever.retrieve("What is machine learning?")
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content']}")
```

Run it:
```bash
python my_rag.py
```

### 2. PDF Processing

Use the CLI to process PDFs:

```bash
# Process a PDF
localrag process-pdf document.pdf --output-dir ./data

# Query the results
localrag query "What are the main topics?" --vector-store-path ./data/vector_store
```

### 3. With Toxicity Filtering

```python
from localrag import RAGRetriever, ToxicityFilter

# Create retriever with toxicity filtering
toxicity_filter = ToxicityFilter()
retriever = RAGRetriever(
    vector_store, 
    use_toxicity_filter=True,
    toxicity_filter=toxicity_filter
)

# Safe queries only
results = retriever.retrieve("How does AI work?")
```

## Configuration

Create a `config.json` file:

```json
{
  "embedding": {
    "model_name": "all-MiniLM-L6-v2",
    "batch_size": 16
  },
  "chunking": {
    "chunk_size": 256,
    "overlap": 25
  },
  "retrieval": {
    "k": 5,
    "use_reranking": true
  }
}
```

Load it in your code:

```python
from localrag.config import load_config
load_config("config.json")
```

## Advanced Features

### Multi-Query Retrieval
```python
from localrag.retrieval import MultiQueryRetriever

retriever = MultiQueryRetriever(vector_store, query_variations=3)
```

### Contextual Retrieval
```python
from localrag.retrieval import ContextualRetriever

retriever = ContextualRetriever(vector_store, context_window=3)
```

### Advanced Toxicity Filtering
```python
from localrag.toxicity_filter import AdvancedToxicityFilter

filter = AdvancedToxicityFilter(
    models=["martin-ha/toxic-comment-model"],
    ensemble_method="average"
)
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you activated the virtual environment
2. **Network errors**: Ensure internet access for model downloads
3. **Memory issues**: Reduce batch size in configuration
4. **Slow performance**: Use GPU if available or smaller models

### Performance Tips

1. **Use appropriate chunk sizes**: 256-512 tokens work well
2. **Enable batch processing**: Larger batches = better throughput
3. **Choose the right index**: Flat for <10K docs, IVF for larger datasets
4. **Cache embeddings**: Models are cached automatically

## Next Steps

1. Check out the examples in `examples/`
2. Read the full documentation
3. Experiment with different models and configurations
4. Build your own RAG applications!

## Getting Help

- Check the examples directory
- Review configuration options
- Open an issue on GitHub
- Read the full README.md