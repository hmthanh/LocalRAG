# LocalRAG 🤖

A comprehensive local RAG (Retrieval-Augmented Generation) system using the uv Python manager. This template provides a complete, ready-to-run implementation of a RAG system with PDF text extraction, intelligent chunking, FAISS vector storage, LangChain retrieval, and built-in toxicity filtering.

## ✨ Features

- **📄 PDF Text Extraction**: Robust PDF processing using PyPDF
- **✂️ Intelligent Text Chunking**: Token-aware chunking with customizable overlap
- **🧠 Advanced Embeddings**: Hugging Face sentence-transformers with multiple model options
- **📊 FAISS Vector Storage**: High-performance similarity search with multiple index types
- **🔍 LangChain Integration**: Compatible retriever interface with advanced features
- **🛡️ Toxicity Filtering**: Built-in content filtering to clean prompts before LLM inference
- **⚡ Performance Optimized**: Efficient batch processing and caching
- **🔧 Highly Configurable**: Comprehensive configuration system
- **📱 CLI Interface**: Easy-to-use command-line tools

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- uv package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hmthanh/LocalRAG.git
   cd LocalRAG
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

### Basic Usage

#### 1. Process a PDF and create vector store:
```bash
localrag process-pdf path/to/your/document.pdf --output-dir ./data
```

#### 2. Query the system:
```bash
localrag query "What is machine learning?" --vector-store-path ./data/vector_store
```

#### 3. Use with toxicity filtering:
```bash
localrag query "Explain neural networks" --filter-toxicity
```

## 📖 Usage Examples

### Python API

```python
from localrag import (
    PDFExtractor, TextChunker, EmbeddingModel, 
    VectorStore, RAGRetriever, ToxicityFilter
)

# Extract text from PDF
extractor = PDFExtractor()
text = extractor.extract_text("document.pdf")

# Chunk the text
chunker = TextChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk_text(text)

# Create embeddings and vector store
embedding_model = EmbeddingModel()
vector_store = VectorStore(embedding_model)
vector_store.add_chunk_data(chunks)

# Set up retrieval with toxicity filtering
toxicity_filter = ToxicityFilter()
retriever = RAGRetriever(
    vector_store, 
    use_toxicity_filter=True,
    toxicity_filter=toxicity_filter
)

# Query the system
results = retriever.retrieve("What are the main concepts?", k=5)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
```

### Advanced Features

#### Multi-Query Retrieval
```python
from localrag.retrieval import MultiQueryRetriever

retriever = MultiQueryRetriever(
    vector_store=vector_store,
    query_variations=3,
    k=5
)
```

#### Contextual Retrieval
```python
from localrag.retrieval import ContextualRetriever

retriever = ContextualRetriever(
    vector_store=vector_store,
    context_window=3  # Remember last 3 queries
)
```

#### Advanced Toxicity Filtering
```python
from localrag.toxicity_filter import AdvancedToxicityFilter

filter = AdvancedToxicityFilter(
    models=["martin-ha/toxic-comment-model", "unitary/toxic-bert"],
    ensemble_method="average"
)
```

## 🔧 Configuration

Create a configuration file to customize behavior:

```json
{
  "embedding": {
    "model_name": "all-MiniLM-L6-v2",
    "device": "cpu",
    "normalize_embeddings": true,
    "batch_size": 32
  },
  "chunking": {
    "chunk_size": 512,
    "overlap": 50,
    "min_chunk_size": 50
  },
  "vector_store": {
    "index_type": "flat",
    "metric_type": "cosine"
  },
  "retrieval": {
    "k": 5,
    "use_reranking": true,
    "diversity_threshold": 0.8
  },
  "toxicity_filter": {
    "enabled": true,
    "threshold": 0.7
  }
}
```

Load configuration:
```python
from localrag.config import load_config
load_config("path/to/config.json")
```

## 📁 Project Structure

```
LocalRAG/
├── src/localrag/           # Main package
│   ├── __init__.py         # Package initialization
│   ├── main.py            # CLI interface
│   ├── config.py          # Configuration management
│   ├── pdf_extractor.py   # PDF text extraction
│   ├── text_chunker.py    # Text chunking
│   ├── embeddings.py      # Embedding models
│   ├── vector_store.py    # FAISS vector storage
│   ├── retrieval.py       # LangChain retrieval
│   └── toxicity_filter.py # Toxicity filtering
├── examples/              # Usage examples
│   ├── basic_usage.py     # Basic example
│   └── pdf_processing.py  # Advanced PDF processing
├── tests/                 # Test suite
├── docs/                  # Documentation
├── data/                  # Data directory (gitignored)
├── pyproject.toml         # Project configuration
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔬 Available Models

### Embedding Models
- **Default**: `all-MiniLM-L6-v2` (fast, good quality)
- **Multilingual**: `paraphrase-multilingual-MiniLM-L12-v2`
- **High Quality**: `all-mpnet-base-v2`
- **Code**: `microsoft/codebert-base`

### Toxicity Detection Models
- **Default**: `martin-ha/toxic-comment-model`
- **Alternative**: `unitary/toxic-bert`
- **Ensemble**: Multiple models with voting/averaging

## 🛠️ Advanced Configuration

### Index Types
- **Flat** (`flat`): Exact search, good for small datasets
- **IVF** (`ivf`): Inverted file index, faster for large datasets  
- **HNSW** (`hnsw`): Hierarchical navigable small world, best for very large datasets

### Distance Metrics
- **Cosine** (`cosine`): Good for normalized embeddings
- **Euclidean** (`euclidean`): L2 distance
- **Inner Product** (`inner_product`): Dot product similarity

## 🧪 Running Examples

1. **Basic usage example:**
   ```bash
   cd examples
   python basic_usage.py
   ```

2. **PDF processing example:**
   ```bash
   cd examples  
   python pdf_processing.py
   ```

## 📊 Performance Tips

1. **Use appropriate chunk sizes**: 256-512 tokens work well for most use cases
2. **Enable batch processing**: Use larger batch sizes for better throughput
3. **Choose the right index**: Flat for <10K docs, IVF for 10K-1M docs, HNSW for >1M docs
4. **Consider GPU**: Use CUDA-enabled models for faster inference
5. **Cache embeddings**: Enable caching to avoid recomputing embeddings

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformers and sentence-transformers
- [Facebook AI Research](https://github.com/facebookresearch/faiss) for FAISS
- [LangChain](https://github.com/langchain-ai/langchain) for the retrieval framework
- [PyPDF](https://github.com/py-pdf/pypdf) for PDF processing

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the examples in the `examples/` directory
2. Review the configuration options in `src/localrag/config.py`
3. Open an issue on GitHub with detailed information

---

**Ready to build your local RAG system? Start with the quick start guide above! 🚀**
