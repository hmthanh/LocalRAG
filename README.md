# LocalRAG

A local RAG (Retrieval-Augmented Generation) system built with modern Python tools including uv package manager, FAISS vector storage, LangChain, and Hugging Face transformers.

## Features

- 📄 **PDF Text Extraction**: Extract text from PDF documents using PyPDF2 and pdfplumber
- 🔪 **Smart Text Chunking**: Intelligent text chunking with LangChain text splitters
- 🗂️ **FAISS Vector Storage**: High-performance vector storage and similarity search
- 🤖 **LangChain Integration**: Powerful retrieval chains and document processing
- 🤗 **Hugging Face Tools**: Open-weight embedding models, bitsandbytes optimization, and evaluation tools
- 🌐 **Language Support**: Google Cloud Translation integration
- 🛡️ **Toxicity Filter**: Clean prompts before LLM inference using detoxify
- 🖥️ **Multiple Interfaces**: CLI, Gradio, and Streamlit UIs
- ⚡ **Modern Python**: Built with uv package manager for fast dependency management

## Quick Start

### Prerequisites

- Python 3.9+
- uv package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hmthanh/LocalRAG.git
cd LocalRAG
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your API keys if using Google Cloud Translation
```

### Basic Usage

#### 1. Ingest Documents

```bash
# Ingest PDF documents into the vector store
uv run localrag-ingest --input-dir ./documents --vector-store ./data/vector_store
```

#### 2. Query the System

```bash
# Query the RAG system
uv run localrag-query "What is the main topic of the documents?"
```

#### 3. Launch Web UI

```bash
# Launch Streamlit UI
uv run localrag-ui --interface streamlit

# Or launch Gradio UI
uv run localrag-ui --interface gradio
```

## Project Structure

```
LocalRAG/
├── src/localrag/           # Main package
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── document_loader.py # PDF and document loading
│   ├── text_processor.py  # Text chunking and processing
│   ├── embeddings.py      # Embedding model management
│   ├── vector_store.py    # FAISS vector storage
│   ├── retrieval.py       # RAG retrieval chains
│   ├── toxicity_filter.py # Prompt toxicity filtering
│   ├── translator.py      # Language translation
│   └── ui/                # User interfaces
│       ├── __init__.py
│       ├── streamlit_app.py
│       └── gradio_app.py
├── examples/              # Example scripts and notebooks
├── tests/                 # Test suite
├── documents/             # Sample documents
├── data/                  # Data storage (vector stores, etc.)
├── pyproject.toml         # Project configuration
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Configuration

The system can be configured through:

1. **Environment Variables**: Set in `.env` file
2. **Configuration File**: `config.yaml` in the project root
3. **Command Line Arguments**: Override settings per command

### Key Configuration Options

- `EMBEDDING_MODEL`: Hugging Face embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `VECTOR_STORE_PATH`: Path to FAISS vector store
- `CHUNK_SIZE`: Text chunk size for processing
- `CHUNK_OVERLAP`: Overlap between text chunks
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials (for translation)

## Development

### Setting up Development Environment

```bash
# Install development dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

### Running Examples

```bash
# Run the basic RAG example
uv run python examples/basic_rag.py

# Run the toxicity filtering example
uv run python examples/toxicity_demo.py

# Run the translation example
uv run python examples/translation_demo.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## Troubleshooting

### Common Issues

1. **FAISS Installation**: If you encounter issues with FAISS, try installing the GPU version:
   ```bash
   uv add faiss-gpu
   ```

2. **Memory Issues**: For large documents, adjust chunk size and batch processing:
   ```bash
   uv run localrag-ingest --chunk-size 512 --batch-size 32
   ```

3. **Model Downloads**: First run will download embedding models. Ensure you have sufficient disk space.

## Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [FAISS](https://faiss.ai/) for vector similarity search
- [Hugging Face](https://huggingface.co/) for transformer models
- [uv](https://github.com/astral-sh/uv) for fast Python package management