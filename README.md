# LocalRAG

Local RAG (Retrieval-Augmented Generation) system with PDF text extraction capabilities.

## Features

- PDF text extraction using multiple methods (pdfplumber and PyPDF2)
- Robust fallback mechanisms for improved reliability
- Command-line interface for easy PDF processing
- PDF metadata extraction
- Modern Python project structure with uv package manager

## Installation

This project uses [uv](https://astral.sh/uv/) for dependency management. Install uv first:

```bash
# Install uv
pip install uv

# Clone the repository
git clone https://github.com/hmthanh/LocalRAG.git
cd LocalRAG

# Install dependencies
uv sync
```

## Usage

### Command Line Interface

Extract text from a PDF file:

```bash
# Using pdfplumber (default)
uv run localrag path/to/your/document.pdf

# Using PyPDF2
uv run localrag path/to/your/document.pdf --method pypdf2

# Get PDF information
uv run localrag path/to/your/document.pdf --info
```

### Python API

```python
from localrag import PDFExtractor, extract_text_from_pdf

# Quick extraction
text = extract_text_from_pdf("document.pdf")
print(text)

# Using the extractor class
extractor = PDFExtractor(preferred_method="pdfplumber")
text = extractor.extract_text("document.pdf")
info = extractor.get_pdf_info("document.pdf")

print(f"Extracted {len(text)} characters from {info['page_count']} pages")
```

## Dependencies

- `pdfplumber>=0.9.0` - Primary PDF extraction library
- `pypdf2>=3.0.0` - Fallback PDF extraction library

## Development

```bash
# Install development dependencies
uv sync --dev

# Run with development environment
uv run localrag --help
```

## License

MIT License - see LICENSE file for details.
