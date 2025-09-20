"""LocalRAG: Local Retrieval-Augmented Generation system."""

from .pdf_extractor import PDFExtractor, extract_text_from_pdf

__version__ = "0.1.0"
__all__ = ["PDFExtractor", "extract_text_from_pdf"]


def main() -> None:
    """Main entry point for the LocalRAG CLI."""
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="LocalRAG PDF text extraction")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--method", 
        choices=["pdfplumber", "pypdf2"], 
        default="pdfplumber",
        help="PDF extraction method"
    )
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Show PDF info instead of extracting text"
    )
    
    args = parser.parse_args()
    
    try:
        pdf_path = Path(args.pdf_path)
        extractor = PDFExtractor(preferred_method=args.method)
        
        if args.info:
            info = extractor.get_pdf_info(pdf_path)
            print(f"PDF Info for: {info['file_path']}")
            print(f"File size: {info['file_size']} bytes")
            print(f"Page count: {info['page_count']}")
            if info['title']:
                print(f"Title: {info['title']}")
            if info['author']:
                print(f"Author: {info['author']}")
            if info['creator']:
                print(f"Creator: {info['creator']}")
        else:
            text = extractor.extract_text(pdf_path)
            print(text)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
