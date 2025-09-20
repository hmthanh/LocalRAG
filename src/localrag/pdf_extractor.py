"""PDF text extraction module for LocalRAG.

This module provides functionality to extract text from PDF files using multiple approaches
for robustness and flexibility.
"""

import logging
from pathlib import Path
from typing import Optional, Union

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF files using available PDF processing libraries."""
    
    def __init__(self, preferred_method: str = "pdfplumber"):
        """Initialize the PDF extractor.
        
        Args:
            preferred_method: Preferred extraction method ("pdfplumber" or "pypdf2")
        """
        self.preferred_method = preferred_method
        
        if not HAS_PYPDF2 and not HAS_PDFPLUMBER:
            raise ImportError(
                "Neither PyPDF2 nor pdfplumber is available. "
                "Please install at least one: pip install pypdf2 pdfplumber"
            )
    
    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the file is not a PDF or cannot be processed
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        # Try preferred method first
        if self.preferred_method == "pdfplumber" and HAS_PDFPLUMBER:
            try:
                return self._extract_with_pdfplumber(pdf_path)
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
                if HAS_PYPDF2:
                    logger.info("Falling back to PyPDF2")
                    return self._extract_with_pypdf2(pdf_path)
                raise
        
        elif self.preferred_method == "pypdf2" and HAS_PYPDF2:
            try:
                return self._extract_with_pypdf2(pdf_path)
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
                if HAS_PDFPLUMBER:
                    logger.info("Falling back to pdfplumber")
                    return self._extract_with_pdfplumber(pdf_path)
                raise
        
        # Fallback to any available method
        if HAS_PDFPLUMBER:
            return self._extract_with_pdfplumber(pdf_path)
        elif HAS_PYPDF2:
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise RuntimeError("No PDF processing library available")
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber library."""
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
        
        return "\n\n".join(text_content)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 library."""
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
        
        return "\n\n".join(text_content)
    
    def get_pdf_info(self, pdf_path: Union[str, Path]) -> dict:
        """Get basic information about a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            dict: PDF metadata including page count, title, etc.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        info = {
            "file_path": str(pdf_path),
            "file_size": pdf_path.stat().st_size,
            "page_count": 0,
            "title": None,
            "author": None,
            "creator": None,
        }
        
        try:
            if HAS_PYPDF2:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    info["page_count"] = len(pdf_reader.pages)
                    
                    if pdf_reader.metadata:
                        info["title"] = pdf_reader.metadata.get("/Title")
                        info["author"] = pdf_reader.metadata.get("/Author")
                        info["creator"] = pdf_reader.metadata.get("/Creator")
            
            elif HAS_PDFPLUMBER:
                with pdfplumber.open(pdf_path) as pdf:
                    info["page_count"] = len(pdf.pages)
                    if hasattr(pdf, 'metadata') and pdf.metadata:
                        info["title"] = pdf.metadata.get("Title")
                        info["author"] = pdf.metadata.get("Author")
                        info["creator"] = pdf.metadata.get("Creator")
        
        except Exception as e:
            logger.warning(f"Failed to extract PDF info: {e}")
        
        return info


def extract_text_from_pdf(pdf_path: Union[str, Path], 
                         method: str = "pdfplumber") -> str:
    """Convenience function to extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        method: Extraction method ("pdfplumber" or "pypdf2")
        
    Returns:
        str: Extracted text from the PDF
    """
    extractor = PDFExtractor(preferred_method=method)
    return extractor.extract_text(pdf_path)