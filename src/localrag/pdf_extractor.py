"""
PDF text extraction functionality using PyPDF.
"""

from pathlib import Path
from typing import List, Optional
import logging

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError("pypdf is required. Install it with: pip install pypdf")

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF files."""
    
    def __init__(self):
        """Initialize PDF extractor."""
        pass
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            reader = PdfReader(str(pdf_path))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                        logger.debug(f"Extracted text from page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Successfully extracted {len(full_text)} characters from {pdf_path}")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise Exception(f"PDF processing failed: {e}")
    
    def extract_text_from_multiple(self, pdf_paths: List[str]) -> List[str]:
        """
        Extract text from multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of extracted texts
        """
        texts = []
        for pdf_path in pdf_paths:
            try:
                text = self.extract_text(pdf_path)
                texts.append(text)
            except Exception as e:
                logger.error(f"Failed to extract from {pdf_path}: {e}")
                texts.append("")  # Add empty string for failed extractions
        
        return texts
    
    def get_pdf_metadata(self, pdf_path: str) -> dict:
        """
        Get metadata from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            reader = PdfReader(str(pdf_path))
            metadata = reader.metadata or {}
            
            return {
                "title": metadata.get("/Title", ""),
                "author": metadata.get("/Author", ""),
                "subject": metadata.get("/Subject", ""),
                "creator": metadata.get("/Creator", ""),
                "producer": metadata.get("/Producer", ""),
                "creation_date": metadata.get("/CreationDate", ""),
                "modification_date": metadata.get("/ModDate", ""),
                "num_pages": len(reader.pages),
                "file_size": pdf_path.stat().st_size,
            }
            
        except Exception as e:
            logger.error(f"Failed to get metadata from {pdf_path}: {e}")
            return {}