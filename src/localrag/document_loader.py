"""Document loading utilities for LocalRAG."""

import logging
from pathlib import Path
from typing import List, Generator, Optional, Union

import pdfplumber
from PyPDF2 import PdfReader
from langchain.schema import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and extract text from various document formats."""
    
    def __init__(self, use_pdfplumber: bool = True):
        """
        Initialize document loader.
        
        Args:
            use_pdfplumber: Whether to use pdfplumber (True) or PyPDF2 (False)
        """
        self.use_pdfplumber = use_pdfplumber
    
    def load_pdf(self, file_path: Union[str, Path]) -> Document:
        """
        Load text from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document object with extracted text and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        
        try:
            if self.use_pdfplumber:
                text = self._extract_with_pdfplumber(file_path)
            else:
                text = self._extract_with_pypdf2(file_path)
            
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "extraction_method": "pdfplumber" if self.use_pdfplumber else "PyPDF2"
            }
            
            return Document(page_content=text, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            # Try fallback method
            if self.use_pdfplumber:
                logger.info("Falling back to PyPDF2")
                try:
                    text = self._extract_with_pypdf2(file_path)
                    metadata = {
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_size": file_path.stat().st_size,
                        "extraction_method": "PyPDF2_fallback"
                    }
                    return Document(page_content=text, metadata=metadata)
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
            raise e
    
    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num} ---\n{page_text}\n")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
        
        return "\n".join(text_parts)
    
    def _extract_with_pypdf2(self, file_path: Path) -> str:
        """Extract text using PyPDF2."""
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num} ---\n{page_text}\n")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
        
        return "\n".join(text_parts)
    
    def load_directory(
        self, 
        directory_path: Union[str, Path],
        max_files: Optional[int] = None,
        recursive: bool = True
    ) -> Generator[Document, None, None]:
        """
        Load all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            max_files: Maximum number of files to process
            recursive: Whether to search subdirectories
            
        Yields:
            Document objects for each PDF file
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find PDF files
        if recursive:
            pdf_files = list(directory_path.rglob("*.pdf"))
        else:
            pdf_files = list(directory_path.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        if max_files:
            pdf_files = pdf_files[:max_files]
            logger.info(f"Processing first {len(pdf_files)} files")
        
        for file_path in pdf_files:
            try:
                document = self.load_pdf(file_path)
                yield document
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
    
    def load_text_file(self, file_path: Union[str, Path]) -> Document:
        """
        Load text from a plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Document object with text content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "extraction_method": "plain_text"
            }
            
            return Document(page_content=text, metadata=metadata)
            
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "extraction_method": "plain_text_latin1"
            }
            
            return Document(page_content=text, metadata=metadata)
    
    def load_mixed_directory(
        self, 
        directory_path: Union[str, Path],
        max_files: Optional[int] = None,
        recursive: bool = True
    ) -> Generator[Document, None, None]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Path to directory
            max_files: Maximum number of files to process
            recursive: Whether to search subdirectories
            
        Yields:
            Document objects for each supported file
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        supported_extensions = {'.pdf', '.txt', '.md'}
        
        # Find supported files
        if recursive:
            files = [f for f in directory_path.rglob("*") 
                    if f.suffix.lower() in supported_extensions]
        else:
            files = [f for f in directory_path.glob("*") 
                    if f.suffix.lower() in supported_extensions]
        
        logger.info(f"Found {len(files)} supported files in {directory_path}")
        
        if max_files:
            files = files[:max_files]
            logger.info(f"Processing first {len(files)} files")
        
        for file_path in files:
            try:
                if file_path.suffix.lower() == '.pdf':
                    document = self.load_pdf(file_path)
                else:
                    document = self.load_text_file(file_path)
                yield document
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue