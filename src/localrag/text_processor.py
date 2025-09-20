"""Text processing and chunking utilities for LocalRAG."""

import logging
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

from .config import config

logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and chunk text documents for RAG system."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        splitter_type: str = "recursive"
    ):
        """
        Initialize text processor.
        
        Args:
            chunk_size: Size of text chunks (defaults to config value)
            chunk_overlap: Overlap between chunks (defaults to config value)
            splitter_type: Type of text splitter ("recursive", "character", "token")
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.splitter_type = splitter_type
        
        self.text_splitter = self._create_text_splitter()
    
    def _create_text_splitter(self):
        """Create appropriate text splitter based on configuration."""
        if self.splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif self.splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
        elif self.splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown splitter type: {self.splitter_type}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace from each line
            cleaned_line = line.strip()
            
            # Skip empty lines and lines with only special characters
            if cleaned_line and not cleaned_line.replace('-', '').replace('=', '').strip() == '':
                cleaned_lines.append(cleaned_line)
        
        # Join lines and normalize whitespace
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Replace multiple consecutive newlines with double newline
        import re
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        # Replace multiple spaces with single space
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocess a document by cleaning its text content.
        
        Args:
            document: Input document
            
        Returns:
            Document with cleaned text
        """
        cleaned_content = self.clean_text(document.page_content)
        
        # Update metadata to indicate preprocessing
        metadata = document.metadata.copy()
        metadata.update({
            "preprocessed": True,
            "original_length": len(document.page_content),
            "cleaned_length": len(cleaned_content)
        })
        
        return Document(page_content=cleaned_content, metadata=metadata)
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks.
        
        Args:
            document: Input document to chunk
            
        Returns:
            List of document chunks
        """
        try:
            # Preprocess the document first
            preprocessed_doc = self.preprocess_document(document)
            
            # Split the document
            chunks = self.text_splitter.split_documents([preprocessed_doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_count": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    "chunk_method": self.splitter_type
                })
            
            logger.debug(f"Split document into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            # Return original document as single chunk
            document.metadata.update({
                "chunk_id": 0,
                "chunk_count": 1,
                "chunk_size": len(document.page_content),
                "chunk_method": "failed_fallback"
            })
            return [document]
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of input documents
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            try:
                chunks = self.chunk_document(document)
                
                # Add document-level metadata to chunks
                for chunk in chunks:
                    chunk.metadata.update({
                        "document_id": doc_idx,
                        "source_document": document.metadata.get("source", "unknown")
                    })
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing document {doc_idx}: {e}")
                continue
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        Get statistics about document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "unique_sources": len(set(chunk.metadata.get("source", "unknown") for chunk in chunks))
        }
        
        return stats
    
    def filter_chunks_by_size(
        self, 
        chunks: List[Document], 
        min_size: int = 50, 
        max_size: Optional[int] = None
    ) -> List[Document]:
        """
        Filter chunks by size constraints.
        
        Args:
            chunks: List of document chunks
            min_size: Minimum chunk size in characters
            max_size: Maximum chunk size in characters
            
        Returns:
            Filtered list of chunks
        """
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_size = len(chunk.page_content)
            
            if chunk_size < min_size:
                logger.debug(f"Skipping chunk (too small): {chunk_size} < {min_size}")
                continue
            
            if max_size and chunk_size > max_size:
                logger.debug(f"Skipping chunk (too large): {chunk_size} > {max_size}")
                continue
            
            filtered_chunks.append(chunk)
        
        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)}")
        return filtered_chunks