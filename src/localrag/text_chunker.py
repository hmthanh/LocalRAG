"""
Text chunking functionality for processing large documents.
"""

from typing import List, Optional
import re
import logging

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)


class TextChunker:
    """Split text into manageable chunks for processing."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separators: Optional[List[str]] = None,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk (in tokens or characters)
            overlap: Number of tokens/characters to overlap between chunks
            separators: List of separators to use for splitting (in order of preference)
            encoding_name: Tiktoken encoding to use for token counting
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation endings
            "? ",    # Question endings
            "; ",    # Semicolon
            ", ",    # Comma
            " ",     # Spaces
            ""       # Character level
        ]
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if tiktoken:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to approximate word count
            return len(text.split())
    
    def chunk_text(self, text: str, use_tokens: bool = True) -> List[dict]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            use_tokens: Whether to use token-based chunking (vs character-based)
            
        Returns:
            List of dictionaries containing chunk data
        """
        if not text.strip():
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        chunks = []
        start_idx = 0
        chunk_id = 0
        
        count_func = self.count_tokens if use_tokens else len
        
        while start_idx < len(text):
            # Find the end of current chunk
            end_idx = self._find_chunk_end(text, start_idx, count_func)
            
            if end_idx <= start_idx:
                # Safety check to avoid infinite loops
                end_idx = min(start_idx + self.chunk_size, len(text))
            
            chunk_text = text[start_idx:end_idx].strip()
            
            if chunk_text:
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text,
                    "start_char": start_idx,
                    "end_char": end_idx,
                    "size": count_func(chunk_text),
                })
                chunk_id += 1
            
            # Calculate next start position with overlap
            if use_tokens:
                overlap_tokens = min(self.overlap, count_func(chunk_text))
                # Approximate character position for overlap
                if overlap_tokens > 0:
                    overlap_ratio = overlap_tokens / count_func(chunk_text)
                    overlap_chars = int((end_idx - start_idx) * overlap_ratio)
                    start_idx = max(start_idx + 1, end_idx - overlap_chars)
                else:
                    start_idx = end_idx
            else:
                start_idx = max(start_idx + 1, end_idx - self.overlap)
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _find_chunk_end(self, text: str, start: int, count_func) -> int:
        """
        Find the optimal end position for a chunk.
        
        Args:
            text: Full text
            start: Start position
            count_func: Function to count tokens/characters
            
        Returns:
            End position for chunk
        """
        if start >= len(text):
            return len(text)
        
        # Initial estimate
        max_end = min(start + self.chunk_size * 5, len(text))  # Generous estimate
        
        # Binary search for optimal chunk size
        low, high = start + 1, max_end
        best_end = high
        
        while low <= high:
            mid = (low + high) // 2
            chunk_text = text[start:mid]
            size = count_func(chunk_text)
            
            if size <= self.chunk_size:
                best_end = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Try to find a good breaking point near the optimal size
        optimal_end = best_end
        for separator in self.separators:
            if not separator:
                continue
                
            # Look for separator within reasonable distance
            search_start = max(start, optimal_end - 100)
            search_end = min(len(text), optimal_end + 100)
            
            last_sep = text.rfind(separator, search_start, search_end)
            if last_sep > start:
                return last_sep + len(separator)
        
        return optimal_end
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for better chunking.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'\x0c', ' ', text)  # Form feed
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)  # Control chars
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def merge_small_chunks(self, chunks: List[dict], min_size: int = 100) -> List[dict]:
        """
        Merge chunks that are too small.
        
        Args:
            chunks: List of chunk dictionaries
            min_size: Minimum chunk size
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
        
        merged = []
        current_chunk = chunks[0].copy()
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            if current_chunk["size"] < min_size:
                # Merge with next chunk
                current_chunk["content"] += " " + next_chunk["content"]
                current_chunk["end_char"] = next_chunk["end_char"]
                current_chunk["size"] = len(current_chunk["content"])
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk.copy()
        
        # Add the last chunk
        merged.append(current_chunk)
        
        # Update chunk IDs
        for i, chunk in enumerate(merged):
            chunk["id"] = i
        
        logger.info(f"Merged {len(chunks)} chunks into {len(merged)} chunks")
        return merged