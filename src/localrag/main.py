"""
Main CLI interface for LocalRAG system.
"""

import click
import os
from pathlib import Path

from .pdf_extractor import PDFExtractor
from .text_chunker import TextChunker
from .vector_store import VectorStore
from .retrieval import RAGRetriever
from .toxicity_filter import ToxicityFilter
from .embeddings import EmbeddingModel


@click.group()
def cli():
    """LocalRAG: A local RAG system with PDF processing and toxicity filtering."""
    pass


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./data', help='Output directory for processed files')
@click.option('--chunk-size', default=512, help='Size of text chunks')
@click.option('--chunk-overlap', default=50, help='Overlap between chunks')
def process_pdf(pdf_path, output_dir, chunk_size, chunk_overlap):
    """Process a PDF file and create a vector store."""
    click.echo(f"Processing PDF: {pdf_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text from PDF
    extractor = PDFExtractor()
    text = extractor.extract_text(pdf_path)
    click.echo(f"Extracted {len(text)} characters from PDF")
    
    # Chunk the text
    chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    chunks = chunker.chunk_text(text)
    click.echo(f"Created {len(chunks)} chunks")
    
    # Create embeddings and vector store
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model=embedding_model)
    vector_store.add_texts(chunks)
    
    # Save vector store
    vector_store_path = Path(output_dir) / "vector_store"
    vector_store.save(str(vector_store_path))
    click.echo(f"Vector store saved to: {vector_store_path}")


@cli.command()
@click.argument('query')
@click.option('--vector-store-path', '-v', default='./data/vector_store', 
              help='Path to vector store')
@click.option('--k', default=3, help='Number of documents to retrieve')
@click.option('--filter-toxicity', is_flag=True, help='Filter toxic content from query')
def query(query, vector_store_path, k, filter_toxicity):
    """Query the RAG system."""
    if filter_toxicity:
        toxicity_filter = ToxicityFilter()
        if toxicity_filter.is_toxic(query):
            click.echo("Warning: Toxic content detected in query. Please rephrase.")
            return
        query = toxicity_filter.clean_text(query)
    
    # Load vector store and query
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model=embedding_model)
    vector_store.load(vector_store_path)
    
    retriever = RAGRetriever(vector_store)
    results = retriever.retrieve(query, k=k)
    
    click.echo(f"Found {len(results)} relevant documents:")
    for i, result in enumerate(results, 1):
        click.echo(f"\n--- Document {i} ---")
        click.echo(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
        click.echo(f"Similarity: {result['score']:.4f}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()