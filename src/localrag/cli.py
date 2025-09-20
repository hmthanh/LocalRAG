"""Command-line interface for LocalRAG."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.logging import RichHandler

from . import (
    config,
    DocumentLoader,
    TextProcessor,
    create_vector_store,
    create_retriever,
    global_toxicity_filter
)

# Set up rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """LocalRAG: Local RAG system with FAISS and LangChain."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing documents to ingest"
)
@click.option(
    "--vector-store",
    "-v",
    type=click.Path(path_type=Path),
    default=config.vector_store_path,
    help="Path to vector store directory"
)
@click.option(
    "--max-files",
    "-m",
    type=int,
    help="Maximum number of files to process"
)
@click.option(
    "--chunk-size",
    "-c",
    type=int,
    default=config.chunk_size,
    help="Text chunk size"
)
@click.option(
    "--chunk-overlap",
    "-o",
    type=int,
    default=config.chunk_overlap,
    help="Text chunk overlap"
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=config.batch_size,
    help="Batch size for processing"
)
def ingest(
    input_dir: Path,
    vector_store: Path,
    max_files: Optional[int],
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int
) -> None:
    """Ingest documents into the vector store."""
    console.print(f"[bold blue]LocalRAG Document Ingestion[/bold blue]")
    console.print(f"Input directory: {input_dir}")
    console.print(f"Vector store: {vector_store}")
    console.print(f"Max files: {max_files or 'unlimited'}")
    console.print(f"Chunk size: {chunk_size}")
    console.print(f"Chunk overlap: {chunk_overlap}")
    
    try:
        # Initialize components
        document_loader = DocumentLoader()
        text_processor = TextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Load documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            load_task = progress.add_task("Loading documents...", total=None)
            
            documents = list(document_loader.load_mixed_directory(
                input_dir,
                max_files=max_files,
                recursive=True
            ))
            
            progress.update(load_task, description=f"Loaded {len(documents)} documents")
        
        if not documents:
            console.print("[red]No documents found to process[/red]")
            return
        
        # Process documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            process_task = progress.add_task("Processing documents...", total=None)
            
            chunks = text_processor.chunk_documents(documents)
            
            progress.update(process_task, description=f"Created {len(chunks)} chunks")
        
        # Get statistics
        stats = text_processor.get_chunk_statistics(chunks)
        
        console.print("\n[bold green]Processing Statistics:[/bold green]")
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")
        
        stats_table.add_row("Total Documents", str(len(documents)))
        stats_table.add_row("Total Chunks", str(stats["total_chunks"]))
        stats_table.add_row("Total Characters", f"{stats['total_characters']:,}")
        stats_table.add_row("Average Chunk Size", f"{stats['average_chunk_size']:.1f}")
        stats_table.add_row("Min Chunk Size", str(stats["min_chunk_size"]))
        stats_table.add_row("Max Chunk Size", str(stats["max_chunk_size"]))
        stats_table.add_row("Unique Sources", str(stats["unique_sources"]))
        
        console.print(stats_table)
        
        # Create/update vector store
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            vector_task = progress.add_task("Creating vector store...", total=None)
            
            vector_store_obj = create_vector_store(
                documents=chunks,
                persist_directory=str(vector_store)
            )
            
            progress.update(vector_task, description="Vector store created")
        
        # Final statistics
        final_stats = vector_store_obj.get_stats()
        console.print(f"\n[bold green]✅ Successfully ingested {final_stats['total_documents']} chunks[/bold green]")
        console.print(f"Vector store saved to: {vector_store}")
        
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        logger.exception("Ingestion failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--vector-store",
    "-v",
    type=click.Path(path_type=Path),
    default=config.vector_store_path,
    help="Path to vector store directory"
)
@click.option(
    "--query",
    "-q",
    type=str,
    help="Query to search for (if not provided, interactive mode)"
)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=4,
    help="Number of results to return"
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Enable interactive query mode"
)
@click.option(
    "--enable-safety",
    is_flag=True,
    default=True,
    help="Enable toxicity filtering"
)
def query(
    vector_store: Path,
    query: Optional[str],
    top_k: int,
    interactive: bool,
    enable_safety: bool
) -> None:
    """Query the RAG system."""
    console.print("[bold blue]LocalRAG Query Interface[/bold blue]")
    
    try:
        # Initialize retriever
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            init_task = progress.add_task("Initializing retriever...", total=None)
            
            retriever = create_retriever(vector_store_path=str(vector_store))
            stats = retriever.get_retriever_stats()
            
            progress.update(init_task, description="Retriever initialized")
        
        # Display system info
        console.print(f"\nVector store: {vector_store}")
        console.print(f"Total documents: {stats['vector_store_stats']['total_documents']}")
        console.print(f"Safety filtering: {'enabled' if enable_safety else 'disabled'}")
        console.print(f"Top-K results: {top_k}")
        
        def process_query(query_text: str) -> None:
            """Process a single query."""
            if not query_text.strip():
                console.print("[yellow]Empty query provided[/yellow]")
                return
            
            console.print(f"\n[bold cyan]Query:[/bold cyan] {query_text}")
            
            # Execute query
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                search_task = progress.add_task("Searching...", total=None)
                
                result = retriever.search_and_respond(
                    query=query_text,
                    k=top_k,
                    enable_safety=enable_safety
                )
                
                progress.update(search_task, description="Search completed")
            
            # Display results
            if result["documents"]:
                console.print(f"\n[bold green]Found {len(result['documents'])} relevant documents:[/bold green]")
                
                for i, doc in enumerate(result["documents"], 1):
                    console.print(f"\n[bold]Document {i}:[/bold]")
                    source = doc.metadata.get("source", "Unknown")
                    console.print(f"[dim]Source: {source}[/dim]")
                    
                    # Show similarity score if available
                    if "similarity_score" in doc.metadata:
                        score = doc.metadata["similarity_score"]
                        console.print(f"[dim]Similarity: {score:.3f}[/dim]")
                    
                    # Show content preview
                    content = doc.page_content[:300]
                    if len(doc.page_content) > 300:
                        content += "..."
                    console.print(f"[white]{content}[/white]")
                
                # Show response
                console.print(f"\n[bold magenta]Response:[/bold magenta]")
                console.print(result["response"])
                
                # Show metadata if safety was applied
                if enable_safety:
                    retrieval_meta = result["retrieval_metadata"]
                    if not retrieval_meta.get("query_safe", True):
                        console.print("\n[yellow]⚠️  Query was filtered for safety[/yellow]")
                    
                    response_meta = result["response_metadata"]
                    if not response_meta.get("response_safe", True):
                        console.print("[yellow]⚠️  Response was filtered for safety[/yellow]")
            else:
                console.print("\n[yellow]No relevant documents found[/yellow]")
        
        # Handle single query or interactive mode
        if query and not interactive:
            process_query(query)
        else:
            console.print("\n[bold]Interactive Query Mode[/bold]")
            console.print("Type your queries (press Ctrl+C to exit)")
            
            try:
                while True:
                    query_text = console.input("\n[bold cyan]Enter query:[/bold cyan] ")
                    if query_text.lower() in ['exit', 'quit', 'q']:
                        break
                    process_query(query_text)
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error during query: {e}[/red]")
        logger.exception("Query failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--interface",
    "-i",
    type=click.Choice(["streamlit", "gradio"]),
    default="streamlit",
    help="UI interface to launch"
)
@click.option(
    "--host",
    "-h",
    type=str,
    default=config.ui_host,
    help="Host address to bind to"
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=config.ui_port,
    help="Port to bind to"
)
@click.option(
    "--vector-store",
    "-v",
    type=click.Path(path_type=Path),
    default=config.vector_store_path,
    help="Path to vector store directory"
)
def launch_ui(
    interface: str,
    host: str,
    port: int,
    vector_store: Path
) -> None:
    """Launch web UI for LocalRAG."""
    console.print(f"[bold blue]Launching {interface.title()} UI[/bold blue]")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"Vector store: {vector_store}")
    
    try:
        if interface == "streamlit":
            from .ui.streamlit_app import main as streamlit_main
            streamlit_main(host=host, port=port, vector_store_path=str(vector_store))
        elif interface == "gradio":
            from .ui.gradio_app import main as gradio_main
            gradio_main(host=host, port=port, vector_store_path=str(vector_store))
    
    except ImportError as e:
        console.print(f"[red]UI dependencies not available: {e}[/red]")
        console.print("Please install UI dependencies:")
        console.print(f"  uv add {interface}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error launching UI: {e}[/red]")
        logger.exception("UI launch failed")
        sys.exit(1)


@cli.command()
def info() -> None:
    """Show LocalRAG system information."""
    console.print("[bold blue]LocalRAG System Information[/bold blue]")
    
    # Configuration info
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Setting")
    config_table.add_column("Value")
    
    config_table.add_row("Embedding Model", config.embedding_model)
    config_table.add_row("Chunk Size", str(config.chunk_size))
    config_table.add_row("Chunk Overlap", str(config.chunk_overlap))
    config_table.add_row("Vector Store Path", config.vector_store_path)
    config_table.add_row("Toxicity Filter", "enabled" if config.enable_toxicity_filter else "disabled")
    config_table.add_row("Toxicity Threshold", str(config.toxicity_threshold))
    config_table.add_row("Device", config.device)
    
    console.print("\n[bold green]Configuration:[/bold green]")
    console.print(config_table)
    
    # Component status
    status_table = Table(show_header=True, header_style="bold magenta")
    status_table.add_column("Component")
    status_table.add_column("Status")
    
    # Check vector store
    vector_store_exists = Path(config.vector_store_path).exists()
    status_table.add_row("Vector Store", "✅ exists" if vector_store_exists else "❌ not found")
    
    # Check toxicity filter
    toxicity_status = "✅ enabled" if global_toxicity_filter.enabled else "❌ disabled"
    status_table.add_row("Toxicity Filter", toxicity_status)
    
    # Check Google credentials
    google_creds = config.google_credentials_path
    google_status = "✅ configured" if google_creds and Path(google_creds).exists() else "❌ not configured"
    status_table.add_row("Google Translation", google_status)
    
    console.print("\n[bold green]Component Status:[/bold green]")
    console.print(status_table)


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()