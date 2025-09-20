"""Streamlit web interface for LocalRAG."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .. import create_retriever, DocumentLoader, TextProcessor, create_vector_store

logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize Streamlit session state."""
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "vector_store_path" not in st.session_state:
        st.session_state.vector_store_path = "./data/vector_store"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def load_retriever(vector_store_path: str):
    """Load the RAG retriever."""
    try:
        if not Path(vector_store_path).exists():
            st.error(f"Vector store not found at {vector_store_path}")
            st.info("Please ingest documents first using the CLI: `uv run localrag-ingest --input-dir ./documents`")
            return None
        
        with st.spinner("Loading retriever..."):
            retriever = create_retriever(vector_store_path=vector_store_path)
            stats = retriever.get_retriever_stats()
            
            st.success(f"Loaded vector store with {stats['vector_store_stats']['total_documents']} documents")
            return retriever
    
    except Exception as e:
        st.error(f"Error loading retriever: {e}")
        return None


def document_upload_interface():
    """Interface for uploading and processing documents."""
    st.header("📄 Document Management")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files to add to the knowledge base"
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000)
        
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)
        
        if st.button("Process Documents", type="primary"):
            try:
                # Save uploaded files temporarily
                temp_dir = Path("/tmp/localrag_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                document_loader = DocumentLoader()
                text_processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                documents = []
                
                with st.spinner("Processing uploaded documents..."):
                    for uploaded_file in uploaded_files:
                        # Save file
                        temp_file = temp_dir / uploaded_file.name
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Load document
                        try:
                            doc = document_loader.load_pdf(temp_file)
                            documents.append(doc)
                            st.success(f"✅ Processed {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                
                if documents:
                    # Chunk documents
                    chunks = text_processor.chunk_documents(documents)
                    
                    # Update vector store
                    vector_store = create_vector_store(
                        documents=chunks,
                        persist_directory=st.session_state.vector_store_path
                    )
                    
                    st.success(f"Added {len(chunks)} chunks to vector store")
                    
                    # Reload retriever
                    st.session_state.retriever = load_retriever(st.session_state.vector_store_path)
                    
                    # Clean up temp files
                    for temp_file in temp_dir.glob("*"):
                        temp_file.unlink()
                
            except Exception as e:
                st.error(f"Error processing documents: {e}")
                logger.exception("Document processing failed")


def query_interface():
    """Interface for querying the RAG system."""
    st.header("💬 Query Interface")
    
    if st.session_state.retriever is None:
        st.warning("Please load a vector store first or upload documents")
        return
    
    # Query input
    query = st.text_area(
        "Enter your query:",
        placeholder="Ask a question about your documents...",
        height=100
    )
    
    # Query options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_k = st.number_input("Number of results", min_value=1, max_value=20, value=4)
    
    with col2:
        enable_safety = st.checkbox("Enable safety filtering", value=True)
    
    with col3:
        show_scores = st.checkbox("Show similarity scores", value=False)
    
    if st.button("Search", type="primary") and query.strip():
        try:
            with st.spinner("Searching..."):
                result = st.session_state.retriever.search_and_respond(
                    query=query,
                    k=top_k,
                    enable_safety=enable_safety
                )
            
            # Display results
            if result["documents"]:
                st.subheader("📋 Response")
                st.write(result["response"])
                
                st.subheader("📚 Source Documents")
                
                for i, doc in enumerate(result["documents"], 1):
                    with st.expander(f"Document {i} - {doc.metadata.get('file_name', 'Unknown')}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        
                        with col2:
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                            if show_scores and "similarity_score" in doc.metadata:
                                st.metric("Similarity", f"{doc.metadata['similarity_score']:.3f}")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": query,
                    "response": result["response"],
                    "document_count": len(result["documents"])
                })
                
                # Show safety info if applicable
                if enable_safety:
                    retrieval_meta = result["retrieval_metadata"]
                    if not retrieval_meta.get("query_safe", True):
                        st.warning("⚠️ Query was filtered for safety")
                    
                    response_meta = result["response_metadata"]
                    if not response_meta.get("response_safe", True):
                        st.warning("⚠️ Response was filtered for safety")
            
            else:
                st.warning("No relevant documents found for your query.")
        
        except Exception as e:
            st.error(f"Error during search: {e}")
            logger.exception("Search failed")


def chat_history_interface():
    """Interface for displaying chat history."""
    if st.session_state.chat_history:
        st.header("💭 Chat History")
        
        for i, item in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
            with st.expander(f"Query {i}: {item['query'][:50]}..."):
                st.write(f"**Query:** {item['query']}")
                st.write(f"**Response:** {item['response']}")
                st.caption(f"Found {item['document_count']} relevant documents")
        
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()


def settings_interface():
    """Interface for system settings."""
    st.header("⚙️ Settings")
    
    # Vector store path
    new_path = st.text_input(
        "Vector Store Path",
        value=st.session_state.vector_store_path,
        help="Path to the vector store directory"
    )
    
    if new_path != st.session_state.vector_store_path:
        if st.button("Update Vector Store Path"):
            st.session_state.vector_store_path = new_path
            st.session_state.retriever = None
            st.success("Vector store path updated. Please reload the retriever.")
    
    # Load/reload retriever
    if st.button("Load/Reload Retriever"):
        st.session_state.retriever = load_retriever(st.session_state.vector_store_path)
    
    # System info
    if st.session_state.retriever:
        st.subheader("System Information")
        stats = st.session_state.retriever.get_retriever_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Documents", stats['vector_store_stats']['total_documents'])
            st.metric("Embedding Model", stats['vector_store_stats']['embedding_model'])
        
        with col2:
            st.metric("Safety Enabled", "✅" if stats['safety_enabled'] else "❌")
            st.metric("Translation Enabled", "✅" if stats['translation_enabled'] else "❌")


def main_app():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LocalRAG",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 LocalRAG")
    st.markdown("*Local RAG system with FAISS, LangChain, and Hugging Face*")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        
        page = st.selectbox(
            "Choose a page:",
            ["Query", "Documents", "History", "Settings"]
        )
        
        st.divider()
        
        # Quick stats
        if st.session_state.retriever:
            stats = st.session_state.retriever.get_retriever_stats()
            st.metric("Documents", stats['vector_store_stats']['total_documents'])
        else:
            st.info("No retriever loaded")
        
        st.divider()
        
        # Footer
        st.markdown("---")
        st.markdown("**LocalRAG v0.1.0**")
        st.markdown("Built with Streamlit")
    
    # Main content
    if page == "Query":
        query_interface()
    elif page == "Documents":
        document_upload_interface()
    elif page == "History":
        chat_history_interface()
    elif page == "Settings":
        settings_interface()


def main(host: str = "0.0.0.0", port: int = 8501, vector_store_path: str = "./data/vector_store"):
    """Launch Streamlit app."""
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit is not available. Please install it with: uv add streamlit")
    
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_ADDRESS"] = host
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    
    # Set vector store path in session state
    if get_script_run_ctx() is None:
        # Running from command line
        import streamlit.web.cli as stcli
        sys.argv = [
            "streamlit",
            "run",
            __file__,
            "--server.address", host,
            "--server.port", str(port),
            "--",
            "--vector-store-path", vector_store_path
        ]
        sys.exit(stcli.main())
    else:
        # Running within Streamlit
        main_app()


if __name__ == "__main__":
    # Handle command line arguments
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-store-path", default="./data/vector_store")
    args = parser.parse_args()
    
    # Set in session state if available
    if STREAMLIT_AVAILABLE and get_script_run_ctx() is not None:
        init_session_state()
        st.session_state.vector_store_path = args.vector_store_path
    
    main_app()