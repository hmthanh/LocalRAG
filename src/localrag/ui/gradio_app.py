"""Gradio web interface for LocalRAG."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from .. import create_retriever, DocumentLoader, TextProcessor, create_vector_store

logger = logging.getLogger(__name__)


class LocalRAGInterface:
    """Gradio interface for LocalRAG system."""
    
    def __init__(self, vector_store_path: str = "./data/vector_store"):
        """Initialize the interface."""
        self.vector_store_path = vector_store_path
        self.retriever = None
        self.chat_history = []
        
        # Try to load existing retriever
        self.load_retriever()
    
    def load_retriever(self) -> str:
        """Load the RAG retriever."""
        try:
            if not Path(self.vector_store_path).exists():
                return f"❌ Vector store not found at {self.vector_store_path}. Please ingest documents first."
            
            self.retriever = create_retriever(vector_store_path=self.vector_store_path)
            stats = self.retriever.get_retriever_stats()
            
            return f"✅ Loaded vector store with {stats['vector_store_stats']['total_documents']} documents"
        
        except Exception as e:
            logger.exception("Error loading retriever")
            return f"❌ Error loading retriever: {e}"
    
    def process_documents(
        self,
        files: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> str:
        """Process uploaded documents."""
        if not files:
            return "❌ No files provided"
        
        try:
            document_loader = DocumentLoader()
            text_processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            documents = []
            processed_files = []
            
            for file_path in files:
                try:
                    doc = document_loader.load_pdf(file_path)
                    documents.append(doc)
                    processed_files.append(Path(file_path).name)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
            
            if not documents:
                return "❌ No documents could be processed"
            
            # Chunk documents
            chunks = text_processor.chunk_documents(documents)
            
            # Update vector store
            vector_store = create_vector_store(
                documents=chunks,
                persist_directory=self.vector_store_path
            )
            
            # Reload retriever
            self.retriever = create_retriever(vector_store_path=self.vector_store_path)
            
            return f"✅ Successfully processed {len(processed_files)} files and added {len(chunks)} chunks to vector store.\nProcessed files: {', '.join(processed_files)}"
        
        except Exception as e:
            logger.exception("Document processing failed")
            return f"❌ Error processing documents: {e}"
    
    def query_system(
        self,
        query: str,
        top_k: int = 4,
        enable_safety: bool = True,
        show_sources: bool = True
    ) -> Tuple[str, str]:
        """Query the RAG system."""
        if not query.strip():
            return "❌ Please enter a query", ""
        
        if self.retriever is None:
            return "❌ No retriever loaded. Please load a vector store or upload documents first.", ""
        
        try:
            result = self.retriever.search_and_respond(
                query=query,
                k=top_k,
                enable_safety=enable_safety
            )
            
            response = result["response"]
            sources_info = ""
            
            if result["documents"] and show_sources:
                sources_info = "\n\n📚 **Source Documents:**\n"
                
                for i, doc in enumerate(result["documents"], 1):
                    source = doc.metadata.get("source", "Unknown")
                    file_name = doc.metadata.get("file_name", Path(source).name)
                    
                    # Show similarity score if available
                    score_info = ""
                    if "similarity_score" in doc.metadata:
                        score = doc.metadata["similarity_score"]
                        score_info = f" (similarity: {score:.3f})"
                    
                    sources_info += f"\n**Document {i}:** {file_name}{score_info}\n"
                    
                    # Show content preview
                    content = doc.page_content[:200]
                    if len(doc.page_content) > 200:
                        content += "..."
                    sources_info += f"*{content}*\n"
            
            # Add to chat history
            self.chat_history.append({
                "query": query,
                "response": response,
                "document_count": len(result["documents"])
            })
            
            # Safety warnings
            safety_info = ""
            if enable_safety:
                retrieval_meta = result["retrieval_metadata"]
                if not retrieval_meta.get("query_safe", True):
                    safety_info += "\n⚠️ Query was filtered for safety\n"
                
                response_meta = result["response_metadata"]
                if not response_meta.get("response_safe", True):
                    safety_info += "⚠️ Response was filtered for safety\n"
            
            full_response = response + sources_info + safety_info
            
            return full_response, f"Found {len(result['documents'])} relevant documents"
        
        except Exception as e:
            logger.exception("Query failed")
            return f"❌ Error during search: {e}", ""
    
    def get_system_info(self) -> str:
        """Get system information."""
        info = f"**Vector Store Path:** {self.vector_store_path}\n"
        
        if self.retriever:
            stats = self.retriever.get_retriever_stats()
            info += f"**Total Documents:** {stats['vector_store_stats']['total_documents']}\n"
            info += f"**Embedding Model:** {stats['vector_store_stats']['embedding_model']}\n"
            info += f"**Safety Filtering:** {'✅ Enabled' if stats['safety_enabled'] else '❌ Disabled'}\n"
            info += f"**Translation:** {'✅ Enabled' if stats['translation_enabled'] else '❌ Disabled'}\n"
        else:
            info += "**Status:** No retriever loaded\n"
        
        info += f"**Chat History:** {len(self.chat_history)} interactions\n"
        
        return info
    
    def get_chat_history(self) -> str:
        """Get formatted chat history."""
        if not self.chat_history:
            return "No chat history available."
        
        history = "**Recent Chat History:**\n\n"
        
        for i, item in enumerate(reversed(self.chat_history[-10:]), 1):
            history += f"**Query {i}:** {item['query']}\n"
            history += f"**Response:** {item['response'][:200]}...\n"
            history += f"**Documents Found:** {item['document_count']}\n\n"
        
        return history
    
    def clear_chat_history(self) -> str:
        """Clear chat history."""
        self.chat_history = []
        return "✅ Chat history cleared"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="LocalRAG",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px;
                margin: auto;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # 🔍 LocalRAG
                *Local RAG system with FAISS, LangChain, and Hugging Face*
                """
            )
            
            with gr.Tabs():
                # Query Tab
                with gr.Tab("💬 Query"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            query_input = gr.Textbox(
                                label="Enter your query",
                                placeholder="Ask a question about your documents...",
                                lines=3
                            )
                            
                            with gr.Row():
                                top_k = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    value=4,
                                    step=1,
                                    label="Number of results"
                                )
                                enable_safety = gr.Checkbox(
                                    label="Enable safety filtering",
                                    value=True
                                )
                                show_sources = gr.Checkbox(
                                    label="Show source documents",
                                    value=True
                                )
                            
                            query_btn = gr.Button("Search", variant="primary")
                        
                        with gr.Column(scale=1):
                            status_output = gr.Textbox(
                                label="Status",
                                interactive=False,
                                lines=2
                            )
                    
                    response_output = gr.Markdown(label="Response")
                    
                    query_btn.click(
                        fn=self.query_system,
                        inputs=[query_input, top_k, enable_safety, show_sources],
                        outputs=[response_output, status_output]
                    )
                
                # Document Management Tab
                with gr.Tab("📄 Documents"):
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(
                                label="Upload PDF documents",
                                file_types=[".pdf"],
                                file_count="multiple"
                            )
                            
                            with gr.Row():
                                chunk_size = gr.Number(
                                    label="Chunk Size",
                                    value=1000,
                                    minimum=100,
                                    maximum=2000
                                )
                                chunk_overlap = gr.Number(
                                    label="Chunk Overlap",
                                    value=200,
                                    minimum=0,
                                    maximum=500
                                )
                            
                            process_btn = gr.Button("Process Documents", variant="primary")
                            
                            process_output = gr.Textbox(
                                label="Processing Status",
                                interactive=False,
                                lines=5
                            )
                    
                    process_btn.click(
                        fn=self.process_documents,
                        inputs=[file_upload, chunk_size, chunk_overlap],
                        outputs=process_output
                    )
                
                # History Tab
                with gr.Tab("💭 History"):
                    with gr.Row():
                        with gr.Column():
                            history_output = gr.Markdown(label="Chat History")
                            
                            with gr.Row():
                                refresh_history_btn = gr.Button("Refresh History")
                                clear_history_btn = gr.Button("Clear History", variant="stop")
                    
                    refresh_history_btn.click(
                        fn=self.get_chat_history,
                        outputs=history_output
                    )
                    
                    clear_history_btn.click(
                        fn=self.clear_chat_history,
                        outputs=gr.Textbox(visible=False)
                    ).then(
                        fn=self.get_chat_history,
                        outputs=history_output
                    )
                
                # Settings Tab
                with gr.Tab("⚙️ Settings"):
                    with gr.Row():
                        with gr.Column():
                            reload_btn = gr.Button("Reload Retriever", variant="secondary")
                            
                            system_info = gr.Markdown(label="System Information")
                            
                            # Auto-refresh system info
                            reload_btn.click(
                                fn=self.load_retriever,
                                outputs=gr.Textbox(visible=False)
                            ).then(
                                fn=self.get_system_info,
                                outputs=system_info
                            )
            
            # Load initial system info
            interface.load(
                fn=self.get_system_info,
                outputs=gr.Markdown(visible=False)
            )
        
        return interface


def main(host: str = "0.0.0.0", port: int = 7860, vector_store_path: str = "./data/vector_store"):
    """Launch Gradio app."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is not available. Please install it with: uv add gradio")
    
    # Create interface
    app = LocalRAGInterface(vector_store_path=vector_store_path)
    interface = app.create_interface()
    
    # Launch
    interface.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()