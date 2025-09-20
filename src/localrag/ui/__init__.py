"""User interface modules for LocalRAG."""

from typing import Optional


def check_streamlit_available() -> bool:
    """Check if Streamlit is available."""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def check_gradio_available() -> bool:
    """Check if Gradio is available."""
    try:
        import gradio
        return True
    except ImportError:
        return False


def get_available_interfaces() -> list[str]:
    """Get list of available UI interfaces."""
    interfaces = []
    
    if check_streamlit_available():
        interfaces.append("streamlit")
    
    if check_gradio_available():
        interfaces.append("gradio")
    
    return interfaces