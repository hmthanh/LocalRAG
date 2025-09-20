"""Translation utilities for LocalRAG using Google Cloud Translation."""

import logging
from typing import Dict, List, Optional, Union

try:
    from google.cloud import translate_v2 as translate
    from google.oauth2 import service_account
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False
    logging.warning("Google Cloud Translation not available. Translation features will be disabled.")

from .config import config

logger = logging.getLogger(__name__)


class Translator:
    """Handle text translation using Google Cloud Translation API."""
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        default_source_language: str = "auto",
        default_target_language: Optional[str] = None
    ):
        """
        Initialize translator.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON
            default_source_language: Default source language ("auto" for detection)
            default_target_language: Default target language
        """
        self.credentials_path = credentials_path or config.google_credentials_path
        self.default_source_language = default_source_language
        self.default_target_language = default_target_language or config.default_language
        
        self.client = None
        self.enabled = False
        
        if GOOGLE_TRANSLATE_AVAILABLE and self.credentials_path:
            try:
                self._initialize_client()
                self.enabled = True
                logger.info("Translation service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize translation service: {e}")
        elif not GOOGLE_TRANSLATE_AVAILABLE:
            logger.warning("Google Cloud Translation not available")
        else:
            logger.info("Translation service disabled (no credentials provided)")
    
    def _initialize_client(self) -> None:
        """Initialize Google Cloud Translation client."""
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self.client = translate.Client(credentials=credentials)
        else:
            # Try to use default credentials
            self.client = translate.Client()
    
    def detect_language(self, text: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language code and confidence, or None if detection fails
        """
        if not self.enabled or not text.strip():
            return None
        
        try:
            result = self.client.detect_language(text)
            return {
                "language": result["language"],
                "confidence": result["confidence"]
            }
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return None
    
    def translate_text(
        self,
        text: str,
        target_language: Optional[str] = None,
        source_language: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code ("auto" for detection)
            
        Returns:
            Dictionary with translated text and metadata, or None if translation fails
        """
        if not self.enabled or not text.strip():
            return None
        
        target_language = target_language or self.default_target_language
        source_language = source_language or self.default_source_language
        
        try:
            # Detect language if source is "auto"
            detected_language = None
            if source_language == "auto":
                detection = self.detect_language(text)
                if detection:
                    detected_language = detection["language"]
                    # Don't translate if already in target language
                    if detected_language == target_language:
                        return {
                            "translated_text": text,
                            "source_language": detected_language,
                            "target_language": target_language,
                            "translation_needed": False
                        }
                    source_language = detected_language
            
            # Perform translation
            result = self.client.translate(
                text,
                target_language=target_language,
                source_language=source_language if source_language != "auto" else None
            )
            
            return {
                "translated_text": result["translatedText"],
                "source_language": result.get("detectedSourceLanguage", detected_language or source_language),
                "target_language": target_language,
                "translation_needed": True
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None
    
    def translate_texts(
        self,
        texts: List[str],
        target_language: Optional[str] = None,
        source_language: Optional[str] = None
    ) -> List[Optional[Dict[str, str]]]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            List of translation results
        """
        return [
            self.translate_text(text, target_language, source_language)
            for text in texts
        ]
    
    def get_supported_languages(self) -> Optional[List[Dict[str, str]]]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported languages with codes and names
        """
        if not self.enabled:
            return None
        
        try:
            languages = self.client.get_languages()
            return [
                {
                    "code": lang["language"],
                    "name": lang.get("name", lang["language"])
                }
                for lang in languages
            ]
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return None
    
    def is_translation_needed(
        self,
        text: str,
        target_language: Optional[str] = None
    ) -> bool:
        """
        Check if translation is needed for text.
        
        Args:
            text: Text to check
            target_language: Target language code
            
        Returns:
            True if translation is needed
        """
        if not self.enabled or not text.strip():
            return False
        
        target_language = target_language or self.default_target_language
        
        detection = self.detect_language(text)
        if detection:
            return detection["language"] != target_language
        
        return True  # Assume translation needed if detection fails


class DocumentTranslator:
    """Translate documents while preserving structure."""
    
    def __init__(self, translator: Optional[Translator] = None):
        """
        Initialize document translator.
        
        Args:
            translator: Translator instance to use
        """
        self.translator = translator or Translator()
    
    def translate_document_content(
        self,
        content: str,
        target_language: Optional[str] = None,
        preserve_structure: bool = True
    ) -> Optional[str]:
        """
        Translate document content.
        
        Args:
            content: Document content to translate
            target_language: Target language code
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Translated content or None if translation fails
        """
        if not self.translator.enabled:
            return content
        
        if preserve_structure:
            # Split content into paragraphs for better translation
            paragraphs = content.split('\n\n')
            translated_paragraphs = []
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    result = self.translator.translate_text(paragraph, target_language)
                    if result:
                        translated_paragraphs.append(result["translated_text"])
                    else:
                        translated_paragraphs.append(paragraph)  # Keep original on failure
                else:
                    translated_paragraphs.append(paragraph)  # Keep empty paragraphs
            
            return '\n\n'.join(translated_paragraphs)
        else:
            # Translate entire content as one block
            result = self.translator.translate_text(content, target_language)
            return result["translated_text"] if result else content
    
    def translate_query(
        self,
        query: str,
        target_language: Optional[str] = None
    ) -> Dict[str, Union[str, bool]]:
        """
        Translate a user query.
        
        Args:
            query: User query to translate
            target_language: Target language code
            
        Returns:
            Dictionary with translation results
        """
        if not self.translator.enabled:
            return {
                "translated_query": query,
                "original_query": query,
                "translation_performed": False,
                "detected_language": None
            }
        
        result = self.translator.translate_text(query, target_language)
        
        if result and result.get("translation_needed", True):
            return {
                "translated_query": result["translated_text"],
                "original_query": query,
                "translation_performed": True,
                "detected_language": result.get("source_language"),
                "target_language": result.get("target_language")
            }
        else:
            return {
                "translated_query": query,
                "original_query": query,
                "translation_performed": False,
                "detected_language": result.get("source_language") if result else None
            }
    
    def translate_response(
        self,
        response: str,
        target_language: str,
        source_language: str = "en"
    ) -> Dict[str, Union[str, bool]]:
        """
        Translate a system response back to user's language.
        
        Args:
            response: System response to translate
            target_language: Target language code (user's language)
            source_language: Source language code (system's language)
            
        Returns:
            Dictionary with translation results
        """
        if not self.translator.enabled or target_language == source_language:
            return {
                "translated_response": response,
                "original_response": response,
                "translation_performed": False
            }
        
        result = self.translator.translate_text(
            response,
            target_language=target_language,
            source_language=source_language
        )
        
        if result:
            return {
                "translated_response": result["translated_text"],
                "original_response": response,
                "translation_performed": True,
                "source_language": source_language,
                "target_language": target_language
            }
        else:
            return {
                "translated_response": response,
                "original_response": response,
                "translation_performed": False
            }


# Global translator instances
global_translator = Translator()
global_document_translator = DocumentTranslator(global_translator)