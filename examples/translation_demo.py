#!/usr/bin/env python3
"""Translation demonstration for LocalRAG."""

import logging

from localrag.translator import Translator, DocumentTranslator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate translation functionality."""
    print("🌐 LocalRAG Translation Demo")
    print("=" * 40)
    
    # Initialize translator
    print("🔧 Initializing translator...")
    translator = Translator()
    document_translator = DocumentTranslator(translator)
    
    if not translator.enabled:
        print("❌ Translation service is not enabled")
        print("This might be because:")
        print("1. Google Cloud Translation API is not available")
        print("2. No credentials are configured")
        print("3. There was an initialization error")
        print("\nTo enable translation:")
        print("1. Install: uv add google-cloud-translate")
        print("2. Set up Google Cloud credentials")
        print("3. Set GOOGLE_APPLICATION_CREDENTIALS in .env")
        
        # Show what would happen with translation enabled
        show_demo_without_translation()
        return
    
    print("✅ Translation service initialized")
    
    # Test language detection
    print("\n🔍 Language Detection Demo")
    print("=" * 40)
    
    test_texts = [
        "Hello, how are you today?",  # English
        "Bonjour, comment allez-vous?",  # French
        "Hola, ¿cómo estás?",  # Spanish
        "Guten Tag, wie geht es Ihnen?",  # German
        "こんにちは、元気ですか？",  # Japanese
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        detection = translator.detect_language(text)
        
        if detection:
            print(f"Detected language: {detection['language']}")
            print(f"Confidence: {detection['confidence']:.3f}")
        else:
            print("Detection failed")
    
    # Test translation
    print("\n🔄 Translation Demo")
    print("=" * 40)
    
    translation_tests = [
        ("Hello, world!", "es"),  # English to Spanish
        ("Machine learning is fascinating", "fr"),  # English to French
        ("How does artificial intelligence work?", "de"),  # English to German
    ]
    
    for text, target_lang in translation_tests:
        print(f"\nOriginal ({translator.default_source_language}): {text}")
        print(f"Target language: {target_lang}")
        
        result = translator.translate_text(text, target_language=target_lang)
        
        if result:
            print(f"Translation: {result['translated_text']}")
            print(f"Source detected: {result.get('source_language', 'unknown')}")
            print(f"Translation needed: {result.get('translation_needed', True)}")
        else:
            print("Translation failed")
    
    # Test query translation
    print("\n💬 Query Translation Demo")
    print("=" * 40)
    
    multilingual_queries = [
        "What is machine learning?",
        "¿Qué es el aprendizaje automático?",  # Spanish
        "Qu'est-ce que l'apprentissage automatique?",  # French
    ]
    
    for query in multilingual_queries:
        print(f"\nOriginal query: {query}")
        
        result = document_translator.translate_query(query, target_language="en")
        
        print(f"Translated: {'Yes' if result['translation_performed'] else 'No'}")
        print(f"Final query: {result['translated_query']}")
        
        if result['translation_performed']:
            print(f"Detected language: {result.get('detected_language', 'unknown')}")
    
    # Test response translation
    print("\n📤 Response Translation Demo")
    print("=" * 40)
    
    english_response = "Machine learning is a method of data analysis that automates analytical model building."
    target_languages = ["es", "fr", "de"]
    
    for target_lang in target_languages:
        print(f"\nTranslating to {target_lang}:")
        
        result = document_translator.translate_response(
            english_response,
            target_language=target_lang,
            source_language="en"
        )
        
        if result['translation_performed']:
            print(f"Translation: {result['translated_response']}")
        else:
            print("No translation performed")
    
    # Test document content translation
    print("\n📄 Document Translation Demo")
    print("=" * 40)
    
    sample_document = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience.
    
    Key Concepts:
    - Supervised Learning
    - Unsupervised Learning
    - Reinforcement Learning
    
    Applications include natural language processing, computer vision, and recommendation systems.
    """
    
    print("Original document (first 200 chars):")
    print(sample_document[:200] + "...")
    
    translated_content = document_translator.translate_document_content(
        sample_document,
        target_language="es",
        preserve_structure=True
    )
    
    if translated_content and translated_content != sample_document:
        print("\nTranslated document (first 200 chars):")
        print(translated_content[:200] + "...")
    else:
        print("\nNo translation performed")
    
    # Show supported languages (limited output)
    print("\n🌍 Supported Languages")
    print("=" * 40)
    
    languages = translator.get_supported_languages()
    if languages:
        print(f"Total supported languages: {len(languages)}")
        print("Sample languages:")
        for lang in languages[:10]:  # Show first 10
            print(f"  {lang['code']}: {lang['name']}")
        print("  ... and more")
    else:
        print("Could not retrieve supported languages")
    
    print("\n✅ Translation demo completed!")


def show_demo_without_translation():
    """Show demo functionality when translation is not available."""
    print("\n📝 Demo Without Translation Service")
    print("=" * 40)
    
    print("Here's what the translation service would provide:")
    print("\n🔍 Language Detection:")
    print("- Detect language of input text")
    print("- Provide confidence scores")
    print("- Support for 100+ languages")
    
    print("\n🔄 Text Translation:")
    print("- Translate between language pairs")
    print("- Preserve document structure")
    print("- Handle batch translations")
    
    print("\n💬 Query Processing:")
    print("- Translate user queries to system language")
    print("- Translate responses back to user language")
    print("- Maintain conversation context")
    
    print("\n📄 Document Processing:")
    print("- Translate document content")
    print("- Preserve formatting and structure")
    print("- Support for multiple document types")
    
    print("\nTo enable these features:")
    print("1. Set up Google Cloud Translation API")
    print("2. Configure credentials")
    print("3. Install required dependencies")


if __name__ == "__main__":
    main()