"""Toxicity filtering for LocalRAG prompts and responses."""

import logging
from typing import Dict, List, Optional, Tuple, Union

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    logging.warning("Detoxify not available. Toxicity filtering will be disabled.")

from .config import config

logger = logging.getLogger(__name__)


class ToxicityFilter:
    """Filter toxic content from prompts and responses."""
    
    def __init__(
        self,
        model_name: str = "original",
        threshold: Optional[float] = None,
        enabled: Optional[bool] = None
    ):
        """
        Initialize toxicity filter.
        
        Args:
            model_name: Detoxify model to use ("original", "unbiased", "multilingual")
            threshold: Toxicity threshold (0-1). Higher values are more permissive.
            enabled: Whether toxicity filtering is enabled
        """
        self.threshold = threshold or config.toxicity_threshold
        self.enabled = enabled if enabled is not None else config.enable_toxicity_filter
        self.model_name = model_name
        
        self.model = None
        if self.enabled and DETOXIFY_AVAILABLE:
            try:
                logger.info(f"Loading toxicity model: {model_name}")
                self.model = Detoxify(model_name)
                logger.info("Toxicity filter initialized successfully")
            except Exception as e:
                logger.error(f"Failed to load toxicity model: {e}")
                self.enabled = False
        elif self.enabled and not DETOXIFY_AVAILABLE:
            logger.warning("Toxicity filtering requested but detoxify not available")
            self.enabled = False
    
    def is_toxic(self, text: str) -> bool:
        """
        Check if text is toxic.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is considered toxic
        """
        if not self.enabled or not self.model:
            return False
        
        if not text or not text.strip():
            return False
        
        try:
            scores = self.model.predict(text)
            
            # Check if any toxicity score exceeds threshold
            for category, score in scores.items():
                if score > self.threshold:
                    logger.warning(f"Text flagged for {category}: {score:.3f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking toxicity: {e}")
            return False
    
    def get_toxicity_scores(self, text: str) -> Dict[str, float]:
        """
        Get detailed toxicity scores.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of toxicity scores by category
        """
        if not self.enabled or not self.model:
            return {}
        
        if not text or not text.strip():
            return {}
        
        try:
            scores = self.model.predict(text)
            # Convert numpy types to float for JSON serialization
            return {k: float(v) for k, v in scores.items()}
            
        except Exception as e:
            logger.error(f"Error getting toxicity scores: {e}")
            return {}
    
    def filter_text(self, text: str, replacement: str = "[FILTERED]") -> str:
        """
        Filter toxic text by replacing it with a placeholder.
        
        Args:
            text: Text to filter
            replacement: Replacement text for toxic content
            
        Returns:
            Filtered text
        """
        if not self.enabled or not self.is_toxic(text):
            return text
        
        logger.info("Filtering toxic content")
        return replacement
    
    def filter_texts(
        self, 
        texts: List[str], 
        replacement: str = "[FILTERED]"
    ) -> List[str]:
        """
        Filter multiple texts.
        
        Args:
            texts: List of texts to filter
            replacement: Replacement text for toxic content
            
        Returns:
            List of filtered texts
        """
        return [self.filter_text(text, replacement) for text in texts]
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[bool, Dict[str, float]]]]:
        """
        Analyze multiple texts for toxicity.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        
        for text in texts:
            scores = self.get_toxicity_scores(text)
            is_toxic = self.is_toxic(text)
            
            results.append({
                "is_toxic": is_toxic,
                "scores": scores,
                "text_preview": text[:100] + "..." if len(text) > 100 else text
            })
        
        return results
    
    def get_statistics(self, texts: List[str]) -> Dict[str, Union[int, float, Dict[str, int]]]:
        """
        Get toxicity statistics for a collection of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Statistics dictionary
        """
        if not self.enabled:
            return {"enabled": False, "total_texts": len(texts)}
        
        total_texts = len(texts)
        toxic_count = 0
        category_counts = {}
        all_scores = []
        
        for text in texts:
            scores = self.get_toxicity_scores(text)
            all_scores.append(scores)
            
            is_text_toxic = False
            for category, score in scores.items():
                if score > self.threshold:
                    is_text_toxic = True
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            if is_text_toxic:
                toxic_count += 1
        
        # Calculate average scores
        avg_scores = {}
        if all_scores:
            categories = set()
            for scores in all_scores:
                categories.update(scores.keys())
            
            for category in categories:
                category_scores = [scores.get(category, 0) for scores in all_scores]
                avg_scores[category] = sum(category_scores) / len(category_scores)
        
        return {
            "enabled": True,
            "total_texts": total_texts,
            "toxic_count": toxic_count,
            "toxic_percentage": (toxic_count / total_texts * 100) if total_texts > 0 else 0,
            "threshold": self.threshold,
            "category_violations": category_counts,
            "average_scores": avg_scores
        }


class SafetyWrapper:
    """Wrapper to ensure safe interactions with the RAG system."""
    
    def __init__(self, toxicity_filter: Optional[ToxicityFilter] = None):
        """
        Initialize safety wrapper.
        
        Args:
            toxicity_filter: Toxicity filter to use
        """
        self.toxicity_filter = toxicity_filter or ToxicityFilter()
    
    def safe_query(self, query: str) -> Tuple[str, bool, Dict[str, float]]:
        """
        Process a query safely.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (filtered_query, is_safe, toxicity_scores)
        """
        if not query or not query.strip():
            return query, True, {}
        
        toxicity_scores = self.toxicity_filter.get_toxicity_scores(query)
        is_safe = not self.toxicity_filter.is_toxic(query)
        
        if is_safe:
            return query, True, toxicity_scores
        else:
            logger.warning(f"Unsafe query detected: {query[:50]}...")
            filtered_query = self.toxicity_filter.filter_text(query)
            return filtered_query, False, toxicity_scores
    
    def safe_response(self, response: str) -> Tuple[str, bool, Dict[str, float]]:
        """
        Process a response safely.
        
        Args:
            response: System response
            
        Returns:
            Tuple of (filtered_response, is_safe, toxicity_scores)
        """
        if not response or not response.strip():
            return response, True, {}
        
        toxicity_scores = self.toxicity_filter.get_toxicity_scores(response)
        is_safe = not self.toxicity_filter.is_toxic(response)
        
        if is_safe:
            return response, True, toxicity_scores
        else:
            logger.warning("Unsafe response detected, filtering...")
            filtered_response = self.toxicity_filter.filter_text(
                response, 
                "[Response filtered due to inappropriate content]"
            )
            return filtered_response, False, toxicity_scores
    
    def log_safety_event(
        self, 
        event_type: str, 
        content: str, 
        scores: Dict[str, float]
    ) -> None:
        """
        Log safety-related events.
        
        Args:
            event_type: Type of event ("query", "response", "document")
            content: Content that triggered the event
            scores: Toxicity scores
        """
        logger.warning(
            f"Safety event - {event_type}: "
            f"Content: {content[:100]}..., "
            f"Scores: {scores}"
        )


# Global toxicity filter instance
global_toxicity_filter = ToxicityFilter()
global_safety_wrapper = SafetyWrapper(global_toxicity_filter)