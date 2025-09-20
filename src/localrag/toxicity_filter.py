"""
Toxicity filtering functionality to clean prompts before LLM inference.
"""

from typing import List, Dict, Optional, Tuple
import re
import logging

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    raise ImportError("transformers is required. Install it with: pip install transformers")

logger = logging.getLogger(__name__)


class ToxicityFilter:
    """Filter toxic content from text inputs."""
    
    def __init__(
        self,
        model_name: str = "martin-ha/toxic-comment-model",
        threshold: float = 0.7,
        use_simple_filter: bool = True
    ):
        """
        Initialize toxicity filter.
        
        Args:
            model_name: Hugging Face model for toxicity detection
            threshold: Threshold for toxicity classification (0-1)
            use_simple_filter: Whether to use simple keyword-based filtering as well
        """
        self.model_name = model_name
        self.threshold = threshold
        self.use_simple_filter = use_simple_filter
        
        self.classifier = None
        self.profanity_words = self._load_profanity_words()
        
        self._load_model()
    
    def _load_model(self):
        """Load the toxicity detection model."""
        try:
            logger.info(f"Loading toxicity detection model: {self.model_name}")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self._has_cuda() else -1,
                return_all_scores=True
            )
            logger.info("Successfully loaded toxicity detection model")
        except Exception as e:
            logger.warning(f"Failed to load model {self.model_name}: {e}")
            logger.info("Falling back to simple keyword-based filtering")
            self.classifier = None
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_profanity_words(self) -> List[str]:
        """Load a basic list of profanity words for simple filtering."""
        # Basic profanity list - in a real implementation, you might load from a file
        return [
            "hate", "kill", "murder", "violence", "attack", "harm", "hurt", "abuse",
            "discriminate", "racist", "sexist", "homophobic", "xenophobic",
            "offensive", "inappropriate", "vulgar", "obscene", "profane"
        ]
    
    def is_toxic(self, text: str) -> bool:
        """
        Check if text contains toxic content.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text is considered toxic
        """
        if not text.strip():
            return False
        
        # Simple keyword-based check
        if self.use_simple_filter and self._simple_toxicity_check(text):
            return True
        
        # Model-based check
        if self.classifier:
            return self._model_toxicity_check(text)
        
        return False
    
    def _simple_toxicity_check(self, text: str) -> bool:
        """
        Simple keyword-based toxicity check.
        
        Args:
            text: Input text
            
        Returns:
            True if potentially toxic
        """
        text_lower = text.lower()
        
        # Check for profanity words
        for word in self.profanity_words:
            if word in text_lower:
                logger.debug(f"Simple filter detected potentially toxic word: {word}")
                return True
        
        # Check for patterns indicating harmful intent
        harmful_patterns = [
            r'\bhow to (kill|murder|harm|hurt)',
            r'\bways to (attack|harm|hurt)',
            r'\bi (hate|despise) .* (people|person|group)',
            r'\b(should|must|need to) (die|suffer)',
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text_lower):
                logger.debug(f"Simple filter detected harmful pattern")
                return True
        
        return False
    
    def _model_toxicity_check(self, text: str) -> bool:
        """
        Model-based toxicity check.
        
        Args:
            text: Input text
            
        Returns:
            True if model classifies as toxic
        """
        try:
            results = self.classifier(text)
            
            # Find toxicity score
            toxic_score = 0.0
            for result in results[0]:  # results is a list with one element
                if result['label'].lower() in ['toxic', 'toxicity', '1', 'positive']:
                    toxic_score = result['score']
                    break
            
            is_toxic = toxic_score > self.threshold
            logger.debug(f"Model toxicity score: {toxic_score:.3f}, threshold: {self.threshold}")
            
            return is_toxic
            
        except Exception as e:
            logger.error(f"Error in model toxicity check: {e}")
            return False
    
    def get_toxicity_score(self, text: str) -> float:
        """
        Get toxicity score for text.
        
        Args:
            text: Input text
            
        Returns:
            Toxicity score (0-1, higher is more toxic)
        """
        if not text.strip():
            return 0.0
        
        if self.classifier:
            try:
                results = self.classifier(text)
                for result in results[0]:
                    if result['label'].lower() in ['toxic', 'toxicity', '1', 'positive']:
                        return result['score']
                return 0.0
            except Exception as e:
                logger.error(f"Error getting toxicity score: {e}")
        
        # Fallback to simple check
        return 1.0 if self._simple_toxicity_check(text) else 0.0
    
    def clean_text(self, text: str, replacement: str = "[FILTERED]") -> str:
        """
        Clean toxic content from text.
        
        Args:
            text: Input text
            replacement: Replacement string for toxic content
            
        Returns:
            Cleaned text
        """
        if not self.is_toxic(text):
            return text
        
        cleaned_text = text
        
        # Remove profanity words
        for word in self.profanity_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        # Remove harmful patterns
        harmful_patterns = [
            (r'\bhow to (kill|murder|harm|hurt)[^.!?]*[.!?]?', replacement),
            (r'\bways to (attack|harm|hurt)[^.!?]*[.!?]?', replacement),
            (r'\bi (hate|despise) .* (people|person|group)[^.!?]*[.!?]?', replacement),
        ]
        
        for pattern, repl in harmful_patterns:
            cleaned_text = re.sub(pattern, repl, cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text.strip()
    
    def filter_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Filter a batch of texts for toxicity.
        
        Args:
            texts: List of texts to filter
            
        Returns:
            List of dictionaries with results
        """
        results = []
        
        for i, text in enumerate(texts):
            is_toxic = self.is_toxic(text)
            toxicity_score = self.get_toxicity_score(text)
            cleaned_text = self.clean_text(text) if is_toxic else text
            
            results.append({
                "index": i,
                "original_text": text,
                "is_toxic": is_toxic,
                "toxicity_score": toxicity_score,
                "cleaned_text": cleaned_text,
            })
        
        return results
    
    def get_filter_stats(self) -> Dict[str, any]:
        """
        Get statistics about the filter configuration.
        
        Returns:
            Dictionary with filter information
        """
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "use_simple_filter": self.use_simple_filter,
            "model_loaded": self.classifier is not None,
            "profanity_words_count": len(self.profanity_words),
        }


class AdvancedToxicityFilter(ToxicityFilter):
    """Advanced toxicity filter with multiple models and techniques."""
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        threshold: float = 0.7,
        ensemble_method: str = "average"
    ):
        """
        Initialize advanced toxicity filter.
        
        Args:
            models: List of model names to use in ensemble
            threshold: Threshold for toxicity classification
            ensemble_method: How to combine multiple model predictions ('average', 'max', 'vote')
        """
        self.models = models or [
            "martin-ha/toxic-comment-model",
            "unitary/toxic-bert"
        ]
        self.ensemble_method = ensemble_method
        self.classifiers = []
        
        super().__init__(self.models[0], threshold, True)
        
        # Load additional models
        self._load_ensemble_models()
    
    def _load_ensemble_models(self):
        """Load multiple models for ensemble prediction."""
        for model_name in self.models:
            try:
                classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    device=0 if self._has_cuda() else -1,
                    return_all_scores=True
                )
                self.classifiers.append((model_name, classifier))
                logger.info(f"Loaded ensemble model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load ensemble model {model_name}: {e}")
    
    def _ensemble_toxicity_check(self, text: str) -> Tuple[bool, float]:
        """
        Ensemble-based toxicity check using multiple models.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_toxic, confidence_score)
        """
        if not self.classifiers:
            # Fallback to parent method
            return self._model_toxicity_check(text), self.get_toxicity_score(text)
        
        scores = []
        
        for model_name, classifier in self.classifiers:
            try:
                results = classifier(text)
                toxic_score = 0.0
                
                for result in results[0]:
                    if result['label'].lower() in ['toxic', 'toxicity', '1', 'positive']:
                        toxic_score = result['score']
                        break
                
                scores.append(toxic_score)
                
            except Exception as e:
                logger.warning(f"Error with model {model_name}: {e}")
                continue
        
        if not scores:
            return False, 0.0
        
        # Ensemble prediction
        if self.ensemble_method == "average":
            final_score = sum(scores) / len(scores)
        elif self.ensemble_method == "max":
            final_score = max(scores)
        elif self.ensemble_method == "vote":
            votes = [score > self.threshold for score in scores]
            final_score = sum(votes) / len(votes)
        else:
            final_score = sum(scores) / len(scores)
        
        is_toxic = final_score > self.threshold
        return is_toxic, final_score
    
    def is_toxic(self, text: str) -> bool:
        """Override parent method to use ensemble."""
        if not text.strip():
            return False
        
        # Simple check first
        if self.use_simple_filter and self._simple_toxicity_check(text):
            return True
        
        # Ensemble model check
        is_toxic, _ = self._ensemble_toxicity_check(text)
        return is_toxic
    
    def get_toxicity_score(self, text: str) -> float:
        """Override parent method to use ensemble."""
        if not text.strip():
            return 0.0
        
        _, score = self._ensemble_toxicity_check(text)
        return score