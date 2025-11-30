"""
Metric implementations for model evaluation.

Provides ROUGE, BLEU, Perplexity, Exact Match, and other metrics.
"""

from typing import List, Optional
import numpy as np

import torch
from rouge_score import rouge_scorer
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.logging import get_logger
from .base import Metric


logger = get_logger(__name__)


# ============================================================================
# ROUGE METRICS
# ============================================================================


class ROUGEMetric(Metric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.
    
    Measures overlap between generated and reference texts.
    """
    
    def __init__(self, rouge_type: str = "rouge1"):
        """
        Initialize ROUGE metric.
        
        Args:
            rouge_type: Type of ROUGE (rouge1, rouge2, rougeL)
        """
        super().__init__()
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute ROUGE score."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.scorer.score(ref, pred)
            scores.append(score[self.rouge_type].fmeasure)
        
        return float(np.mean(scores))
    
    def get_name(self) -> str:
        return self.rouge_type


class ROUGE1Metric(ROUGEMetric):
    """ROUGE-1 (unigram overlap)."""
    def __init__(self):
        super().__init__("rouge1")


class ROUGE2Metric(ROUGEMetric):
    """ROUGE-2 (bigram overlap)."""
    def __init__(self):
        super().__init__("rouge2")


class ROUGELMetric(ROUGEMetric):
    """ROUGE-L (longest common subsequence)."""
    def __init__(self):
        super().__init__("rougeL")


# ============================================================================
# BLEU METRIC
# ============================================================================


class BLEUMetric(Metric):
    """
    BLEU (Bilingual Evaluation Understudy) metric.
    
    Measures n-gram precision between generated and reference texts.
    """
    
    def __init__(self, max_order: int = 4):
        """
        Initialize BLEU metric.
        
        Args:
            max_order: Maximum n-gram order (default: 4)
        """
        super().__init__()
        self.max_order = max_order
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute BLEU score."""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        smoothing = SmoothingFunction().method1
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]  # BLEU expects list of references
            
            try:
                score = sentence_bleu(
                    ref_tokens,
                    pred_tokens,
                    smoothing_function=smoothing
                )
                scores.append(score)
            except:
                scores.append(0.0)
        
        return float(np.mean(scores))
    
    def get_name(self) -> str:
        return "bleu"


# ============================================================================
# PERPLEXITY METRIC
# ============================================================================


class PerplexityMetric(Metric):
    """
    Perplexity metric.
    
    Measures how well the model predicts the reference text.
    Lower is better.
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize perplexity metric.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute perplexity on reference texts.
        
        Note: predictions are ignored for perplexity computation.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for ref in references:
                # Tokenize
                inputs = self.tokenizer(
                    ref,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Compute loss
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                num_tokens = inputs["input_ids"].size(1)
                
                total_loss += loss * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return float(perplexity)
    
    def get_name(self) -> str:
        return "perplexity"
    
    def is_higher_better(self) -> bool:
        """Lower perplexity is better."""
        return False
    
    def get_range(self) -> tuple:
        """Perplexity ranges from 1 to infinity."""
        return (1.0, float('inf'))


# ============================================================================
# EXACT MATCH METRIC
# ============================================================================


class ExactMatchMetric(Metric):
    """
    Exact match metric.
    
    Measures percentage of predictions that exactly match references.
    """
    
    def __init__(self, ignore_case: bool = True, ignore_punctuation: bool = True):
        """
        Initialize exact match metric.
        
        Args:
            ignore_case: Whether to ignore case differences
            ignore_punctuation: Whether to ignore punctuation
        """
        super().__init__()
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.ignore_case:
            text = text.lower()
        
        if self.ignore_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text.strip()
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute exact match accuracy."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        matches = 0
        for pred, ref in zip(predictions, references):
            pred_norm = self._normalize(pred)
            ref_norm = self._normalize(ref)
            
            if pred_norm == ref_norm:
                matches += 1
        
        return matches / len(predictions)
    
    def get_name(self) -> str:
        return "exact_match"


# ============================================================================
# F1 SCORE METRIC
# ============================================================================


class F1Metric(Metric):
    """
    Token-level F1 score.
    
    Measures precision and recall of token overlap.
    """
    
    def __init__(self):
        super().__init__()
    
    def _get_tokens(self, text: str) -> set:
        """Get set of tokens from text."""
        return set(text.lower().split())
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute token F1 score."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._get_tokens(pred)
            ref_tokens = self._get_tokens(ref)
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
                continue
            
            # Calculate precision and recall
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            
            # Calculate F1
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            f1_scores.append(f1)
        
        return float(np.mean(f1_scores))
    
    def get_name(self) -> str:
        return "f1"


# ============================================================================
# ACCURACY METRIC
# ============================================================================


class AccuracyMetric(Metric):
    """
    Classification accuracy metric.
    
    For classification tasks, measures percentage of correct predictions.
    """
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute accuracy."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        return correct / len(predictions)
    
    def get_name(self) -> str:
        return "accuracy"


# ============================================================================
# METRIC FACTORY
# ============================================================================


def create_metric(
    metric_name: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **kwargs
) -> Metric:
    """
    Factory function to create metrics.
    
    Args:
        metric_name: Name of metric
        model: Optional model (required for perplexity)
        tokenizer: Optional tokenizer (required for perplexity)
        **kwargs: Additional metric-specific arguments
    
    Returns:
        Metric instance
    """
    metric_map = {
        'rouge1': ROUGE1Metric,
        'rouge2': ROUGE2Metric,
        'rougeL': ROUGELMetric,
        'bleu': BLEUMetric,
        'exact_match': ExactMatchMetric,
        'f1': F1Metric,
        'accuracy': AccuracyMetric,
    }
    
    # Special handling for perplexity
    if metric_name.lower() == 'perplexity':
        if model is None or tokenizer is None:
            raise ValueError("Perplexity requires model and tokenizer")
        return PerplexityMetric(model, tokenizer)
    
    metric_class = metric_map.get(metric_name.lower())
    if metric_class is None:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    return metric_class(**kwargs)


# ============================================================================
# AUTO-REGISTER METRICS
# ============================================================================


def register_default_metrics():
    """Register all default metrics."""
    from .base import register_metric
    
    register_metric(ROUGE1Metric())
    register_metric(ROUGE2Metric())
    register_metric(ROUGELMetric())
    register_metric(BLEUMetric())
    register_metric(ExactMatchMetric())
    register_metric(F1Metric())
    register_metric(AccuracyMetric())
    
    logger.debug("Registered default metrics")


# Auto-register on import
register_default_metrics()