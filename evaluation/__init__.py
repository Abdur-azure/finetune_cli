"""
Evaluation package for model assessment and benchmarking.

Provides comprehensive evaluation capabilities:
- Multiple metrics (ROUGE, BLEU, Perplexity, F1, Exact Match)
- Model comparison and benchmarking
- Report generation (Markdown, JSON, HTML)
- Quick evaluation utilities

High-level interface:
- evaluate_model: Comprehensive evaluation
- quick_evaluate: Fast evaluation without config
- benchmark_models: Compare base vs fine-tuned
"""

from .base import (
    Metric,
    Evaluator,
    Benchmarker,
    MetricResult,
    ComparisonResult,
    MetricRegistry,
    register_metric,
    get_metric,
    get_all_metrics,
    list_available_metrics
)

from .metrics import (
    ROUGEMetric,
    ROUGE1Metric,
    ROUGE2Metric,
    ROUGELMetric,
    BLEUMetric,
    PerplexityMetric,
    ExactMatchMetric,
    F1Metric,
    AccuracyMetric,
    create_metric
)

from .evaluator import (
    StandardEvaluator,
    QuickEvaluator,
    evaluate_model,
    quick_evaluate
)

from .benchmarker import (
    StandardBenchmarker,
    ReportGenerator,
    benchmark_models,
    compare_metrics
)


__all__ = [
    # High-level functions
    'evaluate_model',
    'quick_evaluate',
    'benchmark_models',
    'compare_metrics',
    
    # Evaluators
    'StandardEvaluator',
    'QuickEvaluator',
    
    # Benchmarking
    'StandardBenchmarker',
    'ReportGenerator',
    
    # Metrics
    'ROUGE1Metric',
    'ROUGE2Metric',
    'ROUGELMetric',
    'BLEUMetric',
    'PerplexityMetric',
    'ExactMatchMetric',
    'F1Metric',
    'AccuracyMetric',
    'create_metric',
    
    # Base classes
    'Metric',
    'Evaluator',
    'Benchmarker',
    
    # Results
    'MetricResult',
    'ComparisonResult',
    
    # Registry
    'MetricRegistry',
    'register_metric',
    'get_metric',
    'get_all_metrics',
    'list_available_metrics',
]