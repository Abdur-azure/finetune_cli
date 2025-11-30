"""
Model benchmarking and comparison system.

Compares base models against fine-tuned models with comprehensive reporting.
"""

from typing import Dict, List, Optional
from pathlib import Path
import json

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from ..core.types import EvaluationConfig
from ..utils.logging import get_logger, LogProgress
from .base import Benchmarker, ComparisonResult
from .evaluator import StandardEvaluator


logger = get_logger(__name__)


# ============================================================================
# STANDARD BENCHMARKER
# ============================================================================


class StandardBenchmarker(Benchmarker):
    """
    Standard implementation of model benchmarking.
    
    Evaluates both base and fine-tuned models on the same dataset
    and computes improvement metrics.
    """
    
    def benchmark(
        self,
        base_model: PreTrainedModel,
        finetuned_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        config: EvaluationConfig
    ) -> ComparisonResult:
        """
        Benchmark base vs fine-tuned model.
        
        Args:
            base_model: Original base model
            finetuned_model: Fine-tuned model
            tokenizer: Tokenizer
            dataset: Evaluation dataset
            config: Evaluation configuration
        
        Returns:
            Comparison results with improvements
        """
        with LogProgress(logger, "Benchmarking models"):
            
            # Evaluate base model
            logger.info("Evaluating base model...")
            base_evaluator = StandardEvaluator(base_model, tokenizer, config)
            base_result = base_evaluator.evaluate(dataset)
            
            # Evaluate fine-tuned model
            logger.info("Evaluating fine-tuned model...")
            ft_evaluator = StandardEvaluator(finetuned_model, tokenizer, config)
            ft_result = ft_evaluator.evaluate(dataset)
            
            # Compute improvements
            improvements = self._compute_improvements(
                base_result.metrics,
                ft_result.metrics
            )
            
            # Build comparison result
            result = ComparisonResult(
                base_metrics=base_result.metrics,
                finetuned_metrics=ft_result.metrics,
                improvements=improvements
            )
            
            # Log summary
            self._log_comparison(result)
            
            return result
    
    def _compute_improvements(
        self,
        base_metrics: Dict[str, float],
        ft_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute improvement percentages.
        
        Args:
            base_metrics: Base model metrics
            ft_metrics: Fine-tuned model metrics
        
        Returns:
            Dictionary of improvement percentages
        """
        improvements = {}
        
        for metric_name in base_metrics:
            base_score = base_metrics[metric_name]
            ft_score = ft_metrics.get(metric_name, 0.0)
            
            # Calculate percentage improvement
            if base_score > 0:
                improvement = ((ft_score - base_score) / base_score) * 100
            else:
                improvement = 0.0
            
            improvements[metric_name] = improvement
        
        return improvements
    
    def _log_comparison(self, result: ComparisonResult) -> None:
        """Log comparison results."""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK RESULTS")
        logger.info("="*70)
        logger.info(f"{'Metric':<15} {'Base':<12} {'Fine-tuned':<12} {'Improvement':<15}")
        logger.info("-"*70)
        
        for metric in result.base_metrics:
            base = result.base_metrics[metric]
            ft = result.finetuned_metrics[metric]
            imp = result.improvements[metric]
            
            logger.info(
                f"{metric:<15} {base:<12.4f} {ft:<12.4f} {imp:>+7.2f}%"
            )
        
        logger.info("="*70)
        avg_improvement = result.get_average_improvement()
        logger.info(f"Average Improvement: {avg_improvement:+.2f}%")
        logger.info("="*70 + "\n")


# ============================================================================
# REPORT GENERATOR
# ============================================================================


class ReportGenerator:
    """
    Generates formatted reports from benchmark results.
    """
    
    @staticmethod
    def generate_markdown(
        result: ComparisonResult,
        title: str = "Model Benchmark Report"
    ) -> str:
        """
        Generate Markdown report.
        
        Args:
            result: Comparison results
            title: Report title
        
        Returns:
            Markdown formatted report
        """
        lines = [
            f"# {title}",
            "",
            f"**Generated:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Results Summary",
            "",
            "| Metric | Base Model | Fine-tuned Model | Improvement |",
            "|--------|------------|------------------|-------------|"
        ]
        
        for metric in result.base_metrics:
            base = result.base_metrics[metric]
            ft = result.finetuned_metrics[metric]
            imp = result.improvements[metric]
            
            lines.append(
                f"| {metric} | {base:.4f} | {ft:.4f} | {imp:+.2f}% |"
            )
        
        lines.extend([
            "",
            "## Key Findings",
            "",
            f"- **Average Improvement:** {result.get_average_improvement():+.2f}%"
        ])
        
        # Find best/worst improvements
        best_metric = max(result.improvements, key=result.improvements.get)
        worst_metric = min(result.improvements, key=result.improvements.get)
        
        lines.extend([
            f"- **Best Improvement:** {best_metric} ({result.improvements[best_metric]:+.2f}%)",
            f"- **Worst Improvement:** {worst_metric} ({result.improvements[worst_metric]:+.2f}%)",
            ""
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_json(result: ComparisonResult) -> str:
        """
        Generate JSON report.
        
        Args:
            result: Comparison results
        
        Returns:
            JSON formatted report
        """
        return json.dumps(result.to_dict(), indent=2)
    
    @staticmethod
    def generate_html(
        result: ComparisonResult,
        title: str = "Model Benchmark Report"
    ) -> str:
        """
        Generate HTML report.
        
        Args:
            result: Comparison results
            title: Report title
        
        Returns:
            HTML formatted report
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .summary {{ background-color: #e7f3fe; padding: 15px; margin: 20px 0; border-left: 6px solid #2196F3; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p><strong>Generated:</strong> {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Results</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Base Model</th>
            <th>Fine-tuned Model</th>
            <th>Improvement</th>
        </tr>
"""
        
        for metric in result.base_metrics:
            base = result.base_metrics[metric]
            ft = result.finetuned_metrics[metric]
            imp = result.improvements[metric]
            
            color_class = "positive" if imp > 0 else "negative"
            
            html += f"""
        <tr>
            <td>{metric}</td>
            <td>{base:.4f}</td>
            <td>{ft:.4f}</td>
            <td class="{color_class}">{imp:+.2f}%</td>
        </tr>
"""
        
        avg_improvement = result.get_average_improvement()
        avg_color = "positive" if avg_improvement > 0 else "negative"
        
        html += f"""
    </table>
    
    <div class="summary">
        <h3>Summary</h3>
        <p><strong>Average Improvement:</strong> <span class="{avg_color}">{avg_improvement:+.2f}%</span></p>
    </div>
</body>
</html>
"""
        
        return html
    
    @staticmethod
    def save_report(
        result: ComparisonResult,
        output_path: Path,
        format: str = "markdown",
        title: str = "Model Benchmark Report"
    ) -> None:
        """
        Save report to file.
        
        Args:
            result: Comparison results
            output_path: Output file path
            format: Report format ('markdown', 'json', 'html')
            title: Report title
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "markdown":
            content = ReportGenerator.generate_markdown(result, title)
        elif format == "json":
            content = ReportGenerator.generate_json(result)
        elif format == "html":
            content = ReportGenerator.generate_html(result, title)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Report saved to: {output_path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def benchmark_models(
    base_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    config: EvaluationConfig,
    save_report: Optional[Path] = None
) -> ComparisonResult:
    """
    Convenience function for benchmarking models.
    
    Args:
        base_model: Original base model
        finetuned_model: Fine-tuned model
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        config: Evaluation configuration
        save_report: Optional path to save report
    
    Returns:
        Comparison results
    
    Example:
        >>> from finetune_cli.evaluation import benchmark_models
        >>> from finetune_cli.core.config import ConfigBuilder
        >>> 
        >>> result = benchmark_models(
        ...     base_model, finetuned_model, tokenizer, test_dataset,
        ...     eval_config,
        ...     save_report=Path("./benchmark_report.md")
        ... )
        >>> print(f"Average improvement: {result.get_average_improvement():.2f}%")
    """
    benchmarker = StandardBenchmarker()
    result = benchmarker.benchmark(
        base_model, finetuned_model, tokenizer, dataset, config
    )
    
    if save_report:
        ReportGenerator.save_report(
            result, save_report, format="markdown"
        )
    
    return result


def compare_metrics(
    base_metrics: Dict[str, float],
    finetuned_metrics: Dict[str, float]
) -> ComparisonResult:
    """
    Compare pre-computed metrics.
    
    Args:
        base_metrics: Base model metrics
        finetuned_metrics: Fine-tuned model metrics
    
    Returns:
        Comparison result
    
    Example:
        >>> base = {'rouge1': 0.25, 'rouge2': 0.15}
        >>> finetuned = {'rouge1': 0.35, 'rouge2': 0.22}
        >>> result = compare_metrics(base, finetuned)
        >>> print(result.improvements)
        {'rouge1': 40.0, 'rouge2': 46.67}
    """
    benchmarker = StandardBenchmarker()
    improvements = benchmarker._compute_improvements(base_metrics, finetuned_metrics)
    
    return ComparisonResult(
        base_metrics=base_metrics,
        finetuned_metrics=finetuned_metrics,
        improvements=improvements
    )