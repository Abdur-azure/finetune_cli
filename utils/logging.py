"""
Logging infrastructure for the fine-tuning framework.

Provides structured logging with consistent formatting, log levels,
and context management.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..core.types import LogLevel


# ============================================================================
# LOGGER SETUP
# ============================================================================


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if sys.stdout.isatty():  # Only use colors in terminal
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{self.BOLD}{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level
        log_file: Optional file path for logging
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.value.upper()))
    logger.handlers.clear()  # Remove existing handlers
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.value.upper()))
    console_formatter = ColoredFormatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (no colors)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with default settings."""
    if name not in logging.root.manager.loggerDict:
        return setup_logger(name)
    return logging.getLogger(name)


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: logging.Logger, level: LogLevel):
        self.logger = logger
        self.new_level = getattr(logging, level.value.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


class LogProgress:
    """Context manager for logging operation progress."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} (took {duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} (after {duration:.2f}s)")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration"):
    """Log configuration dictionary in readable format."""
    logger.info(f"{title}:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")


def log_model_info(logger: logging.Logger, model):
    """Log model information (parameters, size, etc.)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    # Estimate memory usage (rough approximation)
    param_memory_gb = total_params * 4 / 1e9  # 4 bytes per param (float32)
    logger.info(f"  Estimated memory (FP32): {param_memory_gb:.2f} GB")


def log_dataset_info(logger: logging.Logger, dataset, name: str = "Dataset"):
    """Log dataset information."""
    logger.info(f"{name} Information:")
    logger.info(f"  Number of samples: {len(dataset):,}")
    if hasattr(dataset, 'column_names'):
        logger.info(f"  Columns: {', '.join(dataset.column_names)}")
    if hasattr(dataset, 'features'):
        logger.info(f"  Features: {list(dataset.features.keys())}")


def log_training_metrics(logger: logging.Logger, metrics: dict, step: int):
    """Log training metrics in consistent format."""
    metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    logger.info(f"Step {step} | {metric_str}")


def log_gpu_memory(logger: logging.Logger):
    """Log GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                logger.debug(
                    f"GPU {i} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )
    except Exception as e:
        logger.debug(f"Could not log GPU memory: {e}")


# ============================================================================
# DECORATOR
# ============================================================================


def log_execution(func):
    """Decorator to log function execution time and errors."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        with LogProgress(logger, func.__name__):
            return func(*args, **kwargs)
    return wrapper