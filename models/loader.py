"""
Model loading system with support for various configurations.

Handles loading models from HuggingFace Hub with quantization,
device mapping, and architecture-specific optimizations.
"""

from typing import Optional, Tuple
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)

from ..core.types import ModelConfig, DeviceType
from ..core.exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedModelError,
    CUDANotAvailableError
)
from ..utils.logging import get_logger, log_model_info, LogProgress


logger = get_logger(__name__)


# ============================================================================
# MODEL LOADER
# ============================================================================


class ModelLoader:
    """
    Handles model and tokenizer loading with various configurations.
    
    Supports:
    - Device mapping (CPU, CUDA, auto)
    - Quantization (4-bit, 8-bit)
    - Flash Attention
    - Custom dtype selection
    """
    
    def __init__(self):
        self._validate_environment()
    
    def _validate_environment(self) -> None:
        """Validate environment for model loading."""
        logger.debug("Validating environment...")
        logger.debug(f"PyTorch version: {torch.__version__}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.debug(f"CUDA version: {torch.version.cuda}")
            logger.debug(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.debug(
                    f"GPU {i}: {props.name} "
                    f"({props.total_memory / 1e9:.1f}GB)"
                )
    
    def load(self, config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer based on configuration.
        
        Args:
            config: Model configuration
        
        Returns:
            Tuple of (model, tokenizer)
        
        Raises:
            ModelLoadError: If loading fails
            ModelNotFoundError: If model doesn't exist
            CUDANotAvailableError: If CUDA required but not available
        """
        with LogProgress(logger, f"Loading model '{config.name}'"):
            try:
                # Load tokenizer first
                tokenizer = self._load_tokenizer(config)
                
                # Load model with appropriate settings
                model = self._load_model(config)
                
                # Log model information
                log_model_info(logger, model)
                
                return model, tokenizer
                
            except FileNotFoundError as e:
                raise ModelNotFoundError(config.name)
            except Exception as e:
                raise ModelLoadError(config.name, str(e))
    
    def _load_tokenizer(self, config: ModelConfig) -> PreTrainedTokenizer:
        """Load tokenizer with configuration."""
        logger.debug(f"Loading tokenizer: {config.name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
            cache_dir=config.cache_dir
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug("Set pad_token to eos_token")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.debug("Added [PAD] token")
        
        logger.debug(f"Tokenizer vocab size: {len(tokenizer)}")
        return tokenizer
    
    def _load_model(self, config: ModelConfig) -> PreTrainedModel:
        """Load model with configuration."""
        logger.debug(f"Loading model: {config.name}")
        
        # Determine device
        device = self._resolve_device(config.device)
        logger.info(f"Target device: {device}")
        
        # Build loading kwargs
        load_kwargs = self._build_load_kwargs(config, device)
        
        # Log configuration
        logger.debug(f"Load kwargs: {load_kwargs}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.name,
            **load_kwargs
        )
        
        # Apply model-specific optimizations
        model = self._apply_optimizations(model, config)
        
        return model
    
    def _resolve_device(self, device: DeviceType) -> str:
        """Resolve device type to actual device string."""
        if device == DeviceType.AUTO:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        elif device == DeviceType.CUDA:
            if not torch.cuda.is_available():
                raise CUDANotAvailableError()
            return "cuda"
        
        else:
            return device.value
    
    def _build_load_kwargs(self, config: ModelConfig, device: str) -> dict:
        """Build keyword arguments for model loading."""
        kwargs = {
            'trust_remote_code': config.trust_remote_code,
            'revision': config.revision,
            'cache_dir': config.cache_dir,
            'low_cpu_mem_usage': True  # Always use efficient loading
        }
        
        # Quantization configuration
        if config.load_in_4bit or config.load_in_8bit:
            kwargs['quantization_config'] = self._build_quantization_config(config)
            kwargs['device_map'] = 'auto'  # Required for quantization
        else:
            # Device mapping
            if device == "cuda":
                kwargs['device_map'] = 'auto'
            else:
                kwargs['device_map'] = None
            
            # Dtype
            if config.torch_dtype is not None:
                kwargs['torch_dtype'] = config.torch_dtype
            elif device == "cuda":
                # Default to float16 on GPU if not specified
                kwargs['torch_dtype'] = torch.float16
            else:
                kwargs['torch_dtype'] = torch.float32
        
        # Flash Attention
        if config.use_flash_attention:
            kwargs['attn_implementation'] = 'flash_attention_2'
            logger.info("Enabled Flash Attention 2")
        
        return kwargs
    
    def _build_quantization_config(self, config: ModelConfig) -> BitsAndBytesConfig:
        """Build quantization configuration."""
        if config.load_in_4bit:
            logger.info("Loading with 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        elif config.load_in_8bit:
            logger.info("Loading with 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        return None
    
    def _apply_optimizations(self, model: PreTrainedModel, 
                            config: ModelConfig) -> PreTrainedModel:
        """Apply model-specific optimizations."""
        
        # Gradient checkpointing (will be enabled during training if needed)
        if hasattr(model, 'supports_gradient_checkpointing'):
            if model.supports_gradient_checkpointing:
                logger.debug("Model supports gradient checkpointing")
        
        # Set to training mode by default
        model.train()
        
        return model


# ============================================================================
# TARGET MODULE DETECTION
# ============================================================================


class TargetModuleDetector:
    """
    Automatically detect target modules for LoRA based on model architecture.
    """
    
    # Common patterns for different architectures
    PATTERNS = [
        # Standard transformer attention
        ["q_proj", "v_proj", "k_proj", "o_proj"],
        ["query", "value", "key", "dense"],
        ["q_lin", "v_lin", "k_lin", "out_lin"],
        
        # GPT-2 style
        ["c_attn", "c_proj"],
        
        # Other variants
        ["qkv_proj", "out_proj"],
        ["Wqkv", "out_proj"],
        
        # MLP layers (for more comprehensive tuning)
        ["fc1", "fc2"],
        ["up_proj", "down_proj", "gate_proj"],
        ["mlp.c_fc", "mlp.c_proj"],
    ]
    
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.logger = get_logger(__name__)
    
    def detect(self) -> list[str]:
        """
        Detect optimal target modules for the model.
        
        Returns:
            List of module names to target
        
        Raises:
            TargetModulesNotFoundError: If no suitable modules found
        """
        self.logger.debug("Detecting target modules...")
        
        # Get all module names
        module_names = self._get_module_names()
        self.logger.debug(f"Found {len(module_names)} unique module names")
        
        # Try each pattern
        for pattern in self.PATTERNS:
            matched = self._match_pattern(pattern, module_names)
            if matched:
                self.logger.info(f"Detected target modules: {matched}")
                return matched
        
        # Fallback: find any linear layers
        linear_modules = self._find_linear_modules(module_names)
        if linear_modules:
            self.logger.warning(
                f"Using fallback linear modules: {linear_modules[:4]}"
            )
            return linear_modules[:4]
        
        # Last resort
        from ..core.exceptions import TargetModulesNotFoundError
        raise TargetModulesNotFoundError(
            self.model.config._name_or_path,
            [str(p) for p in self.PATTERNS]
        )
    
    def _get_module_names(self) -> set[str]:
        """Get all leaf module names from the model."""
        names = set()
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                # Extract just the module type name
                module_name = name.split('.')[-1]
                names.add(module_name)
        return names
    
    def _match_pattern(self, pattern: list[str], 
                       module_names: set[str]) -> Optional[list[str]]:
        """Check if pattern matches available modules."""
        # Need at least 2 modules from pattern to match
        matched = [name for name in pattern if name in module_names]
        if len(matched) >= 2:
            return matched
        return None
    
    def _find_linear_modules(self, module_names: set[str]) -> list[str]:
        """Find modules that are likely linear layers."""
        keywords = ['lin', 'proj', 'fc', 'dense', 'query', 'key', 'value']
        candidates = []
        
        for name in module_names:
            name_lower = name.lower()
            if any(kw in name_lower for kw in keywords):
                candidates.append(name)
        
        return sorted(candidates)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def load_model_and_tokenizer(
    config: ModelConfig
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Convenience function to load model and tokenizer.
    
    Args:
        config: Model configuration
    
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader()
    return loader.load(config)


def detect_target_modules(model: PreTrainedModel) -> list[str]:
    """
    Convenience function to detect target modules.
    
    Args:
        model: Model to analyze
    
    Returns:
        List of target module names
    """
    detector = TargetModuleDetector(model)
    return detector.detect()