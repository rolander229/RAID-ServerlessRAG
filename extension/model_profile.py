"""
ModelProfile dataclass extracted to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from Ayo.engines.engine_types import EngineType


@dataclass
class ModelProfile:
    """Model characteristics"""

    model_name: str
    engine_type: EngineType
    hf_model_path: Optional[str] = None  # Hugging Face model path for estimation
    model_size: Optional[float] = None  # Model size (GB)
    param_count: Optional[float] = None  # In billions
    embedding_dim: Optional[int] = None  # For embedding models
    context_length: Optional[int] = None  # For LLMs
    estimated_memory_gb: Optional[float] = None  # Estimated memory requirement
    precision: str = "fp16"  # Model precision for estimation

    def dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "engine_type": self.engine_type,
            "model_size": self.model_size,
            "param_count": self.param_count,
            "embedding_dim": self.embedding_dim,
            "context_length": self.context_length,
            "estimated_memory_gb": self.estimated_memory_gb,
            "hf_model_path": self.hf_model_path if self.hf_model_path else self.model_name,
            "precision": self.precision,
        }
