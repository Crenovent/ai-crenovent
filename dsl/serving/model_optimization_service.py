"""
Task 6.3.16: Quantization/pruning paths (ONNX/TensorRT)
Model optimization service for quantization and pruning
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Model optimization types"""
    QUANTIZATION_INT8 = "quantization_int8"
    QUANTIZATION_FP16 = "quantization_fp16"
    PRUNING_STRUCTURED = "pruning_structured"
    PRUNING_UNSTRUCTURED = "pruning_unstructured"
    DISTILLATION = "distillation"

@dataclass
class OptimizationConfig:
    """Model optimization configuration"""
    model_id: str
    optimization_type: OptimizationType
    target_format: str  # "onnx", "tensorrt", "pytorch"
    compression_ratio: float = 0.5
    accuracy_threshold: float = 0.95
    optimization_params: Dict[str, Any] = None

@dataclass
class OptimizationResult:
    """Model optimization result"""
    original_model_id: str
    optimized_model_id: str
    optimization_type: OptimizationType
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    accuracy_retained: float
    latency_improvement: float
    created_at: datetime

class ModelOptimizationService:
    """
    Model optimization service for quantization and pruning
    Task 6.3.16: Lower latency/cost with model variants
    """
    
    def __init__(self):
        self.optimization_results: Dict[str, OptimizationResult] = {}
        self.optimization_configs: Dict[str, OptimizationConfig] = {}
    
    def register_optimization_config(self, config: OptimizationConfig) -> bool:
        """Register optimization configuration"""
        try:
            self.optimization_configs[config.model_id] = config
            logger.info(f"Registered optimization config for model: {config.model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register optimization config: {e}")
            return False
    
    async def optimize_model(self, model_id: str, optimization_type: OptimizationType) -> OptimizationResult:
        """Optimize model with specified technique"""
        try:
            config = self.optimization_configs.get(model_id)
            if not config:
                config = OptimizationConfig(
                    model_id=model_id,
                    optimization_type=optimization_type,
                    target_format="onnx"
                )
            
            # Simulate model optimization
            result = await self._perform_optimization(config)
            
            # Store result
            optimized_id = f"{model_id}_optimized_{optimization_type.value}"
            self.optimization_results[optimized_id] = result
            
            logger.info(f"Optimized model {model_id} -> {optimized_id}")
            return result
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise
    
    async def _perform_optimization(self, config: OptimizationConfig) -> OptimizationResult:
        """Perform the actual optimization"""
        # Simulate optimization process
        original_size = 100.0  # MB
        
        if config.optimization_type == OptimizationType.QUANTIZATION_INT8:
            optimized_size = original_size * 0.25  # 4x compression
            accuracy_retained = 0.98
            latency_improvement = 2.5
        elif config.optimization_type == OptimizationType.QUANTIZATION_FP16:
            optimized_size = original_size * 0.5   # 2x compression
            accuracy_retained = 0.995
            latency_improvement = 1.8
        elif config.optimization_type == OptimizationType.PRUNING_STRUCTURED:
            optimized_size = original_size * 0.6   # 40% size reduction
            accuracy_retained = 0.96
            latency_improvement = 1.5
        elif config.optimization_type == OptimizationType.PRUNING_UNSTRUCTURED:
            optimized_size = original_size * 0.4   # 60% size reduction
            accuracy_retained = 0.93
            latency_improvement = 2.0
        else:  # DISTILLATION
            optimized_size = original_size * 0.3   # 70% size reduction
            accuracy_retained = 0.92
            latency_improvement = 3.0
        
        return OptimizationResult(
            original_model_id=config.model_id,
            optimized_model_id=f"{config.model_id}_optimized_{config.optimization_type.value}",
            optimization_type=config.optimization_type,
            original_size_mb=original_size,
            optimized_size_mb=optimized_size,
            compression_ratio=optimized_size / original_size,
            accuracy_retained=accuracy_retained,
            latency_improvement=latency_improvement,
            created_at=datetime.utcnow()
        )
    
    def get_optimization_variants(self, model_id: str) -> List[OptimizationResult]:
        """Get all optimization variants for a model"""
        variants = []
        for result in self.optimization_results.values():
            if result.original_model_id == model_id:
                variants.append(result)
        return variants
    
    def select_optimal_variant(self, model_id: str, latency_priority: float = 0.5) -> Optional[OptimizationResult]:
        """Select optimal variant based on latency vs accuracy trade-off"""
        variants = self.get_optimization_variants(model_id)
        if not variants:
            return None
        
        # Score variants based on weighted latency and accuracy
        best_variant = None
        best_score = -1
        
        for variant in variants:
            # Normalize scores (0-1)
            latency_score = min(variant.latency_improvement / 5.0, 1.0)  # Max 5x improvement
            accuracy_score = variant.accuracy_retained
            
            # Weighted score
            score = (latency_priority * latency_score) + ((1 - latency_priority) * accuracy_score)
            
            if score > best_score:
                best_score = score
                best_variant = variant
        
        return best_variant
    
    def get_optimization_report(self, model_id: str) -> Dict[str, Any]:
        """Generate optimization report for a model"""
        variants = self.get_optimization_variants(model_id)
        
        if not variants:
            return {"model_id": model_id, "variants": [], "summary": "No optimizations performed"}
        
        total_size_reduction = sum(
            (v.original_size_mb - v.optimized_size_mb) for v in variants
        )
        
        avg_accuracy = sum(v.accuracy_retained for v in variants) / len(variants)
        avg_latency_improvement = sum(v.latency_improvement for v in variants) / len(variants)
        
        return {
            "model_id": model_id,
            "total_variants": len(variants),
            "total_size_reduction_mb": total_size_reduction,
            "avg_accuracy_retained": avg_accuracy,
            "avg_latency_improvement": avg_latency_improvement,
            "variants": [
                {
                    "optimized_model_id": v.optimized_model_id,
                    "type": v.optimization_type.value,
                    "compression_ratio": v.compression_ratio,
                    "accuracy_retained": v.accuracy_retained,
                    "latency_improvement": v.latency_improvement
                }
                for v in variants
            ]
        }

# Global model optimization service
model_optimization_service = ModelOptimizationService()
