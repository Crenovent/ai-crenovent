"""
ML Score Operator - Implements score primitive for ML model scoring
Task 4.1.1: Extend workflow DSL with ML primitives
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging
import json
import numpy as np
from datetime import datetime

from .base import BaseOperator, OperatorContext, OperatorResult

logger = logging.getLogger(__name__)

@dataclass
class ScoringResult:
    """Result of ML model scoring"""
    score: float
    score_range: tuple[float, float]  # (min, max)
    percentile: float  # Score percentile
    model_id: str
    model_version: str
    input_features: Dict[str, Any]
    feature_contributions: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    fallback_used: bool = False

class MLScoreOperator(BaseOperator):
    """
    ML Score operator for model scoring in workflows
    
    Supports:
    - Risk scoring (0-100)
    - Probability scoring (0-1)
    - Custom score ranges
    - Feature contribution analysis
    - Explainability (SHAP/LIME)
    - Fallback to deterministic scoring
    """
    
    def __init__(self, operator_id: str = "ml_score"):
        super().__init__(operator_id)
        self.requires_trust_check = True
        self.model_registry = {}  # Will be injected from Node Registry service
        
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate ML score operator configuration"""
        errors = []
        
        # Check required fields
        required_fields = ['model_id', 'input_data']
        for field in required_fields:
            if field not in config:
                errors.append(f"'{field}' is required")
        
        # Validate model_id exists in registry
        if 'model_id' in config:
            model_id = config['model_id']
            if model_id not in self.model_registry:
                errors.append(f"Model '{model_id}' not found in registry")
        
        # Validate input_data structure
        if 'input_data' in config:
            input_data = config['input_data']
            if not isinstance(input_data, dict):
                errors.append("'input_data' must be a dictionary")
        
        # Validate score range
        if 'score_range' in config:
            score_range = config['score_range']
            if not isinstance(score_range, (list, tuple)) or len(score_range) != 2:
                errors.append("'score_range' must be a list/tuple of [min, max]")
            elif not all(isinstance(x, (int, float)) for x in score_range):
                errors.append("'score_range' values must be numeric")
            elif score_range[0] >= score_range[1]:
                errors.append("'score_range' min must be less than max")
        
        # Validate threshold settings
        if 'thresholds' in config:
            thresholds = config['thresholds']
            if not isinstance(thresholds, dict):
                errors.append("'thresholds' must be a dictionary")
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute ML scoring"""
        try:
            model_id = config['model_id']
            input_data = config['input_data']
            score_range = config.get('score_range', [0, 100])
            thresholds = config.get('thresholds', {})
            explainability_enabled = config.get('explainability_enabled', True)
            fallback_enabled = config.get('fallback_enabled', True)
            
            # Get model configuration
            model_config = self.model_registry.get(model_id)
            if not model_config:
                return OperatorResult(
                    success=False,
                    error_message=f"Model '{model_id}' not found in registry"
                )
            
            # Validate input features
            validation_result = await self._validate_input_features(input_data, model_config)
            if not validation_result.success:
                return validation_result
            
            # Perform scoring
            scoring_result = await self._perform_scoring(
                model_id, input_data, model_config, score_range, explainability_enabled
            )
            
            # Check if fallback is needed (low confidence scenarios)
            if scoring_result.score < 0.1 and fallback_enabled:  # Very low score might indicate model failure
                fallback_result = await self._execute_fallback_scoring(
                    input_data, config, context, score_range
                )
                scoring_result.fallback_used = True
                scoring_result.score = fallback_result.get('score', 50.0)
                scoring_result.percentile = fallback_result.get('percentile', 50.0)
            
            # Apply thresholds for categorization
            threshold_categories = self._apply_thresholds(scoring_result.score, thresholds, score_range)
            
            # Prepare output data
            output_data = {
                'score': scoring_result.score,
                'score_range': scoring_result.score_range,
                'percentile': scoring_result.percentile,
                'model_id': scoring_result.model_id,
                'model_version': scoring_result.model_version,
                'input_features': scoring_result.input_features,
                'feature_contributions': scoring_result.feature_contributions,
                'explanation': scoring_result.explanation,
                'threshold_categories': threshold_categories,
                'fallback_used': scoring_result.fallback_used,
                'execution_time_ms': scoring_result.execution_time_ms
            }
            
            # Add scoring evidence
            evidence_data = {
                'scoring_type': 'ml_scoring',
                'model_id': model_id,
                'score': scoring_result.score,
                'percentile': scoring_result.percentile,
                'input_features': input_data,
                'threshold_categories': threshold_categories,
                'timestamp': datetime.utcnow().isoformat(),
                'tenant_id': context.tenant_id,
                'user_id': context.user_id
            }
            
            if scoring_result.explanation:
                evidence_data['explanation'] = scoring_result.explanation
            
            if scoring_result.feature_contributions:
                evidence_data['feature_contributions'] = scoring_result.feature_contributions
            
            if scoring_result.fallback_used:
                evidence_data['fallback_triggered'] = True
                evidence_data['fallback_reason'] = "Low confidence in ML scoring"
            
            return OperatorResult(
                success=True,
                output_data=output_data,
                confidence_score=min(scoring_result.score / 100.0, 1.0),  # Convert score to confidence
                evidence_data=evidence_data
            )
            
        except Exception as e:
            logger.error(f"ML Score execution failed: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Scoring failed: {e}",
                evidence_data={
                    'scoring_type': 'ml_scoring',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat(),
                    'tenant_id': context.tenant_id
                }
            )
    
    async def _validate_input_features(self, input_data: Dict[str, Any], model_config) -> OperatorResult:
        """Validate input features against model requirements"""
        try:
            required_features = model_config.input_features
            
            # Check for missing features
            missing_features = [f for f in required_features if f not in input_data]
            if missing_features:
                return OperatorResult(
                    success=False,
                    error_message=f"Missing required features: {missing_features}"
                )
            
            return OperatorResult(success=True)
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Input validation failed: {e}"
            )
    
    async def _perform_scoring(
        self, 
        model_id: str, 
        input_data: Dict[str, Any], 
        model_config,
        score_range: List[float],
        explainability_enabled: bool
    ) -> ScoringResult:
        """Perform the actual ML scoring"""
        start_time = time.time()
        
        try:
            # Simulate model scoring
            score, percentile = await self._simulate_model_scoring(input_data, model_config, score_range)
            
            # Generate feature contributions if explainability enabled
            feature_contributions = None
            explanation = None
            if explainability_enabled:
                feature_contributions, explanation = await self._generate_scoring_explainability(
                    input_data, score, model_config
                )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return ScoringResult(
                score=score,
                score_range=tuple(score_range),
                percentile=percentile,
                model_id=model_id,
                model_version=model_config.model_version,
                input_features=input_data,
                feature_contributions=feature_contributions,
                explanation=explanation,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Model scoring failed: {e}")
            raise
    
    async def _simulate_model_scoring(
        self, 
        input_data: Dict[str, Any], 
        model_config,
        score_range: List[float]
    ) -> tuple[float, float]:
        """Simulate ML model scoring (placeholder for actual model integration)"""
        # This is a simulation - in real implementation, this would call actual ML models
        
        min_score, max_score = score_range
        
        # Simulate scoring based on input characteristics
        base_score = 50.0  # Start with middle score
        
        # Adjust score based on input features
        for feature, value in input_data.items():
            if isinstance(value, (int, float)):
                # Numeric features contribute to score
                if 'risk' in feature.lower() or 'danger' in feature.lower():
                    base_score += value * 0.1
                elif 'positive' in feature.lower() or 'good' in feature.lower():
                    base_score -= value * 0.05
                else:
                    base_score += value * 0.02
            elif isinstance(value, str):
                # String features contribute based on length and content
                if 'high' in value.lower() or 'critical' in value.lower():
                    base_score += 20
                elif 'low' in value.lower() or 'safe' in value.lower():
                    base_score -= 15
                else:
                    base_score += len(value) * 0.5
        
        # Normalize score to range
        score = max(min_score, min(max_score, base_score))
        
        # Calculate percentile (simplified)
        percentile = ((score - min_score) / (max_score - min_score)) * 100
        
        return score, percentile
    
    async def _generate_scoring_explainability(
        self, 
        input_data: Dict[str, Any], 
        score: float, 
        model_config
    ) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        """Generate explainability data for scoring (SHAP/LIME simulation)"""
        try:
            # Simulate feature contributions
            feature_contributions = {}
            total_contribution = 0
            
            for feature, value in input_data.items():
                if isinstance(value, (int, float)):
                    contribution = abs(value) * 0.1
                elif isinstance(value, str):
                    contribution = len(value) * 0.2
                else:
                    contribution = 0.1
                
                feature_contributions[feature] = contribution
                total_contribution += contribution
            
            # Normalize contributions
            if total_contribution > 0:
                feature_contributions = {
                    k: v / total_contribution * 100 
                    for k, v in feature_contributions.items()
                }
            
            # Generate explanation
            top_contributors = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)[:3]
            explanation = {
                'method': 'simulated_shap',
                'score': score,
                'top_contributors': [{'feature': k, 'contribution': v} for k, v in top_contributors],
                'reasoning': f"Score of {score:.1f} primarily driven by {', '.join([k for k, v in top_contributors])}",
                'confidence': 0.85
            }
            
            return feature_contributions, explanation
            
        except Exception as e:
            logger.warning(f"Scoring explainability generation failed: {e}")
            return None, None
    
    def _apply_thresholds(self, score: float, thresholds: Dict[str, Any], score_range: List[float]) -> Dict[str, Any]:
        """Apply threshold-based categorization to score"""
        categories = {}
        
        # Default thresholds if none provided
        if not thresholds:
            min_score, max_score = score_range
            mid_score = (min_score + max_score) / 2
            thresholds = {
                'low': max_score * 0.3,
                'medium': mid_score,
                'high': max_score * 0.8
            }
        
        # Apply thresholds
        for category, threshold in thresholds.items():
            if score >= threshold:
                categories[category] = True
            else:
                categories[category] = False
        
        # Determine primary category
        if score >= thresholds.get('high', score_range[1] * 0.8):
            categories['primary_category'] = 'high'
        elif score >= thresholds.get('medium', (score_range[0] + score_range[1]) / 2):
            categories['primary_category'] = 'medium'
        else:
            categories['primary_category'] = 'low'
        
        return categories
    
    async def _execute_fallback_scoring(
        self, 
        input_data: Dict[str, Any], 
        config: Dict[str, Any], 
        context: OperatorContext,
        score_range: List[float]
    ) -> Dict[str, Any]:
        """Execute fallback deterministic scoring when ML confidence is low"""
        try:
            # Simple rule-based scoring fallback
            fallback_rules = config.get('fallback_rules', {})
            
            # Calculate score based on simple rules
            score = 50.0  # Default middle score
            min_score, max_score = score_range
            
            if 'field_based_scoring' in fallback_rules:
                field = fallback_rules['field_based_scoring'].get('field', 'value')
                if field in input_data:
                    value = input_data[field]
                    if isinstance(value, (int, float)):
                        score = min(max_score, max(min_score, value))
            
            # Normalize to range
            score = max(min_score, min(max_score, score))
            percentile = ((score - min_score) / (max_score - min_score)) * 100
            
            return {
                'score': score,
                'percentile': percentile,
                'method': 'fallback_rule'
            }
            
        except Exception as e:
            logger.error(f"Fallback scoring failed: {e}")
            return {
                'score': 50.0,
                'percentile': 50.0,
                'method': 'error_fallback'
            }
    
    def set_model_registry(self, registry: Dict[str, Any]):
        """Set the model registry (injected by Node Registry service)"""
        self.model_registry = registry
