"""
ML Predict Operator - Implements predict primitive for ML model inference
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
from .override_service import OverrideService

logger = logging.getLogger(__name__)

@dataclass
class MLModelConfig:
    """Configuration for ML model used in prediction"""
    model_id: str
    model_type: str  # 'classification', 'regression', 'anomaly_detection'
    model_version: str
    input_features: List[str]
    output_schema: Dict[str, Any]
    confidence_threshold: float = 0.7
    fallback_enabled: bool = True
    explainability_enabled: bool = True

@dataclass
class PredictionResult:
    """Result of ML model prediction"""
    prediction: Any
    confidence: float
    model_id: str
    model_version: str
    input_features: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    fallback_used: bool = False

class MLPredictOperator(BaseOperator):
    """
    ML Predict operator for model inference in workflows
    
    Supports:
    - Classification predictions
    - Regression predictions  
    - Anomaly detection
    - Confidence scoring
    - Explainability (SHAP/LIME)
    - Fallback to deterministic rules
    """
    
    def __init__(self, operator_id: str = "ml_predict"):
        super().__init__(operator_id)
        self.requires_trust_check = True
        self.model_registry = {}  # Will be injected from Node Registry service
        self.override_service = OverrideService()
        
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate ML predict operator configuration"""
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
        
        # Validate confidence threshold
        if 'confidence_threshold' in config:
            threshold = config['confidence_threshold']
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                errors.append("'confidence_threshold' must be between 0 and 1")
        
        # Validate explainability settings
        if 'explainability_enabled' in config:
            if not isinstance(config['explainability_enabled'], bool):
                errors.append("'explainability_enabled' must be boolean")
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute ML prediction"""
        try:
            model_id = config['model_id']
            input_data = config['input_data']
            confidence_threshold = config.get('confidence_threshold', 0.7)
            explainability_enabled = config.get('explainability_enabled', True)
            fallback_enabled = config.get('fallback_enabled', True)
            
            # Get model configuration
            model_config = self.model_registry.get(model_id)
            if not model_config:
                return OperatorResult(
                    success=False,
                    error_message=f"Model '{model_id}' not found in registry"
                )
            
            # Check for override requests
            override_requests = await self.override_service.get_pending_overrides(
                tenant_id=context.tenant_id,
                limit=10
            )
            
            # Check if there's an override for this specific prediction
            for override_request in override_requests:
                if (override_request.workflow_id == context.workflow_id and 
                    override_request.step_id == context.step_id and
                    override_request.model_id == model_id):
                    
                    # Use override prediction
                    prediction_result = override_request.override_prediction
                    confidence_score = 0.9  # High confidence for manual overrides
                    
                    # Log override usage
                    logger.info(f"Using override prediction for {model_id}: {prediction_result}")
                    
                    return OperatorResult(
                        success=True,
                        output_data=prediction_result,
                        confidence_score=confidence_score,
                        evidence_data={
                            'model_id': model_id,
                            'model_input': input_data,
                            'prediction_result': prediction_result,
                            'override_used': True,
                            'override_id': override_request.override_id,
                            'override_justification': override_request.justification
                        }
                    )
            
            # Validate input features
            validation_result = await self._validate_input_features(input_data, model_config)
            if not validation_result.success:
                return validation_result
            
            # Perform prediction
            prediction_result = await self._perform_prediction(
                model_id, input_data, model_config, explainability_enabled
            )
            
            # Check confidence threshold
            if prediction_result.confidence < confidence_threshold:
                if fallback_enabled:
                    # Use fallback deterministic rule
                    fallback_result = await self._execute_fallback_rule(
                        input_data, config, context
                    )
                    prediction_result.fallback_used = True
                    prediction_result.prediction = fallback_result.get('prediction')
                    prediction_result.confidence = fallback_result.get('confidence', 0.5)
                else:
                    return OperatorResult(
                        success=False,
                        error_message=f"Prediction confidence {prediction_result.confidence:.3f} below threshold {confidence_threshold}"
                    )
            
            # Prepare output data
            output_data = {
                'prediction': prediction_result.prediction,
                'confidence': prediction_result.confidence,
                'model_id': prediction_result.model_id,
                'model_version': prediction_result.model_version,
                'input_features': prediction_result.input_features,
                'feature_importance': prediction_result.feature_importance,
                'explanation': prediction_result.explanation,
                'fallback_used': prediction_result.fallback_used,
                'execution_time_ms': prediction_result.execution_time_ms
            }
            
            # Add explainability evidence
            evidence_data = {
                'prediction_type': 'ml_prediction',
                'model_id': model_id,
                'confidence_score': prediction_result.confidence,
                'input_features': input_data,
                'prediction_result': prediction_result.prediction,
                'timestamp': datetime.utcnow().isoformat(),
                'tenant_id': context.tenant_id,
                'user_id': context.user_id
            }
            
            if prediction_result.explanation:
                evidence_data['explanation'] = prediction_result.explanation
            
            if prediction_result.feature_importance:
                evidence_data['feature_importance'] = prediction_result.feature_importance
            
            if prediction_result.fallback_used:
                evidence_data['fallback_triggered'] = True
                evidence_data['fallback_reason'] = f"Confidence {prediction_result.confidence:.3f} below threshold {confidence_threshold}"
            
            return OperatorResult(
                success=True,
                output_data=output_data,
                confidence_score=prediction_result.confidence,
                evidence_data=evidence_data
            )
            
        except Exception as e:
            logger.error(f"ML Predict execution failed: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Prediction failed: {e}",
                evidence_data={
                    'prediction_type': 'ml_prediction',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat(),
                    'tenant_id': context.tenant_id
                }
            )
    
    async def _validate_input_features(self, input_data: Dict[str, Any], model_config: MLModelConfig) -> OperatorResult:
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
            
            # Check for extra features (warn but don't fail)
            extra_features = [f for f in input_data.keys() if f not in required_features]
            if extra_features:
                logger.warning(f"Extra features provided: {extra_features}")
            
            return OperatorResult(success=True)
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Input validation failed: {e}"
            )
    
    async def _perform_prediction(
        self, 
        model_id: str, 
        input_data: Dict[str, Any], 
        model_config: MLModelConfig,
        explainability_enabled: bool
    ) -> PredictionResult:
        """Perform the actual ML prediction"""
        start_time = time.time()
        
        try:
            # This is where we would integrate with actual ML models
            # For now, we'll simulate the prediction process
            
            # Simulate model inference
            prediction, confidence = await self._simulate_model_inference(input_data, model_config)
            
            # Generate explainability if enabled
            explanation = None
            feature_importance = None
            if explainability_enabled:
                explanation, feature_importance = await self._generate_explainability(
                    input_data, prediction, model_config
                )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return PredictionResult(
                prediction=prediction,
                confidence=confidence,
                model_id=model_id,
                model_version=model_config.model_version,
                input_features=input_data,
                feature_importance=feature_importance,
                explanation=explanation,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise
    
    async def _simulate_model_inference(self, input_data: Dict[str, Any], model_config: MLModelConfig) -> tuple[Any, float]:
        """Simulate ML model inference (placeholder for actual model integration)"""
        # This is a simulation - in real implementation, this would call actual ML models
        
        # Simulate different model types
        if model_config.model_type == 'classification':
            # Simulate classification (e.g., churn prediction)
            prediction = 'high_risk' if len(input_data) > 5 else 'low_risk'
            confidence = 0.85 if prediction == 'high_risk' else 0.75
            
        elif model_config.model_type == 'regression':
            # Simulate regression (e.g., revenue prediction)
            prediction = sum(len(str(v)) for v in input_data.values()) * 1000
            confidence = 0.80
            
        elif model_config.model_type == 'anomaly_detection':
            # Simulate anomaly detection
            prediction = 'anomaly' if any(isinstance(v, str) and 'test' in str(v).lower() for v in input_data.values()) else 'normal'
            confidence = 0.90 if prediction == 'anomaly' else 0.70
            
        else:
            prediction = 'unknown'
            confidence = 0.5
        
        return prediction, confidence
    
    async def _generate_explainability(
        self, 
        input_data: Dict[str, Any], 
        prediction: Any, 
        model_config: MLModelConfig
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, float]]]:
        """Generate explainability data (SHAP/LIME simulation)"""
        try:
            # Simulate SHAP/LIME explainability
            # In real implementation, this would call actual explainability libraries
            
            feature_importance = {}
            for feature, value in input_data.items():
                # Simulate feature importance based on value characteristics
                if isinstance(value, (int, float)):
                    importance = abs(value) / 100.0
                elif isinstance(value, str):
                    importance = len(value) / 50.0
                else:
                    importance = 0.1
                
                feature_importance[feature] = min(importance, 1.0)
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            # Generate explanation text
            top_features = list(feature_importance.keys())[:3]
            explanation = {
                'method': 'simulated_shap',
                'prediction': str(prediction),
                'top_features': top_features,
                'reasoning': f"Prediction based on {', '.join(top_features)} with importance scores {[feature_importance[f] for f in top_features]}",
                'confidence': 0.85
            }
            
            return explanation, feature_importance
            
        except Exception as e:
            logger.warning(f"Explainability generation failed: {e}")
            return None, None
    
    async def _execute_fallback_rule(
        self, 
        input_data: Dict[str, Any], 
        config: Dict[str, Any], 
        context: OperatorContext
    ) -> Dict[str, Any]:
        """Execute fallback deterministic rule when ML confidence is low"""
        try:
            # This would integrate with existing RBA rules
            # For now, we'll simulate a simple fallback
            
            fallback_rules = config.get('fallback_rules', {})
            
            # Simple rule-based fallback
            if 'threshold_rule' in fallback_rules:
                threshold = fallback_rules['threshold_rule'].get('threshold', 0.5)
                field = fallback_rules['threshold_rule'].get('field', 'value')
                
                if field in input_data:
                    value = input_data[field]
                    if isinstance(value, (int, float)) and value > threshold:
                        return {
                            'prediction': 'high_risk',
                            'confidence': 0.6,
                            'method': 'fallback_rule'
                        }
            
            # Default fallback
            return {
                'prediction': 'medium_risk',
                'confidence': 0.5,
                'method': 'default_fallback'
            }
            
        except Exception as e:
            logger.error(f"Fallback rule execution failed: {e}")
            return {
                'prediction': 'unknown',
                'confidence': 0.3,
                'method': 'error_fallback'
            }
    
    def set_model_registry(self, registry: Dict[str, MLModelConfig]):
        """Set the model registry (injected by Node Registry service)"""
        self.model_registry = registry
