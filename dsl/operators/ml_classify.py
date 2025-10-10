"""
ML Classify Operator - Implements classify primitive for ML model classification
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
class ClassificationResult:
    """Result of ML model classification"""
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    model_id: str
    model_version: str
    input_features: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    fallback_used: bool = False

class MLClassifyOperator(BaseOperator):
    """
    ML Classify operator for model classification in workflows
    
    Supports:
    - Multi-class classification
    - Binary classification
    - Class probability distributions
    - Confidence scoring per class
    - Explainability (SHAP/LIME)
    - Fallback to deterministic classification
    """
    
    def __init__(self, operator_id: str = "ml_classify"):
        super().__init__(operator_id)
        self.requires_trust_check = True
        self.model_registry = {}  # Will be injected from Node Registry service
        
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate ML classify operator configuration"""
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
        
        # Validate class mapping
        if 'class_mapping' in config:
            class_mapping = config['class_mapping']
            if not isinstance(class_mapping, dict):
                errors.append("'class_mapping' must be a dictionary")
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute ML classification"""
        try:
            model_id = config['model_id']
            input_data = config['input_data']
            confidence_threshold = config.get('confidence_threshold', 0.7)
            class_mapping = config.get('class_mapping', {})
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
            
            # Perform classification
            classification_result = await self._perform_classification(
                model_id, input_data, model_config, class_mapping, explainability_enabled
            )
            
            # Check confidence threshold
            if classification_result.confidence < confidence_threshold:
                if fallback_enabled:
                    # Use fallback deterministic classification
                    fallback_result = await self._execute_fallback_classification(
                        input_data, config, context, class_mapping
                    )
                    classification_result.fallback_used = True
                    classification_result.predicted_class = fallback_result.get('predicted_class', 'unknown')
                    classification_result.confidence = fallback_result.get('confidence', 0.5)
                    classification_result.class_probabilities = fallback_result.get('class_probabilities', {})
                else:
                    return OperatorResult(
                        success=False,
                        error_message=f"Classification confidence {classification_result.confidence:.3f} below threshold {confidence_threshold}"
                    )
            
            # Prepare output data
            output_data = {
                'predicted_class': classification_result.predicted_class,
                'confidence': classification_result.confidence,
                'class_probabilities': classification_result.class_probabilities,
                'model_id': classification_result.model_id,
                'model_version': classification_result.model_version,
                'input_features': classification_result.input_features,
                'feature_importance': classification_result.feature_importance,
                'explanation': classification_result.explanation,
                'fallback_used': classification_result.fallback_used,
                'execution_time_ms': classification_result.execution_time_ms
            }
            
            # Add classification evidence
            evidence_data = {
                'classification_type': 'ml_classification',
                'model_id': model_id,
                'predicted_class': classification_result.predicted_class,
                'confidence': classification_result.confidence,
                'class_probabilities': classification_result.class_probabilities,
                'input_features': input_data,
                'timestamp': datetime.utcnow().isoformat(),
                'tenant_id': context.tenant_id,
                'user_id': context.user_id
            }
            
            if classification_result.explanation:
                evidence_data['explanation'] = classification_result.explanation
            
            if classification_result.feature_importance:
                evidence_data['feature_importance'] = classification_result.feature_importance
            
            if classification_result.fallback_used:
                evidence_data['fallback_triggered'] = True
                evidence_data['fallback_reason'] = f"Confidence {classification_result.confidence:.3f} below threshold {confidence_threshold}"
            
            return OperatorResult(
                success=True,
                output_data=output_data,
                confidence_score=classification_result.confidence,
                evidence_data=evidence_data
            )
            
        except Exception as e:
            logger.error(f"ML Classify execution failed: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Classification failed: {e}",
                evidence_data={
                    'classification_type': 'ml_classification',
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
    
    async def _perform_classification(
        self, 
        model_id: str, 
        input_data: Dict[str, Any], 
        model_config,
        class_mapping: Dict[str, str],
        explainability_enabled: bool
    ) -> ClassificationResult:
        """Perform the actual ML classification"""
        start_time = time.time()
        
        try:
            # Simulate model classification
            predicted_class, confidence, class_probabilities = await self._simulate_model_classification(
                input_data, model_config, class_mapping
            )
            
            # Generate explainability if enabled
            feature_importance = None
            explanation = None
            if explainability_enabled:
                feature_importance, explanation = await self._generate_classification_explainability(
                    input_data, predicted_class, class_probabilities, model_config
                )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return ClassificationResult(
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probabilities,
                model_id=model_id,
                model_version=model_config.model_version,
                input_features=input_data,
                feature_importance=feature_importance,
                explanation=explanation,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Model classification failed: {e}")
            raise
    
    async def _simulate_model_classification(
        self, 
        input_data: Dict[str, Any], 
        model_config,
        class_mapping: Dict[str, str]
    ) -> tuple[str, float, Dict[str, float]]:
        """Simulate ML model classification (placeholder for actual model integration)"""
        # This is a simulation - in real implementation, this would call actual ML models
        
        # Define default classes if none provided
        default_classes = ['low_risk', 'medium_risk', 'high_risk']
        classes = list(class_mapping.keys()) if class_mapping else default_classes
        
        # Simulate classification based on input characteristics
        risk_score = 0.0
        
        for feature, value in input_data.items():
            if isinstance(value, (int, float)):
                if 'risk' in feature.lower() or 'danger' in feature.lower():
                    risk_score += value * 0.1
                elif 'positive' in feature.lower() or 'good' in feature.lower():
                    risk_score -= value * 0.05
                else:
                    risk_score += value * 0.02
            elif isinstance(value, str):
                if 'high' in value.lower() or 'critical' in value.lower():
                    risk_score += 0.3
                elif 'low' in value.lower() or 'safe' in value.lower():
                    risk_score -= 0.2
                else:
                    risk_score += len(value) * 0.01
        
        # Normalize risk score to 0-1 range
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Determine predicted class based on risk score
        if len(classes) == 2:  # Binary classification
            if risk_score > 0.5:
                predicted_class = classes[1]  # High risk
                confidence = risk_score
            else:
                predicted_class = classes[0]  # Low risk
                confidence = 1.0 - risk_score
        else:  # Multi-class classification
            if risk_score > 0.7:
                predicted_class = classes[-1]  # High risk
                confidence = risk_score
            elif risk_score > 0.3:
                predicted_class = classes[1] if len(classes) > 1 else classes[0]  # Medium risk
                confidence = 0.8
            else:
                predicted_class = classes[0]  # Low risk
                confidence = 1.0 - risk_score
        
        # Generate class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(classes):
            if class_name == predicted_class:
                class_probabilities[class_name] = confidence
            else:
                # Distribute remaining probability among other classes
                remaining_prob = (1.0 - confidence) / (len(classes) - 1)
                class_probabilities[class_name] = remaining_prob
        
        return predicted_class, confidence, class_probabilities
    
    async def _generate_classification_explainability(
        self, 
        input_data: Dict[str, Any], 
        predicted_class: str,
        class_probabilities: Dict[str, float],
        model_config
    ) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        """Generate explainability data for classification (SHAP/LIME simulation)"""
        try:
            # Simulate feature importance
            feature_importance = {}
            total_importance = 0
            
            for feature, value in input_data.items():
                if isinstance(value, (int, float)):
                    importance = abs(value) * 0.1
                elif isinstance(value, str):
                    importance = len(value) * 0.05
                else:
                    importance = 0.1
                
                feature_importance[feature] = importance
                total_importance += importance
            
            # Normalize importance
            if total_importance > 0:
                feature_importance = {
                    k: v / total_importance * 100 
                    for k, v in feature_importance.items()
                }
            
            # Generate explanation
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            explanation = {
                'method': 'simulated_shap',
                'predicted_class': predicted_class,
                'class_probabilities': class_probabilities,
                'top_features': [{'feature': k, 'importance': v} for k, v in top_features],
                'reasoning': f"Classified as '{predicted_class}' with {class_probabilities[predicted_class]:.3f} confidence based on {', '.join([k for k, v in top_features])}",
                'confidence': 0.85
            }
            
            return feature_importance, explanation
            
        except Exception as e:
            logger.warning(f"Classification explainability generation failed: {e}")
            return None, None
    
    async def _execute_fallback_classification(
        self, 
        input_data: Dict[str, Any], 
        config: Dict[str, Any], 
        context: OperatorContext,
        class_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Execute fallback deterministic classification when ML confidence is low"""
        try:
            # Simple rule-based classification fallback
            fallback_rules = config.get('fallback_rules', {})
            
            # Default classes
            classes = list(class_mapping.keys()) if class_mapping else ['low_risk', 'medium_risk', 'high_risk']
            
            # Simple rule-based classification
            predicted_class = classes[0]  # Default to first class
            confidence = 0.5
            
            if 'threshold_rule' in fallback_rules:
                threshold = fallback_rules['threshold_rule'].get('threshold', 0.5)
                field = fallback_rules['threshold_rule'].get('field', 'value')
                
                if field in input_data:
                    value = input_data[field]
                    if isinstance(value, (int, float)):
                        if value > threshold * 1.5:
                            predicted_class = classes[-1] if len(classes) > 1 else classes[0]  # High
                            confidence = 0.7
                        elif value > threshold:
                            predicted_class = classes[1] if len(classes) > 2 else classes[0]  # Medium
                            confidence = 0.6
                        else:
                            predicted_class = classes[0]  # Low
                            confidence = 0.6
            
            # Generate class probabilities
            class_probabilities = {}
            for class_name in classes:
                if class_name == predicted_class:
                    class_probabilities[class_name] = confidence
                else:
                    class_probabilities[class_name] = (1.0 - confidence) / (len(classes) - 1)
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'method': 'fallback_rule'
            }
            
        except Exception as e:
            logger.error(f"Fallback classification failed: {e}")
            classes = list(class_mapping.keys()) if class_mapping else ['unknown']
            return {
                'predicted_class': classes[0],
                'confidence': 0.3,
                'class_probabilities': {classes[0]: 1.0},
                'method': 'error_fallback'
            }
    
    def set_model_registry(self, registry: Dict[str, Any]):
        """Set the model registry (injected by Node Registry service)"""
        self.model_registry = registry
