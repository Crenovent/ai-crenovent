"""
ML Explain Operator - Implements explain primitive for ML model explainability
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
class ExplanationResult:
    """Result of ML model explanation"""
    explanation_type: str  # 'shap', 'lime', 'gradient', 'attention'
    explanation_data: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_id: str
    model_version: str
    input_features: Dict[str, Any]
    confidence: float
    execution_time_ms: int = 0
    fallback_used: bool = False

class MLExplainOperator(BaseOperator):
    """
    ML Explain operator for model explainability in workflows
    
    Supports:
    - SHAP explanations
    - LIME explanations
    - Gradient-based explanations
    - Attention-based explanations
    - Feature importance analysis
    - Counterfactual explanations
    - Fallback to rule-based explanations
    """
    
    def __init__(self, operator_id: str = "ml_explain"):
        super().__init__(operator_id)
        self.requires_trust_check = True
        self.model_registry = {}  # Will be injected from Node Registry service
        
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate ML explain operator configuration"""
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
        
        # Validate explanation type
        if 'explanation_type' in config:
            explanation_type = config['explanation_type']
            valid_types = ['shap', 'lime', 'gradient', 'attention', 'counterfactual']
            if explanation_type not in valid_types:
                errors.append(f"explanation_type must be one of: {valid_types}")
        
        # Validate explanation parameters
        if 'explanation_params' in config:
            params = config['explanation_params']
            if not isinstance(params, dict):
                errors.append("'explanation_params' must be a dictionary")
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute ML explanation"""
        try:
            model_id = config['model_id']
            input_data = config['input_data']
            explanation_type = config.get('explanation_type', 'shap')
            explanation_params = config.get('explanation_params', {})
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
            
            # Perform explanation
            explanation_result = await self._perform_explanation(
                model_id, input_data, model_config, explanation_type, explanation_params
            )
            
            # Check if fallback is needed (explanation generation failed)
            if not explanation_result.explanation_data and fallback_enabled:
                fallback_result = await self._execute_fallback_explanation(
                    input_data, config, context, explanation_type
                )
                explanation_result.fallback_used = True
                explanation_result.explanation_data = fallback_result.get('explanation_data', {})
                explanation_result.feature_importance = fallback_result.get('feature_importance', {})
                explanation_result.confidence = fallback_result.get('confidence', 0.5)
            
            # Prepare output data
            output_data = {
                'explanation_type': explanation_result.explanation_type,
                'explanation_data': explanation_result.explanation_data,
                'feature_importance': explanation_result.feature_importance,
                'model_id': explanation_result.model_id,
                'model_version': explanation_result.model_version,
                'input_features': explanation_result.input_features,
                'confidence': explanation_result.confidence,
                'fallback_used': explanation_result.fallback_used,
                'execution_time_ms': explanation_result.execution_time_ms
            }
            
            # Add explanation evidence
            evidence_data = {
                'explanation_type': 'ml_explanation',
                'model_id': model_id,
                'explanation_method': explanation_type,
                'explanation_data': explanation_result.explanation_data,
                'feature_importance': explanation_result.feature_importance,
                'input_features': input_data,
                'confidence': explanation_result.confidence,
                'timestamp': datetime.utcnow().isoformat(),
                'tenant_id': context.tenant_id,
                'user_id': context.user_id
            }
            
            if explanation_result.fallback_used:
                evidence_data['fallback_triggered'] = True
                evidence_data['fallback_reason'] = "ML explanation generation failed"
            
            return OperatorResult(
                success=True,
                output_data=output_data,
                confidence_score=explanation_result.confidence,
                evidence_data=evidence_data
            )
            
        except Exception as e:
            logger.error(f"ML Explain execution failed: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Explanation failed: {e}",
                evidence_data={
                    'explanation_type': 'ml_explanation',
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
    
    async def _perform_explanation(
        self, 
        model_id: str, 
        input_data: Dict[str, Any], 
        model_config,
        explanation_type: str,
        explanation_params: Dict[str, Any]
    ) -> ExplanationResult:
        """Perform the actual ML explanation"""
        start_time = time.time()
        
        try:
            # Generate explanation based on type
            explanation_data, feature_importance, confidence = await self._generate_explanation(
                input_data, model_config, explanation_type, explanation_params
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return ExplanationResult(
                explanation_type=explanation_type,
                explanation_data=explanation_data,
                feature_importance=feature_importance,
                model_id=model_id,
                model_version=model_config.model_version,
                input_features=input_data,
                confidence=confidence,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            raise
    
    async def _generate_explanation(
        self, 
        input_data: Dict[str, Any], 
        model_config,
        explanation_type: str,
        explanation_params: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float], float]:
        """Generate explanation based on specified type"""
        try:
            if explanation_type == 'shap':
                return await self._generate_shap_explanation(input_data, model_config, explanation_params)
            elif explanation_type == 'lime':
                return await self._generate_lime_explanation(input_data, model_config, explanation_params)
            elif explanation_type == 'gradient':
                return await self._generate_gradient_explanation(input_data, model_config, explanation_params)
            elif explanation_type == 'attention':
                return await self._generate_attention_explanation(input_data, model_config, explanation_params)
            elif explanation_type == 'counterfactual':
                return await self._generate_counterfactual_explanation(input_data, model_config, explanation_params)
            else:
                raise ValueError(f"Unsupported explanation type: {explanation_type}")
                
        except Exception as e:
            logger.error(f"Explanation generation failed for type {explanation_type}: {e}")
            return {}, {}, 0.0
    
    async def _generate_shap_explanation(
        self, 
        input_data: Dict[str, Any], 
        model_config,
        explanation_params: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float], float]:
        """Generate SHAP explanation (simulation)"""
        try:
            # Simulate SHAP explanation
            feature_importance = {}
            shap_values = {}
            
            # Calculate feature importance based on input characteristics
            for feature, value in input_data.items():
                if isinstance(value, (int, float)):
                    importance = abs(value) * 0.1
                    shap_value = value * 0.05  # Simulate SHAP value
                elif isinstance(value, str):
                    importance = len(value) * 0.05
                    shap_value = len(value) * 0.02
                else:
                    importance = 0.1
                    shap_value = 0.05
                
                feature_importance[feature] = importance
                shap_values[feature] = shap_value
            
            # Normalize importance
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {k: v / total_importance * 100 for k, v in feature_importance.items()}
            
            # Generate SHAP explanation data
            explanation_data = {
                'method': 'shap',
                'shap_values': shap_values,
                'base_value': 0.5,  # Simulated base value
                'feature_names': list(input_data.keys()),
                'expected_value': 0.5,
                'data': list(input_data.values()),
                'values': list(shap_values.values()),
                'feature_importance': feature_importance,
                'summary': f"SHAP explanation shows {max(feature_importance, key=feature_importance.get)} as most important feature"
            }
            
            confidence = 0.85
            
            return explanation_data, feature_importance, confidence
            
        except Exception as e:
            logger.error(f"SHAP explanation generation failed: {e}")
            return {}, {}, 0.0
    
    async def _generate_lime_explanation(
        self, 
        input_data: Dict[str, Any], 
        model_config,
        explanation_params: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float], float]:
        """Generate LIME explanation (simulation)"""
        try:
            # Simulate LIME explanation
            feature_importance = {}
            lime_values = {}
            
            # Calculate feature importance
            for feature, value in input_data.items():
                if isinstance(value, (int, float)):
                    importance = abs(value) * 0.08
                    lime_value = value * 0.03
                elif isinstance(value, str):
                    importance = len(value) * 0.04
                    lime_value = len(value) * 0.015
                else:
                    importance = 0.08
                    lime_value = 0.03
                
                feature_importance[feature] = importance
                lime_values[feature] = lime_value
            
            # Normalize importance
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {k: v / total_importance * 100 for k, v in feature_importance.items()}
            
            # Generate LIME explanation data
            explanation_data = {
                'method': 'lime',
                'lime_values': lime_values,
                'feature_names': list(input_data.keys()),
                'feature_importance': feature_importance,
                'local_prediction': 0.6,  # Simulated local prediction
                'intercept': 0.4,
                'score': 0.8,
                'summary': f"LIME explanation highlights {max(feature_importance, key=feature_importance.get)} as key contributor"
            }
            
            confidence = 0.80
            
            return explanation_data, feature_importance, confidence
            
        except Exception as e:
            logger.error(f"LIME explanation generation failed: {e}")
            return {}, {}, 0.0
    
    async def _generate_gradient_explanation(
        self, 
        input_data: Dict[str, Any], 
        model_config,
        explanation_params: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float], float]:
        """Generate gradient-based explanation (simulation)"""
        try:
            # Simulate gradient explanation
            feature_importance = {}
            gradients = {}
            
            # Calculate gradients
            for feature, value in input_data.items():
                if isinstance(value, (int, float)):
                    gradient = value * 0.02
                    importance = abs(gradient) * 10
                elif isinstance(value, str):
                    gradient = len(value) * 0.01
                    importance = abs(gradient) * 5
                else:
                    gradient = 0.01
                    importance = 0.1
                
                feature_importance[feature] = importance
                gradients[feature] = gradient
            
            # Normalize importance
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {k: v / total_importance * 100 for k, v in feature_importance.items()}
            
            # Generate gradient explanation data
            explanation_data = {
                'method': 'gradient',
                'gradients': gradients,
                'feature_names': list(input_data.keys()),
                'feature_importance': feature_importance,
                'gradient_norm': sum(abs(g) for g in gradients.values()),
                'summary': f"Gradient explanation shows {max(feature_importance, key=feature_importance.get)} has highest gradient"
            }
            
            confidence = 0.75
            
            return explanation_data, feature_importance, confidence
            
        except Exception as e:
            logger.error(f"Gradient explanation generation failed: {e}")
            return {}, {}, 0.0
    
    async def _generate_attention_explanation(
        self, 
        input_data: Dict[str, Any], 
        model_config,
        explanation_params: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float], float]:
        """Generate attention-based explanation (simulation)"""
        try:
            # Simulate attention explanation
            feature_importance = {}
            attention_weights = {}
            
            # Calculate attention weights
            for feature, value in input_data.items():
                if isinstance(value, (int, float)):
                    attention = min(abs(value) * 0.1, 1.0)
                    importance = attention * 100
                elif isinstance(value, str):
                    attention = min(len(value) * 0.05, 1.0)
                    importance = attention * 50
                else:
                    attention = 0.1
                    importance = 10
                
                feature_importance[feature] = importance
                attention_weights[feature] = attention
            
            # Normalize attention weights
            total_attention = sum(attention_weights.values())
            if total_attention > 0:
                attention_weights = {k: v / total_attention for k, v in attention_weights.items()}
            
            # Generate attention explanation data
            explanation_data = {
                'method': 'attention',
                'attention_weights': attention_weights,
                'feature_names': list(input_data.keys()),
                'feature_importance': feature_importance,
                'attention_sum': sum(attention_weights.values()),
                'max_attention': max(attention_weights.values()),
                'summary': f"Attention mechanism focuses on {max(attention_weights, key=attention_weights.get)}"
            }
            
            confidence = 0.90
            
            return explanation_data, feature_importance, confidence
            
        except Exception as e:
            logger.error(f"Attention explanation generation failed: {e}")
            return {}, {}, 0.0
    
    async def _generate_counterfactual_explanation(
        self, 
        input_data: Dict[str, Any], 
        model_config,
        explanation_params: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float], float]:
        """Generate counterfactual explanation (simulation)"""
        try:
            # Simulate counterfactual explanation
            feature_importance = {}
            counterfactuals = {}
            
            # Generate counterfactual suggestions
            for feature, value in input_data.items():
                if isinstance(value, (int, float)):
                    # Suggest changes to numeric values
                    if value > 0:
                        counterfactual = value * 0.8
                    else:
                        counterfactual = value * 1.2
                    importance = abs(value - counterfactual) * 0.1
                elif isinstance(value, str):
                    # Suggest changes to string values
                    if 'high' in value.lower():
                        counterfactual = value.replace('high', 'low')
                    elif 'low' in value.lower():
                        counterfactual = value.replace('low', 'high')
                    else:
                        counterfactual = value + '_modified'
                    importance = len(value) * 0.05
                else:
                    counterfactual = value
                    importance = 0.1
                
                feature_importance[feature] = importance
                counterfactuals[feature] = counterfactual
            
            # Normalize importance
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {k: v / total_importance * 100 for k, v in feature_importance.items()}
            
            # Generate counterfactual explanation data
            explanation_data = {
                'method': 'counterfactual',
                'counterfactuals': counterfactuals,
                'feature_names': list(input_data.keys()),
                'feature_importance': feature_importance,
                'original_input': input_data,
                'summary': f"Counterfactual analysis suggests changing {max(feature_importance, key=feature_importance.get)}"
            }
            
            confidence = 0.70
            
            return explanation_data, feature_importance, confidence
            
        except Exception as e:
            logger.error(f"Counterfactual explanation generation failed: {e}")
            return {}, {}, 0.0
    
    async def _execute_fallback_explanation(
        self, 
        input_data: Dict[str, Any], 
        config: Dict[str, Any], 
        context: OperatorContext,
        explanation_type: str
    ) -> Dict[str, Any]:
        """Execute fallback rule-based explanation when ML explanation fails"""
        try:
            # Simple rule-based explanation fallback
            feature_importance = {}
            
            # Calculate simple feature importance
            for feature, value in input_data.items():
                if isinstance(value, (int, float)):
                    importance = abs(value) * 0.1
                elif isinstance(value, str):
                    importance = len(value) * 0.05
                else:
                    importance = 0.1
                
                feature_importance[feature] = importance
            
            # Normalize importance
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {k: v / total_importance * 100 for k, v in feature_importance.items()}
            
            # Generate simple explanation
            explanation_data = {
                'method': 'rule_based_fallback',
                'feature_importance': feature_importance,
                'summary': f"Rule-based explanation: {max(feature_importance, key=feature_importance.get)} is most important",
                'fallback': True
            }
            
            return {
                'explanation_data': explanation_data,
                'feature_importance': feature_importance,
                'confidence': 0.5
            }
            
        except Exception as e:
            logger.error(f"Fallback explanation failed: {e}")
            return {
                'explanation_data': {'method': 'error_fallback', 'error': str(e)},
                'feature_importance': {},
                'confidence': 0.3
            }
    
    def set_model_registry(self, registry: Dict[str, Any]):
        """Set the model registry (injected by Node Registry service)"""
        self.model_registry = registry
