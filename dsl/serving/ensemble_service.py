"""
Task 6.3.27: Multi-model ensembles (stacked/weighted)
Support multi-model ensembles for improved accuracy
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    STACKED = "stacked"
    VOTING = "voting"
    MAX_CONFIDENCE = "max_confidence"

@dataclass
class ModelWeight:
    """Model weight configuration"""
    model_id: str
    weight: float
    confidence_threshold: float = 0.0

@dataclass
class EnsembleConfig:
    """Ensemble configuration"""
    ensemble_id: str
    method: EnsembleMethod
    models: List[ModelWeight]
    meta_model_id: Optional[str] = None  # For stacked ensembles
    explainability_enabled: bool = True

class EnsembleService:
    """
    Multi-model ensemble service
    Task 6.3.27: Accuracy improvement through ensemble methods
    """
    
    def __init__(self):
        self.ensemble_configs: Dict[str, EnsembleConfig] = {}
    
    def register_ensemble(self, config: EnsembleConfig) -> bool:
        """Register an ensemble configuration"""
        try:
            # Validate weights sum to 1.0 for weighted methods
            if config.method == EnsembleMethod.WEIGHTED_AVERAGE:
                total_weight = sum(model.weight for model in config.models)
                if abs(total_weight - 1.0) > 0.001:
                    raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
            self.ensemble_configs[config.ensemble_id] = config
            logger.info(f"Registered ensemble: {config.ensemble_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register ensemble {config.ensemble_id}: {e}")
            return False
    
    async def predict_ensemble(self, ensemble_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using ensemble"""
        if ensemble_id not in self.ensemble_configs:
            raise ValueError(f"Ensemble {ensemble_id} not found")
        
        config = self.ensemble_configs[ensemble_id]
        
        # Get predictions from all models
        model_predictions = []
        model_confidences = []
        
        for model_weight in config.models:
            prediction = await self._get_model_prediction(model_weight.model_id, input_data)
            
            # Skip models below confidence threshold
            if prediction['confidence'] >= model_weight.confidence_threshold:
                model_predictions.append(prediction)
                model_confidences.append(prediction['confidence'])
        
        if not model_predictions:
            raise ValueError("No models met confidence threshold")
        
        # Combine predictions based on method
        if config.method == EnsembleMethod.WEIGHTED_AVERAGE:
            result = self._weighted_average_ensemble(model_predictions, config.models)
        elif config.method == EnsembleMethod.STACKED:
            result = await self._stacked_ensemble(model_predictions, config.meta_model_id, input_data)
        elif config.method == EnsembleMethod.VOTING:
            result = self._voting_ensemble(model_predictions)
        elif config.method == EnsembleMethod.MAX_CONFIDENCE:
            result = self._max_confidence_ensemble(model_predictions)
        else:
            raise ValueError(f"Unsupported ensemble method: {config.method}")
        
        # Add ensemble metadata
        result['ensemble_id'] = ensemble_id
        result['ensemble_method'] = config.method.value
        result['models_used'] = len(model_predictions)
        
        return result
    
    async def _get_model_prediction(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from individual model"""
        # Simulate model prediction
        return {
            'model_id': model_id,
            'prediction': np.random.random(),
            'confidence': np.random.uniform(0.5, 1.0),
            'probabilities': [np.random.random() for _ in range(3)]
        }
    
    def _weighted_average_ensemble(self, predictions: List[Dict[str, Any]], model_weights: List[ModelWeight]) -> Dict[str, Any]:
        """Combine predictions using weighted average"""
        weight_map = {mw.model_id: mw.weight for mw in model_weights}
        
        weighted_prediction = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            weight = weight_map.get(pred['model_id'], 0.0)
            weighted_prediction += pred['prediction'] * weight
            weighted_confidence += pred['confidence'] * weight
            total_weight += weight
        
        return {
            'prediction': weighted_prediction / total_weight if total_weight > 0 else 0.0,
            'confidence': weighted_confidence / total_weight if total_weight > 0 else 0.0
        }
    
    async def _stacked_ensemble(self, predictions: List[Dict[str, Any]], meta_model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions using stacked ensemble with meta-model"""
        # Prepare meta-features from base model predictions
        meta_features = []
        for pred in predictions:
            meta_features.extend([pred['prediction'], pred['confidence']])
            if 'probabilities' in pred:
                meta_features.extend(pred['probabilities'])
        
        # Get meta-model prediction
        meta_input = {**input_data, 'meta_features': meta_features}
        meta_prediction = await self._get_model_prediction(meta_model_id, meta_input)
        
        return {
            'prediction': meta_prediction['prediction'],
            'confidence': meta_prediction['confidence'],
            'meta_model_used': meta_model_id
        }
    
    def _voting_ensemble(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine predictions using majority voting"""
        # Convert predictions to binary votes (>0.5)
        votes = [1 if pred['prediction'] > 0.5 else 0 for pred in predictions]
        final_prediction = 1 if sum(votes) > len(votes) / 2 else 0
        
        # Average confidence
        avg_confidence = sum(pred['confidence'] for pred in predictions) / len(predictions)
        
        return {
            'prediction': float(final_prediction),
            'confidence': avg_confidence,
            'votes': votes
        }
    
    def _max_confidence_ensemble(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use prediction from model with highest confidence"""
        best_pred = max(predictions, key=lambda x: x['confidence'])
        
        return {
            'prediction': best_pred['prediction'],
            'confidence': best_pred['confidence'],
            'selected_model': best_pred['model_id']
        }

# Global ensemble service instance
ensemble_service = EnsembleService()
