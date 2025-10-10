"""
Template Confidence Threshold System
Task 4.2.14: Enforce confidence thresholds in templates
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

from ..operators.fallback_service import FallbackService
from .template_explainability_system import TemplateExplainabilitySystem

logger = logging.getLogger(__name__)

class ConfidenceAction(Enum):
    """Actions to take when confidence threshold is not met"""
    REJECT = "reject"
    FALLBACK = "fallback"
    HUMAN_REVIEW = "human_review"
    LOWER_CONFIDENCE = "lower_confidence"
    REQUEST_MORE_DATA = "request_more_data"
    USE_ENSEMBLE = "use_ensemble"

@dataclass
class ConfidenceThreshold:
    """Configuration for confidence thresholds"""
    threshold_value: float
    action: ConfidenceAction
    fallback_method: Optional[str] = None
    human_review_queue: Optional[str] = None
    explanation_required: bool = True
    retry_attempts: int = 0
    escalation_threshold: float = 0.0
    custom_message: Optional[str] = None

@dataclass
class ConfidenceThresholdConfig:
    """Complete confidence threshold configuration for a template"""
    template_id: str
    model_thresholds: Dict[str, ConfidenceThreshold]  # model_id -> threshold config
    global_threshold: Optional[ConfidenceThreshold] = None
    step_thresholds: Dict[str, ConfidenceThreshold] = field(default_factory=dict)  # step_id -> threshold
    dynamic_adjustment: bool = False
    adjustment_factor: float = 0.05
    min_threshold: float = 0.3
    max_threshold: float = 0.95
    performance_tracking: bool = True

@dataclass
class ConfidenceEvaluation:
    """Result of confidence threshold evaluation"""
    template_id: str
    step_id: str
    model_id: str
    confidence_score: float
    threshold_value: float
    threshold_met: bool
    action_taken: ConfidenceAction
    fallback_used: bool = False
    human_review_required: bool = False
    explanation_generated: bool = False
    evaluation_details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class TemplateConfidenceManager:
    """
    Comprehensive confidence threshold management for RBIA templates
    
    Provides:
    - Template-specific confidence thresholds
    - Model-specific and step-specific thresholds
    - Dynamic threshold adjustment
    - Multiple confidence actions (reject, fallback, human review)
    - Performance tracking and optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fallback_service = FallbackService()
        self.explainability_system = TemplateExplainabilitySystem()
        self.threshold_configs: Dict[str, ConfidenceThresholdConfig] = {}
        self.evaluation_history: List[ConfidenceEvaluation] = []
        self._initialize_template_thresholds()
    
    def _initialize_template_thresholds(self):
        """Initialize confidence thresholds for all templates"""
        
        # SaaS templates
        self.threshold_configs["saas_churn_risk_alert"] = ConfidenceThresholdConfig(
            template_id="saas_churn_risk_alert",
            model_thresholds={
                "saas_churn_predictor_v3": ConfidenceThreshold(
                    threshold_value=0.75,
                    action=ConfidenceAction.FALLBACK,
                    fallback_method="rule_based",
                    explanation_required=True
                ),
                "saas_engagement_scorer_v2": ConfidenceThreshold(
                    threshold_value=0.65,
                    action=ConfidenceAction.LOWER_CONFIDENCE,
                    explanation_required=False
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.70,
                action=ConfidenceAction.HUMAN_REVIEW,
                human_review_queue="churn_review_queue"
            ),
            dynamic_adjustment=True,
            performance_tracking=True
        )
        
        self.threshold_configs["saas_forecast_variance_detector"] = ConfidenceThresholdConfig(
            template_id="saas_forecast_variance_detector",
            model_thresholds={
                "saas_forecast_anomaly_detector_v1": ConfidenceThreshold(
                    threshold_value=0.70,
                    action=ConfidenceAction.FALLBACK,
                    fallback_method="statistical_analysis"
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.65,
                action=ConfidenceAction.LOWER_CONFIDENCE
            )
        )
        
        # Banking templates
        self.threshold_configs["banking_credit_scoring_check"] = ConfidenceThresholdConfig(
            template_id="banking_credit_scoring_check",
            model_thresholds={
                "banking_credit_scorer_v4": ConfidenceThreshold(
                    threshold_value=0.80,
                    action=ConfidenceAction.HUMAN_REVIEW,
                    human_review_queue="credit_review_queue",
                    explanation_required=True,
                    escalation_threshold=0.85
                ),
                "banking_income_verifier_v2": ConfidenceThreshold(
                    threshold_value=0.75,
                    action=ConfidenceAction.REQUEST_MORE_DATA,
                    custom_message="Additional income verification required"
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.75,
                action=ConfidenceAction.HUMAN_REVIEW,
                human_review_queue="credit_review_queue"
            ),
            performance_tracking=True
        )
        
        self.threshold_configs["banking_fraudulent_disbursal_detector"] = ConfidenceThresholdConfig(
            template_id="banking_fraudulent_disbursal_detector",
            model_thresholds={
                "banking_fraud_detector_v5": ConfidenceThreshold(
                    threshold_value=0.85,
                    action=ConfidenceAction.REJECT,
                    explanation_required=True,
                    escalation_threshold=0.90
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.80,
                action=ConfidenceAction.HUMAN_REVIEW,
                human_review_queue="fraud_review_queue"
            )
        )
        
        # Insurance templates
        self.threshold_configs["insurance_claim_fraud_anomaly"] = ConfidenceThresholdConfig(
            template_id="insurance_claim_fraud_anomaly",
            model_thresholds={
                "insurance_fraud_detector_v3": ConfidenceThreshold(
                    threshold_value=0.75,
                    action=ConfidenceAction.HUMAN_REVIEW,
                    human_review_queue="siu_queue",
                    explanation_required=True
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.70,
                action=ConfidenceAction.FALLBACK,
                fallback_method="rule_based_fraud_check"
            )
        )
        
        self.threshold_configs["insurance_policy_lapse_predictor"] = ConfidenceThresholdConfig(
            template_id="insurance_policy_lapse_predictor",
            model_thresholds={
                "insurance_lapse_predictor_v2": ConfidenceThreshold(
                    threshold_value=0.70,
                    action=ConfidenceAction.FALLBACK,
                    fallback_method="actuarial_analysis"
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.65,
                action=ConfidenceAction.LOWER_CONFIDENCE
            )
        )
        
        # E-commerce templates
        self.threshold_configs["ecommerce_checkout_fraud_scoring"] = ConfidenceThresholdConfig(
            template_id="ecommerce_checkout_fraud_scoring",
            model_thresholds={
                "ecommerce_fraud_scorer_v4": ConfidenceThreshold(
                    threshold_value=0.75,
                    action=ConfidenceAction.HUMAN_REVIEW,
                    human_review_queue="fraud_review_queue",
                    explanation_required=True
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.70,
                action=ConfidenceAction.FALLBACK,
                fallback_method="velocity_check"
            )
        )
        
        self.threshold_configs["ecommerce_refund_delay_predictor"] = ConfidenceThresholdConfig(
            template_id="ecommerce_refund_delay_predictor",
            model_thresholds={
                "ecommerce_refund_delay_predictor_v2": ConfidenceThreshold(
                    threshold_value=0.70,
                    action=ConfidenceAction.FALLBACK,
                    fallback_method="queue_analysis"
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.65,
                action=ConfidenceAction.LOWER_CONFIDENCE
            )
        )
        
        # Financial Services templates
        self.threshold_configs["fs_liquidity_risk_early_warning"] = ConfidenceThresholdConfig(
            template_id="fs_liquidity_risk_early_warning",
            model_thresholds={
                "fs_liquidity_risk_detector_v3": ConfidenceThreshold(
                    threshold_value=0.80,
                    action=ConfidenceAction.HUMAN_REVIEW,
                    human_review_queue="risk_management_queue",
                    explanation_required=True,
                    escalation_threshold=0.85
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.75,
                action=ConfidenceAction.FALLBACK,
                fallback_method="regulatory_calculation"
            )
        )
        
        self.threshold_configs["fs_mifid_reporting_anomaly_detection"] = ConfidenceThresholdConfig(
            template_id="fs_mifid_reporting_anomaly_detection",
            model_thresholds={
                "fs_reporting_anomaly_detector_v2": ConfidenceThreshold(
                    threshold_value=0.75,
                    action=ConfidenceAction.HUMAN_REVIEW,
                    human_review_queue="compliance_review_queue",
                    explanation_required=True
                )
            },
            global_threshold=ConfidenceThreshold(
                threshold_value=0.70,
                action=ConfidenceAction.FALLBACK,
                fallback_method="regulatory_rules"
            )
        )
        
        self.logger.info(f"Initialized confidence thresholds for {len(self.threshold_configs)} templates")
    
    async def evaluate_confidence_threshold(
        self,
        template_id: str,
        step_id: str,
        model_id: str,
        confidence_score: float,
        prediction_output: Dict[str, Any],
        input_features: Dict[str, Any],
        workflow_context: Dict[str, Any]
    ) -> ConfidenceEvaluation:
        """Evaluate confidence threshold and determine action"""
        
        try:
            config = self.threshold_configs.get(template_id)
            if not config:
                # Use default threshold if no config found
                default_threshold = ConfidenceThreshold(
                    threshold_value=0.70,
                    action=ConfidenceAction.LOWER_CONFIDENCE
                )
                return await self._apply_threshold(
                    template_id, step_id, model_id, confidence_score,
                    default_threshold, prediction_output, input_features, workflow_context
                )
            
            # Determine which threshold to use (priority: step > model > global)
            threshold_config = None
            
            if step_id in config.step_thresholds:
                threshold_config = config.step_thresholds[step_id]
            elif model_id in config.model_thresholds:
                threshold_config = config.model_thresholds[model_id]
            elif config.global_threshold:
                threshold_config = config.global_threshold
            else:
                # Use default if no specific threshold found
                threshold_config = ConfidenceThreshold(
                    threshold_value=0.70,
                    action=ConfidenceAction.LOWER_CONFIDENCE
                )
            
            # Apply dynamic adjustment if enabled
            if config.dynamic_adjustment:
                threshold_config = await self._apply_dynamic_adjustment(
                    template_id, model_id, threshold_config, config
                )
            
            # Evaluate threshold
            evaluation = await self._apply_threshold(
                template_id, step_id, model_id, confidence_score,
                threshold_config, prediction_output, input_features, workflow_context
            )
            
            # Track performance if enabled
            if config.performance_tracking:
                self.evaluation_history.append(evaluation)
                await self._track_performance(evaluation)
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate confidence threshold: {e}")
            return ConfidenceEvaluation(
                template_id=template_id,
                step_id=step_id,
                model_id=model_id,
                confidence_score=confidence_score,
                threshold_value=0.70,
                threshold_met=False,
                action_taken=ConfidenceAction.REJECT,
                evaluation_details={"error": str(e)}
            )
    
    async def _apply_threshold(
        self,
        template_id: str,
        step_id: str,
        model_id: str,
        confidence_score: float,
        threshold_config: ConfidenceThreshold,
        prediction_output: Dict[str, Any],
        input_features: Dict[str, Any],
        workflow_context: Dict[str, Any]
    ) -> ConfidenceEvaluation:
        """Apply confidence threshold and execute action"""
        
        threshold_met = confidence_score >= threshold_config.threshold_value
        action_taken = ConfidenceAction.LOWER_CONFIDENCE  # Default
        fallback_used = False
        human_review_required = False
        explanation_generated = False
        evaluation_details = {
            "original_confidence": confidence_score,
            "threshold_value": threshold_config.threshold_value,
            "threshold_met": threshold_met
        }
        
        if not threshold_met:
            action_taken = threshold_config.action
            
            if action_taken == ConfidenceAction.REJECT:
                evaluation_details["rejection_reason"] = "Confidence below threshold"
                if threshold_config.custom_message:
                    evaluation_details["custom_message"] = threshold_config.custom_message
            
            elif action_taken == ConfidenceAction.FALLBACK:
                fallback_used = True
                if threshold_config.fallback_method:
                    fallback_result = await self._execute_fallback(
                        template_id, model_id, threshold_config.fallback_method,
                        input_features, workflow_context
                    )
                    evaluation_details["fallback_result"] = fallback_result
            
            elif action_taken == ConfidenceAction.HUMAN_REVIEW:
                human_review_required = True
                if threshold_config.human_review_queue:
                    evaluation_details["review_queue"] = threshold_config.human_review_queue
                    await self._queue_for_human_review(
                        template_id, step_id, model_id, confidence_score,
                        threshold_config.human_review_queue, prediction_output, input_features
                    )
            
            elif action_taken == ConfidenceAction.LOWER_CONFIDENCE:
                # Continue with lower confidence but flag it
                evaluation_details["confidence_lowered"] = True
                evaluation_details["warning"] = "Proceeding with lower confidence"
            
            elif action_taken == ConfidenceAction.REQUEST_MORE_DATA:
                evaluation_details["data_request"] = "Additional data required"
                if threshold_config.custom_message:
                    evaluation_details["request_message"] = threshold_config.custom_message
            
            elif action_taken == ConfidenceAction.USE_ENSEMBLE:
                # Trigger ensemble model if available
                ensemble_result = await self._use_ensemble_model(
                    template_id, model_id, input_features, workflow_context
                )
                evaluation_details["ensemble_result"] = ensemble_result
        
        # Generate explanation if required
        if threshold_config.explanation_required:
            try:
                explanation = await self.explainability_system.generate_template_explanation(
                    template_id=template_id,
                    workflow_id=workflow_context.get("workflow_id", "unknown"),
                    step_id=step_id,
                    model_id=model_id,
                    input_features=input_features,
                    prediction_output=prediction_output,
                    confidence_score=confidence_score,
                    tenant_id=workflow_context.get("tenant_id", "unknown")
                )
                explanation_generated = True
                evaluation_details["explanation_id"] = explanation.explanation_id
            except Exception as e:
                self.logger.error(f"Failed to generate explanation: {e}")
        
        return ConfidenceEvaluation(
            template_id=template_id,
            step_id=step_id,
            model_id=model_id,
            confidence_score=confidence_score,
            threshold_value=threshold_config.threshold_value,
            threshold_met=threshold_met,
            action_taken=action_taken,
            fallback_used=fallback_used,
            human_review_required=human_review_required,
            explanation_generated=explanation_generated,
            evaluation_details=evaluation_details
        )
    
    async def _apply_dynamic_adjustment(
        self,
        template_id: str,
        model_id: str,
        threshold_config: ConfidenceThreshold,
        config: ConfidenceThresholdConfig
    ) -> ConfidenceThreshold:
        """Apply dynamic threshold adjustment based on performance"""
        
        try:
            # Get recent performance for this template/model
            recent_evaluations = [
                eval for eval in self.evaluation_history[-100:]  # Last 100 evaluations
                if eval.template_id == template_id and eval.model_id == model_id
            ]
            
            if len(recent_evaluations) < 10:
                return threshold_config  # Not enough data for adjustment
            
            # Calculate success rate
            successful_predictions = sum(1 for eval in recent_evaluations if eval.threshold_met)
            success_rate = successful_predictions / len(recent_evaluations)
            
            # Adjust threshold based on success rate
            adjusted_threshold = threshold_config.threshold_value
            
            if success_rate < 0.7:  # Too many failures, lower threshold
                adjusted_threshold = max(
                    config.min_threshold,
                    threshold_config.threshold_value - config.adjustment_factor
                )
            elif success_rate > 0.9:  # Too many passes, raise threshold
                adjusted_threshold = min(
                    config.max_threshold,
                    threshold_config.threshold_value + config.adjustment_factor
                )
            
            # Create adjusted threshold config
            adjusted_config = ConfidenceThreshold(
                threshold_value=adjusted_threshold,
                action=threshold_config.action,
                fallback_method=threshold_config.fallback_method,
                human_review_queue=threshold_config.human_review_queue,
                explanation_required=threshold_config.explanation_required,
                retry_attempts=threshold_config.retry_attempts,
                escalation_threshold=threshold_config.escalation_threshold,
                custom_message=threshold_config.custom_message
            )
            
            if adjusted_threshold != threshold_config.threshold_value:
                self.logger.info(f"Adjusted threshold for {template_id}/{model_id}: {threshold_config.threshold_value} -> {adjusted_threshold}")
            
            return adjusted_config
            
        except Exception as e:
            self.logger.error(f"Failed to apply dynamic adjustment: {e}")
            return threshold_config
    
    async def _execute_fallback(
        self,
        template_id: str,
        model_id: str,
        fallback_method: str,
        input_features: Dict[str, Any],
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fallback method when confidence threshold is not met"""
        
        try:
            # Use the fallback service
            fallback_result = await self.fallback_service.execute_fallback(
                workflow_id=workflow_context.get("workflow_id", "unknown"),
                step_id=workflow_context.get("step_id", "unknown"),
                model_id=model_id,
                trigger_reason="confidence_below_threshold",
                original_input=input_features,
                context=workflow_context.get("operator_context")
            )
            
            return {
                "fallback_method": fallback_method,
                "fallback_success": fallback_result.success,
                "fallback_output": fallback_result.output_data if fallback_result.success else None,
                "fallback_error": fallback_result.error_message if not fallback_result.success else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute fallback {fallback_method}: {e}")
            return {
                "fallback_method": fallback_method,
                "fallback_success": False,
                "fallback_error": str(e)
            }
    
    async def _queue_for_human_review(
        self,
        template_id: str,
        step_id: str,
        model_id: str,
        confidence_score: float,
        review_queue: str,
        prediction_output: Dict[str, Any],
        input_features: Dict[str, Any]
    ):
        """Queue item for human review"""
        
        try:
            review_item = {
                "review_id": str(uuid.uuid4()),
                "template_id": template_id,
                "step_id": step_id,
                "model_id": model_id,
                "confidence_score": confidence_score,
                "prediction_output": prediction_output,
                "input_features": input_features,
                "queue": review_queue,
                "created_at": datetime.utcnow().isoformat(),
                "status": "pending_review"
            }
            
            # In a real implementation, this would integrate with a review system
            self.logger.info(f"Queued item {review_item['review_id']} for human review in {review_queue}")
            
        except Exception as e:
            self.logger.error(f"Failed to queue for human review: {e}")
    
    async def _use_ensemble_model(
        self,
        template_id: str,
        model_id: str,
        input_features: Dict[str, Any],
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use ensemble model when primary model confidence is low"""
        
        try:
            # Simulate ensemble model usage
            # In a real implementation, this would call multiple models and combine results
            
            ensemble_models = [f"{model_id}_ensemble_1", f"{model_id}_ensemble_2", f"{model_id}_ensemble_3"]
            ensemble_results = []
            
            for ensemble_model in ensemble_models:
                # Simulate ensemble model prediction
                confidence = np.random.uniform(0.6, 0.9)
                result = {
                    "model_id": ensemble_model,
                    "confidence": confidence,
                    "prediction": f"ensemble_prediction_{ensemble_model}"
                }
                ensemble_results.append(result)
            
            # Combine ensemble results
            avg_confidence = sum(r["confidence"] for r in ensemble_results) / len(ensemble_results)
            
            return {
                "ensemble_used": True,
                "ensemble_models": ensemble_models,
                "ensemble_results": ensemble_results,
                "combined_confidence": avg_confidence,
                "ensemble_prediction": "combined_ensemble_result"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to use ensemble model: {e}")
            return {
                "ensemble_used": False,
                "ensemble_error": str(e)
            }
    
    async def _track_performance(self, evaluation: ConfidenceEvaluation):
        """Track performance metrics for threshold optimization"""
        
        try:
            # Calculate performance metrics
            template_evaluations = [
                eval for eval in self.evaluation_history
                if eval.template_id == evaluation.template_id
            ]
            
            if len(template_evaluations) >= 10:
                success_rate = sum(1 for eval in template_evaluations if eval.threshold_met) / len(template_evaluations)
                avg_confidence = sum(eval.confidence_score for eval in template_evaluations) / len(template_evaluations)
                
                performance_metrics = {
                    "template_id": evaluation.template_id,
                    "total_evaluations": len(template_evaluations),
                    "success_rate": success_rate,
                    "average_confidence": avg_confidence,
                    "fallback_usage_rate": sum(1 for eval in template_evaluations if eval.fallback_used) / len(template_evaluations),
                    "human_review_rate": sum(1 for eval in template_evaluations if eval.human_review_required) / len(template_evaluations)
                }
                
                self.logger.info(f"Performance metrics for {evaluation.template_id}: {performance_metrics}")
            
        except Exception as e:
            self.logger.error(f"Failed to track performance: {e}")
    
    def get_template_threshold_config(self, template_id: str) -> Optional[ConfidenceThresholdConfig]:
        """Get threshold configuration for a template"""
        return self.threshold_configs.get(template_id)
    
    def update_template_threshold_config(
        self,
        template_id: str,
        config: ConfidenceThresholdConfig
    ):
        """Update threshold configuration for a template"""
        self.threshold_configs[template_id] = config
        self.logger.info(f"Updated threshold config for template {template_id}")
    
    def get_evaluation_history(
        self,
        template_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ConfidenceEvaluation]:
        """Get evaluation history"""
        if template_id:
            return [
                eval for eval in self.evaluation_history
                if eval.template_id == template_id
            ][-limit:]
        return self.evaluation_history[-limit:]
    
    async def get_performance_summary(
        self,
        template_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance summary for templates"""
        
        try:
            evaluations = self.get_evaluation_history(template_id, 1000)
            
            if not evaluations:
                return {"message": "No evaluation data available"}
            
            # Group by template
            template_stats = {}
            
            for eval in evaluations:
                tid = eval.template_id
                if tid not in template_stats:
                    template_stats[tid] = {
                        "total_evaluations": 0,
                        "threshold_met": 0,
                        "fallback_used": 0,
                        "human_review_required": 0,
                        "confidence_scores": [],
                        "actions": {}
                    }
                
                stats = template_stats[tid]
                stats["total_evaluations"] += 1
                
                if eval.threshold_met:
                    stats["threshold_met"] += 1
                
                if eval.fallback_used:
                    stats["fallback_used"] += 1
                
                if eval.human_review_required:
                    stats["human_review_required"] += 1
                
                stats["confidence_scores"].append(eval.confidence_score)
                
                action = eval.action_taken.value
                stats["actions"][action] = stats["actions"].get(action, 0) + 1
            
            # Calculate summary metrics
            summary = {}
            for tid, stats in template_stats.items():
                total = stats["total_evaluations"]
                summary[tid] = {
                    "total_evaluations": total,
                    "success_rate": stats["threshold_met"] / total if total > 0 else 0,
                    "fallback_rate": stats["fallback_used"] / total if total > 0 else 0,
                    "human_review_rate": stats["human_review_required"] / total if total > 0 else 0,
                    "average_confidence": sum(stats["confidence_scores"]) / len(stats["confidence_scores"]) if stats["confidence_scores"] else 0,
                    "min_confidence": min(stats["confidence_scores"]) if stats["confidence_scores"] else 0,
                    "max_confidence": max(stats["confidence_scores"]) if stats["confidence_scores"] else 0,
                    "action_distribution": stats["actions"]
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    async def optimize_thresholds(self, template_id: str) -> Dict[str, Any]:
        """Optimize confidence thresholds based on performance data"""
        
        try:
            config = self.threshold_configs.get(template_id)
            if not config:
                return {"error": f"No configuration found for template {template_id}"}
            
            evaluations = self.get_evaluation_history(template_id, 500)
            if len(evaluations) < 50:
                return {"message": "Insufficient data for optimization"}
            
            # Analyze performance by model
            model_performance = {}
            
            for eval in evaluations:
                model_id = eval.model_id
                if model_id not in model_performance:
                    model_performance[model_id] = {
                        "evaluations": [],
                        "success_rate": 0,
                        "avg_confidence": 0
                    }
                
                model_performance[model_id]["evaluations"].append(eval)
            
            # Calculate optimal thresholds
            optimization_results = {}
            
            for model_id, perf in model_performance.items():
                evaluations = perf["evaluations"]
                confidences = [e.confidence_score for e in evaluations]
                successes = [e.threshold_met for e in evaluations]
                
                # Find optimal threshold (maximize success rate while maintaining reasonable threshold)
                best_threshold = config.model_thresholds.get(model_id, config.global_threshold).threshold_value
                best_score = 0
                
                for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
                    success_count = sum(1 for c in confidences if c >= threshold)
                    success_rate = success_count / len(confidences) if confidences else 0
                    
                    # Score combines success rate and threshold level
                    score = success_rate * (1 - abs(threshold - 0.75))  # Prefer thresholds around 0.75
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                
                optimization_results[model_id] = {
                    "current_threshold": config.model_thresholds.get(model_id, config.global_threshold).threshold_value,
                    "optimal_threshold": best_threshold,
                    "improvement": best_score,
                    "recommendation": "update" if abs(best_threshold - config.model_thresholds.get(model_id, config.global_threshold).threshold_value) > 0.05 else "no_change"
                }
            
            return {
                "template_id": template_id,
                "optimization_results": optimization_results,
                "total_evaluations_analyzed": len(evaluations)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize thresholds: {e}")
            return {"error": str(e)}
