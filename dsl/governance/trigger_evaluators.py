"""
Task 6.4.8: Build trigger evaluators (SLA, policy, drift/bias, residency)
=========================================================================

Real-time decisions rule engine with low latency
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)

class TriggerType(str, Enum):
    """Trigger types for evaluation"""
    SLA = "sla"
    POLICY = "policy"
    DRIFT_BIAS = "drift_bias"
    RESIDENCY = "residency"

class EvaluationResult(str, Enum):
    """Evaluation result"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class TriggerEvaluationRequest:
    """Trigger evaluation request"""
    trigger_type: TriggerType
    context: Dict[str, Any]
    tenant_id: str
    workflow_id: str
    thresholds: Optional[Dict[str, Any]] = None

@dataclass
class TriggerEvaluationResponse:
    """Trigger evaluation response"""
    trigger_type: TriggerType
    result: EvaluationResult
    triggered: bool
    message: str
    evaluation_time_ms: float
    details: Dict[str, Any]

class SLATriggerEvaluator:
    """SLA trigger evaluator - Task 6.4.8"""
    
    def evaluate(self, context: Dict[str, Any], thresholds: Dict[str, Any]) -> TriggerEvaluationResponse:
        """Evaluate SLA triggers"""
        start_time = time.time()
        
        response_time = context.get("response_time_ms", 0)
        error_rate = context.get("error_rate", 0.0)
        availability = context.get("availability", 100.0)
        
        sla_response_time_threshold = thresholds.get("response_time_ms", 5000)
        sla_error_rate_threshold = thresholds.get("error_rate", 0.01)
        sla_availability_threshold = thresholds.get("availability", 99.0)
        
        triggered = False
        messages = []
        
        if response_time > sla_response_time_threshold:
            triggered = True
            messages.append(f"Response time {response_time}ms > {sla_response_time_threshold}ms")
        
        if error_rate > sla_error_rate_threshold:
            triggered = True
            messages.append(f"Error rate {error_rate} > {sla_error_rate_threshold}")
        
        if availability < sla_availability_threshold:
            triggered = True
            messages.append(f"Availability {availability}% < {sla_availability_threshold}%")
        
        result = EvaluationResult.FAIL if triggered else EvaluationResult.PASS
        message = "; ".join(messages) if messages else "SLA within limits"
        
        evaluation_time = (time.time() - start_time) * 1000
        
        return TriggerEvaluationResponse(
            trigger_type=TriggerType.SLA,
            result=result,
            triggered=triggered,
            message=message,
            evaluation_time_ms=evaluation_time,
            details={
                "response_time_ms": response_time,
                "error_rate": error_rate,
                "availability": availability
            }
        )

class PolicyTriggerEvaluator:
    """Policy trigger evaluator - Task 6.4.8"""
    
    def evaluate(self, context: Dict[str, Any], thresholds: Dict[str, Any]) -> TriggerEvaluationResponse:
        """Evaluate policy triggers"""
        start_time = time.time()
        
        policy_violations = context.get("policy_violations", [])
        compliance_score = context.get("compliance_score", 1.0)
        access_denied = context.get("access_denied", False)
        
        min_compliance_score = thresholds.get("min_compliance_score", 0.8)
        
        triggered = False
        messages = []
        
        if policy_violations:
            triggered = True
            messages.append(f"Policy violations: {', '.join(policy_violations)}")
        
        if compliance_score < min_compliance_score:
            triggered = True
            messages.append(f"Compliance score {compliance_score} < {min_compliance_score}")
        
        if access_denied:
            triggered = True
            messages.append("Access denied by policy")
        
        result = EvaluationResult.FAIL if triggered else EvaluationResult.PASS
        message = "; ".join(messages) if messages else "Policy compliance OK"
        
        evaluation_time = (time.time() - start_time) * 1000
        
        return TriggerEvaluationResponse(
            trigger_type=TriggerType.POLICY,
            result=result,
            triggered=triggered,
            message=message,
            evaluation_time_ms=evaluation_time,
            details={
                "policy_violations": policy_violations,
                "compliance_score": compliance_score,
                "access_denied": access_denied
            }
        )

class DriftBiasTriggerEvaluator:
    """Drift/bias trigger evaluator - Task 6.4.8"""
    
    def evaluate(self, context: Dict[str, Any], thresholds: Dict[str, Any]) -> TriggerEvaluationResponse:
        """Evaluate drift/bias triggers"""
        start_time = time.time()
        
        drift_score = context.get("drift_score", 0.0)
        bias_score = context.get("bias_score", 0.0)
        fairness_score = context.get("fairness_score", 1.0)
        
        max_drift_score = thresholds.get("max_drift_score", 0.3)
        max_bias_score = thresholds.get("max_bias_score", 0.2)
        min_fairness_score = thresholds.get("min_fairness_score", 0.8)
        
        triggered = False
        messages = []
        
        if drift_score > max_drift_score:
            triggered = True
            messages.append(f"Data drift {drift_score} > {max_drift_score}")
        
        if bias_score > max_bias_score:
            triggered = True
            messages.append(f"Bias score {bias_score} > {max_bias_score}")
        
        if fairness_score < min_fairness_score:
            triggered = True
            messages.append(f"Fairness score {fairness_score} < {min_fairness_score}")
        
        result = EvaluationResult.FAIL if triggered else EvaluationResult.PASS
        message = "; ".join(messages) if messages else "Drift/bias within limits"
        
        evaluation_time = (time.time() - start_time) * 1000
        
        return TriggerEvaluationResponse(
            trigger_type=TriggerType.DRIFT_BIAS,
            result=result,
            triggered=triggered,
            message=message,
            evaluation_time_ms=evaluation_time,
            details={
                "drift_score": drift_score,
                "bias_score": bias_score,
                "fairness_score": fairness_score
            }
        )

class ResidencyTriggerEvaluator:
    """Residency trigger evaluator - Task 6.4.8"""
    
    def evaluate(self, context: Dict[str, Any], thresholds: Dict[str, Any]) -> TriggerEvaluationResponse:
        """Evaluate residency triggers"""
        start_time = time.time()
        
        requested_region = context.get("requested_region")
        allowed_regions = context.get("allowed_regions", [])
        geo_compliance_required = context.get("geo_compliance_required", False)
        data_residency_violation = context.get("data_residency_violation", False)
        
        triggered = False
        messages = []
        
        if requested_region and allowed_regions and requested_region not in allowed_regions:
            triggered = True
            messages.append(f"Region {requested_region} not in allowed regions {allowed_regions}")
        
        if geo_compliance_required and not context.get("geo_compliance_verified", False):
            triggered = True
            messages.append("Geo compliance verification required but not verified")
        
        if data_residency_violation:
            triggered = True
            messages.append("Data residency violation detected")
        
        result = EvaluationResult.FAIL if triggered else EvaluationResult.PASS
        message = "; ".join(messages) if messages else "Residency requirements met"
        
        evaluation_time = (time.time() - start_time) * 1000
        
        return TriggerEvaluationResponse(
            trigger_type=TriggerType.RESIDENCY,
            result=result,
            triggered=triggered,
            message=message,
            evaluation_time_ms=evaluation_time,
            details={
                "requested_region": requested_region,
                "allowed_regions": allowed_regions,
                "geo_compliance_required": geo_compliance_required,
                "data_residency_violation": data_residency_violation
            }
        )

class TriggerEvaluationEngine:
    """Trigger evaluation engine - Task 6.4.8"""
    
    def __init__(self):
        self.evaluators = {
            TriggerType.SLA: SLATriggerEvaluator(),
            TriggerType.POLICY: PolicyTriggerEvaluator(),
            TriggerType.DRIFT_BIAS: DriftBiasTriggerEvaluator(),
            TriggerType.RESIDENCY: ResidencyTriggerEvaluator()
        }
    
    def evaluate_trigger(self, request: TriggerEvaluationRequest) -> TriggerEvaluationResponse:
        """Evaluate trigger with real-time decision making"""
        evaluator = self.evaluators.get(request.trigger_type)
        if not evaluator:
            return TriggerEvaluationResponse(
                trigger_type=request.trigger_type,
                result=EvaluationResult.FAIL,
                triggered=True,
                message=f"Unknown trigger type: {request.trigger_type}",
                evaluation_time_ms=0.0,
                details={}
            )
        
        thresholds = request.thresholds or {}
        response = evaluator.evaluate(request.context, thresholds)
        
        logger.info(f"Trigger evaluation: {request.trigger_type} -> {response.result} ({response.evaluation_time_ms:.2f}ms)")
        return response
    
    def evaluate_multiple_triggers(self, requests: List[TriggerEvaluationRequest]) -> List[TriggerEvaluationResponse]:
        """Evaluate multiple triggers efficiently"""
        responses = []
        for request in requests:
            response = self.evaluate_trigger(request)
            responses.append(response)
        return responses
