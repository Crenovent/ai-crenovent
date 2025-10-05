# Auto-Scaling and Performance Management System
# Tasks 6.6-T04, T05, T06, T07, T28, T36: HPA, KEDA, quotas, rate limiting, backpressure, predictive scaling

import asyncio
import json
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ScalingTrigger(Enum):
    """Types of scaling triggers"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_DEPTH = "queue_depth"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"

class ScalingDirection(Enum):
    """Scaling direction"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"

class ScalingStrategy(Enum):
    """Scaling strategies"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"

@dataclass
class ScalingPolicy:
    """Auto-scaling policy definition"""
    policy_id: str
    name: str
    tenant_id: Optional[int]
    industry_code: str
    tenant_tier: str
    trigger_type: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    min_replicas: int
    max_replicas: int
    scale_up_cooldown_seconds: int
    scale_down_cooldown_seconds: int
    strategy: ScalingStrategy
    enabled: bool
    created_at: str
    updated_at: str

@dataclass
class ScalingEvent:
    """Scaling event record"""
    event_id: str
    policy_id: str
    tenant_id: int
    trigger_value: float
    threshold_value: float
    scaling_direction: ScalingDirection
    current_replicas: int
    target_replicas: int
    timestamp: str
    reason: str
    success: bool
    error_message: Optional[str]

@dataclass
class ResourceQuota:
    """Resource quota for tenant scaling"""
    quota_id: str
    tenant_id: int
    max_cpu_cores: float
    max_memory_gb: float
    max_replicas: int
    max_concurrent_workflows: int
    max_queue_size: int
    current_usage: Dict[str, float]
    quota_exceeded: bool
    last_updated: str

@dataclass
class RateLimitRule:
    """Rate limiting rule"""
    rule_id: str
    tenant_id: int
    resource_type: str  # api_calls, workflows, data_processing
    limit_per_minute: int
    limit_per_hour: int
    burst_allowance: int
    current_usage: Dict[str, int]
    violations_count: int
    last_reset: str

class BackpressureManager:
    """
    Backpressure management for runtime queues
    Task 6.6-T28: Implement backpressure in runtime queues
    """
    
    def __init__(self):
        self.queue_metrics: Dict[str, Dict[str, Any]] = {}
        self.backpressure_thresholds = {
            'high_priority': 1000,
            'normal_priority': 5000,
            'low_priority': 10000
        }
        self.backpressure_active: Dict[str, bool] = {}
    
    async def check_backpressure(
        self,
        queue_name: str,
        current_depth: int,
        priority: str = "normal_priority"
    ) -> Dict[str, Any]:
        """Check if backpressure should be applied"""
        
        threshold = self.backpressure_thresholds.get(priority, 5000)
        should_apply_backpressure = current_depth > threshold
        
        # Update queue metrics
        self.queue_metrics[queue_name] = {
            'current_depth': current_depth,
            'threshold': threshold,
            'priority': priority,
            'backpressure_active': should_apply_backpressure,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
        self.backpressure_active[queue_name] = should_apply_backpressure
        
        if should_apply_backpressure:
            logger.warning(f"Backpressure activated for queue {queue_name}: depth={current_depth}, threshold={threshold}")
        
        return {
            'apply_backpressure': should_apply_backpressure,
            'queue_depth': current_depth,
            'threshold': threshold,
            'utilization_percent': (current_depth / threshold) * 100,
            'recommended_action': self._get_backpressure_action(current_depth, threshold)
        }
    
    def _get_backpressure_action(self, current_depth: int, threshold: int) -> str:
        """Get recommended backpressure action"""
        
        utilization = (current_depth / threshold) * 100
        
        if utilization > 150:
            return "reject_new_requests"
        elif utilization > 120:
            return "throttle_requests"
        elif utilization > 100:
            return "delay_low_priority"
        else:
            return "no_action"

class PredictiveScaler:
    """
    Predictive scaling based on telemetry
    Task 6.6-T36: Implement predictive scaling based on telemetry
    """
    
    def __init__(self):
        self.historical_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_models: Dict[str, Any] = {}
    
    async def predict_scaling_need(
        self,
        tenant_id: int,
        current_metrics: Dict[str, float],
        prediction_horizon_minutes: int = 30
    ) -> Dict[str, Any]:
        """Predict scaling needs based on historical patterns"""
        
        # Store current metrics
        timestamp = time.time()
        metric_key = f"tenant_{tenant_id}"
        
        self.historical_metrics[metric_key].append({
            'timestamp': timestamp,
            'metrics': current_metrics
        })
        
        # Simple trend-based prediction (in practice would use ML models)
        prediction = await self._calculate_trend_prediction(
            metric_key,
            prediction_horizon_minutes
        )
        
        return {
            'tenant_id': tenant_id,
            'prediction_horizon_minutes': prediction_horizon_minutes,
            'predicted_metrics': prediction,
            'scaling_recommendation': self._generate_scaling_recommendation(prediction),
            'confidence_score': prediction.get('confidence', 0.5),
            'predicted_at': datetime.now(timezone.utc).isoformat()
        }
    
    async def _calculate_trend_prediction(
        self,
        metric_key: str,
        horizon_minutes: int
    ) -> Dict[str, Any]:
        """Calculate trend-based prediction"""
        
        historical_data = list(self.historical_metrics[metric_key])
        
        if len(historical_data) < 10:
            return {'confidence': 0.1, 'cpu_utilization': 50.0, 'memory_utilization': 50.0}
        
        # Get recent data points (last hour)
        recent_data = [
            point for point in historical_data
            if point['timestamp'] > time.time() - 3600
        ]
        
        if len(recent_data) < 5:
            return {'confidence': 0.2, 'cpu_utilization': 50.0, 'memory_utilization': 50.0}
        
        # Calculate trends for key metrics
        cpu_values = [point['metrics'].get('cpu_utilization', 0) for point in recent_data]
        memory_values = [point['metrics'].get('memory_utilization', 0) for point in recent_data]
        
        # Simple linear trend calculation
        cpu_trend = self._calculate_linear_trend(cpu_values)
        memory_trend = self._calculate_linear_trend(memory_values)
        
        # Project forward
        current_cpu = cpu_values[-1] if cpu_values else 50.0
        current_memory = memory_values[-1] if memory_values else 50.0
        
        predicted_cpu = max(0, min(100, current_cpu + (cpu_trend * horizon_minutes)))
        predicted_memory = max(0, min(100, current_memory + (memory_trend * horizon_minutes)))
        
        # Calculate confidence based on data consistency
        cpu_variance = statistics.variance(cpu_values) if len(cpu_values) > 1 else 100
        memory_variance = statistics.variance(memory_values) if len(memory_values) > 1 else 100
        
        confidence = max(0.1, min(0.9, 1.0 - (cpu_variance + memory_variance) / 200))
        
        return {
            'confidence': confidence,
            'cpu_utilization': predicted_cpu,
            'memory_utilization': predicted_memory,
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'data_points': len(recent_data)
        }
    
    def _calculate_linear_trend(self, values: List[float]) -> float:
        """Calculate linear trend from values"""
        
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _generate_scaling_recommendation(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scaling recommendation from prediction"""
        
        predicted_cpu = prediction.get('cpu_utilization', 50.0)
        predicted_memory = prediction.get('memory_utilization', 50.0)
        confidence = prediction.get('confidence', 0.5)
        
        # Scaling thresholds
        scale_up_cpu_threshold = 80.0
        scale_down_cpu_threshold = 30.0
        scale_up_memory_threshold = 85.0
        scale_down_memory_threshold = 25.0
        
        recommendation = "no_change"
        reason = "Predicted utilization within normal range"
        
        if confidence > 0.6:  # Only act on high-confidence predictions
            if predicted_cpu > scale_up_cpu_threshold or predicted_memory > scale_up_memory_threshold:
                recommendation = "scale_up"
                reason = f"Predicted high utilization: CPU={predicted_cpu:.1f}%, Memory={predicted_memory:.1f}%"
            elif predicted_cpu < scale_down_cpu_threshold and predicted_memory < scale_down_memory_threshold:
                recommendation = "scale_down"
                reason = f"Predicted low utilization: CPU={predicted_cpu:.1f}%, Memory={predicted_memory:.1f}%"
        
        return {
            'action': recommendation,
            'reason': reason,
            'confidence': confidence,
            'predicted_cpu': predicted_cpu,
            'predicted_memory': predicted_memory
        }

class AutoScalingManager:
    """
    Comprehensive Auto-Scaling Management System
    Tasks 6.6-T04, T05, T06, T07, T28, T36: HPA, KEDA, quotas, rate limiting, backpressure, predictive scaling
    """
    
    def __init__(self, tenant_context_manager=None, metrics_collector=None):
        self.tenant_context_manager = tenant_context_manager
        self.metrics_collector = metrics_collector
        
        # Core components
        self.backpressure_manager = BackpressureManager()
        self.predictive_scaler = PredictiveScaler()
        
        # Storage for policies and state
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.resource_quotas: Dict[int, ResourceQuota] = {}
        self.rate_limit_rules: Dict[int, List[RateLimitRule]] = defaultdict(list)
        self.scaling_events: List[ScalingEvent] = []
        
        # Current scaling state
        self.current_replicas: Dict[str, int] = defaultdict(lambda: 1)
        self.last_scaling_action: Dict[str, datetime] = {}
        
        # Initialize default policies
        self._initialize_default_scaling_policies()
        self._initialize_default_quotas()
        self._initialize_default_rate_limits()
    
    def _initialize_default_scaling_policies(self) -> None:
        """Initialize default scaling policies for different industries"""
        
        default_policies = [
            # SaaS CPU-based scaling
            {
                'policy_id': 'saas_cpu_scaling',
                'name': 'SaaS CPU-based Auto Scaling',
                'tenant_id': None,
                'industry_code': 'SaaS',
                'tenant_tier': 'professional',
                'trigger_type': ScalingTrigger.CPU_UTILIZATION,
                'scale_up_threshold': 70.0,
                'scale_down_threshold': 30.0,
                'min_replicas': 2,
                'max_replicas': 10,
                'scale_up_cooldown_seconds': 300,
                'scale_down_cooldown_seconds': 600,
                'strategy': ScalingStrategy.REACTIVE
            },
            # Banking queue-based scaling (stricter)
            {
                'policy_id': 'banking_queue_scaling',
                'name': 'Banking Queue-based Auto Scaling',
                'tenant_id': None,
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'trigger_type': ScalingTrigger.QUEUE_DEPTH,
                'scale_up_threshold': 100.0,  # Queue depth
                'scale_down_threshold': 10.0,
                'min_replicas': 3,
                'max_replicas': 20,
                'scale_up_cooldown_seconds': 180,  # Faster scaling for banking
                'scale_down_cooldown_seconds': 900,
                'strategy': ScalingStrategy.HYBRID
            },
            # Insurance response time scaling
            {
                'policy_id': 'insurance_latency_scaling',
                'name': 'Insurance Latency-based Auto Scaling',
                'tenant_id': None,
                'industry_code': 'Insurance',
                'tenant_tier': 'enterprise',
                'trigger_type': ScalingTrigger.RESPONSE_TIME,
                'scale_up_threshold': 2000.0,  # 2 seconds
                'scale_down_threshold': 500.0,  # 0.5 seconds
                'min_replicas': 2,
                'max_replicas': 15,
                'scale_up_cooldown_seconds': 240,
                'scale_down_cooldown_seconds': 720,
                'strategy': ScalingStrategy.PREDICTIVE
            }
        ]
        
        for policy_data in default_policies:
            policy = ScalingPolicy(
                policy_id=policy_data['policy_id'],
                name=policy_data['name'],
                tenant_id=policy_data['tenant_id'],
                industry_code=policy_data['industry_code'],
                tenant_tier=policy_data['tenant_tier'],
                trigger_type=policy_data['trigger_type'],
                scale_up_threshold=policy_data['scale_up_threshold'],
                scale_down_threshold=policy_data['scale_down_threshold'],
                min_replicas=policy_data['min_replicas'],
                max_replicas=policy_data['max_replicas'],
                scale_up_cooldown_seconds=policy_data['scale_up_cooldown_seconds'],
                scale_down_cooldown_seconds=policy_data['scale_down_cooldown_seconds'],
                strategy=policy_data['strategy'],
                enabled=True,
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat()
            )
            self.scaling_policies[policy.policy_id] = policy
    
    def _initialize_default_quotas(self) -> None:
        """Initialize default resource quotas by tenant tier"""
        
        # These would be set per tenant during onboarding
        default_quotas = {
            'starter': {
                'max_cpu_cores': 4.0,
                'max_memory_gb': 8.0,
                'max_replicas': 5,
                'max_concurrent_workflows': 50,
                'max_queue_size': 1000
            },
            'professional': {
                'max_cpu_cores': 16.0,
                'max_memory_gb': 32.0,
                'max_replicas': 15,
                'max_concurrent_workflows': 500,
                'max_queue_size': 10000
            },
            'enterprise': {
                'max_cpu_cores': 64.0,
                'max_memory_gb': 128.0,
                'max_replicas': 50,
                'max_concurrent_workflows': 2000,
                'max_queue_size': 50000
            },
            'enterprise_plus': {
                'max_cpu_cores': 256.0,
                'max_memory_gb': 512.0,
                'max_replicas': 200,
                'max_concurrent_workflows': 10000,
                'max_queue_size': 200000
            }
        }
        
        # Store as templates (actual quotas created per tenant)
        self.default_quota_templates = default_quotas
    
    def _initialize_default_rate_limits(self) -> None:
        """Initialize default rate limiting rules"""
        
        # These would be applied per tenant based on their tier
        self.default_rate_limit_templates = {
            'starter': {
                'api_calls': {'per_minute': 100, 'per_hour': 5000, 'burst': 20},
                'workflows': {'per_minute': 10, 'per_hour': 500, 'burst': 5},
                'data_processing': {'per_minute': 50, 'per_hour': 2000, 'burst': 10}
            },
            'professional': {
                'api_calls': {'per_minute': 1000, 'per_hour': 50000, 'burst': 200},
                'workflows': {'per_minute': 100, 'per_hour': 5000, 'burst': 50},
                'data_processing': {'per_minute': 500, 'per_hour': 20000, 'burst': 100}
            },
            'enterprise': {
                'api_calls': {'per_minute': 10000, 'per_hour': 500000, 'burst': 2000},
                'workflows': {'per_minute': 1000, 'per_hour': 50000, 'burst': 500},
                'data_processing': {'per_minute': 5000, 'per_hour': 200000, 'burst': 1000}
            },
            'enterprise_plus': {
                'api_calls': {'per_minute': 100000, 'per_hour': 5000000, 'burst': 20000},
                'workflows': {'per_minute': 10000, 'per_hour': 500000, 'burst': 5000},
                'data_processing': {'per_minute': 50000, 'per_hour': 2000000, 'burst': 10000}
            }
        }
    
    async def create_tenant_quota(self, tenant_id: int) -> ResourceQuota:
        """Create resource quota for tenant"""
        
        # Get tenant context
        tenant_context = None
        if self.tenant_context_manager:
            tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
        
        tier_key = tenant_context.tenant_tier.value if tenant_context else 'professional'
        quota_template = self.default_quota_templates.get(tier_key, self.default_quota_templates['professional'])
        
        quota = ResourceQuota(
            quota_id=f"quota_{tenant_id}_{int(time.time())}",
            tenant_id=tenant_id,
            max_cpu_cores=quota_template['max_cpu_cores'],
            max_memory_gb=quota_template['max_memory_gb'],
            max_replicas=quota_template['max_replicas'],
            max_concurrent_workflows=quota_template['max_concurrent_workflows'],
            max_queue_size=quota_template['max_queue_size'],
            current_usage={
                'cpu_cores': 0.0,
                'memory_gb': 0.0,
                'replicas': 1,
                'concurrent_workflows': 0,
                'queue_size': 0
            },
            quota_exceeded=False,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        self.resource_quotas[tenant_id] = quota
        logger.info(f"Created resource quota for tenant {tenant_id}: {tier_key} tier")
        return quota
    
    async def create_tenant_rate_limits(self, tenant_id: int) -> List[RateLimitRule]:
        """Create rate limiting rules for tenant"""
        
        # Get tenant context
        tenant_context = None
        if self.tenant_context_manager:
            tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
        
        tier_key = tenant_context.tenant_tier.value if tenant_context else 'professional'
        rate_template = self.default_rate_limit_templates.get(tier_key, self.default_rate_limit_templates['professional'])
        
        rules = []
        for resource_type, limits in rate_template.items():
            rule = RateLimitRule(
                rule_id=f"rate_limit_{tenant_id}_{resource_type}_{int(time.time())}",
                tenant_id=tenant_id,
                resource_type=resource_type,
                limit_per_minute=limits['per_minute'],
                limit_per_hour=limits['per_hour'],
                burst_allowance=limits['burst'],
                current_usage={'minute': 0, 'hour': 0, 'burst_used': 0},
                violations_count=0,
                last_reset=datetime.now(timezone.utc).isoformat()
            )
            rules.append(rule)
        
        self.rate_limit_rules[tenant_id] = rules
        logger.info(f"Created rate limiting rules for tenant {tenant_id}: {len(rules)} rules")
        return rules
    
    async def check_rate_limit(
        self,
        tenant_id: int,
        resource_type: str,
        requested_count: int = 1
    ) -> Dict[str, Any]:
        """Check if request is within rate limits"""
        
        if tenant_id not in self.rate_limit_rules:
            await self.create_tenant_rate_limits(tenant_id)
        
        # Find applicable rule
        applicable_rule = None
        for rule in self.rate_limit_rules[tenant_id]:
            if rule.resource_type == resource_type:
                applicable_rule = rule
                break
        
        if not applicable_rule:
            return {'allowed': True, 'reason': 'no_rate_limit_defined'}
        
        current_time = datetime.now(timezone.utc)
        last_reset = datetime.fromisoformat(applicable_rule.last_reset.replace('Z', '+00:00'))
        
        # Reset counters if needed
        if (current_time - last_reset).total_seconds() >= 60:  # Reset every minute
            applicable_rule.current_usage['minute'] = 0
            if (current_time - last_reset).total_seconds() >= 3600:  # Reset every hour
                applicable_rule.current_usage['hour'] = 0
                applicable_rule.current_usage['burst_used'] = 0
            applicable_rule.last_reset = current_time.isoformat()
        
        # Check limits
        minute_usage = applicable_rule.current_usage['minute']
        hour_usage = applicable_rule.current_usage['hour']
        burst_used = applicable_rule.current_usage['burst_used']
        
        # Check if request would exceed limits
        if minute_usage + requested_count > applicable_rule.limit_per_minute:
            # Check if burst allowance can cover it
            if burst_used + requested_count <= applicable_rule.burst_allowance:
                # Allow using burst
                applicable_rule.current_usage['burst_used'] += requested_count
                return {
                    'allowed': True,
                    'reason': 'burst_allowance_used',
                    'burst_remaining': applicable_rule.burst_allowance - applicable_rule.current_usage['burst_used']
                }
            else:
                # Rate limit exceeded
                applicable_rule.violations_count += 1
                return {
                    'allowed': False,
                    'reason': 'minute_rate_limit_exceeded',
                    'limit': applicable_rule.limit_per_minute,
                    'current_usage': minute_usage,
                    'reset_in_seconds': 60 - (current_time - last_reset).total_seconds()
                }
        
        if hour_usage + requested_count > applicable_rule.limit_per_hour:
            applicable_rule.violations_count += 1
            return {
                'allowed': False,
                'reason': 'hour_rate_limit_exceeded',
                'limit': applicable_rule.limit_per_hour,
                'current_usage': hour_usage,
                'reset_in_seconds': 3600 - (current_time - last_reset).total_seconds()
            }
        
        # Update usage
        applicable_rule.current_usage['minute'] += requested_count
        applicable_rule.current_usage['hour'] += requested_count
        
        return {
            'allowed': True,
            'reason': 'within_limits',
            'minute_remaining': applicable_rule.limit_per_minute - applicable_rule.current_usage['minute'],
            'hour_remaining': applicable_rule.limit_per_hour - applicable_rule.current_usage['hour']
        }
    
    async def evaluate_scaling_decision(
        self,
        tenant_id: int,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate scaling decision based on policies and metrics"""
        
        # Get tenant context for industry/tier
        tenant_context = None
        if self.tenant_context_manager:
            tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
        
        industry_code = tenant_context.industry_code.value if tenant_context else 'SaaS'
        tenant_tier = tenant_context.tenant_tier.value if tenant_context else 'professional'
        
        # Find applicable scaling policies
        applicable_policies = []
        for policy in self.scaling_policies.values():
            if (policy.industry_code == industry_code and 
                policy.tenant_tier == tenant_tier and 
                policy.enabled):
                applicable_policies.append(policy)
        
        if not applicable_policies:
            return {
                'scaling_decision': ScalingDirection.NO_CHANGE,
                'reason': 'No applicable scaling policies found',
                'tenant_id': tenant_id
            }
        
        scaling_decisions = []
        
        for policy in applicable_policies:
            decision = await self._evaluate_single_policy(
                policy, tenant_id, current_metrics
            )
            scaling_decisions.append(decision)
        
        # Aggregate decisions (most conservative approach)
        final_decision = self._aggregate_scaling_decisions(scaling_decisions)
        
        # Check resource quotas
        quota_check = await self._check_scaling_quota(tenant_id, final_decision)
        if not quota_check['allowed']:
            final_decision['scaling_decision'] = ScalingDirection.NO_CHANGE
            final_decision['reason'] = quota_check['reason']
        
        return final_decision
    
    async def _evaluate_single_policy(
        self,
        policy: ScalingPolicy,
        tenant_id: int,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate single scaling policy"""
        
        trigger_metric = policy.trigger_type.value
        current_value = current_metrics.get(trigger_metric, 0.0)
        
        # Check cooldown period
        policy_key = f"{policy.policy_id}_{tenant_id}"
        last_action = self.last_scaling_action.get(policy_key)
        current_time = datetime.now(timezone.utc)
        
        if last_action:
            time_since_last_action = (current_time - last_action).total_seconds()
            if time_since_last_action < policy.scale_up_cooldown_seconds:
                return {
                    'policy_id': policy.policy_id,
                    'scaling_decision': ScalingDirection.NO_CHANGE,
                    'reason': f'Cooldown period active ({time_since_last_action:.0f}s remaining)',
                    'current_value': current_value
                }
        
        # Determine scaling direction
        if current_value > policy.scale_up_threshold:
            scaling_decision = ScalingDirection.SCALE_UP
            reason = f'{trigger_metric} ({current_value}) > threshold ({policy.scale_up_threshold})'
        elif current_value < policy.scale_down_threshold:
            scaling_decision = ScalingDirection.SCALE_DOWN
            reason = f'{trigger_metric} ({current_value}) < threshold ({policy.scale_down_threshold})'
        else:
            scaling_decision = ScalingDirection.NO_CHANGE
            reason = f'{trigger_metric} ({current_value}) within normal range'
        
        # For predictive strategies, also consider prediction
        if policy.strategy == ScalingStrategy.PREDICTIVE:
            prediction = await self.predictive_scaler.predict_scaling_need(
                tenant_id, current_metrics, 30
            )
            
            if prediction['confidence_score'] > 0.6:
                predicted_action = prediction['scaling_recommendation']['action']
                if predicted_action != 'no_change':
                    scaling_decision = ScalingDirection.SCALE_UP if predicted_action == 'scale_up' else ScalingDirection.SCALE_DOWN
                    reason += f" + Predictive: {prediction['scaling_recommendation']['reason']}"
        
        return {
            'policy_id': policy.policy_id,
            'scaling_decision': scaling_decision,
            'reason': reason,
            'current_value': current_value,
            'threshold': policy.scale_up_threshold if scaling_decision == ScalingDirection.SCALE_UP else policy.scale_down_threshold
        }
    
    def _aggregate_scaling_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple scaling decisions"""
        
        scale_up_votes = sum(1 for d in decisions if d['scaling_decision'] == ScalingDirection.SCALE_UP)
        scale_down_votes = sum(1 for d in decisions if d['scaling_decision'] == ScalingDirection.SCALE_DOWN)
        no_change_votes = sum(1 for d in decisions if d['scaling_decision'] == ScalingDirection.NO_CHANGE)
        
        # Conservative approach: require majority for scaling
        total_votes = len(decisions)
        
        if scale_up_votes > total_votes / 2:
            final_decision = ScalingDirection.SCALE_UP
            reason = f"Scale up: {scale_up_votes}/{total_votes} policies recommend scaling up"
        elif scale_down_votes > total_votes / 2:
            final_decision = ScalingDirection.SCALE_DOWN
            reason = f"Scale down: {scale_down_votes}/{total_votes} policies recommend scaling down"
        else:
            final_decision = ScalingDirection.NO_CHANGE
            reason = f"No consensus: up={scale_up_votes}, down={scale_down_votes}, no_change={no_change_votes}"
        
        return {
            'scaling_decision': final_decision,
            'reason': reason,
            'policy_decisions': decisions,
            'vote_summary': {
                'scale_up': scale_up_votes,
                'scale_down': scale_down_votes,
                'no_change': no_change_votes
            }
        }
    
    async def _check_scaling_quota(
        self,
        tenant_id: int,
        scaling_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if scaling decision is within resource quotas"""
        
        if tenant_id not in self.resource_quotas:
            await self.create_tenant_quota(tenant_id)
        
        quota = self.resource_quotas[tenant_id]
        current_replicas = self.current_replicas.get(f"tenant_{tenant_id}", 1)
        
        if scaling_decision['scaling_decision'] == ScalingDirection.SCALE_UP:
            if current_replicas >= quota.max_replicas:
                return {
                    'allowed': False,
                    'reason': f'Maximum replicas quota reached ({quota.max_replicas})'
                }
        elif scaling_decision['scaling_decision'] == ScalingDirection.SCALE_DOWN:
            # Find minimum replicas from applicable policies
            min_replicas = 1
            for policy in self.scaling_policies.values():
                if policy.enabled:
                    min_replicas = max(min_replicas, policy.min_replicas)
            
            if current_replicas <= min_replicas:
                return {
                    'allowed': False,
                    'reason': f'Minimum replicas limit reached ({min_replicas})'
                }
        
        return {'allowed': True, 'reason': 'Within quota limits'}
    
    async def execute_scaling_action(
        self,
        tenant_id: int,
        scaling_decision: Dict[str, Any]
    ) -> ScalingEvent:
        """Execute scaling action (mock implementation)"""
        
        current_replicas = self.current_replicas.get(f"tenant_{tenant_id}", 1)
        
        if scaling_decision['scaling_decision'] == ScalingDirection.SCALE_UP:
            target_replicas = min(current_replicas + 1, 50)  # Scale up by 1
        elif scaling_decision['scaling_decision'] == ScalingDirection.SCALE_DOWN:
            target_replicas = max(current_replicas - 1, 1)  # Scale down by 1
        else:
            target_replicas = current_replicas
        
        # Create scaling event
        event = ScalingEvent(
            event_id=str(uuid.uuid4()),
            policy_id=scaling_decision.get('policy_decisions', [{}])[0].get('policy_id', 'unknown'),
            tenant_id=tenant_id,
            trigger_value=scaling_decision.get('policy_decisions', [{}])[0].get('current_value', 0.0),
            threshold_value=scaling_decision.get('policy_decisions', [{}])[0].get('threshold', 0.0),
            scaling_direction=scaling_decision['scaling_decision'],
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=scaling_decision['reason'],
            success=True,  # Mock success
            error_message=None
        )
        
        # Update current replicas
        self.current_replicas[f"tenant_{tenant_id}"] = target_replicas
        
        # Update last scaling action time
        for decision in scaling_decision.get('policy_decisions', []):
            policy_key = f"{decision.get('policy_id')}_{tenant_id}"
            self.last_scaling_action[policy_key] = datetime.now(timezone.utc)
        
        # Store event
        self.scaling_events.append(event)
        
        logger.info(f"Executed scaling action for tenant {tenant_id}: {current_replicas} -> {target_replicas}")
        return event
    
    async def get_scaling_status(self, tenant_id: int) -> Dict[str, Any]:
        """Get current scaling status for tenant"""
        
        current_replicas = self.current_replicas.get(f"tenant_{tenant_id}", 1)
        
        # Get recent scaling events
        recent_events = [
            event for event in self.scaling_events[-50:]  # Last 50 events
            if event.tenant_id == tenant_id
        ]
        
        # Get resource quota
        quota = self.resource_quotas.get(tenant_id)
        
        # Get rate limit status
        rate_limits = self.rate_limit_rules.get(tenant_id, [])
        
        return {
            'tenant_id': tenant_id,
            'current_replicas': current_replicas,
            'resource_quota': asdict(quota) if quota else None,
            'rate_limits': [asdict(rule) for rule in rate_limits],
            'recent_scaling_events': [asdict(event) for event in recent_events[-10:]],
            'backpressure_status': {
                queue_name: metrics for queue_name, metrics in self.backpressure_manager.queue_metrics.items()
            },
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

# Global auto-scaling manager
auto_scaling_manager = AutoScalingManager()
