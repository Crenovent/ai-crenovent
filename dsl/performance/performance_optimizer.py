# Performance Optimization and Anomaly Detection System
# Tasks 6.6-T24, T26, T27, T35, T37: Fallback behavior, anomaly detection, noisy neighbor mitigation, governance rules

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

class PerformanceAnomalyType(Enum):
    """Types of performance anomalies"""
    LATENCY_SPIKE = "latency_spike"
    THROUGHPUT_DROP = "throughput_drop"
    ERROR_RATE_INCREASE = "error_rate_increase"
    MEMORY_LEAK = "memory_leak"
    CPU_SATURATION = "cpu_saturation"
    QUEUE_BUILDUP = "queue_buildup"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class FallbackMode(Enum):
    """Fallback modes when scaling fails"""
    SAFE_MODE = "safe_mode"
    DEGRADED_SERVICE = "degraded_service"
    CIRCUIT_BREAKER = "circuit_breaker"
    QUEUE_ONLY = "queue_only"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

class NoisyNeighborAction(Enum):
    """Actions for noisy neighbor mitigation"""
    THROTTLE = "throttle"
    ISOLATE = "isolate"
    SUSPEND = "suspend"
    ALERT_ONLY = "alert_only"

@dataclass
class PerformanceAnomaly:
    """Performance anomaly detection result"""
    anomaly_id: str
    tenant_id: int
    anomaly_type: PerformanceAnomalyType
    metric_name: str
    current_value: float
    expected_value: float
    deviation_percent: float
    severity: str  # low, medium, high, critical
    confidence_score: float
    detected_at: str
    duration_minutes: int
    root_cause_analysis: Dict[str, Any]
    recommended_actions: List[str]

@dataclass
class NoisyNeighborDetection:
    """Noisy neighbor detection result"""
    detection_id: str
    tenant_id: int
    resource_type: str
    usage_percent: float
    threshold_percent: float
    impact_on_others: List[int]  # Affected tenant IDs
    detected_at: str
    duration_minutes: int
    mitigation_action: NoisyNeighborAction
    action_taken: bool

@dataclass
class FallbackConfiguration:
    """Fallback configuration for scaling failures"""
    config_id: str
    tenant_id: Optional[int]
    industry_code: str
    tenant_tier: str
    trigger_conditions: List[str]
    fallback_mode: FallbackMode
    max_requests_per_minute: int
    max_concurrent_workflows: int
    timeout_seconds: int
    retry_attempts: int
    enabled: bool
    created_at: str

@dataclass
class ScalingOverride:
    """Scaling governance override"""
    override_id: str
    tenant_id: int
    requested_by: str
    override_type: str  # quota_increase, policy_bypass, emergency_scaling
    justification: str
    approval_status: str  # pending, approved, rejected
    approved_by: Optional[str]
    approved_at: Optional[str]
    expiry_time: str
    applied: bool
    created_at: str

class PerformanceAnomalyDetector:
    """
    Performance anomaly detection using statistical analysis
    Task 6.6-T26: Document performance anomaly detection rules
    """
    
    def __init__(self):
        self.baseline_metrics: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        self.anomaly_thresholds = {
            'latency_spike': {'deviation_percent': 50.0, 'confidence_threshold': 0.8},
            'throughput_drop': {'deviation_percent': 30.0, 'confidence_threshold': 0.7},
            'error_rate_increase': {'deviation_percent': 100.0, 'confidence_threshold': 0.9},
            'memory_leak': {'trend_threshold': 10.0, 'confidence_threshold': 0.8},
            'cpu_saturation': {'absolute_threshold': 90.0, 'confidence_threshold': 0.9}
        }
    
    async def detect_anomalies(
        self,
        tenant_id: int,
        current_metrics: Dict[str, float]
    ) -> List[PerformanceAnomaly]:
        """Detect performance anomalies in current metrics"""
        
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            # Update baseline
            self.baseline_metrics[tenant_id][metric_name].append(current_value)
            
            # Skip if insufficient baseline data
            if len(self.baseline_metrics[tenant_id][metric_name]) < 10:
                continue
            
            # Detect different types of anomalies
            anomaly = await self._detect_metric_anomaly(
                tenant_id, metric_name, current_value
            )
            
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_metric_anomaly(
        self,
        tenant_id: int,
        metric_name: str,
        current_value: float
    ) -> Optional[PerformanceAnomaly]:
        """Detect anomaly for specific metric"""
        
        baseline_values = list(self.baseline_metrics[tenant_id][metric_name])
        
        if len(baseline_values) < 10:
            return None
        
        # Calculate baseline statistics
        baseline_mean = statistics.mean(baseline_values[:-1])  # Exclude current value
        baseline_std = statistics.stdev(baseline_values[:-1]) if len(baseline_values) > 2 else 0
        
        # Detect latency spikes
        if 'latency' in metric_name.lower() or 'response_time' in metric_name.lower():
            return await self._detect_latency_anomaly(
                tenant_id, metric_name, current_value, baseline_mean, baseline_std
            )
        
        # Detect throughput drops
        elif 'throughput' in metric_name.lower() or 'requests_per' in metric_name.lower():
            return await self._detect_throughput_anomaly(
                tenant_id, metric_name, current_value, baseline_mean, baseline_std
            )
        
        # Detect error rate increases
        elif 'error' in metric_name.lower():
            return await self._detect_error_rate_anomaly(
                tenant_id, metric_name, current_value, baseline_mean, baseline_std
            )
        
        # Detect memory leaks
        elif 'memory' in metric_name.lower():
            return await self._detect_memory_anomaly(
                tenant_id, metric_name, current_value, baseline_values
            )
        
        # Detect CPU saturation
        elif 'cpu' in metric_name.lower():
            return await self._detect_cpu_anomaly(
                tenant_id, metric_name, current_value, baseline_mean
            )
        
        return None
    
    async def _detect_latency_anomaly(
        self,
        tenant_id: int,
        metric_name: str,
        current_value: float,
        baseline_mean: float,
        baseline_std: float
    ) -> Optional[PerformanceAnomaly]:
        """Detect latency spike anomaly"""
        
        threshold_config = self.anomaly_thresholds['latency_spike']
        
        if baseline_std == 0:
            return None
        
        # Z-score based detection
        z_score = abs((current_value - baseline_mean) / baseline_std)
        deviation_percent = ((current_value - baseline_mean) / baseline_mean) * 100
        
        if (deviation_percent > threshold_config['deviation_percent'] and 
            z_score > 2.0):  # 2 standard deviations
            
            confidence_score = min(0.99, z_score / 4.0)  # Normalize to 0-1
            
            if confidence_score >= threshold_config['confidence_threshold']:
                return PerformanceAnomaly(
                    anomaly_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    anomaly_type=PerformanceAnomalyType.LATENCY_SPIKE,
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_value=baseline_mean,
                    deviation_percent=deviation_percent,
                    severity=self._calculate_severity(deviation_percent),
                    confidence_score=confidence_score,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                    duration_minutes=1,  # Will be updated if anomaly persists
                    root_cause_analysis={
                        'z_score': z_score,
                        'baseline_mean': baseline_mean,
                        'baseline_std': baseline_std,
                        'possible_causes': ['resource_contention', 'network_latency', 'database_slowdown']
                    },
                    recommended_actions=[
                        'scale_up_resources',
                        'check_database_performance',
                        'investigate_network_latency'
                    ]
                )
        
        return None
    
    async def _detect_throughput_anomaly(
        self,
        tenant_id: int,
        metric_name: str,
        current_value: float,
        baseline_mean: float,
        baseline_std: float
    ) -> Optional[PerformanceAnomaly]:
        """Detect throughput drop anomaly"""
        
        threshold_config = self.anomaly_thresholds['throughput_drop']
        
        if baseline_mean == 0:
            return None
        
        deviation_percent = ((baseline_mean - current_value) / baseline_mean) * 100
        
        if deviation_percent > threshold_config['deviation_percent']:
            confidence_score = min(0.99, deviation_percent / 50.0)
            
            if confidence_score >= threshold_config['confidence_threshold']:
                return PerformanceAnomaly(
                    anomaly_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    anomaly_type=PerformanceAnomalyType.THROUGHPUT_DROP,
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_value=baseline_mean,
                    deviation_percent=deviation_percent,
                    severity=self._calculate_severity(deviation_percent),
                    confidence_score=confidence_score,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                    duration_minutes=1,
                    root_cause_analysis={
                        'baseline_mean': baseline_mean,
                        'possible_causes': ['resource_exhaustion', 'scaling_issues', 'downstream_bottleneck']
                    },
                    recommended_actions=[
                        'check_resource_utilization',
                        'investigate_scaling_policies',
                        'check_downstream_services'
                    ]
                )
        
        return None
    
    async def _detect_error_rate_anomaly(
        self,
        tenant_id: int,
        metric_name: str,
        current_value: float,
        baseline_mean: float,
        baseline_std: float
    ) -> Optional[PerformanceAnomaly]:
        """Detect error rate increase anomaly"""
        
        threshold_config = self.anomaly_thresholds['error_rate_increase']
        
        if baseline_mean == 0:
            baseline_mean = 0.1  # Assume very low baseline
        
        deviation_percent = ((current_value - baseline_mean) / baseline_mean) * 100
        
        if deviation_percent > threshold_config['deviation_percent']:
            confidence_score = min(0.99, deviation_percent / 200.0)
            
            if confidence_score >= threshold_config['confidence_threshold']:
                return PerformanceAnomaly(
                    anomaly_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    anomaly_type=PerformanceAnomalyType.ERROR_RATE_INCREASE,
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_value=baseline_mean,
                    deviation_percent=deviation_percent,
                    severity=self._calculate_severity(deviation_percent),
                    confidence_score=confidence_score,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                    duration_minutes=1,
                    root_cause_analysis={
                        'baseline_mean': baseline_mean,
                        'possible_causes': ['service_degradation', 'configuration_error', 'dependency_failure']
                    },
                    recommended_actions=[
                        'check_service_health',
                        'review_recent_deployments',
                        'investigate_dependencies'
                    ]
                )
        
        return None
    
    async def _detect_memory_anomaly(
        self,
        tenant_id: int,
        metric_name: str,
        current_value: float,
        baseline_values: List[float]
    ) -> Optional[PerformanceAnomaly]:
        """Detect memory leak anomaly"""
        
        if len(baseline_values) < 20:
            return None
        
        # Calculate trend over recent values
        recent_values = baseline_values[-20:]
        trend = self._calculate_trend(recent_values)
        
        threshold_config = self.anomaly_thresholds['memory_leak']
        
        if trend > threshold_config['trend_threshold']:  # Positive trend indicating growth
            confidence_score = min(0.99, trend / 20.0)
            
            if confidence_score >= threshold_config['confidence_threshold']:
                return PerformanceAnomaly(
                    anomaly_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    anomaly_type=PerformanceAnomalyType.MEMORY_LEAK,
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_value=recent_values[0],
                    deviation_percent=((current_value - recent_values[0]) / recent_values[0]) * 100,
                    severity=self._calculate_severity(trend * 5),  # Amplify for severity
                    confidence_score=confidence_score,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                    duration_minutes=20,  # Based on observation window
                    root_cause_analysis={
                        'trend_slope': trend,
                        'possible_causes': ['memory_leak', 'inefficient_caching', 'resource_accumulation']
                    },
                    recommended_actions=[
                        'investigate_memory_usage',
                        'check_for_memory_leaks',
                        'review_caching_strategy'
                    ]
                )
        
        return None
    
    async def _detect_cpu_anomaly(
        self,
        tenant_id: int,
        metric_name: str,
        current_value: float,
        baseline_mean: float
    ) -> Optional[PerformanceAnomaly]:
        """Detect CPU saturation anomaly"""
        
        threshold_config = self.anomaly_thresholds['cpu_saturation']
        
        if current_value > threshold_config['absolute_threshold']:
            deviation_percent = ((current_value - baseline_mean) / baseline_mean) * 100
            confidence_score = min(0.99, current_value / 100.0)
            
            if confidence_score >= threshold_config['confidence_threshold']:
                return PerformanceAnomaly(
                    anomaly_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    anomaly_type=PerformanceAnomalyType.CPU_SATURATION,
                    metric_name=metric_name,
                    current_value=current_value,
                    expected_value=baseline_mean,
                    deviation_percent=deviation_percent,
                    severity=self._calculate_severity(current_value),
                    confidence_score=confidence_score,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                    duration_minutes=1,
                    root_cause_analysis={
                        'baseline_mean': baseline_mean,
                        'possible_causes': ['cpu_intensive_workload', 'inefficient_algorithms', 'resource_contention']
                    },
                    recommended_actions=[
                        'scale_up_cpu_resources',
                        'optimize_algorithms',
                        'investigate_workload_patterns'
                    ]
                )
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for values"""
        
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
    
    def _calculate_severity(self, deviation_percent: float) -> str:
        """Calculate severity based on deviation percentage"""
        
        if deviation_percent > 100:
            return "critical"
        elif deviation_percent > 50:
            return "high"
        elif deviation_percent > 25:
            return "medium"
        else:
            return "low"

class NoisyNeighborMitigator:
    """
    Noisy neighbor detection and mitigation
    Task 6.6-T35: Document noisy neighbor mitigation policies
    """
    
    def __init__(self):
        self.resource_usage_history: Dict[int, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=50)))
        self.mitigation_thresholds = {
            'cpu': {'warning': 80.0, 'critical': 95.0},
            'memory': {'warning': 85.0, 'critical': 95.0},
            'network': {'warning': 80.0, 'critical': 90.0},
            'storage': {'warning': 85.0, 'critical': 95.0}
        }
        self.mitigation_actions = {
            'warning': NoisyNeighborAction.ALERT_ONLY,
            'critical': NoisyNeighborAction.THROTTLE
        }
    
    async def detect_noisy_neighbors(
        self,
        tenant_usage: Dict[int, Dict[str, float]]
    ) -> List[NoisyNeighborDetection]:
        """Detect tenants consuming excessive resources"""
        
        detections = []
        
        for tenant_id, usage_metrics in tenant_usage.items():
            # Update usage history
            for resource_type, usage_value in usage_metrics.items():
                self.resource_usage_history[tenant_id][resource_type].append(usage_value)
            
            # Check each resource type
            for resource_type, usage_value in usage_metrics.items():
                detection = await self._check_resource_usage(
                    tenant_id, resource_type, usage_value, tenant_usage
                )
                
                if detection:
                    detections.append(detection)
        
        return detections
    
    async def _check_resource_usage(
        self,
        tenant_id: int,
        resource_type: str,
        usage_percent: float,
        all_tenant_usage: Dict[int, Dict[str, float]]
    ) -> Optional[NoisyNeighborDetection]:
        """Check if tenant is consuming excessive resources"""
        
        thresholds = self.mitigation_thresholds.get(resource_type, {'warning': 80.0, 'critical': 95.0})
        
        severity_level = None
        if usage_percent >= thresholds['critical']:
            severity_level = 'critical'
        elif usage_percent >= thresholds['warning']:
            severity_level = 'warning'
        
        if not severity_level:
            return None
        
        # Calculate impact on other tenants
        affected_tenants = []
        for other_tenant_id, other_usage in all_tenant_usage.items():
            if other_tenant_id != tenant_id:
                other_resource_usage = other_usage.get(resource_type, 0)
                # If other tenants are also high, this tenant might be impacting them
                if other_resource_usage > 70.0:  # Threshold for being affected
                    affected_tenants.append(other_tenant_id)
        
        # Determine mitigation action
        mitigation_action = self.mitigation_actions.get(severity_level, NoisyNeighborAction.ALERT_ONLY)
        
        return NoisyNeighborDetection(
            detection_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            resource_type=resource_type,
            usage_percent=usage_percent,
            threshold_percent=thresholds[severity_level],
            impact_on_others=affected_tenants,
            detected_at=datetime.now(timezone.utc).isoformat(),
            duration_minutes=self._calculate_duration(tenant_id, resource_type, thresholds[severity_level]),
            mitigation_action=mitigation_action,
            action_taken=False
        )
    
    def _calculate_duration(self, tenant_id: int, resource_type: str, threshold: float) -> int:
        """Calculate how long tenant has been above threshold"""
        
        usage_history = list(self.resource_usage_history[tenant_id][resource_type])
        
        if len(usage_history) < 2:
            return 1
        
        # Count consecutive measurements above threshold
        duration = 0
        for usage in reversed(usage_history):
            if usage >= threshold:
                duration += 1
            else:
                break
        
        return max(1, duration)  # At least 1 minute

class PerformanceOptimizer:
    """
    Comprehensive Performance Optimization System
    Tasks 6.6-T24, T26, T27, T35, T37: Fallback behavior, anomaly detection, noisy neighbor mitigation, governance rules
    """
    
    def __init__(self, tenant_context_manager=None, auto_scaling_manager=None):
        self.tenant_context_manager = tenant_context_manager
        self.auto_scaling_manager = auto_scaling_manager
        
        # Core components
        self.anomaly_detector = PerformanceAnomalyDetector()
        self.noisy_neighbor_mitigator = NoisyNeighborMitigator()
        
        # Storage for configurations and state
        self.fallback_configurations: Dict[str, FallbackConfiguration] = {}
        self.scaling_overrides: Dict[str, ScalingOverride] = {}
        self.detected_anomalies: List[PerformanceAnomaly] = []
        self.noisy_neighbor_detections: List[NoisyNeighborDetection] = []
        
        # Initialize default configurations
        self._initialize_fallback_configurations()
    
    def _initialize_fallback_configurations(self) -> None:
        """Initialize default fallback configurations"""
        
        fallback_configs = [
            # SaaS fallback configuration
            {
                'config_id': 'saas_fallback',
                'tenant_id': None,
                'industry_code': 'SaaS',
                'tenant_tier': 'professional',
                'trigger_conditions': ['scaling_failure', 'resource_exhaustion', 'high_error_rate'],
                'fallback_mode': FallbackMode.SAFE_MODE,
                'max_requests_per_minute': 1000,
                'max_concurrent_workflows': 50,
                'timeout_seconds': 30,
                'retry_attempts': 3
            },
            # Banking fallback configuration (more conservative)
            {
                'config_id': 'banking_fallback',
                'tenant_id': None,
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'trigger_conditions': ['scaling_failure', 'compliance_violation', 'security_alert'],
                'fallback_mode': FallbackMode.CIRCUIT_BREAKER,
                'max_requests_per_minute': 500,
                'max_concurrent_workflows': 20,
                'timeout_seconds': 60,
                'retry_attempts': 1
            },
            # Insurance fallback configuration
            {
                'config_id': 'insurance_fallback',
                'tenant_id': None,
                'industry_code': 'Insurance',
                'tenant_tier': 'enterprise',
                'trigger_conditions': ['scaling_failure', 'data_quality_issue', 'regulatory_alert'],
                'fallback_mode': FallbackMode.DEGRADED_SERVICE,
                'max_requests_per_minute': 750,
                'max_concurrent_workflows': 30,
                'timeout_seconds': 45,
                'retry_attempts': 2
            }
        ]
        
        for config_data in fallback_configs:
            config = FallbackConfiguration(
                config_id=config_data['config_id'],
                tenant_id=config_data['tenant_id'],
                industry_code=config_data['industry_code'],
                tenant_tier=config_data['tenant_tier'],
                trigger_conditions=config_data['trigger_conditions'],
                fallback_mode=config_data['fallback_mode'],
                max_requests_per_minute=config_data['max_requests_per_minute'],
                max_concurrent_workflows=config_data['max_concurrent_workflows'],
                timeout_seconds=config_data['timeout_seconds'],
                retry_attempts=config_data['retry_attempts'],
                enabled=True,
                created_at=datetime.now(timezone.utc).isoformat()
            )
            self.fallback_configurations[config.config_id] = config
    
    async def analyze_performance(
        self,
        tenant_id: int,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        
        analysis_start = time.time()
        
        # Detect anomalies
        anomalies = await self.anomaly_detector.detect_anomalies(tenant_id, current_metrics)
        
        # Store detected anomalies
        self.detected_anomalies.extend(anomalies)
        
        # Keep only recent anomalies (last 1000)
        self.detected_anomalies = self.detected_anomalies[-1000:]
        
        # Check for fallback triggers
        fallback_recommendation = await self._check_fallback_triggers(tenant_id, current_metrics, anomalies)
        
        # Generate optimization recommendations
        optimization_recommendations = await self._generate_optimization_recommendations(
            tenant_id, current_metrics, anomalies
        )
        
        analysis_time = (time.time() - analysis_start) * 1000
        
        return {
            'tenant_id': tenant_id,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_time_ms': analysis_time,
            'anomalies_detected': [asdict(anomaly) for anomaly in anomalies],
            'fallback_recommendation': fallback_recommendation,
            'optimization_recommendations': optimization_recommendations,
            'overall_health_score': self._calculate_health_score(current_metrics, anomalies)
        }
    
    async def detect_noisy_neighbors(
        self,
        tenant_usage_data: Dict[int, Dict[str, float]]
    ) -> List[NoisyNeighborDetection]:
        """Detect and mitigate noisy neighbors"""
        
        detections = await self.noisy_neighbor_mitigator.detect_noisy_neighbors(tenant_usage_data)
        
        # Store detections
        self.noisy_neighbor_detections.extend(detections)
        
        # Keep only recent detections (last 500)
        self.noisy_neighbor_detections = self.noisy_neighbor_detections[-500:]
        
        # Apply mitigation actions
        for detection in detections:
            await self._apply_noisy_neighbor_mitigation(detection)
        
        return detections
    
    async def _check_fallback_triggers(
        self,
        tenant_id: int,
        current_metrics: Dict[str, float],
        anomalies: List[PerformanceAnomaly]
    ) -> Dict[str, Any]:
        """Check if fallback mode should be triggered"""
        
        # Get tenant context
        tenant_context = None
        if self.tenant_context_manager:
            tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
        
        industry_code = tenant_context.industry_code.value if tenant_context else 'SaaS'
        tenant_tier = tenant_context.tenant_tier.value if tenant_context else 'professional'
        
        # Find applicable fallback configuration
        applicable_config = None
        for config in self.fallback_configurations.values():
            if (config.industry_code == industry_code and 
                config.tenant_tier == tenant_tier and 
                config.enabled):
                applicable_config = config
                break
        
        if not applicable_config:
            return {'fallback_required': False, 'reason': 'No applicable fallback configuration'}
        
        # Check trigger conditions
        triggered_conditions = []
        
        # Check for scaling failure (mock - would integrate with actual scaling system)
        if current_metrics.get('scaling_failures', 0) > 0:
            triggered_conditions.append('scaling_failure')
        
        # Check for resource exhaustion
        if (current_metrics.get('cpu_utilization', 0) > 95 or 
            current_metrics.get('memory_utilization', 0) > 95):
            triggered_conditions.append('resource_exhaustion')
        
        # Check for high error rate
        if current_metrics.get('error_rate', 0) > 10:
            triggered_conditions.append('high_error_rate')
        
        # Check for critical anomalies
        critical_anomalies = [a for a in anomalies if a.severity == 'critical']
        if critical_anomalies:
            triggered_conditions.append('critical_anomalies')
        
        # Determine if fallback should be triggered
        matching_conditions = [
            condition for condition in triggered_conditions
            if condition in applicable_config.trigger_conditions
        ]
        
        fallback_required = len(matching_conditions) > 0
        
        return {
            'fallback_required': fallback_required,
            'triggered_conditions': triggered_conditions,
            'matching_conditions': matching_conditions,
            'recommended_fallback_mode': applicable_config.fallback_mode.value if fallback_required else None,
            'fallback_config': asdict(applicable_config) if fallback_required else None
        }
    
    async def _generate_optimization_recommendations(
        self,
        tenant_id: int,
        current_metrics: Dict[str, float],
        anomalies: List[PerformanceAnomaly]
    ) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        # Resource utilization recommendations
        cpu_util = current_metrics.get('cpu_utilization', 0)
        memory_util = current_metrics.get('memory_utilization', 0)
        
        if cpu_util > 80:
            recommendations.append({
                'type': 'resource_optimization',
                'priority': 'high' if cpu_util > 90 else 'medium',
                'recommendation': 'Scale up CPU resources',
                'reason': f'CPU utilization is high at {cpu_util:.1f}%',
                'estimated_impact': 'Reduce latency by 20-30%'
            })
        
        if memory_util > 85:
            recommendations.append({
                'type': 'resource_optimization',
                'priority': 'high' if memory_util > 95 else 'medium',
                'recommendation': 'Scale up memory resources',
                'reason': f'Memory utilization is high at {memory_util:.1f}%',
                'estimated_impact': 'Prevent out-of-memory errors'
            })
        
        # Anomaly-based recommendations
        for anomaly in anomalies:
            for action in anomaly.recommended_actions:
                recommendations.append({
                    'type': 'anomaly_mitigation',
                    'priority': anomaly.severity,
                    'recommendation': action,
                    'reason': f'Detected {anomaly.anomaly_type.value} anomaly',
                    'estimated_impact': 'Address performance degradation'
                })
        
        # Performance pattern recommendations
        error_rate = current_metrics.get('error_rate', 0)
        if error_rate > 5:
            recommendations.append({
                'type': 'reliability_improvement',
                'priority': 'high',
                'recommendation': 'Implement circuit breaker pattern',
                'reason': f'Error rate is elevated at {error_rate:.1f}%',
                'estimated_impact': 'Improve system resilience'
            })
        
        return recommendations
    
    async def _apply_noisy_neighbor_mitigation(
        self,
        detection: NoisyNeighborDetection
    ) -> None:
        """Apply mitigation action for noisy neighbor"""
        
        try:
            if detection.mitigation_action == NoisyNeighborAction.THROTTLE:
                # Apply throttling (mock implementation)
                logger.warning(f"Applying throttling to tenant {detection.tenant_id} for {detection.resource_type}")
                detection.action_taken = True
                
            elif detection.mitigation_action == NoisyNeighborAction.ISOLATE:
                # Apply isolation (mock implementation)
                logger.warning(f"Isolating tenant {detection.tenant_id} for excessive {detection.resource_type} usage")
                detection.action_taken = True
                
            elif detection.mitigation_action == NoisyNeighborAction.SUSPEND:
                # Suspend tenant (mock implementation)
                logger.critical(f"Suspending tenant {detection.tenant_id} for excessive {detection.resource_type} usage")
                detection.action_taken = True
                
            elif detection.mitigation_action == NoisyNeighborAction.ALERT_ONLY:
                # Send alert only
                logger.info(f"Alert: Tenant {detection.tenant_id} has high {detection.resource_type} usage")
                detection.action_taken = True
                
        except Exception as e:
            logger.error(f"Failed to apply mitigation for tenant {detection.tenant_id}: {e}")
            detection.action_taken = False
    
    def _calculate_health_score(
        self,
        current_metrics: Dict[str, float],
        anomalies: List[PerformanceAnomaly]
    ) -> float:
        """Calculate overall health score (0-100)"""
        
        base_score = 100.0
        
        # Deduct for high resource utilization
        cpu_util = current_metrics.get('cpu_utilization', 0)
        memory_util = current_metrics.get('memory_utilization', 0)
        
        if cpu_util > 80:
            base_score -= (cpu_util - 80) * 0.5
        
        if memory_util > 80:
            base_score -= (memory_util - 80) * 0.5
        
        # Deduct for errors
        error_rate = current_metrics.get('error_rate', 0)
        base_score -= error_rate * 2
        
        # Deduct for anomalies
        for anomaly in anomalies:
            if anomaly.severity == 'critical':
                base_score -= 20
            elif anomaly.severity == 'high':
                base_score -= 10
            elif anomaly.severity == 'medium':
                base_score -= 5
            else:
                base_score -= 2
        
        return max(0.0, min(100.0, base_score))
    
    async def create_scaling_override(
        self,
        tenant_id: int,
        requested_by: str,
        override_type: str,
        justification: str,
        expiry_hours: int = 24
    ) -> ScalingOverride:
        """Create scaling governance override"""
        
        override = ScalingOverride(
            override_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            requested_by=requested_by,
            override_type=override_type,
            justification=justification,
            approval_status='pending',
            approved_by=None,
            approved_at=None,
            expiry_time=(datetime.now(timezone.utc) + timedelta(hours=expiry_hours)).isoformat(),
            applied=False,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        self.scaling_overrides[override.override_id] = override
        
        logger.info(f"Created scaling override request: {override.override_id} for tenant {tenant_id}")
        return override
    
    async def approve_scaling_override(
        self,
        override_id: str,
        approved_by: str
    ) -> bool:
        """Approve scaling override"""
        
        if override_id not in self.scaling_overrides:
            return False
        
        override = self.scaling_overrides[override_id]
        override.approval_status = 'approved'
        override.approved_by = approved_by
        override.approved_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Approved scaling override: {override_id} by {approved_by}")
        return True
    
    async def get_performance_summary(self, tenant_id: int) -> Dict[str, Any]:
        """Get performance summary for tenant"""
        
        # Get recent anomalies
        recent_anomalies = [
            anomaly for anomaly in self.detected_anomalies[-100:]
            if anomaly.tenant_id == tenant_id
        ]
        
        # Get recent noisy neighbor detections
        recent_detections = [
            detection for detection in self.noisy_neighbor_detections[-50:]
            if detection.tenant_id == tenant_id
        ]
        
        # Get pending overrides
        pending_overrides = [
            override for override in self.scaling_overrides.values()
            if override.tenant_id == tenant_id and override.approval_status == 'pending'
        ]
        
        return {
            'tenant_id': tenant_id,
            'recent_anomalies': [asdict(anomaly) for anomaly in recent_anomalies[-10:]],
            'noisy_neighbor_detections': [asdict(detection) for detection in recent_detections[-5:]],
            'pending_overrides': [asdict(override) for override in pending_overrides],
            'anomaly_summary': {
                'total_anomalies_24h': len([a for a in recent_anomalies if 
                    datetime.fromisoformat(a.detected_at.replace('Z', '+00:00')) > 
                    datetime.now(timezone.utc) - timedelta(hours=24)]),
                'critical_anomalies_24h': len([a for a in recent_anomalies if 
                    a.severity == 'critical' and
                    datetime.fromisoformat(a.detected_at.replace('Z', '+00:00')) > 
                    datetime.now(timezone.utc) - timedelta(hours=24)])
            },
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

# Global performance optimizer
performance_optimizer = PerformanceOptimizer()
