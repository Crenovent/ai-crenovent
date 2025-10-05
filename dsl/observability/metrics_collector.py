#!/usr/bin/env python3
"""
Metrics Collector for RBA Workflows
===================================

Implements Tasks 6.2-T23, 6.3-T15 to T17, 6.7-T07 to 6.7-T12 and 20.1-T04 to 20.1-T15:
- Metrics collection service (latency, throughput, SLA hit rate)
- Error rate and governance metrics
- Error handling framework metrics (MTTR, retry success rate, override frequency)
- Escalation tracking and SLA monitoring
- Circuit breaker state monitoring
- Adoption and business metrics
- Tenant, industry, and region labeling
- Prometheus integration

Features:
- Multi-tenant metrics with proper labeling
- Business KPI tracking and correlation
- Governance metrics (policy compliance, override frequency)
- SLA adherence monitoring
- Industry overlay metrics
- Real-time metric aggregation and export
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import defaultdict, deque
import statistics

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    from prometheus_client.exposition import MetricsHandler
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock implementations
    class MockMetric:
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class MockCollectorRegistry:
        pass
    
    Counter = Histogram = Gauge = Info = MockMetric
    CollectorRegistry = MockCollectorRegistry
    
    def generate_latest(registry):
        return b"# Prometheus not available\n"

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of RBA metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SLA_ADHERENCE = "sla_adherence"
    GOVERNANCE_COMPLIANCE = "governance_compliance"
    TRUST_SCORE = "trust_score"
    BUSINESS_KPI = "business_kpi"

@dataclass
class RBAMetrics:
    """Standard RBA metrics data structure"""
    metric_type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    tenant_id: int
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    industry_code: str = 'SaaS'
    region: str = 'US'
    
    def to_prometheus_labels(self) -> Dict[str, str]:
        """Convert to Prometheus label format"""
        base_labels = {
            'tenant_id': str(self.tenant_id),
            'industry_code': self.industry_code,
            'region': self.region,
            'metric_type': self.metric_type.value
        }
        
        if self.workflow_id:
            base_labels['workflow_id'] = self.workflow_id
        if self.execution_id:
            base_labels['execution_id'] = self.execution_id
        
        # Add custom labels
        base_labels.update(self.labels)
        
        return base_labels

class MetricsCollector:
    """
    Enhanced metrics collector for RBA workflows
    
    Implements comprehensive metrics collection with:
    - Multi-tenant labeling and isolation
    - Business KPI correlation
    - Governance and compliance metrics
    - Real-time aggregation and export
    - Industry overlay support
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.metrics_buffer = deque(maxlen=10000)  # Rolling buffer
        self.aggregation_cache = {}
        self.cache_ttl = 60  # 1 minute cache
        
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        else:
            logger.warning("⚠️ Prometheus client not available, using mock metrics")
            self._setup_mock_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics with proper labeling (Tasks 20.1-T13 to 20.1-T15)"""
        
        # Common label names for multi-tenant metrics
        common_labels = ['tenant_id', 'industry_code', 'region', 'workflow_id']
        
        # Latency metrics (Task 20.1-T04, 20.1-T05)
        self.workflow_latency = Histogram(
            'rba_workflow_execution_duration_seconds',
            'Time spent executing RBA workflows',
            labelnames=common_labels + ['status'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float('inf'))
        )
        
        self.step_latency = Histogram(
            'rba_step_execution_duration_seconds',
            'Time spent executing individual workflow steps',
            labelnames=common_labels + ['step_name', 'step_type'],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float('inf'))
        )
        
        # Throughput metrics (Task 20.1-T05)
        self.workflow_executions_total = Counter(
            'rba_workflow_executions_total',
            'Total number of workflow executions',
            labelnames=common_labels + ['status'],
            registry=self.registry
        )
        
        self.step_executions_total = Counter(
            'rba_step_executions_total',
            'Total number of step executions',
            labelnames=common_labels + ['step_name', 'step_type', 'status'],
            registry=self.registry
        )
        
        # Error rate metrics (Task 6.7-T08, 20.1-T06)
        self.workflow_errors_total = Counter(
            'rba_workflow_errors_total',
            'Total number of workflow errors',
            labelnames=common_labels + ['error_type', 'error_category'],
            registry=self.registry
        )
        
        self.retry_attempts_total = Counter(
            'rba_retry_attempts_total',
            'Total number of retry attempts',
            labelnames=common_labels + ['retry_reason'],
            registry=self.registry
        )
        
        # Governance metrics (Task 6.7-T10, 20.1-T09)
        self.policy_evaluations_total = Counter(
            'rba_policy_evaluations_total',
            'Total number of policy evaluations',
            labelnames=common_labels + ['policy_id', 'result'],
            registry=self.registry
        )
        
        self.override_frequency = Counter(
            'rba_overrides_total',
            'Total number of policy overrides',
            labelnames=common_labels + ['override_type', 'approver_role'],
            registry=self.registry
        )
        
        self.trust_score = Gauge(
            'rba_trust_score',
            'Current trust score for workflows',
            labelnames=common_labels + ['trust_level'],
            registry=self.registry
        )
        
        # Evidence and audit metrics (Task 20.1-T08)
        self.evidence_packs_generated = Counter(
            'rba_evidence_packs_total',
            'Total number of evidence packs generated',
            labelnames=common_labels + ['pack_type', 'compliance_framework'],
            registry=self.registry
        )
        
        self.audit_events_total = Counter(
            'rba_audit_events_total',
            'Total number of audit events',
            labelnames=common_labels + ['event_type', 'severity'],
            registry=self.registry
        )
        
        # Chapter 20 Enhanced Metrics - Risk Dashboard Metrics (Tasks 20.3-T05 to T09)
        self.risk_events_total = Counter(
            'rba_risk_events_total',
            'Total number of risk events detected',
            labelnames=common_labels + ['risk_category', 'severity_level', 'source_service'],
            registry=self.registry
        )
        
        self.risk_severity_gauge = Gauge(
            'rba_risk_severity_current',
            'Current risk severity level by category',
            labelnames=common_labels + ['risk_category'],
            registry=self.registry
        )
        
        self.governance_anomalies_total = Counter(
            'rba_governance_anomalies_total',
            'Total number of governance anomalies detected',
            labelnames=common_labels + ['anomaly_type', 'detection_method'],
            registry=self.registry
        )
        
        self.stress_test_outcomes = Counter(
            'rba_stress_test_outcomes_total',
            'Total stress test outcomes',
            labelnames=common_labels + ['test_type', 'outcome', 'scenario'],
            registry=self.registry
        )
        
        self.rollback_events_total = Counter(
            'rba_rollback_events_total',
            'Total number of rollback events',
            labelnames=common_labels + ['rollback_reason', 'rollback_type'],
            registry=self.registry
        )
        
        # Chapter 20 Enhanced Governance Metrics (Tasks 20.2-T03 to T08)
        self.policy_enforcement_latency = Histogram(
            'rba_policy_enforcement_duration_seconds',
            'Time spent enforcing policies',
            labelnames=common_labels + ['policy_type', 'enforcement_result'],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float('inf'))
        )
        
        self.sod_violations_total = Counter(
            'rba_sod_violations_total',
            'Total number of Segregation of Duties violations',
            labelnames=common_labels + ['violation_type', 'severity'],
            registry=self.registry
        )
        
        self.trust_score_updates_total = Counter(
            'rba_trust_score_updates_total',
            'Total number of trust score updates',
            labelnames=common_labels + ['update_trigger', 'score_change_direction'],
            registry=self.registry
        )
        
        self.risk_register_updates_total = Counter(
            'rba_risk_register_updates_total',
            'Total number of risk register updates',
            labelnames=common_labels + ['update_type', 'risk_level'],
            registry=self.registry
        )
        
        # SLA metrics (Task 20.1-T21, 20.1-T22)
        self.sla_adherence = Gauge(
            'rba_sla_adherence_percentage',
            'SLA adherence percentage',
            labelnames=common_labels + ['sla_tier', 'metric_type'],
            registry=self.registry
        )
        
        self.sla_violations_total = Counter(
            'rba_sla_violations_total',
            'Total number of SLA violations',
            labelnames=common_labels + ['sla_tier', 'violation_type'],
            registry=self.registry
        )
        
        # Business KPI metrics (Task 6.7-T09, 20.1-T03)
        self.business_impact = Gauge(
            'rba_business_impact',
            'Business impact metrics',
            labelnames=common_labels + ['kpi_type', 'impact_category'],
            registry=self.registry
        )
        
        self.adoption_metrics = Gauge(
            'rba_adoption_metrics',
            'Workflow adoption metrics',
            labelnames=common_labels + ['adoption_type'],
            registry=self.registry
        )
        
        logger.info("✅ Prometheus metrics initialized with multi-tenant labeling")
    
    def _setup_mock_metrics(self):
        """Setup mock metrics for environments without Prometheus"""
        self.workflow_latency = MockMetric()
        self.step_latency = MockMetric()
        self.workflow_executions_total = MockMetric()
        self.step_executions_total = MockMetric()
        self.workflow_errors_total = MockMetric()
        self.retry_attempts_total = MockMetric()
        self.policy_evaluations_total = MockMetric()
        self.override_frequency = MockMetric()
        self.trust_score = MockMetric()
        self.evidence_packs_generated = MockMetric()
        self.audit_events_total = MockMetric()
        self.sla_adherence = MockMetric()
        self.sla_violations_total = MockMetric()
        self.business_impact = MockMetric()
        self.adoption_metrics = MockMetric()
    
    def record_workflow_execution(self, 
                                workflow_id: str,
                                execution_time_seconds: float,
                                status: str,
                                tenant_id: int,
                                industry_code: str = 'SaaS',
                                region: str = 'US',
                                **kwargs):
        """
        Record workflow execution metrics (Task 20.1-T05)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'status': status
            }
            
            # Record latency
            self.workflow_latency.labels(**labels).observe(execution_time_seconds)
            
            # Record execution count
            self.workflow_executions_total.labels(**labels).inc()
            
            # Store in buffer for aggregation
            metric = RBAMetrics(
                metric_type=MetricType.LATENCY,
                value=execution_time_seconds,
                labels={'status': status},
                timestamp=datetime.utcnow(),
                tenant_id=tenant_id,
                workflow_id=workflow_id,
                industry_code=industry_code,
                region=region
            )
            self.metrics_buffer.append(metric)
            
            logger.debug(f"📊 Recorded workflow execution: {workflow_id} ({execution_time_seconds:.3f}s)")
            
        except Exception as e:
            logger.error(f"❌ Failed to record workflow execution metrics: {e}")
    
    def record_step_execution(self,
                            workflow_id: str,
                            step_name: str,
                            step_type: str,
                            execution_time_seconds: float,
                            status: str,
                            tenant_id: int,
                            industry_code: str = 'SaaS',
                            region: str = 'US',
                            **kwargs):
        """
        Record step execution metrics with step-level granularity (Task 20.1-T05)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'step_name': step_name,
                'step_type': step_type,
                'status': status
            }
            
            # Record step latency
            self.step_latency.labels(**{k: v for k, v in labels.items() if k != 'status'}).observe(execution_time_seconds)
            
            # Record step execution count
            self.step_executions_total.labels(**labels).inc()
            
        except Exception as e:
            logger.error(f"❌ Failed to record step execution metrics: {e}")
    
    def record_policy_evaluation(self,
                               workflow_id: str,
                               policy_id: str,
                               result: str,  # 'compliant', 'violation', 'override'
                               tenant_id: int,
                               industry_code: str = 'SaaS',
                               region: str = 'US',
                               **kwargs):
        """
        Record policy evaluation metrics (Task 6.7-T10)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'policy_id': policy_id,
                'result': result
            }
            
            self.policy_evaluations_total.labels(**labels).inc()
            
            # Store governance metric
            metric = RBAMetrics(
                metric_type=MetricType.GOVERNANCE_COMPLIANCE,
                value=1.0 if result == 'compliant' else 0.0,
                labels={'policy_id': policy_id, 'result': result},
                timestamp=datetime.utcnow(),
                tenant_id=tenant_id,
                workflow_id=workflow_id,
                industry_code=industry_code,
                region=region
            )
            self.metrics_buffer.append(metric)
            
        except Exception as e:
            logger.error(f"❌ Failed to record policy evaluation metrics: {e}")
    
    def record_override(self,
                       workflow_id: str,
                       override_type: str,
                       approver_role: str,
                       tenant_id: int,
                       industry_code: str = 'SaaS',
                       region: str = 'US',
                       **kwargs):
        """
        Record policy override metrics (Task 20.1-T09)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'override_type': override_type,
                'approver_role': approver_role
            }
            
            self.override_frequency.labels(**labels).inc()
            
        except Exception as e:
            logger.error(f"❌ Failed to record override metrics: {e}")
    
    def record_trust_score(self,
                          workflow_id: str,
                          trust_score: float,
                          trust_level: str,
                          tenant_id: int,
                          industry_code: str = 'SaaS',
                          region: str = 'US',
                          **kwargs):
        """
        Record trust score metrics (Task 14.2 integration)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'trust_level': trust_level
            }
            
            self.trust_score.labels(**labels).set(trust_score)
            
            # Store trust metric
            metric = RBAMetrics(
                metric_type=MetricType.TRUST_SCORE,
                value=trust_score,
                labels={'trust_level': trust_level},
                timestamp=datetime.utcnow(),
                tenant_id=tenant_id,
                workflow_id=workflow_id,
                industry_code=industry_code,
                region=region
            )
            self.metrics_buffer.append(metric)
            
        except Exception as e:
            logger.error(f"❌ Failed to record trust score metrics: {e}")
    
    def record_evidence_pack(self,
                           workflow_id: str,
                           pack_type: str,
                           compliance_framework: str,
                           tenant_id: int,
                           industry_code: str = 'SaaS',
                           region: str = 'US',
                           **kwargs):
        """
        Record evidence pack generation metrics (Task 20.1-T08)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'pack_type': pack_type,
                'compliance_framework': compliance_framework
            }
            
            self.evidence_packs_generated.labels(**labels).inc()
            
        except Exception as e:
            logger.error(f"❌ Failed to record evidence pack metrics: {e}")
    
    def record_sla_adherence(self,
                           sla_tier: str,
                           metric_type: str,
                           adherence_percentage: float,
                           tenant_id: int,
                           industry_code: str = 'SaaS',
                           region: str = 'US',
                           **kwargs):
        """
        Record SLA adherence metrics (Task 20.1-T21, 20.1-T22)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': kwargs.get('workflow_id', ''),
                'sla_tier': sla_tier,
                'metric_type': metric_type
            }
            
            self.sla_adherence.labels(**labels).set(adherence_percentage)
            
            # Record SLA violation if below threshold
            if adherence_percentage < 99.0:  # Configurable threshold
                violation_labels = {k: v for k, v in labels.items() if k != 'metric_type'}
                violation_labels['violation_type'] = f"{metric_type}_breach"
                self.sla_violations_total.labels(**violation_labels).inc()
            
        except Exception as e:
            logger.error(f"❌ Failed to record SLA adherence metrics: {e}")
    
    def record_business_impact(self,
                             kpi_type: str,
                             impact_value: float,
                             impact_category: str,
                             tenant_id: int,
                             workflow_id: Optional[str] = None,
                             industry_code: str = 'SaaS',
                             region: str = 'US',
                             **kwargs):
        """
        Record business KPI impact metrics (Task 20.1-T03)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id or '',
                'kpi_type': kpi_type,
                'impact_category': impact_category
            }
            
            self.business_impact.labels(**labels).set(impact_value)
            
            # Store business metric
            metric = RBAMetrics(
                metric_type=MetricType.BUSINESS_KPI,
                value=impact_value,
                labels={'kpi_type': kpi_type, 'impact_category': impact_category},
                timestamp=datetime.utcnow(),
                tenant_id=tenant_id,
                workflow_id=workflow_id,
                industry_code=industry_code,
                region=region
            )
            self.metrics_buffer.append(metric)
            
        except Exception as e:
            logger.error(f"❌ Failed to record business impact metrics: {e}")
    
    def get_aggregated_metrics(self, 
                             tenant_id: Optional[int] = None,
                             time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get aggregated metrics for dashboards and reporting
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            
            # Filter metrics by tenant and time window
            relevant_metrics = [
                m for m in self.metrics_buffer
                if m.timestamp >= cutoff_time and (tenant_id is None or m.tenant_id == tenant_id)
            ]
            
            if not relevant_metrics:
                return {}
            
            # Aggregate by metric type
            aggregated = {}
            
            for metric_type in MetricType:
                type_metrics = [m for m in relevant_metrics if m.metric_type == metric_type]
                
                if type_metrics:
                    values = [m.value for m in type_metrics]
                    aggregated[metric_type.value] = {
                        'count': len(values),
                        'avg': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                        'latest': values[-1] if values else 0
                    }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"❌ Failed to get aggregated metrics: {e}")
            return {}
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format (Task 20.1-T11)
        """
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus not available\n"
    
    def get_tenant_metrics_summary(self, tenant_id: int) -> Dict[str, Any]:
        """
        Get tenant-specific metrics summary for dashboards
        """
        try:
            tenant_metrics = [m for m in self.metrics_buffer if m.tenant_id == tenant_id]
            
            if not tenant_metrics:
                return {'tenant_id': tenant_id, 'metrics': {}}
            
            # Calculate summary statistics
            recent_metrics = [
                m for m in tenant_metrics 
                if m.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ]
            
            summary = {
                'tenant_id': tenant_id,
                'total_executions': len([m for m in recent_metrics if m.metric_type == MetricType.LATENCY]),
                'avg_latency_ms': statistics.mean([m.value * 1000 for m in recent_metrics if m.metric_type == MetricType.LATENCY]) if recent_metrics else 0,
                'compliance_rate': statistics.mean([m.value for m in recent_metrics if m.metric_type == MetricType.GOVERNANCE_COMPLIANCE]) if recent_metrics else 1.0,
                'trust_score': statistics.mean([m.value for m in recent_metrics if m.metric_type == MetricType.TRUST_SCORE]) if recent_metrics else 0.8,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get tenant metrics summary: {e}")
            return {'tenant_id': tenant_id, 'error': str(e)}
    
    # =====================================================
    # CHAPTER 20 ENHANCED EMITTERS
    # =====================================================
    
    def record_risk_event(self,
                         workflow_id: str,
                         risk_category: str,
                         severity_level: str,
                         source_service: str,
                         tenant_id: int,
                         industry_code: str = 'SaaS',
                         region: str = 'US',
                         **kwargs):
        """
        Record risk events for risk dashboard metrics (Task 20.3-T05)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'risk_category': risk_category,
                'severity_level': severity_level,
                'source_service': source_service
            }
            
            self.risk_events_total.labels(**labels).inc()
            
            # Update current risk severity gauge
            severity_labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'risk_category': risk_category
            }
            
            # Convert severity to numeric value
            severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            severity_value = severity_map.get(severity_level.lower(), 1)
            self.risk_severity_gauge.labels(**severity_labels).set(severity_value)
            
            logger.info(f"📊 Risk event recorded: {risk_category} - {severity_level}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record risk event metrics: {e}")
    
    def record_governance_anomaly(self,
                                workflow_id: str,
                                anomaly_type: str,
                                detection_method: str,
                                tenant_id: int,
                                industry_code: str = 'SaaS',
                                region: str = 'US',
                                **kwargs):
        """
        Record governance anomalies (Task 20.3-T12)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'anomaly_type': anomaly_type,
                'detection_method': detection_method
            }
            
            self.governance_anomalies_total.labels(**labels).inc()
            logger.info(f"📊 Governance anomaly recorded: {anomaly_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record governance anomaly metrics: {e}")
    
    def record_stress_test_outcome(self,
                                 workflow_id: str,
                                 test_type: str,
                                 outcome: str,
                                 scenario: str,
                                 tenant_id: int,
                                 industry_code: str = 'SaaS',
                                 region: str = 'US',
                                 **kwargs):
        """
        Record stress test outcomes (Task 20.3-T10)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'test_type': test_type,
                'outcome': outcome,
                'scenario': scenario
            }
            
            self.stress_test_outcomes.labels(**labels).inc()
            logger.info(f"📊 Stress test outcome recorded: {test_type} - {outcome}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record stress test outcome metrics: {e}")
    
    def record_rollback_event(self,
                            workflow_id: str,
                            rollback_reason: str,
                            rollback_type: str,
                            tenant_id: int,
                            industry_code: str = 'SaaS',
                            region: str = 'US',
                            **kwargs):
        """
        Record rollback events (Task 20.3-T11)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'rollback_reason': rollback_reason,
                'rollback_type': rollback_type
            }
            
            self.rollback_events_total.labels(**labels).inc()
            logger.info(f"📊 Rollback event recorded: {rollback_type} - {rollback_reason}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record rollback event metrics: {e}")
    
    def record_policy_enforcement(self,
                                workflow_id: str,
                                policy_type: str,
                                enforcement_result: str,
                                enforcement_duration: float,
                                tenant_id: int,
                                industry_code: str = 'SaaS',
                                region: str = 'US',
                                **kwargs):
        """
        Record policy enforcement metrics (Task 20.2-T03)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'policy_type': policy_type,
                'enforcement_result': enforcement_result
            }
            
            self.policy_enforcement_latency.labels(**labels).observe(enforcement_duration)
            logger.info(f"📊 Policy enforcement recorded: {policy_type} - {enforcement_duration:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to record policy enforcement metrics: {e}")
    
    def record_sod_violation(self,
                           workflow_id: str,
                           violation_type: str,
                           severity: str,
                           tenant_id: int,
                           industry_code: str = 'SaaS',
                           region: str = 'US',
                           **kwargs):
        """
        Record Segregation of Duties violations (Task 20.2-T05)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'violation_type': violation_type,
                'severity': severity
            }
            
            self.sod_violations_total.labels(**labels).inc()
            logger.warning(f"⚠️ SoD violation recorded: {violation_type} - {severity}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record SoD violation metrics: {e}")
    
    def record_trust_score_update(self,
                                workflow_id: str,
                                update_trigger: str,
                                score_change_direction: str,
                                tenant_id: int,
                                industry_code: str = 'SaaS',
                                region: str = 'US',
                                **kwargs):
        """
        Record trust score updates (Task 20.2-T06)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'update_trigger': update_trigger,
                'score_change_direction': score_change_direction
            }
            
            self.trust_score_updates_total.labels(**labels).inc()
            logger.info(f"📊 Trust score update recorded: {update_trigger} - {score_change_direction}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record trust score update metrics: {e}")
    
    def record_risk_register_update(self,
                                  workflow_id: str,
                                  update_type: str,
                                  risk_level: str,
                                  tenant_id: int,
                                  industry_code: str = 'SaaS',
                                  region: str = 'US',
                                  **kwargs):
        """
        Record risk register updates (Task 20.2-T07)
        """
        try:
            labels = {
                'tenant_id': str(tenant_id),
                'industry_code': industry_code,
                'region': region,
                'workflow_id': workflow_id,
                'update_type': update_type,
                'risk_level': risk_level
            }
            
            self.risk_register_updates_total.labels(**labels).inc()
            logger.info(f"📊 Risk register update recorded: {update_type} - {risk_level}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record risk register update metrics: {e}")

# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """
    Get or create global metrics collector instance
    """
    global _global_collector
    
    if _global_collector is None:
        _global_collector = MetricsCollector()
    
    return _global_collector
