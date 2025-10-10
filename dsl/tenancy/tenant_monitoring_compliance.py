# Tenant Monitoring and Compliance System
# Tasks 6.5-T12, T13, T33, T37, T38: Monitoring, health, trust scoring, KPIs, SLO tracking

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

class TenantHealthStatus(Enum):
    """Tenant health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    OFFLINE = "offline"

class SLOMetricType(Enum):
    """Types of SLO metrics to track"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    COMPLIANCE_SCORE = "compliance_score"

@dataclass
class TenantHealthMetrics:
    """Health metrics for a tenant"""
    tenant_id: int
    overall_health: TenantHealthStatus
    uptime_percent: float
    avg_response_time_ms: float
    error_rate_percent: float
    active_workflows: int
    resource_utilization: Dict[str, float]
    compliance_score: float
    trust_score: float
    last_updated: str
    issues: List[Dict[str, Any]]

@dataclass
class SLOTarget:
    """SLO target definition"""
    metric_type: SLOMetricType
    target_value: float
    comparison: str  # >=, <=, ==
    measurement_window_hours: int
    alert_threshold_percent: float

@dataclass
class SLOStatus:
    """Current SLO status"""
    tenant_id: int
    metric_type: SLOMetricType
    target: SLOTarget
    current_value: float
    compliance_percent: float
    status: str  # compliant, at_risk, violated
    violation_duration_minutes: int
    last_violation: Optional[str]
    measurement_period: str

@dataclass
class TenantKPI:
    """Key Performance Indicator for tenant"""
    kpi_name: str
    current_value: float
    target_value: float
    trend: str  # improving, stable, declining
    measurement_unit: str
    category: str  # operational, business, compliance
    last_updated: str

class TenantTrustScoreCalculator:
    """
    Calculate comprehensive trust scores for tenants
    Task 6.5-T33: Build trust scoring per tenant
    """
    
    def __init__(self):
        self.trust_factors = {
            'compliance_adherence': 0.30,  # 30% weight
            'workflow_success_rate': 0.25,  # 25% weight
            'policy_violations': 0.20,  # 20% weight
            'override_frequency': 0.15,  # 15% weight
            'sla_compliance': 0.10  # 10% weight
        }
    
    def calculate_trust_score(
        self,
        tenant_metrics: Dict[str, Any],
        historical_data: Dict[str, List[float]] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive trust score"""
        
        # Extract metrics with defaults
        compliance_score = tenant_metrics.get('compliance_score', 1.0)
        success_rate = tenant_metrics.get('workflow_success_rate', 1.0)
        policy_violations = tenant_metrics.get('policy_violations_count', 0)
        override_count = tenant_metrics.get('override_count', 0)
        sla_compliance = tenant_metrics.get('sla_compliance_percent', 100.0) / 100.0
        
        total_workflows = tenant_metrics.get('total_workflows', 1)
        
        # Calculate individual factor scores (0.0 to 1.0)
        compliance_factor = min(compliance_score, 1.0)
        success_factor = min(success_rate, 1.0)
        
        # Policy violations penalty (more violations = lower score)
        violation_penalty = min(policy_violations / max(total_workflows, 1), 1.0)
        policy_factor = max(1.0 - violation_penalty, 0.0)
        
        # Override frequency penalty
        override_penalty = min(override_count / max(total_workflows, 1), 1.0)
        override_factor = max(1.0 - override_penalty, 0.0)
        
        sla_factor = min(sla_compliance, 1.0)
        
        # Calculate weighted trust score
        trust_score = (
            compliance_factor * self.trust_factors['compliance_adherence'] +
            success_factor * self.trust_factors['workflow_success_rate'] +
            policy_factor * self.trust_factors['policy_violations'] +
            override_factor * self.trust_factors['override_frequency'] +
            sla_factor * self.trust_factors['sla_compliance']
        )
        
        # Determine trust level
        if trust_score >= 0.9:
            trust_level = "excellent"
        elif trust_score >= 0.8:
            trust_level = "good"
        elif trust_score >= 0.7:
            trust_level = "fair"
        elif trust_score >= 0.6:
            trust_level = "poor"
        else:
            trust_level = "critical"
        
        # Calculate trend if historical data available
        trend = "stable"
        if historical_data and 'trust_scores' in historical_data:
            recent_scores = historical_data['trust_scores'][-5:]  # Last 5 measurements
            if len(recent_scores) >= 3:
                if recent_scores[-1] > recent_scores[0] + 0.05:
                    trend = "improving"
                elif recent_scores[-1] < recent_scores[0] - 0.05:
                    trend = "declining"
        
        return {
            'trust_score': round(trust_score, 3),
            'trust_level': trust_level,
            'trend': trend,
            'factor_scores': {
                'compliance_adherence': round(compliance_factor, 3),
                'workflow_success_rate': round(success_factor, 3),
                'policy_violations': round(policy_factor, 3),
                'override_frequency': round(override_factor, 3),
                'sla_compliance': round(sla_factor, 3)
            },
            'calculated_at': datetime.now(timezone.utc).isoformat()
        }

class TenantMonitoringCompliance:
    """
    Comprehensive tenant monitoring and compliance system
    Tasks 6.5-T12, T13, T33, T37, T38: Monitoring, health, trust scoring, KPIs, SLO tracking
    """
    
    def __init__(
        self,
        tenant_context_manager=None,
        tenant_isolation_sandbox=None,
        metrics_collector=None
    ):
        self.tenant_context_manager = tenant_context_manager
        self.tenant_isolation_sandbox = tenant_isolation_sandbox
        self.metrics_collector = metrics_collector
        self.trust_calculator = TenantTrustScoreCalculator()
        
        # Monitoring data storage
        self.tenant_health_metrics: Dict[int, TenantHealthMetrics] = {}
        self.tenant_slo_targets: Dict[int, List[SLOTarget]] = {}
        self.tenant_slo_status: Dict[int, List[SLOStatus]] = {}
        self.tenant_kpis: Dict[int, List[TenantKPI]] = {}
        self.tenant_historical_data: Dict[int, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        
        # Default SLO targets by tenant tier
        self.default_slo_targets = {
            'starter': [
                SLOTarget(SLOMetricType.AVAILABILITY, 99.0, ">=", 24, 95.0),
                SLOTarget(SLOMetricType.LATENCY, 5000.0, "<=", 1, 90.0),
                SLOTarget(SLOMetricType.ERROR_RATE, 5.0, "<=", 1, 90.0)
            ],
            'professional': [
                SLOTarget(SLOMetricType.AVAILABILITY, 99.5, ">=", 24, 95.0),
                SLOTarget(SLOMetricType.LATENCY, 2000.0, "<=", 1, 90.0),
                SLOTarget(SLOMetricType.ERROR_RATE, 2.0, "<=", 1, 90.0)
            ],
            'enterprise': [
                SLOTarget(SLOMetricType.AVAILABILITY, 99.9, ">=", 24, 95.0),
                SLOTarget(SLOMetricType.LATENCY, 1000.0, "<=", 1, 90.0),
                SLOTarget(SLOMetricType.ERROR_RATE, 1.0, "<=", 1, 90.0),
                SLOTarget(SLOMetricType.COMPLIANCE_SCORE, 95.0, ">=", 24, 90.0)
            ],
            'enterprise_plus': [
                SLOTarget(SLOMetricType.AVAILABILITY, 99.99, ">=", 24, 95.0),
                SLOTarget(SLOMetricType.LATENCY, 500.0, "<=", 1, 90.0),
                SLOTarget(SLOMetricType.ERROR_RATE, 0.5, "<=", 1, 90.0),
                SLOTarget(SLOMetricType.COMPLIANCE_SCORE, 98.0, ">=", 24, 90.0)
            ]
        }
    
    async def initialize_tenant_monitoring(self, tenant_id: int) -> bool:
        """Initialize monitoring for a tenant"""
        
        try:
            tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
            if not tenant_context:
                return False
            
            # Set up SLO targets based on tenant tier
            tier_key = tenant_context.tenant_tier.value
            slo_targets = self.default_slo_targets.get(tier_key, self.default_slo_targets['professional'])
            self.tenant_slo_targets[tenant_id] = slo_targets
            
            # Initialize SLO status tracking
            self.tenant_slo_status[tenant_id] = []
            for target in slo_targets:
                slo_status = SLOStatus(
                    tenant_id=tenant_id,
                    metric_type=target.metric_type,
                    target=target,
                    current_value=0.0,
                    compliance_percent=100.0,
                    status="compliant",
                    violation_duration_minutes=0,
                    last_violation=None,
                    measurement_period=f"last_{target.measurement_window_hours}h"
                )
                self.tenant_slo_status[tenant_id].append(slo_status)
            
            # Initialize KPIs
            await self._initialize_tenant_kpis(tenant_id, tenant_context)
            
            logger.info(f"Initialized monitoring for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring for tenant {tenant_id}: {e}")
            return False
    
    async def collect_tenant_health_metrics(self, tenant_id: int) -> TenantHealthMetrics:
        """Collect comprehensive health metrics for tenant"""
        
        try:
            # Get resource usage from sandbox
            resource_usage = {}
            if self.tenant_isolation_sandbox:
                usage_data = await self.tenant_isolation_sandbox.get_tenant_resource_usage(tenant_id)
                if 'resource_utilization' in usage_data:
                    resource_usage = usage_data['resource_utilization']
            
            # Get workflow metrics
            workflow_metrics = await self._get_tenant_workflow_metrics(tenant_id)
            
            # Calculate compliance score
            compliance_score = await self._calculate_tenant_compliance_score(tenant_id)
            
            # Calculate trust score
            trust_metrics = {
                'compliance_score': compliance_score,
                'workflow_success_rate': workflow_metrics.get('success_rate', 1.0),
                'policy_violations_count': workflow_metrics.get('policy_violations', 0),
                'override_count': workflow_metrics.get('overrides', 0),
                'total_workflows': workflow_metrics.get('total_workflows', 1),
                'sla_compliance_percent': await self._get_tenant_sla_compliance(tenant_id)
            }
            
            historical_data = dict(self.tenant_historical_data[tenant_id])
            trust_data = self.trust_calculator.calculate_trust_score(trust_metrics, historical_data)
            
            # Determine overall health status
            health_status = self._determine_health_status(
                resource_usage,
                workflow_metrics,
                compliance_score,
                trust_data['trust_score']
            )
            
            # Identify issues
            issues = await self._identify_tenant_issues(tenant_id, resource_usage, workflow_metrics)
            
            # Create health metrics
            health_metrics = TenantHealthMetrics(
                tenant_id=tenant_id,
                overall_health=health_status,
                uptime_percent=workflow_metrics.get('uptime_percent', 100.0),
                avg_response_time_ms=workflow_metrics.get('avg_response_time_ms', 0.0),
                error_rate_percent=workflow_metrics.get('error_rate_percent', 0.0),
                active_workflows=workflow_metrics.get('active_workflows', 0),
                resource_utilization=resource_usage,
                compliance_score=compliance_score,
                trust_score=trust_data['trust_score'],
                last_updated=datetime.now(timezone.utc).isoformat(),
                issues=issues
            )
            
            # Store metrics
            self.tenant_health_metrics[tenant_id] = health_metrics
            
            # Update historical data
            self._update_historical_data(tenant_id, health_metrics, trust_data)
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect health metrics for tenant {tenant_id}: {e}")
            return self._get_default_health_metrics(tenant_id)
    
    async def update_tenant_slo_status(self, tenant_id: int) -> List[SLOStatus]:
        """Update SLO status for tenant"""
        
        if tenant_id not in self.tenant_slo_status:
            await self.initialize_tenant_monitoring(tenant_id)
        
        slo_statuses = self.tenant_slo_status[tenant_id]
        current_time = datetime.now(timezone.utc)
        
        for slo_status in slo_statuses:
            try:
                # Get current metric value
                current_value = await self._get_current_metric_value(
                    tenant_id,
                    slo_status.metric_type,
                    slo_status.target.measurement_window_hours
                )
                
                slo_status.current_value = current_value
                
                # Check compliance
                is_compliant = self._check_slo_compliance(current_value, slo_status.target)
                
                if is_compliant:
                    if slo_status.status != "compliant":
                        slo_status.status = "compliant"
                        slo_status.violation_duration_minutes = 0
                else:
                    # Check if at risk or violated
                    compliance_percent = self._calculate_compliance_percent(current_value, slo_status.target)
                    slo_status.compliance_percent = compliance_percent
                    
                    if compliance_percent < slo_status.target.alert_threshold_percent:
                        if slo_status.status == "compliant":
                            slo_status.status = "violated"
                            slo_status.last_violation = current_time.isoformat()
                            slo_status.violation_duration_minutes = 0
                        else:
                            # Update violation duration
                            if slo_status.last_violation:
                                last_violation_time = datetime.fromisoformat(slo_status.last_violation.replace('Z', '+00:00'))
                                slo_status.violation_duration_minutes = int((current_time - last_violation_time).total_seconds() / 60)
                    else:
                        slo_status.status = "at_risk"
                
            except Exception as e:
                logger.error(f"Failed to update SLO status for tenant {tenant_id}, metric {slo_status.metric_type.value}: {e}")
        
        return slo_statuses
    
    async def get_tenant_dashboard_data(self, tenant_id: int) -> Dict[str, Any]:
        """Get comprehensive dashboard data for tenant"""
        
        # Collect latest health metrics
        health_metrics = await self.collect_tenant_health_metrics(tenant_id)
        
        # Update SLO status
        slo_statuses = await self.update_tenant_slo_status(tenant_id)
        
        # Get KPIs
        kpis = self.tenant_kpis.get(tenant_id, [])
        
        # Get recent trends
        trends = self._get_tenant_trends(tenant_id)
        
        return {
            'tenant_id': tenant_id,
            'health_metrics': asdict(health_metrics),
            'slo_status': [asdict(slo) for slo in slo_statuses],
            'kpis': [asdict(kpi) for kpi in kpis],
            'trends': trends,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    async def _get_tenant_workflow_metrics(self, tenant_id: int) -> Dict[str, Any]:
        """Get workflow metrics for tenant"""
        # Mock implementation - in practice would query actual metrics
        return {
            'total_workflows': 100,
            'active_workflows': 5,
            'success_rate': 0.95,
            'avg_response_time_ms': 1500.0,
            'error_rate_percent': 2.0,
            'uptime_percent': 99.5,
            'policy_violations': 2,
            'overrides': 1
        }
    
    async def _calculate_tenant_compliance_score(self, tenant_id: int) -> float:
        """Calculate compliance score for tenant"""
        # Mock implementation - in practice would calculate from actual compliance data
        return 0.92
    
    async def _get_tenant_sla_compliance(self, tenant_id: int) -> float:
        """Get SLA compliance percentage for tenant"""
        # Mock implementation
        return 98.5
    
    def _determine_health_status(
        self,
        resource_usage: Dict[str, float],
        workflow_metrics: Dict[str, Any],
        compliance_score: float,
        trust_score: float
    ) -> TenantHealthStatus:
        """Determine overall health status"""
        
        # Check critical thresholds
        if trust_score < 0.6 or compliance_score < 0.7:
            return TenantHealthStatus.CRITICAL
        
        # Check resource utilization
        max_utilization = max(resource_usage.values()) if resource_usage else 0
        if max_utilization > 90:
            return TenantHealthStatus.CRITICAL
        elif max_utilization > 80:
            return TenantHealthStatus.WARNING
        
        # Check workflow metrics
        error_rate = workflow_metrics.get('error_rate_percent', 0)
        if error_rate > 10:
            return TenantHealthStatus.CRITICAL
        elif error_rate > 5:
            return TenantHealthStatus.WARNING
        
        # Check uptime
        uptime = workflow_metrics.get('uptime_percent', 100)
        if uptime < 95:
            return TenantHealthStatus.DEGRADED
        elif uptime < 99:
            return TenantHealthStatus.WARNING
        
        return TenantHealthStatus.HEALTHY
    
    async def _identify_tenant_issues(
        self,
        tenant_id: int,
        resource_usage: Dict[str, float],
        workflow_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify issues for tenant"""
        
        issues = []
        
        # Resource utilization issues
        for resource, utilization in resource_usage.items():
            if utilization > 90:
                issues.append({
                    'type': 'resource_critical',
                    'resource': resource,
                    'utilization': utilization,
                    'severity': 'critical',
                    'message': f"{resource} utilization is critically high at {utilization:.1f}%"
                })
            elif utilization > 80:
                issues.append({
                    'type': 'resource_warning',
                    'resource': resource,
                    'utilization': utilization,
                    'severity': 'warning',
                    'message': f"{resource} utilization is high at {utilization:.1f}%"
                })
        
        # Workflow issues
        error_rate = workflow_metrics.get('error_rate_percent', 0)
        if error_rate > 5:
            issues.append({
                'type': 'high_error_rate',
                'error_rate': error_rate,
                'severity': 'critical' if error_rate > 10 else 'warning',
                'message': f"Error rate is high at {error_rate:.1f}%"
            })
        
        return issues
    
    async def _get_current_metric_value(
        self,
        tenant_id: int,
        metric_type: SLOMetricType,
        window_hours: int
    ) -> float:
        """Get current value for SLO metric"""
        
        # Mock implementation - in practice would query actual metrics
        mock_values = {
            SLOMetricType.AVAILABILITY: 99.5,
            SLOMetricType.LATENCY: 1200.0,
            SLOMetricType.ERROR_RATE: 1.5,
            SLOMetricType.THROUGHPUT: 150.0,
            SLOMetricType.COMPLIANCE_SCORE: 92.0
        }
        
        return mock_values.get(metric_type, 0.0)
    
    def _check_slo_compliance(self, current_value: float, target: SLOTarget) -> bool:
        """Check if current value meets SLO target"""
        
        if target.comparison == ">=":
            return current_value >= target.target_value
        elif target.comparison == "<=":
            return current_value <= target.target_value
        elif target.comparison == "==":
            return abs(current_value - target.target_value) < 0.01
        
        return False
    
    def _calculate_compliance_percent(self, current_value: float, target: SLOTarget) -> float:
        """Calculate compliance percentage"""
        
        if target.comparison == ">=":
            if current_value >= target.target_value:
                return 100.0
            else:
                return max((current_value / target.target_value) * 100, 0.0)
        elif target.comparison == "<=":
            if current_value <= target.target_value:
                return 100.0
            else:
                return max((target.target_value / current_value) * 100, 0.0)
        
        return 100.0 if abs(current_value - target.target_value) < 0.01 else 0.0
    
    def _update_historical_data(
        self,
        tenant_id: int,
        health_metrics: TenantHealthMetrics,
        trust_data: Dict[str, Any]
    ) -> None:
        """Update historical data for tenant"""
        
        historical = self.tenant_historical_data[tenant_id]
        timestamp = time.time()
        
        # Store key metrics
        historical['timestamps'].append(timestamp)
        historical['trust_scores'].append(trust_data['trust_score'])
        historical['compliance_scores'].append(health_metrics.compliance_score)
        historical['error_rates'].append(health_metrics.error_rate_percent)
        historical['response_times'].append(health_metrics.avg_response_time_ms)
        historical['uptime_percentages'].append(health_metrics.uptime_percent)
    
    def _get_tenant_trends(self, tenant_id: int) -> Dict[str, Any]:
        """Get trend analysis for tenant"""
        
        historical = self.tenant_historical_data[tenant_id]
        trends = {}
        
        for metric_name, values in historical.items():
            if metric_name == 'timestamps' or len(values) < 5:
                continue
            
            recent_values = list(values)[-10:]  # Last 10 measurements
            if len(recent_values) >= 3:
                # Simple trend calculation
                first_half = recent_values[:len(recent_values)//2]
                second_half = recent_values[len(recent_values)//2:]
                
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                
                if second_avg > first_avg * 1.05:
                    trend = "improving" if metric_name in ['trust_scores', 'compliance_scores', 'uptime_percentages'] else "declining"
                elif second_avg < first_avg * 0.95:
                    trend = "declining" if metric_name in ['trust_scores', 'compliance_scores', 'uptime_percentages'] else "improving"
                else:
                    trend = "stable"
                
                trends[metric_name] = {
                    'trend': trend,
                    'current_value': recent_values[-1],
                    'change_percent': ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
                }
        
        return trends
    
    async def _initialize_tenant_kpis(self, tenant_id: int, tenant_context) -> None:
        """Initialize KPIs for tenant"""
        
        kpis = [
            TenantKPI(
                kpi_name="Workflow Success Rate",
                current_value=95.0,
                target_value=98.0,
                trend="stable",
                measurement_unit="percent",
                category="operational",
                last_updated=datetime.now(timezone.utc).isoformat()
            ),
            TenantKPI(
                kpi_name="Average Response Time",
                current_value=1200.0,
                target_value=1000.0,
                trend="improving",
                measurement_unit="milliseconds",
                category="operational",
                last_updated=datetime.now(timezone.utc).isoformat()
            ),
            TenantKPI(
                kpi_name="Compliance Score",
                current_value=92.0,
                target_value=95.0,
                trend="improving",
                measurement_unit="percent",
                category="compliance",
                last_updated=datetime.now(timezone.utc).isoformat()
            )
        ]
        
        self.tenant_kpis[tenant_id] = kpis
    
    def _get_default_health_metrics(self, tenant_id: int) -> TenantHealthMetrics:
        """Get default health metrics when collection fails"""
        return TenantHealthMetrics(
            tenant_id=tenant_id,
            overall_health=TenantHealthStatus.OFFLINE,
            uptime_percent=0.0,
            avg_response_time_ms=0.0,
            error_rate_percent=100.0,
            active_workflows=0,
            resource_utilization={},
            compliance_score=0.0,
            trust_score=0.0,
            last_updated=datetime.now(timezone.utc).isoformat(),
            issues=[{
                'type': 'monitoring_failure',
                'severity': 'critical',
                'message': 'Failed to collect health metrics'
            }]
        )

# Global tenant monitoring and compliance system
tenant_monitoring_compliance = TenantMonitoringCompliance()
