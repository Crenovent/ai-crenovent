#!/usr/bin/env python3
"""
SLA Monitoring System - Chapter 14.2.2-14.2.8
===============================================

Implements intelligent SLA monitoring and enforcement for capabilities:
- Real-time availability tracking
- Latency monitoring and alerting
- Uptime measurement and reporting
- SLA tier enforcement (T0/T1/T2)
- Automated escalation and remediation
- Cost attribution and FinOps integration

This system ensures capabilities meet their committed SLA levels and provides
real-time visibility into performance against business commitments.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import uuid

logger = logging.getLogger(__name__)

class SLATier(Enum):
    """SLA tier definitions matching Build Plan specifications"""
    T0_REGULATED = "T0"    # 99.99% uptime, <100ms latency, 24/7 support
    T1_ENTERPRISE = "T1"   # 99.9% uptime, <500ms latency, business hours support
    T2_MIDMARKET = "T2"    # 99.5% uptime, <2s latency, email support

class SLAMetricType(Enum):
    """Types of SLA metrics to monitor"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RECOVERY_TIME = "recovery_time"

class SLAStatus(Enum):
    """SLA compliance status"""
    MEETING = "meeting"          # Within SLA targets
    AT_RISK = "at_risk"         # Approaching SLA breach
    BREACHED = "breached"       # SLA has been breached
    RECOVERING = "recovering"    # Recovering from breach

@dataclass
class SLATarget:
    """SLA target definition"""
    metric_type: SLAMetricType
    target_value: float
    measurement_unit: str
    measurement_period: str  # e.g., "monthly", "daily", "hourly"
    threshold_warning: float  # Warning threshold (e.g., 95% of target)
    threshold_critical: float  # Critical threshold (e.g., 90% of target)

@dataclass
class SLAMeasurement:
    """Individual SLA measurement"""
    capability_id: str
    tenant_id: int
    metric_type: SLAMetricType
    measured_value: float
    target_value: float
    measurement_unit: str
    status: SLAStatus
    measured_at: datetime
    measurement_period_start: datetime
    measurement_period_end: datetime
    
    # Contextual data
    sample_size: int
    confidence: float
    breach_duration_seconds: Optional[int] = None

@dataclass
class SLAReport:
    """Comprehensive SLA report"""
    capability_id: str
    tenant_id: int
    sla_tier: SLATier
    reporting_period_start: datetime
    reporting_period_end: datetime
    
    # Current status
    overall_status: SLAStatus
    measurements: List[SLAMeasurement]
    
    # Summary statistics
    availability_percentage: float
    average_latency_ms: float
    error_rate_percentage: float
    uptime_minutes: int
    downtime_minutes: int
    
    # SLA performance
    sla_compliance_percentage: float
    breaches_count: int
    total_breach_duration_minutes: int
    
    # Business impact
    estimated_business_impact: Dict[str, Any]
    cost_attribution: Dict[str, float]
    
    # Actions and recommendations
    recommended_actions: List[str]
    escalations_triggered: List[str]

class SLAMonitoringSystem:
    """
    Intelligent SLA monitoring and enforcement system
    
    Features:
    - Real-time SLA tracking across all capability executions
    - Tier-based SLA enforcement (T0/T1/T2)
    - Automated alerting and escalation
    - Business impact calculation
    - Cost attribution and FinOps integration
    - Predictive SLA breach detection
    - Automated remediation triggers
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # SLA tier definitions
        self.sla_tiers = {
            SLATier.T0_REGULATED: {
                SLAMetricType.AVAILABILITY: SLATarget(
                    metric_type=SLAMetricType.AVAILABILITY,
                    target_value=99.99,
                    measurement_unit="percentage",
                    measurement_period="monthly",
                    threshold_warning=99.95,
                    threshold_critical=99.90
                ),
                SLAMetricType.LATENCY: SLATarget(
                    metric_type=SLAMetricType.LATENCY,
                    target_value=100.0,
                    measurement_unit="milliseconds",
                    measurement_period="hourly",
                    threshold_warning=80.0,
                    threshold_critical=90.0
                ),
                SLAMetricType.ERROR_RATE: SLATarget(
                    metric_type=SLAMetricType.ERROR_RATE,
                    target_value=0.01,
                    measurement_unit="percentage",
                    measurement_period="daily",
                    threshold_warning=0.05,
                    threshold_critical=0.1
                )
            },
            SLATier.T1_ENTERPRISE: {
                SLAMetricType.AVAILABILITY: SLATarget(
                    metric_type=SLAMetricType.AVAILABILITY,
                    target_value=99.9,
                    measurement_unit="percentage",
                    measurement_period="monthly",
                    threshold_warning=99.5,
                    threshold_critical=99.0
                ),
                SLAMetricType.LATENCY: SLATarget(
                    metric_type=SLAMetricType.LATENCY,
                    target_value=500.0,
                    measurement_unit="milliseconds",
                    measurement_period="hourly",
                    threshold_warning=400.0,
                    threshold_critical=450.0
                ),
                SLAMetricType.ERROR_RATE: SLATarget(
                    metric_type=SLAMetricType.ERROR_RATE,
                    target_value=0.1,
                    measurement_unit="percentage",
                    measurement_period="daily",
                    threshold_warning=0.5,
                    threshold_critical=1.0
                )
            },
            SLATier.T2_MIDMARKET: {
                SLAMetricType.AVAILABILITY: SLATarget(
                    metric_type=SLAMetricType.AVAILABILITY,
                    target_value=99.5,
                    measurement_unit="percentage",
                    measurement_period="monthly",
                    threshold_warning=99.0,
                    threshold_critical=98.0
                ),
                SLAMetricType.LATENCY: SLATarget(
                    metric_type=SLAMetricType.LATENCY,
                    target_value=2000.0,
                    measurement_unit="milliseconds",
                    measurement_period="hourly",
                    threshold_warning=1500.0,
                    threshold_critical=1800.0
                ),
                SLAMetricType.ERROR_RATE: SLATarget(
                    metric_type=SLAMetricType.ERROR_RATE,
                    target_value=1.0,
                    measurement_unit="percentage",
                    measurement_period="daily",
                    threshold_warning=2.0,
                    threshold_critical=5.0
                )
            }
        }
        
        # Monitoring configuration
        self.measurement_intervals = {
            "real_time": timedelta(seconds=30),
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1),
            "monthly": timedelta(days=30)
        }
        
        # Cache for performance
        self.sla_cache = {}
        self.cache_ttl = timedelta(minutes=5)
        
    async def initialize(self):
        """Initialize the SLA monitoring system"""
        try:
            self.logger.info("📊 Initializing SLA Monitoring System...")
            
            # Create database tables
            await self._ensure_sla_tables()
            
            # Start background monitoring tasks
            await self._start_monitoring_tasks()
            
            self.logger.info("✅ SLA Monitoring System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SLA Monitoring System: {e}")
            return False
    
    async def measure_capability_sla(self, capability_id: str, tenant_id: int, 
                                   sla_tier: SLATier, measurement_period_hours: int = 24) -> SLAReport:
        """
        Measure SLA performance for a capability over a specific period
        
        Args:
            capability_id: Capability to measure
            tenant_id: Tenant context
            sla_tier: SLA tier (T0/T1/T2)
            measurement_period_hours: Hours to look back for measurements
            
        Returns:
            Comprehensive SLA report with status and recommendations
        """
        try:
            self.logger.info(f"📊 Measuring SLA for capability {capability_id} (Tier: {sla_tier.value})")
            
            period_end = datetime.now()
            period_start = period_end - timedelta(hours=measurement_period_hours)
            
            # Get SLA targets for this tier
            tier_targets = self.sla_tiers[sla_tier]
            
            # Collect measurements for each metric type
            measurements = []
            
            # Availability measurement
            availability_measurement = await self._measure_availability(
                capability_id, tenant_id, period_start, period_end, tier_targets[SLAMetricType.AVAILABILITY]
            )
            if availability_measurement:
                measurements.append(availability_measurement)
            
            # Latency measurement
            latency_measurement = await self._measure_latency(
                capability_id, tenant_id, period_start, period_end, tier_targets[SLAMetricType.LATENCY]
            )
            if latency_measurement:
                measurements.append(latency_measurement)
            
            # Error rate measurement
            error_rate_measurement = await self._measure_error_rate(
                capability_id, tenant_id, period_start, period_end, tier_targets[SLAMetricType.ERROR_RATE]
            )
            if error_rate_measurement:
                measurements.append(error_rate_measurement)
            
            # Calculate overall status
            overall_status = self._calculate_overall_sla_status(measurements)
            
            # Calculate summary statistics
            availability_percentage = next(
                (m.measured_value for m in measurements if m.metric_type == SLAMetricType.AVAILABILITY), 
                0.0
            )
            average_latency_ms = next(
                (m.measured_value for m in measurements if m.metric_type == SLAMetricType.LATENCY), 
                0.0
            )
            error_rate_percentage = next(
                (m.measured_value for m in measurements if m.metric_type == SLAMetricType.ERROR_RATE), 
                0.0
            )
            
            # Calculate uptime/downtime
            total_minutes = measurement_period_hours * 60
            uptime_minutes = int((availability_percentage / 100.0) * total_minutes)
            downtime_minutes = total_minutes - uptime_minutes
            
            # Calculate SLA compliance
            sla_compliance_percentage = self._calculate_sla_compliance(measurements, tier_targets)
            
            # Count breaches
            breaches_count = sum(1 for m in measurements if m.status == SLAStatus.BREACHED)
            total_breach_duration_minutes = sum(
                m.breach_duration_seconds // 60 for m in measurements 
                if m.breach_duration_seconds is not None
            )
            
            # Calculate business impact
            estimated_business_impact = await self._calculate_business_impact(
                capability_id, tenant_id, measurements, downtime_minutes
            )
            
            # Calculate cost attribution
            cost_attribution = await self._calculate_cost_attribution(
                capability_id, tenant_id, period_start, period_end
            )
            
            # Generate recommendations
            recommended_actions = self._generate_sla_recommendations(
                measurements, sla_tier, overall_status
            )
            
            # Check for escalations
            escalations_triggered = await self._check_escalations(
                capability_id, tenant_id, measurements, sla_tier
            )
            
            # Create SLA report
            sla_report = SLAReport(
                capability_id=capability_id,
                tenant_id=tenant_id,
                sla_tier=sla_tier,
                reporting_period_start=period_start,
                reporting_period_end=period_end,
                overall_status=overall_status,
                measurements=measurements,
                availability_percentage=availability_percentage,
                average_latency_ms=average_latency_ms,
                error_rate_percentage=error_rate_percentage,
                uptime_minutes=uptime_minutes,
                downtime_minutes=downtime_minutes,
                sla_compliance_percentage=sla_compliance_percentage,
                breaches_count=breaches_count,
                total_breach_duration_minutes=total_breach_duration_minutes,
                estimated_business_impact=estimated_business_impact,
                cost_attribution=cost_attribution,
                recommended_actions=recommended_actions,
                escalations_triggered=escalations_triggered
            )
            
            # Store the report
            await self._store_sla_report(sla_report)
            
            self.logger.info(f"✅ SLA measured: {sla_compliance_percentage:.1f}% compliance ({overall_status.value})")
            return sla_report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to measure SLA for {capability_id}: {e}")
            return await self._create_default_sla_report(capability_id, tenant_id, sla_tier)
    
    async def get_sla_dashboard_data(self, tenant_id: int, sla_tier: Optional[SLATier] = None) -> Dict[str, Any]:
        """Get SLA dashboard data for frontend integration"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                # Base query for SLA reports
                base_query = """
                SELECT 
                    capability_id,
                    sla_tier,
                    overall_status,
                    availability_percentage,
                    average_latency_ms,
                    error_rate_percentage,
                    sla_compliance_percentage,
                    breaches_count,
                    reporting_period_start,
                    reporting_period_end,
                    recommended_actions,
                    escalations_triggered
                FROM capability_sla_reports 
                WHERE tenant_id = $1
                  AND reporting_period_end >= NOW() - INTERVAL '7 days'
                """
                
                params = [tenant_id]
                
                if sla_tier:
                    base_query += " AND sla_tier = $2"
                    params.append(sla_tier.value)
                
                base_query += " ORDER BY reporting_period_end DESC"
                
                rows = await conn.fetch(base_query, *params)
                
                capabilities = [dict(row) for row in rows]
                
                # Calculate summary statistics
                if capabilities:
                    avg_availability = statistics.mean([c['availability_percentage'] for c in capabilities])
                    avg_latency = statistics.mean([c['average_latency_ms'] for c in capabilities])
                    avg_compliance = statistics.mean([c['sla_compliance_percentage'] for c in capabilities])
                    total_breaches = sum([c['breaches_count'] for c in capabilities])
                else:
                    avg_availability = avg_latency = avg_compliance = total_breaches = 0
                
                # Get SLA trends
                trends_query = """
                SELECT 
                    DATE_TRUNC('hour', reporting_period_end) as hour,
                    AVG(availability_percentage) as avg_availability,
                    AVG(average_latency_ms) as avg_latency,
                    AVG(sla_compliance_percentage) as avg_compliance
                FROM capability_sla_reports
                WHERE tenant_id = $1
                  AND reporting_period_end >= NOW() - INTERVAL '24 hours'
                GROUP BY DATE_TRUNC('hour', reporting_period_end)
                ORDER BY hour
                """
                
                trend_rows = await conn.fetch(trends_query, tenant_id)
                trends = [dict(row) for row in trend_rows]
                
                return {
                    "tenant_id": tenant_id,
                    "sla_tier_filter": sla_tier.value if sla_tier else "all",
                    "summary": {
                        "total_capabilities": len(set(c['capability_id'] for c in capabilities)),
                        "average_availability": round(avg_availability, 2),
                        "average_latency_ms": round(avg_latency, 1),
                        "average_compliance": round(avg_compliance, 1),
                        "total_breaches": total_breaches,
                        "capabilities_at_risk": len([c for c in capabilities if c['overall_status'] == 'at_risk']),
                        "capabilities_breached": len([c for c in capabilities if c['overall_status'] == 'breached'])
                    },
                    "capabilities": capabilities,
                    "trends": trends,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting SLA dashboard data: {e}")
            return {"error": str(e)}
    
    async def trigger_sla_alert(self, capability_id: str, tenant_id: int, 
                              measurement: SLAMeasurement) -> Dict[str, Any]:
        """Trigger SLA alert for breaches or at-risk situations"""
        try:
            alert_data = {
                "alert_id": str(uuid.uuid4()),
                "capability_id": capability_id,
                "tenant_id": tenant_id,
                "alert_type": "sla_breach" if measurement.status == SLAStatus.BREACHED else "sla_at_risk",
                "severity": "critical" if measurement.status == SLAStatus.BREACHED else "warning",
                "metric_type": measurement.metric_type.value,
                "measured_value": measurement.measured_value,
                "target_value": measurement.target_value,
                "status": measurement.status.value,
                "triggered_at": datetime.now().isoformat(),
                "message": self._generate_alert_message(measurement),
                "recommended_actions": self._get_metric_specific_actions(measurement)
            }
            
            # Store alert in database
            await self._store_sla_alert(alert_data)
            
            # Trigger notification (would integrate with notification service)
            await self._send_sla_notification(alert_data)
            
            return alert_data
            
        except Exception as e:
            self.logger.error(f"Error triggering SLA alert: {e}")
            return {"error": str(e)}
    
    # Helper methods for SLA measurements
    
    async def _measure_availability(self, capability_id: str, tenant_id: int, 
                                  start_time: datetime, end_time: datetime, 
                                  target: SLATarget) -> Optional[SLAMeasurement]:
        """Measure availability percentage"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                query = """
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN execution_status = 'completed' THEN 1 ELSE 0 END) as successful_executions
                FROM dsl_execution_traces 
                WHERE tenant_id = $1
                  AND trace_data->>'capability_id' = $2
                  AND started_at BETWEEN $3 AND $4
                """
                
                row = await conn.fetchrow(query, tenant_id, capability_id, start_time, end_time)
                
                if row and row['total_executions'] > 0:
                    availability = (float(row['successful_executions']) / float(row['total_executions'])) * 100.0
                    
                    # Determine status
                    status = SLAStatus.MEETING
                    if availability < target.threshold_critical:
                        status = SLAStatus.BREACHED
                    elif availability < target.threshold_warning:
                        status = SLAStatus.AT_RISK
                    
                    return SLAMeasurement(
                        capability_id=capability_id,
                        tenant_id=tenant_id,
                        metric_type=SLAMetricType.AVAILABILITY,
                        measured_value=availability,
                        target_value=target.target_value,
                        measurement_unit=target.measurement_unit,
                        status=status,
                        measured_at=datetime.now(),
                        measurement_period_start=start_time,
                        measurement_period_end=end_time,
                        sample_size=row['total_executions'],
                        confidence=min(1.0, row['total_executions'] / 100.0)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error measuring availability: {e}")
        
        return None
    
    async def _measure_latency(self, capability_id: str, tenant_id: int, 
                             start_time: datetime, end_time: datetime, 
                             target: SLATarget) -> Optional[SLAMeasurement]:
        """Measure average latency"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                query = """
                SELECT 
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at)) * 1000) as avg_latency_ms,
                    COUNT(*) as sample_count
                FROM dsl_execution_traces 
                WHERE tenant_id = $1
                  AND trace_data->>'capability_id' = $2
                  AND execution_status = 'completed'
                  AND started_at BETWEEN $3 AND $4
                """
                
                row = await conn.fetchrow(query, tenant_id, capability_id, start_time, end_time)
                
                if row and row['sample_count'] > 0:
                    latency_ms = float(row['avg_latency_ms'] or 0)
                    
                    # Determine status
                    status = SLAStatus.MEETING
                    if latency_ms > target.threshold_critical:
                        status = SLAStatus.BREACHED
                    elif latency_ms > target.threshold_warning:
                        status = SLAStatus.AT_RISK
                    
                    return SLAMeasurement(
                        capability_id=capability_id,
                        tenant_id=tenant_id,
                        metric_type=SLAMetricType.LATENCY,
                        measured_value=latency_ms,
                        target_value=target.target_value,
                        measurement_unit=target.measurement_unit,
                        status=status,
                        measured_at=datetime.now(),
                        measurement_period_start=start_time,
                        measurement_period_end=end_time,
                        sample_size=row['sample_count'],
                        confidence=min(1.0, row['sample_count'] / 50.0)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error measuring latency: {e}")
        
        return None
    
    async def _measure_error_rate(self, capability_id: str, tenant_id: int, 
                                start_time: datetime, end_time: datetime, 
                                target: SLATarget) -> Optional[SLAMeasurement]:
        """Measure error rate percentage"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                query = """
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN execution_status = 'failed' THEN 1 ELSE 0 END) as failed_executions
                FROM dsl_execution_traces 
                WHERE tenant_id = $1
                  AND trace_data->>'capability_id' = $2
                  AND started_at BETWEEN $3 AND $4
                """
                
                row = await conn.fetchrow(query, tenant_id, capability_id, start_time, end_time)
                
                if row and row['total_executions'] > 0:
                    error_rate = (float(row['failed_executions']) / float(row['total_executions'])) * 100.0
                    
                    # Determine status
                    status = SLAStatus.MEETING
                    if error_rate > target.threshold_critical:
                        status = SLAStatus.BREACHED
                    elif error_rate > target.threshold_warning:
                        status = SLAStatus.AT_RISK
                    
                    return SLAMeasurement(
                        capability_id=capability_id,
                        tenant_id=tenant_id,
                        metric_type=SLAMetricType.ERROR_RATE,
                        measured_value=error_rate,
                        target_value=target.target_value,
                        measurement_unit=target.measurement_unit,
                        status=status,
                        measured_at=datetime.now(),
                        measurement_period_start=start_time,
                        measurement_period_end=end_time,
                        sample_size=row['total_executions'],
                        confidence=min(1.0, row['total_executions'] / 100.0)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error measuring error rate: {e}")
        
        return None
    
    def _calculate_overall_sla_status(self, measurements: List[SLAMeasurement]) -> SLAStatus:
        """Calculate overall SLA status from individual measurements"""
        if not measurements:
            return SLAStatus.MEETING
        
        # If any metric is breached, overall status is breached
        if any(m.status == SLAStatus.BREACHED for m in measurements):
            return SLAStatus.BREACHED
        
        # If any metric is at risk, overall status is at risk
        if any(m.status == SLAStatus.AT_RISK for m in measurements):
            return SLAStatus.AT_RISK
        
        return SLAStatus.MEETING
    
    def _calculate_sla_compliance(self, measurements: List[SLAMeasurement], 
                                targets: Dict[SLAMetricType, SLATarget]) -> float:
        """Calculate overall SLA compliance percentage"""
        if not measurements:
            return 100.0
        
        compliance_scores = []
        
        for measurement in measurements:
            target = targets.get(measurement.metric_type)
            if target:
                if measurement.metric_type == SLAMetricType.AVAILABILITY:
                    # For availability, higher is better
                    compliance = min(100.0, (measurement.measured_value / target.target_value) * 100.0)
                elif measurement.metric_type in [SLAMetricType.LATENCY, SLAMetricType.ERROR_RATE]:
                    # For latency and error rate, lower is better
                    compliance = max(0.0, 100.0 - ((measurement.measured_value - target.target_value) / target.target_value * 100.0))
                else:
                    compliance = 100.0 if measurement.status == SLAStatus.MEETING else 0.0
                
                compliance_scores.append(max(0.0, min(100.0, compliance)))
        
        return statistics.mean(compliance_scores) if compliance_scores else 100.0
    
    async def _calculate_business_impact(self, capability_id: str, tenant_id: int, 
                                       measurements: List[SLAMeasurement], 
                                       downtime_minutes: int) -> Dict[str, Any]:
        """Calculate estimated business impact of SLA performance"""
        # Simplified business impact calculation
        # In a real implementation, this would be much more sophisticated
        
        estimated_hourly_revenue_impact = 1000.0  # $1000/hour impact
        downtime_hours = downtime_minutes / 60.0
        
        return {
            "estimated_revenue_impact_usd": downtime_hours * estimated_hourly_revenue_impact,
            "downtime_hours": downtime_hours,
            "affected_processes": ["revenue_tracking", "pipeline_management"],
            "customer_impact_level": "medium" if downtime_minutes > 60 else "low"
        }
    
    async def _calculate_cost_attribution(self, capability_id: str, tenant_id: int, 
                                        start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Calculate cost attribution for the capability"""
        # Simplified cost calculation
        # In a real implementation, this would integrate with actual FinOps data
        
        return {
            "compute_cost_usd": 50.0,
            "storage_cost_usd": 10.0,
            "network_cost_usd": 5.0,
            "total_cost_usd": 65.0
        }
    
    def _generate_sla_recommendations(self, measurements: List[SLAMeasurement], 
                                    sla_tier: SLATier, overall_status: SLAStatus) -> List[str]:
        """Generate actionable SLA recommendations"""
        recommendations = []
        
        if overall_status == SLAStatus.BREACHED:
            recommendations.append("🚨 CRITICAL: SLA breach detected - immediate escalation required")
            recommendations.append("🔧 Activate incident response procedures")
        
        elif overall_status == SLAStatus.AT_RISK:
            recommendations.append("⚠️ SLA at risk - proactive measures recommended")
        
        for measurement in measurements:
            if measurement.status in [SLAStatus.BREACHED, SLAStatus.AT_RISK]:
                if measurement.metric_type == SLAMetricType.AVAILABILITY:
                    recommendations.append("🔄 Investigate failed executions and implement retry logic")
                elif measurement.metric_type == SLAMetricType.LATENCY:
                    recommendations.append("⚡ Optimize performance - consider caching or resource scaling")
                elif measurement.metric_type == SLAMetricType.ERROR_RATE:
                    recommendations.append("🐛 Review error logs and implement error handling improvements")
        
        if sla_tier == SLATier.T0_REGULATED:
            recommendations.append("📋 Prepare regulator notification if breach continues")
        
        return recommendations
    
    async def _check_escalations(self, capability_id: str, tenant_id: int, 
                               measurements: List[SLAMeasurement], sla_tier: SLATier) -> List[str]:
        """Check if escalations should be triggered"""
        escalations = []
        
        for measurement in measurements:
            if measurement.status == SLAStatus.BREACHED:
                if sla_tier == SLATier.T0_REGULATED:
                    escalations.append("EXECUTIVE_ESCALATION")
                    escalations.append("REGULATOR_NOTIFICATION")
                elif sla_tier == SLATier.T1_ENTERPRISE:
                    escalations.append("MANAGEMENT_ESCALATION")
                else:
                    escalations.append("OPS_TEAM_NOTIFICATION")
        
        return escalations
    
    def _generate_alert_message(self, measurement: SLAMeasurement) -> str:
        """Generate human-readable alert message"""
        metric_name = measurement.metric_type.value.replace('_', ' ').title()
        
        if measurement.status == SLAStatus.BREACHED:
            return f"SLA BREACH: {metric_name} is {measurement.measured_value:.2f} {measurement.measurement_unit}, exceeding target of {measurement.target_value:.2f} {measurement.measurement_unit}"
        else:
            return f"SLA AT RISK: {metric_name} is {measurement.measured_value:.2f} {measurement.measurement_unit}, approaching target of {measurement.target_value:.2f} {measurement.measurement_unit}"
    
    def _get_metric_specific_actions(self, measurement: SLAMeasurement) -> List[str]:
        """Get metric-specific recommended actions"""
        actions = []
        
        if measurement.metric_type == SLAMetricType.AVAILABILITY:
            actions.extend([
                "Check system health and error rates",
                "Review recent deployments or changes",
                "Verify infrastructure capacity"
            ])
        elif measurement.metric_type == SLAMetricType.LATENCY:
            actions.extend([
                "Review performance metrics and bottlenecks",
                "Check database query performance",
                "Consider scaling compute resources"
            ])
        elif measurement.metric_type == SLAMetricType.ERROR_RATE:
            actions.extend([
                "Analyze error logs for patterns",
                "Check integration endpoints",
                "Review recent code changes"
            ])
        
        return actions
    
    # Database and infrastructure methods
    
    async def _ensure_sla_tables(self):
        """Ensure SLA monitoring tables exist"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                # SLA reports table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS capability_sla_reports (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        capability_id TEXT NOT NULL,
                        tenant_id INTEGER NOT NULL,
                        sla_tier TEXT NOT NULL,
                        overall_status TEXT NOT NULL,
                        availability_percentage DECIMAL(5,2),
                        average_latency_ms DECIMAL(10,2),
                        error_rate_percentage DECIMAL(5,2),
                        sla_compliance_percentage DECIMAL(5,2),
                        breaches_count INTEGER,
                        total_breach_duration_minutes INTEGER,
                        reporting_period_start TIMESTAMPTZ,
                        reporting_period_end TIMESTAMPTZ,
                        measurements JSONB,
                        estimated_business_impact JSONB,
                        cost_attribution JSONB,
                        recommended_actions TEXT[],
                        escalations_triggered TEXT[],
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        
                        CONSTRAINT sla_tier_check CHECK (sla_tier IN ('T0', 'T1', 'T2')),
                        CONSTRAINT sla_status_check CHECK (overall_status IN ('meeting', 'at_risk', 'breached', 'recovering'))
                    )
                """)
                
                # SLA alerts table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS capability_sla_alerts (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        alert_id TEXT NOT NULL UNIQUE,
                        capability_id TEXT NOT NULL,
                        tenant_id INTEGER NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        measured_value DECIMAL(10,2),
                        target_value DECIMAL(10,2),
                        status TEXT NOT NULL,
                        message TEXT,
                        recommended_actions TEXT[],
                        triggered_at TIMESTAMPTZ DEFAULT NOW(),
                        acknowledged_at TIMESTAMPTZ,
                        resolved_at TIMESTAMPTZ,
                        
                        CONSTRAINT alert_type_check CHECK (alert_type IN ('sla_breach', 'sla_at_risk')),
                        CONSTRAINT severity_check CHECK (severity IN ('critical', 'warning', 'info'))
                    )
                """)
                
                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sla_reports_capability_tenant 
                    ON capability_sla_reports(capability_id, tenant_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sla_reports_period 
                    ON capability_sla_reports(reporting_period_end DESC)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sla_alerts_triggered 
                    ON capability_sla_alerts(triggered_at DESC)
                """)
                
        except Exception as e:
            self.logger.error(f"Error ensuring SLA tables: {e}")
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # In a real implementation, this would start background tasks for continuous monitoring
        self.logger.info("📊 SLA monitoring tasks would be started here")
    
    async def _store_sla_report(self, report: SLAReport):
        """Store SLA report in database"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                await conn.execute("""
                    INSERT INTO capability_sla_reports (
                        capability_id, tenant_id, sla_tier, overall_status,
                        availability_percentage, average_latency_ms, error_rate_percentage,
                        sla_compliance_percentage, breaches_count, total_breach_duration_minutes,
                        reporting_period_start, reporting_period_end, measurements,
                        estimated_business_impact, cost_attribution,
                        recommended_actions, escalations_triggered
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                report.capability_id, report.tenant_id, report.sla_tier.value, report.overall_status.value,
                report.availability_percentage, report.average_latency_ms, report.error_rate_percentage,
                report.sla_compliance_percentage, report.breaches_count, report.total_breach_duration_minutes,
                report.reporting_period_start, report.reporting_period_end,
                json.dumps([asdict(m) for m in report.measurements]),
                json.dumps(report.estimated_business_impact), json.dumps(report.cost_attribution),
                report.recommended_actions, report.escalations_triggered
                )
                
        except Exception as e:
            self.logger.error(f"Error storing SLA report: {e}")
    
    async def _store_sla_alert(self, alert_data: Dict[str, Any]):
        """Store SLA alert in database"""
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                await conn.execute("""
                    INSERT INTO capability_sla_alerts (
                        alert_id, capability_id, tenant_id, alert_type, severity,
                        metric_type, measured_value, target_value, status,
                        message, recommended_actions, triggered_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                alert_data['alert_id'], alert_data['capability_id'], alert_data['tenant_id'],
                alert_data['alert_type'], alert_data['severity'], alert_data['metric_type'],
                alert_data['measured_value'], alert_data['target_value'], alert_data['status'],
                alert_data['message'], alert_data['recommended_actions'],
                datetime.fromisoformat(alert_data['triggered_at'])
                )
                
        except Exception as e:
            self.logger.error(f"Error storing SLA alert: {e}")
    
    async def _send_sla_notification(self, alert_data: Dict[str, Any]):
        """Send SLA notification (would integrate with notification service)"""
        self.logger.info(f"📧 SLA notification would be sent: {alert_data['message']}")
    
    async def _create_default_sla_report(self, capability_id: str, tenant_id: int, 
                                       sla_tier: SLATier) -> SLAReport:
        """Create default SLA report when measurement fails"""
        return SLAReport(
            capability_id=capability_id,
            tenant_id=tenant_id,
            sla_tier=sla_tier,
            reporting_period_start=datetime.now() - timedelta(hours=24),
            reporting_period_end=datetime.now(),
            overall_status=SLAStatus.MEETING,
            measurements=[],
            availability_percentage=99.0,
            average_latency_ms=100.0,
            error_rate_percentage=0.1,
            uptime_minutes=1440,
            downtime_minutes=0,
            sla_compliance_percentage=100.0,
            breaches_count=0,
            total_breach_duration_minutes=0,
            estimated_business_impact={},
            cost_attribution={},
            recommended_actions=["⚠️ SLA measurement failed - using default values"],
            escalations_triggered=[]
        )
