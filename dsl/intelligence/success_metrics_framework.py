#!/usr/bin/env python3
"""
Success Metrics Framework - Task 1.1-T5
========================================
Document success metrics (accuracy, SLA adherence, adoption, governance)
Clear measurement framework aligned with Vision Ch.12

Features:
- Comprehensive metrics collection and calculation
- Real-time and batch metric processing
- Multi-tenant metrics with tenant isolation
- Industry-specific metric overlays
- SLA tier-aware performance tracking
- Governance and compliance metrics
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json
import asyncio
from dataclasses import dataclass

from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# TASK 1.1-T5: SUCCESS METRICS FRAMEWORK
# =====================================================

class MetricCategory(str, Enum):
    ACCURACY = "accuracy"
    SLA_ADHERENCE = "sla_adherence"
    ADOPTION = "adoption"
    GOVERNANCE = "governance"
    BUSINESS_IMPACT = "business_impact"

class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"
    PERCENTAGE = "percentage"

class TenantTier(str, Enum):
    T0 = "T0"  # Premium: <200ms, 99.99% availability
    T1 = "T1"  # Standard: <500ms, 99.9% availability
    T2 = "T2"  # Basic: <1000ms, 99.5% availability

class IndustryCode(str, Enum):
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"

@dataclass
class MetricDefinition:
    """Task 1.1-T5: Metric definition structure"""
    metric_id: str
    name: str
    category: MetricCategory
    metric_type: MetricType
    description: str
    target_value: float
    unit: str
    calculation_method: str
    frequency: str
    tenant_tier_specific: bool = False
    industry_specific: bool = False

class SuccessMetricsFramework:
    """
    Task 1.1-T5: Success Metrics Framework Implementation
    Comprehensive metrics collection and calculation system
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize metric definitions (Task 1.1-T5)
        self.metric_definitions = self._initialize_metric_definitions()
        
        # SLA targets per tenant tier (Task 1.1-T5)
        self.sla_targets = {
            TenantTier.T0: {
                "response_time_ms": 200,
                "throughput_per_min": 10000,
                "availability_percent": 99.99,
                "recovery_time_min": 5
            },
            TenantTier.T1: {
                "response_time_ms": 500,
                "throughput_per_min": 5000,
                "availability_percent": 99.9,
                "recovery_time_min": 15
            },
            TenantTier.T2: {
                "response_time_ms": 1000,
                "throughput_per_min": 1000,
                "availability_percent": 99.5,
                "recovery_time_min": 30
            }
        }
    
    def _initialize_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Initialize all metric definitions (Task 1.1-T5)"""
        
        metrics = {}
        
        # ACCURACY METRICS
        metrics["workflow_success_rate"] = MetricDefinition(
            metric_id="workflow_success_rate",
            name="Workflow Success Rate",
            category=MetricCategory.ACCURACY,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of successful workflow executions",
            target_value=99.5,
            unit="percent",
            calculation_method="successful_executions / total_executions * 100",
            frequency="real-time"
        )
        
        metrics["data_quality_score"] = MetricDefinition(
            metric_id="data_quality_score",
            name="Data Quality Score",
            category=MetricCategory.ACCURACY,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of valid data points",
            target_value=95.0,
            unit="percent",
            calculation_method="valid_data_points / total_data_points * 100",
            frequency="daily"
        )
        
        metrics["policy_compliance_rate"] = MetricDefinition(
            metric_id="policy_compliance_rate",
            name="Policy Compliance Rate",
            category=MetricCategory.ACCURACY,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of policy-compliant executions",
            target_value=100.0,
            unit="percent",
            calculation_method="compliant_executions / total_executions * 100",
            frequency="real-time"
        )
        
        # SLA ADHERENCE METRICS
        metrics["response_time"] = MetricDefinition(
            metric_id="response_time",
            name="Response Time",
            category=MetricCategory.SLA_ADHERENCE,
            metric_type=MetricType.HISTOGRAM,
            description="API response time distribution",
            target_value=500.0,  # Default T1 target
            unit="milliseconds",
            calculation_method="p95(response_times)",
            frequency="real-time",
            tenant_tier_specific=True
        )
        
        metrics["availability"] = MetricDefinition(
            metric_id="availability",
            name="System Availability",
            category=MetricCategory.SLA_ADHERENCE,
            metric_type=MetricType.PERCENTAGE,
            description="System uptime percentage",
            target_value=99.9,  # Default T1 target
            unit="percent",
            calculation_method="uptime / total_time * 100",
            frequency="real-time",
            tenant_tier_specific=True
        )
        
        # ADOPTION METRICS
        metrics["active_users"] = MetricDefinition(
            metric_id="active_users",
            name="Active Users",
            category=MetricCategory.ADOPTION,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of licensed users who are active",
            target_value=80.0,
            unit="percent",
            calculation_method="monthly_active_users / licensed_users * 100",
            frequency="monthly"
        )
        
        metrics["workflow_creation_rate"] = MetricDefinition(
            metric_id="workflow_creation_rate",
            name="Workflow Creation Rate",
            category=MetricCategory.ADOPTION,
            metric_type=MetricType.RATE,
            description="New workflows created per user per month",
            target_value=5.0,
            unit="workflows/user/month",
            calculation_method="new_workflows / active_users",
            frequency="monthly"
        )
        
        # GOVERNANCE METRICS
        metrics["policy_violation_rate"] = MetricDefinition(
            metric_id="policy_violation_rate",
            name="Policy Violation Rate",
            category=MetricCategory.GOVERNANCE,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of executions with policy violations",
            target_value=0.1,
            unit="percent",
            calculation_method="policy_violations / total_executions * 100",
            frequency="real-time"
        )
        
        metrics["audit_readiness_score"] = MetricDefinition(
            metric_id="audit_readiness_score",
            name="Audit Readiness Score",
            category=MetricCategory.GOVERNANCE,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of executions with complete audit trails",
            target_value=100.0,
            unit="percent",
            calculation_method="complete_audit_trails / total_executions * 100",
            frequency="daily"
        )
        
        # BUSINESS IMPACT METRICS
        metrics["process_automation_rate"] = MetricDefinition(
            metric_id="process_automation_rate",
            name="Process Automation Rate",
            category=MetricCategory.BUSINESS_IMPACT,
            metric_type=MetricType.PERCENTAGE,
            description="Percentage of process steps that are automated",
            target_value=70.0,
            unit="percent",
            calculation_method="automated_steps / total_process_steps * 100",
            frequency="monthly"
        )
        
        metrics["cost_savings"] = MetricDefinition(
            metric_id="cost_savings",
            name="Cost Savings",
            category=MetricCategory.BUSINESS_IMPACT,
            metric_type=MetricType.GAUGE,
            description="Operational cost reduction percentage",
            target_value=30.0,
            unit="percent",
            calculation_method="(baseline_cost - current_cost) / baseline_cost * 100",
            frequency="quarterly"
        )
        
        return metrics
    
    async def calculate_metric(
        self,
        metric_id: str,
        tenant_id: int,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate specific metric value (Task 1.1-T5)
        """
        try:
            if metric_id not in self.metric_definitions:
                raise HTTPException(status_code=404, detail=f"Metric {metric_id} not found")
            
            metric_def = self.metric_definitions[metric_id]
            
            # Get tenant tier for SLA-specific metrics
            tenant_tier = await self._get_tenant_tier(tenant_id)
            
            # Calculate metric based on type and category
            if metric_def.category == MetricCategory.ACCURACY:
                value = await self._calculate_accuracy_metric(
                    metric_id, tenant_id, start_time, end_time, filters
                )
            elif metric_def.category == MetricCategory.SLA_ADHERENCE:
                value = await self._calculate_sla_metric(
                    metric_id, tenant_id, tenant_tier, start_time, end_time, filters
                )
            elif metric_def.category == MetricCategory.ADOPTION:
                value = await self._calculate_adoption_metric(
                    metric_id, tenant_id, start_time, end_time, filters
                )
            elif metric_def.category == MetricCategory.GOVERNANCE:
                value = await self._calculate_governance_metric(
                    metric_id, tenant_id, start_time, end_time, filters
                )
            elif metric_def.category == MetricCategory.BUSINESS_IMPACT:
                value = await self._calculate_business_impact_metric(
                    metric_id, tenant_id, start_time, end_time, filters
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown metric category: {metric_def.category}")
            
            # Get target value (adjust for tenant tier if applicable)
            target_value = self._get_target_value(metric_def, tenant_tier)
            
            # Calculate performance vs target
            performance_ratio = value / target_value if target_value > 0 else 0
            status = "meeting_target" if performance_ratio >= 1.0 else "below_target"
            
            return {
                "metric_id": metric_id,
                "tenant_id": tenant_id,
                "value": value,
                "target_value": target_value,
                "performance_ratio": performance_ratio,
                "status": status,
                "unit": metric_def.unit,
                "calculation_time": datetime.utcnow().isoformat(),
                "period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metric {metric_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Metric calculation failed: {e}")
    
    async def _calculate_accuracy_metric(
        self,
        metric_id: str,
        tenant_id: int,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate accuracy metrics (Task 1.1-T5)"""
        
        if metric_id == "workflow_success_rate":
            # Query workflow execution results
            query = """
                SELECT 
                    COUNT(*) FILTER (WHERE status = 'completed') as successful,
                    COUNT(*) as total
                FROM dsl_execution_traces 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                
                if result['total'] == 0:
                    return 0.0
                
                return (result['successful'] / result['total']) * 100
        
        elif metric_id == "data_quality_score":
            # Query data quality metrics
            query = """
                SELECT 
                    COUNT(*) FILTER (WHERE data_quality_score >= 0.95) as valid,
                    COUNT(*) as total
                FROM workflow_executions 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                
                if result['total'] == 0:
                    return 0.0
                
                return (result['valid'] / result['total']) * 100
        
        elif metric_id == "policy_compliance_rate":
            # Query policy compliance
            query = """
                SELECT 
                    COUNT(*) FILTER (WHERE policy_violations = 0) as compliant,
                    COUNT(*) as total
                FROM dsl_execution_traces 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                
                if result['total'] == 0:
                    return 100.0  # No executions = 100% compliant
                
                return (result['compliant'] / result['total']) * 100
        
        return 0.0
    
    async def _calculate_sla_metric(
        self,
        metric_id: str,
        tenant_id: int,
        tenant_tier: TenantTier,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate SLA adherence metrics (Task 1.1-T5)"""
        
        if metric_id == "response_time":
            # Query response times and calculate P95
            query = """
                SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_response_time
                FROM dsl_execution_traces 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                return float(result['p95_response_time'] or 0)
        
        elif metric_id == "availability":
            # Calculate system availability
            total_minutes = (end_time - start_time).total_seconds() / 60
            
            # Query downtime minutes
            query = """
                SELECT COALESCE(SUM(EXTRACT(EPOCH FROM (resolved_at - created_at))/60), 0) as downtime_minutes
                FROM system_incidents 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
                AND incident_type = 'outage'
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                downtime_minutes = float(result['downtime_minutes'])
                
                if total_minutes == 0:
                    return 100.0
                
                uptime_minutes = total_minutes - downtime_minutes
                return (uptime_minutes / total_minutes) * 100
        
        return 0.0
    
    async def _calculate_adoption_metric(
        self,
        metric_id: str,
        tenant_id: int,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate adoption metrics (Task 1.1-T5)"""
        
        if metric_id == "active_users":
            # Query active vs licensed users
            query = """
                SELECT 
                    COUNT(DISTINCT user_id) FILTER (WHERE last_login >= $2) as active_users,
                    COUNT(*) as licensed_users
                FROM users 
                WHERE tenant_id = $1 
                AND status = 'active'
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time)
                
                if result['licensed_users'] == 0:
                    return 0.0
                
                return (result['active_users'] / result['licensed_users']) * 100
        
        elif metric_id == "workflow_creation_rate":
            # Query workflow creation rate
            query = """
                SELECT 
                    COUNT(*) as new_workflows,
                    COUNT(DISTINCT created_by_user_id) as active_creators
                FROM dsl_workflows 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                
                if result['active_creators'] == 0:
                    return 0.0
                
                return result['new_workflows'] / result['active_creators']
        
        return 0.0
    
    async def _calculate_governance_metric(
        self,
        metric_id: str,
        tenant_id: int,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate governance metrics (Task 1.1-T5)"""
        
        if metric_id == "policy_violation_rate":
            # Query policy violations
            query = """
                SELECT 
                    COUNT(*) FILTER (WHERE policy_violations > 0) as violations,
                    COUNT(*) as total
                FROM dsl_execution_traces 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                
                if result['total'] == 0:
                    return 0.0
                
                return (result['violations'] / result['total']) * 100
        
        elif metric_id == "audit_readiness_score":
            # Query audit trail completeness
            query = """
                SELECT 
                    COUNT(*) FILTER (WHERE evidence_pack_id IS NOT NULL) as complete_trails,
                    COUNT(*) as total
                FROM dsl_execution_traces 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                
                if result['total'] == 0:
                    return 100.0
                
                return (result['complete_trails'] / result['total']) * 100
        
        return 0.0
    
    async def _calculate_business_impact_metric(
        self,
        metric_id: str,
        tenant_id: int,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate business impact metrics (Task 1.1-T5)"""
        
        if metric_id == "process_automation_rate":
            # Query automation coverage
            query = """
                SELECT 
                    COUNT(*) FILTER (WHERE automation_type != 'manual') as automated_steps,
                    COUNT(*) as total_steps
                FROM process_steps 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                
                if result['total_steps'] == 0:
                    return 0.0
                
                return (result['automated_steps'] / result['total_steps']) * 100
        
        elif metric_id == "cost_savings":
            # Query cost savings (simplified calculation)
            query = """
                SELECT 
                    baseline_cost,
                    current_cost
                FROM tenant_cost_metrics 
                WHERE tenant_id = $1 
                AND period_start >= $2 AND period_end <= $3
                ORDER BY period_end DESC
                LIMIT 1
            """
            
            async with self.pool_manager.get_connection() as conn:
                result = await conn.fetchrow(query, tenant_id, start_time, end_time)
                
                if not result or result['baseline_cost'] == 0:
                    return 0.0
                
                baseline = float(result['baseline_cost'])
                current = float(result['current_cost'])
                
                return ((baseline - current) / baseline) * 100
        
        return 0.0
    
    async def _get_tenant_tier(self, tenant_id: int) -> TenantTier:
        """Get tenant tier for SLA-specific metrics"""
        query = "SELECT tenant_tier FROM tenant_metadata WHERE tenant_id = $1"
        
        async with self.pool_manager.get_connection() as conn:
            result = await conn.fetchrow(query, tenant_id)
            
            if result:
                return TenantTier(result['tenant_tier'])
            
            return TenantTier.T1  # Default to T1
    
    def _get_target_value(self, metric_def: MetricDefinition, tenant_tier: TenantTier) -> float:
        """Get target value, adjusted for tenant tier if applicable"""
        
        if not metric_def.tenant_tier_specific:
            return metric_def.target_value
        
        # Adjust targets based on tenant tier
        if metric_def.metric_id == "response_time":
            return float(self.sla_targets[tenant_tier]["response_time_ms"])
        elif metric_def.metric_id == "availability":
            return self.sla_targets[tenant_tier]["availability_percent"]
        
        return metric_def.target_value

# Initialize service
success_metrics_framework = None

def get_success_metrics_framework():
    """Get success metrics framework instance"""
    global success_metrics_framework
    if success_metrics_framework is None:
        pool_manager = get_pool_manager()
        success_metrics_framework = SuccessMetricsFramework(pool_manager)
    return success_metrics_framework

# =====================================================
# API ENDPOINTS
# =====================================================

@router.get("/metrics/definitions")
async def get_metric_definitions(
    category: Optional[MetricCategory] = None,
    framework: SuccessMetricsFramework = Depends(get_success_metrics_framework)
):
    """
    Get all metric definitions (Task 1.1-T5)
    """
    definitions = framework.metric_definitions
    
    if category:
        definitions = {
            k: v for k, v in definitions.items() 
            if v.category == category
        }
    
    return {
        "metric_definitions": {
            k: {
                "metric_id": v.metric_id,
                "name": v.name,
                "category": v.category,
                "metric_type": v.metric_type,
                "description": v.description,
                "target_value": v.target_value,
                "unit": v.unit,
                "frequency": v.frequency
            }
            for k, v in definitions.items()
        }
    }

@router.get("/metrics/{metric_id}/calculate")
async def calculate_metric_endpoint(
    metric_id: str,
    tenant_id: int = Query(..., description="Tenant ID"),
    start_time: datetime = Query(..., description="Start time for calculation"),
    end_time: datetime = Query(..., description="End time for calculation"),
    framework: SuccessMetricsFramework = Depends(get_success_metrics_framework)
):
    """
    Calculate specific metric value (Task 1.1-T5)
    """
    return await framework.calculate_metric(
        metric_id=metric_id,
        tenant_id=tenant_id,
        start_time=start_time,
        end_time=end_time
    )

@router.get("/metrics/dashboard/{tenant_id}")
async def get_metrics_dashboard(
    tenant_id: int,
    category: Optional[MetricCategory] = None,
    framework: SuccessMetricsFramework = Depends(get_success_metrics_framework)
):
    """
    Get comprehensive metrics dashboard for tenant (Task 1.1-T5)
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)  # Last 30 days
    
    metrics_to_calculate = framework.metric_definitions.keys()
    if category:
        metrics_to_calculate = [
            k for k, v in framework.metric_definitions.items()
            if v.category == category
        ]
    
    results = {}
    for metric_id in metrics_to_calculate:
        try:
            result = await framework.calculate_metric(
                metric_id=metric_id,
                tenant_id=tenant_id,
                start_time=start_time,
                end_time=end_time
            )
            results[metric_id] = result
        except Exception as e:
            logger.error(f"Error calculating metric {metric_id}: {e}")
            results[metric_id] = {"error": str(e)}
    
    return {
        "tenant_id": tenant_id,
        "dashboard_generated_at": datetime.utcnow().isoformat(),
        "period": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        },
        "metrics": results
    }
