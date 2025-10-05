"""
SLA Enforcement API
Task 9.1.11: Automate SLA tier enforcement in Control Plane

REST API endpoints for SLA tier management, enforcement, and monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, Body, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from dsl.orchestration.sla_enforcement_engine import (
    SLAEnforcementEngine, SLATier, SLAStatus, ResourceType
)
from src.services.connection_pool_manager import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API requests/responses

class EnforceRequestSLARequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant identifier")
    request_type: str = Field(..., description="Type of request")
    estimated_duration_ms: Optional[int] = Field(None, description="Estimated request duration")

class UpdateSLATierRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant identifier")
    new_tier: SLATier = Field(..., description="New SLA tier")
    updated_by: str = Field(..., description="User making the change")
    reason: Optional[str] = Field("", description="Reason for tier change")

class TriggerEscalationRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant identifier")
    violation_type: str = Field(..., description="Type of SLA violation")
    details: Dict[str, Any] = Field(..., description="Violation details")

class SLAEnforcementResponse(BaseModel):
    tenant_id: int
    sla_tier: str
    decision: str  # allow, queue, throttle, deny
    priority: int
    queue_position: Optional[int] = None
    estimated_wait_ms: Optional[int] = None
    resource_allocation: Optional[Dict[str, Any]] = None
    actions_taken: int
    compliance_status: str

class SLAMetricsResponse(BaseModel):
    tenant_id: int
    sla_tier: str
    current_response_time_ms: float
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    uptime_percentage: float
    error_rate_percentage: float
    success_rate_percentage: float
    cpu_utilization_percentage: float
    memory_utilization_percentage: float
    concurrent_requests_count: int
    api_calls_current_minute: int
    sla_status: str
    error_budget_remaining_percentage: float
    time_to_sla_violation_minutes: Optional[int]
    measurement_timestamp: datetime
    last_violation_timestamp: Optional[datetime] = None

class SLAConfigurationResponse(BaseModel):
    tier: str
    max_response_time_ms: int
    uptime_percentage: float
    availability_slo: float
    cpu_cores: float
    memory_gb: float
    storage_gb: float
    concurrent_requests: int
    api_calls_per_minute: int
    priority_weight: int
    queue_timeout_ms: int
    retry_attempts: int
    error_budget_percentage: float
    alert_threshold_percentage: float
    escalation_threshold_percentage: float
    cost_multiplier: float
    billing_tier: str

class SLAStatusReportResponse(BaseModel):
    tenant_id: Optional[int] = None
    sla_tier: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None
    compliance: Optional[bool] = None
    total_tenants: Optional[int] = None
    tenants_by_tier: Optional[Dict[str, int]] = None
    compliance_summary: Optional[Dict[str, int]] = None
    enforcement_stats: Optional[Dict[str, int]] = None
    timestamp: str
    error: Optional[str] = None

# Global SLA enforcement engine instance
sla_engine = None

async def get_sla_engine(pool_manager=Depends(get_pool_manager)) -> SLAEnforcementEngine:
    """Get or create SLA enforcement engine instance"""
    global sla_engine
    if sla_engine is None:
        sla_engine = SLAEnforcementEngine(pool_manager)
        await sla_engine.initialize()
    return sla_engine

@router.post("/enforce-request", response_model=SLAEnforcementResponse)
async def enforce_sla_for_request(
    request: EnforceRequestSLARequest,
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Enforce SLA for incoming request
    Task 9.1.11: Predictable service levels with tier enforcement
    """
    try:
        enforcement_result = await engine.enforce_sla_for_request(
            tenant_id=request.tenant_id,
            request_type=request.request_type,
            estimated_duration_ms=request.estimated_duration_ms
        )
        
        return SLAEnforcementResponse(
            tenant_id=enforcement_result["tenant_id"],
            sla_tier=enforcement_result["sla_tier"],
            decision=enforcement_result["decision"],
            priority=enforcement_result["priority"],
            queue_position=enforcement_result.get("queue_position"),
            estimated_wait_ms=enforcement_result.get("estimated_wait_ms"),
            resource_allocation=enforcement_result.get("resource_allocation"),
            actions_taken=enforcement_result["actions_taken"],
            compliance_status=enforcement_result["compliance_status"]
        )
        
    except Exception as e:
        logger.error(f"SLA enforcement failed: {e}")
        raise HTTPException(status_code=500, detail=f"SLA enforcement failed: {str(e)}")

@router.post("/update-tier", response_model=Dict[str, Any])
async def update_tenant_sla_tier(
    request: UpdateSLATierRequest,
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Update tenant's SLA tier
    Task 9.1.11: SLA tier management
    """
    try:
        success = await engine.update_tenant_sla_tier(
            tenant_id=request.tenant_id,
            new_tier=request.new_tier,
            updated_by=request.updated_by
        )
        
        if success:
            return {
                "tenant_id": request.tenant_id,
                "new_tier": request.new_tier.value,
                "updated_by": request.updated_by,
                "message": "SLA tier updated successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update SLA tier")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SLA tier update failed: {e}")
        raise HTTPException(status_code=500, detail=f"SLA tier update failed: {str(e)}")

@router.get("/metrics/{tenant_id}", response_model=SLAMetricsResponse)
async def get_tenant_sla_metrics(
    tenant_id: int,
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Get current SLA metrics for tenant
    Task 9.1.11: SLA monitoring and observability
    """
    try:
        metrics = await engine.get_sla_metrics(tenant_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No SLA metrics found for tenant {tenant_id}")
        
        return SLAMetricsResponse(
            tenant_id=metrics.tenant_id,
            sla_tier=metrics.sla_tier.value,
            current_response_time_ms=metrics.current_response_time_ms,
            average_response_time_ms=metrics.average_response_time_ms,
            p95_response_time_ms=metrics.p95_response_time_ms,
            p99_response_time_ms=metrics.p99_response_time_ms,
            uptime_percentage=metrics.uptime_percentage,
            error_rate_percentage=metrics.error_rate_percentage,
            success_rate_percentage=metrics.success_rate_percentage,
            cpu_utilization_percentage=metrics.cpu_utilization_percentage,
            memory_utilization_percentage=metrics.memory_utilization_percentage,
            concurrent_requests_count=metrics.concurrent_requests_count,
            api_calls_current_minute=metrics.api_calls_current_minute,
            sla_status=metrics.sla_status.value,
            error_budget_remaining_percentage=metrics.error_budget_remaining_percentage,
            time_to_sla_violation_minutes=metrics.time_to_sla_violation_minutes,
            measurement_timestamp=metrics.measurement_timestamp,
            last_violation_timestamp=metrics.last_violation_timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get SLA metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLA metrics: {str(e)}")

@router.get("/status-report", response_model=SLAStatusReportResponse)
async def get_sla_status_report(
    tenant_id: Optional[int] = Query(None, description="Specific tenant ID (optional)"),
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Generate SLA status report for tenant or all tenants
    Task 9.1.11: SLA compliance reporting
    """
    try:
        report = await engine.get_sla_status_report(tenant_id)
        
        return SLAStatusReportResponse(
            tenant_id=report.get("tenant_id"),
            sla_tier=report.get("sla_tier"),
            metrics=report.get("metrics"),
            configuration=report.get("configuration"),
            compliance=report.get("compliance"),
            total_tenants=report.get("total_tenants"),
            tenants_by_tier=report.get("tenants_by_tier"),
            compliance_summary=report.get("compliance_summary"),
            enforcement_stats=report.get("enforcement_stats"),
            timestamp=report.get("timestamp", datetime.now().isoformat()),
            error=report.get("error")
        )
        
    except Exception as e:
        logger.error(f"Failed to generate SLA status report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.post("/trigger-escalation", response_model=Dict[str, Any])
async def trigger_sla_escalation(
    request: TriggerEscalationRequest,
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Trigger SLA escalation for violations
    Task 9.1.11: SLA violation escalation
    """
    try:
        success = await engine.trigger_sla_escalation(
            tenant_id=request.tenant_id,
            violation_type=request.violation_type,
            details=request.details
        )
        
        if success:
            return {
                "tenant_id": request.tenant_id,
                "violation_type": request.violation_type,
                "escalation_triggered": True,
                "message": "SLA escalation triggered successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to trigger SLA escalation")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SLA escalation failed: {e}")
        raise HTTPException(status_code=500, detail=f"SLA escalation failed: {str(e)}")

@router.get("/configurations", response_model=Dict[str, SLAConfigurationResponse])
async def get_sla_configurations(
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Get SLA configurations for all tiers
    Task 9.1.11: SLA configuration management
    """
    try:
        configurations = {}
        
        for tier, config in engine.sla_configs.items():
            configurations[tier.value] = SLAConfigurationResponse(
                tier=config.tier.value,
                max_response_time_ms=config.max_response_time_ms,
                uptime_percentage=config.uptime_percentage,
                availability_slo=config.availability_slo,
                cpu_cores=config.cpu_cores,
                memory_gb=config.memory_gb,
                storage_gb=config.storage_gb,
                concurrent_requests=config.concurrent_requests,
                api_calls_per_minute=config.api_calls_per_minute,
                priority_weight=config.priority_weight,
                queue_timeout_ms=config.queue_timeout_ms,
                retry_attempts=config.retry_attempts,
                error_budget_percentage=config.error_budget_percentage,
                alert_threshold_percentage=config.alert_threshold_percentage,
                escalation_threshold_percentage=config.escalation_threshold_percentage,
                cost_multiplier=config.cost_multiplier,
                billing_tier=config.billing_tier
            )
        
        return configurations
        
    except Exception as e:
        logger.error(f"Failed to get SLA configurations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configurations: {str(e)}")

@router.get("/tiers", response_model=List[str])
async def get_available_sla_tiers():
    """
    Get available SLA tiers
    Task 9.1.11: SLA tier discovery
    """
    return [tier.value for tier in SLATier]

@router.get("/resource-types", response_model=List[str])
async def get_resource_types():
    """
    Get available resource types for SLA enforcement
    Task 9.1.11: Resource type discovery
    """
    return [resource.value for resource in ResourceType]

@router.get("/tenant/{tenant_id}/resource-allocation", response_model=Dict[str, Any])
async def get_tenant_resource_allocation(
    tenant_id: int,
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Get current resource allocation for tenant
    Task 9.1.11: Resource allocation visibility
    """
    try:
        allocation = engine.resource_allocations.get(tenant_id)
        
        if not allocation:
            raise HTTPException(status_code=404, detail=f"No resource allocation found for tenant {tenant_id}")
        
        return {
            "tenant_id": tenant_id,
            "resource_allocation": allocation,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get resource allocation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get allocation: {str(e)}")

@router.get("/enforcement-stats", response_model=Dict[str, Any])
async def get_enforcement_statistics(
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Get SLA enforcement statistics
    Task 9.1.11: Enforcement observability
    """
    try:
        return {
            "enforcement_stats": engine.enforcement_stats.copy(),
            "active_tenants": len(engine.tenant_metrics),
            "total_resource_allocations": len(engine.resource_allocations),
            "active_enforcement_actions": len(engine.enforcement_actions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get enforcement statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/simulate-load", response_model=Dict[str, Any])
async def simulate_sla_load(
    tenant_id: int = Body(..., description="Tenant ID"),
    request_count: int = Body(..., description="Number of requests to simulate"),
    request_type: str = Body("api_call", description="Type of requests"),
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Simulate load for SLA testing
    Task 9.1.11: SLA load testing
    """
    try:
        results = []
        
        for i in range(request_count):
            result = await engine.enforce_sla_for_request(
                tenant_id=tenant_id,
                request_type=f"{request_type}_{i}",
                estimated_duration_ms=1000
            )
            results.append(result)
        
        # Analyze results
        decisions = {}
        for result in results:
            decision = result["decision"]
            decisions[decision] = decisions.get(decision, 0) + 1
        
        return {
            "tenant_id": tenant_id,
            "requests_simulated": request_count,
            "decision_summary": decisions,
            "average_priority": sum(r["priority"] for r in results) / len(results),
            "actions_triggered": sum(r["actions_taken"] for r in results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"SLA load simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Load simulation failed: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    engine: SLAEnforcementEngine = Depends(get_sla_engine)
):
    """
    Health check for SLA enforcement service
    """
    try:
        # Basic health checks
        active_tenants = len(engine.tenant_metrics)
        total_actions = engine.enforcement_stats["actions_executed"]
        
        return {
            "status": "healthy",
            "service": "sla_enforcement",
            "active_tenants": active_tenants,
            "total_actions_executed": total_actions,
            "available_tiers": [tier.value for tier in SLATier],
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "sla_enforcement",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Export router
__all__ = ["router"]
