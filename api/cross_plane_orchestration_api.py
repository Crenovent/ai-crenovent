"""
Cross-Plane Orchestration API
Task 9.3.3: Build orchestration API for cross-plane calls

REST API endpoints for cross-plane message orchestration and workflow execution
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from dsl.orchestration.cross_plane_orchestrator import (
    CrossPlaneOrchestrator, PlaneType, MessageType, MessagePriority,
    GovernanceMetadata, ExecutionStatus
)
from src.services.connection_pool_manager import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API requests/responses

class GovernanceMetadataRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant identifier")
    region_id: str = Field(..., description="Region identifier")
    policy_pack_id: Optional[str] = Field(None, description="Policy pack ID")
    evidence_pack_id: Optional[str] = Field(None, description="Evidence pack ID")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")
    sla_tier: str = Field("T2", description="SLA tier (T0, T1, T2)")

class SendMessageRequest(BaseModel):
    target_plane: PlaneType = Field(..., description="Target plane")
    operation: str = Field(..., description="Operation to execute")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    governance: GovernanceMetadataRequest = Field(..., description="Governance metadata")
    source_plane: PlaneType = Field(PlaneType.CONTROL, description="Source plane")
    message_type: MessageType = Field(MessageType.COMMAND, description="Message type")
    priority: Optional[MessagePriority] = Field(None, description="Message priority")
    timeout_ms: Optional[int] = Field(None, description="Timeout in milliseconds")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")

class SendNotificationRequest(BaseModel):
    target_plane: PlaneType = Field(..., description="Target plane")
    event_type: str = Field(..., description="Event type")
    event_data: Dict[str, Any] = Field(..., description="Event data")
    governance: GovernanceMetadataRequest = Field(..., description="Governance metadata")
    source_plane: PlaneType = Field(PlaneType.CONTROL, description="Source plane")

class QueryPlaneRequest(BaseModel):
    target_plane: PlaneType = Field(..., description="Target plane to query")
    query_type: str = Field(..., description="Type of query")
    query_params: Dict[str, Any] = Field(..., description="Query parameters")
    governance: GovernanceMetadataRequest = Field(..., description="Governance metadata")
    source_plane: PlaneType = Field(PlaneType.CONTROL, description="Source plane")

class WorkflowStep(BaseModel):
    target_plane: PlaneType = Field(..., description="Target plane for this step")
    operation: str = Field(..., description="Operation to execute")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Step payload")
    use_previous_results: bool = Field(False, description="Include previous step results")
    condition: Optional[Dict[str, Any]] = Field(None, description="Conditional execution")

class ExecuteWorkflowRequest(BaseModel):
    workflow_name: str = Field(..., description="Workflow name")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    governance: GovernanceMetadataRequest = Field(..., description="Governance metadata")
    description: Optional[str] = Field("", description="Workflow description")

class MessageResponse(BaseModel):
    response_id: str
    original_message_id: str
    source_plane: PlaneType
    target_plane: PlaneType
    status: ExecutionStatus
    result: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: int
    created_at: datetime

class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    results: Dict[str, Any]
    steps_executed: int
    execution_time_ms: float
    error: Optional[str] = None

class PlaneHealthResponse(BaseModel):
    plane: str
    healthy: bool
    response_time_ms: int
    details: Dict[str, Any]
    error: Optional[str] = None

class OrchestratorMetricsResponse(BaseModel):
    metrics: Dict[str, int]
    registered_planes: List[str]
    active_correlations: int
    timestamp: str

# Global orchestrator instance
orchestrator = None

async def get_orchestrator(pool_manager=Depends(get_pool_manager)) -> CrossPlaneOrchestrator:
    """Get or create cross-plane orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = CrossPlaneOrchestrator(pool_manager)
        await orchestrator.initialize()
    return orchestrator

def _convert_governance_metadata(governance_req: GovernanceMetadataRequest) -> GovernanceMetadata:
    """Convert API request model to internal governance metadata"""
    return GovernanceMetadata(
        tenant_id=governance_req.tenant_id,
        region_id=governance_req.region_id,
        policy_pack_id=governance_req.policy_pack_id,
        evidence_pack_id=governance_req.evidence_pack_id,
        compliance_frameworks=governance_req.compliance_frameworks,
        sla_tier=governance_req.sla_tier
    )

@router.post("/send-message", response_model=MessageResponse)
async def send_cross_plane_message(
    request: SendMessageRequest,
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Send a cross-plane message and wait for response
    Task 9.3.3: Standardized control messages
    """
    try:
        governance = _convert_governance_metadata(request.governance)
        
        response = await orch.send_message(
            target_plane=request.target_plane,
            operation=request.operation,
            payload=request.payload,
            governance=governance,
            source_plane=request.source_plane,
            message_type=request.message_type,
            priority=request.priority,
            timeout_ms=request.timeout_ms,
            correlation_id=request.correlation_id
        )
        
        return MessageResponse(
            response_id=response.response_id,
            original_message_id=response.original_message_id,
            source_plane=response.source_plane,
            target_plane=response.target_plane,
            status=response.status,
            result=response.result,
            error=response.error,
            execution_time_ms=response.execution_time_ms,
            created_at=response.created_at
        )
        
    except Exception as e:
        logger.error(f"Cross-plane message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Message failed: {str(e)}")

@router.post("/send-notification", response_model=Dict[str, Any])
async def send_cross_plane_notification(
    request: SendNotificationRequest,
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Send a fire-and-forget notification to a plane
    Task 9.3.3: Cross-plane event notifications
    """
    try:
        governance = _convert_governance_metadata(request.governance)
        
        success = await orch.send_notification(
            target_plane=request.target_plane,
            event_type=request.event_type,
            event_data=request.event_data,
            governance=governance,
            source_plane=request.source_plane
        )
        
        return {
            "success": success,
            "message": "Notification sent successfully" if success else "Notification failed",
            "target_plane": request.target_plane.value,
            "event_type": request.event_type
        }
        
    except Exception as e:
        logger.error(f"Cross-plane notification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Notification failed: {str(e)}")

@router.post("/query-plane", response_model=Dict[str, Any])
async def query_plane(
    request: QueryPlaneRequest,
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Query a plane for information
    Task 9.3.3: Cross-plane queries
    """
    try:
        governance = _convert_governance_metadata(request.governance)
        
        result = await orch.query_plane(
            target_plane=request.target_plane,
            query_type=request.query_type,
            query_params=request.query_params,
            governance=governance,
            source_plane=request.source_plane
        )
        
        return {
            "query_type": request.query_type,
            "target_plane": request.target_plane.value,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Plane query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.post("/execute-workflow", response_model=WorkflowResponse)
async def execute_cross_plane_workflow(
    request: ExecuteWorkflowRequest,
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Execute a multi-plane workflow with orchestration
    Task 9.3.3: Cross-plane workflow orchestration
    """
    try:
        governance = _convert_governance_metadata(request.governance)
        
        # Convert workflow steps to dictionary format
        workflow_definition = {
            "name": request.workflow_name,
            "description": request.description,
            "steps": [
                {
                    "target_plane": step.target_plane.value,
                    "operation": step.operation,
                    "payload": step.payload,
                    "use_previous_results": step.use_previous_results,
                    "condition": step.condition
                }
                for step in request.steps
            ]
        }
        
        result = await orch.execute_cross_plane_workflow(
            workflow_definition=workflow_definition,
            governance=governance
        )
        
        return WorkflowResponse(
            workflow_id=result["workflow_id"],
            status=result["status"],
            results=result["results"],
            steps_executed=result.get("steps_executed", 0),
            execution_time_ms=result.get("execution_time_ms", 0),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Cross-plane workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

@router.get("/plane-health/{plane}", response_model=PlaneHealthResponse)
async def get_plane_health(
    plane: PlaneType,
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Get health status of a specific plane
    Task 9.3.3: Plane health monitoring
    """
    try:
        health_info = await orch.get_plane_health(plane)
        
        return PlaneHealthResponse(
            plane=health_info["plane"],
            healthy=health_info["healthy"],
            response_time_ms=health_info["response_time_ms"],
            details=health_info.get("details", {}),
            error=health_info.get("error")
        )
        
    except Exception as e:
        logger.error(f"Plane health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def get_all_planes_health(
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Get health status of all planes
    Task 9.3.3: System-wide health monitoring
    """
    try:
        health_results = {}
        
        for plane in PlaneType:
            try:
                health_info = await orch.get_plane_health(plane)
                health_results[plane.value] = health_info
            except Exception as e:
                health_results[plane.value] = {
                    "plane": plane.value,
                    "healthy": False,
                    "error": str(e),
                    "response_time_ms": -1
                }
        
        # Calculate overall health
        healthy_planes = sum(1 for h in health_results.values() if h["healthy"])
        total_planes = len(health_results)
        overall_healthy = healthy_planes == total_planes
        
        return {
            "overall_healthy": overall_healthy,
            "healthy_planes": healthy_planes,
            "total_planes": total_planes,
            "planes": health_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")

@router.get("/metrics", response_model=OrchestratorMetricsResponse)
async def get_orchestrator_metrics(
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Get orchestrator performance metrics
    Task 9.3.3: Orchestrator observability
    """
    try:
        metrics = await orch.get_orchestrator_metrics()
        
        return OrchestratorMetricsResponse(
            metrics=metrics["metrics"],
            registered_planes=[plane.value for plane in metrics["registered_planes"]],
            active_correlations=metrics["active_correlations"],
            timestamp=metrics["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get orchestrator metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics failed: {str(e)}")

@router.post("/register-handler", response_model=Dict[str, str])
async def register_plane_handler(
    plane: PlaneType = Body(..., description="Plane type"),
    handler_url: str = Body(..., description="Handler endpoint URL"),
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Register a handler for a specific plane
    Task 9.3.3: Dynamic plane handler registration
    """
    try:
        # Create a handler function that calls the provided URL
        async def url_handler(message):
            # This would make HTTP calls to the handler URL
            # For now, return a placeholder response
            from dsl.orchestration.cross_plane_orchestrator import CrossPlaneResponse, ExecutionStatus
            import uuid
            
            return CrossPlaneResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=message.message_id,
                source_plane=message.target_plane,
                target_plane=message.source_plane,
                status=ExecutionStatus.COMPLETED,
                result={"handler_url": handler_url, "message": "Handler registered"},
                governance=message.governance,
                execution_time_ms=100,
                created_at=datetime.now()
            )
        
        success = await orch.register_plane_handler(plane, url_handler)
        
        if success:
            return {
                "message": f"Handler registered for {plane.value} plane",
                "plane": plane.value,
                "handler_url": handler_url
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to register handler")
        
    except Exception as e:
        logger.error(f"Handler registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Handler registration failed: {str(e)}")

@router.get("/planes", response_model=List[str])
async def list_available_planes():
    """
    List all available plane types
    Task 9.3.3: Plane discovery
    """
    return [plane.value for plane in PlaneType]

@router.get("/message-types", response_model=List[str])
async def list_message_types():
    """
    List all available message types
    Task 9.3.3: Message type discovery
    """
    return [msg_type.value for msg_type in MessageType]

@router.get("/sla-tiers", response_model=Dict[str, Dict[str, Any]])
async def get_sla_configurations(
    orch: CrossPlaneOrchestrator = Depends(get_orchestrator)
):
    """
    Get SLA tier configurations
    Task 9.3.3: SLA tier information
    """
    return {
        "sla_configs": orch.sla_configs,
        "description": "SLA tier configurations for cross-plane messaging"
    }

# Export router
__all__ = ["router"]
