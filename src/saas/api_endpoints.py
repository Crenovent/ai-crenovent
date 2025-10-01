"""
SaaS API Endpoints
FastAPI endpoints for SaaS workflow execution and management
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Body
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json
import uuid
import logging
import asyncio
from io import BytesIO
import csv

from .dsl_templates import SaaSDSLTemplateManager, SaaSWorkflowType
from .parameter_packs import SaaSParameterPackManager, SaaSParameterCategory
from .schemas import SaaSSchemaFactory, SaaSWorkflowTrace, SaaSAuditTrail, SaaSEvidencePack
from .compiler_enhancements import SaaSRuntimeEnhancer, PolicyInjector

logger = logging.getLogger(__name__)

# Initialize components
template_manager = SaaSDSLTemplateManager()
parameter_manager = SaaSParameterPackManager()
schema_factory = SaaSSchemaFactory()
runtime_enhancer = SaaSRuntimeEnhancer()
policy_injector = PolicyInjector()

# Request/Response Models
class WorkflowExecutionRequest(BaseModel):
    workflow_type: SaaSWorkflowType
    tenant_id: str
    user_id: Optional[str] = None
    parameters: Dict[str, Any] = {}
    execution_mode: str = "sync"  # "sync", "async"
    priority: str = "normal"  # "low", "normal", "high", "critical"
    
    class Config:
        use_enum_values = True

class WorkflowExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: str
    status: str  # "running", "completed", "failed", "cancelled"
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    trace_id: Optional[str] = None
    audit_trail_id: Optional[str] = None

class WorkflowListResponse(BaseModel):
    workflows: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int

class ParameterPackResponse(BaseModel):
    pack_id: str
    name: str
    description: str
    category: str
    version: str
    parameters: Dict[str, Any]
    compliance_frameworks: List[str]

class PolicyEvaluationResponse(BaseModel):
    policy_id: str
    policy_name: str
    is_compliant: bool
    violations: List[Dict[str, Any]] = []
    warnings: List[str] = []
    evidence_collected: Dict[str, Any] = {}

class AuditTrailResponse(BaseModel):
    audit_id: str
    workflow_execution_id: str
    audit_type: str
    compliance_status: Dict[str, str]
    compliance_score: float
    total_execution_time_ms: int
    created_at: datetime

# Create router
saas_router = APIRouter(prefix="/api/v1/saas", tags=["SaaS Workflows"])

# Workflow Template Endpoints
@saas_router.get("/templates", response_model=List[Dict[str, Any]])
async def list_workflow_templates():
    """List all available SaaS workflow templates"""
    try:
        available_templates = template_manager.list_available_templates()
        
        templates_info = []
        for template_type in available_templates:
            workflow_type = SaaSWorkflowType(template_type)
            template = template_manager.get_template(workflow_type)
            
            templates_info.append({
                "workflow_type": template_type,
                "workflow_id": template["workflow_id"],
                "name": template["name"],
                "description": template["description"],
                "version": template["version"],
                "compliance_frameworks": template.get("compliance_frameworks", []),
                "parameters_count": len(template.get("parameters", {})),
                "steps_count": len(template.get("steps", []))
            })
        
        return templates_info
        
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.get("/templates/{workflow_type}", response_model=Dict[str, Any])
async def get_workflow_template(workflow_type: SaaSWorkflowType):
    """Get a specific workflow template"""
    try:
        template = template_manager.get_template(workflow_type)
        return template
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.post("/templates/{workflow_type}/validate")
async def validate_workflow_template(
    workflow_type: SaaSWorkflowType,
    template_data: Dict[str, Any] = Body(...)
):
    """Validate a workflow template"""
    try:
        validation_result = template_manager.validate_template(template_data)
        
        return {
            "is_valid": validation_result["is_valid"],
            "errors": validation_result["errors"],
            "warnings": validation_result["warnings"]
        }
        
    except Exception as e:
        logger.error(f"Error validating template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Parameter Pack Endpoints
@saas_router.get("/parameter-packs", response_model=List[ParameterPackResponse])
async def list_parameter_packs(category: Optional[SaaSParameterCategory] = None):
    """List all parameter packs, optionally filtered by category"""
    try:
        packs = parameter_manager.list_parameter_packs(category)
        
        return [
            ParameterPackResponse(
                pack_id=pack.pack_id,
                name=pack.name,
                description=pack.description,
                category=pack.category.value,
                version=pack.version,
                parameters=pack.parameters,
                compliance_frameworks=[cf.value for cf in pack.compliance_frameworks]
            )
            for pack in packs
        ]
        
    except Exception as e:
        logger.error(f"Error listing parameter packs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.get("/parameter-packs/{pack_id}", response_model=ParameterPackResponse)
async def get_parameter_pack(pack_id: str):
    """Get a specific parameter pack"""
    try:
        pack = parameter_manager.get_parameter_pack(pack_id)
        if not pack:
            raise HTTPException(status_code=404, detail="Parameter pack not found")
        
        return ParameterPackResponse(
            pack_id=pack.pack_id,
            name=pack.name,
            description=pack.description,
            category=pack.category.value,
            version=pack.version,
            parameters=pack.parameters,
            compliance_frameworks=[cf.value for cf in pack.compliance_frameworks]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting parameter pack: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.post("/parameter-packs/{pack_id}/validate")
async def validate_parameters(
    pack_id: str,
    parameters: Dict[str, Any] = Body(...)
):
    """Validate workflow parameters against a parameter pack"""
    try:
        validation_result = parameter_manager.validate_parameters(pack_id, parameters)
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Workflow Execution Endpoints
@saas_router.post("/workflows/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks
):
    """Execute a SaaS workflow"""
    try:
        execution_id = str(uuid.uuid4())
        
        # Get workflow template
        template = template_manager.get_template(request.workflow_type)
        workflow_id = template["workflow_id"]
        
        # Merge parameters with defaults
        merged_parameters = parameter_manager.merge_parameters(
            f"saas_{request.workflow_type.value}_v1.0",
            request.parameters
        )
        
        # Inject policies
        enhanced_template = policy_injector.inject_policies_into_workflow(template)
        
        # Create execution response
        response = WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status="running",
            started_at=datetime.utcnow()
        )
        
        if request.execution_mode == "async":
            # Execute asynchronously
            background_tasks.add_task(
                _execute_workflow_async,
                execution_id,
                enhanced_template,
                merged_parameters,
                request.tenant_id,
                request.user_id
            )
            return response
        else:
            # Execute synchronously
            result = await _execute_workflow_sync(
                execution_id,
                enhanced_template,
                merged_parameters,
                request.tenant_id,
                request.user_id
            )
            
            response.status = result["status"]
            response.completed_at = result["completed_at"]
            response.duration_ms = result["duration_ms"]
            response.result = result.get("result")
            response.error = result.get("error")
            response.trace_id = result.get("trace_id")
            response.audit_trail_id = result.get("audit_trail_id")
            
            return response
            
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.get("/workflows/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(execution_id: str):
    """Get workflow execution status and results"""
    try:
        # In a real implementation, this would query the database
        # For now, return a mock response
        return WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id="saas_pipeline_hygiene_v1.0",
            status="completed",
            started_at=datetime.utcnow() - timedelta(minutes=5),
            completed_at=datetime.utcnow(),
            duration_ms=300000,
            result={"message": "Workflow completed successfully"}
        )
        
    except Exception as e:
        logger.error(f"Error getting workflow execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.get("/workflows/executions", response_model=WorkflowListResponse)
async def list_workflow_executions(
    tenant_id: Optional[str] = Query(None),
    workflow_type: Optional[SaaSWorkflowType] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """List workflow executions with filtering and pagination"""
    try:
        # Mock response - in real implementation, query database
        mock_executions = [
            {
                "execution_id": f"exec_{i}",
                "workflow_id": "saas_pipeline_hygiene_v1.0",
                "workflow_type": "pipeline_hygiene",
                "tenant_id": tenant_id or "tenant_123",
                "status": "completed",
                "started_at": datetime.utcnow() - timedelta(hours=i),
                "duration_ms": 300000 + (i * 1000)
            }
            for i in range(1, 11)
        ]
        
        # Apply filters
        if workflow_type:
            mock_executions = [
                e for e in mock_executions 
                if e["workflow_type"] == workflow_type.value
            ]
        
        if status:
            mock_executions = [
                e for e in mock_executions 
                if e["status"] == status
            ]
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_executions = mock_executions[start_idx:end_idx]
        
        return WorkflowListResponse(
            workflows=paginated_executions,
            total_count=len(mock_executions),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing workflow executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Policy and Compliance Endpoints
@saas_router.post("/workflows/{workflow_id}/policies/evaluate")
async def evaluate_workflow_policies(
    workflow_id: str,
    execution_context: Dict[str, Any] = Body(...)
):
    """Evaluate policies for a workflow"""
    try:
        applicable_policies = policy_injector.get_applicable_policies(workflow_id)
        
        evaluation_results = []
        for policy in applicable_policies:
            # Mock evaluation - in real implementation, use PolicyEnforcer
            evaluation_results.append(
                PolicyEvaluationResponse(
                    policy_id=policy.policy_id,
                    policy_name=policy.name,
                    is_compliant=True,  # Mock result
                    violations=[],
                    warnings=[],
                    evidence_collected={}
                )
            )
        
        return {
            "workflow_id": workflow_id,
            "evaluation_results": evaluation_results,
            "overall_compliance": all(r.is_compliant for r in evaluation_results),
            "evaluated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error evaluating policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.get("/workflows/executions/{execution_id}/audit-trail", response_model=AuditTrailResponse)
async def get_audit_trail(execution_id: str):
    """Get audit trail for a workflow execution"""
    try:
        # Mock response - in real implementation, query database
        return AuditTrailResponse(
            audit_id=f"audit_{execution_id}",
            workflow_execution_id=execution_id,
            audit_type="execution",
            compliance_status={"SOX": "compliant", "GDPR": "compliant"},
            compliance_score=1.0,
            total_execution_time_ms=300000,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting audit trail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.get("/workflows/executions/{execution_id}/evidence")
async def get_evidence_pack(execution_id: str):
    """Get evidence pack for a workflow execution"""
    try:
        # Create mock evidence pack
        evidence_pack = schema_factory.create_evidence_pack(
            tenant_id="tenant_123",
            workflow_type="pipeline_hygiene",
            workflow_execution_id=execution_id,
            pack_name=f"Evidence Pack for {execution_id}"
        )
        
        return evidence_pack.dict()
        
    except Exception as e:
        logger.error(f"Error getting evidence pack: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export and Reporting Endpoints
@saas_router.get("/workflows/executions/{execution_id}/export")
async def export_execution_data(
    execution_id: str,
    format: str = Query("json", regex="^(json|csv|pdf)$")
):
    """Export workflow execution data in various formats"""
    try:
        # Mock data - in real implementation, query database
        execution_data = {
            "execution_id": execution_id,
            "workflow_id": "saas_pipeline_hygiene_v1.0",
            "status": "completed",
            "started_at": datetime.utcnow() - timedelta(minutes=5),
            "completed_at": datetime.utcnow(),
            "duration_ms": 300000,
            "steps_executed": [
                {"step_id": "calculate_metrics", "status": "completed", "duration_ms": 150000},
                {"step_id": "evaluate_rules", "status": "completed", "duration_ms": 100000},
                {"step_id": "send_notifications", "status": "completed", "duration_ms": 50000}
            ]
        }
        
        if format == "json":
            return JSONResponse(content=execution_data)
        
        elif format == "csv":
            # Create CSV export
            output = BytesIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(["Field", "Value"])
            
            # Write data
            for key, value in execution_data.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                writer.writerow([key, value])
            
            output.seek(0)
            
            return StreamingResponse(
                BytesIO(output.read()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=execution_{execution_id}.csv"}
            )
        
        elif format == "pdf":
            # Mock PDF export - in real implementation, generate actual PDF
            return JSONResponse(
                content={"message": "PDF export not implemented yet"},
                status_code=501
            )
        
    except Exception as e:
        logger.error(f"Error exporting execution data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@saas_router.get("/analytics/dashboard")
async def get_analytics_dashboard(
    tenant_id: Optional[str] = Query(None),
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d)$")
):
    """Get analytics dashboard data"""
    try:
        # Mock analytics data
        return {
            "tenant_id": tenant_id,
            "time_range": time_range,
            "metrics": {
                "total_executions": 1250,
                "successful_executions": 1200,
                "failed_executions": 50,
                "success_rate": 0.96,
                "average_execution_time_ms": 285000,
                "policy_violations": 15,
                "compliance_score": 0.98
            },
            "workflow_performance": {
                "pipeline_hygiene": {"executions": 800, "avg_time_ms": 250000, "success_rate": 0.98},
                "forecast_accuracy": {"executions": 300, "avg_time_ms": 350000, "success_rate": 0.95},
                "lead_scoring": {"executions": 150, "avg_time_ms": 180000, "success_rate": 0.99}
            },
            "compliance_status": {
                "SOX": {"compliant": 1180, "violations": 8, "compliance_rate": 0.993},
                "GDPR": {"compliant": 145, "violations": 2, "compliance_rate": 0.986}
            },
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions for workflow execution
async def _execute_workflow_sync(
    execution_id: str,
    workflow_template: Dict[str, Any],
    parameters: Dict[str, Any],
    tenant_id: str,
    user_id: Optional[str]
) -> Dict[str, Any]:
    """Execute workflow synchronously"""
    start_time = datetime.utcnow()
    
    try:
        async with runtime_enhancer.enhanced_execution_context(
            execution_id, workflow_template["workflow_id"], tenant_id, user_id
        ) as execution_context:
            
            # Mock execution - in real implementation, execute actual workflow steps
            await asyncio.sleep(0.3)  # Simulate execution time
            
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return {
                "status": "completed",
                "completed_at": end_time,
                "duration_ms": duration_ms,
                "result": {
                    "message": "Workflow executed successfully",
                    "steps_completed": len(workflow_template.get("steps", [])),
                    "parameters_used": parameters
                },
                "trace_id": f"trace_{execution_id}",
                "audit_trail_id": f"audit_{execution_id}"
            }
    
    except Exception as e:
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return {
            "status": "failed",
            "completed_at": end_time,
            "duration_ms": duration_ms,
            "error": str(e)
        }

async def _execute_workflow_async(
    execution_id: str,
    workflow_template: Dict[str, Any],
    parameters: Dict[str, Any],
    tenant_id: str,
    user_id: Optional[str]
):
    """Execute workflow asynchronously"""
    logger.info(f"Starting async execution: {execution_id}")
    
    try:
        result = await _execute_workflow_sync(
            execution_id, workflow_template, parameters, tenant_id, user_id
        )
        
        logger.info(f"Async execution completed: {execution_id} - {result['status']}")
        
        # In real implementation, update database with results
        
    except Exception as e:
        logger.error(f"Async execution failed: {execution_id} - {e}")

# Include router in main application
def get_saas_router():
    """Get the SaaS API router"""
    return saas_router
