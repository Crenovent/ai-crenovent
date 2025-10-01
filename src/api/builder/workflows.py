"""
Builder API - Workflow Management Endpoints
==========================================

Implements Chapter 12.1 Builder APIs for frontend integration:
- Workflow CRUD operations
- Template management
- Validation and compilation
- Integration with DSL Compiler & Runtime

Frontend-Ready Components for user testing.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from datetime import datetime

from dsl.compiler import DSLCompiler, WorkflowValidator, WorkflowRuntime
from dsl.compiler.parser import DSLWorkflowAST, SAMPLE_PIPELINE_HYGIENE_WORKFLOW, SAMPLE_FORECAST_APPROVAL_WORKFLOW
from src.rba.builder import RBABuilder, RBAWorkflow, WorkflowState, BlockType

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for API
class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description") 
    industry: str = Field(default="SaaS", description="Industry focus")
    template_id: Optional[str] = Field(None, description="Template to base workflow on")
    tenant_id: int = Field(..., description="Tenant ID")
    user_id: int = Field(..., description="User ID")

class WorkflowUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    workflow_definition: Optional[str] = None
    state: Optional[str] = None

class WorkflowValidationRequest(BaseModel):
    workflow_definition: str = Field(..., description="YAML workflow definition")
    format: str = Field(default="yaml", description="Format: yaml or json")

class WorkflowResponse(BaseModel):
    workflow_id: str
    name: str
    description: str
    industry: str
    automation_type: str
    version: str
    state: str
    created_at: datetime
    created_by: int
    tenant_id: int

class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    plan_hash: Optional[str] = None

# Dependency to get app state
async def get_app_state(request: Request):
    return request.app.state

@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    workflow_request: WorkflowCreateRequest,
    app_state = Depends(get_app_state)
):
    """
    Create new RBA workflow
    Task 12.1-T15: Build workflow CRUD APIs
    """
    try:
        logger.info(f"🆕 Creating workflow: {workflow_request.name}")
        
        # Initialize RBA Builder
        builder = RBABuilder()
        
        # Create workflow
        workflow = await builder.create_workflow(
            name=workflow_request.name,
            description=workflow_request.description,
            tenant_id=workflow_request.tenant_id,
            user_id=workflow_request.user_id,
            industry=workflow_request.industry
        )
        
        # If template specified, apply template
        if workflow_request.template_id:
            await builder.apply_template(workflow.workflow_id, workflow_request.template_id)
        
        return WorkflowResponse(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            industry=workflow.industry,
            automation_type=workflow.automation_type,
            version=workflow.version,
            state=workflow.state.value,
            created_at=workflow.created_at,
            created_by=workflow.created_by,
            tenant_id=workflow.tenant_id
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")

@router.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows(
    tenant_id: int,
    user_id: Optional[int] = None,
    state: Optional[str] = None,
    industry: Optional[str] = None,
    app_state = Depends(get_app_state)
):
    """
    List workflows with filtering
    Task 12.1-T16: Build workflow listing APIs
    """
    try:
        logger.info(f"📋 Listing workflows for tenant: {tenant_id}")
        
        builder = RBABuilder()
        workflows = await builder.list_workflows(
            tenant_id=tenant_id,
            user_id=user_id,
            state=state,
            industry=industry
        )
        
        return [
            WorkflowResponse(
                workflow_id=wf.workflow_id,
                name=wf.name,
                description=wf.description,
                industry=wf.industry,
                automation_type=wf.automation_type,
                version=wf.version,
                state=wf.state.value,
                created_at=wf.created_at,
                created_by=wf.created_by,
                tenant_id=wf.tenant_id
            )
            for wf in workflows
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    app_state = Depends(get_app_state)
):
    """
    Get specific workflow
    Task 12.1-T17: Build workflow retrieval APIs
    """
    try:
        logger.info(f"🔍 Getting workflow: {workflow_id}")
        
        builder = RBABuilder()
        workflow = await builder.get_workflow(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowResponse(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            industry=workflow.industry,
            automation_type=workflow.automation_type,
            version=workflow.version,
            state=workflow.state.value,
            created_at=workflow.created_at,
            created_by=workflow.created_by,
            tenant_id=workflow.tenant_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow: {str(e)}")

@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    update_request: WorkflowUpdateRequest,
    app_state = Depends(get_app_state)
):
    """
    Update workflow
    Task 12.1-T18: Build workflow update APIs
    """
    try:
        logger.info(f"📝 Updating workflow: {workflow_id}")
        
        builder = RBABuilder()
        workflow = await builder.update_workflow(workflow_id, update_request.dict(exclude_unset=True))
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowResponse(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            industry=workflow.industry,
            automation_type=workflow.automation_type,
            version=workflow.version,
            state=workflow.state.value,
            created_at=workflow.created_at,
            created_by=workflow.created_by,
            tenant_id=workflow.tenant_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

@router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    app_state = Depends(get_app_state)
):
    """
    Delete workflow
    Task 12.1-T19: Build workflow deletion APIs
    """
    try:
        logger.info(f"🗑️ Deleting workflow: {workflow_id}")
        
        builder = RBABuilder()
        success = await builder.delete_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {"message": "Workflow deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.post("/workflows/validate", response_model=ValidationResponse)
async def validate_workflow(
    validation_request: WorkflowValidationRequest,
    app_state = Depends(get_app_state)
):
    """
    Validate workflow definition
    Task 12.1-T35: Build DSL validation APIs
    """
    try:
        logger.info("🔍 Validating workflow definition")
        
        # Compile workflow
        compiler = DSLCompiler()
        ast = compiler.compile_workflow(
            validation_request.workflow_definition, 
            validation_request.format
        )
        
        # Validate workflow
        validator = WorkflowValidator()
        validation_result = validator.validate_workflow(ast)
        
        # Separate errors and warnings
        errors = [
            {
                "level": issue.level.value,
                "message": issue.message,
                "step_id": issue.step_id,
                "field": issue.field,
                "suggestion": issue.suggestion
            }
            for issue in validation_result.issues 
            if issue.level.value == "error"
        ]
        
        warnings = [
            {
                "level": issue.level.value,
                "message": issue.message,
                "step_id": issue.step_id,
                "field": issue.field,
                "suggestion": issue.suggestion
            }
            for issue in validation_result.issues 
            if issue.level.value == "warning"
        ]
        
        return ValidationResponse(
            is_valid=validation_result.is_valid,
            errors=errors,
            warnings=warnings,
            plan_hash=ast.plan_hash
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to validate workflow: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

@router.get("/workflows/samples/pipeline-hygiene")
async def get_pipeline_hygiene_sample():
    """
    Get sample Pipeline Hygiene workflow
    Task 19.3-T03: SaaS template - Pipeline hygiene + quota compliance
    """
    return {
        "name": "SaaS Pipeline Hygiene Check",
        "description": "Automated pipeline health monitoring with quota compliance checking",
        "workflow_definition": SAMPLE_PIPELINE_HYGIENE_WORKFLOW,
        "industry": "SaaS",
        "automation_type": "RBA",
        "template_id": "pipeline_hygiene"
    }

@router.get("/workflows/samples/forecast-approval")
async def get_forecast_approval_sample():
    """
    Get sample Forecast Approval workflow
    Task 19.3-T04: SaaS template - Forecast approval governance
    """
    return {
        "name": "SaaS Forecast Approval Governance",
        "description": "Automated forecast variance analysis with governance-driven approval workflows",
        "workflow_definition": SAMPLE_FORECAST_APPROVAL_WORKFLOW,
        "industry": "SaaS",
        "automation_type": "RBA",
        "template_id": "forecast_approval"
    }

@router.post("/workflows/{workflow_id}/publish")
async def publish_workflow(
    workflow_id: str,
    app_state = Depends(get_app_state)
):
    """
    Publish workflow to production
    Task 12.1-T26: Implement approval → publish state
    """
    try:
        logger.info(f"🚀 Publishing workflow: {workflow_id}")
        
        builder = RBABuilder()
        workflow = await builder.publish_workflow(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "message": "Workflow published successfully",
            "workflow_id": workflow_id,
            "state": workflow.state.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to publish workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to publish workflow: {str(e)}")
