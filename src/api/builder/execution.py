"""
Builder API - Workflow Execution Endpoints
==========================================

Implements Chapter 6.2 Runtime APIs:
- Workflow execution with DSL Runtime Engine
- Test execution with sample data
- Execution status and results
- Integration with Routing Orchestrator

Provides execution capabilities for frontend testing.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from dsl.compiler import DSLCompiler, WorkflowValidator, WorkflowRuntime
from dsl.compiler.parser import DSLWorkflowAST
from dsl.compiler.runtime import WorkflowExecution, WorkflowStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for API
class ExecutionRequest(BaseModel):
    workflow_definition: str = Field(..., description="YAML workflow definition")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for execution")
    tenant_id: str = Field(..., description="Tenant ID")
    user_id: str = Field(..., description="User ID")
    format: str = Field(default="yaml", description="Format: yaml or json")

class ExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    final_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    evidence_pack_id: Optional[str] = None

class ExecutionStatusResponse(BaseModel):
    execution_id: str
    status: str
    progress: Dict[str, Any]
    current_step: Optional[str] = None
    completed_steps: List[str]
    failed_steps: List[str]

class TestExecutionRequest(BaseModel):
    workflow_definition: str = Field(..., description="YAML workflow definition")
    test_scenario: str = Field(default="default", description="Test scenario name")
    format: str = Field(default="yaml", description="Format: yaml or json")

# Dependency to get app state
async def get_app_state(request: Request):
    return request.app.state

@router.post("/execute", response_model=ExecutionResponse)
async def execute_workflow(
    execution_request: ExecutionRequest,
    background_tasks: BackgroundTasks,
    app_state = Depends(get_app_state)
):
    """
    Execute workflow with DSL Runtime Engine
    Task 6.2-T05: Runtime step executor (deterministic execution)
    """
    try:
        logger.info(f"🚀 Starting workflow execution for tenant: {execution_request.tenant_id}")
        
        # Compile workflow
        compiler = DSLCompiler()
        ast = compiler.compile_workflow(
            execution_request.workflow_definition,
            execution_request.format
        )
        
        # Validate workflow
        validator = WorkflowValidator()
        validation_result = validator.validate_workflow(ast)
        
        if not validation_result.is_valid:
            error_messages = [issue.message for issue in validation_result.issues if issue.level.value == "error"]
            raise HTTPException(
                status_code=400, 
                detail=f"Workflow validation failed: {'; '.join(error_messages)}"
            )
        
        # Initialize runtime
        runtime = WorkflowRuntime(app_state.pool_manager)
        
        # Execute workflow
        execution = await runtime.execute_workflow(
            ast=ast,
            input_data=execution_request.input_data,
            tenant_id=execution_request.tenant_id,
            user_id=execution_request.user_id
        )
        
        return ExecutionResponse(
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            status=execution.status.value,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            final_result=execution.final_result,
            error=execution.error,
            evidence_pack_id=execution.evidence_pack_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to execute workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution_status(
    execution_id: str,
    app_state = Depends(get_app_state)
):
    """
    Get workflow execution status
    Task 6.2-T30: Build execution status APIs
    """
    try:
        logger.info(f"🔍 Getting execution status: {execution_id}")
        
        # Initialize runtime to access execution records
        runtime = WorkflowRuntime(app_state.pool_manager)
        
        # Check active executions first
        if execution_id in runtime.active_executions:
            execution = runtime.active_executions[execution_id]
        else:
            # In production, this would query the database
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return ExecutionResponse(
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            status=execution.status.value,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            final_result=execution.final_result,
            error=execution.error,
            evidence_pack_id=execution.evidence_pack_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get execution status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get execution status: {str(e)}")

@router.post("/test", response_model=ExecutionResponse)
async def test_workflow_execution(
    test_request: TestExecutionRequest,
    app_state = Depends(get_app_state)
):
    """
    Test workflow execution with sample data
    Task 12.1-T36: Build workflow testing APIs
    """
    try:
        logger.info(f"🧪 Testing workflow execution - scenario: {test_request.test_scenario}")
        
        # Get test data based on scenario
        test_data = _get_test_data(test_request.test_scenario)
        
        # Create execution request
        execution_request = ExecutionRequest(
            workflow_definition=test_request.workflow_definition,
            input_data=test_data,
            tenant_id="1300",  # Default test tenant
            user_id="1319",    # Default test user
            format=test_request.format
        )
        
        # Execute workflow (reuse the execute_workflow logic)
        return await execute_workflow(execution_request, BackgroundTasks(), app_state)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to test workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")

@router.get("/test-scenarios")
async def list_test_scenarios():
    """
    List available test scenarios with sample data
    """
    return {
        "scenarios": {
            "pipeline_hygiene": {
                "name": "Pipeline Hygiene Test",
                "description": "Test pipeline hygiene workflow with sample opportunities",
                "sample_data": {
                    "opportunities": [
                        {
                            "Id": "006xx000004TmiQAAS",
                            "Name": "Acme Corp - Q4 Deal",
                            "Amount": 50000,
                            "StageName": "Negotiation/Review",
                            "CloseDate": "2024-12-31",
                            "LastActivityDate": "2024-11-15T10:30:00Z",
                            "Probability": 75
                        },
                        {
                            "Id": "006xx000004TmiRAAS", 
                            "Name": "TechStart - Annual License",
                            "Amount": 25000,
                            "StageName": "Proposal/Price Quote",
                            "CloseDate": "2024-11-30",
                            "LastActivityDate": "2024-10-20T14:15:00Z",
                            "Probability": 60
                        }
                    ],
                    "quota_targets": {
                        "target_quota": 100000,
                        "current_quarter": "Q4 2024"
                    }
                }
            },
            "forecast_approval": {
                "name": "Forecast Approval Test",
                "description": "Test forecast approval workflow with variance scenarios",
                "sample_data": {
                    "forecast_data": {
                        "previous_forecast": 100000,
                        "current_forecast": 115000,
                        "variance_amount": 15000,
                        "variance_percentage": 0.15,
                        "quarter": "Q4 2024"
                    },
                    "approval_hierarchy": {
                        "manager": "sales_manager_001",
                        "director": "sales_director_001"
                    }
                }
            },
            "default": {
                "name": "Default Test",
                "description": "Basic test scenario with minimal data",
                "sample_data": {
                    "test": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        }
    }

@router.post("/validate-and-execute", response_model=ExecutionResponse)
async def validate_and_execute_workflow(
    execution_request: ExecutionRequest,
    app_state = Depends(get_app_state)
):
    """
    Validate and execute workflow in one step
    Combines validation + execution for frontend convenience
    """
    try:
        logger.info("🔍 Validating and executing workflow")
        
        # Compile workflow
        compiler = DSLCompiler()
        ast = compiler.compile_workflow(
            execution_request.workflow_definition,
            execution_request.format
        )
        
        # Validate workflow
        validator = WorkflowValidator()
        validation_result = validator.validate_workflow(ast)
        
        if not validation_result.is_valid:
            # Return validation errors
            error_messages = [
                f"{issue.message} (Step: {issue.step_id or 'N/A'})"
                for issue in validation_result.issues 
                if issue.level.value == "error"
            ]
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "validation_error",
                    "message": "Workflow validation failed",
                    "errors": error_messages,
                    "warnings": [
                        f"{issue.message} (Step: {issue.step_id or 'N/A'})"
                        for issue in validation_result.issues 
                        if issue.level.value == "warning"
                    ]
                }
            )
        
        # If validation passes, execute workflow
        runtime = WorkflowRuntime(app_state.pool_manager)
        execution = await runtime.execute_workflow(
            ast=ast,
            input_data=execution_request.input_data,
            tenant_id=execution_request.tenant_id,
            user_id=execution_request.user_id
        )
        
        return ExecutionResponse(
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            status=execution.status.value,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            final_result=execution.final_result,
            error=execution.error,
            evidence_pack_id=execution.evidence_pack_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to validate and execute workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation and execution failed: {str(e)}")

def _get_test_data(scenario: str) -> Dict[str, Any]:
    """Get test data for different scenarios"""
    
    test_scenarios = {
        "pipeline_hygiene": {
            "opportunities": [
                {
                    "Id": "006xx000004TmiQAAS",
                    "Name": "Acme Corp - Q4 Deal",
                    "Amount": 50000,
                    "StageName": "Negotiation/Review",
                    "CloseDate": "2024-12-31",
                    "LastActivityDate": "2024-11-15T10:30:00+00:00",  # Recent activity
                    "Probability": 75
                },
                {
                    "Id": "006xx000004TmiRAAS",
                    "Name": "TechStart - Annual License", 
                    "Amount": 25000,
                    "StageName": "Proposal/Price Quote",
                    "CloseDate": "2024-11-30",
                    "LastActivityDate": "2024-10-20T14:15:00+00:00",  # Stale (>15 days)
                    "Probability": 60
                },
                {
                    "Id": "006xx000004TmiSAAS",
                    "Name": "StartupXYZ - Platform Deal",
                    "Amount": None,  # Missing required field
                    "StageName": "Qualification",
                    "CloseDate": "2025-01-15",
                    "LastActivityDate": "2024-09-30T09:00:00+00:00",  # Very stale
                    "Probability": 30
                }
            ],
            "quota_targets": {
                "target_quota": 100000,
                "current_quarter": "Q4 2024"
            }
        },
        "forecast_approval": {
            "forecast_data": {
                "previous_forecast": 100000,
                "current_forecast": 115000,
                "variance_amount": 15000,
                "variance_percentage": 0.15,  # 15% variance - MEDIUM level
                "quarter": "Q4 2024"
            },
            "variance_thresholds": {
                "low": 0.05,
                "medium": 0.15,
                "high": 0.25
            },
            "approval_hierarchy": {
                "manager": "sales_manager_001",
                "director": "sales_director_001"
            }
        },
        "default": {
            "test": True,
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": "1300",
            "user_id": "1319"
        }
    }
    
    return test_scenarios.get(scenario, test_scenarios["default"])
