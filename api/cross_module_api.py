#!/usr/bin/env python3
"""
Cross-Module Orchestration API
==============================
REST API endpoints for multi-module workflow chaining

Endpoints:
- POST /cross-module/create-workflow - Create cross-module workflow
- POST /cross-module/execute/{workflow_id} - Execute cross-module workflow
- GET /cross-module/status/{workflow_id} - Get workflow status
- GET /cross-module/capabilities - List module capabilities
- POST /cross-module/register-capability - Register new module capability
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

# Cross-Module Components
from dsl.orchestration.cross_module_orchestrator import (
    CrossModuleOrchestrator,
    ModuleType,
    ModuleCapability,
    CrossModuleWorkflow,
    CrossModuleExecutionResult,
    get_cross_module_orchestrator
)
from src.services.connection_pool_manager import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cross-module", tags=["Cross-Module Orchestration"])

# Request/Response Models
class CreateWorkflowRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    modules: List[str] = Field(..., description="List of module names")
    workflow_definition: Dict[str, Any] = Field(..., description="Workflow definition")
    tenant_id: int = Field(..., description="Tenant ID")
    user_id: int = Field(..., description="User ID")

class RegisterCapabilityRequest(BaseModel):
    module: str = Field(..., description="Module name")
    capability_id: str = Field(..., description="Capability ID")
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    input_schema: Dict[str, Any] = Field(..., description="Input schema")
    output_schema: Dict[str, Any] = Field(..., description="Output schema")
    dependencies: List[str] = Field(default=[], description="Dependencies")
    sla_ms: int = Field(default=5000, description="SLA in milliseconds")

# Chapter 19.2 Models
class ModuleExpansionKitRequest(BaseModel):
    modules: List[str] = Field(..., description="List of module names to include in expansion kit")
    tenant_id: int = Field(..., description="Tenant ID")
    industry: str = Field(default="SaaS", description="Industry type")

class ExecuteExpansionRequest(BaseModel):
    kit_id: str = Field(..., description="Expansion kit ID")
    user_id: int = Field(..., description="User ID")
    auto_start: bool = Field(default=True, description="Auto-start expansion")

class WorkflowResponse(BaseModel):
    success: bool
    workflow_id: Optional[str] = None
    message: str = ""
    processing_time_ms: int = 0

class ExecutionResponse(BaseModel):
    success: bool
    workflow_id: str
    status: str
    module_results: List[Dict[str, Any]] = []
    total_execution_time_ms: int = 0
    evidence_pack_ids: List[str] = []
    rollback_performed: bool = False
    error_message: Optional[str] = None

class StatusResponse(BaseModel):
    workflow_id: str
    name: str
    modules: List[str]
    status: str
    created_at: Optional[str] = None

class CapabilitiesResponse(BaseModel):
    modules: Dict[str, List[Dict[str, Any]]]
    total_capabilities: int

@router.post("/create-workflow", response_model=WorkflowResponse)
async def create_cross_module_workflow(
    request: CreateWorkflowRequest,
    pool_manager=Depends(get_pool_manager)
):
    """Create a new cross-module workflow"""
    try:
        start_time = datetime.now()
        
        logger.info(f"🔗 Creating cross-module workflow '{request.name}' for tenant {request.tenant_id}")
        
        # Get cross-module orchestrator
        orchestrator = await get_cross_module_orchestrator(pool_manager)
        
        # Convert module names to ModuleType enums
        try:
            modules = [ModuleType(module.lower()) for module in request.modules]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid module name: {str(e)}")
        
        # Create workflow
        workflow = await orchestrator.create_cross_module_workflow(
            modules=modules,
            workflow_definition=request.workflow_definition,
            tenant_id=request.tenant_id,
            user_id=request.user_id
        )
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        logger.info(f"✅ Cross-module workflow created: {workflow.workflow_id} in {processing_time}ms")
        
        return WorkflowResponse(
            success=True,
            workflow_id=workflow.workflow_id,
            message=f"Workflow '{request.name}' created successfully",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to create cross-module workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow creation failed: {str(e)}")

@router.post("/execute/{workflow_id}", response_model=ExecutionResponse)
async def execute_cross_module_workflow(
    workflow_id: str,
    pool_manager=Depends(get_pool_manager)
):
    """Execute a cross-module workflow"""
    try:
        logger.info(f"🚀 Executing cross-module workflow: {workflow_id}")
        
        # Get cross-module orchestrator
        orchestrator = await get_cross_module_orchestrator(pool_manager)
        
        # Execute workflow
        result = await orchestrator.execute_cross_module_workflow(workflow_id)
        
        logger.info(f"✅ Cross-module workflow execution completed: {result.status.value}")
        
        return ExecutionResponse(
            success=result.status.value == "completed",
            workflow_id=result.workflow_id,
            status=result.status.value,
            module_results=[
                {
                    "module": r.module.value,
                    "capability_id": r.capability_id,
                    "status": r.status.value,
                    "data": r.data,
                    "error": r.error,
                    "execution_time_ms": r.execution_time_ms,
                    "evidence_pack_id": r.evidence_pack_id
                }
                for r in result.module_results
            ],
            total_execution_time_ms=result.total_execution_time_ms,
            evidence_pack_ids=result.evidence_pack_ids,
            rollback_performed=result.rollback_performed,
            error_message=result.error_message
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to execute cross-module workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@router.get("/status/{workflow_id}", response_model=StatusResponse)
async def get_workflow_status(
    workflow_id: str,
    pool_manager=Depends(get_pool_manager)
):
    """Get status of a cross-module workflow"""
    try:
        logger.info(f"📊 Getting status for workflow: {workflow_id}")
        
        # Get cross-module orchestrator
        orchestrator = await get_cross_module_orchestrator(pool_manager)
        
        # Get workflow status
        status = await orchestrator.get_workflow_status(workflow_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return StatusResponse(
            workflow_id=status["workflow_id"],
            name=status["name"],
            modules=status["modules"],
            status=status["status"],
            created_at=status.get("created_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/capabilities", response_model=CapabilitiesResponse)
async def list_module_capabilities(
    pool_manager=Depends(get_pool_manager)
):
    """List all registered module capabilities"""
    try:
        logger.info("📋 Listing module capabilities")
        
        # Get cross-module orchestrator
        orchestrator = await get_cross_module_orchestrator(pool_manager)
        
        # Get capabilities
        capabilities = await orchestrator.list_module_capabilities()
        
        total_capabilities = sum(len(caps) for caps in capabilities.values())
        
        logger.info(f"✅ Retrieved {total_capabilities} capabilities across {len(capabilities)} modules")
        
        return CapabilitiesResponse(
            modules=capabilities,
            total_capabilities=total_capabilities
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to list capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Capabilities listing failed: {str(e)}")

@router.post("/register-capability")
async def register_module_capability(
    request: RegisterCapabilityRequest,
    pool_manager=Depends(get_pool_manager)
):
    """Register a new module capability"""
    try:
        logger.info(f"📝 Registering capability '{request.name}' for module {request.module}")
        
        # Get cross-module orchestrator
        orchestrator = await get_cross_module_orchestrator(pool_manager)
        
        # Convert module name to ModuleType enum
        try:
            module_type = ModuleType(request.module.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid module name: {request.module}")
        
        # Create capability
        capability = ModuleCapability(
            module=module_type,
            capability_id=request.capability_id,
            name=request.name,
            description=request.description,
            input_schema=request.input_schema,
            output_schema=request.output_schema,
            dependencies=request.dependencies,
            sla_ms=request.sla_ms
        )
        
        # Register capability
        await orchestrator.register_module_capability(capability)
        
        logger.info(f"✅ Capability '{request.name}' registered successfully")
        
        return {
            "success": True,
            "message": f"Capability '{request.name}' registered for module {request.module}",
            "capability_id": request.capability_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to register capability: {e}")
        raise HTTPException(status_code=500, detail=f"Capability registration failed: {str(e)}")

@router.get("/modules")
async def list_supported_modules():
    """List all supported modules"""
    try:
        modules = [
            {
                "name": module.value,
                "display_name": module.value.replace("_", " ").title(),
                "description": _get_module_description(module)
            }
            for module in ModuleType
        ]
        
        return {
            "modules": modules,
            "total_modules": len(modules)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list modules: {e}")
        raise HTTPException(status_code=500, detail=f"Module listing failed: {str(e)}")

@router.post("/simulate-workflow")
async def simulate_cross_module_workflow(
    request: CreateWorkflowRequest,
    pool_manager=Depends(get_pool_manager)
):
    """Simulate a cross-module workflow without execution"""
    try:
        logger.info(f"🧪 Simulating cross-module workflow '{request.name}'")
        
        # Get cross-module orchestrator
        orchestrator = await get_cross_module_orchestrator(pool_manager)
        
        # Convert module names to ModuleType enums
        try:
            modules = [ModuleType(module.lower()) for module in request.modules]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid module name: {str(e)}")
        
        # Build execution graph
        execution_graph = await orchestrator._build_execution_graph(modules, request.workflow_definition)
        
        # Resolve execution order
        execution_order = orchestrator._resolve_execution_order(execution_graph)
        
        simulation_result = {
            "workflow_name": request.name,
            "modules": [m.value for m in modules],
            "execution_graph": execution_graph,
            "execution_order": [
                {
                    "step": i + 1,
                    "module": step["module"].value,
                    "capability_id": step["capability_id"],
                    "dependencies": step["dependencies"]
                }
                for i, step in enumerate(execution_order)
            ],
            "estimated_execution_time_ms": len(execution_order) * 1000,  # Rough estimate
            "validation": "valid"
        }
        
        logger.info(f"✅ Workflow simulation completed with {len(execution_order)} steps")
        
        return simulation_result
        
    except Exception as e:
        logger.error(f"❌ Workflow simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

def _get_module_description(module: ModuleType) -> str:
    """Get description for a module"""
    descriptions = {
        ModuleType.PIPELINE: "Sales pipeline management and hygiene",
        ModuleType.FORECASTING: "Revenue forecasting and prediction",
        ModuleType.COMPENSATION: "Sales compensation and commission management",
        ModuleType.BILLING: "Billing and invoice management",
        ModuleType.CLM: "Contract lifecycle management",
        ModuleType.CRUXX: "Customer relationship and experience management",
        ModuleType.REVENUE_DASHBOARD: "Revenue analytics and dashboards",
        ModuleType.PLANNING: "Strategic account planning"
    }
    
    return descriptions.get(module, "Module description not available")

# =====================================================
# CHAPTER 19.2 - MODULE EXPANSION ENDPOINTS
# =====================================================

@router.post("/expansion/create-kit")
async def create_module_expansion_kit(request: ModuleExpansionKitRequest):
    """
    Create multi-module expansion kit (Task 19.2-T15)
    """
    try:
        orchestrator = await get_cross_module_orchestrator()
        
        # Convert module names to ModuleType enums
        modules = [ModuleType(module) for module in request.modules]
        
        expansion_kit = await orchestrator.create_module_expansion_kit(
            modules=modules,
            tenant_id=request.tenant_id,
            industry=request.industry
        )
        
        return {
            "success": True,
            "expansion_kit": expansion_kit,
            "message": f"Created expansion kit for {len(modules)} modules"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create expansion kit: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create expansion kit: {str(e)}")

@router.post("/expansion/execute")
async def execute_module_expansion(request: ExecuteExpansionRequest):
    """
    Execute multi-module expansion workflow (Task 19.2-T10)
    """
    try:
        orchestrator = await get_cross_module_orchestrator()
        
        expansion_config = {
            "user_id": request.user_id,
            "auto_start": request.auto_start
        }
        
        result = await orchestrator.execute_module_expansion_workflow(
            kit_id=request.kit_id,
            expansion_config=expansion_config
        )
        
        return {
            "success": True,
            "expansion_result": result,
            "message": "Module expansion executed successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to execute expansion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute expansion: {str(e)}")

@router.get("/expansion/heatmap/{tenant_id}")
async def get_expansion_adoption_heatmap(tenant_id: int):
    """
    Get expansion adoption heatmap (Task 19.2-T36)
    """
    try:
        orchestrator = await get_cross_module_orchestrator()
        
        heatmap_data = await orchestrator.get_expansion_adoption_heatmap(tenant_id)
        
        return {
            "success": True,
            "heatmap_data": heatmap_data,
            "message": "Expansion heatmap generated successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get expansion heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get heatmap: {str(e)}")

@router.get("/expansion/roi/{expansion_id}")
async def calculate_expansion_roi(expansion_id: str):
    """
    Calculate ROI for multi-module expansion (Task 19.2-T37)
    """
    try:
        orchestrator = await get_cross_module_orchestrator()
        
        roi_calculation = await orchestrator.calculate_expansion_roi(expansion_id)
        
        return {
            "success": True,
            "roi_calculation": roi_calculation,
            "message": "Expansion ROI calculated successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to calculate expansion ROI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate ROI: {str(e)}")
