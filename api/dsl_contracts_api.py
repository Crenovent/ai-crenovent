"""
DSL Contracts API
Task 9.3.4: Implement workflow DSL contracts API

REST API endpoints for workflow DSL contract management and validation
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from dsl.contracts.workflow_dsl_contracts import (
    WorkflowDSLContractManager, ContractType, ContractStatus, 
    ValidationLevel, GovernanceFields
)
from src.services.connection_pool_manager import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API requests/responses

class GovernanceFieldsRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant identifier")
    region_id: str = Field(..., description="Region identifier")
    policy_pack_id: Optional[str] = Field(None, description="Policy pack ID")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")
    sla_tier: str = Field("T2", description="SLA tier (T0, T1, T2)")
    data_classification: str = Field("internal", description="Data classification level")
    retention_days: int = Field(2555, description="Data retention period in days")

class WorkflowStepRequest(BaseModel):
    step_id: str = Field(..., description="Unique step identifier")
    step_name: str = Field(..., description="Human-readable step name")
    step_type: str = Field(..., description="Step type (query, decision, ml_decision, agent_call, notify, governance)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    inputs: List[str] = Field(default_factory=list, description="Input dependencies")
    outputs: List[str] = Field(default_factory=list, description="Output definitions")
    timeout_ms: int = Field(30000, description="Step timeout in milliseconds")
    retry_count: int = Field(3, description="Number of retries")
    fallback_step: Optional[str] = Field(None, description="Fallback step ID")
    governance_required: bool = Field(True, description="Governance validation required")
    evidence_capture: bool = Field(True, description="Evidence capture enabled")
    policy_checks: List[str] = Field(default_factory=list, description="Policy checks to perform")

class CreateContractRequest(BaseModel):
    workflow_name: str = Field(..., description="Workflow name")
    workflow_description: str = Field(..., description="Workflow description")
    steps: List[WorkflowStepRequest] = Field(..., description="Workflow steps")
    governance: GovernanceFieldsRequest = Field(..., description="Governance metadata")
    created_by: str = Field(..., description="Contract creator")
    contract_type: ContractType = Field(ContractType.WORKFLOW, description="Contract type")
    validation_level: ValidationLevel = Field(ValidationLevel.STANDARD, description="Validation level")
    
    # Optional workflow configuration
    execution_mode: str = Field("sequential", description="Execution mode")
    max_execution_time_ms: int = Field(300000, description="Maximum execution time")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output schema")
    error_handling: Dict[str, Any] = Field(default_factory=dict, description="Error handling configuration")
    rollback_strategy: str = Field("none", description="Rollback strategy")

class ValidateInputRequest(BaseModel):
    contract_id: str = Field(..., description="Contract ID")
    input_data: Dict[str, Any] = Field(..., description="Input data to validate")

class ContractResponse(BaseModel):
    contract_id: str
    contract_name: str
    contract_type: ContractType
    version: str
    status: ContractStatus
    created_by: str
    created_at: datetime
    updated_at: datetime
    description: str

class ValidationResponse(BaseModel):
    valid: bool
    errors: List[str]
    warnings: List[str] = Field(default_factory=list)
    step_count: Optional[int] = None
    governance_compliant: Optional[bool] = None

class ContractDetailsResponse(BaseModel):
    contract_id: str
    metadata: Dict[str, Any]
    governance: Dict[str, Any]
    workflow_definition: Dict[str, Any]
    validation_result: ValidationResponse

# Global contract manager instance
contract_manager = None

async def get_contract_manager(pool_manager=Depends(get_pool_manager)) -> WorkflowDSLContractManager:
    """Get or create contract manager instance"""
    global contract_manager
    if contract_manager is None:
        contract_manager = WorkflowDSLContractManager(pool_manager)
        await contract_manager.initialize()
    return contract_manager

@router.post("/create", response_model=Dict[str, str])
async def create_workflow_contract(
    request: CreateContractRequest,
    manager: WorkflowDSLContractManager = Depends(get_contract_manager)
):
    """
    Create a new workflow DSL contract
    Task 9.3.4: Declarative workflow definition
    """
    try:
        # Convert request to internal format
        steps_data = []
        for step in request.steps:
            step_data = {
                "step_id": step.step_id,
                "step_name": step.step_name,
                "step_type": step.step_type,
                "parameters": step.parameters,
                "inputs": step.inputs,
                "outputs": step.outputs,
                "timeout_ms": step.timeout_ms,
                "retry_count": step.retry_count,
                "fallback_step": step.fallback_step,
                "governance_required": step.governance_required,
                "evidence_capture": step.evidence_capture,
                "policy_checks": step.policy_checks
            }
            steps_data.append(step_data)
        
        governance_data = {
            "tenant_id": request.governance.tenant_id,
            "region_id": request.governance.region_id,
            "policy_pack_id": request.governance.policy_pack_id,
            "compliance_frameworks": request.governance.compliance_frameworks,
            "sla_tier": request.governance.sla_tier,
            "data_classification": request.governance.data_classification,
            "retention_days": request.governance.retention_days
        }
        
        contract_id = await manager.create_contract(
            workflow_name=request.workflow_name,
            workflow_description=request.workflow_description,
            steps=steps_data,
            governance=governance_data,
            created_by=request.created_by,
            contract_type=request.contract_type,
            validation_level=request.validation_level
        )
        
        return {
            "contract_id": contract_id,
            "message": f"Contract '{request.workflow_name}' created successfully",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Contract creation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Contract creation failed: {str(e)}")

@router.get("/contract/{contract_id}", response_model=ContractDetailsResponse)
async def get_workflow_contract(
    contract_id: str,
    manager: WorkflowDSLContractManager = Depends(get_contract_manager)
):
    """
    Get workflow contract by ID
    Task 9.3.4: Contract retrieval
    """
    try:
        contract = await manager.get_contract(contract_id)
        
        if not contract:
            raise HTTPException(status_code=404, detail=f"Contract not found: {contract_id}")
        
        # Validate contract
        validation_result = contract.validate()
        
        return ContractDetailsResponse(
            contract_id=contract.metadata.contract_id,
            metadata=contract.metadata.to_dict(),
            governance=contract.governance.__dict__,
            workflow_definition=contract.to_dict(),
            validation_result=ValidationResponse(
                valid=validation_result["valid"],
                errors=validation_result["errors"],
                warnings=validation_result.get("warnings", []),
                step_count=validation_result.get("step_count"),
                governance_compliant=validation_result.get("governance_compliant")
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get contract {contract_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve contract: {str(e)}")

@router.get("/contracts", response_model=List[ContractResponse])
async def list_workflow_contracts(
    tenant_id: Optional[int] = Query(None, description="Filter by tenant ID"),
    contract_type: Optional[ContractType] = Query(None, description="Filter by contract type"),
    status: Optional[ContractStatus] = Query(None, description="Filter by status"),
    manager: WorkflowDSLContractManager = Depends(get_contract_manager)
):
    """
    List workflow contracts with optional filters
    Task 9.3.4: Contract discovery
    """
    try:
        contracts = await manager.list_contracts(
            tenant_id=tenant_id,
            contract_type=contract_type,
            status=status
        )
        
        return [
            ContractResponse(
                contract_id=contract.contract_id,
                contract_name=contract.contract_name,
                contract_type=contract.contract_type,
                version=contract.version,
                status=contract.status,
                created_by=contract.created_by,
                created_at=contract.created_at,
                updated_at=contract.updated_at,
                description=contract.description
            )
            for contract in contracts
        ]
        
    except Exception as e:
        logger.error(f"Failed to list contracts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list contracts: {str(e)}")

@router.post("/validate/{contract_id}", response_model=ValidationResponse)
async def validate_workflow_contract(
    contract_id: str,
    manager: WorkflowDSLContractManager = Depends(get_contract_manager)
):
    """
    Validate a workflow contract
    Task 9.3.4: Contract validation
    """
    try:
        validation_result = await manager.validate_contract(contract_id)
        
        return ValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result["errors"],
            warnings=validation_result.get("warnings", []),
            step_count=validation_result.get("step_count"),
            governance_compliant=validation_result.get("governance_compliant")
        )
        
    except Exception as e:
        logger.error(f"Contract validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Contract validation failed: {str(e)}")

@router.post("/validate-input", response_model=ValidationResponse)
async def validate_contract_input(
    request: ValidateInputRequest,
    manager: WorkflowDSLContractManager = Depends(get_contract_manager)
):
    """
    Validate input data against contract's input schema
    Task 9.3.4: Input validation
    """
    try:
        validation_result = await manager.execute_contract_validation(
            contract_id=request.contract_id,
            input_data=request.input_data
        )
        
        return ValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result["errors"],
            warnings=validation_result.get("warnings", [])
        )
        
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Input validation failed: {str(e)}")

@router.get("/templates/{industry}", response_model=Dict[str, Any])
async def get_industry_contract_templates(
    industry: str,
    manager: WorkflowDSLContractManager = Depends(get_contract_manager)
):
    """
    Get industry-specific contract templates
    Task 9.3.4: Industry template management
    """
    try:
        templates = await manager.get_industry_templates(industry)
        
        return {
            "industry": templates["industry"],
            "templates": templates["templates"],
            "count": templates["count"],
            "available_industries": ["SaaS", "Banking", "Insurance"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get industry templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.get("/step-types", response_model=List[str])
async def get_available_step_types():
    """
    Get available DSL step types
    Task 9.3.4: Step type discovery
    """
    return [
        "query",
        "decision", 
        "ml_decision",
        "agent_call",
        "notify",
        "governance"
    ]

@router.get("/validation-levels", response_model=List[str])
async def get_validation_levels():
    """
    Get available validation levels
    Task 9.3.4: Validation level discovery
    """
    return [level.value for level in ValidationLevel]

@router.get("/execution-modes", response_model=List[str])
async def get_execution_modes():
    """
    Get available execution modes
    Task 9.3.4: Execution mode discovery
    """
    return ["sequential", "parallel", "conditional"]

@router.post("/create-from-template", response_model=Dict[str, str])
async def create_contract_from_template(
    industry: str,
    template_name: str,
    workflow_name: str,
    governance: GovernanceFieldsRequest,
    created_by: str,
    manager: WorkflowDSLContractManager = Depends(get_contract_manager)
):
    """
    Create a contract from an industry template
    Task 9.3.4: Template-based contract creation
    """
    try:
        # Get industry templates
        templates = await manager.get_industry_templates(industry)
        
        if template_name not in templates["templates"]:
            raise HTTPException(
                status_code=404, 
                detail=f"Template '{template_name}' not found for industry '{industry}'"
            )
        
        template = templates["templates"][template_name]
        
        # Convert template to contract format
        steps_data = []
        for i, step in enumerate(template["steps"]):
            step_data = {
                "step_id": step["step_id"],
                "step_name": step.get("step_name", f"Step {i+1}"),
                "step_type": step["step_type"],
                "parameters": step.get("parameters", {}),
                "inputs": step.get("inputs", []),
                "outputs": step.get("outputs", []),
                "timeout_ms": step.get("timeout_ms", 30000),
                "retry_count": step.get("retry_count", 3),
                "governance_required": step.get("governance_required", True),
                "evidence_capture": step.get("evidence_capture", True),
                "policy_checks": step.get("policy_checks", [])
            }
            steps_data.append(step_data)
        
        governance_data = {
            "tenant_id": governance.tenant_id,
            "region_id": governance.region_id,
            "policy_pack_id": governance.policy_pack_id,
            "compliance_frameworks": governance.compliance_frameworks,
            "sla_tier": governance.sla_tier,
            "data_classification": governance.data_classification,
            "retention_days": governance.retention_days
        }
        
        contract_id = await manager.create_contract(
            workflow_name=workflow_name,
            workflow_description=template.get("description", f"Contract created from {template_name} template"),
            steps=steps_data,
            governance=governance_data,
            created_by=created_by,
            contract_type=ContractType.WORKFLOW,
            validation_level=ValidationLevel.STANDARD
        )
        
        return {
            "contract_id": contract_id,
            "message": f"Contract '{workflow_name}' created from template '{template_name}'",
            "template_used": template_name,
            "industry": industry,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template-based contract creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template creation failed: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    manager: WorkflowDSLContractManager = Depends(get_contract_manager)
):
    """
    Health check for DSL contracts service
    """
    try:
        # Basic health checks
        contract_count = len(manager.contract_cache)
        
        return {
            "status": "healthy",
            "service": "dsl_contracts",
            "cached_contracts": contract_count,
            "available_industries": len(manager.industry_templates),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "dsl_contracts",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Export router
__all__ = ["router"]
