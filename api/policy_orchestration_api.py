"""
CHAPTER 16.1: POLICY ORCHESTRATION API
Tasks 16.1.9, 16.1.18-16.1.22: Policy Registry APIs and Management
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from src.database.connection_pool import get_pool_manager, ConnectionPoolManager
from src.auth.auth_utils import get_current_user, get_tenant_id_from_user
from dsl.governance.policy_registry_service import (
    PolicyRegistryService, PolicyDefinition, PolicyType, 
    ComplianceFramework, EnforcementMode
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/policy-orchestration", tags=["Policy Orchestration"])

# Pydantic models for API requests/responses
class CreatePolicyRequest(BaseModel):
    policy_name: str = Field(..., description="Policy name")
    policy_description: str = Field(..., description="Policy description")
    policy_type: str = Field(..., description="Policy type: compliance, business_rule, security, governance")
    industry_code: str = Field(default="SaaS", description="Industry code")
    compliance_framework: str = Field(..., description="Compliance framework")
    jurisdiction: str = Field(default="US", description="Jurisdiction")
    sla_tier: str = Field(default="standard", description="SLA tier")
    policy_content: Dict[str, Any] = Field(..., description="Policy rules and content")
    enforcement_mode: str = Field(default="enforcing", description="Enforcement mode")
    fail_closed: bool = Field(default=True, description="Fail-closed enforcement")
    override_allowed: bool = Field(default=True, description="Allow overrides")
    target_automation_types: List[str] = Field(default=["RBA"], description="Target automation types")

class PolicyEnforcementRequest(BaseModel):
    policy_id: str = Field(..., description="Policy ID to enforce")
    context: Dict[str, Any] = Field(..., description="Enforcement context data")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    automation_type: str = Field(default="RBA", description="Automation type")

class PolicyValidationRequest(BaseModel):
    policy_content: Dict[str, Any] = Field(..., description="Policy content to validate")

@router.post("/policies", response_model=Dict[str, Any])
async def create_policy(
    request: CreatePolicyRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.1.9: Create new policy
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        user_id = current_user['user_id']
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        # Create policy definition
        policy_def = PolicyDefinition(
            policy_id="",  # Will be generated
            policy_name=request.policy_name,
            policy_description=request.policy_description,
            policy_type=PolicyType(request.policy_type),
            industry_code=request.industry_code,
            compliance_framework=ComplianceFramework(request.compliance_framework),
            jurisdiction=request.jurisdiction,
            sla_tier=request.sla_tier,
            policy_content=request.policy_content,
            enforcement_mode=EnforcementMode(request.enforcement_mode),
            fail_closed=request.fail_closed,
            override_allowed=request.override_allowed,
            target_automation_types=request.target_automation_types,
            tenant_id=tenant_id,
            created_by_user_id=user_id
        )
        
        # Create policy using service
        policy_service = PolicyRegistryService(pool_manager)
        policy_id = await policy_service.create_policy(policy_def, tenant_id, user_id)
        
        return {
            "status": "success",
            "policy_id": policy_id,
            "message": "Policy created successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"❌ Invalid policy data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to create policy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create policy: {str(e)}")

@router.get("/policies", response_model=Dict[str, Any])
async def get_tenant_policies(
    industry_filter: Optional[str] = None,
    compliance_filter: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.1.10: Get tenant policies with filters
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        policy_service = PolicyRegistryService(pool_manager)
        policies = await policy_service.get_tenant_policies(
            tenant_id, industry_filter, compliance_filter
        )
        
        return {
            "status": "success",
            "policies": policies,
            "total_policies": len(policies),
            "filters": {
                "industry_filter": industry_filter,
                "compliance_filter": compliance_filter
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get policies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get policies: {str(e)}")

@router.post("/policies/enforce", response_model=Dict[str, Any])
async def enforce_policy(
    request: PolicyEnforcementRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.1.18: Policy enforcement with evidence logging
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        user_id = current_user['user_id']
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        # Add user context to enforcement context
        enforcement_context = {
            **request.context,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "workflow_id": request.workflow_id,
            "automation_type": request.automation_type,
            "enforcement_timestamp": datetime.now().isoformat()
        }
        
        policy_service = PolicyRegistryService(pool_manager)
        result = await policy_service.enforce_policy(
            tenant_id, request.policy_id, enforcement_context
        )
        
        return {
            "status": "success",
            "enforcement_result": result.enforcement_result,
            "policy_decision": result.policy_decision,
            "execution_time_ms": result.execution_time_ms,
            "evidence_pack_id": result.evidence_pack_id,
            "policy_id": result.policy_id,
            "version_id": result.version_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Policy enforcement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy enforcement failed: {str(e)}")

@router.get("/templates/saas", response_model=Dict[str, Any])
async def get_saas_policy_templates(
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.1.5: Get SaaS policy pack templates
    """
    try:
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        policy_service = PolicyRegistryService(pool_manager)
        templates = await policy_service.get_saas_policy_templates()
        
        return {
            "status": "success",
            "saas_policy_templates": templates,
            "template_count": len(templates.get("templates", {})),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get SaaS templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SaaS templates: {str(e)}")

@router.post("/policies/validate", response_model=Dict[str, Any])
async def validate_policy_syntax(
    request: PolicyValidationRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.1.11: Policy syntax validation
    """
    try:
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        policy_service = PolicyRegistryService(pool_manager)
        validation_result = await policy_service.validate_policy_syntax(request.policy_content)
        
        return {
            "status": "success",
            "validation_result": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Policy validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy validation failed: {str(e)}")

@router.get("/metrics", response_model=Dict[str, Any])
async def get_policy_metrics(
    days: int = 30,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.1.22: Policy enforcement metrics and analytics
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        policy_service = PolicyRegistryService(pool_manager)
        metrics = await policy_service.get_policy_metrics(tenant_id, days)
        
        return {
            "status": "success",
            "metrics": metrics,
            "tenant_id": tenant_id,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get policy metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get policy metrics: {str(e)}")

@router.get("/compliance-frameworks", response_model=Dict[str, Any])
async def get_compliance_frameworks(
    current_user: dict = Depends(get_current_user)
):
    """
    Task 16.1.20: Get available compliance frameworks
    """
    try:
        frameworks = {
            "SOX": {
                "name": "Sarbanes-Oxley Act",
                "jurisdiction": "US",
                "industry": ["SaaS", "FinTech", "E-commerce"],
                "sections": ["302", "404", "409", "906"],
                "description": "Financial reporting and internal controls"
            },
            "GDPR": {
                "name": "General Data Protection Regulation",
                "jurisdiction": "EU",
                "industry": ["SaaS", "E-commerce", "IT_Services"],
                "articles": ["6", "7", "17", "20", "25"],
                "description": "Data protection and privacy"
            },
            "RBI": {
                "name": "Reserve Bank of India Guidelines",
                "jurisdiction": "IN",
                "industry": ["Banking", "FinTech"],
                "guidelines": ["Digital Lending", "Outsourcing", "Cyber Security"],
                "description": "Banking and financial services regulation"
            },
            "HIPAA": {
                "name": "Health Insurance Portability and Accountability Act",
                "jurisdiction": "US",
                "industry": ["Healthcare", "SaaS"],
                "rules": ["Privacy Rule", "Security Rule", "Breach Notification"],
                "description": "Healthcare data protection"
            },
            "SAAS_BUSINESS_RULES": {
                "name": "SaaS Business Rules Framework",
                "jurisdiction": "Global",
                "industry": ["SaaS"],
                "categories": ["Subscription", "Usage", "Billing", "Churn"],
                "description": "SaaS-specific business logic and workflows"
            }
        }
        
        return {
            "status": "success",
            "compliance_frameworks": frameworks,
            "framework_count": len(frameworks),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get compliance frameworks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance frameworks: {str(e)}")

@router.get("/enforcement-modes", response_model=Dict[str, Any])
async def get_enforcement_modes(
    current_user: dict = Depends(get_current_user)
):
    """
    Task 16.1.21: Get available enforcement modes
    """
    try:
        enforcement_modes = {
            "enforcing": {
                "description": "Strict enforcement - violations block execution",
                "fail_closed": True,
                "override_allowed": True,
                "use_cases": ["Production", "Compliance-critical workflows"]
            },
            "permissive": {
                "description": "Warning mode - violations logged but execution continues",
                "fail_closed": False,
                "override_allowed": True,
                "use_cases": ["Testing", "Development", "Gradual rollout"]
            },
            "disabled": {
                "description": "Policy disabled - no enforcement",
                "fail_closed": False,
                "override_allowed": False,
                "use_cases": ["Maintenance", "Emergency bypass"]
            }
        }
        
        return {
            "status": "success",
            "enforcement_modes": enforcement_modes,
            "default_mode": "enforcing",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get enforcement modes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enforcement modes: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def policy_orchestration_health(
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Health check for policy orchestration service
    """
    try:
        health_status = {
            "service": "policy_orchestration",
            "status": "healthy",
            "database_connected": pool_manager is not None,
            "opa_integration": "simulated",  # Would check actual OPA connection
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "service": "policy_orchestration",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
