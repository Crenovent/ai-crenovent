"""
CHAPTER 16.3: COMPLIANCE OVERLAY API
Tasks 16.3.6-16.3.11: Multi-industry compliance overlay management APIs
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from src.database.connection_pool import get_pool_manager, ConnectionPoolManager
from src.auth.auth_utils import get_current_user, get_tenant_id_from_user
from dsl.governance.compliance_overlay_service import (
    ComplianceOverlayService, ComplianceFramework, EnforcementMode
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/compliance-overlays", tags=["Compliance Overlays"])

# Pydantic models for API requests/responses
class CreateComplianceOverlayRequest(BaseModel):
    overlay_name: str = Field(..., description="Compliance overlay name")
    overlay_description: str = Field(..., description="Overlay description")
    compliance_framework: str = Field(..., description="Compliance framework (SOX, GDPR, etc.)")
    industry_code: str = Field(default="SaaS", description="Industry code")
    jurisdiction: str = Field(default="US", description="Jurisdiction")
    regulatory_authority: str = Field(..., description="Regulatory authority")
    overlay_config: Dict[str, Any] = Field(..., description="Overlay configuration")
    enforcement_rules: Dict[str, Any] = Field(..., description="Enforcement rules")
    applicable_workflows: List[str] = Field(default=[], description="Applicable workflows")
    enforcement_mode: str = Field(default="enforcing", description="Enforcement mode")

class ComplianceEnforcementRequest(BaseModel):
    workflow_context: Dict[str, Any] = Field(..., description="Workflow context for compliance check")

@router.post("/overlays", response_model=Dict[str, Any])
async def create_compliance_overlay(
    request: CreateComplianceOverlayRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.3.6: Create new compliance overlay
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        user_id = current_user['user_id']
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        # Create overlay configuration
        overlay_config = {
            "overlay_name": request.overlay_name,
            "overlay_description": request.overlay_description,
            "compliance_framework": request.compliance_framework,
            "industry_code": request.industry_code,
            "jurisdiction": request.jurisdiction,
            "regulatory_authority": request.regulatory_authority,
            "overlay_config": request.overlay_config,
            "enforcement_rules": request.enforcement_rules,
            "applicable_workflows": request.applicable_workflows,
            "enforcement_mode": request.enforcement_mode
        }
        
        # Create overlay using service
        compliance_service = ComplianceOverlayService(pool_manager)
        overlay_id = await compliance_service.create_compliance_overlay(overlay_config, tenant_id, user_id)
        
        return {
            "status": "success",
            "overlay_id": overlay_id,
            "message": "Compliance overlay created successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"❌ Invalid overlay data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to create compliance overlay: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create compliance overlay: {str(e)}")

@router.get("/overlays", response_model=Dict[str, Any])
async def get_tenant_compliance_overlays(
    framework_filter: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.3.9: Get tenant compliance overlays
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        compliance_service = ComplianceOverlayService(pool_manager)
        overlays = await compliance_service.get_tenant_compliance_overlays(tenant_id, framework_filter)
        
        return {
            "status": "success",
            "overlays": overlays,
            "total_overlays": len(overlays),
            "framework_filter": framework_filter,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get compliance overlays: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance overlays: {str(e)}")

@router.post("/enforce", response_model=Dict[str, Any])
async def enforce_compliance(
    request: ComplianceEnforcementRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.3.8: Enforce compliance overlays for workflow
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        user_id = current_user['user_id']
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        # Add user context to workflow context
        workflow_context = {
            **request.workflow_context,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "enforcement_timestamp": datetime.now().isoformat()
        }
        
        compliance_service = ComplianceOverlayService(pool_manager)
        result = await compliance_service.enforce_compliance(tenant_id, workflow_context)
        
        return {
            "status": "success",
            "enforcement_result": result.enforcement_result.value,
            "compliance_score": result.compliance_score,
            "violation_count": result.violation_count,
            "violation_details": result.violation_details,
            "remediation_required": result.remediation_required,
            "evidence_pack_id": result.evidence_pack_id,
            "enforcement_id": result.enforcement_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Compliance enforcement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance enforcement failed: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_compliance_analytics(
    period_days: int = 30,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.3.10: Get compliance analytics and metrics
    """
    try:
        tenant_id = await get_tenant_id_from_user(current_user)
        
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        compliance_service = ComplianceOverlayService(pool_manager)
        analytics = await compliance_service.get_compliance_analytics(tenant_id, period_days)
        
        return {
            "status": "success",
            "analytics": analytics,
            "tenant_id": tenant_id,
            "period_days": period_days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get compliance analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get compliance analytics: {str(e)}")

@router.get("/templates/saas", response_model=Dict[str, Any])
async def get_saas_compliance_templates(
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 16.3.11: Get SaaS compliance overlay templates
    """
    try:
        if not pool_manager:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        compliance_service = ComplianceOverlayService(pool_manager)
        templates = await compliance_service.get_saas_compliance_templates()
        
        return {
            "status": "success",
            "saas_compliance_templates": templates,
            "template_count": len(templates.get("saas_overlays", {})),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get SaaS compliance templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SaaS compliance templates: {str(e)}")

@router.get("/frameworks", response_model=Dict[str, Any])
async def get_compliance_frameworks(
    current_user: dict = Depends(get_current_user)
):
    """
    Get available compliance frameworks
    """
    try:
        frameworks = {
            "SOX": {
                "name": "Sarbanes-Oxley Act",
                "jurisdiction": "US",
                "industry": ["SaaS", "FinTech", "E-commerce"],
                "sections": ["302", "404", "409", "906"],
                "description": "Financial reporting and internal controls",
                "enforcement_level": "mandatory"
            },
            "GDPR": {
                "name": "General Data Protection Regulation",
                "jurisdiction": "EU",
                "industry": ["SaaS", "E-commerce", "IT_Services"],
                "articles": ["6", "7", "17", "20", "25"],
                "description": "Data protection and privacy",
                "enforcement_level": "mandatory"
            },
            "RBI": {
                "name": "Reserve Bank of India Guidelines",
                "jurisdiction": "IN",
                "industry": ["Banking", "FinTech"],
                "guidelines": ["Digital Lending", "Outsourcing", "Cyber Security"],
                "description": "Banking and financial services regulation",
                "enforcement_level": "mandatory"
            },
            "HIPAA": {
                "name": "Health Insurance Portability and Accountability Act",
                "jurisdiction": "US",
                "industry": ["Healthcare", "SaaS"],
                "rules": ["Privacy Rule", "Security Rule", "Breach Notification"],
                "description": "Healthcare data protection",
                "enforcement_level": "mandatory"
            },
            "NAIC": {
                "name": "National Association of Insurance Commissioners",
                "jurisdiction": "US",
                "industry": ["Insurance"],
                "standards": ["Solvency", "Market Conduct", "Privacy"],
                "description": "Insurance regulation and oversight",
                "enforcement_level": "mandatory"
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
    Get available enforcement modes
    """
    try:
        enforcement_modes = {
            "enforcing": {
                "description": "Strict enforcement - violations block execution",
                "fail_closed": True,
                "override_allowed": False,
                "use_cases": ["Production", "Compliance-critical workflows"]
            },
            "monitoring": {
                "description": "Monitoring mode - violations logged but execution continues",
                "fail_closed": False,
                "override_allowed": True,
                "use_cases": ["Testing", "Development", "Gradual rollout"]
            },
            "disabled": {
                "description": "Overlay disabled - no enforcement",
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
async def compliance_overlay_health(
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Health check for compliance overlay service
    """
    try:
        health_status = {
            "service": "compliance_overlays",
            "status": "healthy",
            "database_connected": pool_manager is not None,
            "features": {
                "overlay_creation": True,
                "compliance_enforcement": True,
                "industry_templates": True,
                "analytics": True,
                "multi_framework_support": True
            },
            "supported_frameworks": [framework.value for framework in ComplianceFramework],
            "supported_modes": [mode.value for mode in EnforcementMode],
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "service": "compliance_overlays",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
