"""
Dynamic RBA Template Registry API - Chapter 19.1
Task 19.1.7: Registry APIs for template CRUD operations
Integrates with existing Enhanced Capability Registry and dynamic template loading
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ai_crenovent.database.database_manager import DatabaseManager
from ai_crenovent.dsl.governance.evidence_service import EvidenceService
from ai_crenovent.dsl.tenancy.tenant_context_manager import TenantContextManager
from ai_crenovent.dsl.templates.rba_template_service import DynamicRBATemplateService, LifecycleState
from ai_crenovent.dsl.registry.enhanced_capability_registry import EnhancedCapabilityRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rba-templates", tags=["RBA Templates"])

# Dependency injection
def get_db_manager() -> DatabaseManager:
    return DatabaseManager()

def get_evidence_service(db: DatabaseManager = Depends(get_db_manager)) -> EvidenceService:
    return EvidenceService(db)

def get_tenant_context_manager(db: DatabaseManager = Depends(get_db_manager)) -> TenantContextManager:
    return TenantContextManager(db)

def get_rba_template_service(
    db: DatabaseManager = Depends(get_db_manager),
    evidence: EvidenceService = Depends(get_evidence_service),
    tenant_context: TenantContextManager = Depends(get_tenant_context_manager)
) -> DynamicRBATemplateService:
    return DynamicRBATemplateService(db, evidence, tenant_context)

# Mock user context for now (would come from JWT in real implementation)
def get_current_user():
    return {"user_id": 1, "tenant_id": 1300}

@router.get("/", summary="List RBA Templates")
async def list_rba_templates(
    industry_code: Optional[str] = Query(None, description="Filter by industry (SaaS, Banking, Insurance)"),
    template_category: Optional[str] = Query(None, description="Filter by template category"),
    lifecycle_state: Optional[str] = Query(None, description="Filter by lifecycle state"),
    current_user: dict = Depends(get_current_user),
    template_service: DynamicRBATemplateService = Depends(get_rba_template_service)
):
    """
    Task 19.1.7: List RBA templates with dynamic filtering
    """
    try:
        tenant_id = current_user["tenant_id"]
        
        if industry_code:
            # Load industry-specific templates dynamically
            templates = await template_service.load_industry_templates(industry_code, tenant_id)
            
            # Apply additional filters
            if template_category:
                templates = [t for t in templates if t['template_category'] == template_category]
            if lifecycle_state:
                templates = [t for t in templates if t['lifecycle_state'] == lifecycle_state]
        else:
            # Load all templates for tenant
            templates = []
            for industry in ['SaaS', 'Banking', 'Insurance']:
                industry_templates = await template_service.load_industry_templates(industry, tenant_id)
                templates.extend(industry_templates)
            
            # Apply filters
            if template_category:
                templates = [t for t in templates if t['template_category'] == template_category]
            if lifecycle_state:
                templates = [t for t in templates if t['lifecycle_state'] == lifecycle_state]
        
        logger.info(f"Listed {len(templates)} RBA templates for tenant {tenant_id}")
        return {
            "templates": templates,
            "total_count": len(templates),
            "filters_applied": {
                "industry_code": industry_code,
                "template_category": template_category,
                "lifecycle_state": lifecycle_state
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list RBA templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")

@router.get("/{template_id}", summary="Get RBA Template Details")
async def get_rba_template(
    template_id: str,
    current_user: dict = Depends(get_current_user),
    template_service: DynamicRBATemplateService = Depends(get_rba_template_service)
):
    """
    Task 19.1.7: Get detailed RBA template information
    """
    try:
        tenant_id = current_user["tenant_id"]
        template = await template_service.get_template(template_id, tenant_id)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get RBA template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")

@router.post("/register-from-yaml", summary="Register Template from YAML")
async def register_template_from_yaml(
    yaml_path: str,
    industry_code: str,
    template_category: str,
    current_user: dict = Depends(get_current_user),
    template_service: DynamicRBATemplateService = Depends(get_rba_template_service)
):
    """
    Task 19.1.7: Register RBA template from existing YAML workflow
    """
    try:
        tenant_id = current_user["tenant_id"]
        user_id = current_user["user_id"]
        
        template_id = await template_service.register_template_from_yaml(
            yaml_path=yaml_path,
            industry_code=industry_code,
            template_category=template_category,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return {
            "template_id": template_id,
            "message": f"Successfully registered RBA template from {yaml_path}",
            "industry_code": industry_code,
            "template_category": template_category
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to register template from YAML {yaml_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register template: {str(e)}")

@router.post("/{template_id}/promote", summary="Promote Template Lifecycle")
async def promote_template(
    template_id: str,
    to_state: str,
    reason: str = "",
    current_user: dict = Depends(get_current_user),
    template_service: DynamicRBATemplateService = Depends(get_rba_template_service)
):
    """
    Task 19.1.35: Promote template through lifecycle states
    """
    try:
        tenant_id = current_user["tenant_id"]
        user_id = current_user["user_id"]
        
        # Validate lifecycle state
        try:
            lifecycle_state = LifecycleState(to_state)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid lifecycle state: {to_state}")
        
        promotion_id = await template_service.promote_template(
            template_id=template_id,
            to_state=lifecycle_state,
            tenant_id=tenant_id,
            user_id=user_id,
            reason=reason
        )
        
        return {
            "promotion_id": promotion_id,
            "template_id": template_id,
            "to_state": to_state,
            "message": f"Successfully promoted template to {to_state}",
            "reason": reason
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to promote template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to promote template: {str(e)}")

@router.post("/{template_id}/pin-version", summary="Pin Template Version")
async def pin_template_version(
    template_id: str,
    version: str,
    reason: str = "",
    expires_days: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
    template_service: DynamicRBATemplateService = Depends(get_rba_template_service)
):
    """
    Task 19.1.37: Pin template version for tenant
    """
    try:
        tenant_id = current_user["tenant_id"]
        user_id = current_user["user_id"]
        
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        pin_id = await template_service.pin_template_version(
            template_id=template_id,
            version=version,
            tenant_id=tenant_id,
            user_id=user_id,
            reason=reason,
            expires_at=expires_at
        )
        
        return {
            "pin_id": pin_id,
            "template_id": template_id,
            "pinned_version": version,
            "message": f"Successfully pinned template to version {version}",
            "expires_at": expires_at.isoformat() if expires_at else None
        }
        
    except Exception as e:
        logger.error(f"Failed to pin template {template_id} version: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pin version: {str(e)}")

@router.get("/industry/{industry_code}", summary="Get Industry Templates")
async def get_industry_templates(
    industry_code: str,
    current_user: dict = Depends(get_current_user),
    template_service: DynamicRBATemplateService = Depends(get_rba_template_service)
):
    """
    Task 19.1.9-19.1.34: Get industry-specific RBA templates dynamically
    """
    try:
        tenant_id = current_user["tenant_id"]
        
        # Validate industry code
        valid_industries = ['SaaS', 'Banking', 'Insurance', 'E-commerce', 'FinTech', 'IT_Services']
        if industry_code not in valid_industries:
            raise HTTPException(status_code=400, detail=f"Invalid industry code. Must be one of: {valid_industries}")
        
        templates = await template_service.load_industry_templates(industry_code, tenant_id)
        
        # Group by template category
        categories = {}
        for template in templates:
            category = template['template_category']
            if category not in categories:
                categories[category] = []
            categories[category].append(template)
        
        return {
            "industry_code": industry_code,
            "total_templates": len(templates),
            "categories": categories,
            "template_categories": list(categories.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get {industry_code} templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get industry templates: {str(e)}")

@router.get("/categories/available", summary="Get Available Template Categories")
async def get_available_categories(
    industry_code: Optional[str] = Query(None, description="Filter by industry"),
    current_user: dict = Depends(get_current_user),
    template_service: DynamicRBATemplateService = Depends(get_rba_template_service)
):
    """
    Get available template categories dynamically
    """
    try:
        tenant_id = current_user["tenant_id"]
        
        # Define industry-specific categories dynamically
        categories = {
            "SaaS": [
                "arr_rollup", "churn_alerts", "pipeline_hygiene", "subscription_management",
                "customer_success", "billing_automation", "usage_tracking", "freemium_conversion"
            ],
            "Banking": [
                "npa_detection", "aml_kyc_compliance", "credit_scoring", "loan_processing",
                "regulatory_reporting", "risk_assessment", "fraud_detection", "customer_onboarding"
            ],
            "Insurance": [
                "claims_aging", "underwriting_validation", "policy_renewal", "fraud_investigation",
                "actuarial_analysis", "regulatory_compliance", "customer_service", "claims_processing"
            ]
        }
        
        if industry_code:
            return {
                "industry_code": industry_code,
                "categories": categories.get(industry_code, [])
            }
        else:
            return {
                "all_categories": categories,
                "industries": list(categories.keys())
            }
        
    except Exception as e:
        logger.error(f"Failed to get template categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@router.get("/{template_id}/evidence", summary="Get Template Evidence")
async def get_template_evidence(
    template_id: str,
    evidence_type: Optional[str] = Query(None, description="Filter by evidence type"),
    current_user: dict = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Task 19.1.12, 19.1.16, 19.1.19, 19.1.23, 19.1.27, 19.1.31, 19.1.34: Get template evidence packs
    """
    try:
        tenant_id = current_user["tenant_id"]
        
        query = """
            SELECT rte.*, rtl.capability_id
            FROM rba_template_evidence rte
            JOIN rba_template_lifecycle rtl ON rte.capability_id = rtl.capability_id
            WHERE rte.tenant_id = ? AND rtl.capability_id IN (
                SELECT capability_id FROM rba_industry_templates WHERE template_id = ?
            )
        """
        params = [tenant_id, template_id]
        
        if evidence_type:
            query += " AND rte.evidence_type = ?"
            params.append(evidence_type)
        
        query += " ORDER BY rte.created_at DESC"
        
        evidence_records = await db_manager.fetch_all(query, params)
        
        return {
            "template_id": template_id,
            "evidence_count": len(evidence_records),
            "evidence_records": [dict(record) for record in evidence_records]
        }
        
    except Exception as e:
        logger.error(f"Failed to get template evidence for {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get evidence: {str(e)}")

@router.get("/lifecycle/states", summary="Get Available Lifecycle States")
async def get_lifecycle_states():
    """
    Task 19.1.35: Get available lifecycle states
    """
    return {
        "lifecycle_states": [state.value for state in LifecycleState],
        "valid_transitions": {
            LifecycleState.DRAFT.value: [LifecycleState.PUBLISHED.value],
            LifecycleState.PUBLISHED.value: [LifecycleState.PROMOTED.value, LifecycleState.DEPRECATED.value],
            LifecycleState.PROMOTED.value: [LifecycleState.DEPRECATED.value],
            LifecycleState.DEPRECATED.value: [LifecycleState.RETIRED.value, LifecycleState.PUBLISHED.value],
            LifecycleState.RETIRED.value: []
        }
    }
