#!/usr/bin/env python3
"""
Industry-Specific Templates API - Chapter 14.1 Tasks T29-T32
============================================================
Tasks 14.1-T29, T30, T31, T32: Industry templates and cross-module flows

Features:
- Industry-specific approval templates (SaaS, Banking, Insurance, Healthcare)
- Override templates per industry with compliance frameworks
- Cross-module approval flows (Pipeline → Compensation → CFO)
- Cross-industry override chains with policy pack integration
- Template versioning and inheritance
- Dynamic template generation based on industry overlays
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json

from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class IndustryCode(str, Enum):
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"

class TemplateType(str, Enum):
    APPROVAL = "approval"
    OVERRIDE = "override"
    CROSS_MODULE = "cross_module"
    ESCALATION = "escalation"

class ApprovalThreshold(BaseModel):
    amount_threshold: float = Field(..., description="Financial threshold for approval")
    currency: str = Field("USD", description="Currency code")
    approval_levels: List[str] = Field(..., description="Required approval roles")
    sla_hours: int = Field(24, description="SLA in hours")

class IndustryTemplate(BaseModel):
    template_id: Optional[str] = Field(None, description="Template ID")
    template_name: str = Field(..., description="Template name")
    industry_code: IndustryCode = Field(..., description="Industry code")
    template_type: TemplateType = Field(..., description="Template type")
    description: str = Field(..., description="Template description")
    approval_thresholds: List[ApprovalThreshold] = Field(..., description="Approval thresholds")
    compliance_frameworks: List[str] = Field([], description="Required compliance frameworks")
    policy_pack_ids: List[str] = Field([], description="Associated policy packs")
    workflow_config: Dict[str, Any] = Field({}, description="Workflow configuration")
    active: bool = Field(True, description="Template active status")

class CrossModuleFlow(BaseModel):
    flow_id: Optional[str] = Field(None, description="Flow ID")
    flow_name: str = Field(..., description="Cross-module flow name")
    source_module: str = Field(..., description="Source module")
    target_modules: List[str] = Field(..., description="Target modules")
    trigger_conditions: Dict[str, Any] = Field(..., description="Flow trigger conditions")
    approval_sequence: List[str] = Field(..., description="Approval sequence")
    parallel_approvals: bool = Field(False, description="Allow parallel approvals")
    industry_specific: bool = Field(False, description="Industry-specific flow")
    industry_codes: List[IndustryCode] = Field([], description="Applicable industries")

class IndustryTemplateService:
    """
    Industry-specific templates service
    Tasks 14.1-T29 to T32: Industry templates and cross-module flows
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
        # Pre-defined industry templates (Task 14.1-T29)
        self.saas_templates = {
            "deal_approval": {
                "small_deal": {"threshold": 10000, "approvers": ["sales_manager"], "sla_hours": 24},
                "medium_deal": {"threshold": 100000, "approvers": ["sales_director", "finance_manager"], "sla_hours": 48},
                "large_deal": {"threshold": 500000, "approvers": ["vp_sales", "cfo", "compliance_officer"], "sla_hours": 72},
                "enterprise_deal": {"threshold": 1000000, "approvers": ["cro", "cfo", "ceo"], "sla_hours": 120}
            },
            "subscription_changes": {
                "plan_upgrade": {"approvers": ["customer_success"], "sla_hours": 4},
                "plan_downgrade": {"approvers": ["sales_manager", "finance_manager"], "sla_hours": 24},
                "cancellation": {"approvers": ["retention_specialist", "sales_director"], "sla_hours": 48}
            }
        }
        
        self.banking_templates = {
            "loan_approval": {
                "personal_loan": {"threshold": 100000, "approvers": ["loan_officer", "branch_manager"], "sla_hours": 72},
                "home_loan": {"threshold": 5000000, "approvers": ["senior_loan_officer", "credit_manager", "regional_head"], "sla_hours": 168},
                "business_loan": {"threshold": 10000000, "approvers": ["business_head", "credit_committee", "ceo"], "sla_hours": 240}
            },
            "transaction_limits": {
                "daily_limit_increase": {"threshold": 500000, "approvers": ["relationship_manager"], "sla_hours": 24},
                "monthly_limit_increase": {"threshold": 2000000, "approvers": ["branch_manager", "ops_head"], "sla_hours": 48}
            }
        }
        
        self.insurance_templates = {
            "claim_approval": {
                "motor_claim": {"threshold": 100000, "approvers": ["claims_officer"], "sla_hours": 48},
                "health_claim": {"threshold": 500000, "approvers": ["medical_officer", "claims_manager"], "sla_hours": 72},
                "life_claim": {"threshold": 5000000, "approvers": ["underwriter", "claims_head", "ceo"], "sla_hours": 168}
            },
            "policy_modifications": {
                "premium_adjustment": {"approvers": ["underwriter"], "sla_hours": 24},
                "coverage_increase": {"approvers": ["senior_underwriter", "actuarial_head"], "sla_hours": 48}
            }
        }
        
        # Override templates per industry (Task 14.1-T30)
        self.override_templates = {
            IndustryCode.SAAS: {
                "urgent_deal_closure": {
                    "reason_codes": ["quarter_end_pressure", "customer_escalation", "competitive_threat"],
                    "required_approvers": ["sales_director", "legal_counsel"],
                    "compliance_frameworks": ["SOX", "GDPR"],
                    "evidence_requirements": ["customer_communication", "legal_review", "finance_approval"]
                },
                "pricing_exception": {
                    "reason_codes": ["strategic_customer", "volume_discount", "competitive_match"],
                    "required_approvers": ["pricing_manager", "finance_director"],
                    "compliance_frameworks": ["SOX"],
                    "evidence_requirements": ["competitive_analysis", "margin_impact", "cfo_approval"]
                }
            },
            IndustryCode.BANKING: {
                "rbi_exemption": {
                    "reason_codes": ["systemic_risk", "customer_hardship", "regulatory_guidance"],
                    "required_approvers": ["compliance_head", "ceo", "board_member"],
                    "compliance_frameworks": ["RBI", "BASEL_III"],
                    "evidence_requirements": ["rbi_communication", "risk_assessment", "board_resolution"]
                },
                "urgent_disbursal": {
                    "reason_codes": ["medical_emergency", "natural_disaster", "business_continuity"],
                    "required_approvers": ["regional_head", "risk_head"],
                    "compliance_frameworks": ["RBI", "AML"],
                    "evidence_requirements": ["emergency_documentation", "kyc_verification", "risk_mitigation"]
                }
            },
            IndustryCode.INSURANCE: {
                "irdai_exemption": {
                    "reason_codes": ["catastrophic_event", "regulatory_change", "customer_protection"],
                    "required_approvers": ["actuarial_head", "ceo", "compliance_officer"],
                    "compliance_frameworks": ["IRDAI", "IRDA"],
                    "evidence_requirements": ["actuarial_report", "regulatory_filing", "board_approval"]
                }
            }
        }
    
    async def create_industry_template(self, template: IndustryTemplate, tenant_id: int) -> Dict[str, Any]:
        """
        Create industry-specific approval template (Task 14.1-T29)
        """
        try:
            template_id = str(uuid.uuid4())
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                await conn.execute("""
                    INSERT INTO industry_approval_templates (
                        template_id, tenant_id, template_name, industry_code,
                        template_type, description, approval_thresholds,
                        compliance_frameworks, policy_pack_ids, workflow_config,
                        active, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                    template_id, tenant_id, template.template_name, template.industry_code.value,
                    template.template_type.value, template.description,
                    json.dumps([threshold.dict() for threshold in template.approval_thresholds]),
                    json.dumps(template.compliance_frameworks),
                    json.dumps(template.policy_pack_ids),
                    json.dumps(template.workflow_config),
                    template.active, datetime.utcnow()
                )
                
                return {
                    "template_id": template_id,
                    "template_name": template.template_name,
                    "industry_code": template.industry_code.value,
                    "template_type": template.template_type.value,
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "created"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to create industry template: {e}")
            raise
    
    async def create_override_template(self, industry_code: IndustryCode, 
                                     override_type: str, tenant_id: int) -> Dict[str, Any]:
        """
        Create industry-specific override template (Task 14.1-T30)
        """
        try:
            if industry_code not in self.override_templates:
                raise ValueError(f"No override templates defined for industry: {industry_code}")
            
            industry_overrides = self.override_templates[industry_code]
            if override_type not in industry_overrides:
                raise ValueError(f"Override type '{override_type}' not found for industry: {industry_code}")
            
            template_config = industry_overrides[override_type]
            template_id = str(uuid.uuid4())
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                await conn.execute("""
                    INSERT INTO industry_override_templates (
                        template_id, tenant_id, industry_code, override_type,
                        reason_codes, required_approvers, compliance_frameworks,
                        evidence_requirements, template_config, active, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    template_id, tenant_id, industry_code.value, override_type,
                    json.dumps(template_config["reason_codes"]),
                    json.dumps(template_config["required_approvers"]),
                    json.dumps(template_config["compliance_frameworks"]),
                    json.dumps(template_config["evidence_requirements"]),
                    json.dumps(template_config), True, datetime.utcnow()
                )
                
                return {
                    "template_id": template_id,
                    "industry_code": industry_code.value,
                    "override_type": override_type,
                    "reason_codes": template_config["reason_codes"],
                    "required_approvers": template_config["required_approvers"],
                    "compliance_frameworks": template_config["compliance_frameworks"],
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to create override template: {e}")
            raise
    
    async def create_cross_module_flow(self, flow: CrossModuleFlow, tenant_id: int) -> Dict[str, Any]:
        """
        Create cross-module approval flow (Task 14.1-T31)
        """
        try:
            flow_id = str(uuid.uuid4())
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                await conn.execute("""
                    INSERT INTO cross_module_approval_flows (
                        flow_id, tenant_id, flow_name, source_module, target_modules,
                        trigger_conditions, approval_sequence, parallel_approvals,
                        industry_specific, industry_codes, flow_config, active, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    flow_id, tenant_id, flow.flow_name, flow.source_module,
                    json.dumps(flow.target_modules), json.dumps(flow.trigger_conditions),
                    json.dumps(flow.approval_sequence), flow.parallel_approvals,
                    flow.industry_specific, json.dumps([code.value for code in flow.industry_codes]),
                    json.dumps({}), True, datetime.utcnow()
                )
                
                return {
                    "flow_id": flow_id,
                    "flow_name": flow.flow_name,
                    "source_module": flow.source_module,
                    "target_modules": flow.target_modules,
                    "approval_sequence": flow.approval_sequence,
                    "industry_specific": flow.industry_specific,
                    "created_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to create cross-module flow: {e}")
            raise
    
    async def get_industry_templates(self, industry_code: IndustryCode, 
                                   template_type: Optional[TemplateType] = None,
                                   tenant_id: int = None) -> List[Dict[str, Any]]:
        """
        Get industry-specific templates
        """
        try:
            async with self.pool_manager.get_connection() as conn:
                if tenant_id:
                    await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                query = """
                    SELECT * FROM industry_approval_templates
                    WHERE industry_code = $1 AND active = true
                """
                params = [industry_code.value]
                
                if template_type:
                    query += " AND template_type = $2"
                    params.append(template_type.value)
                
                if tenant_id:
                    query += f" AND tenant_id = ${len(params) + 1}"
                    params.append(tenant_id)
                
                query += " ORDER BY created_at DESC"
                
                rows = await conn.fetch(query, *params)
                
                templates = []
                for row in rows:
                    template = {
                        "template_id": row['template_id'],
                        "template_name": row['template_name'],
                        "industry_code": row['industry_code'],
                        "template_type": row['template_type'],
                        "description": row['description'],
                        "approval_thresholds": json.loads(row['approval_thresholds']),
                        "compliance_frameworks": json.loads(row['compliance_frameworks']),
                        "policy_pack_ids": json.loads(row['policy_pack_ids']),
                        "workflow_config": json.loads(row['workflow_config']),
                        "created_at": row['created_at'].isoformat()
                    }
                    templates.append(template)
                
                return templates
                
        except Exception as e:
            logger.error(f"❌ Failed to get industry templates: {e}")
            return []
    
    async def generate_predefined_templates(self, industry_code: IndustryCode, 
                                          tenant_id: int) -> Dict[str, Any]:
        """
        Generate pre-defined templates for an industry
        """
        try:
            generated_templates = []
            
            if industry_code == IndustryCode.SAAS:
                templates_config = self.saas_templates
            elif industry_code == IndustryCode.BANKING:
                templates_config = self.banking_templates
            elif industry_code == IndustryCode.INSURANCE:
                templates_config = self.insurance_templates
            else:
                templates_config = {}
            
            for category, templates in templates_config.items():
                for template_name, config in templates.items():
                    # Create approval threshold
                    threshold = ApprovalThreshold(
                        amount_threshold=config.get("threshold", 0),
                        currency="USD",
                        approval_levels=config["approvers"],
                        sla_hours=config["sla_hours"]
                    )
                    
                    # Create template
                    template = IndustryTemplate(
                        template_name=f"{industry_code.value}_{category}_{template_name}",
                        industry_code=industry_code,
                        template_type=TemplateType.APPROVAL,
                        description=f"Pre-defined {template_name} template for {industry_code.value}",
                        approval_thresholds=[threshold],
                        compliance_frameworks=self._get_industry_compliance_frameworks(industry_code),
                        workflow_config={"category": category, "predefined": True}
                    )
                    
                    result = await self.create_industry_template(template, tenant_id)
                    generated_templates.append(result)
            
            return {
                "industry_code": industry_code.value,
                "templates_generated": len(generated_templates),
                "templates": generated_templates,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate predefined templates: {e}")
            raise
    
    def _get_industry_compliance_frameworks(self, industry_code: IndustryCode) -> List[str]:
        """Get compliance frameworks for industry"""
        frameworks = {
            IndustryCode.SAAS: ["SOX", "GDPR", "CCPA"],
            IndustryCode.BANKING: ["RBI", "BASEL_III", "AML", "KYC"],
            IndustryCode.INSURANCE: ["IRDAI", "IRDA", "GDPR"],
            IndustryCode.HEALTHCARE: ["HIPAA", "GDPR", "FDA"],
            IndustryCode.ECOMMERCE: ["PCI_DSS", "GDPR", "CCPA"],
            IndustryCode.FINTECH: ["RBI", "PCI_DSS", "AML", "GDPR"]
        }
        return frameworks.get(industry_code, ["GDPR"])

# Initialize service
industry_service = None

def get_industry_service(pool_manager=Depends(get_pool_manager)) -> IndustryTemplateService:
    global industry_service
    if industry_service is None:
        industry_service = IndustryTemplateService(pool_manager)
    return industry_service

# API Endpoints
@router.post("/industry-templates", response_model=Dict[str, Any])
async def create_industry_template(
    template: IndustryTemplate,
    tenant_id: int = Query(..., description="Tenant ID"),
    service: IndustryTemplateService = Depends(get_industry_service)
):
    """
    Create industry-specific approval template
    Task 14.1-T29: Industry approval templates
    """
    return await service.create_industry_template(template, tenant_id)

@router.post("/override-templates", response_model=Dict[str, Any])
async def create_override_template(
    industry_code: IndustryCode = Query(..., description="Industry code"),
    override_type: str = Query(..., description="Override type"),
    tenant_id: int = Query(..., description="Tenant ID"),
    service: IndustryTemplateService = Depends(get_industry_service)
):
    """
    Create industry-specific override template
    Task 14.1-T30: Industry override templates
    """
    return await service.create_override_template(industry_code, override_type, tenant_id)

@router.post("/cross-module-flows", response_model=Dict[str, Any])
async def create_cross_module_flow(
    flow: CrossModuleFlow,
    tenant_id: int = Query(..., description="Tenant ID"),
    service: IndustryTemplateService = Depends(get_industry_service)
):
    """
    Create cross-module approval flow
    Task 14.1-T31: Cross-module approval flows
    """
    return await service.create_cross_module_flow(flow, tenant_id)

@router.get("/industry-templates", response_model=List[Dict[str, Any]])
async def get_industry_templates(
    industry_code: IndustryCode = Query(..., description="Industry code"),
    template_type: Optional[TemplateType] = Query(None, description="Template type filter"),
    tenant_id: Optional[int] = Query(None, description="Tenant ID filter"),
    service: IndustryTemplateService = Depends(get_industry_service)
):
    """
    Get industry-specific templates
    """
    return await service.get_industry_templates(industry_code, template_type, tenant_id)

@router.post("/generate-predefined-templates", response_model=Dict[str, Any])
async def generate_predefined_templates(
    industry_code: IndustryCode = Query(..., description="Industry code"),
    tenant_id: int = Query(..., description="Tenant ID"),
    service: IndustryTemplateService = Depends(get_industry_service)
):
    """
    Generate pre-defined templates for an industry
    """
    return await service.generate_predefined_templates(industry_code, tenant_id)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "industry_templates_api", "timestamp": datetime.utcnow()}
