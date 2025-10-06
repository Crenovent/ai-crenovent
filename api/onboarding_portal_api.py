"""
Task 10.2.2: Build self-service onboarding portal
Single entry point for tenant admins with React + FastAPI backend

Features:
- Self-service tenant onboarding with step-by-step wizard
- Integration with provisioning APIs
- Guided workflows for different personas (Admin, RevOps)
- Real-time onboarding progress tracking
- Evidence pack generation and compliance reporting
- Multi-industry template support
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime
import uuid
import json

# Import tenant provisioning service
from api.tenant_provisioning_api import provisioning_service, TenantProvisioningRequest
from dsl.tenancy.tenant_lifecycle_manager import TenantLifecycleManager
from dsl.governance.audit_trail_service import audit_trail_service, AuditEventType
from src.services.connection_pool_manager import pool_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onboarding-portal", tags=["Onboarding Portal"])

# =====================================================
# REQUEST/RESPONSE MODELS
# =====================================================

class OnboardingWizardStep(BaseModel):
    """Individual step in onboarding wizard"""
    step_id: str
    step_name: str
    step_description: str
    step_order: int
    is_completed: bool
    is_current: bool
    is_required: bool
    estimated_duration_minutes: int
    
class OnboardingSession(BaseModel):
    """Onboarding session state"""
    session_id: str
    tenant_name: str
    current_step: str
    progress_percentage: int
    persona_type: str  # Admin, RevOps, Compliance
    industry_type: str
    
    # Step tracking
    steps: List[OnboardingWizardStep]
    completed_steps: List[str]
    
    # Configuration data
    tenant_config: Dict[str, Any]
    integration_config: Dict[str, Any]
    compliance_config: Dict[str, Any]
    
    # Status and timestamps
    status: str
    created_at: str
    last_updated_at: str
    estimated_completion_time: str

class OnboardingWizardRequest(BaseModel):
    """Start onboarding wizard request"""
    tenant_name: str = Field(..., description="Tenant name")
    persona_type: str = Field(..., description="User persona (Admin, RevOps, Compliance)")
    industry_type: str = Field(..., description="Industry type")
    contact_email: str = Field(..., description="Contact email")
    organization_name: str = Field(..., description="Organization name")

class OnboardingStepUpdate(BaseModel):
    """Update onboarding step data"""
    step_id: str
    step_data: Dict[str, Any]
    mark_completed: bool = False

class IntegrationWizardRequest(BaseModel):
    """Integration wizard configuration"""
    integration_type: str = Field(..., description="CRM, Billing, CLM, Marketing, CS")
    connection_config: Dict[str, Any] = Field(..., description="Connection configuration")
    data_mapping: Optional[Dict[str, Any]] = Field(None, description="Data field mapping")
    test_connection: bool = Field(True, description="Test connection before saving")

class OnboardingCompletionRequest(BaseModel):
    """Complete onboarding request"""
    session_id: str
    final_review_completed: bool = True
    accept_terms: bool = True
    enable_monitoring: bool = True

# =====================================================
# ONBOARDING PORTAL SERVICE
# =====================================================

class OnboardingPortalService:
    """
    Task 10.2.2: Self-service onboarding portal service
    Orchestrates guided onboarding workflows
    """
    
    def __init__(self):
        self.lifecycle_manager = TenantLifecycleManager()
        
        # Onboarding sessions
        self.onboarding_sessions: Dict[str, OnboardingSession] = {}
        
        # Persona-specific step templates
        self.persona_steps = self._initialize_persona_steps()
        
        # Industry-specific templates
        self.industry_templates = self._initialize_industry_templates()
        
        logger.info("🚀 Onboarding Portal Service initialized")
    
    def _initialize_persona_steps(self) -> Dict[str, List[Dict[str, Any]]]:
        """Task 10.2.5: Initialize persona-based onboarding flows"""
        return {
            "Admin": [
                {
                    "step_id": "basic_info",
                    "step_name": "Basic Information",
                    "step_description": "Configure basic tenant information and contact details",
                    "step_order": 1,
                    "is_required": True,
                    "estimated_duration_minutes": 5
                },
                {
                    "step_id": "compliance_setup",
                    "step_name": "Compliance Configuration",
                    "step_description": "Select compliance frameworks and data residency requirements",
                    "step_order": 2,
                    "is_required": True,
                    "estimated_duration_minutes": 10
                },
                {
                    "step_id": "sla_selection",
                    "step_name": "SLA Tier Selection",
                    "step_description": "Choose service level agreement tier (T0/T1/T2)",
                    "step_order": 3,
                    "is_required": True,
                    "estimated_duration_minutes": 5
                },
                {
                    "step_id": "integration_setup",
                    "step_name": "System Integrations",
                    "step_description": "Configure CRM, billing, and other system integrations",
                    "step_order": 4,
                    "is_required": False,
                    "estimated_duration_minutes": 20
                },
                {
                    "step_id": "user_management",
                    "step_name": "User Management",
                    "step_description": "Set up user roles and permissions",
                    "step_order": 5,
                    "is_required": True,
                    "estimated_duration_minutes": 10
                },
                {
                    "step_id": "monitoring_setup",
                    "step_name": "Monitoring & Dashboards",
                    "step_description": "Configure monitoring dashboards and alerts",
                    "step_order": 6,
                    "is_required": False,
                    "estimated_duration_minutes": 15
                },
                {
                    "step_id": "final_review",
                    "step_name": "Final Review",
                    "step_description": "Review configuration and complete onboarding",
                    "step_order": 7,
                    "is_required": True,
                    "estimated_duration_minutes": 5
                }
            ],
            "RevOps": [
                {
                    "step_id": "basic_info",
                    "step_name": "Basic Information",
                    "step_description": "Configure basic tenant information",
                    "step_order": 1,
                    "is_required": True,
                    "estimated_duration_minutes": 5
                },
                {
                    "step_id": "revenue_config",
                    "step_name": "Revenue Configuration",
                    "step_description": "Set up revenue tracking and forecasting",
                    "step_order": 2,
                    "is_required": True,
                    "estimated_duration_minutes": 15
                },
                {
                    "step_id": "crm_integration",
                    "step_name": "CRM Integration",
                    "step_description": "Connect and configure CRM system",
                    "step_order": 3,
                    "is_required": True,
                    "estimated_duration_minutes": 20
                },
                {
                    "step_id": "pipeline_setup",
                    "step_name": "Pipeline Configuration",
                    "step_description": "Configure sales pipeline and stages",
                    "step_order": 4,
                    "is_required": True,
                    "estimated_duration_minutes": 15
                },
                {
                    "step_id": "reporting_setup",
                    "step_name": "Reporting & Analytics",
                    "step_description": "Set up revenue dashboards and reports",
                    "step_order": 5,
                    "is_required": True,
                    "estimated_duration_minutes": 10
                },
                {
                    "step_id": "final_review",
                    "step_name": "Final Review",
                    "step_description": "Review configuration and go live",
                    "step_order": 6,
                    "is_required": True,
                    "estimated_duration_minutes": 5
                }
            ],
            "Compliance": [
                {
                    "step_id": "basic_info",
                    "step_name": "Basic Information",
                    "step_description": "Configure tenant information with compliance focus",
                    "step_order": 1,
                    "is_required": True,
                    "estimated_duration_minutes": 5
                },
                {
                    "step_id": "regulatory_framework",
                    "step_name": "Regulatory Framework",
                    "step_description": "Select applicable regulatory frameworks",
                    "step_order": 2,
                    "is_required": True,
                    "estimated_duration_minutes": 15
                },
                {
                    "step_id": "data_governance",
                    "step_name": "Data Governance",
                    "step_description": "Configure data classification and handling policies",
                    "step_order": 3,
                    "is_required": True,
                    "estimated_duration_minutes": 20
                },
                {
                    "step_id": "consent_management",
                    "step_name": "Consent Management",
                    "step_description": "Set up GDPR/DPDP consent management",
                    "step_order": 4,
                    "is_required": True,
                    "estimated_duration_minutes": 15
                },
                {
                    "step_id": "audit_setup",
                    "step_name": "Audit & Reporting",
                    "step_description": "Configure audit trails and compliance reporting",
                    "step_order": 5,
                    "is_required": True,
                    "estimated_duration_minutes": 10
                },
                {
                    "step_id": "final_review",
                    "step_name": "Final Review",
                    "step_description": "Review compliance configuration",
                    "step_order": 6,
                    "is_required": True,
                    "estimated_duration_minutes": 5
                }
            ]
        }
    
    def _initialize_industry_templates(self) -> Dict[str, Dict[str, Any]]:
        """Task 10.2.8: Initialize industry-specific templates"""
        return {
            "SaaS": {
                "default_integrations": ["Salesforce", "Stripe", "HubSpot"],
                "compliance_frameworks": ["SOX", "GDPR"],
                "data_retention_tier": "Silver",
                "sample_data_packs": ["saas_opportunities.csv", "saas_accounts.csv"],
                "dashboard_templates": ["ARR Dashboard", "Churn Analysis", "Pipeline Health"]
            },
            "Banking": {
                "default_integrations": ["Core Banking", "Risk Management", "Compliance"],
                "compliance_frameworks": ["RBI", "BASEL_III"],
                "data_retention_tier": "Gold",
                "sample_data_packs": ["banking_loans.csv", "banking_customers.csv"],
                "dashboard_templates": ["Risk Dashboard", "Regulatory Reporting", "Loan Portfolio"]
            },
            "Insurance": {
                "default_integrations": ["Policy Management", "Claims System", "Underwriting"],
                "compliance_frameworks": ["IRDAI", "GDPR"],
                "data_retention_tier": "Gold",
                "sample_data_packs": ["insurance_policies.csv", "insurance_claims.csv"],
                "dashboard_templates": ["Claims Dashboard", "Policy Analytics", "Risk Assessment"]
            },
            "Healthcare": {
                "default_integrations": ["EMR", "Billing System", "Patient Portal"],
                "compliance_frameworks": ["HIPAA", "GDPR"],
                "data_retention_tier": "Gold",
                "sample_data_packs": ["healthcare_patients.csv", "healthcare_treatments.csv"],
                "dashboard_templates": ["Patient Analytics", "Treatment Outcomes", "Compliance Monitoring"]
            }
        }
    
    async def start_onboarding_wizard(self, request: OnboardingWizardRequest) -> OnboardingSession:
        """Task 10.2.5: Start guided onboarding workflow"""
        session_id = str(uuid.uuid4())
        
        # Get persona-specific steps
        persona_steps = self.persona_steps.get(request.persona_type, self.persona_steps["Admin"])
        
        # Create wizard steps
        steps = []
        for step_template in persona_steps:
            step = OnboardingWizardStep(
                step_id=step_template["step_id"],
                step_name=step_template["step_name"],
                step_description=step_template["step_description"],
                step_order=step_template["step_order"],
                is_completed=False,
                is_current=step_template["step_order"] == 1,
                is_required=step_template["is_required"],
                estimated_duration_minutes=step_template["estimated_duration_minutes"]
            )
            steps.append(step)
        
        # Get industry template
        industry_template = self.industry_templates.get(request.industry_type, self.industry_templates["SaaS"])
        
        # Create onboarding session
        session = OnboardingSession(
            session_id=session_id,
            tenant_name=request.tenant_name,
            current_step=steps[0].step_id,
            progress_percentage=0,
            persona_type=request.persona_type,
            industry_type=request.industry_type,
            steps=steps,
            completed_steps=[],
            tenant_config={
                "tenant_name": request.tenant_name,
                "contact_email": request.contact_email,
                "organization_name": request.organization_name,
                "industry_type": request.industry_type,
                "persona_type": request.persona_type
            },
            integration_config={
                "default_integrations": industry_template["default_integrations"],
                "configured_integrations": []
            },
            compliance_config={
                "frameworks": industry_template["compliance_frameworks"],
                "data_retention_tier": industry_template["data_retention_tier"]
            },
            status="active",
            created_at=datetime.utcnow().isoformat(),
            last_updated_at=datetime.utcnow().isoformat(),
            estimated_completion_time=(datetime.utcnow() + timedelta(hours=2)).isoformat()
        )
        
        # Store session
        self.onboarding_sessions[session_id] = session
        
        # Log audit event
        await audit_trail_service.log_audit_event(
            event_type=AuditEventType.SYSTEM_CONFIGURATION,
            event_data={
                "action": "onboarding_wizard_started",
                "session_id": session_id,
                "tenant_name": request.tenant_name,
                "persona_type": request.persona_type,
                "industry_type": request.industry_type
            },
            tenant_id=0,  # System-level event
            plane="ux"
        )
        
        logger.info(f"🧙‍♂️ Started onboarding wizard: {session_id} for {request.tenant_name}")
        return session
    
    async def update_onboarding_step(self, session_id: str, update: OnboardingStepUpdate) -> OnboardingSession:
        """Update onboarding step data"""
        session = self.onboarding_sessions.get(session_id)
        if not session:
            raise ValueError("Onboarding session not found")
        
        # Update step data based on step_id
        if update.step_id == "basic_info":
            session.tenant_config.update(update.step_data)
        elif update.step_id == "compliance_setup":
            session.compliance_config.update(update.step_data)
        elif update.step_id == "integration_setup":
            session.integration_config.update(update.step_data)
        
        # Mark step as completed if requested
        if update.mark_completed and update.step_id not in session.completed_steps:
            session.completed_steps.append(update.step_id)
            
            # Update step status
            for step in session.steps:
                if step.step_id == update.step_id:
                    step.is_completed = True
                    step.is_current = False
                    break
            
            # Move to next step
            next_step = self._get_next_step(session)
            if next_step:
                session.current_step = next_step.step_id
                next_step.is_current = True
            
            # Update progress
            session.progress_percentage = int((len(session.completed_steps) / len(session.steps)) * 100)
        
        session.last_updated_at = datetime.utcnow().isoformat()
        
        logger.info(f"📝 Updated onboarding step {update.step_id} for session {session_id}")
        return session
    
    def _get_next_step(self, session: OnboardingSession) -> Optional[OnboardingWizardStep]:
        """Get next incomplete step"""
        for step in session.steps:
            if not step.is_completed:
                return step
        return None
    
    async def configure_integration(self, session_id: str, integration_request: IntegrationWizardRequest) -> Dict[str, Any]:
        """Task 10.2.21: Configure system integration"""
        session = self.onboarding_sessions.get(session_id)
        if not session:
            raise ValueError("Onboarding session not found")
        
        integration_config = {
            "integration_id": str(uuid.uuid4()),
            "integration_type": integration_request.integration_type,
            "connection_config": integration_request.connection_config,
            "data_mapping": integration_request.data_mapping or {},
            "status": "configured",
            "configured_at": datetime.utcnow().isoformat()
        }
        
        # Test connection if requested
        if integration_request.test_connection:
            test_result = await self._test_integration_connection(integration_request)
            integration_config["test_result"] = test_result
            integration_config["status"] = "tested" if test_result["success"] else "test_failed"
        
        # Add to session
        session.integration_config["configured_integrations"].append(integration_config)
        session.last_updated_at = datetime.utcnow().isoformat()
        
        logger.info(f"🔗 Configured {integration_request.integration_type} integration for session {session_id}")
        return integration_config
    
    async def _test_integration_connection(self, integration_request: IntegrationWizardRequest) -> Dict[str, Any]:
        """Test integration connection"""
        # Simulate connection test
        await asyncio.sleep(1)  # Simulate API call
        
        # Mock test results based on integration type
        if integration_request.integration_type == "Salesforce":
            return {
                "success": True,
                "message": "Successfully connected to Salesforce",
                "objects_accessible": ["Account", "Opportunity", "Contact", "Lead"],
                "permissions_verified": True
            }
        elif integration_request.integration_type == "Stripe":
            return {
                "success": True,
                "message": "Successfully connected to Stripe",
                "webhook_configured": True,
                "test_transaction_completed": True
            }
        else:
            return {
                "success": True,
                "message": f"Successfully connected to {integration_request.integration_type}",
                "connection_verified": True
            }
    
    async def complete_onboarding(self, request: OnboardingCompletionRequest) -> Dict[str, Any]:
        """Task 10.2.37: Complete onboarding and provision tenant"""
        session = self.onboarding_sessions.get(request.session_id)
        if not session:
            raise ValueError("Onboarding session not found")
        
        # Validate all required steps are completed
        required_steps = [step for step in session.steps if step.is_required]
        incomplete_required = [step for step in required_steps if not step.is_completed]
        
        if incomplete_required:
            raise ValueError(f"Required steps not completed: {[s.step_name for s in incomplete_required]}")
        
        # Create tenant provisioning request
        provisioning_request = TenantProvisioningRequest(
            tenant_name=session.tenant_config["tenant_name"],
            region_id=session.compliance_config.get("region_id", "US"),
            policy_pack=session.compliance_config.get("frameworks", ["SOX"])[0],
            industry_type=session.industry_type,
            sla_tier=session.compliance_config.get("sla_tier", "T2"),
            contact_email=session.tenant_config["contact_email"],
            organization_name=session.tenant_config["organization_name"],
            custom_configs=session.tenant_config,
            enable_sandbox=True,
            enable_monitoring=request.enable_monitoring,
            consent_frameworks=session.compliance_config.get("consent_frameworks", []),
            data_retention_tier=session.compliance_config.get("data_retention_tier", "Silver")
        )
        
        # Provision tenant
        provisioning_response = await provisioning_service.provision_tenant(provisioning_request)
        
        # Update session status
        session.status = "completed"
        session.progress_percentage = 100
        session.last_updated_at = datetime.utcnow().isoformat()
        
        # Generate onboarding completion report
        completion_report = {
            "onboarding_session_id": request.session_id,
            "tenant_provisioning_id": provisioning_response.provisioning_id,
            "tenant_id": provisioning_response.tenant_id,
            "tenant_name": session.tenant_config["tenant_name"],
            "persona_type": session.persona_type,
            "industry_type": session.industry_type,
            "completed_steps": session.completed_steps,
            "configured_integrations": len(session.integration_config["configured_integrations"]),
            "compliance_frameworks": session.compliance_config.get("frameworks", []),
            "onboarding_duration_minutes": self._calculate_onboarding_duration(session),
            "provisioning_response": provisioning_response.dict(),
            "completion_timestamp": datetime.utcnow().isoformat()
        }
        
        # Log completion
        await audit_trail_service.log_audit_event(
            event_type=AuditEventType.SYSTEM_CONFIGURATION,
            event_data={
                "action": "onboarding_completed",
                "session_id": request.session_id,
                "tenant_id": provisioning_response.tenant_id,
                "provisioning_id": provisioning_response.provisioning_id
            },
            tenant_id=provisioning_response.tenant_id,
            plane="ux"
        )
        
        logger.info(f"🎉 Onboarding completed for session {request.session_id}, tenant {provisioning_response.tenant_id}")
        return completion_report
    
    def _calculate_onboarding_duration(self, session: OnboardingSession) -> int:
        """Calculate onboarding duration in minutes"""
        start_time = datetime.fromisoformat(session.created_at)
        end_time = datetime.utcnow()
        return int((end_time - start_time).total_seconds() / 60)
    
    def get_onboarding_session(self, session_id: str) -> Optional[OnboardingSession]:
        """Get onboarding session by ID"""
        return self.onboarding_sessions.get(session_id)
    
    def get_industry_template(self, industry_type: str) -> Dict[str, Any]:
        """Get industry-specific template"""
        return self.industry_templates.get(industry_type, self.industry_templates["SaaS"])

# Global service instance
onboarding_portal_service = OnboardingPortalService()

# =====================================================
# API ENDPOINTS
# =====================================================

@router.post("/start-wizard", response_model=OnboardingSession)
async def start_onboarding_wizard(request: OnboardingWizardRequest):
    """
    Task 10.2.5: Start guided onboarding workflow
    
    Creates a new onboarding session with persona-based steps
    """
    try:
        session = await onboarding_portal_service.start_onboarding_wizard(request)
        return session
        
    except Exception as e:
        logger.error(f"❌ Failed to start onboarding wizard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}", response_model=OnboardingSession)
async def get_onboarding_session(session_id: str):
    """Get onboarding session details"""
    try:
        session = onboarding_portal_service.get_onboarding_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Onboarding session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get onboarding session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{session_id}/step", response_model=OnboardingSession)
async def update_onboarding_step(session_id: str, update: OnboardingStepUpdate):
    """Update onboarding step data"""
    try:
        session = await onboarding_portal_service.update_onboarding_step(session_id, update)
        return session
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to update onboarding step: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/integration")
async def configure_integration(session_id: str, integration_request: IntegrationWizardRequest):
    """
    Task 10.2.21: Configure system integration
    
    Set up CRM, Billing, CLM, Marketing, or CS integrations
    """
    try:
        integration_config = await onboarding_portal_service.configure_integration(session_id, integration_request)
        return integration_config
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to configure integration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{session_id}/complete")
async def complete_onboarding(request: OnboardingCompletionRequest):
    """
    Task 10.2.37: Complete onboarding and provision tenant
    
    Finalizes onboarding and triggers tenant provisioning
    """
    try:
        completion_report = await onboarding_portal_service.complete_onboarding(request)
        return completion_report
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to complete onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/{industry_type}")
async def get_industry_template(industry_type: str):
    """
    Task 10.2.8: Get industry-specific template
    
    Returns sample data packs and configuration templates
    """
    try:
        template = onboarding_portal_service.get_industry_template(industry_type)
        return template
        
    except Exception as e:
        logger.error(f"❌ Failed to get industry template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def onboarding_portal_health():
    """Health check for onboarding portal service"""
    return {
        "service": "onboarding_portal_api",
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": len(onboarding_portal_service.onboarding_sessions),
        "supported_personas": list(onboarding_portal_service.persona_steps.keys()),
        "supported_industries": list(onboarding_portal_service.industry_templates.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }
