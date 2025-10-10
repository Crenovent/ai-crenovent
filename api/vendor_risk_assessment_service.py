"""
Task 3.2.38: Third-party model/vendor due-diligence (SLAs, DPAs, sub-processors)
- Vendor risk controlled
- VRA checklist
- Re-cert annually
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json

app = FastAPI(title="RBIA Third-Party Vendor Risk Assessment Service")
logger = logging.getLogger(__name__)

class VendorType(str, Enum):
    MODEL_PROVIDER = "model_provider"      # OpenAI, Anthropic, etc.
    DATA_PROCESSOR = "data_processor"      # Data processing services
    CLOUD_PROVIDER = "cloud_provider"      # AWS, Azure, GCP
    ANALYTICS_PLATFORM = "analytics_platform"  # Analytics and BI tools
    SECURITY_VENDOR = "security_vendor"    # Security and compliance tools
    INTEGRATION_PARTNER = "integration_partner"  # API and integration services

class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AssessmentStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    RENEWAL_REQUIRED = "renewal_required"

class ComplianceFramework(str, Enum):
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    FedRAMP = "fedramp"

class VendorContact(BaseModel):
    name: str
    role: str
    email: str
    phone: Optional[str] = None
    is_primary: bool = False

class SLARequirement(BaseModel):
    requirement_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: str  # "availability", "performance", "support", "security"
    description: str
    required_value: str  # e.g., "99.9%", "<100ms", "24/7"
    actual_value: Optional[str] = None
    compliant: Optional[bool] = None
    notes: Optional[str] = None

class DataProcessingAgreement(BaseModel):
    dpa_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dpa_title: str
    version: str
    
    # Legal details
    effective_date: datetime
    expiration_date: datetime
    governing_law: str
    
    # Data processing specifics
    processing_purposes: List[str] = Field(default_factory=list)
    data_categories: List[str] = Field(default_factory=list)
    data_subjects: List[str] = Field(default_factory=list)
    retention_period: str
    
    # Security and compliance
    security_measures: List[str] = Field(default_factory=list)
    compliance_certifications: List[ComplianceFramework] = Field(default_factory=list)
    
    # Sub-processors
    sub_processors_allowed: bool = True
    sub_processor_approval_required: bool = True
    current_sub_processors: List[str] = Field(default_factory=list)
    
    # Breach notification
    breach_notification_timeline: str = "72 hours"
    breach_notification_contact: str
    
    # Data subject rights
    data_subject_rights_supported: List[str] = Field(default_factory=list)
    
    # Status
    signed: bool = False
    signed_date: Optional[datetime] = None
    reviewed_by_legal: bool = False

class VRAChecklist(BaseModel):
    checklist_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor_name: str
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    assessor: str
    
    # Security Assessment
    security_controls: Dict[str, Any] = Field(default_factory=lambda: {
        "access_controls": {"status": "pending", "score": 0, "notes": ""},
        "data_encryption": {"status": "pending", "score": 0, "notes": ""},
        "network_security": {"status": "pending", "score": 0, "notes": ""},
        "incident_response": {"status": "pending", "score": 0, "notes": ""},
        "vulnerability_management": {"status": "pending", "score": 0, "notes": ""},
        "backup_recovery": {"status": "pending", "score": 0, "notes": ""}
    })
    
    # Compliance Assessment
    compliance_requirements: Dict[str, Any] = Field(default_factory=lambda: {
        "gdpr_compliance": {"status": "pending", "evidence": [], "notes": ""},
        "soc2_certification": {"status": "pending", "evidence": [], "notes": ""},
        "iso27001_certification": {"status": "pending", "evidence": [], "notes": ""},
        "data_residency": {"status": "pending", "evidence": [], "notes": ""},
        "audit_rights": {"status": "pending", "evidence": [], "notes": ""}
    })
    
    # Operational Assessment
    operational_criteria: Dict[str, Any] = Field(default_factory=lambda: {
        "financial_stability": {"status": "pending", "rating": 0, "notes": ""},
        "business_continuity": {"status": "pending", "rating": 0, "notes": ""},
        "support_quality": {"status": "pending", "rating": 0, "notes": ""},
        "technical_capability": {"status": "pending", "rating": 0, "notes": ""},
        "scalability": {"status": "pending", "rating": 0, "notes": ""}
    })
    
    # AI/ML Specific Assessment (for model providers)
    ai_ml_criteria: Dict[str, Any] = Field(default_factory=lambda: {
        "model_transparency": {"status": "pending", "score": 0, "notes": ""},
        "bias_testing": {"status": "pending", "score": 0, "notes": ""},
        "explainability": {"status": "pending", "score": 0, "notes": ""},
        "data_provenance": {"status": "pending", "score": 0, "notes": ""},
        "model_governance": {"status": "pending", "score": 0, "notes": ""}
    })
    
    # Overall Assessment
    overall_risk_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    risk_level: Optional[RiskLevel] = None
    recommendation: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    
    # Completion tracking
    completed_sections: List[str] = Field(default_factory=list)
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0)

class VendorAssessment(BaseModel):
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor_id: str
    assessment_type: str = "annual_review"  # initial, annual_review, incident_driven
    
    # Assessment details
    initiated_by: str
    initiated_date: datetime = Field(default_factory=datetime.utcnow)
    target_completion_date: datetime
    actual_completion_date: Optional[datetime] = None
    
    # Status tracking
    status: AssessmentStatus = AssessmentStatus.PENDING
    current_phase: str = "initiation"  # initiation, documentation_review, assessment, approval
    
    # Assessment components
    vra_checklist: VRAChecklist
    sla_requirements: List[SLARequirement] = Field(default_factory=list)
    dpa_review: Optional[DataProcessingAgreement] = None
    
    # Results
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    
    # Approval
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None
    approval_conditions: List[str] = Field(default_factory=list)
    
    # Next review
    next_review_date: Optional[datetime] = None
    
    # Tenant context
    tenant_id: str

class ThirdPartyVendor(BaseModel):
    vendor_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor_name: str
    vendor_type: VendorType
    
    # Basic information
    description: str
    website: str
    headquarters_location: str
    data_processing_locations: List[str] = Field(default_factory=list)
    
    # Contacts
    contacts: List[VendorContact] = Field(default_factory=list)
    
    # Services provided
    services_description: str
    models_provided: List[str] = Field(default_factory=list)  # For model providers
    data_types_processed: List[str] = Field(default_factory=list)
    
    # Current status
    current_risk_level: RiskLevel = RiskLevel.MEDIUM
    status: str = "active"  # active, inactive, suspended, terminated
    onboarding_date: datetime
    last_assessment_date: Optional[datetime] = None
    next_assessment_due: Optional[datetime] = None
    
    # Compliance and certifications
    certifications: List[ComplianceFramework] = Field(default_factory=list)
    certification_expiry_dates: Dict[str, datetime] = Field(default_factory=dict)
    
    # Contracts and agreements
    master_agreement_signed: bool = False
    dpa_signed: bool = False
    sla_agreed: bool = False
    
    # Assessment history
    assessments: List[str] = Field(default_factory=list)  # Assessment IDs
    
    # Metadata
    created_by: str
    tenant_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with proper database in production)
vendors: Dict[str, ThirdPartyVendor] = {}
assessments: Dict[str, VendorAssessment] = {}
vra_checklists: Dict[str, VRAChecklist] = {}

def _calculate_risk_score(checklist: VRAChecklist) -> tuple[float, RiskLevel]:
    """Calculate overall risk score from VRA checklist."""
    
    total_score = 0.0
    total_weight = 0.0
    
    # Security controls (40% weight)
    security_scores = [ctrl.get("score", 0) for ctrl in checklist.security_controls.values()]
    if security_scores:
        security_avg = sum(security_scores) / len(security_scores)
        total_score += security_avg * 0.4
        total_weight += 0.4
    
    # Compliance requirements (30% weight)
    compliance_scores = []
    for req in checklist.compliance_requirements.values():
        if req.get("status") == "compliant":
            compliance_scores.append(100)
        elif req.get("status") == "non_compliant":
            compliance_scores.append(0)
        else:
            compliance_scores.append(50)  # Pending/partial
    
    if compliance_scores:
        compliance_avg = sum(compliance_scores) / len(compliance_scores)
        total_score += compliance_avg * 0.3
        total_weight += 0.3
    
    # Operational criteria (20% weight)
    operational_scores = [crit.get("rating", 0) * 20 for crit in checklist.operational_criteria.values()]  # Convert 1-5 to 0-100
    if operational_scores:
        operational_avg = sum(operational_scores) / len(operational_scores)
        total_score += operational_avg * 0.2
        total_weight += 0.2
    
    # AI/ML criteria (10% weight)
    ai_scores = [crit.get("score", 0) for crit in checklist.ai_ml_criteria.values()]
    if ai_scores:
        ai_avg = sum(ai_scores) / len(ai_scores)
        total_score += ai_avg * 0.1
        total_weight += 0.1
    
    final_score = total_score / total_weight if total_weight > 0 else 0.0
    
    # Determine risk level (inverted - higher score = lower risk)
    if final_score >= 80:
        risk_level = RiskLevel.LOW
    elif final_score >= 60:
        risk_level = RiskLevel.MEDIUM
    elif final_score >= 40:
        risk_level = RiskLevel.HIGH
    else:
        risk_level = RiskLevel.CRITICAL
    
    return final_score, risk_level

def _generate_default_sla_requirements(vendor_type: VendorType) -> List[SLARequirement]:
    """Generate default SLA requirements based on vendor type."""
    
    common_requirements = [
        SLARequirement(
            category="availability",
            description="Service uptime guarantee",
            required_value="99.9%"
        ),
        SLARequirement(
            category="support",
            description="Response time for critical issues",
            required_value="4 hours"
        ),
        SLARequirement(
            category="security",
            description="Security incident notification",
            required_value="24 hours"
        )
    ]
    
    if vendor_type == VendorType.MODEL_PROVIDER:
        common_requirements.extend([
            SLARequirement(
                category="performance",
                description="API response time (95th percentile)",
                required_value="<2 seconds"
            ),
            SLARequirement(
                category="performance",
                description="Model accuracy degradation threshold",
                required_value="<5% from baseline"
            )
        ])
    elif vendor_type == VendorType.CLOUD_PROVIDER:
        common_requirements.extend([
            SLARequirement(
                category="availability",
                description="Compute service availability",
                required_value="99.95%"
            ),
            SLARequirement(
                category="performance",
                description="Network latency",
                required_value="<50ms regional"
            )
        ])
    
    return common_requirements

@app.post("/vendors", response_model=ThirdPartyVendor)
async def create_vendor(vendor: ThirdPartyVendor):
    """Register a new third-party vendor."""
    
    vendor.updated_at = datetime.utcnow()
    
    # Set next assessment due date (1 year from onboarding)
    vendor.next_assessment_due = vendor.onboarding_date + timedelta(days=365)
    
    vendors[vendor.vendor_id] = vendor
    
    logger.info(f"Registered vendor {vendor.vendor_id}: {vendor.vendor_name}")
    return vendor

@app.post("/vendors/{vendor_id}/assessments", response_model=VendorAssessment)
async def initiate_vendor_assessment(
    vendor_id: str,
    assessment_type: str,
    initiated_by: str,
    target_completion_days: int = 30
):
    """Initiate a vendor risk assessment."""
    
    if vendor_id not in vendors:
        raise HTTPException(status_code=404, detail="Vendor not found")
    
    vendor = vendors[vendor_id]
    
    # Create VRA checklist
    vra_checklist = VRAChecklist(
        vendor_name=vendor.vendor_name,
        assessor=initiated_by
    )
    
    # Generate default SLA requirements
    sla_requirements = _generate_default_sla_requirements(vendor.vendor_type)
    
    assessment = VendorAssessment(
        vendor_id=vendor_id,
        assessment_type=assessment_type,
        initiated_by=initiated_by,
        target_completion_date=datetime.utcnow() + timedelta(days=target_completion_days),
        vra_checklist=vra_checklist,
        sla_requirements=sla_requirements,
        tenant_id=vendor.tenant_id
    )
    
    assessments[assessment.assessment_id] = assessment
    vra_checklists[vra_checklist.checklist_id] = vra_checklist
    
    # Add to vendor's assessment history
    vendor.assessments.append(assessment.assessment_id)
    
    logger.info(f"Initiated assessment {assessment.assessment_id} for vendor {vendor_id}")
    return assessment

@app.post("/assessments/{assessment_id}/checklist/update")
async def update_vra_checklist(
    assessment_id: str,
    section: str,
    criteria: str,
    status: str,
    score: Optional[int] = None,
    rating: Optional[int] = None,
    notes: Optional[str] = None,
    evidence: Optional[List[str]] = None
):
    """Update a specific criteria in the VRA checklist."""
    
    if assessment_id not in assessments:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    assessment = assessments[assessment_id]
    checklist = assessment.vra_checklist
    
    # Update the specific criteria
    section_data = getattr(checklist, section, {})
    if criteria in section_data:
        section_data[criteria]["status"] = status
        if score is not None:
            section_data[criteria]["score"] = score
        if rating is not None:
            section_data[criteria]["rating"] = rating
        if notes:
            section_data[criteria]["notes"] = notes
        if evidence:
            section_data[criteria]["evidence"] = evidence
    
    # Calculate completion percentage
    total_criteria = 0
    completed_criteria = 0
    
    for section_name in ["security_controls", "compliance_requirements", "operational_criteria", "ai_ml_criteria"]:
        section_data = getattr(checklist, section_name, {})
        for criteria_data in section_data.values():
            total_criteria += 1
            if criteria_data.get("status") not in ["pending", ""]:
                completed_criteria += 1
    
    checklist.completion_percentage = (completed_criteria / total_criteria * 100) if total_criteria > 0 else 0
    
    # Update completed sections
    if section not in checklist.completed_sections and checklist.completion_percentage > 0:
        checklist.completed_sections.append(section)
    
    # Calculate overall risk score if assessment is sufficiently complete
    if checklist.completion_percentage >= 80:
        risk_score, risk_level = _calculate_risk_score(checklist)
        checklist.overall_risk_score = risk_score
        checklist.risk_level = risk_level
    
    logger.info(f"Updated VRA checklist for assessment {assessment_id}: {section}.{criteria}")
    
    return {
        "status": "updated",
        "assessment_id": assessment_id,
        "completion_percentage": checklist.completion_percentage,
        "overall_risk_score": checklist.overall_risk_score,
        "risk_level": checklist.risk_level.value if checklist.risk_level else None
    }

@app.post("/assessments/{assessment_id}/sla/validate")
async def validate_sla_requirements(
    assessment_id: str,
    requirement_id: str,
    actual_value: str,
    compliant: bool,
    notes: Optional[str] = None
):
    """Validate SLA requirements against actual vendor performance."""
    
    if assessment_id not in assessments:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    assessment = assessments[assessment_id]
    
    # Find and update the SLA requirement
    for req in assessment.sla_requirements:
        if req.requirement_id == requirement_id:
            req.actual_value = actual_value
            req.compliant = compliant
            if notes:
                req.notes = notes
            break
    else:
        raise HTTPException(status_code=404, detail="SLA requirement not found")
    
    logger.info(f"Validated SLA requirement {requirement_id} for assessment {assessment_id}: {'compliant' if compliant else 'non-compliant'}")
    
    return {
        "status": "validated",
        "requirement_id": requirement_id,
        "compliant": compliant,
        "actual_value": actual_value
    }

@app.post("/assessments/{assessment_id}/dpa", response_model=DataProcessingAgreement)
async def create_dpa_review(assessment_id: str, dpa: DataProcessingAgreement):
    """Create or update DPA review for an assessment."""
    
    if assessment_id not in assessments:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    assessment = assessments[assessment_id]
    assessment.dpa_review = dpa
    
    logger.info(f"Created DPA review for assessment {assessment_id}")
    return dpa

@app.post("/assessments/{assessment_id}/complete")
async def complete_assessment(
    assessment_id: str,
    findings: List[str],
    recommendations: List[str],
    action_items: List[str],
    approved_by: str,
    approval_conditions: List[str] = None
):
    """Complete a vendor assessment."""
    
    if assessment_id not in assessments:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    assessment = assessments[assessment_id]
    
    assessment.status = AssessmentStatus.COMPLETED
    assessment.actual_completion_date = datetime.utcnow()
    assessment.findings = findings
    assessment.recommendations = recommendations
    assessment.action_items = action_items
    assessment.approved_by = approved_by
    assessment.approval_date = datetime.utcnow()
    assessment.approval_conditions = approval_conditions or []
    
    # Set next review date (1 year from completion)
    assessment.next_review_date = assessment.actual_completion_date + timedelta(days=365)
    
    # Update vendor status
    vendor = vendors[assessment.vendor_id]
    vendor.last_assessment_date = assessment.actual_completion_date
    vendor.next_assessment_due = assessment.next_review_date
    vendor.current_risk_level = assessment.vra_checklist.risk_level or RiskLevel.MEDIUM
    
    logger.info(f"Completed assessment {assessment_id} for vendor {assessment.vendor_id}")
    
    return {
        "status": "completed",
        "assessment_id": assessment_id,
        "completion_date": assessment.actual_completion_date,
        "risk_level": assessment.vra_checklist.risk_level.value if assessment.vra_checklist.risk_level else None,
        "next_review_date": assessment.next_review_date,
        "findings_count": len(findings),
        "action_items_count": len(action_items)
    }

@app.get("/vendors", response_model=List[ThirdPartyVendor])
async def list_vendors(
    tenant_id: Optional[str] = None,
    vendor_type: Optional[VendorType] = None,
    risk_level: Optional[RiskLevel] = None,
    status: Optional[str] = None
):
    """List vendors with optional filters."""
    
    filtered_vendors = list(vendors.values())
    
    if tenant_id:
        filtered_vendors = [v for v in filtered_vendors if v.tenant_id == tenant_id]
    
    if vendor_type:
        filtered_vendors = [v for v in filtered_vendors if v.vendor_type == vendor_type]
    
    if risk_level:
        filtered_vendors = [v for v in filtered_vendors if v.current_risk_level == risk_level]
    
    if status:
        filtered_vendors = [v for v in filtered_vendors if v.status == status]
    
    return filtered_vendors

@app.get("/vendors/assessments-due")
async def get_assessments_due(
    tenant_id: Optional[str] = None,
    days_ahead: int = 30
):
    """Get vendors with assessments due within specified days."""
    
    cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
    
    due_vendors = [
        v for v in vendors.values()
        if v.next_assessment_due and v.next_assessment_due <= cutoff_date
        and (tenant_id is None or v.tenant_id == tenant_id)
    ]
    
    # Sort by due date
    due_vendors.sort(key=lambda x: x.next_assessment_due or datetime.max)
    
    return {
        "vendors_due": due_vendors,
        "count": len(due_vendors),
        "cutoff_date": cutoff_date
    }

@app.get("/assessments/{assessment_id}/report")
async def generate_assessment_report(assessment_id: str):
    """Generate a comprehensive assessment report."""
    
    if assessment_id not in assessments:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    assessment = assessments[assessment_id]
    vendor = vendors[assessment.vendor_id]
    checklist = assessment.vra_checklist
    
    # Calculate SLA compliance
    total_sla = len(assessment.sla_requirements)
    compliant_sla = len([req for req in assessment.sla_requirements if req.compliant])
    sla_compliance_rate = (compliant_sla / total_sla * 100) if total_sla > 0 else 0
    
    # Generate summary
    report = {
        "assessment_id": assessment_id,
        "vendor_name": vendor.vendor_name,
        "vendor_type": vendor.vendor_type.value,
        "assessment_date": assessment.initiated_date,
        "completion_date": assessment.actual_completion_date,
        "assessor": checklist.assessor,
        
        # Risk assessment
        "overall_risk_score": checklist.overall_risk_score,
        "risk_level": checklist.risk_level.value if checklist.risk_level else None,
        "completion_percentage": checklist.completion_percentage,
        
        # SLA compliance
        "sla_compliance_rate": sla_compliance_rate,
        "sla_requirements_total": total_sla,
        "sla_requirements_compliant": compliant_sla,
        
        # Findings and recommendations
        "findings": assessment.findings,
        "recommendations": assessment.recommendations,
        "action_items": assessment.action_items,
        
        # DPA status
        "dpa_signed": assessment.dpa_review.signed if assessment.dpa_review else False,
        "dpa_reviewed_by_legal": assessment.dpa_review.reviewed_by_legal if assessment.dpa_review else False,
        
        # Next steps
        "next_review_date": assessment.next_review_date,
        "approval_conditions": assessment.approval_conditions,
        
        # Detailed checklist results
        "security_controls": checklist.security_controls,
        "compliance_requirements": checklist.compliance_requirements,
        "operational_criteria": checklist.operational_criteria,
        "ai_ml_criteria": checklist.ai_ml_criteria
    }
    
    return report

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Third-Party Vendor Risk Assessment Service", "task": "3.2.38"}
