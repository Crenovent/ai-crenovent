# Industry-specific Governance Overlays
# Tasks 8.4-T01 to T30: Industry overlays for SaaS, Banking, Insurance, Healthcare

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Industry(Enum):
    """Supported industries"""
    SAAS = "saas"
    BANKING = "banking"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    ECOMMERCE = "ecommerce"
    MANUFACTURING = "manufacturing"

class ComplianceFramework(Enum):
    """Compliance frameworks by industry"""
    # SaaS
    SOX_SAAS = "sox_saas"
    GDPR_SAAS = "gdpr_saas"
    CCPA_SAAS = "ccpa_saas"
    
    # Banking
    RBI = "rbi"
    BASEL_III = "basel_iii"
    SOX_BANKING = "sox_banking"
    GDPR_BANKING = "gdpr_banking"
    PCI_DSS = "pci_dss"
    
    # Insurance
    IRDAI = "irdai"
    SOLVENCY_II = "solvency_ii"
    GDPR_INSURANCE = "gdpr_insurance"
    
    # Healthcare
    HIPAA = "hipaa"
    FDA_21CFR11 = "fda_21cfr11"
    GDPR_HEALTHCARE = "gdpr_healthcare"
    
    # Cross-industry
    GDPR = "gdpr"
    DPDP = "dpdp"
    SOX = "sox"
    ISO27001 = "iso27001"

class GovernanceControl(Enum):
    """Types of governance controls"""
    DATA_RETENTION = "data_retention"
    ACCESS_CONTROL = "access_control"
    AUDIT_LOGGING = "audit_logging"
    ENCRYPTION = "encryption"
    SEGREGATION_OF_DUTIES = "segregation_of_duties"
    APPROVAL_WORKFLOW = "approval_workflow"
    RISK_ASSESSMENT = "risk_assessment"
    INCIDENT_RESPONSE = "incident_response"
    BUSINESS_CONTINUITY = "business_continuity"
    VENDOR_MANAGEMENT = "vendor_management"

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    # All required fields first
    requirement_id: str
    requirement_name: str
    description: str
    control_type: GovernanceControl
    framework: ComplianceFramework
    industry: Industry
    
    # All optional fields with defaults
    mandatory: bool = True
    implementation_guidance: str = ""
    validation_criteria: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high, critical
    non_compliance_impact: str = ""
    effective_date: Optional[str] = None
    review_frequency_days: int = 365
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['control_type'] = self.control_type.value
        data['framework'] = self.framework.value
        data['industry'] = self.industry.value
        return data

@dataclass
class IndustryOverlay:
    """Complete industry-specific governance overlay"""
    overlay_id: str
    industry: Industry
    overlay_name: str
    description: str
    
    # Compliance frameworks
    applicable_frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    # Requirements
    requirements: List[ComplianceRequirement] = field(default_factory=list)
    
    # Policy overrides
    policy_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Workflow restrictions
    restricted_workflows: List[str] = field(default_factory=list)
    required_approvals: Dict[str, List[str]] = field(default_factory=dict)  # workflow_type -> approver_roles
    
    # Data handling
    data_classification_rules: Dict[str, str] = field(default_factory=dict)
    retention_policies: Dict[str, int] = field(default_factory=dict)  # data_type -> retention_days
    
    # Metadata
    version: str = "1.0.0"
    created_by: str = "system"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['industry'] = self.industry.value
        data['applicable_frameworks'] = [f.value for f in self.applicable_frameworks]
        data['requirements'] = [req.to_dict() for req in self.requirements]
        return data

class IndustryGovernanceManager:
    """
    Industry-specific Governance Overlay Manager
    Tasks 8.4-T01 to T30: Industry overlays, compliance frameworks, governance controls
    """
    
    def __init__(self):
        self.overlays: Dict[str, IndustryOverlay] = {}
        self.framework_mappings: Dict[ComplianceFramework, List[ComplianceRequirement]] = {}
        
        # Initialize industry overlays
        self._initialize_industry_overlays()
    
    def _initialize_industry_overlays(self) -> None:
        """Initialize default industry governance overlays"""
        
        # SaaS Industry Overlay
        saas_overlay = self._create_saas_overlay()
        self.overlays[saas_overlay.overlay_id] = saas_overlay
        
        # Banking Industry Overlay
        banking_overlay = self._create_banking_overlay()
        self.overlays[banking_overlay.overlay_id] = banking_overlay
        
        # Insurance Industry Overlay
        insurance_overlay = self._create_insurance_overlay()
        self.overlays[insurance_overlay.overlay_id] = insurance_overlay
        
        # Healthcare Industry Overlay
        healthcare_overlay = self._create_healthcare_overlay()
        self.overlays[healthcare_overlay.overlay_id] = healthcare_overlay
        
        logger.info(f"✅ Initialized {len(self.overlays)} industry governance overlays")
    
    def _create_saas_overlay(self) -> IndustryOverlay:
        """Create SaaS industry governance overlay"""
        
        # SaaS-specific compliance requirements
        saas_requirements = [
            ComplianceRequirement(
                requirement_id="saas_sox_revenue_recognition",
                requirement_name="SOX Revenue Recognition Controls",
                description="Ensure accurate revenue recognition and financial reporting",
                control_type=GovernanceControl.APPROVAL_WORKFLOW,
                framework=ComplianceFramework.SOX_SAAS,
                industry=Industry.SAAS,
                mandatory=True,
                implementation_guidance="All revenue-impacting workflows must have dual approval",
                validation_criteria=[
                    "Revenue workflows require CFO or Finance Manager approval",
                    "All revenue adjustments must be logged and auditable",
                    "Segregation of duties between revenue recognition and billing"
                ],
                risk_level="high",
                non_compliance_impact="SOX violations, financial restatements, regulatory penalties"
            ),
            
            ComplianceRequirement(
                requirement_id="saas_gdpr_customer_data",
                requirement_name="GDPR Customer Data Protection",
                description="Protect customer personal data according to GDPR requirements",
                control_type=GovernanceControl.DATA_RETENTION,
                framework=ComplianceFramework.GDPR_SAAS,
                industry=Industry.SAAS,
                mandatory=True,
                implementation_guidance="Implement data minimization and retention policies",
                validation_criteria=[
                    "Customer data retention limited to business necessity",
                    "Right to erasure implemented for customer requests",
                    "Data processing consent tracked and auditable"
                ],
                risk_level="critical",
                non_compliance_impact="GDPR fines up to 4% of annual revenue"
            ),
            
            ComplianceRequirement(
                requirement_id="saas_subscription_lifecycle",
                requirement_name="SaaS Subscription Lifecycle Controls",
                description="Ensure proper controls over subscription lifecycle management",
                control_type=GovernanceControl.APPROVAL_WORKFLOW,
                framework=ComplianceFramework.SOX_SAAS,
                industry=Industry.SAAS,
                mandatory=True,
                implementation_guidance="Implement controls for subscription changes affecting revenue",
                validation_criteria=[
                    "Subscription upgrades/downgrades require approval",
                    "Churn and retention workflows must be auditable",
                    "Revenue impact of subscription changes must be tracked"
                ],
                risk_level="medium",
                non_compliance_impact="Revenue recognition errors, audit findings"
            )
        ]
        
        return IndustryOverlay(
            overlay_id="saas_governance_overlay_v1",
            industry=Industry.SAAS,
            overlay_name="SaaS Governance Overlay",
            description="Governance overlay for SaaS companies focusing on revenue recognition and customer data protection",
            applicable_frameworks=[
                ComplianceFramework.SOX_SAAS,
                ComplianceFramework.GDPR_SAAS,
                ComplianceFramework.CCPA_SAAS
            ],
            requirements=saas_requirements,
            policy_overrides={
                "max_customer_data_retention_days": 2555,  # 7 years for SOX
                "require_dual_approval_for_revenue": True,
                "enable_gdpr_right_to_erasure": True
            },
            required_approvals={
                "revenue_recognition": ["cfo", "finance_manager"],
                "customer_data_processing": ["compliance_manager", "dpo"],
                "subscription_lifecycle": ["revops_manager", "finance_manager"]
            },
            data_classification_rules={
                "customer_pii": "sensitive",
                "financial_data": "confidential",
                "usage_analytics": "internal"
            },
            retention_policies={
                "customer_pii": 2555,  # 7 years
                "financial_records": 2555,  # 7 years for SOX
                "audit_logs": 2555,  # 7 years
                "usage_data": 1095  # 3 years
            }
        )
    
    def _create_banking_overlay(self) -> IndustryOverlay:
        """Create Banking industry governance overlay"""
        
        banking_requirements = [
            ComplianceRequirement(
                requirement_id="banking_rbi_loan_sanction",
                requirement_name="RBI Loan Sanction Controls",
                description="Ensure compliance with RBI guidelines for loan sanctioning",
                control_type=GovernanceControl.APPROVAL_WORKFLOW,
                framework=ComplianceFramework.RBI,
                industry=Industry.BANKING,
                mandatory=True,
                implementation_guidance="Implement multi-level approval for loan sanctions",
                validation_criteria=[
                    "Loan sanctions require credit committee approval",
                    "All loan decisions must be documented and auditable",
                    "Risk assessment mandatory for all loan applications"
                ],
                risk_level="critical",
                non_compliance_impact="RBI penalties, license suspension"
            ),
            
            ComplianceRequirement(
                requirement_id="banking_basel_capital_adequacy",
                requirement_name="Basel III Capital Adequacy",
                description="Maintain capital adequacy ratios as per Basel III norms",
                control_type=GovernanceControl.RISK_ASSESSMENT,
                framework=ComplianceFramework.BASEL_III,
                industry=Industry.BANKING,
                mandatory=True,
                implementation_guidance="Monitor and report capital adequacy ratios",
                validation_criteria=[
                    "Capital adequacy ratio maintained above regulatory minimum",
                    "Risk-weighted assets calculated accurately",
                    "Regular stress testing performed"
                ],
                risk_level="critical",
                non_compliance_impact="Regulatory action, capital infusion requirements"
            ),
            
            ComplianceRequirement(
                requirement_id="banking_pci_dss_payments",
                requirement_name="PCI DSS Payment Security",
                description="Secure payment card data according to PCI DSS standards",
                control_type=GovernanceControl.ENCRYPTION,
                framework=ComplianceFramework.PCI_DSS,
                industry=Industry.BANKING,
                mandatory=True,
                implementation_guidance="Encrypt all payment card data at rest and in transit",
                validation_criteria=[
                    "Payment card data encrypted with approved algorithms",
                    "Access to card data restricted and logged",
                    "Regular security assessments performed"
                ],
                risk_level="critical",
                non_compliance_impact="Loss of payment processing privileges, fines"
            )
        ]
        
        return IndustryOverlay(
            overlay_id="banking_governance_overlay_v1",
            industry=Industry.BANKING,
            overlay_name="Banking Governance Overlay",
            description="Governance overlay for banking institutions with RBI, Basel III, and PCI DSS compliance",
            applicable_frameworks=[
                ComplianceFramework.RBI,
                ComplianceFramework.BASEL_III,
                ComplianceFramework.SOX_BANKING,
                ComplianceFramework.PCI_DSS,
                ComplianceFramework.GDPR_BANKING
            ],
            requirements=banking_requirements,
            policy_overrides={
                "max_loan_amount_single_approval": 1000000,  # 10 lakh INR
                "require_credit_committee_approval": True,
                "enable_real_time_fraud_detection": True,
                "mandatory_kyc_verification": True
            },
            required_approvals={
                "loan_sanction": ["credit_manager", "branch_manager", "regional_head"],
                "large_transactions": ["branch_manager", "compliance_officer"],
                "policy_changes": ["compliance_head", "cro"]
            },
            data_classification_rules={
                "customer_financial_data": "confidential",
                "payment_card_data": "restricted",
                "loan_application_data": "confidential",
                "kyc_documents": "restricted"
            },
            retention_policies={
                "loan_records": 3650,  # 10 years
                "transaction_records": 3650,  # 10 years
                "kyc_documents": 2555,  # 7 years
                "audit_trails": 3650  # 10 years
            }
        )
    
    def _create_insurance_overlay(self) -> IndustryOverlay:
        """Create Insurance industry governance overlay"""
        
        insurance_requirements = [
            ComplianceRequirement(
                requirement_id="insurance_irdai_solvency",
                requirement_name="IRDAI Solvency Requirements",
                description="Maintain solvency ratios as per IRDAI regulations",
                control_type=GovernanceControl.RISK_ASSESSMENT,
                framework=ComplianceFramework.IRDAI,
                industry=Industry.INSURANCE,
                mandatory=True,
                implementation_guidance="Monitor solvency ratios and maintain adequate reserves",
                validation_criteria=[
                    "Solvency ratio maintained above regulatory minimum",
                    "Regular actuarial valuations performed",
                    "Risk management framework implemented"
                ],
                risk_level="critical",
                non_compliance_impact="IRDAI penalties, business restrictions"
            ),
            
            ComplianceRequirement(
                requirement_id="insurance_claims_processing",
                requirement_name="Claims Processing Controls",
                description="Ensure fair and timely claims processing",
                control_type=GovernanceControl.APPROVAL_WORKFLOW,
                framework=ComplianceFramework.IRDAI,
                industry=Industry.INSURANCE,
                mandatory=True,
                implementation_guidance="Implement automated claims processing with manual oversight",
                validation_criteria=[
                    "Claims processed within regulatory timelines",
                    "Fraud detection mechanisms in place",
                    "Claims settlement ratios monitored"
                ],
                risk_level="high",
                non_compliance_impact="Customer complaints, regulatory action"
            ),
            
            ComplianceRequirement(
                requirement_id="insurance_underwriting_controls",
                requirement_name="Underwriting Risk Controls",
                description="Implement proper underwriting controls and risk assessment",
                control_type=GovernanceControl.RISK_ASSESSMENT,
                framework=ComplianceFramework.IRDAI,
                industry=Industry.INSURANCE,
                mandatory=True,
                implementation_guidance="Use actuarial models for risk assessment and pricing",
                validation_criteria=[
                    "Underwriting guidelines followed consistently",
                    "Risk-based pricing implemented",
                    "Regular review of underwriting performance"
                ],
                risk_level="high",
                non_compliance_impact="Adverse selection, profitability issues"
            )
        ]
        
        return IndustryOverlay(
            overlay_id="insurance_governance_overlay_v1",
            industry=Industry.INSURANCE,
            overlay_name="Insurance Governance Overlay",
            description="Governance overlay for insurance companies with IRDAI and Solvency II compliance",
            applicable_frameworks=[
                ComplianceFramework.IRDAI,
                ComplianceFramework.SOLVENCY_II,
                ComplianceFramework.GDPR_INSURANCE
            ],
            requirements=insurance_requirements,
            policy_overrides={
                "max_claims_auto_approval": 100000,  # 1 lakh INR
                "require_actuarial_review": True,
                "enable_fraud_detection": True,
                "mandatory_medical_underwriting_threshold": 500000  # 5 lakh INR
            },
            required_approvals={
                "large_claims": ["claims_manager", "medical_officer", "cfo"],
                "underwriting_exceptions": ["chief_underwriter", "risk_manager"],
                "policy_launches": ["actuarial_head", "compliance_head", "ceo"]
            },
            data_classification_rules={
                "medical_records": "restricted",
                "claims_data": "confidential",
                "actuarial_models": "confidential",
                "customer_pii": "sensitive"
            },
            retention_policies={
                "policy_records": 3650,  # 10 years
                "claims_records": 3650,  # 10 years
                "medical_records": 2555,  # 7 years
                "audit_trails": 3650  # 10 years
            }
        )
    
    def _create_healthcare_overlay(self) -> IndustryOverlay:
        """Create Healthcare industry governance overlay"""
        
        healthcare_requirements = [
            ComplianceRequirement(
                requirement_id="healthcare_hipaa_phi",
                requirement_name="HIPAA PHI Protection",
                description="Protect patient health information according to HIPAA requirements",
                control_type=GovernanceControl.ACCESS_CONTROL,
                framework=ComplianceFramework.HIPAA,
                industry=Industry.HEALTHCARE,
                mandatory=True,
                implementation_guidance="Implement strict access controls and encryption for PHI",
                validation_criteria=[
                    "PHI access limited to authorized personnel only",
                    "All PHI access logged and monitored",
                    "Patient consent obtained for data use"
                ],
                risk_level="critical",
                non_compliance_impact="HIPAA violations, fines up to $1.5M per incident"
            ),
            
            ComplianceRequirement(
                requirement_id="healthcare_fda_21cfr11",
                requirement_name="FDA 21 CFR Part 11 Electronic Records",
                description="Ensure electronic records and signatures meet FDA requirements",
                control_type=GovernanceControl.AUDIT_LOGGING,
                framework=ComplianceFramework.FDA_21CFR11,
                industry=Industry.HEALTHCARE,
                mandatory=True,
                implementation_guidance="Implement audit trails and electronic signature validation",
                validation_criteria=[
                    "Electronic records are tamper-evident",
                    "Electronic signatures are legally binding",
                    "Complete audit trails maintained"
                ],
                risk_level="critical",
                non_compliance_impact="FDA enforcement actions, product recalls"
            ),
            
            ComplianceRequirement(
                requirement_id="healthcare_patient_safety",
                requirement_name="Patient Safety Controls",
                description="Implement controls to ensure patient safety in healthcare workflows",
                control_type=GovernanceControl.APPROVAL_WORKFLOW,
                framework=ComplianceFramework.HIPAA,
                industry=Industry.HEALTHCARE,
                mandatory=True,
                implementation_guidance="Require medical professional approval for patient-impacting workflows",
                validation_criteria=[
                    "Medical decisions require licensed professional approval",
                    "Patient safety checks integrated into workflows",
                    "Adverse events tracked and reported"
                ],
                risk_level="critical",
                non_compliance_impact="Patient harm, malpractice liability"
            )
        ]
        
        return IndustryOverlay(
            overlay_id="healthcare_governance_overlay_v1",
            industry=Industry.HEALTHCARE,
            overlay_name="Healthcare Governance Overlay",
            description="Governance overlay for healthcare organizations with HIPAA and FDA compliance",
            applicable_frameworks=[
                ComplianceFramework.HIPAA,
                ComplianceFramework.FDA_21CFR11,
                ComplianceFramework.GDPR_HEALTHCARE
            ],
            requirements=healthcare_requirements,
            policy_overrides={
                "phi_access_logging_required": True,
                "require_patient_consent": True,
                "enable_audit_trails": True,
                "phi_encryption_mandatory": True
            },
            required_approvals={
                "phi_access": ["medical_director", "privacy_officer"],
                "clinical_workflows": ["medical_director", "chief_medical_officer"],
                "research_data_use": ["irb_chair", "privacy_officer", "research_director"]
            },
            data_classification_rules={
                "phi": "restricted",
                "clinical_data": "confidential",
                "research_data": "confidential",
                "administrative_data": "internal"
            },
            retention_policies={
                "medical_records": 2555,  # 7 years minimum
                "clinical_trial_data": 5475,  # 15 years
                "audit_logs": 2555,  # 7 years
                "phi_access_logs": 2555  # 7 years
            }
        )
    
    def get_overlay(self, industry: Industry) -> Optional[IndustryOverlay]:
        """Get governance overlay for industry"""
        
        overlay_id = f"{industry.value}_governance_overlay_v1"
        return self.overlays.get(overlay_id)
    
    def get_applicable_requirements(
        self,
        industry: Industry,
        workflow_type: Optional[str] = None,
        data_types: Optional[List[str]] = None
    ) -> List[ComplianceRequirement]:
        """Get applicable compliance requirements for industry and context"""
        
        overlay = self.get_overlay(industry)
        if not overlay:
            return []
        
        applicable_requirements = overlay.requirements.copy()
        
        # Filter by workflow type if specified
        if workflow_type:
            # This would be more sophisticated in production
            filtered_requirements = []
            for req in applicable_requirements:
                if workflow_type in req.description.lower() or workflow_type in req.requirement_name.lower():
                    filtered_requirements.append(req)
            
            if filtered_requirements:
                applicable_requirements = filtered_requirements
        
        # Filter by data types if specified
        if data_types:
            # This would be more sophisticated in production
            filtered_requirements = []
            for req in applicable_requirements:
                for data_type in data_types:
                    if data_type in req.description.lower() or data_type in req.requirement_name.lower():
                        filtered_requirements.append(req)
                        break
            
            if filtered_requirements:
                applicable_requirements = filtered_requirements
        
        return applicable_requirements
    
    def validate_compliance(
        self,
        industry: Industry,
        workflow_data: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate compliance against industry requirements"""
        
        overlay = self.get_overlay(industry)
        if not overlay:
            return {
                'compliant': True,
                'violations': [],
                'warnings': [],
                'applicable_requirements': []
            }
        
        violations = []
        warnings = []
        applicable_requirements = []
        
        for requirement in overlay.requirements:
            applicable_requirements.append(requirement.requirement_id)
            
            # Check compliance based on requirement type
            violation = self._check_requirement_compliance(
                requirement, workflow_data, execution_context
            )
            
            if violation:
                if requirement.mandatory:
                    violations.append(violation)
                else:
                    warnings.append(violation)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'applicable_requirements': applicable_requirements,
            'industry': industry.value,
            'overlay_version': overlay.version
        }
    
    def _check_requirement_compliance(
        self,
        requirement: ComplianceRequirement,
        workflow_data: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Optional[str]:
        """Check compliance for individual requirement"""
        
        # This is a simplified compliance check
        # In production, would have sophisticated rule engines
        
        try:
            if requirement.control_type == GovernanceControl.APPROVAL_WORKFLOW:
                # Check if required approvals are present
                approvals = execution_context.get('approvals', [])
                if not approvals and requirement.mandatory:
                    return f"Missing required approval for {requirement.requirement_name}"
            
            elif requirement.control_type == GovernanceControl.DATA_RETENTION:
                # Check data retention policies
                data_retention_days = execution_context.get('data_retention_days')
                if data_retention_days and requirement.framework == ComplianceFramework.GDPR:
                    # GDPR requires data minimization
                    if data_retention_days > 2555:  # 7 years max
                        return f"Data retention period exceeds GDPR limits for {requirement.requirement_name}"
            
            elif requirement.control_type == GovernanceControl.ACCESS_CONTROL:
                # Check access controls
                user_clearance = execution_context.get('user_clearance_level', 0)
                required_clearance = execution_context.get('required_clearance_level', 1)
                if user_clearance < required_clearance:
                    return f"Insufficient access clearance for {requirement.requirement_name}"
            
            elif requirement.control_type == GovernanceControl.ENCRYPTION:
                # Check encryption requirements
                data_encrypted = execution_context.get('data_encrypted', False)
                if not data_encrypted and requirement.mandatory:
                    return f"Data encryption required for {requirement.requirement_name}"
            
            # Add more compliance checks as needed
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error checking compliance for {requirement.requirement_id}: {e}")
            return f"Compliance check error for {requirement.requirement_name}: {str(e)}"
    
    def get_required_approvals(
        self,
        industry: Industry,
        workflow_type: str
    ) -> List[str]:
        """Get required approver roles for workflow type in industry"""
        
        overlay = self.get_overlay(industry)
        if not overlay:
            return []
        
        # Check exact match first
        if workflow_type in overlay.required_approvals:
            return overlay.required_approvals[workflow_type]
        
        # Check pattern matches
        for pattern, approvers in overlay.required_approvals.items():
            if pattern.endswith('*') and workflow_type.startswith(pattern[:-1]):
                return approvers
            elif '*' in pattern and pattern.replace('*', '') in workflow_type:
                return approvers
        
        return []
    
    def get_data_retention_policy(
        self,
        industry: Industry,
        data_type: str
    ) -> Optional[int]:
        """Get data retention policy for data type in industry"""
        
        overlay = self.get_overlay(industry)
        if not overlay:
            return None
        
        return overlay.retention_policies.get(data_type)
    
    def list_overlays(self) -> List[IndustryOverlay]:
        """List all industry overlays"""
        return list(self.overlays.values())
    
    def get_overlay_statistics(self) -> Dict[str, Any]:
        """Get industry overlay statistics"""
        
        total_overlays = len(self.overlays)
        total_requirements = sum(len(overlay.requirements) for overlay in self.overlays.values())
        
        requirements_by_industry = {}
        requirements_by_framework = {}
        requirements_by_control_type = {}
        
        for overlay in self.overlays.values():
            industry = overlay.industry.value
            requirements_by_industry[industry] = len(overlay.requirements)
            
            for requirement in overlay.requirements:
                framework = requirement.framework.value
                requirements_by_framework[framework] = requirements_by_framework.get(framework, 0) + 1
                
                control_type = requirement.control_type.value
                requirements_by_control_type[control_type] = requirements_by_control_type.get(control_type, 0) + 1
        
        return {
            'total_overlays': total_overlays,
            'total_requirements': total_requirements,
            'requirements_by_industry': requirements_by_industry,
            'requirements_by_framework': requirements_by_framework,
            'requirements_by_control_type': requirements_by_control_type,
            'supported_industries': [industry.value for industry in Industry],
            'supported_frameworks': [framework.value for framework in ComplianceFramework]
        }

# Global instance
industry_governance_manager = IndustryGovernanceManager()
