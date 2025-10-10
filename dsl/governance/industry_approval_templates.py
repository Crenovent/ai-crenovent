"""
Tasks 14.1-T29, 14.1-T30: Create approval and override templates per industry
Industry-specific templates for SaaS, Banking, Insurance, E-commerce, FS, IT Services
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IndustryCode(Enum):
    """Supported industry codes"""
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    ECOMMERCE = "E-commerce"
    FINANCIAL_SERVICES = "FS"
    IT_SERVICES = "IT_Services"
    HEALTHCARE = "Healthcare"


class ApprovalType(Enum):
    """Types of approval workflows"""
    DEAL_APPROVAL = "deal_approval"
    POLICY_EXCEPTION = "policy_exception"
    EMERGENCY_ACCESS = "emergency_access"
    COMPLIANCE_OVERRIDE = "compliance_override"
    FINANCIAL_TRANSACTION = "financial_transaction"
    DATA_ACCESS = "data_access"
    SYSTEM_CHANGE = "system_change"
    REGULATORY_EXEMPTION = "regulatory_exemption"


class RiskLevel(Enum):
    """Risk levels for approvals"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalTemplate:
    """Approval workflow template"""
    template_id: str
    name: str
    description: str
    
    # Industry context
    industry_code: IndustryCode
    approval_type: ApprovalType
    
    # Approval chain configuration
    approval_chain: List[str]  # Roles in order
    quorum_required: int = 1
    parallel_approval: bool = False
    
    # Risk and thresholds
    risk_level: RiskLevel = RiskLevel.MEDIUM
    value_threshold: Optional[float] = None
    currency: str = "USD"
    
    # SLA configuration
    sla_hours: int = 24
    escalation_hours: int = 4
    auto_escalate: bool = True
    
    # Compliance requirements
    compliance_frameworks: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    mandatory_fields: List[str] = field(default_factory=list)
    
    # Notification settings
    notification_channels: List[str] = field(default_factory=lambda: ["email"])
    notification_templates: Dict[str, str] = field(default_factory=dict)
    
    # Conditions and rules
    approval_conditions: Dict[str, Any] = field(default_factory=dict)
    bypass_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OverrideTemplate:
    """Override workflow template"""
    template_id: str
    name: str
    description: str
    
    # Industry context
    industry_code: IndustryCode
    override_type: str
    component_affected: str
    
    # Override configuration
    approval_required: bool = True
    approval_chain: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_level: RiskLevel = RiskLevel.HIGH
    business_impact_required: bool = True
    
    # Expiration and limits
    default_expiry_hours: int = 24
    max_uses: Optional[int] = None
    cooldown_hours: int = 0
    
    # Compliance requirements
    compliance_frameworks: List[str] = field(default_factory=list)
    required_justifications: List[str] = field(default_factory=list)
    audit_requirements: List[str] = field(default_factory=list)
    
    # Conditions
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    approval_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IndustryApprovalTemplateManager:
    """
    Industry Approval Template Manager
    Tasks 14.1-T29, 14.1-T30: Create approval and override templates per industry
    
    Provides pre-built templates for:
    - SaaS: Deal approvals, subscription changes, customer escalations
    - Banking: Loan approvals, KYC overrides, regulatory exemptions
    - Insurance: Claims approvals, policy exceptions, solvency overrides
    - E-commerce: Refund approvals, fraud overrides, payment exceptions
    - Financial Services: Transaction approvals, compliance overrides
    - IT Services: Access approvals, system changes, emergency access
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Template storage
        self.approval_templates: Dict[str, ApprovalTemplate] = {}
        self.override_templates: Dict[str, OverrideTemplate] = {}
        
        # Initialize industry templates
        self._initialize_saas_templates()
        self._initialize_banking_templates()
        self._initialize_insurance_templates()
        self._initialize_ecommerce_templates()
        self._initialize_fs_templates()
        self._initialize_it_services_templates()
        
        self.logger.info(f"✅ Initialized {len(self.approval_templates)} approval templates and {len(self.override_templates)} override templates")
    
    def _initialize_saas_templates(self):
        """Initialize SaaS industry templates"""
        
        # SaaS High-Value Deal Approval
        saas_deal_approval = ApprovalTemplate(
            template_id="saas_deal_approval_high_value",
            name="SaaS High-Value Deal Approval",
            description="Approval workflow for SaaS deals above $500K ARR",
            industry_code=IndustryCode.SAAS,
            approval_type=ApprovalType.DEAL_APPROVAL,
            approval_chain=["sales_manager", "sales_director", "cfo", "ceo"],
            quorum_required=3,
            risk_level=RiskLevel.HIGH,
            value_threshold=500000.0,
            sla_hours=48,
            escalation_hours=8,
            compliance_frameworks=["SOX", "Revenue_Recognition"],
            required_evidence=["deal_summary", "customer_credit_check", "legal_review"],
            mandatory_fields=["customer_name", "deal_value", "contract_term", "payment_terms"],
            notification_channels=["email", "slack"],
            approval_conditions={
                "min_deal_value": 500000,
                "max_discount_percent": 30,
                "requires_legal_review": True
            },
            tags=["saas", "high_value", "revenue_critical"]
        )
        
        # SaaS Subscription Change Approval
        saas_subscription_approval = ApprovalTemplate(
            template_id="saas_subscription_change",
            name="SaaS Subscription Change Approval",
            description="Approval for subscription plan changes and pricing adjustments",
            industry_code=IndustryCode.SAAS,
            approval_type=ApprovalType.POLICY_EXCEPTION,
            approval_chain=["customer_success_manager", "revenue_operations"],
            quorum_required=1,
            risk_level=RiskLevel.MEDIUM,
            sla_hours=24,
            escalation_hours=4,
            compliance_frameworks=["Revenue_Recognition"],
            required_evidence=["customer_request", "impact_analysis"],
            mandatory_fields=["customer_id", "current_plan", "requested_plan", "effective_date"],
            tags=["saas", "subscription", "customer_success"]
        )
        
        # SaaS Emergency Customer Access
        saas_emergency_access = OverrideTemplate(
            template_id="saas_emergency_customer_access",
            name="SaaS Emergency Customer Access Override",
            description="Emergency access to customer data for critical support issues",
            industry_code=IndustryCode.SAAS,
            override_type="emergency_access",
            component_affected="customer_data_access_policy",
            approval_required=True,
            approval_chain=["support_manager", "data_protection_officer"],
            risk_level=RiskLevel.HIGH,
            default_expiry_hours=4,
            max_uses=1,
            compliance_frameworks=["GDPR", "SOC2"],
            required_justifications=["customer_impact", "business_justification", "security_assessment"],
            audit_requirements=["access_log", "data_accessed", "resolution_summary"],
            tags=["saas", "emergency", "customer_data", "gdpr"]
        )
        
        # Store SaaS templates
        self.approval_templates.update({
            saas_deal_approval.template_id: saas_deal_approval,
            saas_subscription_approval.template_id: saas_subscription_approval
        })
        
        self.override_templates.update({
            saas_emergency_access.template_id: saas_emergency_access
        })
    
    def _initialize_banking_templates(self):
        """Initialize Banking industry templates"""
        
        # Banking Loan Approval
        banking_loan_approval = ApprovalTemplate(
            template_id="banking_loan_approval_high_value",
            name="Banking High-Value Loan Approval",
            description="Approval workflow for loans above ₹1 Crore",
            industry_code=IndustryCode.BANKING,
            approval_type=ApprovalType.FINANCIAL_TRANSACTION,
            approval_chain=["relationship_manager", "credit_manager", "regional_head", "chief_risk_officer"],
            quorum_required=3,
            risk_level=RiskLevel.CRITICAL,
            value_threshold=10000000.0,  # ₹1 Crore
            currency="INR",
            sla_hours=72,
            escalation_hours=12,
            compliance_frameworks=["RBI", "Basel_III", "KYC", "AML"],
            required_evidence=["credit_report", "kyc_documents", "financial_statements", "collateral_valuation"],
            mandatory_fields=["customer_id", "loan_amount", "loan_type", "tenure", "interest_rate", "collateral_details"],
            approval_conditions={
                "min_credit_score": 700,
                "max_ltv_ratio": 80,
                "kyc_status": "complete",
                "aml_cleared": True
            },
            tags=["banking", "loan", "high_value", "rbi_compliance"]
        )
        
        # Banking KYC Override
        banking_kyc_override = OverrideTemplate(
            template_id="banking_kyc_emergency_override",
            name="Banking KYC Emergency Override",
            description="Emergency KYC override for urgent financial transactions",
            industry_code=IndustryCode.BANKING,
            override_type="regulatory_exemption",
            component_affected="kyc_verification_policy",
            approval_required=True,
            approval_chain=["compliance_head", "chief_risk_officer", "md_ceo"],
            risk_level=RiskLevel.CRITICAL,
            default_expiry_hours=24,
            max_uses=1,
            cooldown_hours=168,  # 1 week
            compliance_frameworks=["RBI", "KYC", "AML", "PMLA"],
            required_justifications=["emergency_nature", "customer_relationship", "risk_mitigation", "regulatory_reporting"],
            audit_requirements=["rbi_reporting", "transaction_monitoring", "post_verification_plan"],
            trigger_conditions={
                "transaction_amount_min": 100000,
                "customer_relationship_years_min": 5,
                "emergency_category": ["medical", "natural_disaster", "family_emergency"]
            },
            tags=["banking", "kyc", "emergency", "rbi", "high_risk"]
        )
        
        # Store Banking templates
        self.approval_templates.update({
            banking_loan_approval.template_id: banking_loan_approval
        })
        
        self.override_templates.update({
            banking_kyc_override.template_id: banking_kyc_override
        })
    
    def _initialize_insurance_templates(self):
        """Initialize Insurance industry templates"""
        
        # Insurance Claims Approval
        insurance_claims_approval = ApprovalTemplate(
            template_id="insurance_claims_high_value",
            name="Insurance High-Value Claims Approval",
            description="Approval workflow for insurance claims above ₹10 Lakhs",
            industry_code=IndustryCode.INSURANCE,
            approval_type=ApprovalType.FINANCIAL_TRANSACTION,
            approval_chain=["claims_officer", "senior_claims_manager", "chief_underwriter"],
            quorum_required=2,
            risk_level=RiskLevel.HIGH,
            value_threshold=1000000.0,  # ₹10 Lakhs
            currency="INR",
            sla_hours=120,  # 5 days
            escalation_hours=24,
            compliance_frameworks=["IRDAI", "Insurance_Act"],
            required_evidence=["claim_form", "medical_reports", "police_report", "surveyor_report"],
            mandatory_fields=["policy_number", "claim_amount", "incident_date", "claim_type"],
            approval_conditions={
                "policy_active": True,
                "premium_paid": True,
                "claim_within_time_limit": True,
                "fraud_score_max": 0.3
            },
            tags=["insurance", "claims", "high_value", "irdai_compliance"]
        )
        
        # Insurance Solvency Override
        insurance_solvency_override = OverrideTemplate(
            template_id="insurance_solvency_emergency_override",
            name="Insurance Solvency Emergency Override",
            description="Emergency override for solvency ratio requirements",
            industry_code=IndustryCode.INSURANCE,
            override_type="regulatory_exemption",
            component_affected="solvency_ratio_policy",
            approval_required=True,
            approval_chain=["chief_actuary", "chief_risk_officer", "board_of_directors"],
            risk_level=RiskLevel.CRITICAL,
            default_expiry_hours=72,
            max_uses=1,
            cooldown_hours=2160,  # 90 days
            compliance_frameworks=["IRDAI", "Solvency_II"],
            required_justifications=["market_conditions", "temporary_nature", "recovery_plan", "irdai_notification"],
            audit_requirements=["irdai_filing", "board_resolution", "recovery_timeline"],
            tags=["insurance", "solvency", "emergency", "irdai", "board_approval"]
        )
        
        # Store Insurance templates
        self.approval_templates.update({
            insurance_claims_approval.template_id: insurance_claims_approval
        })
        
        self.override_templates.update({
            insurance_solvency_override.template_id: insurance_solvency_override
        })
    
    def _initialize_ecommerce_templates(self):
        """Initialize E-commerce industry templates"""
        
        # E-commerce Refund Approval
        ecommerce_refund_approval = ApprovalTemplate(
            template_id="ecommerce_refund_high_value",
            name="E-commerce High-Value Refund Approval",
            description="Approval workflow for refunds above $10,000",
            industry_code=IndustryCode.ECOMMERCE,
            approval_type=ApprovalType.FINANCIAL_TRANSACTION,
            approval_chain=["customer_service_manager", "finance_manager"],
            quorum_required=2,
            risk_level=RiskLevel.MEDIUM,
            value_threshold=10000.0,
            sla_hours=48,
            escalation_hours=8,
            compliance_frameworks=["PCI_DSS", "Consumer_Protection"],
            required_evidence=["order_details", "return_reason", "customer_history"],
            mandatory_fields=["order_id", "refund_amount", "refund_reason", "customer_id"],
            tags=["ecommerce", "refund", "customer_service"]
        )
        
        # E-commerce Fraud Override
        ecommerce_fraud_override = OverrideTemplate(
            template_id="ecommerce_fraud_detection_override",
            name="E-commerce Fraud Detection Override",
            description="Override fraud detection system for legitimate transactions",
            industry_code=IndustryCode.ECOMMERCE,
            override_type="fraud_detection_override",
            component_affected="fraud_detection_system",
            approval_required=True,
            approval_chain=["fraud_analyst", "risk_manager"],
            risk_level=RiskLevel.HIGH,
            default_expiry_hours=1,
            max_uses=1,
            compliance_frameworks=["PCI_DSS"],
            required_justifications=["customer_verification", "transaction_analysis", "risk_assessment"],
            audit_requirements=["transaction_log", "verification_evidence", "outcome_tracking"],
            tags=["ecommerce", "fraud", "risk_management", "pci_dss"]
        )
        
        # Store E-commerce templates
        self.approval_templates.update({
            ecommerce_refund_approval.template_id: ecommerce_refund_approval
        })
        
        self.override_templates.update({
            ecommerce_fraud_override.template_id: ecommerce_fraud_override
        })
    
    def _initialize_fs_templates(self):
        """Initialize Financial Services templates"""
        
        # FS Transaction Approval
        fs_transaction_approval = ApprovalTemplate(
            template_id="fs_transaction_approval_high_value",
            name="FS High-Value Transaction Approval",
            description="Approval workflow for financial transactions above $1M",
            industry_code=IndustryCode.FINANCIAL_SERVICES,
            approval_type=ApprovalType.FINANCIAL_TRANSACTION,
            approval_chain=["portfolio_manager", "compliance_officer", "cfo"],
            quorum_required=3,
            risk_level=RiskLevel.CRITICAL,
            value_threshold=1000000.0,
            sla_hours=24,
            escalation_hours=4,
            compliance_frameworks=["SOX", "SEC", "FINRA"],
            required_evidence=["transaction_rationale", "risk_analysis", "compliance_review"],
            mandatory_fields=["transaction_type", "amount", "counterparty", "settlement_date"],
            tags=["financial_services", "transaction", "high_value", "sox_compliance"]
        )
        
        # FS Compliance Override
        fs_compliance_override = OverrideTemplate(
            template_id="fs_compliance_emergency_override",
            name="FS Compliance Emergency Override",
            description="Emergency override for compliance controls",
            industry_code=IndustryCode.FINANCIAL_SERVICES,
            override_type="compliance_override",
            component_affected="compliance_controls",
            approval_required=True,
            approval_chain=["chief_compliance_officer", "general_counsel", "ceo"],
            risk_level=RiskLevel.CRITICAL,
            default_expiry_hours=12,
            max_uses=1,
            cooldown_hours=720,  # 30 days
            compliance_frameworks=["SOX", "SEC", "FINRA"],
            required_justifications=["regulatory_deadline", "business_continuity", "risk_mitigation"],
            audit_requirements=["regulatory_filing", "board_notification", "external_audit_notification"],
            tags=["financial_services", "compliance", "emergency", "regulatory"]
        )
        
        # Store FS templates
        self.approval_templates.update({
            fs_transaction_approval.template_id: fs_transaction_approval
        })
        
        self.override_templates.update({
            fs_compliance_override.template_id: fs_compliance_override
        })
    
    def _initialize_it_services_templates(self):
        """Initialize IT Services templates"""
        
        # IT Services System Change Approval
        it_system_change_approval = ApprovalTemplate(
            template_id="it_system_change_critical",
            name="IT Critical System Change Approval",
            description="Approval workflow for critical system changes",
            industry_code=IndustryCode.IT_SERVICES,
            approval_type=ApprovalType.SYSTEM_CHANGE,
            approval_chain=["technical_lead", "it_manager", "cto"],
            quorum_required=2,
            risk_level=RiskLevel.HIGH,
            sla_hours=48,
            escalation_hours=8,
            compliance_frameworks=["SOC2", "ISO27001"],
            required_evidence=["change_request", "impact_analysis", "rollback_plan", "testing_results"],
            mandatory_fields=["system_affected", "change_description", "implementation_date", "rollback_procedure"],
            tags=["it_services", "system_change", "critical", "soc2"]
        )
        
        # IT Services Emergency Access Override
        it_emergency_access_override = OverrideTemplate(
            template_id="it_emergency_access_override",
            name="IT Emergency Access Override",
            description="Emergency access override for critical system issues",
            industry_code=IndustryCode.IT_SERVICES,
            override_type="emergency_access",
            component_affected="access_control_system",
            approval_required=True,
            approval_chain=["it_security_manager", "cto"],
            risk_level=RiskLevel.HIGH,
            default_expiry_hours=8,
            max_uses=1,
            compliance_frameworks=["SOC2", "ISO27001"],
            required_justifications=["system_outage", "business_impact", "security_assessment"],
            audit_requirements=["access_log", "actions_performed", "incident_report"],
            tags=["it_services", "emergency_access", "security", "incident_response"]
        )
        
        # Store IT Services templates
        self.approval_templates.update({
            it_system_change_approval.template_id: it_system_change_approval
        })
        
        self.override_templates.update({
            it_emergency_access_override.template_id: it_emergency_access_override
        })
    
    def get_approval_templates_by_industry(self, industry_code: IndustryCode) -> List[ApprovalTemplate]:
        """Get approval templates for specific industry"""
        
        return [
            template for template in self.approval_templates.values()
            if template.industry_code == industry_code and template.is_active
        ]
    
    def get_override_templates_by_industry(self, industry_code: IndustryCode) -> List[OverrideTemplate]:
        """Get override templates for specific industry"""
        
        return [
            template for template in self.override_templates.values()
            if template.industry_code == industry_code and template.is_active
        ]
    
    def get_template_by_id(self, template_id: str) -> Optional[ApprovalTemplate | OverrideTemplate]:
        """Get template by ID"""
        
        if template_id in self.approval_templates:
            return self.approval_templates[template_id]
        elif template_id in self.override_templates:
            return self.override_templates[template_id]
        else:
            return None
    
    def get_templates_by_risk_level(self, risk_level: RiskLevel) -> Dict[str, List]:
        """Get templates by risk level"""
        
        approval_templates = [
            template for template in self.approval_templates.values()
            if template.risk_level == risk_level and template.is_active
        ]
        
        override_templates = [
            template for template in self.override_templates.values()
            if template.risk_level == risk_level and template.is_active
        ]
        
        return {
            'approval_templates': approval_templates,
            'override_templates': override_templates
        }
    
    def get_templates_by_compliance_framework(self, framework: str) -> Dict[str, List]:
        """Get templates by compliance framework"""
        
        approval_templates = [
            template for template in self.approval_templates.values()
            if framework in template.compliance_frameworks and template.is_active
        ]
        
        override_templates = [
            template for template in self.override_templates.values()
            if framework in template.compliance_frameworks and template.is_active
        ]
        
        return {
            'approval_templates': approval_templates,
            'override_templates': override_templates
        }
    
    def create_workflow_from_template(
        self,
        template_id: str,
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create workflow configuration from template"""
        
        template = self.get_template_by_id(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        if isinstance(template, ApprovalTemplate):
            return self._create_approval_workflow(template, workflow_context)
        elif isinstance(template, OverrideTemplate):
            return self._create_override_workflow(template, workflow_context)
        else:
            raise ValueError(f"Unknown template type: {type(template)}")
    
    def _create_approval_workflow(
        self, template: ApprovalTemplate, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create approval workflow from template"""
        
        workflow_config = {
            'workflow_id': context.get('workflow_id', str(uuid.uuid4())),
            'template_id': template.template_id,
            'workflow_type': 'approval',
            'industry_code': template.industry_code.value,
            'approval_type': template.approval_type.value,
            
            # Approval chain configuration
            'approval_chain': template.approval_chain.copy(),
            'quorum_required': template.quorum_required,
            'parallel_approval': template.parallel_approval,
            
            # Risk and thresholds
            'risk_level': template.risk_level.value,
            'value_threshold': template.value_threshold,
            'currency': template.currency,
            
            # SLA configuration
            'sla_hours': template.sla_hours,
            'escalation_hours': template.escalation_hours,
            'auto_escalate': template.auto_escalate,
            
            # Compliance
            'compliance_frameworks': template.compliance_frameworks.copy(),
            'required_evidence': template.required_evidence.copy(),
            'mandatory_fields': template.mandatory_fields.copy(),
            
            # Notifications
            'notification_channels': template.notification_channels.copy(),
            'notification_templates': template.notification_templates.copy(),
            
            # Conditions
            'approval_conditions': template.approval_conditions.copy(),
            'bypass_conditions': template.bypass_conditions.copy(),
            
            # Context data
            'context': context,
            
            # Timing
            'created_at': datetime.now(timezone.utc).isoformat(),
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=template.sla_hours)).isoformat()
        }
        
        return workflow_config
    
    def _create_override_workflow(
        self, template: OverrideTemplate, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create override workflow from template"""
        
        workflow_config = {
            'workflow_id': context.get('workflow_id', str(uuid.uuid4())),
            'template_id': template.template_id,
            'workflow_type': 'override',
            'industry_code': template.industry_code.value,
            'override_type': template.override_type,
            'component_affected': template.component_affected,
            
            # Override configuration
            'approval_required': template.approval_required,
            'approval_chain': template.approval_chain.copy(),
            
            # Risk assessment
            'risk_level': template.risk_level.value,
            'business_impact_required': template.business_impact_required,
            
            # Expiration and limits
            'default_expiry_hours': template.default_expiry_hours,
            'max_uses': template.max_uses,
            'cooldown_hours': template.cooldown_hours,
            
            # Compliance
            'compliance_frameworks': template.compliance_frameworks.copy(),
            'required_justifications': template.required_justifications.copy(),
            'audit_requirements': template.audit_requirements.copy(),
            
            # Conditions
            'trigger_conditions': template.trigger_conditions.copy(),
            'approval_conditions': template.approval_conditions.copy(),
            
            # Context data
            'context': context,
            
            # Timing
            'created_at': datetime.now(timezone.utc).isoformat(),
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=template.default_expiry_hours)).isoformat()
        }
        
        return workflow_config
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get template statistics"""
        
        stats = {
            'total_approval_templates': len(self.approval_templates),
            'total_override_templates': len(self.override_templates),
            'templates_by_industry': {},
            'templates_by_risk_level': {},
            'templates_by_compliance_framework': {}
        }
        
        # By industry
        for industry in IndustryCode:
            approval_count = len(self.get_approval_templates_by_industry(industry))
            override_count = len(self.get_override_templates_by_industry(industry))
            stats['templates_by_industry'][industry.value] = {
                'approval_templates': approval_count,
                'override_templates': override_count,
                'total': approval_count + override_count
            }
        
        # By risk level
        for risk_level in RiskLevel:
            templates = self.get_templates_by_risk_level(risk_level)
            stats['templates_by_risk_level'][risk_level.value] = {
                'approval_templates': len(templates['approval_templates']),
                'override_templates': len(templates['override_templates']),
                'total': len(templates['approval_templates']) + len(templates['override_templates'])
            }
        
        # By compliance framework
        frameworks = set()
        for template in list(self.approval_templates.values()) + list(self.override_templates.values()):
            frameworks.update(template.compliance_frameworks)
        
        for framework in frameworks:
            templates = self.get_templates_by_compliance_framework(framework)
            stats['templates_by_compliance_framework'][framework] = {
                'approval_templates': len(templates['approval_templates']),
                'override_templates': len(templates['override_templates']),
                'total': len(templates['approval_templates']) + len(templates['override_templates'])
            }
        
        return stats


# Global template manager instance
industry_template_manager = IndustryApprovalTemplateManager()
