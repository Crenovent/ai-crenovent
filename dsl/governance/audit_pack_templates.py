"""
Tasks 14.4-T10, 14.4-T11, 14.4-T12, 14.4-T13: Build audit pack templates for SOX, RBI, IRDAI, GDPR/DPDP
Industry-specific audit pack templates for regulatory compliance
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "SOX"
    RBI = "RBI"
    IRDAI = "IRDAI"
    GDPR = "GDPR"
    DPDP = "DPDP"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"


class AuditSectionType(Enum):
    """Types of audit sections"""
    EXECUTIVE_SUMMARY = "executive_summary"
    CONTROL_TESTING = "control_testing"
    SEGREGATION_OF_DUTIES = "segregation_of_duties"
    APPROVAL_ANALYSIS = "approval_analysis"
    OVERRIDE_ANALYSIS = "override_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_VIOLATIONS = "compliance_violations"
    EVIDENCE_REVIEW = "evidence_review"
    RECOMMENDATIONS = "recommendations"
    APPENDICES = "appendices"


@dataclass
class AuditSection:
    """Audit pack section definition"""
    section_id: str
    name: str
    description: str
    section_type: AuditSectionType
    
    # Content requirements
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    
    # Data sources
    data_sources: List[str] = field(default_factory=list)  # Tables/APIs to query
    
    # Validation rules
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Formatting
    template: str = ""
    include_charts: bool = False
    include_tables: bool = True
    
    # Metadata
    is_mandatory: bool = True
    order_index: int = 0


@dataclass
class AuditPackTemplate:
    """Audit pack template definition"""
    template_id: str
    name: str
    description: str
    framework: ComplianceFramework
    
    # Template configuration
    sections: List[AuditSection] = field(default_factory=list)
    
    # Compliance requirements
    regulatory_requirements: List[str] = field(default_factory=list)
    retention_years: int = 7
    signature_required: bool = True
    
    # Output configuration
    supported_formats: List[str] = field(default_factory=lambda: ["PDF", "CSV"])
    default_format: str = "PDF"
    
    # Validation
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    version: str = "1.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


class AuditPackTemplateManager:
    """
    Audit Pack Template Manager
    Tasks 14.4-T10, 14.4-T11, 14.4-T12, 14.4-T13
    
    Provides industry-specific audit pack templates:
    - SOX: Financial controls, segregation of duties, approval chains
    - RBI: Banking regulations, loan approvals, risk management
    - IRDAI: Insurance regulations, claims processing, solvency
    - GDPR/DPDP: Data protection, consent management, breach reporting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Template storage
        self.templates: Dict[str, AuditPackTemplate] = {}
        
        # Initialize framework templates
        self._initialize_sox_template()
        self._initialize_rbi_template()
        self._initialize_irdai_template()
        self._initialize_gdpr_template()
        self._initialize_dpdp_template()
        
        self.logger.info(f"✅ Initialized {len(self.templates)} audit pack templates")
    
    def _initialize_sox_template(self):
        """Initialize SOX audit pack template - Task 14.4-T10"""
        
        # SOX Executive Summary Section
        sox_executive_summary = AuditSection(
            section_id="sox_executive_summary",
            name="Executive Summary",
            description="High-level overview of SOX compliance status",
            section_type=AuditSectionType.EXECUTIVE_SUMMARY,
            required_fields=[
                "audit_period", "total_transactions", "control_deficiencies",
                "material_weaknesses", "overall_assessment", "management_conclusion"
            ],
            optional_fields=["key_improvements", "prior_period_comparison"],
            data_sources=["approval_ledger", "override_ledger", "trust_scores"],
            validation_rules={
                "min_audit_period_days": 90,
                "required_management_sign_off": True
            },
            template="""
            SOX COMPLIANCE AUDIT REPORT - EXECUTIVE SUMMARY
            
            Audit Period: {audit_period}
            Report Date: {report_date}
            
            OVERALL ASSESSMENT: {overall_assessment}
            
            Key Metrics:
            - Total Financial Transactions: {total_transactions:,}
            - Control Tests Performed: {control_tests_performed:,}
            - Control Deficiencies Identified: {control_deficiencies}
            - Material Weaknesses: {material_weaknesses}
            
            Management Conclusion: {management_conclusion}
            """,
            is_mandatory=True,
            order_index=1
        )
        
        # SOX Segregation of Duties Section
        sox_sod_analysis = AuditSection(
            section_id="sox_segregation_of_duties",
            name="Segregation of Duties Analysis",
            description="Analysis of segregation of duties controls",
            section_type=AuditSectionType.SEGREGATION_OF_DUTIES,
            required_fields=[
                "sod_controls_tested", "sod_violations", "compensating_controls",
                "maker_checker_compliance", "approval_chain_analysis"
            ],
            data_sources=["approval_ledger", "users", "users_role"],
            validation_rules={
                "max_sod_violations": 0,
                "min_approval_chain_length": 2
            },
            template="""
            SEGREGATION OF DUTIES ANALYSIS
            
            Controls Tested: {sod_controls_tested}
            Violations Identified: {sod_violations}
            
            Maker-Checker Compliance: {maker_checker_compliance}%
            
            Approval Chain Analysis:
            {approval_chain_analysis}
            
            Compensating Controls:
            {compensating_controls}
            """,
            include_tables=True,
            is_mandatory=True,
            order_index=2
        )
        
        # SOX Financial Controls Section
        sox_financial_controls = AuditSection(
            section_id="sox_financial_controls",
            name="Financial Controls Testing",
            description="Testing of key financial controls",
            section_type=AuditSectionType.CONTROL_TESTING,
            required_fields=[
                "controls_tested", "control_effectiveness", "deficiencies_identified",
                "remediation_status", "testing_methodology"
            ],
            data_sources=["approval_ledger", "evidence_packs", "trust_scores"],
            validation_rules={
                "min_control_effectiveness": 95.0,
                "max_deficiency_severity": "medium"
            },
            template="""
            FINANCIAL CONTROLS TESTING
            
            Testing Period: {testing_period}
            Controls Tested: {controls_tested}
            
            Control Effectiveness: {control_effectiveness}%
            
            Deficiencies Identified:
            {deficiencies_identified}
            
            Remediation Status:
            {remediation_status}
            """,
            include_charts=True,
            is_mandatory=True,
            order_index=3
        )
        
        # SOX Override Analysis Section
        sox_override_analysis = AuditSection(
            section_id="sox_override_analysis",
            name="Management Override Analysis",
            description="Analysis of management overrides and exceptions",
            section_type=AuditSectionType.OVERRIDE_ANALYSIS,
            required_fields=[
                "total_overrides", "override_categories", "business_justifications",
                "approval_status", "risk_assessment", "follow_up_actions"
            ],
            data_sources=["override_ledger", "approval_ledger"],
            validation_rules={
                "max_override_frequency": 5.0,  # % of total transactions
                "required_cfo_approval_threshold": 100000  # USD
            },
            is_mandatory=True,
            order_index=4
        )
        
        # Create SOX template
        sox_template = AuditPackTemplate(
            template_id="sox_audit_pack_template",
            name="SOX Compliance Audit Pack",
            description="Comprehensive SOX compliance audit pack for financial controls",
            framework=ComplianceFramework.SOX,
            sections=[
                sox_executive_summary, sox_sod_analysis, 
                sox_financial_controls, sox_override_analysis
            ],
            regulatory_requirements=[
                "SOX Section 302 - Corporate Responsibility",
                "SOX Section 404 - Management Assessment of Internal Controls",
                "SOX Section 906 - Corporate Responsibility for Financial Reports"
            ],
            retention_years=7,
            signature_required=True,
            supported_formats=["PDF", "CSV", "JSON"],
            validation_rules={
                "min_audit_period_days": 90,
                "required_cfo_certification": True,
                "required_external_auditor_review": True
            }
        )
        
        self.templates[sox_template.template_id] = sox_template
    
    def _initialize_rbi_template(self):
        """Initialize RBI audit pack template - Task 14.4-T11"""
        
        # RBI Executive Summary
        rbi_executive_summary = AuditSection(
            section_id="rbi_executive_summary",
            name="RBI Compliance Executive Summary",
            description="Overview of RBI regulatory compliance",
            section_type=AuditSectionType.EXECUTIVE_SUMMARY,
            required_fields=[
                "reporting_period", "total_loan_approvals", "kyc_compliance_rate",
                "aml_violations", "regulatory_breaches", "corrective_actions"
            ],
            data_sources=["approval_ledger", "kyc_records", "aml_reports"],
            validation_rules={
                "min_kyc_compliance_rate": 100.0,
                "max_aml_violations": 0
            },
            is_mandatory=True,
            order_index=1
        )
        
        # RBI Loan Approval Analysis
        rbi_loan_analysis = AuditSection(
            section_id="rbi_loan_approval_analysis",
            name="Loan Approval Process Analysis",
            description="Analysis of loan approval processes and compliance",
            section_type=AuditSectionType.APPROVAL_ANALYSIS,
            required_fields=[
                "loan_applications_processed", "approval_rates", "rejection_reasons",
                "credit_assessment_quality", "documentation_completeness",
                "turnaround_time_compliance"
            ],
            data_sources=["loan_applications", "approval_ledger", "credit_assessments"],
            validation_rules={
                "max_turnaround_time_days": 30,
                "min_documentation_completeness": 95.0
            },
            is_mandatory=True,
            order_index=2
        )
        
        # RBI KYC Compliance Section
        rbi_kyc_compliance = AuditSection(
            section_id="rbi_kyc_compliance",
            name="KYC Compliance Analysis",
            description="Know Your Customer compliance analysis",
            section_type=AuditSectionType.COMPLIANCE_VIOLATIONS,
            required_fields=[
                "kyc_verifications_completed", "kyc_deficiencies", "customer_risk_profiling",
                "periodic_kyc_updates", "suspicious_transaction_reports"
            ],
            data_sources=["kyc_records", "customer_profiles", "str_reports"],
            validation_rules={
                "max_kyc_deficiencies": 0,
                "min_periodic_update_compliance": 100.0
            },
            is_mandatory=True,
            order_index=3
        )
        
        # RBI Risk Management Section
        rbi_risk_management = AuditSection(
            section_id="rbi_risk_management",
            name="Risk Management Framework",
            description="Risk management and capital adequacy analysis",
            section_type=AuditSectionType.RISK_ASSESSMENT,
            required_fields=[
                "capital_adequacy_ratio", "credit_risk_exposure", "operational_risk_events",
                "market_risk_metrics", "liquidity_ratios", "stress_test_results"
            ],
            data_sources=["risk_reports", "capital_calculations", "stress_tests"],
            validation_rules={
                "min_capital_adequacy_ratio": 9.0,  # Basel III minimum
                "max_npa_ratio": 4.0
            },
            is_mandatory=True,
            order_index=4
        )
        
        # Create RBI template
        rbi_template = AuditPackTemplate(
            template_id="rbi_audit_pack_template",
            name="RBI Regulatory Compliance Audit Pack",
            description="Comprehensive RBI compliance audit pack for banking operations",
            framework=ComplianceFramework.RBI,
            sections=[
                rbi_executive_summary, rbi_loan_analysis,
                rbi_kyc_compliance, rbi_risk_management
            ],
            regulatory_requirements=[
                "RBI Master Direction on KYC",
                "RBI Guidelines on Credit Risk Management",
                "RBI Framework for Dealing with Loan Frauds",
                "Basel III Capital Regulations"
            ],
            retention_years=10,
            signature_required=True,
            supported_formats=["PDF", "XML", "CSV"],
            validation_rules={
                "required_rbi_filing": True,
                "min_audit_frequency_months": 6
            }
        )
        
        self.templates[rbi_template.template_id] = rbi_template
    
    def _initialize_irdai_template(self):
        """Initialize IRDAI audit pack template - Task 14.4-T12"""
        
        # IRDAI Executive Summary
        irdai_executive_summary = AuditSection(
            section_id="irdai_executive_summary",
            name="IRDAI Compliance Executive Summary",
            description="Overview of IRDAI regulatory compliance",
            section_type=AuditSectionType.EXECUTIVE_SUMMARY,
            required_fields=[
                "reporting_period", "total_claims_processed", "claims_settlement_ratio",
                "solvency_margin", "regulatory_violations", "customer_complaints"
            ],
            data_sources=["claims_register", "solvency_reports", "complaint_register"],
            validation_rules={
                "min_claims_settlement_ratio": 95.0,
                "min_solvency_margin": 150.0
            },
            is_mandatory=True,
            order_index=1
        )
        
        # IRDAI Claims Processing Analysis
        irdai_claims_analysis = AuditSection(
            section_id="irdai_claims_analysis",
            name="Claims Processing Analysis",
            description="Analysis of claims processing and settlement",
            section_type=AuditSectionType.APPROVAL_ANALYSIS,
            required_fields=[
                "claims_received", "claims_settled", "claims_rejected", "settlement_time",
                "claims_above_threshold", "surveyor_reports", "fraud_detection"
            ],
            data_sources=["claims_register", "approval_ledger", "surveyor_reports"],
            validation_rules={
                "max_settlement_time_days": 30,
                "min_fraud_detection_rate": 2.0
            },
            is_mandatory=True,
            order_index=2
        )
        
        # IRDAI Solvency Analysis
        irdai_solvency_analysis = AuditSection(
            section_id="irdai_solvency_analysis",
            name="Solvency and Capital Management",
            description="Solvency margin and capital adequacy analysis",
            section_type=AuditSectionType.RISK_ASSESSMENT,
            required_fields=[
                "available_solvency_margin", "required_solvency_margin", "solvency_ratio",
                "capital_adequacy", "investment_portfolio", "reinsurance_arrangements"
            ],
            data_sources=["solvency_calculations", "investment_reports", "reinsurance_contracts"],
            validation_rules={
                "min_solvency_ratio": 1.5,
                "max_investment_concentration": 10.0
            },
            is_mandatory=True,
            order_index=3
        )
        
        # IRDAI Customer Protection
        irdai_customer_protection = AuditSection(
            section_id="irdai_customer_protection",
            name="Customer Protection and Fair Practices",
            description="Customer protection and fair business practices",
            section_type=AuditSectionType.COMPLIANCE_VIOLATIONS,
            required_fields=[
                "customer_complaints", "complaint_resolution_time", "mis_selling_cases",
                "policy_cancellations", "premium_collection", "disclosure_compliance"
            ],
            data_sources=["complaint_register", "policy_records", "disclosure_reports"],
            validation_rules={
                "max_complaint_resolution_days": 15,
                "max_mis_selling_rate": 1.0
            },
            is_mandatory=True,
            order_index=4
        )
        
        # Create IRDAI template
        irdai_template = AuditPackTemplate(
            template_id="irdai_audit_pack_template",
            name="IRDAI Regulatory Compliance Audit Pack",
            description="Comprehensive IRDAI compliance audit pack for insurance operations",
            framework=ComplianceFramework.IRDAI,
            sections=[
                irdai_executive_summary, irdai_claims_analysis,
                irdai_solvency_analysis, irdai_customer_protection
            ],
            regulatory_requirements=[
                "IRDAI (Protection of Policyholders' Interests) Regulations",
                "IRDAI (Solvency Margin) Regulations",
                "IRDAI (Claims Settlement) Regulations",
                "IRDAI (Corporate Governance) Guidelines"
            ],
            retention_years=10,
            signature_required=True,
            supported_formats=["PDF", "CSV", "XML"],
            validation_rules={
                "required_irdai_filing": True,
                "min_board_approval": True
            }
        )
        
        self.templates[irdai_template.template_id] = irdai_template
    
    def _initialize_gdpr_template(self):
        """Initialize GDPR audit pack template - Task 14.4-T13"""
        
        # GDPR Executive Summary
        gdpr_executive_summary = AuditSection(
            section_id="gdpr_executive_summary",
            name="GDPR Compliance Executive Summary",
            description="Overview of GDPR data protection compliance",
            section_type=AuditSectionType.EXECUTIVE_SUMMARY,
            required_fields=[
                "audit_period", "data_processing_activities", "consent_compliance_rate",
                "data_breaches", "subject_access_requests", "dpo_activities"
            ],
            data_sources=["data_processing_register", "consent_records", "breach_register"],
            validation_rules={
                "min_consent_compliance_rate": 95.0,
                "max_breach_notification_hours": 72
            },
            is_mandatory=True,
            order_index=1
        )
        
        # GDPR Data Processing Analysis
        gdpr_data_processing = AuditSection(
            section_id="gdpr_data_processing",
            name="Data Processing Activities Analysis",
            description="Analysis of personal data processing activities",
            section_type=AuditSectionType.APPROVAL_ANALYSIS,
            required_fields=[
                "processing_purposes", "legal_bases", "data_categories", "data_subjects",
                "retention_periods", "third_party_transfers", "privacy_impact_assessments"
            ],
            data_sources=["processing_register", "pia_reports", "transfer_agreements"],
            validation_rules={
                "required_legal_basis": True,
                "max_retention_period_years": 7
            },
            is_mandatory=True,
            order_index=2
        )
        
        # GDPR Consent Management
        gdpr_consent_management = AuditSection(
            section_id="gdpr_consent_management",
            name="Consent Management Analysis",
            description="Analysis of consent collection and management",
            section_type=AuditSectionType.COMPLIANCE_VIOLATIONS,
            required_fields=[
                "consent_requests", "consent_granted", "consent_withdrawn",
                "consent_validity", "granular_consent", "consent_records"
            ],
            data_sources=["consent_records", "withdrawal_requests", "consent_audits"],
            validation_rules={
                "min_consent_validity_rate": 98.0,
                "required_granular_consent": True
            },
            is_mandatory=True,
            order_index=3
        )
        
        # GDPR Rights Management
        gdpr_rights_management = AuditSection(
            section_id="gdpr_rights_management",
            name="Data Subject Rights Management",
            description="Management of data subject rights requests",
            section_type=AuditSectionType.COMPLIANCE_VIOLATIONS,
            required_fields=[
                "access_requests", "rectification_requests", "erasure_requests",
                "portability_requests", "objection_requests", "response_times"
            ],
            data_sources=["rights_requests", "response_logs", "erasure_logs"],
            validation_rules={
                "max_response_time_days": 30,
                "min_fulfillment_rate": 95.0
            },
            is_mandatory=True,
            order_index=4
        )
        
        # Create GDPR template
        gdpr_template = AuditPackTemplate(
            template_id="gdpr_audit_pack_template",
            name="GDPR Data Protection Audit Pack",
            description="Comprehensive GDPR compliance audit pack for data protection",
            framework=ComplianceFramework.GDPR,
            sections=[
                gdpr_executive_summary, gdpr_data_processing,
                gdpr_consent_management, gdpr_rights_management
            ],
            regulatory_requirements=[
                "GDPR Article 5 - Principles of Processing",
                "GDPR Article 6 - Lawfulness of Processing",
                "GDPR Article 7 - Conditions for Consent",
                "GDPR Article 32 - Security of Processing",
                "GDPR Article 33 - Notification of Breach",
                "GDPR Article 35 - Data Protection Impact Assessment"
            ],
            retention_years=3,
            signature_required=True,
            supported_formats=["PDF", "JSON", "CSV"],
            validation_rules={
                "required_dpo_sign_off": True,
                "required_legal_review": True
            }
        )
        
        self.templates[gdpr_template.template_id] = gdpr_template
    
    def _initialize_dpdp_template(self):
        """Initialize DPDP (India) audit pack template - Task 14.4-T13"""
        
        # DPDP Executive Summary
        dpdp_executive_summary = AuditSection(
            section_id="dpdp_executive_summary",
            name="DPDP Compliance Executive Summary",
            description="Overview of DPDP Act compliance (India)",
            section_type=AuditSectionType.EXECUTIVE_SUMMARY,
            required_fields=[
                "audit_period", "personal_data_processing", "consent_compliance",
                "data_breaches", "grievance_redressal", "cross_border_transfers"
            ],
            data_sources=["processing_activities", "consent_logs", "breach_incidents"],
            validation_rules={
                "min_consent_compliance_rate": 100.0,
                "max_breach_notification_hours": 72
            },
            is_mandatory=True,
            order_index=1
        )
        
        # DPDP Consent Framework
        dpdp_consent_framework = AuditSection(
            section_id="dpdp_consent_framework",
            name="Consent Framework Analysis",
            description="Analysis of consent collection under DPDP Act",
            section_type=AuditSectionType.COMPLIANCE_VIOLATIONS,
            required_fields=[
                "consent_notices", "consent_mechanisms", "consent_withdrawal",
                "verifiable_parental_consent", "consent_manager_integration"
            ],
            data_sources=["consent_records", "parental_consent", "consent_manager_logs"],
            validation_rules={
                "required_clear_consent": True,
                "required_withdrawal_mechanism": True
            },
            is_mandatory=True,
            order_index=2
        )
        
        # DPDP Data Governance
        dpdp_data_governance = AuditSection(
            section_id="dpdp_data_governance",
            name="Data Governance and Protection",
            description="Data governance framework under DPDP Act",
            section_type=AuditSectionType.RISK_ASSESSMENT,
            required_fields=[
                "data_protection_measures", "data_localization", "data_retention",
                "data_accuracy", "data_minimization", "purpose_limitation"
            ],
            data_sources=["data_inventory", "retention_policies", "accuracy_reports"],
            validation_rules={
                "required_data_localization": True,
                "max_retention_period": "necessary_period"
            },
            is_mandatory=True,
            order_index=3
        )
        
        # Create DPDP template
        dpdp_template = AuditPackTemplate(
            template_id="dpdp_audit_pack_template",
            name="DPDP Act Compliance Audit Pack",
            description="Comprehensive DPDP Act compliance audit pack for India",
            framework=ComplianceFramework.DPDP,
            sections=[
                dpdp_executive_summary, dpdp_consent_framework, dpdp_data_governance
            ],
            regulatory_requirements=[
                "DPDP Act Section 6 - Grounds for Processing",
                "DPDP Act Section 7 - Consent",
                "DPDP Act Section 8 - Deemed Consent",
                "DPDP Act Section 16 - Data Localization",
                "DPDP Act Section 25 - Breach Notification"
            ],
            retention_years=3,
            signature_required=True,
            supported_formats=["PDF", "JSON", "CSV"],
            validation_rules={
                "required_dpo_appointment": True,
                "required_grievance_officer": True
            }
        )
        
        self.templates[dpdp_template.template_id] = dpdp_template
    
    def get_template_by_framework(self, framework: ComplianceFramework) -> Optional[AuditPackTemplate]:
        """Get audit pack template by compliance framework"""
        
        for template in self.templates.values():
            if template.framework == framework and template.is_active:
                return template
        return None
    
    def get_template_by_id(self, template_id: str) -> Optional[AuditPackTemplate]:
        """Get audit pack template by ID"""
        
        return self.templates.get(template_id)
    
    def get_all_templates(self) -> List[AuditPackTemplate]:
        """Get all active audit pack templates"""
        
        return [template for template in self.templates.values() if template.is_active]
    
    def validate_template_data(
        self, template_id: str, audit_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate audit data against template requirements"""
        
        template = self.get_template_by_id(template_id)
        if not template:
            return {"valid": False, "error": f"Template not found: {template_id}"}
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_required_fields": [],
            "validation_details": {}
        }
        
        # Validate each section
        for section in template.sections:
            section_data = audit_data.get(section.section_id, {})
            
            # Check required fields
            for required_field in section.required_fields:
                if required_field not in section_data:
                    validation_result["missing_required_fields"].append(
                        f"{section.section_id}.{required_field}"
                    )
            
            # Apply validation rules
            for rule_name, rule_value in section.validation_rules.items():
                if rule_name.startswith("min_") and rule_name.replace("min_", "") in section_data:
                    field_name = rule_name.replace("min_", "")
                    if section_data[field_name] < rule_value:
                        validation_result["errors"].append(
                            f"{section.section_id}.{field_name} below minimum: {section_data[field_name]} < {rule_value}"
                        )
                
                elif rule_name.startswith("max_") and rule_name.replace("max_", "") in section_data:
                    field_name = rule_name.replace("max_", "")
                    if section_data[field_name] > rule_value:
                        validation_result["errors"].append(
                            f"{section.section_id}.{field_name} above maximum: {section_data[field_name]} > {rule_value}"
                        )
        
        # Apply template-level validation rules
        for rule_name, rule_value in template.validation_rules.items():
            if rule_name == "min_audit_period_days":
                audit_period = audit_data.get("audit_period_days", 0)
                if audit_period < rule_value:
                    validation_result["errors"].append(
                        f"Audit period too short: {audit_period} < {rule_value} days"
                    )
        
        # Determine overall validity
        if validation_result["errors"] or validation_result["missing_required_fields"]:
            validation_result["valid"] = False
        
        return validation_result
    
    def generate_template_schema(self, template_id: str) -> Dict[str, Any]:
        """Generate JSON schema for template data structure"""
        
        template = self.get_template_by_id(template_id)
        if not template:
            return {}
        
        schema = {
            "type": "object",
            "title": template.name,
            "description": template.description,
            "properties": {},
            "required": []
        }
        
        for section in template.sections:
            section_schema = {
                "type": "object",
                "title": section.name,
                "description": section.description,
                "properties": {},
                "required": section.required_fields
            }
            
            # Add required fields
            for field in section.required_fields:
                section_schema["properties"][field] = {
                    "type": "string",
                    "description": f"Required field for {section.name}"
                }
            
            # Add optional fields
            for field in section.optional_fields:
                section_schema["properties"][field] = {
                    "type": "string",
                    "description": f"Optional field for {section.name}"
                }
            
            schema["properties"][section.section_id] = section_schema
            
            if section.is_mandatory:
                schema["required"].append(section.section_id)
        
        return schema
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get template statistics"""
        
        stats = {
            "total_templates": len(self.templates),
            "active_templates": len([t for t in self.templates.values() if t.is_active]),
            "templates_by_framework": {},
            "total_sections": 0,
            "mandatory_sections": 0,
            "supported_formats": set()
        }
        
        for template in self.templates.values():
            if template.is_active:
                framework = template.framework.value
                if framework not in stats["templates_by_framework"]:
                    stats["templates_by_framework"][framework] = 0
                stats["templates_by_framework"][framework] += 1
                
                stats["total_sections"] += len(template.sections)
                stats["mandatory_sections"] += len([s for s in template.sections if s.is_mandatory])
                stats["supported_formats"].update(template.supported_formats)
        
        stats["supported_formats"] = list(stats["supported_formats"])
        
        return stats


# Global template manager instance
audit_pack_template_manager = AuditPackTemplateManager()
