# Cross-Industry Schema Extensions for RBA Traces
# Task 7.1-T46: Implement cross-industry schema extensions

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

try:
    from .canonical_trace_schema import WorkflowTrace, TraceContext, GovernanceEvent
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from canonical_trace_schema import WorkflowTrace, TraceContext, GovernanceEvent

class IndustryCode(Enum):
    """Supported industry codes"""
    SAAS = "SaaS"
    BANKING = "BANK"
    INSURANCE = "INSUR"
    ECOMMERCE = "ECOMM"
    FINANCIAL_SERVICES = "FS"
    IT_SERVICES = "IT"
    HEALTHCARE = "HEALTH"
    MANUFACTURING = "MANUF"
    RETAIL = "RETAIL"
    TELECOM = "TELECOM"

@dataclass
class IndustryExtensionConfig:
    """Configuration for industry-specific extensions"""
    
    industry_code: IndustryCode
    extension_version: str = "v1.0"
    
    # Field extensions
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    custom_validators: List[str] = field(default_factory=list)
    
    # Compliance requirements
    compliance_frameworks: List[str] = field(default_factory=list)
    regulatory_tags: List[str] = field(default_factory=list)
    data_residency_requirements: List[str] = field(default_factory=list)
    
    # Business logic
    trust_score_adjustments: Dict[str, float] = field(default_factory=dict)
    governance_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    trace_retention_days: int = 2555  # 7 years default
    compression_enabled: bool = True
    sampling_rate: float = 1.0

class IndustryExtension(ABC):
    """Abstract base class for industry-specific extensions"""
    
    def __init__(self, config: IndustryExtensionConfig):
        self.config = config
        self.industry_code = config.industry_code
        
    @abstractmethod
    def extend_trace_context(self, context: TraceContext) -> TraceContext:
        """Extend trace context with industry-specific fields"""
        pass
    
    @abstractmethod
    def validate_industry_compliance(self, trace: WorkflowTrace) -> Dict[str, Any]:
        """Validate trace against industry-specific requirements"""
        pass
    
    @abstractmethod
    def calculate_industry_trust_score(self, trace: WorkflowTrace) -> float:
        """Calculate industry-specific trust score adjustments"""
        pass
    
    @abstractmethod
    def generate_industry_governance_events(self, trace: WorkflowTrace) -> List[GovernanceEvent]:
        """Generate industry-specific governance events"""
        pass

class SaaSIndustryExtension(IndustryExtension):
    """SaaS industry-specific extensions"""
    
    def __init__(self):
        config = IndustryExtensionConfig(
            industry_code=IndustryCode.SAAS,
            required_fields=[
                "saas_metrics.mrr_impact",
                "saas_metrics.customer_tier",
                "saas_metrics.subscription_status"
            ],
            optional_fields=[
                "saas_metrics.feature_usage",
                "saas_metrics.api_calls",
                "saas_metrics.storage_usage",
                "saas_metrics.user_engagement_score"
            ],
            compliance_frameworks=["SOX", "GDPR", "CCPA"],
            regulatory_tags=["data_processing", "subscription_billing"],
            trust_score_adjustments={
                "high_value_customer": 0.1,
                "enterprise_tier": 0.05,
                "api_abuse_detected": -0.2
            }
        )
        super().__init__(config)
    
    def extend_trace_context(self, context: TraceContext) -> TraceContext:
        """Extend context with SaaS-specific fields"""
        
        saas_fields = {
            "subscription_tier": "standard",  # free, standard, premium, enterprise
            "customer_segment": "smb",  # smb, mid_market, enterprise
            "product_line": "core",  # core, addon, integration
            "feature_flags": {},
            "usage_metrics": {
                "api_calls_last_30d": 0,
                "storage_gb": 0,
                "active_users": 0,
                "feature_adoption_score": 0.0
            },
            "billing_info": {
                "mrr": 0.0,
                "arr": 0.0,
                "payment_method": "credit_card",
                "billing_cycle": "monthly",
                "next_billing_date": None
            },
            "support_tier": "standard",  # basic, standard, premium, white_glove
            "onboarding_status": "completed",  # pending, in_progress, completed
            "health_score": 0.8,  # 0.0 - 1.0
            "churn_risk": "low"  # low, medium, high, critical
        }
        
        context.industry_specific_fields.update(saas_fields)
        return context
    
    def validate_industry_compliance(self, trace: WorkflowTrace) -> Dict[str, Any]:
        """Validate SaaS-specific compliance requirements"""
        
        validation_result = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        saas_fields = trace.context.industry_specific_fields
        
        # Validate subscription data handling
        if "billing_info" in saas_fields:
            billing_info = saas_fields["billing_info"]
            if billing_info.get("mrr", 0) > 0 and not trace.context.pii_redaction_enabled:
                validation_result["violations"].append(
                    "Financial data present but PII redaction not enabled"
                )
                validation_result["compliant"] = False
        
        # Validate GDPR compliance for EU customers
        if trace.context.data_residency in ["EU", "UK"]:
            if "GDPR" not in trace.context.compliance_frameworks:
                validation_result["warnings"].append(
                    "EU customer data without GDPR compliance framework"
                )
        
        # Validate feature usage tracking
        if "feature_usage" in saas_fields:
            usage_data = saas_fields["feature_usage"]
            if isinstance(usage_data, dict) and any(
                key.startswith("pii_") for key in usage_data.keys()
            ):
                validation_result["violations"].append(
                    "PII detected in feature usage tracking"
                )
                validation_result["compliant"] = False
        
        return validation_result
    
    def calculate_industry_trust_score(self, trace: WorkflowTrace) -> float:
        """Calculate SaaS-specific trust score adjustments"""
        
        base_adjustment = 0.0
        saas_fields = trace.context.industry_specific_fields
        
        # Customer tier adjustment
        tier_adjustments = {
            "enterprise": 0.1,
            "premium": 0.05,
            "standard": 0.0,
            "free": -0.05
        }
        
        customer_tier = saas_fields.get("subscription_tier", "standard")
        base_adjustment += tier_adjustments.get(customer_tier, 0.0)
        
        # Health score adjustment
        health_score = saas_fields.get("health_score", 0.8)
        if health_score < 0.5:
            base_adjustment -= 0.1
        elif health_score > 0.9:
            base_adjustment += 0.05
        
        # Churn risk adjustment
        churn_risk = saas_fields.get("churn_risk", "low")
        churn_adjustments = {
            "low": 0.02,
            "medium": 0.0,
            "high": -0.05,
            "critical": -0.1
        }
        base_adjustment += churn_adjustments.get(churn_risk, 0.0)
        
        return max(-0.2, min(0.2, base_adjustment))  # Cap adjustments
    
    def generate_industry_governance_events(self, trace: WorkflowTrace) -> List[GovernanceEvent]:
        """Generate SaaS-specific governance events"""
        
        events = []
        saas_fields = trace.context.industry_specific_fields
        
        # Billing data access event
        if "billing_info" in saas_fields and saas_fields["billing_info"].get("mrr", 0) > 0:
            events.append(GovernanceEvent(
                event_id=str(uuid.uuid4()),
                event_type="financial_data_access",
                policy_id="saas_billing_data_policy",
                policy_version="v1.0",
                decision="allowed",
                reason="Authorized access to billing data for workflow execution",
                compliance_framework="SOX",
                regulatory_requirement="Financial data handling",
                risk_level="medium"
            ))
        
        # Feature usage tracking event
        if "feature_usage" in saas_fields:
            events.append(GovernanceEvent(
                event_id=str(uuid.uuid4()),
                event_type="usage_analytics",
                policy_id="saas_analytics_policy",
                policy_version="v1.0",
                decision="allowed",
                reason="Feature usage analytics for product optimization",
                compliance_framework="GDPR",
                regulatory_requirement="Analytics data processing"
            ))
        
        return events

class BankingIndustryExtension(IndustryExtension):
    """Banking industry-specific extensions"""
    
    def __init__(self):
        config = IndustryExtensionConfig(
            industry_code=IndustryCode.BANKING,
            required_fields=[
                "banking_metrics.account_type",
                "banking_metrics.transaction_category",
                "banking_metrics.risk_rating"
            ],
            optional_fields=[
                "banking_metrics.credit_score",
                "banking_metrics.loan_details",
                "banking_metrics.kyc_status",
                "banking_metrics.aml_flags"
            ],
            compliance_frameworks=["RBI", "BASEL_III", "SOX", "GDPR"],
            regulatory_tags=["banking_regulation", "financial_services", "kyc_aml"],
            data_residency_requirements=["IN"],  # India data localization
            trust_score_adjustments={
                "high_net_worth": 0.1,
                "aml_flag_detected": -0.3,
                "kyc_incomplete": -0.2
            },
            trace_retention_days=3650  # 10 years for banking
        )
        super().__init__(config)
    
    def extend_trace_context(self, context: TraceContext) -> TraceContext:
        """Extend context with Banking-specific fields"""
        
        banking_fields = {
            "account_type": "savings",  # savings, current, loan, credit_card, investment
            "customer_category": "retail",  # retail, corporate, sme, private_banking
            "branch_code": None,
            "relationship_manager_id": None,
            "risk_profile": "medium",  # low, medium, high, very_high
            "kyc_details": {
                "kyc_status": "completed",  # pending, in_progress, completed, expired
                "kyc_last_updated": None,
                "kyc_level": "full",  # basic, full, enhanced
                "document_verification": "verified"
            },
            "aml_screening": {
                "last_screened": None,
                "pep_status": False,  # Politically Exposed Person
                "sanctions_hit": False,
                "risk_score": 0.0,
                "screening_provider": "internal"
            },
            "credit_information": {
                "credit_score": None,
                "credit_bureau": None,
                "last_updated": None,
                "credit_limit": 0.0
            },
            "transaction_limits": {
                "daily_limit": 0.0,
                "monthly_limit": 0.0,
                "per_transaction_limit": 0.0
            },
            "regulatory_flags": [],
            "basel_classification": "standard",  # standard, sub_standard, doubtful, loss
            "rbi_category": "individual"  # individual, huf, partnership, company, etc.
        }
        
        context.industry_specific_fields.update(banking_fields)
        return context
    
    def validate_industry_compliance(self, trace: WorkflowTrace) -> Dict[str, Any]:
        """Validate Banking-specific compliance requirements"""
        
        validation_result = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        banking_fields = trace.context.industry_specific_fields
        
        # RBI Data Localization Check
        if trace.context.data_residency != "IN":
            validation_result["violations"].append(
                "Banking data must be stored in India per RBI guidelines"
            )
            validation_result["compliant"] = False
        
        # KYC Compliance Check
        kyc_details = banking_fields.get("kyc_details", {})
        if kyc_details.get("kyc_status") != "completed":
            validation_result["violations"].append(
                "KYC must be completed for banking operations"
            )
            validation_result["compliant"] = False
        
        # AML Screening Check
        aml_screening = banking_fields.get("aml_screening", {})
        if aml_screening.get("sanctions_hit", False):
            validation_result["violations"].append(
                "Sanctions hit detected - transaction blocked"
            )
            validation_result["compliant"] = False
        
        # PEP Status Check
        if aml_screening.get("pep_status", False):
            validation_result["warnings"].append(
                "PEP customer - enhanced due diligence required"
            )
        
        # Basel III Classification
        basel_class = banking_fields.get("basel_classification", "standard")
        if basel_class in ["doubtful", "loss"]:
            validation_result["warnings"].append(
                f"Asset classification: {basel_class} - special monitoring required"
            )
        
        return validation_result
    
    def calculate_industry_trust_score(self, trace: WorkflowTrace) -> float:
        """Calculate Banking-specific trust score adjustments"""
        
        base_adjustment = 0.0
        banking_fields = trace.context.industry_specific_fields
        
        # Customer category adjustment
        category_adjustments = {
            "private_banking": 0.15,
            "corporate": 0.1,
            "sme": 0.05,
            "retail": 0.0
        }
        
        customer_category = banking_fields.get("customer_category", "retail")
        base_adjustment += category_adjustments.get(customer_category, 0.0)
        
        # Risk profile adjustment
        risk_adjustments = {
            "low": 0.1,
            "medium": 0.0,
            "high": -0.1,
            "very_high": -0.2
        }
        
        risk_profile = banking_fields.get("risk_profile", "medium")
        base_adjustment += risk_adjustments.get(risk_profile, 0.0)
        
        # AML flags adjustment
        aml_screening = banking_fields.get("aml_screening", {})
        if aml_screening.get("sanctions_hit", False):
            base_adjustment -= 0.5  # Severe penalty
        if aml_screening.get("pep_status", False):
            base_adjustment -= 0.1  # Moderate penalty
        
        # Basel classification adjustment
        basel_adjustments = {
            "standard": 0.0,
            "sub_standard": -0.1,
            "doubtful": -0.2,
            "loss": -0.3
        }
        
        basel_class = banking_fields.get("basel_classification", "standard")
        base_adjustment += basel_adjustments.get(basel_class, 0.0)
        
        return max(-0.5, min(0.2, base_adjustment))  # Banking has stricter penalties
    
    def generate_industry_governance_events(self, trace: WorkflowTrace) -> List[GovernanceEvent]:
        """Generate Banking-specific governance events"""
        
        events = []
        banking_fields = trace.context.industry_specific_fields
        
        # KYC verification event
        kyc_details = banking_fields.get("kyc_details", {})
        if kyc_details.get("kyc_status") == "completed":
            events.append(GovernanceEvent(
                event_id=str(uuid.uuid4()),
                event_type="kyc_verification",
                policy_id="rbi_kyc_policy",
                policy_version="v2.0",
                decision="allowed",
                reason="KYC verification completed",
                compliance_framework="RBI",
                regulatory_requirement="KYC compliance",
                risk_level="low"
            ))
        
        # AML screening event
        aml_screening = banking_fields.get("aml_screening", {})
        if aml_screening.get("last_screened"):
            decision = "denied" if aml_screening.get("sanctions_hit", False) else "allowed"
            events.append(GovernanceEvent(
                event_id=str(uuid.uuid4()),
                event_type="aml_screening",
                policy_id="aml_screening_policy",
                policy_version="v1.0",
                decision=decision,
                reason=f"AML screening result: {'Hit' if aml_screening.get('sanctions_hit') else 'Clear'}",
                compliance_framework="RBI",
                regulatory_requirement="AML compliance",
                risk_level="high" if aml_screening.get("sanctions_hit") else "low"
            ))
        
        return events

class InsuranceIndustryExtension(IndustryExtension):
    """Insurance industry-specific extensions"""
    
    def __init__(self):
        config = IndustryExtensionConfig(
            industry_code=IndustryCode.INSURANCE,
            required_fields=[
                "insurance_metrics.policy_type",
                "insurance_metrics.coverage_amount",
                "insurance_metrics.risk_assessment"
            ],
            optional_fields=[
                "insurance_metrics.claim_history",
                "insurance_metrics.underwriting_score",
                "insurance_metrics.actuarial_data",
                "insurance_metrics.reinsurance_details"
            ],
            compliance_frameworks=["IRDAI", "SOLVENCY_II", "SOX"],
            regulatory_tags=["insurance_regulation", "solvency", "claims_processing"],
            trust_score_adjustments={
                "high_value_policy": 0.1,
                "frequent_claims": -0.15,
                "fraud_indicator": -0.3
            },
            trace_retention_days=2555  # 7 years for insurance
        )
        super().__init__(config)
    
    def extend_trace_context(self, context: TraceContext) -> TraceContext:
        """Extend context with Insurance-specific fields"""
        
        insurance_fields = {
            "policy_type": "life",  # life, health, motor, property, marine, etc.
            "policy_status": "active",  # active, lapsed, surrendered, matured
            "coverage_details": {
                "sum_assured": 0.0,
                "premium_amount": 0.0,
                "premium_frequency": "annual",  # annual, semi_annual, quarterly, monthly
                "policy_term": 0,  # in years
                "premium_paying_term": 0
            },
            "underwriting_info": {
                "underwriting_score": 0.0,
                "risk_category": "standard",  # preferred, standard, sub_standard, declined
                "medical_exam_required": False,
                "financial_underwriting": False,
                "underwriter_id": None
            },
            "claims_history": {
                "total_claims": 0,
                "claims_amount": 0.0,
                "last_claim_date": None,
                "claim_frequency": 0.0,
                "fraud_flags": []
            },
            "actuarial_data": {
                "mortality_rating": "standard",
                "morbidity_rating": "standard",
                "loading_percentage": 0.0,
                "discount_percentage": 0.0
            },
            "agent_details": {
                "agent_id": None,
                "agent_license": None,
                "commission_rate": 0.0,
                "agency_code": None
            },
            "reinsurance": {
                "reinsured": False,
                "reinsurer_name": None,
                "retention_amount": 0.0,
                "cession_rate": 0.0
            },
            "solvency_impact": {
                "capital_requirement": 0.0,
                "risk_margin": 0.0,
                "technical_provisions": 0.0
            },
            "regulatory_returns": [],  # List of regulatory returns this affects
            "irdai_category": "individual"  # individual, group, corporate
        }
        
        context.industry_specific_fields.update(insurance_fields)
        return context
    
    def validate_industry_compliance(self, trace: WorkflowTrace) -> Dict[str, Any]:
        """Validate Insurance-specific compliance requirements"""
        
        validation_result = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        insurance_fields = trace.context.industry_specific_fields
        
        # IRDAI Compliance Checks
        coverage_details = insurance_fields.get("coverage_details", {})
        sum_assured = coverage_details.get("sum_assured", 0)
        
        # High value policy checks
        if sum_assured > 10000000:  # 1 Crore
            underwriting_info = insurance_fields.get("underwriting_info", {})
            if not underwriting_info.get("medical_exam_required", False):
                validation_result["warnings"].append(
                    "High value policy without medical exam - IRDAI guidelines check required"
                )
        
        # Claims fraud detection
        claims_history = insurance_fields.get("claims_history", {})
        fraud_flags = claims_history.get("fraud_flags", [])
        if fraud_flags:
            validation_result["violations"].append(
                f"Fraud indicators detected: {', '.join(fraud_flags)}"
            )
            validation_result["compliant"] = False
        
        # Agent licensing check
        agent_details = insurance_fields.get("agent_details", {})
        if agent_details.get("agent_id") and not agent_details.get("agent_license"):
            validation_result["violations"].append(
                "Agent license information missing"
            )
            validation_result["compliant"] = False
        
        # Solvency II compliance (if applicable)
        if "SOLVENCY_II" in trace.context.compliance_frameworks:
            solvency_impact = insurance_fields.get("solvency_impact", {})
            if solvency_impact.get("capital_requirement", 0) == 0:
                validation_result["warnings"].append(
                    "Solvency II capital requirement not calculated"
                )
        
        return validation_result
    
    def calculate_industry_trust_score(self, trace: WorkflowTrace) -> float:
        """Calculate Insurance-specific trust score adjustments"""
        
        base_adjustment = 0.0
        insurance_fields = trace.context.industry_specific_fields
        
        # Policy type adjustment
        type_adjustments = {
            "life": 0.05,
            "health": 0.0,
            "motor": -0.02,
            "property": 0.02,
            "marine": 0.03
        }
        
        policy_type = insurance_fields.get("policy_type", "life")
        base_adjustment += type_adjustments.get(policy_type, 0.0)
        
        # Claims history adjustment
        claims_history = insurance_fields.get("claims_history", {})
        claim_frequency = claims_history.get("claim_frequency", 0.0)
        
        if claim_frequency > 0.5:  # More than 0.5 claims per year
            base_adjustment -= 0.1
        elif claim_frequency == 0:  # No claims
            base_adjustment += 0.05
        
        # Fraud flags severe penalty
        fraud_flags = claims_history.get("fraud_flags", [])
        if fraud_flags:
            base_adjustment -= 0.3
        
        # Underwriting score adjustment
        underwriting_info = insurance_fields.get("underwriting_info", {})
        risk_category = underwriting_info.get("risk_category", "standard")
        
        risk_adjustments = {
            "preferred": 0.1,
            "standard": 0.0,
            "sub_standard": -0.1,
            "declined": -0.3
        }
        
        base_adjustment += risk_adjustments.get(risk_category, 0.0)
        
        return max(-0.4, min(0.2, base_adjustment))
    
    def generate_industry_governance_events(self, trace: WorkflowTrace) -> List[GovernanceEvent]:
        """Generate Insurance-specific governance events"""
        
        events = []
        insurance_fields = trace.context.industry_specific_fields
        
        # Underwriting decision event
        underwriting_info = insurance_fields.get("underwriting_info", {})
        if underwriting_info.get("underwriting_score", 0) > 0:
            events.append(GovernanceEvent(
                event_id=str(uuid.uuid4()),
                event_type="underwriting_decision",
                policy_id="irdai_underwriting_policy",
                policy_version="v1.0",
                decision="allowed",
                reason=f"Underwriting approved with score {underwriting_info['underwriting_score']}",
                compliance_framework="IRDAI",
                regulatory_requirement="Underwriting guidelines",
                risk_level="medium"
            ))
        
        # Claims processing event
        claims_history = insurance_fields.get("claims_history", {})
        if claims_history.get("total_claims", 0) > 0:
            events.append(GovernanceEvent(
                event_id=str(uuid.uuid4()),
                event_type="claims_processing",
                policy_id="claims_processing_policy",
                policy_version="v1.0",
                decision="allowed",
                reason="Claims history accessed for processing",
                compliance_framework="IRDAI",
                regulatory_requirement="Claims handling",
                risk_level="low"
            ))
        
        return events

class CrossIndustryExtensionManager:
    """Manages cross-industry schema extensions"""
    
    def __init__(self):
        self.extensions = {
            IndustryCode.SAAS: SaaSIndustryExtension(),
            IndustryCode.BANKING: BankingIndustryExtension(),
            IndustryCode.INSURANCE: InsuranceIndustryExtension()
        }
        
        # Register additional extensions dynamically
        self._register_additional_extensions()
    
    def _register_additional_extensions(self):
        """Register additional industry extensions"""
        
        # E-commerce extension (simplified)
        ecommerce_config = IndustryExtensionConfig(
            industry_code=IndustryCode.ECOMMERCE,
            required_fields=["ecommerce_metrics.order_value", "ecommerce_metrics.customer_segment"],
            compliance_frameworks=["GDPR", "CCPA", "PCI_DSS"],
            regulatory_tags=["payment_processing", "customer_data"]
        )
        
        # Healthcare extension (simplified)
        healthcare_config = IndustryExtensionConfig(
            industry_code=IndustryCode.HEALTHCARE,
            required_fields=["healthcare_metrics.patient_category", "healthcare_metrics.phi_present"],
            compliance_frameworks=["HIPAA", "GDPR"],
            regulatory_tags=["phi_handling", "medical_records"],
            trace_retention_days=2190  # 6 years for healthcare
        )
        
        # Note: Full implementations would be similar to the detailed ones above
    
    def get_extension(self, industry_code: IndustryCode) -> Optional[IndustryExtension]:
        """Get industry extension by code"""
        return self.extensions.get(industry_code)
    
    def apply_industry_extension(self, trace: WorkflowTrace) -> WorkflowTrace:
        """Apply industry-specific extensions to trace"""
        
        industry_code_str = trace.context.industry_code
        
        try:
            industry_code = IndustryCode(industry_code_str)
        except ValueError:
            # Unknown industry code - use default behavior
            return trace
        
        extension = self.get_extension(industry_code)
        if not extension:
            return trace
        
        # Apply extensions
        trace.context = extension.extend_trace_context(trace.context)
        
        # Add industry-specific governance events
        industry_events = extension.generate_industry_governance_events(trace)
        trace.governance_events.extend(industry_events)
        
        # Adjust trust score
        trust_adjustment = extension.calculate_industry_trust_score(trace)
        trace.trust_score = max(0.0, min(1.0, trace.trust_score + trust_adjustment))
        
        # Update trust factors
        trace.trust_factors[f"{industry_code.value}_adjustment"] = trust_adjustment
        
        return trace
    
    def validate_industry_compliance(self, trace: WorkflowTrace) -> Dict[str, Any]:
        """Validate trace against industry-specific compliance requirements"""
        
        industry_code_str = trace.context.industry_code
        
        try:
            industry_code = IndustryCode(industry_code_str)
        except ValueError:
            return {"compliant": True, "violations": [], "warnings": ["Unknown industry code"]}
        
        extension = self.get_extension(industry_code)
        if not extension:
            return {"compliant": True, "violations": [], "warnings": ["No extension available"]}
        
        return extension.validate_industry_compliance(trace)
    
    def get_supported_industries(self) -> List[str]:
        """Get list of supported industry codes"""
        return [code.value for code in self.extensions.keys()]
    
    def get_industry_requirements(self, industry_code: IndustryCode) -> Dict[str, Any]:
        """Get requirements for specific industry"""
        
        extension = self.get_extension(industry_code)
        if not extension:
            return {}
        
        return {
            "industry_code": industry_code.value,
            "required_fields": extension.config.required_fields,
            "optional_fields": extension.config.optional_fields,
            "compliance_frameworks": extension.config.compliance_frameworks,
            "regulatory_tags": extension.config.regulatory_tags,
            "data_residency_requirements": extension.config.data_residency_requirements,
            "trace_retention_days": extension.config.trace_retention_days
        }

# Global extension manager instance
extension_manager = CrossIndustryExtensionManager()

# Utility functions for easy integration
def apply_industry_extensions(trace: WorkflowTrace) -> WorkflowTrace:
    """Apply industry extensions to trace"""
    return extension_manager.apply_industry_extension(trace)

def validate_industry_compliance(trace: WorkflowTrace) -> Dict[str, Any]:
    """Validate industry compliance"""
    return extension_manager.validate_industry_compliance(trace)

def get_industry_requirements(industry_code: str) -> Dict[str, Any]:
    """Get industry requirements"""
    try:
        code = IndustryCode(industry_code)
        return extension_manager.get_industry_requirements(code)
    except ValueError:
        return {}

# Example usage
if __name__ == "__main__":
    try:
        from .canonical_trace_schema import TraceBuilder
    except ImportError:
        from canonical_trace_schema import TraceBuilder
    
    # Create trace builder
    builder = TraceBuilder()
    
    # Create sample SaaS trace
    saas_trace = builder.create_workflow_trace(
        workflow_id="saas_pipeline_hygiene",
        execution_id=str(uuid.uuid4()),
        workflow_version="v1.0",
        tenant_id=1000,
        user_id="user123",
        actor_name="saas_user"
    )
    
    # Set industry code
    saas_trace.context.industry_code = "SaaS"
    
    # Apply industry extensions
    extended_trace = apply_industry_extensions(saas_trace)
    
    print(f"✅ Applied SaaS extensions to trace: {extended_trace.trace_id}")
    print(f"SaaS fields: {list(extended_trace.context.industry_specific_fields.keys())}")
    
    # Validate compliance
    compliance_result = validate_industry_compliance(extended_trace)
    print(f"Compliance result: {compliance_result}")
    
    # Test banking extension
    banking_trace = builder.create_workflow_trace(
        workflow_id="banking_loan_approval",
        execution_id=str(uuid.uuid4()),
        workflow_version="v1.0",
        tenant_id=2000,
        user_id="banker123",
        actor_name="bank_user"
    )
    
    banking_trace.context.industry_code = "BANK"
    banking_trace.context.data_residency = "IN"
    
    extended_banking_trace = apply_industry_extensions(banking_trace)
    
    print(f"✅ Applied Banking extensions to trace: {extended_banking_trace.trace_id}")
    print(f"Banking fields: {list(extended_banking_trace.context.industry_specific_fields.keys())}")
    
    banking_compliance = validate_industry_compliance(extended_banking_trace)
    print(f"Banking compliance: {banking_compliance}")
    
    print("✅ Cross-industry schema extensions implementation complete")
