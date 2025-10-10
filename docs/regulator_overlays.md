# Regulator Overlays Documentation

**Task 7.4-T31: Document regulator overlays (SOX, RBI, HIPAA)**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Compliance Team

---

## Overview

This document defines regulatory compliance overlays for evidence packs, ensuring that all workflow executions meet jurisdiction-specific requirements. Each overlay specifies additional fields, validation rules, and retention policies required by regulatory frameworks.

---

## SOX (Sarbanes-Oxley Act) Overlay

### Scope
**Applicable To:** All tenants with financial reporting requirements  
**Industries:** SaaS, Banking, Insurance, Financial Services  
**Jurisdiction:** United States  
**Effective Date:** 2024-01-01

### Required Evidence Fields

```json
{
  "sox_overlay": {
    "financial_controls": {
      "control_id": "string, required - Internal control identifier",
      "control_type": "enum: preventive|detective|corrective",
      "control_owner": "string, required - Control owner employee ID",
      "control_frequency": "enum: daily|weekly|monthly|quarterly|annual",
      "control_automation_level": "enum: manual|semi_automated|fully_automated"
    },
    "segregation_of_duties": {
      "maker_id": "string, required - User who initiated transaction",
      "checker_id": "string, required - User who approved transaction", 
      "maker_role": "string, required - Role of initiating user",
      "checker_role": "string, required - Role of approving user",
      "sod_validation": "boolean, required - True if SoD rules passed",
      "sod_exceptions": "array - List of any SoD exceptions with justification"
    },
    "financial_impact": {
      "revenue_impact": "decimal - Impact on revenue recognition",
      "expense_impact": "decimal - Impact on expense recognition", 
      "balance_sheet_impact": "decimal - Impact on balance sheet items",
      "materiality_threshold": "decimal - Materiality threshold applied",
      "is_material": "boolean - Whether impact exceeds materiality threshold"
    },
    "audit_trail": {
      "journal_entry_id": "string - Related journal entry if applicable",
      "supporting_documents": "array - List of supporting document IDs",
      "approval_chain": "array - Complete approval chain with timestamps",
      "system_generated": "boolean - Whether transaction was system generated",
      "manual_intervention": "boolean - Whether manual intervention occurred"
    },
    "retention_metadata": {
      "retention_period_years": "integer, default: 7 - SOX requires 7 years",
      "legal_hold_flag": "boolean - Whether under legal hold",
      "destruction_date": "date - Scheduled destruction date",
      "retention_justification": "string - Business justification for retention"
    }
  }
}
```

### Validation Rules

```javascript
// SOX Validation Rules
function validateSOXCompliance(evidencePack) {
  const rules = [
    {
      rule: "segregation_of_duties",
      validation: evidencePack.sox_overlay.segregation_of_duties.maker_id !== 
                 evidencePack.sox_overlay.segregation_of_duties.checker_id,
      message: "Maker and checker must be different users"
    },
    {
      rule: "financial_controls_owner",
      validation: evidencePack.sox_overlay.financial_controls.control_owner !== null,
      message: "All financial controls must have an assigned owner"
    },
    {
      rule: "material_transaction_approval",
      validation: !evidencePack.sox_overlay.financial_impact.is_material || 
                 evidencePack.sox_overlay.audit_trail.approval_chain.length >= 2,
      message: "Material transactions require at least 2-level approval"
    },
    {
      rule: "audit_trail_completeness",
      validation: evidencePack.sox_overlay.audit_trail.supporting_documents.length > 0,
      message: "All transactions must have supporting documentation"
    }
  ];
  
  return rules.every(rule => rule.validation);
}
```

### Retention Policy
- **Standard Retention:** 7 years from transaction date
- **Legal Hold:** Indefinite retention during litigation
- **Destruction:** Automated after retention period unless under legal hold
- **Backup:** Quarterly backups maintained for disaster recovery

---

## RBI (Reserve Bank of India) Overlay

### Scope
**Applicable To:** Banking and financial institutions in India  
**Industries:** Banking, NBFC, Payment Systems  
**Jurisdiction:** India  
**Effective Date:** 2024-01-01

### Required Evidence Fields

```json
{
  "rbi_overlay": {
    "regulatory_compliance": {
      "rbi_circular_reference": "string, required - Applicable RBI circular number",
      "compliance_officer_id": "string, required - Compliance officer employee ID",
      "regulatory_reporting_required": "boolean - Whether regulatory reporting needed",
      "reporting_deadline": "date - Deadline for regulatory reporting",
      "compliance_status": "enum: compliant|non_compliant|under_review"
    },
    "customer_protection": {
      "customer_consent": "boolean, required - Customer consent obtained",
      "consent_timestamp": "datetime - When consent was obtained",
      "consent_method": "enum: digital|physical|verbal - Method of consent",
      "grievance_redressal": "boolean - Grievance mechanism available",
      "fair_practices_code": "boolean - Fair practices code followed"
    },
    "data_localization": {
      "data_residency": "enum: india|offshore - Where data is stored",
      "cross_border_transfer": "boolean - Whether data crossed borders",
      "transfer_justification": "string - Justification for cross-border transfer",
      "data_classification": "enum: critical|sensitive|normal - RBI data classification",
      "encryption_applied": "boolean - Whether data is encrypted"
    },
    "risk_management": {
      "risk_category": "enum: credit|operational|market|liquidity - Risk type",
      "risk_rating": "enum: low|medium|high|critical - Risk rating",
      "risk_mitigation": "array - Risk mitigation measures applied",
      "capital_impact": "decimal - Impact on capital adequacy",
      "provisioning_required": "decimal - Provisioning amount required"
    },
    "kyc_aml": {
      "kyc_status": "enum: completed|pending|rejected - KYC status",
      "kyc_last_updated": "date - Last KYC update date",
      "aml_screening": "boolean - AML screening performed",
      "pep_screening": "boolean - PEP screening performed",
      "sanctions_screening": "boolean - Sanctions list screening performed",
      "suspicious_activity": "boolean - Suspicious activity detected"
    },
    "audit_requirements": {
      "internal_audit_required": "boolean - Internal audit requirement",
      "external_audit_required": "boolean - External audit requirement", 
      "rbi_inspection_ready": "boolean - Ready for RBI inspection",
      "documentation_complete": "boolean - All documentation complete",
      "audit_trail_preserved": "boolean - Audit trail preservation confirmed"
    }
  }
}
```

### Validation Rules

```javascript
// RBI Validation Rules
function validateRBICompliance(evidencePack) {
  const rules = [
    {
      rule: "data_localization",
      validation: evidencePack.rbi_overlay.data_localization.data_residency === "india" ||
                 evidencePack.rbi_overlay.data_localization.transfer_justification !== null,
      message: "Critical data must be stored in India or have valid transfer justification"
    },
    {
      rule: "customer_consent",
      validation: evidencePack.rbi_overlay.customer_protection.customer_consent === true,
      message: "Customer consent is mandatory for all data processing"
    },
    {
      rule: "kyc_compliance",
      validation: evidencePack.rbi_overlay.kyc_aml.kyc_status === "completed",
      message: "KYC must be completed before transaction processing"
    },
    {
      rule: "aml_screening",
      validation: evidencePack.rbi_overlay.kyc_aml.aml_screening === true &&
                 evidencePack.rbi_overlay.kyc_aml.sanctions_screening === true,
      message: "AML and sanctions screening mandatory for all transactions"
    },
    {
      rule: "risk_assessment",
      validation: evidencePack.rbi_overlay.risk_management.risk_rating !== null,
      message: "Risk assessment required for all banking transactions"
    }
  ];
  
  return rules.every(rule => rule.validation);
}
```

### Retention Policy
- **Transaction Records:** 10 years from transaction completion
- **KYC Documents:** 10 years after account closure
- **AML Records:** 10 years from transaction date
- **Regulatory Reports:** 10 years from submission
- **Audit Trails:** Permanent retention for critical transactions

---

## HIPAA (Health Insurance Portability and Accountability Act) Overlay

### Scope
**Applicable To:** Healthcare providers, insurers, and business associates  
**Industries:** Healthcare, Insurance (Health), SaaS (Healthcare)  
**Jurisdiction:** United States  
**Effective Date:** 2024-01-01

### Required Evidence Fields

```json
{
  "hipaa_overlay": {
    "phi_handling": {
      "phi_present": "boolean, required - Whether PHI is present in data",
      "phi_categories": "array - Types of PHI present (medical, financial, etc.)",
      "minimum_necessary": "boolean - Minimum necessary standard applied",
      "purpose_limitation": "string, required - Purpose for PHI processing",
      "authorized_users": "array - List of authorized users who accessed PHI"
    },
    "patient_rights": {
      "patient_consent": "boolean - Patient consent obtained",
      "consent_type": "enum: general|specific|research - Type of consent",
      "consent_date": "date - Date consent was obtained",
      "right_to_access": "boolean - Patient right to access honored",
      "right_to_amend": "boolean - Patient right to amend honored",
      "accounting_of_disclosures": "boolean - Disclosure accounting maintained"
    },
    "security_safeguards": {
      "access_controls": "boolean, required - Access controls implemented",
      "audit_controls": "boolean, required - Audit controls active",
      "integrity": "boolean, required - Data integrity measures applied",
      "transmission_security": "boolean, required - Secure transmission used",
      "encryption_at_rest": "boolean - Data encrypted at rest",
      "encryption_in_transit": "boolean - Data encrypted in transit"
    },
    "breach_prevention": {
      "risk_assessment": "boolean - Risk assessment performed",
      "vulnerability_scan": "boolean - Vulnerability scanning performed",
      "penetration_test": "boolean - Penetration testing performed",
      "incident_response": "boolean - Incident response plan active",
      "breach_notification": "boolean - Breach notification procedures ready"
    },
    "business_associate": {
      "ba_agreement": "boolean - Business Associate Agreement in place",
      "ba_compliance": "boolean - BA compliance verified",
      "subcontractor_agreements": "array - Subcontractor agreements",
      "due_diligence": "boolean - Due diligence performed on BAs",
      "monitoring": "boolean - BA monitoring procedures active"
    },
    "administrative_safeguards": {
      "security_officer": "string - Security officer assigned",
      "workforce_training": "boolean - Workforce training completed",
      "access_management": "boolean - Access management procedures active",
      "contingency_plan": "boolean - Contingency plan in place",
      "evaluation": "boolean - Regular security evaluations performed"
    }
  }
}
```

### Validation Rules

```javascript
// HIPAA Validation Rules
function validateHIPAACompliance(evidencePack) {
  const rules = [
    {
      rule: "phi_authorization",
      validation: !evidencePack.hipaa_overlay.phi_handling.phi_present ||
                 evidencePack.hipaa_overlay.patient_rights.patient_consent === true,
      message: "PHI processing requires patient authorization"
    },
    {
      rule: "minimum_necessary",
      validation: !evidencePack.hipaa_overlay.phi_handling.phi_present ||
                 evidencePack.hipaa_overlay.phi_handling.minimum_necessary === true,
      message: "Minimum necessary standard must be applied to PHI"
    },
    {
      rule: "encryption_required",
      validation: !evidencePack.hipaa_overlay.phi_handling.phi_present ||
                 (evidencePack.hipaa_overlay.security_safeguards.encryption_at_rest === true &&
                  evidencePack.hipaa_overlay.security_safeguards.encryption_in_transit === true),
      message: "PHI must be encrypted at rest and in transit"
    },
    {
      rule: "access_controls",
      validation: evidencePack.hipaa_overlay.security_safeguards.access_controls === true,
      message: "Access controls are mandatory for all healthcare systems"
    },
    {
      rule: "audit_controls",
      validation: evidencePack.hipaa_overlay.security_safeguards.audit_controls === true,
      message: "Audit controls must be implemented and active"
    },
    {
      rule: "business_associate_agreement",
      validation: evidencePack.hipaa_overlay.business_associate.ba_agreement === true,
      message: "Business Associate Agreement required for third-party processing"
    }
  ];
  
  return rules.every(rule => rule.validation);
}
```

### Retention Policy
- **Medical Records:** 6 years from creation or last use
- **PHI Access Logs:** 6 years from access date
- **Security Incident Records:** 6 years from incident resolution
- **Training Records:** 6 years from completion
- **Business Associate Agreements:** 6 years after termination

---

## GDPR (General Data Protection Regulation) Overlay

### Scope
**Applicable To:** All organizations processing EU personal data  
**Industries:** All industries with EU data subjects  
**Jurisdiction:** European Union  
**Effective Date:** 2024-01-01

### Required Evidence Fields

```json
{
  "gdpr_overlay": {
    "lawful_basis": {
      "processing_basis": "enum: consent|contract|legal_obligation|vital_interests|public_task|legitimate_interests",
      "basis_justification": "string, required - Justification for processing basis",
      "consent_obtained": "boolean - Whether explicit consent obtained",
      "consent_timestamp": "datetime - When consent was obtained",
      "consent_withdrawable": "boolean - Whether consent can be withdrawn"
    },
    "data_subject_rights": {
      "right_to_access": "boolean - Right to access honored",
      "right_to_rectification": "boolean - Right to rectification honored",
      "right_to_erasure": "boolean - Right to erasure honored",
      "right_to_portability": "boolean - Right to portability honored",
      "right_to_object": "boolean - Right to object honored",
      "automated_decision_making": "boolean - Automated decision making used"
    },
    "data_protection": {
      "data_minimization": "boolean, required - Data minimization applied",
      "purpose_limitation": "boolean, required - Purpose limitation respected",
      "accuracy": "boolean, required - Data accuracy maintained",
      "storage_limitation": "boolean, required - Storage limitation applied",
      "pseudonymization": "boolean - Pseudonymization applied",
      "anonymization": "boolean - Anonymization applied"
    },
    "cross_border_transfer": {
      "transfer_occurred": "boolean - Whether data crossed EU borders",
      "adequacy_decision": "boolean - Adequacy decision exists",
      "safeguards_applied": "array - Safeguards applied for transfer",
      "transfer_mechanism": "enum: adequacy|sccs|bcrs|derogation - Transfer mechanism",
      "third_country": "string - Third country receiving data"
    },
    "privacy_by_design": {
      "dpia_conducted": "boolean - Data Protection Impact Assessment conducted",
      "dpia_outcome": "enum: low_risk|high_risk|unacceptable_risk - DPIA outcome",
      "privacy_measures": "array - Privacy measures implemented",
      "data_protection_officer": "string - DPO contact information",
      "privacy_notice": "boolean - Privacy notice provided"
    }
  }
}
```

### Validation Rules

```javascript
// GDPR Validation Rules
function validateGDPRCompliance(evidencePack) {
  const rules = [
    {
      rule: "lawful_basis_required",
      validation: evidencePack.gdpr_overlay.lawful_basis.processing_basis !== null,
      message: "Lawful basis required for all personal data processing"
    },
    {
      rule: "consent_validity",
      validation: evidencePack.gdpr_overlay.lawful_basis.processing_basis !== "consent" ||
                 evidencePack.gdpr_overlay.lawful_basis.consent_obtained === true,
      message: "Valid consent required when consent is the lawful basis"
    },
    {
      rule: "data_minimization",
      validation: evidencePack.gdpr_overlay.data_protection.data_minimization === true,
      message: "Data minimization principle must be applied"
    },
    {
      rule: "purpose_limitation",
      validation: evidencePack.gdpr_overlay.data_protection.purpose_limitation === true,
      message: "Purpose limitation principle must be respected"
    },
    {
      rule: "cross_border_safeguards",
      validation: !evidencePack.gdpr_overlay.cross_border_transfer.transfer_occurred ||
                 evidencePack.gdpr_overlay.cross_border_transfer.safeguards_applied.length > 0,
      message: "Safeguards required for cross-border data transfers"
    }
  ];
  
  return rules.every(rule => rule.validation);
}
```

### Retention Policy
- **Personal Data:** As specified in privacy notice, typically 2-7 years
- **Consent Records:** 3 years after consent withdrawal
- **Processing Records:** 3 years after processing completion
- **DPIA Records:** 3 years after project completion
- **Breach Records:** 5 years from breach resolution

---

## Implementation Guidelines

### Evidence Pack Generation
```python
def generate_evidence_pack_with_overlays(workflow_execution, tenant_config):
    """Generate evidence pack with appropriate regulatory overlays"""
    
    base_evidence = create_base_evidence_pack(workflow_execution)
    
    # Apply industry-specific overlays
    if tenant_config.industry == "BANK" and tenant_config.region == "IN":
        base_evidence.update(apply_rbi_overlay(workflow_execution))
    
    if tenant_config.compliance_frameworks.includes("SOX"):
        base_evidence.update(apply_sox_overlay(workflow_execution))
    
    if tenant_config.compliance_frameworks.includes("HIPAA"):
        base_evidence.update(apply_hipaa_overlay(workflow_execution))
    
    if tenant_config.region in ["EU", "UK"] or tenant_config.has_eu_data_subjects:
        base_evidence.update(apply_gdpr_overlay(workflow_execution))
    
    # Validate compliance
    validate_regulatory_compliance(base_evidence, tenant_config)
    
    return base_evidence
```

### Validation Framework
```python
class RegulatoryValidator:
    """Validates evidence packs against regulatory requirements"""
    
    def __init__(self):
        self.validators = {
            "SOX": validateSOXCompliance,
            "RBI": validateRBICompliance, 
            "HIPAA": validateHIPAACompliance,
            "GDPR": validateGDPRCompliance
        }
    
    def validate(self, evidence_pack, frameworks):
        """Validate evidence pack against specified frameworks"""
        results = {}
        
        for framework in frameworks:
            if framework in self.validators:
                results[framework] = self.validators[framework](evidence_pack)
            else:
                results[framework] = {"status": "unknown_framework"}
        
        return results
```

### Retention Management
```python
class RetentionManager:
    """Manages retention policies for regulatory compliance"""
    
    RETENTION_POLICIES = {
        "SOX": {"years": 7, "legal_hold_supported": True},
        "RBI": {"years": 10, "legal_hold_supported": True},
        "HIPAA": {"years": 6, "legal_hold_supported": True},
        "GDPR": {"years": 3, "legal_hold_supported": False, "erasure_required": True}
    }
    
    def calculate_retention_period(self, evidence_pack):
        """Calculate retention period based on applicable regulations"""
        frameworks = evidence_pack.get("compliance_frameworks", [])
        max_retention = 0
        
        for framework in frameworks:
            if framework in self.RETENTION_POLICIES:
                retention = self.RETENTION_POLICIES[framework]["years"]
                max_retention = max(max_retention, retention)
        
        return max_retention
```

---

## Compliance Monitoring

### Automated Checks
- **Real-time Validation:** Evidence packs validated during generation
- **Batch Validation:** Daily batch validation of all evidence packs
- **Compliance Scoring:** Automated compliance scoring per framework
- **Alert Generation:** Automated alerts for compliance violations

### Reporting
- **Compliance Dashboards:** Real-time compliance status dashboards
- **Regulatory Reports:** Automated generation of regulatory reports
- **Audit Trails:** Complete audit trails for regulatory inspections
- **Breach Notifications:** Automated breach notification workflows

### Maintenance
- **Regulatory Updates:** Quarterly review of regulatory requirements
- **Overlay Updates:** Updates to overlays based on regulatory changes
- **Validation Updates:** Updates to validation rules and logic
- **Training Updates:** Regular training on new compliance requirements

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** compliance-team@company.com
