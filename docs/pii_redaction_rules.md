# PII Redaction Rules for RBA Traces

**Task 7.1-T24: Document redaction rules for PII in traces**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Compliance Team

---

## Overview

This document defines comprehensive PII (Personally Identifiable Information) redaction rules for RBA trace data to ensure compliance with GDPR, DPDP, HIPAA, and other privacy regulations. All trace data must be processed according to these rules before storage or transmission.

---

## PII Classification Framework

### **Level 1: Direct Identifiers (MUST REDACT)**
- **Social Security Numbers (SSN)**
- **National ID Numbers** (Aadhaar, PAN, etc.)
- **Passport Numbers**
- **Driver's License Numbers**
- **Credit Card Numbers** (PCI-DSS scope)
- **Bank Account Numbers**
- **Medical Record Numbers**
- **Biometric Data** (fingerprints, facial recognition)

### **Level 2: Quasi-Identifiers (CONDITIONAL REDACTION)**
- **Full Names** (first + last name combinations)
- **Email Addresses** (personal domains)
- **Phone Numbers** (mobile/landline)
- **Physical Addresses** (street level)
- **IP Addresses** (when linked to individuals)
- **Device IDs** (when persistent)
- **Employee IDs** (in certain contexts)

### **Level 3: Sensitive Personal Data (CONTEXT-DEPENDENT)**
- **Date of Birth** (exact dates)
- **Age** (when combined with other identifiers)
- **Gender** (in small populations)
- **Race/Ethnicity**
- **Religious Affiliation**
- **Political Opinions**
- **Health Information**
- **Financial Status**
- **Location Data** (precise coordinates)

---

## Dynamic Redaction Rules

### **Industry-Specific Rules**

```yaml
# SaaS Industry Rules
saas_redaction_rules:
  always_redact:
    - email_addresses
    - phone_numbers
    - credit_card_numbers
    - ssn
  conditional_redact:
    - full_names: "when not business contacts"
    - ip_addresses: "when linked to individuals"
  preserve:
    - company_names
    - business_titles
    - aggregate_metrics

# Banking Industry Rules (RBI Compliance)
banking_redaction_rules:
  always_redact:
    - account_numbers
    - pan_numbers
    - aadhaar_numbers
    - credit_scores
    - transaction_details
  conditional_redact:
    - customer_names: "when not authorized personnel"
    - branch_codes: "when combined with personal data"
  preserve:
    - transaction_types
    - aggregate_amounts
    - risk_categories

# Insurance Industry Rules (IRDAI Compliance)
insurance_redaction_rules:
  always_redact:
    - policy_numbers
    - claim_amounts
    - medical_conditions
    - beneficiary_details
  conditional_redact:
    - agent_names: "when not business context"
    - location_data: "when precise"
  preserve:
    - policy_types
    - coverage_categories
    - risk_assessments
```

### **Context-Aware Redaction**

```python
def determine_redaction_level(field_name: str, context: Dict[str, Any]) -> str:
    """
    Dynamically determine redaction level based on context
    """
    tenant_id = context.get('tenant_id')
    industry = context.get('industry_code')
    user_role = context.get('user_role')
    data_purpose = context.get('data_purpose')
    
    # Get tenant-specific rules
    tenant_rules = get_tenant_redaction_rules(tenant_id)
    
    # Get industry-specific rules
    industry_rules = get_industry_redaction_rules(industry)
    
    # Get role-based access rules
    role_rules = get_role_based_rules(user_role)
    
    # Combine rules with precedence: tenant > industry > role > default
    combined_rules = merge_redaction_rules(
        tenant_rules, industry_rules, role_rules
    )
    
    return combined_rules.get(field_name, 'redact')
```

---

## Redaction Techniques

### **1. Complete Redaction**
```json
{
  "original": "john.doe@example.com",
  "redacted": "[REDACTED_EMAIL]",
  "technique": "complete_removal"
}
```

### **2. Partial Redaction**
```json
{
  "original": "john.doe@example.com",
  "redacted": "j***@example.com",
  "technique": "partial_masking"
}
```

### **3. Tokenization**
```json
{
  "original": "john.doe@example.com",
  "redacted": "TOKEN_EMAIL_12345",
  "technique": "tokenization",
  "reversible": true
}
```

### **4. Hashing**
```json
{
  "original": "john.doe@example.com",
  "redacted": "sha256:a1b2c3d4e5f6...",
  "technique": "cryptographic_hash",
  "reversible": false
}
```

### **5. Generalization**
```json
{
  "original": "1985-03-15",
  "redacted": "1980s",
  "technique": "generalization"
}
```

### **6. Synthetic Replacement**
```json
{
  "original": "John Doe",
  "redacted": "Person_A",
  "technique": "synthetic_replacement"
}
```

---

## Implementation Guidelines

### **Trace-Level Configuration**

```python
@dataclass
class PIIRedactionConfig:
    """Configuration for PII redaction in traces"""
    
    # Global settings
    redaction_enabled: bool = True
    redaction_level: str = "strict"  # strict, standard, minimal
    preserve_analytics: bool = True
    
    # Industry-specific settings
    industry_code: str = "SaaS"
    compliance_frameworks: List[str] = field(default_factory=list)
    
    # Field-specific rules
    field_redaction_rules: Dict[str, str] = field(default_factory=dict)
    
    # Retention and audit
    retain_redaction_log: bool = True
    audit_redaction_decisions: bool = True
    
    # Performance settings
    lazy_redaction: bool = False  # Redact on access vs. on storage
    cache_redacted_data: bool = True
```

### **Automatic PII Detection**

```python
class PIIDetector:
    """Automatic PII detection in trace data"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        # ML-based detection for complex patterns
        self.ml_detector = load_pii_detection_model()
    
    def detect_pii_fields(self, data: Dict[str, Any]) -> List[str]:
        """Detect PII fields in trace data"""
        pii_fields = []
        
        for field_name, field_value in data.items():
            if isinstance(field_value, str):
                # Pattern-based detection
                for pii_type, pattern in self.patterns.items():
                    if re.search(pattern, field_value):
                        pii_fields.append(field_name)
                        break
                
                # ML-based detection
                if self.ml_detector.is_pii(field_value):
                    pii_fields.append(field_name)
        
        return pii_fields
```

### **Redaction Execution**

```python
class TraceRedactor:
    """Execute PII redaction on trace data"""
    
    def __init__(self, config: PIIRedactionConfig):
        self.config = config
        self.detector = PIIDetector()
        self.techniques = {
            'complete': self._complete_redaction,
            'partial': self._partial_redaction,
            'tokenize': self._tokenization,
            'hash': self._hashing,
            'generalize': self._generalization,
            'synthetic': self._synthetic_replacement
        }
    
    def redact_trace(self, trace: WorkflowTrace) -> WorkflowTrace:
        """Apply PII redaction to entire trace"""
        
        if not self.config.redaction_enabled:
            return trace
        
        # Detect PII fields
        pii_fields = self._detect_trace_pii(trace)
        
        # Apply redaction
        redacted_trace = self._apply_redaction(trace, pii_fields)
        
        # Update trace metadata
        redacted_trace.context.pii_redaction_enabled = True
        
        # Log redaction actions
        if self.config.retain_redaction_log:
            self._log_redaction_actions(trace.trace_id, pii_fields)
        
        return redacted_trace
    
    def _detect_trace_pii(self, trace: WorkflowTrace) -> Dict[str, List[str]]:
        """Detect PII across all trace components"""
        pii_map = {}
        
        # Check inputs/outputs
        if trace.inputs:
            pii_map['inputs'] = self.detector.detect_pii_fields(trace.inputs.input_data)
        
        if trace.outputs:
            pii_map['outputs'] = self.detector.detect_pii_fields(trace.outputs.output_data)
        
        # Check steps
        for i, step in enumerate(trace.steps):
            if step.inputs:
                pii_map[f'step_{i}_inputs'] = self.detector.detect_pii_fields(step.inputs.input_data)
            if step.outputs:
                pii_map[f'step_{i}_outputs'] = self.detector.detect_pii_fields(step.outputs.output_data)
        
        return pii_map
```

---

## Compliance Mapping

### **GDPR Compliance**
- **Article 25**: Privacy by Design - automatic PII detection
- **Article 32**: Security of Processing - encryption of PII
- **Article 17**: Right to Erasure - complete redaction capability
- **Article 20**: Data Portability - tokenization for data export

### **HIPAA Compliance**
- **§164.514**: De-identification standards
- **§164.312**: Technical safeguards for PHI
- **§164.308**: Administrative safeguards

### **DPDP Act 2023 (India)**
- **Section 8**: Data minimization principles
- **Section 12**: Rights of data principals
- **Section 16**: Data breach notification

### **RBI Guidelines**
- **Data Localization**: Redact before cross-border transfer
- **Customer Consent**: Log consent for data processing
- **Audit Requirements**: Maintain redaction audit trails

---

## Operational Procedures

### **Redaction Validation**

```python
def validate_redaction_completeness(trace: WorkflowTrace) -> Dict[str, Any]:
    """Validate that all PII has been properly redacted"""
    
    validation_result = {
        'compliant': True,
        'issues': [],
        'pii_detected': [],
        'redaction_coverage': 100.0
    }
    
    # Re-scan for PII in redacted trace
    detector = PIIDetector()
    remaining_pii = detector.detect_pii_fields(trace.to_dict())
    
    if remaining_pii:
        validation_result['compliant'] = False
        validation_result['pii_detected'] = remaining_pii
        validation_result['issues'].append("PII detected in redacted trace")
    
    return validation_result
```

### **Audit Trail Requirements**

```python
@dataclass
class RedactionAuditEntry:
    """Audit entry for PII redaction actions"""
    
    trace_id: str
    timestamp: str
    redaction_technique: str
    fields_redacted: List[str]
    compliance_framework: str
    user_id: str
    tenant_id: int
    
    # Reversibility information
    reversible: bool = False
    token_mapping_id: Optional[str] = None
    
    # Validation
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
```

### **Emergency Procedures**

1. **Data Breach Response**:
   - Immediate complete redaction of affected traces
   - Audit trail preservation
   - Regulator notification preparation

2. **Compliance Audit**:
   - Export redaction audit logs
   - Demonstrate redaction effectiveness
   - Provide compliance evidence

3. **Right to Erasure**:
   - Locate all traces containing individual's data
   - Apply complete redaction
   - Update audit trails

---

## Configuration Examples

### **Tenant-Specific Configuration**

```yaml
# Tenant 1000 - SaaS Company
tenant_1000_redaction:
  industry: "SaaS"
  compliance_frameworks: ["GDPR", "SOX"]
  redaction_level: "standard"
  field_rules:
    customer_email: "partial"
    customer_name: "tokenize"
    payment_info: "complete"
    usage_metrics: "preserve"

# Tenant 2000 - Bank
tenant_2000_redaction:
  industry: "BANK"
  compliance_frameworks: ["RBI", "BASEL_III"]
  redaction_level: "strict"
  field_rules:
    account_number: "complete"
    customer_name: "hash"
    transaction_amount: "generalize"
    risk_score: "preserve"
```

### **Role-Based Access**

```yaml
role_based_redaction:
  compliance_officer:
    can_view_redacted: true
    can_access_tokens: true
    audit_all_access: true
  
  developer:
    can_view_redacted: true
    can_access_tokens: false
    synthetic_data_only: true
  
  support_agent:
    can_view_redacted: false
    tokenized_view_only: true
    limited_fields: ["case_id", "status", "category"]
```

---

## Monitoring and Alerts

### **Key Metrics**
- **Redaction Coverage**: % of PII fields successfully redacted
- **Detection Accuracy**: % of PII correctly identified
- **Performance Impact**: Redaction processing time
- **Compliance Score**: Overall compliance with regulations

### **Alert Conditions**
- PII detected in "redacted" traces
- Redaction failure rate > 1%
- Unauthorized access to non-redacted data
- Compliance framework violations

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** compliance-team@company.com
