# Schema Architecture Decision Records (ADRs)

**Tasks 7.1-T34, 7.4-T35, 7.5-T57: Document schema ADRs**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Architecture Team

---

## Overview

This document contains Architecture Decision Records (ADRs) for all schema design decisions in the RevOps platform. These decisions provide context, rationale, and consequences for future reference and evolution.

---

## ADR-001: Canonical Trace Schema Design (7.1-T34)

**Status:** Accepted  
**Date:** 2024-10-08  
**Deciders:** Platform Architecture Team, Data Engineering Team  

### Context
We needed to design a canonical trace schema that captures all workflow execution details across different industries (SaaS, Banking, Insurance) while maintaining compliance with various regulatory frameworks.

### Decision
We will use a hierarchical JSON schema with the following key design principles:

1. **Immutable Core Fields**: `trace_id`, `workflow_id`, `execution_id`, `tenant_id` are immutable
2. **Industry Overlays**: Industry-specific fields in `context.regulatory_context`
3. **Governance-First**: Mandatory `governance_events` array for all executions
4. **Performance Tracking**: Built-in `performance_metrics` and `sla_metrics`
5. **Evidence Linkage**: Direct references to evidence packs and override ledger

### Rationale
- **Compliance**: Regulatory frameworks require different fields, overlays provide flexibility
- **Auditability**: Immutable fields ensure trace integrity for audit purposes
- **Performance**: Built-in metrics eliminate need for separate monitoring schemas
- **Scalability**: JSON structure allows for future field additions without breaking changes

### Consequences
**Positive:**
- Single schema supports all industries and compliance frameworks
- Built-in governance and performance tracking
- Future-proof with extensible JSON structure
- Strong audit trail capabilities

**Negative:**
- Larger storage footprint due to comprehensive field set
- Complex validation logic for industry-specific overlays
- Potential performance impact from large JSON documents

### Implementation Details
```json
{
  "trace_id": "UUID v7 for time-ordered sorting",
  "workflow_id": "Semantic versioning with industry prefix",
  "governance_events": "Array of policy applications and compliance checks",
  "industry_overlay": "Enum: SaaS, BANK, INSUR, ECOMM, FS, IT",
  "evidence_pack_id": "Direct linkage to evidence storage",
  "schema_version": "SemVer for schema evolution"
}
```

---

## ADR-002: Evidence Pack Schema Structure (7.4-T35)

**Status:** Accepted  
**Date:** 2024-10-08  
**Deciders:** Compliance Team, Security Team, Platform Architecture Team  

### Context
Evidence packs must provide immutable, tamper-proof records of workflow executions that satisfy regulatory requirements across multiple jurisdictions (US SOX, EU GDPR, India RBI, etc.).

### Decision
We will implement a cryptographically-secured evidence pack schema with:

1. **Immutable Core**: Evidence pack content cannot be modified after creation
2. **Digital Signatures**: All evidence packs signed with platform private key
3. **Hash Chains**: SHA-256 hashing with optional Merkle tree anchoring
4. **Regulatory Overlays**: Industry-specific evidence requirements
5. **PII Protection**: Built-in redaction and masking capabilities

### Rationale
- **Legal Admissibility**: Digital signatures and hash chains provide legal-grade evidence
- **Regulatory Compliance**: Overlays ensure jurisdiction-specific requirements are met
- **Privacy Protection**: Built-in PII handling prevents compliance violations
- **Tamper Evidence**: Cryptographic integrity prevents evidence manipulation

### Consequences
**Positive:**
- Legal-grade evidence suitable for regulatory audits
- Strong privacy protection with built-in PII handling
- Cross-jurisdictional compliance support
- Tamper-evident storage with cryptographic integrity

**Negative:**
- Increased storage costs due to cryptographic overhead
- Complex key management requirements
- Performance impact from signature generation/verification
- Immutability makes error correction difficult

### Implementation Details
```json
{
  "evidence_pack_id": "UUID v7 with timestamp ordering",
  "evidence_hash": "SHA-256 of complete evidence content",
  "digital_signature": "RSA-2048 signature with platform key",
  "regulatory_overlays": {
    "SOX": "Financial controls evidence",
    "GDPR": "Data processing lawful basis",
    "RBI": "Banking compliance evidence",
    "HIPAA": "PHI handling evidence"
  },
  "pii_redaction_map": "Field-level redaction metadata",
  "retention_policy": "Industry-specific retention rules"
}
```

---

## ADR-003: Override Ledger Append-Only Design (7.5-T57)

**Status:** Accepted  
**Date:** 2024-10-08  
**Deciders:** Governance Team, Security Team, Compliance Team  

### Context
The override ledger must provide an immutable, auditable record of all governance exceptions while supporting maker-checker workflows and segregation of duties requirements.

### Decision
We will implement an append-only ledger with:

1. **Append-Only Storage**: No updates or deletes, only new entries
2. **Event Sourcing**: Complete lifecycle captured as discrete events
3. **Maker-Checker**: Separate request, approval, and execution events
4. **Cryptographic Integrity**: Hash chains and digital signatures
5. **Time Synchronization**: NTP-synchronized timestamps for forensic accuracy

### Rationale
- **Immutability**: Append-only design prevents tampering or revision
- **Auditability**: Complete event history provides full audit trail
- **Segregation of Duties**: Maker-checker pattern enforces governance
- **Forensic Accuracy**: Synchronized timestamps enable precise reconstruction

### Consequences
**Positive:**
- Tamper-proof governance record suitable for regulatory audit
- Complete audit trail with segregation of duties enforcement
- Forensic-grade timestamp accuracy
- Event sourcing enables complete state reconstruction

**Negative:**
- Storage growth over time (no deletion capability)
- Complex query patterns for current state reconstruction
- Requires robust time synchronization infrastructure
- Cannot correct erroneous entries (only add corrections)

### Implementation Details
```json
{
  "event_id": "UUID v7 for chronological ordering",
  "event_type": "request|approval|execution|revocation",
  "event_hash": "SHA-256 of event content",
  "previous_hash": "Hash chain linkage to previous event",
  "timestamp": "NTP-synchronized UTC timestamp",
  "maker_checker": {
    "requester_id": "User who requested override",
    "approver_id": "User who approved override",
    "executor_id": "User/system who executed override"
  },
  "segregation_of_duties": "Validation that requester ≠ approver"
}
```

---

## ADR-004: Multi-Tenant Schema Isolation Strategy

**Status:** Accepted  
**Date:** 2024-10-08  
**Deciders:** Security Team, Platform Architecture Team  

### Context
Multi-tenant architecture requires complete data isolation while maintaining performance and operational simplicity across all schema types (traces, evidence, overrides, risk register).

### Decision
We will use Row Level Security (RLS) with shared schema approach:

1. **Shared Schema**: Single database schema with tenant_id column
2. **RLS Policies**: PostgreSQL Row Level Security for automatic filtering
3. **Application Context**: `current_setting('app.tenant_id')` for tenant context
4. **Fail-Closed**: Queries fail if tenant context not set
5. **Audit Logging**: All cross-tenant access attempts logged

### Rationale
- **Performance**: Shared schema avoids connection multiplexing overhead
- **Operational Simplicity**: Single schema to maintain and backup
- **Security**: RLS provides database-level isolation guarantee
- **Compliance**: Tenant isolation required for regulatory compliance

### Consequences
**Positive:**
- Strong security with database-level isolation
- Operational simplicity with single schema
- Good performance characteristics
- Automatic tenant filtering at database level

**Negative:**
- PostgreSQL dependency for RLS functionality
- Application must always set tenant context
- Complex backup/restore for single tenant
- Potential for RLS policy bugs affecting isolation

### Implementation Details
```sql
-- RLS policy example
CREATE POLICY tenant_isolation_traces ON traces
FOR ALL USING (tenant_id = current_setting('app.tenant_id')::integer);

-- Application context setting
SET app.tenant_id = 1000;

-- Fail-closed validation
CREATE OR REPLACE FUNCTION validate_tenant_context()
RETURNS TRIGGER AS $$
BEGIN
  IF current_setting('app.tenant_id', true) IS NULL THEN
    RAISE EXCEPTION 'Tenant context not set';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

## ADR-005: Schema Versioning and Evolution Strategy

**Status:** Accepted  
**Date:** 2024-10-08  
**Deciders:** Platform Engineering Team, API Team  

### Context
Schemas must evolve over time while maintaining backward compatibility and supporting gradual migration across tenants and environments.

### Decision
We will use semantic versioning with backward-compatible evolution:

1. **Semantic Versioning**: MAJOR.MINOR.PATCH versioning for all schemas
2. **Backward Compatibility**: MINOR and PATCH versions must be backward compatible
3. **Additive Changes**: New fields added as optional with safe defaults
4. **Deprecation Process**: 6-month deprecation period before removal
5. **Migration Framework**: Automated migration tools for schema updates

### Rationale
- **Stability**: Backward compatibility prevents breaking existing integrations
- **Flexibility**: Additive changes allow for feature evolution
- **Predictability**: Semantic versioning provides clear upgrade expectations
- **Safety**: Automated migrations reduce human error risk

### Consequences
**Positive:**
- Predictable schema evolution with clear compatibility guarantees
- Automated migration reduces operational overhead
- Gradual rollout capability for large changes
- Clear deprecation process for obsolete features

**Negative:**
- Schema bloat from maintaining deprecated fields
- Complex validation logic for multiple schema versions
- Migration complexity for major version changes
- Potential performance impact from backward compatibility layers

### Implementation Details
```json
{
  "schema_version": "1.2.3",
  "compatibility_matrix": {
    "1.0.x": "fully_compatible",
    "1.1.x": "fully_compatible", 
    "1.2.x": "current",
    "2.0.x": "breaking_changes"
  },
  "deprecated_fields": [
    {
      "field": "old_field_name",
      "deprecated_in": "1.1.0",
      "removal_planned": "2.0.0",
      "replacement": "new_field_name"
    }
  ]
}
```

---

## ADR-006: Industry Overlay Implementation Pattern

**Status:** Accepted  
**Date:** 2024-10-08  
**Deciders:** Business Team, Compliance Team, Platform Architecture Team  

### Context
Different industries (SaaS, Banking, Insurance) have unique compliance requirements and data needs that must be supported within the same platform architecture.

### Decision
We will implement industry overlays using composition pattern:

1. **Base Schema**: Core fields common to all industries
2. **Overlay Extensions**: Industry-specific field additions
3. **Validation Rules**: Industry-specific validation logic
4. **Compliance Mappings**: Regulatory framework mappings per industry
5. **Template Inheritance**: Industry templates inherit from base templates

### Rationale
- **Flexibility**: Each industry can have specific requirements
- **Maintainability**: Core schema remains stable across industries
- **Compliance**: Industry-specific regulatory requirements supported
- **Reusability**: Common patterns shared across industries

### Consequences
**Positive:**
- Industry-specific compliance requirements fully supported
- Core platform stability with industry flexibility
- Reusable patterns across similar industries
- Clear separation of concerns

**Negative:**
- Increased complexity in validation and processing logic
- Potential for industry-specific bugs
- Documentation overhead for multiple variants
- Testing complexity across industry combinations

### Implementation Details
```json
{
  "base_schema": {
    "trace_id": "required",
    "workflow_id": "required",
    "tenant_id": "required"
  },
  "industry_overlays": {
    "SaaS": {
      "additional_fields": ["subscription_id", "mrr_impact"],
      "compliance_frameworks": ["SOX", "GDPR"],
      "validation_rules": ["subscription_validation"]
    },
    "BANK": {
      "additional_fields": ["loan_id", "regulatory_capital_impact"],
      "compliance_frameworks": ["RBI", "BASEL_III", "SOX"],
      "validation_rules": ["banking_compliance_validation"]
    },
    "INSUR": {
      "additional_fields": ["policy_id", "solvency_impact"],
      "compliance_frameworks": ["IRDAI", "SOLVENCY_II", "SOX"],
      "validation_rules": ["insurance_solvency_validation"]
    }
  }
}
```

---

## ADR-007: Performance vs. Compliance Trade-offs

**Status:** Accepted  
**Date:** 2024-10-08  
**Deciders:** Performance Team, Compliance Team, Platform Architecture Team  

### Context
Comprehensive compliance tracking and audit capabilities can impact system performance, requiring careful balance between compliance requirements and performance goals.

### Decision
We will prioritize compliance with performance optimizations:

1. **Compliance First**: All compliance requirements are non-negotiable
2. **Async Processing**: Heavy compliance operations performed asynchronously
3. **Caching Strategy**: Aggressive caching of compliance metadata
4. **Batch Operations**: Bulk processing for evidence pack generation
5. **Performance Monitoring**: SLA tracking for compliance operations

### Rationale
- **Regulatory Requirements**: Compliance violations have severe consequences
- **Business Continuity**: Performance optimizations maintain user experience
- **Risk Management**: Async processing prevents compliance from blocking operations
- **Scalability**: Batch operations handle high-volume scenarios efficiently

### Consequences
**Positive:**
- Full compliance with regulatory requirements
- Maintained system performance through optimization
- Scalable architecture for high-volume operations
- Clear SLA tracking for compliance operations

**Negative:**
- Increased system complexity from async processing
- Potential for compliance lag in async operations
- Higher infrastructure costs for performance optimization
- Complex error handling for async compliance operations

### Implementation Details
```json
{
  "compliance_sla": {
    "evidence_pack_generation": "< 5 seconds",
    "override_ledger_write": "< 1 second",
    "trace_ingestion": "< 2 seconds",
    "compliance_validation": "< 3 seconds"
  },
  "async_operations": [
    "evidence_pack_signing",
    "regulatory_reporting",
    "audit_trail_generation",
    "compliance_analytics"
  ],
  "caching_strategy": {
    "policy_definitions": "1 hour TTL",
    "compliance_rules": "30 minutes TTL",
    "regulatory_mappings": "24 hours TTL"
  }
}
```

---

## Review and Maintenance

### Review Schedule
- **Quarterly Reviews**: Architecture team reviews all ADRs
- **Annual Updates**: Major review with business stakeholders
- **Ad-hoc Reviews**: Triggered by significant platform changes

### Change Process
1. **Proposal**: New ADR proposed with context and alternatives
2. **Review**: Architecture team and relevant stakeholders review
3. **Decision**: Formal decision with rationale documented
4. **Implementation**: Changes implemented with migration plan
5. **Monitoring**: Impact monitored and ADR updated if needed

### ADR Status Lifecycle
- **Proposed**: Under consideration
- **Accepted**: Approved and being implemented
- **Implemented**: Fully implemented in production
- **Deprecated**: Being phased out
- **Superseded**: Replaced by newer ADR

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** architecture-team@company.com
