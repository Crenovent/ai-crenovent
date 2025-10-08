# RevOps Schema Conformance Rules

**Task 7.2-T54: Publish conformance rules (what "canonical" means)**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Engineering Team

---

## Overview

This document defines the conformance rules that determine what constitutes "canonical" data within the RevOps platform. These rules ensure data consistency, quality, and governance across all tenants, industries, and integrations.

---

## Canonical Data Definition

**Canonical Data** is the authoritative, standardized representation of business entities that serves as the single source of truth across the RevOps platform. It must conform to all rules defined in this document.

### Core Principles

1. **Single Source of Truth:** Each business entity has exactly one canonical representation
2. **Standardized Format:** All canonical data follows consistent naming, typing, and structure conventions
3. **Quality Assured:** Canonical data must pass all defined quality checks
4. **Governance Compliant:** All canonical data includes required governance metadata
5. **Audit Ready:** Canonical data maintains complete lineage and change history

---

## Entity Conformance Rules

### 1. Tenant Metadata Conformance

**Table:** `tenant_metadata`

#### Required Fields
- `tenant_id` (INTEGER): Must be unique across platform
- `tenant_name` (VARCHAR): Must be non-empty, max 255 characters
- `industry_code` (ENUM): Must be one of: SaaS, BANK, INSUR, ECOMM, FS, IT
- `region_code` (ENUM): Must be one of: US, EU, IN, APAC
- `compliance_requirements` (JSONB): Must be valid JSON array
- `status` (ENUM): Must be one of: active, suspended, offboarding
- `created_at` (TIMESTAMPTZ): Must be in UTC, not null
- `updated_at` (TIMESTAMPTZ): Must be in UTC, not null

#### Business Rules
- Tenant name must be unique within region
- Industry code determines available workflow templates
- Compliance requirements must match industry standards
- Status transitions must follow defined lifecycle

#### Quality Checks
```sql
-- Tenant conformance validation
SELECT tenant_id, 
       CASE 
         WHEN tenant_name IS NULL OR LENGTH(tenant_name) = 0 THEN 'FAIL: Empty tenant name'
         WHEN industry_code NOT IN ('SaaS', 'BANK', 'INSUR', 'ECOMM', 'FS', 'IT') THEN 'FAIL: Invalid industry code'
         WHEN region_code NOT IN ('US', 'EU', 'IN', 'APAC') THEN 'FAIL: Invalid region code'
         WHEN status NOT IN ('active', 'suspended', 'offboarding') THEN 'FAIL: Invalid status'
         WHEN created_at IS NULL OR updated_at IS NULL THEN 'FAIL: Missing timestamps'
         ELSE 'PASS'
       END as conformance_status
FROM tenant_metadata;
```

### 2. Account Conformance

**Table:** `accounts`

#### Required Fields
- `account_id` (UUID): Must be valid UUID v4 or v7
- `tenant_id` (INTEGER): Must reference existing tenant
- `account_name` (VARCHAR): Must be non-empty, max 255 characters
- `account_type` (ENUM): Must be one of: prospect, customer, partner, competitor
- `created_at` (TIMESTAMPTZ): Must be in UTC, not null
- `updated_at` (TIMESTAMPTZ): Must be in UTC, not null

#### Optional Fields with Rules
- `annual_revenue` (DECIMAL): If provided, must be >= 0
- `employee_count` (INTEGER): If provided, must be >= 1
- `website` (VARCHAR): If provided, must be valid URL format
- `external_id` (VARCHAR): If provided, must be unique within tenant

#### Business Rules
- Account name must be unique within tenant
- External ID must be unique within tenant if provided
- Annual revenue and employee count must be consistent (large revenue = more employees)
- Website must match account name domain when possible

#### Quality Checks
```sql
-- Account conformance validation
SELECT account_id,
       CASE 
         WHEN account_name IS NULL OR LENGTH(account_name) = 0 THEN 'FAIL: Empty account name'
         WHEN account_type NOT IN ('prospect', 'customer', 'partner', 'competitor') THEN 'FAIL: Invalid account type'
         WHEN annual_revenue < 0 THEN 'FAIL: Negative revenue'
         WHEN employee_count < 1 THEN 'FAIL: Invalid employee count'
         WHEN website IS NOT NULL AND website NOT LIKE 'http%' THEN 'FAIL: Invalid website format'
         WHEN created_at > updated_at THEN 'FAIL: Invalid timestamp order'
         ELSE 'PASS'
       END as conformance_status
FROM accounts;
```

### 3. Opportunity Conformance

**Table:** `opportunities`

#### Required Fields
- `opportunity_id` (UUID): Must be valid UUID v4 or v7
- `tenant_id` (INTEGER): Must reference existing tenant
- `account_id` (UUID): Must reference existing account
- `opportunity_name` (VARCHAR): Must be non-empty, max 255 characters
- `stage` (VARCHAR): Must be valid stage for tenant's industry
- `amount` (DECIMAL): Must be >= 0
- `probability` (DECIMAL): Must be between 0 and 100
- `close_date` (DATE): Must be future date for open opportunities

#### Business Rules
- Opportunity name must be unique within account
- Stage must follow defined progression for industry
- Probability must align with stage (later stages = higher probability)
- Close date cannot be in past for open opportunities
- Amount must be reasonable for account size

#### Quality Checks
```sql
-- Opportunity conformance validation
SELECT opportunity_id,
       CASE 
         WHEN opportunity_name IS NULL OR LENGTH(opportunity_name) = 0 THEN 'FAIL: Empty opportunity name'
         WHEN amount < 0 THEN 'FAIL: Negative amount'
         WHEN probability < 0 OR probability > 100 THEN 'FAIL: Invalid probability range'
         WHEN close_date < CURRENT_DATE AND stage NOT IN ('Closed Won', 'Closed Lost') THEN 'FAIL: Past close date for open opp'
         WHEN stage = 'Closed Won' AND probability != 100 THEN 'FAIL: Won opportunity must have 100% probability'
         WHEN stage = 'Closed Lost' AND probability != 0 THEN 'FAIL: Lost opportunity must have 0% probability'
         ELSE 'PASS'
       END as conformance_status
FROM opportunities;
```

---

## Data Type Conformance

### 1. UUID Standards
- **Format:** UUID v4 (random) or UUID v7 (time-ordered) preferred
- **Storage:** Native UUID type in PostgreSQL
- **Generation:** Use `gen_random_uuid()` for v4, application-side for v7
- **Validation:** Must pass UUID format validation

### 2. Timestamp Standards
- **Type:** TIMESTAMPTZ (timezone-aware)
- **Timezone:** All timestamps stored in UTC
- **Format:** ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ)
- **Precision:** Microsecond precision supported
- **Validation:** Must be valid timestamp, not null for audit fields

### 3. Monetary Values
- **Type:** DECIMAL(15,2) for currency amounts
- **Precision:** 2 decimal places for currency
- **Currency:** All amounts in USD unless specified
- **Validation:** Must be >= 0 for revenue/amounts
- **Rounding:** Banker's rounding (round half to even)

### 4. JSON/JSONB Fields
- **Type:** JSONB preferred for performance
- **Validation:** Must be valid JSON
- **Schema:** Follow defined JSON schema where applicable
- **Indexing:** Use GIN indexes for query performance
- **Null Handling:** Use `{}` or `[]` instead of NULL

---

## Naming Conformance

### 1. Table Names
- **Format:** snake_case, lowercase
- **Pattern:** `{domain}_{entity}` (e.g., `tenant_metadata`, `opportunity_stages`)
- **Singular:** Use singular nouns (e.g., `account` not `accounts`)
- **Prefixes:** Use consistent prefixes for related tables

### 2. Column Names
- **Format:** snake_case, lowercase
- **Suffixes:** 
  - `_id` for foreign keys (e.g., `tenant_id`, `account_id`)
  - `_at` for timestamps (e.g., `created_at`, `updated_at`)
  - `_by` for user references (e.g., `created_by_user_id`)
  - `is_` for booleans (e.g., `is_active`, `is_deleted`)
- **Consistency:** Use same column name across tables for same concept

### 3. Index Names
- **Format:** `idx_{table}_{columns}` for regular indexes
- **Unique:** `uk_{table}_{columns}` for unique constraints
- **Foreign Key:** `fk_{table}_{column}` for foreign key constraints
- **Partial:** Include condition in name for partial indexes

---

## Multi-Tenant Conformance

### 1. Tenant Isolation
- **RLS Policies:** All tables must have Row Level Security enabled
- **Tenant ID:** All tenant-scoped tables must include `tenant_id` column
- **Policy Expression:** Use `tenant_id = current_setting('app.tenant_id')::integer`
- **Validation:** Cross-tenant data access must be prevented

### 2. Data Residency
- **Regional Storage:** Data must be stored in tenant's specified region
- **Compliance:** Must follow regional data protection laws
- **Validation:** Automated checks for data residency compliance

---

## Quality Conformance Rules

### 1. Completeness Rules
- **Required Fields:** All NOT NULL fields must have values
- **Business Required:** Fields marked as business-required must be populated
- **Referential Integrity:** All foreign keys must reference existing records

### 2. Accuracy Rules
- **Data Types:** Values must match column data types
- **Constraints:** All check constraints must be satisfied
- **Business Logic:** Values must make business sense (e.g., positive revenue)

### 3. Consistency Rules
- **Cross-Field:** Related fields must be consistent (e.g., stage and probability)
- **Temporal:** Timestamps must be in logical order
- **Hierarchical:** Parent-child relationships must be valid

### 4. Uniqueness Rules
- **Primary Keys:** Must be unique across all records
- **Business Keys:** Natural keys must be unique within scope
- **External IDs:** Must be unique within tenant

---

## Governance Conformance

### 1. Audit Trail Requirements
- **Created Fields:** `created_at` and `created_by_user_id` required
- **Updated Fields:** `updated_at` required, auto-updated on changes
- **Soft Delete:** Use `is_deleted` flag instead of hard deletes
- **Change History:** Maintain change log for sensitive data

### 2. Evidence Pack Integration
- **Linkage:** All workflow-generated data must link to evidence packs
- **Traceability:** Source system and transformation must be recorded
- **Validation:** Evidence pack references must be valid

### 3. Policy Compliance
- **Industry Rules:** Data must comply with industry-specific regulations
- **Regional Laws:** Must follow regional data protection requirements
- **Retention Policies:** Data must follow defined retention schedules

---

## Validation Framework

### 1. Automated Validation
```sql
-- Create conformance validation function
CREATE OR REPLACE FUNCTION validate_canonical_conformance(table_name TEXT)
RETURNS TABLE(record_id TEXT, conformance_status TEXT, issues JSONB) AS $$
BEGIN
    -- Implementation varies by table
    -- Returns conformance status for each record
END;
$$ LANGUAGE plpgsql;
```

### 2. Quality Metrics
- **Conformance Score:** Percentage of records passing all rules
- **Issue Categories:** Breakdown of conformance issues by type
- **Trend Analysis:** Conformance score trends over time
- **Tenant Comparison:** Conformance scores by tenant

### 3. Monitoring and Alerting
- **Real-time Checks:** Validate data on insert/update
- **Batch Validation:** Daily conformance reports
- **Threshold Alerts:** Alert when conformance drops below threshold
- **Dashboard Integration:** Conformance metrics in governance dashboards

---

## Non-Conformant Data Handling

### 1. Quarantine Process
- **Detection:** Automated identification of non-conformant data
- **Isolation:** Move non-conformant data to quarantine tables
- **Notification:** Alert data stewards of conformance issues
- **Resolution:** Provide tools for data correction

### 2. Exception Handling
- **Temporary Exceptions:** Allow temporary non-conformance with approval
- **Documentation:** Record reason and timeline for exceptions
- **Monitoring:** Track exception resolution progress
- **Escalation:** Escalate unresolved exceptions

### 3. Data Correction
- **Automated Fixes:** Apply automatic corrections where safe
- **Manual Review:** Require human review for complex issues
- **Validation:** Re-validate after corrections
- **Audit Trail:** Log all correction activities

---

## Compliance and Certification

### 1. Industry Standards
- **SaaS:** SOX compliance for financial data
- **Banking:** RBI guidelines for financial institutions
- **Insurance:** IRDAI regulations for insurance data
- **Healthcare:** HIPAA compliance for PHI data

### 2. Regional Requirements
- **GDPR:** EU data protection requirements
- **DPDP:** India data protection requirements
- **CCPA:** California privacy requirements
- **SOX:** US financial reporting requirements

### 3. Certification Process
- **Self-Assessment:** Regular conformance self-checks
- **External Audit:** Annual third-party conformance audits
- **Certification:** Formal conformance certification
- **Continuous Monitoring:** Ongoing conformance validation

---

## Implementation Guidelines

### 1. Development Process
- **Schema Design:** Follow conformance rules from design phase
- **Code Review:** Include conformance checks in code reviews
- **Testing:** Test conformance validation in CI/CD pipeline
- **Documentation:** Document conformance decisions

### 2. Data Migration
- **Assessment:** Assess existing data conformance before migration
- **Transformation:** Transform non-conformant data during migration
- **Validation:** Validate conformance after migration
- **Rollback:** Plan for rollback if conformance fails

### 3. Ongoing Maintenance
- **Regular Reviews:** Quarterly conformance rule reviews
- **Updates:** Update rules based on business changes
- **Training:** Train teams on conformance requirements
- **Improvement:** Continuously improve conformance processes

---

## Approval and Maintenance

**Approved By:** Platform Engineering Team, Data Governance Committee  
**Review Cycle:** Quarterly  
**Change Process:** RFC process with stakeholder review  
**Distribution:** All development teams, data stewards  
**Questions:** Contact platform-engineering@company.com

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025
