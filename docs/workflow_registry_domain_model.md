# Workflow Registry Domain Model

**Task 7.3-T01: Define registry domain model (Workflow, Version, Artifact, Overlay, Signature)**  
**Version:** 1.0.0  
**Last Updated:** October 8, 2024  
**Owner:** Platform Architecture Team

---

## Overview

This document defines the comprehensive domain model for the Workflow Registry, which serves as the single source of truth for all RBA workflows and related artifacts. The registry enforces governance-by-design with strict immutability, provenance, and auditability across multi-tenant environments.

---

## Core Domain Entities

### **1. Registry Workflow**

```yaml
entity: registry_workflow
description: "Canonical workflow definition in the registry"
attributes:
  # Identity
  - workflow_id: "Unique workflow identifier (UUID)"
  - workflow_name: "Human-readable workflow name"
  - workflow_slug: "URL-safe identifier (kebab-case)"
  - namespace: "Organizational namespace (tenant/org scoped)"
  
  # Classification
  - workflow_type: "rba, rbia, aala, hybrid"
  - automation_type: "pipeline_hygiene, forecast_accuracy, lead_scoring, etc."
  - industry_overlay: "SaaS, BANK, INSUR, UNIVERSAL"
  - category: "data_sync, governance, notification, approval, custom"
  
  # Metadata
  - description: "Detailed workflow description"
  - keywords: "Array of searchable keywords"
  - tags: "Array of classification tags"
  - documentation_url: "Link to detailed documentation"
  
  # Ownership & Stewardship
  - owner_user_id: "Primary owner/maintainer"
  - steward_team: "Responsible team/department"
  - contact_email: "Support/maintenance contact"
  
  # Lifecycle
  - status: "draft, active, deprecated, retired, archived"
  - visibility: "public, private, organization, tenant"
  - created_at: "Initial creation timestamp"
  - updated_at: "Last modification timestamp"
  - deprecated_at: "Deprecation timestamp"
  - retired_at: "Retirement timestamp"
  
  # Business Context
  - business_value: "Description of business impact"
  - customer_impact: "Customer-facing impact description"
  - risk_level: "low, medium, high, critical"
  - compliance_requirements: "Array of compliance frameworks"

relationships:
  - has_many: workflow_versions, workflow_overlays, workflow_dependencies
  - belongs_to: tenant, owner (user)
  - has_many: usage_analytics, trust_scores, audit_logs
```

### **2. Workflow Version**

```yaml
entity: workflow_version
description: "Immutable version of a workflow with semantic versioning"
attributes:
  # Identity
  - version_id: "Unique version identifier (UUID)"
  - workflow_id: "Parent workflow reference"
  - version_number: "Semantic version (e.g., v1.2.3)"
  - version_alias: "Optional alias (latest, stable, lts)"
  
  # Version Metadata
  - major_version: "Major version number (breaking changes)"
  - minor_version: "Minor version number (new features)"
  - patch_version: "Patch version number (bug fixes)"
  - pre_release: "Pre-release identifier (alpha, beta, rc)"
  - build_metadata: "Build metadata (+build.123)"
  
  # Lifecycle Status
  - status: "draft, review, approved, published, deprecated, retired"
  - published_at: "Publication timestamp"
  - deprecated_at: "Deprecation timestamp"
  - retirement_date: "Planned retirement date"
  
  # Change Information
  - changelog: "Human-readable change description"
  - breaking_changes: "Array of breaking change descriptions"
  - migration_notes: "Migration guidance from previous versions"
  - compatibility_notes: "Compatibility information"
  
  # Approval & Governance
  - approval_status: "pending, approved, rejected"
  - approved_by: "Approver user ID"
  - approved_at: "Approval timestamp"
  - approval_notes: "Approval/rejection notes"
  - review_checklist: "JSONB checklist completion status"
  
  # Technical Metadata
  - dsl_schema_version: "DSL schema version used"
  - runtime_requirements: "Minimum runtime version requirements"
  - compiler_version: "Compiler version used"
  - feature_flags: "Required feature flags"
  
  # Immutability & Integrity
  - is_immutable: "Boolean immutability flag (true after publish)"
  - content_hash: "SHA-256 hash of all artifacts"
  - signature: "Cryptographic signature"
  - attestation: "SLSA attestation metadata"

relationships:
  - belongs_to: workflow, approver (user)
  - has_many: artifacts, policy_bindings, compatibility_entries
  - has_one: sbom, provenance_record
```

### **3. Workflow Artifact**

```yaml
entity: workflow_artifact
description: "Individual artifacts that comprise a workflow version"
attributes:
  # Identity
  - artifact_id: "Unique artifact identifier (UUID)"
  - version_id: "Parent version reference"
  - artifact_name: "Artifact filename/identifier"
  - artifact_type: "dsl, compiled_plan, documentation, sbom, signature"
  
  # Storage Information
  - storage_path: "Object storage path/key"
  - storage_bucket: "Storage bucket/container"
  - content_type: "MIME type"
  - file_size_bytes: "File size in bytes"
  - encoding: "File encoding (utf-8, base64, etc.)"
  
  # Integrity & Security
  - content_hash: "SHA-256 hash of content"
  - checksum_algorithm: "Hash algorithm used"
  - is_signed: "Boolean signature flag"
  - signature_path: "Path to detached signature"
  - encryption_status: "none, at_rest, in_transit, end_to_end"
  
  # Metadata
  - description: "Artifact description"
  - created_at: "Creation timestamp"
  - uploaded_at: "Upload completion timestamp"
  - last_accessed_at: "Last access timestamp"
  - access_count: "Number of times accessed"
  
  # Lifecycle
  - status: "uploading, available, archived, deleted"
  - retention_until: "Retention expiry date"
  - archived_at: "Archive timestamp"
  - archive_location: "Cold storage location"

relationships:
  - belongs_to: workflow_version
  - has_many: access_logs, integrity_checks
```

### **4. Workflow Overlay**

```yaml
entity: workflow_overlay
description: "Industry or tenant-specific customizations of base workflows"
attributes:
  # Identity
  - overlay_id: "Unique overlay identifier (UUID)"
  - base_workflow_id: "Base workflow reference"
  - overlay_name: "Overlay name"
  - overlay_type: "industry, tenant, organization, custom"
  
  # Scope & Targeting
  - target_industry: "Industry code (SaaS, BANK, INSUR)"
  - target_tenant_id: "Specific tenant (if tenant-scoped)"
  - target_organization: "Organization scope"
  - applicability_rules: "JSONB rules for when overlay applies"
  
  # Customization Content
  - field_overrides: "JSONB field value overrides"
  - policy_additions: "Additional policy requirements"
  - validation_rules: "Custom validation rules"
  - ui_customizations: "UI/UX customizations"
  - documentation_overrides: "Custom documentation"
  
  # Inheritance & Composition
  - parent_overlay_id: "Parent overlay (for inheritance)"
  - composition_order: "Order in overlay chain"
  - merge_strategy: "replace, merge, append"
  - conflict_resolution: "Strategy for resolving conflicts"
  
  # Lifecycle
  - status: "draft, active, deprecated, retired"
  - effective_from: "When overlay becomes effective"
  - effective_until: "When overlay expires"
  - created_by: "Creator user ID"
  - maintained_by: "Maintainer user ID"

relationships:
  - belongs_to: base_workflow, target_tenant, creator (user)
  - has_many: overlay_versions, usage_analytics
  - belongs_to: parent_overlay (self-referential)
```

### **5. Registry Signature**

```yaml
entity: registry_signature
description: "Cryptographic signatures for workflow artifacts"
attributes:
  # Identity
  - signature_id: "Unique signature identifier (UUID)"
  - artifact_id: "Signed artifact reference"
  - signature_type: "detached, embedded, cosign, gpg"
  
  # Signature Content
  - signature_data: "Base64-encoded signature"
  - signature_algorithm: "RSA-SHA256, ECDSA-SHA256, Ed25519"
  - hash_algorithm: "SHA-256, SHA-512"
  - signature_format: "PKCS#7, JWS, COSE"
  
  # Signing Identity
  - signer_identity: "Signer identifier"
  - signer_email: "Signer email address"
  - signing_key_id: "Key identifier"
  - certificate_chain: "X.509 certificate chain"
  - key_source: "kms, vault, local, hardware"
  
  # Timestamp & Validity
  - signed_at: "Signature creation timestamp"
  - valid_from: "Signature validity start"
  - valid_until: "Signature expiry"
  - timestamp_authority: "TSA URL (if timestamped)"
  - revocation_status: "valid, revoked, expired, unknown"
  
  # Verification
  - verification_status: "unverified, valid, invalid, expired"
  - last_verified_at: "Last verification timestamp"
  - verification_details: "JSONB verification results"
  - trust_chain_valid: "Boolean trust chain validity"

relationships:
  - belongs_to: workflow_artifact, signer (user)
  - has_many: verification_logs
```

### **6. Policy Binding**

```yaml
entity: policy_binding
description: "Links workflow versions to applicable policy packs"
attributes:
  # Identity
  - binding_id: "Unique binding identifier (UUID)"
  - version_id: "Workflow version reference"
  - policy_pack_id: "Policy pack reference"
  
  # Binding Configuration
  - binding_type: "required, recommended, optional"
  - enforcement_level: "strict, advisory, disabled"
  - binding_scope: "workflow, step, tenant, global"
  - precedence: "Binding precedence order"
  
  # Applicability
  - condition_rules: "JSONB rules for when binding applies"
  - environment_scope: "dev, staging, prod, all"
  - tenant_scope: "specific tenant or all"
  - effective_from: "When binding becomes effective"
  - effective_until: "When binding expires"
  
  # Metadata
  - binding_reason: "Why this policy is bound"
  - compliance_framework: "Associated compliance framework"
  - created_by: "Binding creator"
  - approved_by: "Binding approver"
  - created_at: "Binding creation timestamp"

relationships:
  - belongs_to: workflow_version, policy_pack, creator (user)
```

### **7. Compatibility Matrix**

```yaml
entity: compatibility_matrix
description: "Version compatibility information for safe upgrades"
attributes:
  # Identity
  - compatibility_id: "Unique compatibility record ID (UUID)"
  - version_id: "Workflow version reference"
  
  # Runtime Compatibility
  - min_runtime_version: "Minimum runtime version required"
  - max_runtime_version: "Maximum runtime version supported"
  - min_compiler_version: "Minimum compiler version required"
  - max_compiler_version: "Maximum compiler version supported"
  
  # Platform Compatibility
  - supported_platforms: "Array of supported platforms"
  - required_features: "Array of required platform features"
  - optional_features: "Array of optional features"
  - deprecated_features: "Array of deprecated features used"
  
  # Dependency Compatibility
  - min_dsl_version: "Minimum DSL schema version"
  - max_dsl_version: "Maximum DSL schema version"
  - required_connectors: "JSONB connector version requirements"
  - optional_connectors: "JSONB optional connector versions"
  
  # Backward Compatibility
  - backward_compatible_with: "Array of compatible previous versions"
  - breaking_changes_from: "Array of versions with breaking changes"
  - migration_required_from: "Array of versions requiring migration"
  - deprecation_warnings: "Array of deprecation warnings"
  
  # Testing & Validation
  - compatibility_tested: "Boolean testing completion flag"
  - test_results: "JSONB test results"
  - last_tested_at: "Last compatibility test timestamp"
  - test_coverage: "Percentage of compatibility scenarios tested"

relationships:
  - belongs_to: workflow_version
  - references: compatible_versions (self-referential)
```

### **8. SBOM (Software Bill of Materials)**

```yaml
entity: workflow_sbom
description: "Software Bill of Materials for workflow dependencies"
attributes:
  # Identity
  - sbom_id: "Unique SBOM identifier (UUID)"
  - version_id: "Workflow version reference"
  - sbom_format: "CycloneDX, SPDX, custom"
  - sbom_version: "SBOM format version"
  
  # SBOM Content
  - sbom_data: "JSONB SBOM content"
  - component_count: "Number of components"
  - dependency_count: "Number of dependencies"
  - license_count: "Number of unique licenses"
  
  # Generation Metadata
  - generated_at: "SBOM generation timestamp"
  - generated_by: "SBOM generator tool/version"
  - generation_method: "automatic, manual, hybrid"
  - source_analyzed: "Boolean source analysis completion"
  
  # Integrity
  - sbom_hash: "SHA-256 hash of SBOM content"
  - is_signed: "Boolean signature flag"
  - signature_reference: "Reference to SBOM signature"
  
  # Compliance
  - compliance_status: "compliant, non_compliant, unknown"
  - license_compliance: "JSONB license compliance details"
  - security_scan_results: "JSONB security scan results"
  - vulnerability_count: "Number of known vulnerabilities"

relationships:
  - belongs_to: workflow_version
  - has_many: sbom_components, vulnerability_reports
```

### **9. Provenance Record**

```yaml
entity: provenance_record
description: "SLSA-style provenance information for supply chain security"
attributes:
  # Identity
  - provenance_id: "Unique provenance identifier (UUID)"
  - version_id: "Workflow version reference"
  - provenance_format: "SLSA, in-toto, custom"
  - provenance_version: "Provenance format version"
  
  # Build Information
  - build_system: "Build system identifier"
  - build_id: "Unique build identifier"
  - build_trigger: "manual, ci, scheduled, api"
  - build_started_at: "Build start timestamp"
  - build_completed_at: "Build completion timestamp"
  
  # Source Information
  - source_repository: "Source code repository URL"
  - source_commit: "Git commit hash"
  - source_branch: "Git branch name"
  - source_tag: "Git tag (if applicable)"
  - source_integrity: "Source code integrity hash"
  
  # Builder Information
  - builder_id: "Builder system identifier"
  - builder_version: "Builder version"
  - build_environment: "JSONB build environment details"
  - build_parameters: "JSONB build parameters"
  - build_artifacts: "Array of build output artifacts"
  
  # Attestation
  - attestation_data: "JSONB attestation content"
  - attestation_signature: "Attestation signature"
  - attestor_identity: "Attestor identifier"
  - attestation_timestamp: "Attestation timestamp"
  
  # Verification
  - verification_status: "unverified, verified, failed"
  - verification_details: "JSONB verification results"
  - last_verified_at: "Last verification timestamp"

relationships:
  - belongs_to: workflow_version
  - has_many: attestation_signatures
```

### **10. Usage Analytics**

```yaml
entity: usage_analytics
description: "Workflow usage and performance analytics"
attributes:
  # Identity
  - analytics_id: "Unique analytics record ID (UUID)"
  - workflow_id: "Workflow reference"
  - version_id: "Version reference (optional)"
  - tenant_id: "Tenant reference"
  
  # Usage Metrics
  - execution_count: "Number of executions"
  - unique_users: "Number of unique users"
  - success_rate: "Success rate percentage"
  - average_execution_time: "Average execution time in ms"
  - error_rate: "Error rate percentage"
  
  # Adoption Metrics
  - first_used_at: "First usage timestamp"
  - last_used_at: "Last usage timestamp"
  - adoption_trend: "growing, stable, declining"
  - user_satisfaction: "User satisfaction score (0-10)"
  - feedback_count: "Number of feedback submissions"
  
  # Performance Metrics
  - p50_execution_time: "50th percentile execution time"
  - p95_execution_time: "95th percentile execution time"
  - p99_execution_time: "99th percentile execution time"
  - resource_utilization: "JSONB resource usage metrics"
  
  # Time Period
  - period_start: "Analytics period start"
  - period_end: "Analytics period end"
  - aggregation_level: "hourly, daily, weekly, monthly"
  
  # Quality Indicators
  - trust_score: "Calculated trust score"
  - reliability_score: "Reliability score"
  - security_incidents: "Number of security incidents"
  - compliance_violations: "Number of compliance violations"

relationships:
  - belongs_to: workflow, workflow_version, tenant
```

### **11. Audit Log**

```yaml
entity: registry_audit_log
description: "Comprehensive audit trail for all registry actions"
attributes:
  # Identity
  - audit_id: "Unique audit record ID (UUID)"
  - correlation_id: "Request correlation ID"
  - session_id: "User session ID"
  
  # Action Information
  - action_type: "create, read, update, delete, publish, deprecate, approve"
  - resource_type: "workflow, version, artifact, overlay, policy_binding"
  - resource_id: "Affected resource ID"
  - resource_name: "Human-readable resource name"
  
  # Actor Information
  - actor_type: "user, system, api_client, service"
  - actor_id: "Actor identifier"
  - actor_name: "Actor display name"
  - actor_ip: "Source IP address (hashed)"
  - user_agent: "User agent string"
  
  # Context
  - tenant_id: "Tenant context"
  - organization_id: "Organization context"
  - environment: "dev, staging, prod"
  - api_version: "API version used"
  - client_version: "Client version"
  
  # Change Details
  - before_state: "JSONB state before change"
  - after_state: "JSONB state after change"
  - change_summary: "Human-readable change summary"
  - change_reason: "Reason for change"
  
  # Outcome
  - status: "success, failure, partial"
  - error_code: "Error code (if failed)"
  - error_message: "Error message (if failed)"
  - warnings: "Array of warning messages"
  
  # Timing
  - started_at: "Action start timestamp"
  - completed_at: "Action completion timestamp"
  - duration_ms: "Action duration in milliseconds"
  
  # Compliance & Security
  - compliance_impact: "Boolean compliance impact flag"
  - security_impact: "Boolean security impact flag"
  - risk_level: "low, medium, high, critical"
  - approval_required: "Boolean approval requirement flag"
  - evidence_generated: "Boolean evidence generation flag"

relationships:
  - belongs_to: actor (user), tenant
  - references: affected_resource (polymorphic)
```

---

## Domain Relationships

### **Core Relationship Patterns**

#### **Workflow Hierarchy**
```
Registry Workflow (1) → (many) Workflow Versions
Workflow Version (1) → (many) Workflow Artifacts
Workflow Version (1) → (1) SBOM
Workflow Version (1) → (1) Provenance Record
```

#### **Governance & Compliance**
```
Workflow Version (many) → (many) Policy Bindings
Policy Binding (many) → (1) Policy Pack
Workflow Artifact (1) → (many) Registry Signatures
Workflow Version (1) → (1) Compatibility Matrix
```

#### **Customization & Overlays**
```
Registry Workflow (1) → (many) Workflow Overlays
Workflow Overlay (many) → (1) Base Workflow
Workflow Overlay (many) → (1) Parent Overlay (inheritance)
```

#### **Analytics & Monitoring**
```
Registry Workflow (1) → (many) Usage Analytics
Workflow Version (1) → (many) Usage Analytics
All Entities (1) → (many) Registry Audit Logs
```

---

## Entity Lifecycle States

### **Workflow Lifecycle**
```
draft → active → deprecated → retired → archived
```

### **Version Lifecycle**
```
draft → review → approved → published → deprecated → retired
```

### **Artifact Lifecycle**
```
uploading → available → archived → deleted
```

### **Overlay Lifecycle**
```
draft → active → deprecated → retired
```

---

## Data Integrity Rules

### **Immutability Constraints**
- Published workflow versions are immutable
- Artifacts cannot be modified after upload
- Audit logs are append-only
- Signatures cannot be altered

### **Referential Integrity**
- All versions must reference valid workflows
- All artifacts must reference valid versions
- All overlays must reference valid base workflows
- All policy bindings must reference valid policies

### **Business Rules**
- Version numbers must follow semantic versioning
- Only one version can have a specific alias per workflow
- Deprecated workflows cannot have new versions published
- Retired workflows cannot be executed

---

## Security & Compliance

### **Access Control**
- Tenant-based isolation for all entities
- Role-based access control for operations
- Audit trail for all access and modifications
- Signature verification for integrity

### **Data Protection**
- Encryption at rest for sensitive artifacts
- Secure key management for signatures
- PII redaction in audit logs
- Compliance framework tagging

### **Supply Chain Security**
- SBOM generation for all workflows
- Provenance tracking for build processes
- Vulnerability scanning for dependencies
- Attestation for critical operations

---

## Performance Considerations

### **Indexing Strategy**
- Tenant-scoped indexes for multi-tenancy
- Version-based indexes for temporal queries
- Full-text indexes for search functionality
- Composite indexes for complex queries

### **Caching Strategy**
- Immutable artifact caching
- Version metadata caching
- Search result caching
- CDN distribution for artifacts

### **Scalability Patterns**
- Horizontal partitioning by tenant
- Artifact storage in object stores
- Read replicas for query performance
- Event-driven architecture for updates

---

**Document Version:** 1.0.0  
**Next Review Date:** January 8, 2025  
**Contact:** platform-architecture@company.com
