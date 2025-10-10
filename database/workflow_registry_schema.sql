-- Task 7.3-T04: Build registry database schema
-- Comprehensive database schema for Workflow Registry with RLS multi-tenancy
-- Version: 1.0.0
-- Date: 2024-10-08

-- ============================================================================
-- WORKFLOW REGISTRY DATABASE SCHEMA
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ============================================================================
-- ENUMS AND TYPES
-- ============================================================================

-- Workflow and version status enums
CREATE TYPE workflow_status AS ENUM (
    'draft', 'active', 'deprecated', 'retired', 'archived'
);

CREATE TYPE version_status AS ENUM (
    'draft', 'review', 'approved', 'published', 'deprecated', 'retired'
);

CREATE TYPE artifact_type AS ENUM (
    'dsl', 'compiled_plan', 'documentation', 'sbom', 'signature', 'attestation', 'metadata'
);

CREATE TYPE overlay_type AS ENUM (
    'industry', 'tenant', 'organization', 'custom'
);

CREATE TYPE signature_type AS ENUM (
    'detached', 'embedded', 'cosign', 'gpg'
);

CREATE TYPE binding_type AS ENUM (
    'required', 'recommended', 'optional'
);

CREATE TYPE approval_status AS ENUM (
    'pending', 'approved', 'rejected'
);

-- ============================================================================
-- CORE REGISTRY TABLES
-- ============================================================================

-- Registry Workflows - Core workflow definitions
CREATE TABLE registry_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    
    -- Identity and naming
    workflow_name VARCHAR(200) NOT NULL,
    workflow_slug VARCHAR(200) NOT NULL,
    namespace VARCHAR(100) NOT NULL,
    
    -- Classification
    workflow_type VARCHAR(50) NOT NULL DEFAULT 'rba',
    automation_type VARCHAR(100) NOT NULL,
    industry_overlay VARCHAR(50) NOT NULL DEFAULT 'UNIVERSAL',
    category VARCHAR(50) NOT NULL DEFAULT 'custom',
    
    -- Metadata
    description TEXT,
    keywords TEXT[],
    tags TEXT[],
    documentation_url TEXT,
    
    -- Ownership and stewardship
    owner_user_id INTEGER,
    steward_team VARCHAR(100),
    contact_email VARCHAR(255),
    
    -- Lifecycle
    status workflow_status NOT NULL DEFAULT 'draft',
    visibility VARCHAR(50) NOT NULL DEFAULT 'private',
    deprecated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    
    -- Business context
    business_value TEXT,
    customer_impact TEXT,
    risk_level VARCHAR(20) NOT NULL DEFAULT 'medium',
    compliance_requirements TEXT[],
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by INTEGER,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by INTEGER,
    deleted_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT chk_workflow_name_length CHECK (LENGTH(workflow_name) >= 3),
    CONSTRAINT chk_workflow_slug_format CHECK (workflow_slug ~ '^[a-z0-9-]+$'),
    CONSTRAINT chk_risk_level CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_visibility CHECK (visibility IN ('public', 'private', 'organization', 'tenant'))
);

-- Workflow Versions - Immutable versions with semantic versioning
CREATE TABLE workflow_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES registry_workflows(workflow_id) ON DELETE CASCADE,
    
    -- Version information
    version_number VARCHAR(50) NOT NULL,
    major_version INTEGER NOT NULL,
    minor_version INTEGER NOT NULL,
    patch_version INTEGER NOT NULL,
    pre_release VARCHAR(50),
    build_metadata VARCHAR(100),
    version_alias VARCHAR(50),
    
    -- Lifecycle status
    status version_status NOT NULL DEFAULT 'draft',
    published_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    retirement_date TIMESTAMPTZ,
    
    -- Change information
    changelog TEXT,
    breaking_changes TEXT[],
    migration_notes TEXT,
    compatibility_notes TEXT,
    
    -- Approval and governance
    approval_status approval_status NOT NULL DEFAULT 'pending',
    approved_by INTEGER,
    approved_at TIMESTAMPTZ,
    approval_notes TEXT,
    review_checklist JSONB,
    
    -- Technical metadata
    dsl_schema_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    runtime_requirements JSONB,
    compiler_version VARCHAR(50),
    feature_flags TEXT[],
    
    -- Immutability and integrity
    is_immutable BOOLEAN NOT NULL DEFAULT FALSE,
    immutable_since TIMESTAMPTZ,
    content_hash VARCHAR(64),
    signature TEXT,
    attestation JSONB,
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by INTEGER,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by INTEGER,
    deleted_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT uq_workflow_version UNIQUE (workflow_id, version_number),
    CONSTRAINT uq_workflow_version_alias UNIQUE (workflow_id, version_alias) 
        WHERE version_alias IS NOT NULL AND deleted_at IS NULL,
    CONSTRAINT chk_version_format CHECK (
        version_number ~ '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)?(?:\+[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)?$'
    ),
    CONSTRAINT chk_immutable_logic CHECK (
        (is_immutable = FALSE) OR 
        (is_immutable = TRUE AND immutable_since IS NOT NULL AND content_hash IS NOT NULL)
    ),
    CONSTRAINT chk_approval_logic CHECK (
        (approval_status != 'approved') OR 
        (approval_status = 'approved' AND approved_by IS NOT NULL AND approved_at IS NOT NULL)
    )
);

-- Workflow Artifacts - Individual artifacts that comprise a version
CREATE TABLE workflow_artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES workflow_versions(version_id) ON DELETE CASCADE,
    
    -- Identity
    artifact_name VARCHAR(255) NOT NULL,
    artifact_type artifact_type NOT NULL,
    
    -- Storage information
    storage_path TEXT NOT NULL,
    storage_bucket VARCHAR(100) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    encoding VARCHAR(20) NOT NULL DEFAULT 'utf-8',
    
    -- Integrity and security
    content_hash VARCHAR(64) NOT NULL,
    checksum_algorithm VARCHAR(20) NOT NULL DEFAULT 'sha256',
    is_signed BOOLEAN NOT NULL DEFAULT FALSE,
    signature_path TEXT,
    encryption_status VARCHAR(20) NOT NULL DEFAULT 'none',
    
    -- Metadata
    description TEXT,
    last_accessed_at TIMESTAMPTZ,
    access_count INTEGER NOT NULL DEFAULT 0,
    
    -- Lifecycle
    status VARCHAR(20) NOT NULL DEFAULT 'available',
    retention_until TIMESTAMPTZ,
    archived_at TIMESTAMPTZ,
    archive_location TEXT,
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    uploaded_at TIMESTAMPTZ,
    created_by INTEGER,
    
    -- Constraints
    CONSTRAINT uq_version_artifact_name UNIQUE (version_id, artifact_name),
    CONSTRAINT chk_file_size_positive CHECK (file_size_bytes > 0),
    CONSTRAINT chk_artifact_status CHECK (status IN ('uploading', 'available', 'archived', 'deleted')),
    CONSTRAINT chk_encryption_status CHECK (encryption_status IN ('none', 'at_rest', 'in_transit', 'end_to_end'))
);

-- Workflow Overlays - Industry/tenant-specific customizations
CREATE TABLE workflow_overlays (
    overlay_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    base_workflow_id UUID NOT NULL REFERENCES registry_workflows(workflow_id) ON DELETE CASCADE,
    parent_overlay_id UUID REFERENCES workflow_overlays(overlay_id),
    
    -- Identity
    overlay_name VARCHAR(200) NOT NULL,
    overlay_type overlay_type NOT NULL,
    
    -- Scope and targeting
    target_industry VARCHAR(50),
    target_tenant_id INTEGER,
    target_organization VARCHAR(100),
    applicability_rules JSONB,
    
    -- Customization content
    field_overrides JSONB,
    policy_additions JSONB,
    validation_rules JSONB,
    ui_customizations JSONB,
    documentation_overrides JSONB,
    
    -- Inheritance and composition
    composition_order INTEGER NOT NULL DEFAULT 0,
    merge_strategy VARCHAR(20) NOT NULL DEFAULT 'merge',
    conflict_resolution VARCHAR(20) NOT NULL DEFAULT 'replace',
    
    -- Lifecycle
    status workflow_status NOT NULL DEFAULT 'draft',
    effective_from TIMESTAMPTZ,
    effective_until TIMESTAMPTZ,
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by INTEGER,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by INTEGER,
    deleted_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT chk_merge_strategy CHECK (merge_strategy IN ('replace', 'merge', 'append')),
    CONSTRAINT chk_conflict_resolution CHECK (conflict_resolution IN ('replace', 'merge', 'error')),
    CONSTRAINT chk_effective_dates CHECK (
        (effective_from IS NULL) OR (effective_until IS NULL) OR (effective_until > effective_from)
    )
);

-- Registry Signatures - Cryptographic signatures for artifacts
CREATE TABLE registry_signatures (
    signature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES workflow_artifacts(artifact_id) ON DELETE CASCADE,
    
    -- Signature type and content
    signature_type signature_type NOT NULL,
    signature_data TEXT NOT NULL,
    signature_algorithm VARCHAR(50) NOT NULL,
    hash_algorithm VARCHAR(20) NOT NULL DEFAULT 'sha256',
    signature_format VARCHAR(20) NOT NULL DEFAULT 'PKCS#7',
    
    -- Signing identity
    signer_identity VARCHAR(255) NOT NULL,
    signer_email VARCHAR(255),
    signing_key_id VARCHAR(100),
    certificate_chain TEXT,
    key_source VARCHAR(20) NOT NULL DEFAULT 'kms',
    
    -- Timestamp and validity
    signed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    timestamp_authority TEXT,
    revocation_status VARCHAR(20) NOT NULL DEFAULT 'valid',
    
    -- Verification
    verification_status VARCHAR(20) NOT NULL DEFAULT 'unverified',
    last_verified_at TIMESTAMPTZ,
    verification_details JSONB,
    trust_chain_valid BOOLEAN,
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by INTEGER,
    
    -- Constraints
    CONSTRAINT chk_key_source CHECK (key_source IN ('kms', 'vault', 'local', 'hardware')),
    CONSTRAINT chk_revocation_status CHECK (revocation_status IN ('valid', 'revoked', 'expired', 'unknown')),
    CONSTRAINT chk_verification_status CHECK (verification_status IN ('unverified', 'valid', 'invalid', 'expired'))
);

-- Policy Bindings - Links versions to applicable policy packs
CREATE TABLE policy_bindings (
    binding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES workflow_versions(version_id) ON DELETE CASCADE,
    policy_pack_id UUID NOT NULL, -- References dsl_policy_packs
    
    -- Binding configuration
    binding_type binding_type NOT NULL DEFAULT 'required',
    enforcement_level VARCHAR(20) NOT NULL DEFAULT 'strict',
    binding_scope VARCHAR(20) NOT NULL DEFAULT 'workflow',
    precedence INTEGER NOT NULL DEFAULT 0,
    
    -- Applicability
    condition_rules JSONB,
    environment_scope VARCHAR(20) NOT NULL DEFAULT 'all',
    tenant_scope INTEGER, -- NULL means all tenants
    effective_from TIMESTAMPTZ,
    effective_until TIMESTAMPTZ,
    
    -- Metadata
    binding_reason TEXT,
    compliance_framework VARCHAR(50),
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by INTEGER,
    approved_by INTEGER,
    approved_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT uq_version_policy_binding UNIQUE (version_id, policy_pack_id, binding_scope),
    CONSTRAINT chk_enforcement_level CHECK (enforcement_level IN ('strict', 'advisory', 'disabled')),
    CONSTRAINT chk_binding_scope CHECK (binding_scope IN ('workflow', 'step', 'tenant', 'global')),
    CONSTRAINT chk_environment_scope CHECK (environment_scope IN ('dev', 'staging', 'prod', 'all'))
);

-- Compatibility Matrix - Version compatibility information
CREATE TABLE compatibility_matrix (
    compatibility_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES workflow_versions(version_id) ON DELETE CASCADE,
    
    -- Runtime compatibility
    min_runtime_version VARCHAR(20),
    max_runtime_version VARCHAR(20),
    min_compiler_version VARCHAR(20),
    max_compiler_version VARCHAR(20),
    
    -- Platform compatibility
    supported_platforms TEXT[],
    required_features TEXT[],
    optional_features TEXT[],
    deprecated_features TEXT[],
    
    -- Dependency compatibility
    min_dsl_version VARCHAR(20),
    max_dsl_version VARCHAR(20),
    required_connectors JSONB,
    optional_connectors JSONB,
    
    -- Backward compatibility
    backward_compatible_with TEXT[],
    breaking_changes_from TEXT[],
    migration_required_from TEXT[],
    deprecation_warnings TEXT[],
    
    -- Testing and validation
    compatibility_tested BOOLEAN NOT NULL DEFAULT FALSE,
    test_results JSONB,
    last_tested_at TIMESTAMPTZ,
    test_coverage DECIMAL(5,2),
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uq_version_compatibility UNIQUE (version_id),
    CONSTRAINT chk_test_coverage CHECK (test_coverage >= 0 AND test_coverage <= 100)
);

-- Workflow SBOM - Software Bill of Materials
CREATE TABLE workflow_sbom (
    sbom_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES workflow_versions(version_id) ON DELETE CASCADE,
    
    -- SBOM metadata
    sbom_format VARCHAR(20) NOT NULL DEFAULT 'CycloneDX',
    sbom_version VARCHAR(10) NOT NULL DEFAULT '1.4',
    sbom_data JSONB NOT NULL,
    
    -- Component counts
    component_count INTEGER NOT NULL DEFAULT 0,
    dependency_count INTEGER NOT NULL DEFAULT 0,
    license_count INTEGER NOT NULL DEFAULT 0,
    
    -- Generation metadata
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    generated_by VARCHAR(100) NOT NULL,
    generation_method VARCHAR(20) NOT NULL DEFAULT 'automatic',
    source_analyzed BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Integrity
    sbom_hash VARCHAR(64) NOT NULL,
    is_signed BOOLEAN NOT NULL DEFAULT FALSE,
    signature_reference UUID REFERENCES registry_signatures(signature_id),
    
    -- Compliance
    compliance_status VARCHAR(20) NOT NULL DEFAULT 'unknown',
    license_compliance JSONB,
    security_scan_results JSONB,
    vulnerability_count INTEGER NOT NULL DEFAULT 0,
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uq_version_sbom UNIQUE (version_id),
    CONSTRAINT chk_sbom_format CHECK (sbom_format IN ('CycloneDX', 'SPDX', 'custom')),
    CONSTRAINT chk_generation_method CHECK (generation_method IN ('automatic', 'manual', 'hybrid')),
    CONSTRAINT chk_compliance_status CHECK (compliance_status IN ('compliant', 'non_compliant', 'unknown'))
);

-- Provenance Records - SLSA-style provenance information
CREATE TABLE provenance_records (
    provenance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES workflow_versions(version_id) ON DELETE CASCADE,
    
    -- Provenance metadata
    provenance_format VARCHAR(20) NOT NULL DEFAULT 'SLSA',
    provenance_version VARCHAR(10) NOT NULL DEFAULT '1.0',
    
    -- Build information
    build_system VARCHAR(100) NOT NULL,
    build_id VARCHAR(100) NOT NULL,
    build_trigger VARCHAR(20) NOT NULL DEFAULT 'manual',
    build_started_at TIMESTAMPTZ NOT NULL,
    build_completed_at TIMESTAMPTZ NOT NULL,
    
    -- Source information
    source_repository TEXT NOT NULL,
    source_commit VARCHAR(40) NOT NULL,
    source_branch VARCHAR(100),
    source_tag VARCHAR(100),
    source_integrity VARCHAR(64) NOT NULL,
    
    -- Builder information
    builder_id VARCHAR(100) NOT NULL,
    builder_version VARCHAR(50) NOT NULL,
    build_environment JSONB,
    build_parameters JSONB,
    build_artifacts TEXT[],
    
    -- Attestation
    attestation_data JSONB,
    attestation_signature TEXT,
    attestor_identity VARCHAR(255),
    attestation_timestamp TIMESTAMPTZ,
    
    -- Verification
    verification_status VARCHAR(20) NOT NULL DEFAULT 'unverified',
    verification_details JSONB,
    last_verified_at TIMESTAMPTZ,
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uq_version_provenance UNIQUE (version_id),
    CONSTRAINT chk_build_trigger CHECK (build_trigger IN ('manual', 'ci', 'scheduled', 'api')),
    CONSTRAINT chk_build_duration CHECK (build_completed_at >= build_started_at),
    CONSTRAINT chk_verification_status CHECK (verification_status IN ('unverified', 'verified', 'failed'))
);

-- Usage Analytics - Workflow usage and performance analytics
CREATE TABLE usage_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES registry_workflows(workflow_id) ON DELETE CASCADE,
    version_id UUID REFERENCES workflow_versions(version_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Usage metrics
    execution_count BIGINT NOT NULL DEFAULT 0,
    unique_users INTEGER NOT NULL DEFAULT 0,
    success_rate DECIMAL(5,2) NOT NULL DEFAULT 0,
    average_execution_time INTEGER NOT NULL DEFAULT 0, -- milliseconds
    error_rate DECIMAL(5,2) NOT NULL DEFAULT 0,
    
    -- Adoption metrics
    first_used_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    adoption_trend VARCHAR(20) NOT NULL DEFAULT 'stable',
    user_satisfaction DECIMAL(3,1), -- 0-10 scale
    feedback_count INTEGER NOT NULL DEFAULT 0,
    
    -- Performance metrics
    p50_execution_time INTEGER, -- milliseconds
    p95_execution_time INTEGER, -- milliseconds
    p99_execution_time INTEGER, -- milliseconds
    resource_utilization JSONB,
    
    -- Time period
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    aggregation_level VARCHAR(20) NOT NULL DEFAULT 'daily',
    
    -- Quality indicators
    trust_score DECIMAL(3,2), -- 0-1 scale
    reliability_score DECIMAL(3,2), -- 0-1 scale
    security_incidents INTEGER NOT NULL DEFAULT 0,
    compliance_violations INTEGER NOT NULL DEFAULT 0,
    
    -- Audit columns
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_success_rate CHECK (success_rate >= 0 AND success_rate <= 100),
    CONSTRAINT chk_error_rate CHECK (error_rate >= 0 AND error_rate <= 100),
    CONSTRAINT chk_adoption_trend CHECK (adoption_trend IN ('growing', 'stable', 'declining')),
    CONSTRAINT chk_user_satisfaction CHECK (user_satisfaction IS NULL OR (user_satisfaction >= 0 AND user_satisfaction <= 10)),
    CONSTRAINT chk_aggregation_level CHECK (aggregation_level IN ('hourly', 'daily', 'weekly', 'monthly')),
    CONSTRAINT chk_period_dates CHECK (period_end > period_start),
    CONSTRAINT chk_trust_score CHECK (trust_score IS NULL OR (trust_score >= 0 AND trust_score <= 1)),
    CONSTRAINT chk_reliability_score CHECK (reliability_score IS NULL OR (reliability_score >= 0 AND reliability_score <= 1))
);

-- Registry Audit Log - Comprehensive audit trail
CREATE TABLE registry_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id UUID,
    session_id VARCHAR(100),
    
    -- Action information
    action_type VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID NOT NULL,
    resource_name VARCHAR(255),
    
    -- Actor information
    actor_type VARCHAR(20) NOT NULL DEFAULT 'user',
    actor_id UUID NOT NULL,
    actor_name VARCHAR(255),
    actor_ip INET,
    user_agent TEXT,
    
    -- Context
    tenant_id INTEGER,
    organization_id UUID,
    environment VARCHAR(20) NOT NULL DEFAULT 'prod',
    api_version VARCHAR(10),
    client_version VARCHAR(50),
    
    -- Change details
    before_state JSONB,
    after_state JSONB,
    change_summary TEXT,
    change_reason TEXT,
    
    -- Outcome
    status VARCHAR(20) NOT NULL DEFAULT 'success',
    error_code VARCHAR(50),
    error_message TEXT,
    warnings TEXT[],
    
    -- Timing
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    
    -- Compliance and security
    compliance_impact BOOLEAN NOT NULL DEFAULT FALSE,
    security_impact BOOLEAN NOT NULL DEFAULT FALSE,
    risk_level VARCHAR(20) NOT NULL DEFAULT 'low',
    approval_required BOOLEAN NOT NULL DEFAULT FALSE,
    evidence_generated BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Immutable audit trail
    audit_hash VARCHAR(64) NOT NULL,
    
    -- Constraints
    CONSTRAINT chk_actor_type CHECK (actor_type IN ('user', 'system', 'api_client', 'service')),
    CONSTRAINT chk_action_status CHECK (status IN ('success', 'failure', 'partial')),
    CONSTRAINT chk_risk_level_audit CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_duration CHECK (
        (completed_at IS NULL) OR 
        (completed_at >= started_at AND duration_ms >= 0)
    )
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Registry Workflows indexes
CREATE INDEX idx_registry_workflows_tenant ON registry_workflows (tenant_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_registry_workflows_status ON registry_workflows (status) WHERE deleted_at IS NULL;
CREATE INDEX idx_registry_workflows_industry ON registry_workflows (industry_overlay) WHERE deleted_at IS NULL;
CREATE INDEX idx_registry_workflows_owner ON registry_workflows (owner_user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_registry_workflows_search ON registry_workflows USING GIN (to_tsvector('english', workflow_name || ' ' || COALESCE(description, ''))) WHERE deleted_at IS NULL;

-- Workflow Versions indexes
CREATE INDEX idx_workflow_versions_workflow ON workflow_versions (workflow_id, version_number DESC);
CREATE INDEX idx_workflow_versions_status ON workflow_versions (status, published_at DESC);
CREATE INDEX idx_workflow_versions_immutable ON workflow_versions (is_immutable, immutable_since DESC);
CREATE INDEX idx_workflow_versions_alias ON workflow_versions (workflow_id, version_alias) WHERE version_alias IS NOT NULL;

-- Workflow Artifacts indexes
CREATE INDEX idx_workflow_artifacts_version ON workflow_artifacts (version_id, artifact_type);
CREATE INDEX idx_workflow_artifacts_storage ON workflow_artifacts (storage_bucket, storage_path);
CREATE INDEX idx_workflow_artifacts_hash ON workflow_artifacts (content_hash);

-- Usage Analytics indexes
CREATE INDEX idx_usage_analytics_workflow ON usage_analytics (workflow_id, period_start DESC);
CREATE INDEX idx_usage_analytics_tenant ON usage_analytics (tenant_id, period_start DESC);
CREATE INDEX idx_usage_analytics_version ON usage_analytics (version_id, period_start DESC);

-- Audit Log indexes
CREATE INDEX idx_registry_audit_log_resource ON registry_audit_log (resource_type, resource_id, started_at DESC);
CREATE INDEX idx_registry_audit_log_actor ON registry_audit_log (actor_id, started_at DESC);
CREATE INDEX idx_registry_audit_log_tenant ON registry_audit_log (tenant_id, started_at DESC);
CREATE INDEX idx_registry_audit_log_compliance ON registry_audit_log (compliance_impact, security_impact, started_at DESC);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE registry_workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_overlays ENABLE ROW LEVEL SECURITY;
ALTER TABLE registry_signatures ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy_bindings ENABLE ROW LEVEL SECURITY;
ALTER TABLE compatibility_matrix ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_sbom ENABLE ROW LEVEL SECURITY;
ALTER TABLE provenance_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE registry_audit_log ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
CREATE POLICY registry_workflows_tenant_isolation ON registry_workflows
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY workflow_versions_tenant_isolation ON workflow_versions
    FOR ALL TO PUBLIC
    USING (
        EXISTS (
            SELECT 1 FROM registry_workflows w 
            WHERE w.workflow_id = workflow_versions.workflow_id 
            AND w.tenant_id = current_setting('app.current_tenant_id')::INTEGER
        )
    );

CREATE POLICY workflow_artifacts_tenant_isolation ON workflow_artifacts
    FOR ALL TO PUBLIC
    USING (
        EXISTS (
            SELECT 1 FROM workflow_versions v
            JOIN registry_workflows w ON v.workflow_id = w.workflow_id
            WHERE v.version_id = workflow_artifacts.version_id 
            AND w.tenant_id = current_setting('app.current_tenant_id')::INTEGER
        )
    );

CREATE POLICY usage_analytics_tenant_isolation ON usage_analytics
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY registry_audit_log_tenant_isolation ON registry_audit_log
    FOR ALL TO PUBLIC
    USING (
        tenant_id IS NULL OR 
        tenant_id = current_setting('app.current_tenant_id')::INTEGER
    );

-- ============================================================================
-- TRIGGERS FOR AUDIT AND INTEGRITY
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
CREATE TRIGGER trg_registry_workflows_updated_at
    BEFORE UPDATE ON registry_workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_workflow_versions_updated_at
    BEFORE UPDATE ON workflow_versions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_workflow_overlays_updated_at
    BEFORE UPDATE ON workflow_overlays
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to prevent updates to immutable versions
CREATE OR REPLACE FUNCTION prevent_immutable_version_updates()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.is_immutable = TRUE THEN
        -- Allow only specific lifecycle state changes
        IF (NEW.status != OLD.status AND NEW.status IN ('deprecated', 'retired')) OR
           (NEW.deprecated_at != OLD.deprecated_at) OR
           (NEW.retired_at != OLD.retired_at) THEN
            -- Allow lifecycle state changes
            RETURN NEW;
        ELSE
            RAISE EXCEPTION 'Cannot modify immutable workflow version: %', OLD.version_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_prevent_immutable_version_updates
    BEFORE UPDATE ON workflow_versions
    FOR EACH ROW
    EXECUTE FUNCTION prevent_immutable_version_updates();

-- Function to generate audit hash
CREATE OR REPLACE FUNCTION generate_audit_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.audit_hash = encode(
        digest(
            NEW.action_type || NEW.resource_type || NEW.resource_id::text || 
            NEW.actor_id::text || NEW.started_at::text ||
            COALESCE(NEW.before_state::text, '') ||
            COALESCE(NEW.after_state::text, ''),
            'sha256'
        ),
        'hex'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_generate_audit_hash
    BEFORE INSERT ON registry_audit_log
    FOR EACH ROW
    EXECUTE FUNCTION generate_audit_hash();

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function to get workflow by name and tenant
CREATE OR REPLACE FUNCTION get_workflow_by_name(
    p_tenant_id INTEGER,
    p_workflow_name VARCHAR
)
RETURNS registry_workflows AS $$
DECLARE
    workflow_record registry_workflows;
BEGIN
    SELECT * INTO workflow_record
    FROM registry_workflows
    WHERE tenant_id = p_tenant_id
    AND workflow_name = p_workflow_name
    AND deleted_at IS NULL;
    
    RETURN workflow_record;
END;
$$ LANGUAGE plpgsql;

-- Function to resolve version alias
CREATE OR REPLACE FUNCTION resolve_version_alias(
    p_workflow_id UUID,
    p_alias VARCHAR
)
RETURNS workflow_versions AS $$
DECLARE
    version_record workflow_versions;
BEGIN
    SELECT * INTO version_record
    FROM workflow_versions
    WHERE workflow_id = p_workflow_id
    AND (version_alias = p_alias OR version_number = p_alias)
    AND status = 'published'
    AND deleted_at IS NULL
    ORDER BY 
        CASE WHEN version_alias = p_alias THEN 1 ELSE 2 END,
        major_version DESC, minor_version DESC, patch_version DESC
    LIMIT 1;
    
    RETURN version_record;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate usage analytics
CREATE OR REPLACE FUNCTION calculate_usage_metrics(
    p_workflow_id UUID,
    p_period_start TIMESTAMPTZ,
    p_period_end TIMESTAMPTZ
)
RETURNS JSONB AS $$
DECLARE
    metrics JSONB;
BEGIN
    SELECT jsonb_build_object(
        'execution_count', COALESCE(SUM(execution_count), 0),
        'unique_users', COALESCE(SUM(unique_users), 0),
        'avg_success_rate', COALESCE(AVG(success_rate), 0),
        'avg_execution_time', COALESCE(AVG(average_execution_time), 0),
        'total_errors', COALESCE(SUM(execution_count * error_rate / 100), 0)
    ) INTO metrics
    FROM usage_analytics
    WHERE workflow_id = p_workflow_id
    AND period_start >= p_period_start
    AND period_end <= p_period_end;
    
    RETURN metrics;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Create default tenant for system workflows
INSERT INTO tenant_metadata (tenant_id, tenant_name, tenant_code, industry_code, region, status, subscription_tier)
VALUES (0, 'System', 'SYSTEM', 'IT', 'GLOBAL', 'active', 'enterprise')
ON CONFLICT (tenant_id) DO NOTHING;

-- Create system user for automated operations
INSERT INTO users (user_id, tenant_id, email, first_name, last_name, status, profile)
VALUES (0, 0, 'system@registry.local', 'System', 'Registry', 'active', '{"role": "system", "automated": true}')
ON CONFLICT (user_id) DO NOTHING;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for published workflows with latest version info
CREATE OR REPLACE VIEW published_workflows AS
SELECT 
    w.workflow_id,
    w.tenant_id,
    w.workflow_name,
    w.workflow_slug,
    w.automation_type,
    w.industry_overlay,
    w.status as workflow_status,
    v.version_id,
    v.version_number,
    v.status as version_status,
    v.published_at,
    v.is_immutable,
    a.execution_count,
    a.success_rate,
    a.trust_score
FROM registry_workflows w
JOIN workflow_versions v ON w.workflow_id = v.workflow_id
LEFT JOIN LATERAL (
    SELECT 
        execution_count,
        success_rate,
        trust_score
    FROM usage_analytics ua
    WHERE ua.workflow_id = w.workflow_id
    AND ua.version_id = v.version_id
    ORDER BY ua.period_end DESC
    LIMIT 1
) a ON true
WHERE w.deleted_at IS NULL
AND v.status = 'published'
AND v.deleted_at IS NULL;

-- View for workflow compatibility information
CREATE OR REPLACE VIEW workflow_compatibility AS
SELECT 
    w.workflow_id,
    w.workflow_name,
    v.version_id,
    v.version_number,
    cm.min_runtime_version,
    cm.max_runtime_version,
    cm.supported_platforms,
    cm.backward_compatible_with,
    cm.compatibility_tested,
    cm.test_coverage
FROM registry_workflows w
JOIN workflow_versions v ON w.workflow_id = v.workflow_id
JOIN compatibility_matrix cm ON v.version_id = cm.version_id
WHERE w.deleted_at IS NULL
AND v.deleted_at IS NULL;

COMMIT;
