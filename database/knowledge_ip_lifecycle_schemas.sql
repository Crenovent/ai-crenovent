-- ===============================================================================
-- Chapter 24: Knowledge & IP Lifecycle Validation Database Schemas
-- Tasks 24.1.1, 24.2.1, 24.3.7: Extend versioning, add rollback, anonymization schemas
-- ===============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===============================================================================
-- Section 24.1: Versioning Schema Extensions
-- Tasks 24.1.1-24.1.3: Extend cap_version schema with lifecycle states and hashing
-- ===============================================================================

-- Task 24.1.1: Extend cap_version schema with lifecycle states
-- Note: This extends the existing cap_version_* tables from Chapter 14
CREATE TABLE IF NOT EXISTS cap_version_lifecycle (
    lifecycle_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_id UUID NOT NULL, -- References existing cap_version tables
    lifecycle_state VARCHAR(50) NOT NULL DEFAULT 'draft', -- draft, published, promoted, deprecated, retired
    parent_version_id UUID, -- For version lineage tracking
    promotion_evidence_id UUID, -- Links to evidence packs for promotion decisions
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    
    -- Task 24.1.2: Lifecycle state constraints
    CONSTRAINT valid_lifecycle_state CHECK (lifecycle_state IN ('draft', 'published', 'promoted', 'deprecated', 'retired')),
    
    -- Ensure only one active lifecycle per version
    UNIQUE(version_id, lifecycle_state)
);

-- Task 24.1.3: Add version hashing for immutability
CREATE TABLE IF NOT EXISTS cap_version_hash (
    hash_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_id UUID NOT NULL, -- References existing cap_version tables
    content_hash VARCHAR(64) NOT NULL, -- SHA256 hash of version content
    hash_algorithm VARCHAR(20) NOT NULL DEFAULT 'SHA256',
    hash_metadata JSONB NOT NULL DEFAULT '{}', -- Additional hash metadata (salt, timestamp, etc.)
    is_verified BOOLEAN DEFAULT FALSE, -- Whether hash has been cryptographically verified
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    
    -- Ensure unique hash per version
    UNIQUE(version_id, content_hash)
);

-- ===============================================================================
-- Section 24.2: Rollback Schema Extensions
-- Tasks 24.2.1-24.2.3: Add rollback target tracking and tenant-specific rollbacks
-- ===============================================================================

-- Task 24.2.1: Extend version schema with rollback targets
CREATE TABLE IF NOT EXISTS cap_version_rollback (
    rollback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    current_version_id UUID NOT NULL, -- Current version that may need rollback
    rollback_target_id UUID NOT NULL, -- Target version to rollback to
    rollback_reason VARCHAR(500) NOT NULL, -- Reason for rollback (SLA breach, drift, security, etc.)
    rollback_trigger VARCHAR(100) NOT NULL, -- automated_sla, automated_drift, automated_security, manual
    rollback_status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, in_progress, completed, failed
    rollback_evidence_id UUID, -- Links to evidence packs for rollback decisions
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_at TIMESTAMPTZ, -- When rollback was actually executed
    executed_by_user_id INTEGER, -- User who executed rollback (if manual)
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    
    -- Task 24.2.2: Rollback status constraints
    CONSTRAINT valid_rollback_status CHECK (rollback_status IN ('pending', 'in_progress', 'completed', 'failed')),
    CONSTRAINT valid_rollback_trigger CHECK (rollback_trigger IN ('automated_sla', 'automated_drift', 'automated_bias', 'automated_security', 'manual_emergency', 'manual_planned'))
);

-- Task 24.2.3: Support tenant-pinned rollbacks
CREATE TABLE IF NOT EXISTS cap_version_tenant_pins (
    pin_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    capability_id UUID NOT NULL, -- References dsl_capability_registry
    pinned_version_id UUID NOT NULL, -- Version pinned for this tenant
    pin_reason VARCHAR(500), -- Why this tenant is pinned to this version
    pin_expiry_date TIMESTAMPTZ, -- Optional expiry for temporary pins
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    
    -- Ensure only one active pin per tenant per capability
    UNIQUE(tenant_id, capability_id, is_active) DEFERRABLE INITIALLY DEFERRED
);

-- ===============================================================================
-- Section 24.3: Anonymization Schema Extensions
-- Tasks 24.3.1-24.3.7: Anonymization policies, transformations, and artifact storage
-- ===============================================================================

-- Task 24.3.1: Define anonymization policies
CREATE TABLE IF NOT EXISTS anonymization_policy (
    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_name VARCHAR(100) UNIQUE NOT NULL,
    industry_code VARCHAR(50) NOT NULL, -- SaaS, Banking, Insurance, etc.
    anonymization_standard VARCHAR(50) NOT NULL, -- k_anonymity, l_diversity, t_closeness
    k_anonymity_threshold INTEGER DEFAULT 5, -- Minimum group size for k-anonymity
    l_diversity_threshold INTEGER DEFAULT 2, -- Minimum diversity for l-diversity
    t_closeness_threshold DECIMAL(3,2) DEFAULT 0.2, -- Maximum distance for t-closeness
    sensitive_attributes TEXT[] DEFAULT '{}', -- List of sensitive attributes to protect
    quasi_identifiers TEXT[] DEFAULT '{}', -- List of quasi-identifier fields
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    
    CONSTRAINT valid_anonymization_standard CHECK (anonymization_standard IN ('k_anonymity', 'l_diversity', 't_closeness', 'differential_privacy'))
);

-- Task 24.3.2: Map sensitive fields per industry
CREATE TABLE IF NOT EXISTS anonymization_field_mapping (
    mapping_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_id UUID NOT NULL REFERENCES anonymization_policy(policy_id),
    field_name VARCHAR(100) NOT NULL, -- e.g., tenant_id, email, SSN, PAN, claims_id
    field_type VARCHAR(50) NOT NULL, -- identifier, quasi_identifier, sensitive_attribute
    sensitivity_level VARCHAR(20) NOT NULL, -- low, medium, high, critical
    transformation_method VARCHAR(50) NOT NULL, -- mask, hash, bucket, suppress, generalize
    transformation_params JSONB NOT NULL DEFAULT '{}', -- Parameters for transformation method
    compliance_frameworks TEXT[] DEFAULT '{}', -- GDPR, HIPAA, DPDP, etc.
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_field_type CHECK (field_type IN ('identifier', 'quasi_identifier', 'sensitive_attribute')),
    CONSTRAINT valid_sensitivity_level CHECK (sensitivity_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT valid_transformation_method CHECK (transformation_method IN ('mask', 'hash', 'bucket', 'suppress', 'generalize', 'encrypt'))
);

-- Task 24.3.7: Store anonymized artifacts separately
CREATE TABLE IF NOT EXISTS anonymized_artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_artifact_id UUID NOT NULL, -- Reference to original artifact
    artifact_type VARCHAR(50) NOT NULL, -- trace, dataset, model_output, template
    anonymization_policy_id UUID NOT NULL REFERENCES anonymization_policy(policy_id),
    anonymized_content JSONB NOT NULL, -- Anonymized version of the content
    anonymization_metadata JSONB NOT NULL DEFAULT '{}', -- Metadata about anonymization process
    privacy_metrics JSONB NOT NULL DEFAULT '{}', -- k-anonymity, l-diversity, t-closeness scores
    source_tenant_count INTEGER DEFAULT 1, -- Number of tenants contributing to this anonymized artifact
    is_cross_tenant_safe BOOLEAN DEFAULT FALSE, -- Whether safe for cross-tenant use
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retention_until TIMESTAMPTZ, -- When this anonymized artifact should be purged
    evidence_pack_id UUID, -- Links to evidence pack proving anonymization compliance
    
    CONSTRAINT valid_artifact_type CHECK (artifact_type IN ('trace', 'dataset', 'model_output', 'template', 'benchmark'))
);

-- Task 24.3.5: Trace anonymization tracking
CREATE TABLE IF NOT EXISTS anonymized_traces (
    trace_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_trace_id VARCHAR(100) NOT NULL, -- Original OpenTelemetry trace ID
    anonymized_trace_id VARCHAR(100) NOT NULL, -- New anonymized trace ID
    anonymization_policy_id UUID NOT NULL REFERENCES anonymization_policy(policy_id),
    tenant_mappings JSONB NOT NULL DEFAULT '{}', -- Original tenant_id -> anonymized_tenant_id mappings
    field_transformations JSONB NOT NULL DEFAULT '{}', -- Field-level transformation log
    anonymization_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_cross_tenant_safe BOOLEAN DEFAULT FALSE,
    
    UNIQUE(original_trace_id),
    UNIQUE(anonymized_trace_id)
);

-- ===============================================================================
-- Performance Indexes
-- ===============================================================================

-- Lifecycle indexes
CREATE INDEX IF NOT EXISTS idx_cap_version_lifecycle_version_id ON cap_version_lifecycle(version_id);
CREATE INDEX IF NOT EXISTS idx_cap_version_lifecycle_state ON cap_version_lifecycle(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_cap_version_lifecycle_tenant ON cap_version_lifecycle(tenant_id);

-- Hash indexes
CREATE INDEX IF NOT EXISTS idx_cap_version_hash_version_id ON cap_version_hash(version_id);
CREATE INDEX IF NOT EXISTS idx_cap_version_hash_content ON cap_version_hash(content_hash);
CREATE INDEX IF NOT EXISTS idx_cap_version_hash_tenant ON cap_version_hash(tenant_id);

-- Rollback indexes
CREATE INDEX IF NOT EXISTS idx_cap_version_rollback_current ON cap_version_rollback(current_version_id);
CREATE INDEX IF NOT EXISTS idx_cap_version_rollback_target ON cap_version_rollback(rollback_target_id);
CREATE INDEX IF NOT EXISTS idx_cap_version_rollback_status ON cap_version_rollback(rollback_status);
CREATE INDEX IF NOT EXISTS idx_cap_version_rollback_tenant ON cap_version_rollback(tenant_id);

-- Tenant pin indexes
CREATE INDEX IF NOT EXISTS idx_cap_version_tenant_pins_tenant ON cap_version_tenant_pins(tenant_id);
CREATE INDEX IF NOT EXISTS idx_cap_version_tenant_pins_capability ON cap_version_tenant_pins(capability_id);
CREATE INDEX IF NOT EXISTS idx_cap_version_tenant_pins_active ON cap_version_tenant_pins(is_active);

-- Anonymization indexes
CREATE INDEX IF NOT EXISTS idx_anonymization_policy_industry ON anonymization_policy(industry_code);
CREATE INDEX IF NOT EXISTS idx_anonymization_field_mapping_policy ON anonymization_field_mapping(policy_id);
CREATE INDEX IF NOT EXISTS idx_anonymized_artifacts_type ON anonymized_artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_anonymized_artifacts_policy ON anonymized_artifacts(anonymization_policy_id);
CREATE INDEX IF NOT EXISTS idx_anonymized_traces_original ON anonymized_traces(original_trace_id);

-- ===============================================================================
-- Row Level Security (RLS) Policies
-- ===============================================================================

-- Enable RLS on all tables
ALTER TABLE cap_version_lifecycle ENABLE ROW LEVEL SECURITY;
ALTER TABLE cap_version_hash ENABLE ROW LEVEL SECURITY;
ALTER TABLE cap_version_rollback ENABLE ROW LEVEL SECURITY;
ALTER TABLE cap_version_tenant_pins ENABLE ROW LEVEL SECURITY;
ALTER TABLE anonymization_policy ENABLE ROW LEVEL SECURITY;
ALTER TABLE anonymization_field_mapping ENABLE ROW LEVEL SECURITY;
ALTER TABLE anonymized_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE anonymized_traces ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation
CREATE POLICY cap_version_lifecycle_tenant_isolation ON cap_version_lifecycle
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY cap_version_hash_tenant_isolation ON cap_version_hash
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY cap_version_rollback_tenant_isolation ON cap_version_rollback
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY cap_version_tenant_pins_tenant_isolation ON cap_version_tenant_pins
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- Anonymization policies are cross-tenant but industry-scoped
CREATE POLICY anonymization_policy_industry_access ON anonymization_policy
    USING (TRUE); -- Cross-tenant access for anonymization policies

CREATE POLICY anonymization_field_mapping_access ON anonymization_field_mapping
    USING (TRUE); -- Cross-tenant access for field mappings

-- Anonymized artifacts are cross-tenant safe by design
CREATE POLICY anonymized_artifacts_cross_tenant_access ON anonymized_artifacts
    USING (is_cross_tenant_safe = TRUE OR 
           artifact_id IN (SELECT artifact_id FROM anonymized_artifacts WHERE source_tenant_count = 1));

CREATE POLICY anonymized_traces_cross_tenant_access ON anonymized_traces
    USING (is_cross_tenant_safe = TRUE);

-- ===============================================================================
-- Initial Data Population
-- ===============================================================================

-- Task 24.3.1: Insert default anonymization policies for each industry
INSERT INTO anonymization_policy (policy_name, industry_code, anonymization_standard, k_anonymity_threshold, sensitive_attributes, quasi_identifiers, created_by_user_id) VALUES
('SaaS K-Anonymity Policy', 'SaaS', 'k_anonymity', 5, 
 ARRAY['user_email', 'customer_id', 'subscription_id', 'billing_info'], 
 ARRAY['company_size', 'industry_vertical', 'plan_tier', 'region'], 1),
('Banking L-Diversity Policy', 'Banking', 'l_diversity', 3, 
 ARRAY['account_number', 'ssn', 'pan_number', 'loan_id', 'transaction_id'], 
 ARRAY['age_group', 'income_bracket', 'credit_score_range', 'region'], 1),
('Insurance T-Closeness Policy', 'Insurance', 't_closeness', 5, 
 ARRAY['policy_number', 'claim_id', 'ssn', 'medical_record_id'], 
 ARRAY['age_group', 'coverage_type', 'risk_category', 'region'], 1)
ON CONFLICT (policy_name) DO NOTHING;

-- Task 24.3.2: Insert default field mappings for sensitive data
INSERT INTO anonymization_field_mapping (policy_id, field_name, field_type, sensitivity_level, transformation_method, transformation_params, compliance_frameworks) 
SELECT 
    p.policy_id,
    field_data.field_name,
    field_data.field_type,
    field_data.sensitivity_level,
    field_data.transformation_method,
    field_data.transformation_params::JSONB,
    field_data.compliance_frameworks
FROM anonymization_policy p
CROSS JOIN (
    VALUES 
    ('tenant_id', 'identifier', 'critical', 'hash', '{"algorithm": "SHA256", "salt": "tenant_salt"}', ARRAY['GDPR', 'DPDP']),
    ('user_email', 'identifier', 'high', 'mask', '{"pattern": "XXXXX@domain.com"}', ARRAY['GDPR', 'HIPAA']),
    ('user_id', 'quasi_identifier', 'medium', 'hash', '{"algorithm": "SHA256"}', ARRAY['GDPR']),
    ('account_id', 'identifier', 'high', 'hash', '{"algorithm": "SHA256"}', ARRAY['SOX', 'GDPR']),
    ('ssn', 'sensitive_attribute', 'critical', 'encrypt', '{"algorithm": "AES256"}', ARRAY['HIPAA', 'SOX']),
    ('pan_number', 'sensitive_attribute', 'critical', 'mask', '{"pattern": "XXXX-XXXX-XXXX-1234"}', ARRAY['RBI', 'DPDP']),
    ('region', 'quasi_identifier', 'low', 'generalize', '{"levels": ["country", "continent"]}', ARRAY['GDPR'])
) AS field_data(field_name, field_type, sensitivity_level, transformation_method, transformation_params, compliance_frameworks)
WHERE p.industry_code IN ('SaaS', 'Banking', 'Insurance')
ON CONFLICT DO NOTHING;

COMMENT ON TABLE cap_version_lifecycle IS 'Chapter 24.1: Tracks lifecycle states for capability versions (draft → published → promoted → deprecated → retired)';
COMMENT ON TABLE cap_version_hash IS 'Chapter 24.1: Stores cryptographic hashes for version immutability and integrity verification';
COMMENT ON TABLE cap_version_rollback IS 'Chapter 24.2: Manages rollback targets and execution for capability versions';
COMMENT ON TABLE cap_version_tenant_pins IS 'Chapter 24.2: Tracks tenant-specific version pinning for isolated rollbacks';
COMMENT ON TABLE anonymization_policy IS 'Chapter 24.3: Defines anonymization standards and thresholds per industry';
COMMENT ON TABLE anonymization_field_mapping IS 'Chapter 24.3: Maps sensitive fields to transformation methods for anonymization';
COMMENT ON TABLE anonymized_artifacts IS 'Chapter 24.3: Stores anonymized versions of traces, datasets, and model outputs';
COMMENT ON TABLE anonymized_traces IS 'Chapter 24.3: Tracks OpenTelemetry trace anonymization for cross-tenant safety';
