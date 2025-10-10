-- Chapter 18: Security & Compliance Foundation Database Schemas
-- Tasks 18.1.1-18.1.2, 18.2.1-18.2.3, 18.3.1-18.3.2, 18.4.1-18.4.2

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ================================
-- Section 18.1: RBAC/ABAC Tables
-- ================================

-- Task 18.1.1: RBAC role hierarchies table
CREATE TABLE IF NOT EXISTS authz_policy_role (
    role_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    role_name VARCHAR(255) NOT NULL UNIQUE,
    role_level VARCHAR(50) NOT NULL CHECK (role_level IN ('executive', 'manager', 'user', 'system')),
    parent_role_id UUID REFERENCES authz_policy_role(role_id),
    permissions JSONB NOT NULL DEFAULT '[]',
    industry_overlay VARCHAR(50) CHECK (industry_overlay IN ('saas', 'banking', 'insurance', 'global')),
    compliance_frameworks JSONB DEFAULT '[]',
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    tenant_id INTEGER NOT NULL,
    created_by_user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    role_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            role_id::text || role_name || role_level || 
            COALESCE(parent_role_id::text, '') || permissions::text || 
            COALESCE(industry_overlay, '') || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Task 18.1.2: ABAC attribute policies table
CREATE TABLE IF NOT EXISTS authz_policy_attr (
    attr_policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_name VARCHAR(255) NOT NULL,
    policy_type VARCHAR(50) NOT NULL CHECK (policy_type IN ('regional', 'temporal', 'sensitivity', 'industry', 'sla_tier')),
    attributes JSONB NOT NULL,
    rules JSONB NOT NULL,
    conditions JSONB NOT NULL,
    actions JSONB NOT NULL,
    compliance_frameworks JSONB DEFAULT '[]',
    industry_overlay VARCHAR(50) CHECK (industry_overlay IN ('saas', 'banking', 'insurance', 'global')),
    is_active BOOLEAN NOT NULL DEFAULT true,
    tenant_id INTEGER NOT NULL,
    created_by_user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    attr_policy_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            attr_policy_id::text || policy_name || policy_type || 
            attributes::text || rules::text || conditions::text || 
            actions::text || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- ================================
-- Section 18.2: PII Handling Tables
-- ================================

-- Task 18.2.1: PII classification categories table
CREATE TABLE IF NOT EXISTS pii_policy_class (
    class_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    class_name VARCHAR(255) NOT NULL,
    class_type VARCHAR(100) NOT NULL,
    industry_overlay VARCHAR(50) NOT NULL CHECK (industry_overlay IN ('saas', 'banking', 'insurance', 'global')),
    pii_categories JSONB NOT NULL,
    sensitivity_level VARCHAR(20) NOT NULL CHECK (sensitivity_level IN ('public', 'internal', 'confidential', 'restricted', 'top_secret')),
    regulatory_frameworks JSONB NOT NULL DEFAULT '[]',
    field_patterns JSONB NOT NULL DEFAULT '[]',
    detection_rules JSONB NOT NULL DEFAULT '[]',
    tenant_id INTEGER NOT NULL,
    created_by_user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    class_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            class_id::text || class_name || class_type || 
            industry_overlay || pii_categories::text || 
            sensitivity_level || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Task 18.2.2: PII masking and anonymization rules table
CREATE TABLE IF NOT EXISTS pii_policy_mask (
    mask_policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_name VARCHAR(255) NOT NULL,
    class_id UUID NOT NULL REFERENCES pii_policy_class(class_id),
    masking_type VARCHAR(50) NOT NULL CHECK (masking_type IN ('redact', 'mask', 'encrypt', 'anonymize', 'pseudonymize')),
    masking_rules JSONB NOT NULL,
    field_mappings JSONB NOT NULL,
    anonymization_method VARCHAR(50) CHECK (anonymization_method IN ('k_anonymity', 'l_diversity', 't_closeness', 'differential_privacy')),
    anonymization_params JSONB,
    industry_overlay VARCHAR(50) NOT NULL CHECK (industry_overlay IN ('saas', 'banking', 'insurance', 'global')),
    compliance_frameworks JSONB NOT NULL DEFAULT '[]',
    is_active BOOLEAN NOT NULL DEFAULT true,
    tenant_id INTEGER NOT NULL,
    created_by_user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    mask_policy_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            mask_policy_id::text || policy_name || masking_type || 
            masking_rules::text || field_mappings::text || 
            industry_overlay || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Task 18.2.3: PII retention policies table
CREATE TABLE IF NOT EXISTS pii_policy_retention (
    retention_policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_name VARCHAR(255) NOT NULL,
    class_id UUID NOT NULL REFERENCES pii_policy_class(class_id),
    sla_tier VARCHAR(20) NOT NULL CHECK (sla_tier IN ('bronze', 'silver', 'gold', 'enterprise')),
    retention_period_days INTEGER NOT NULL,
    purge_method VARCHAR(50) NOT NULL CHECK (purge_method IN ('soft_delete', 'hard_delete', 'archive', 'anonymize')),
    purge_schedule VARCHAR(100) NOT NULL,
    legal_hold_exemptions JSONB DEFAULT '[]',
    industry_overlay VARCHAR(50) NOT NULL CHECK (industry_overlay IN ('saas', 'banking', 'insurance', 'global')),
    compliance_frameworks JSONB NOT NULL DEFAULT '[]',
    auto_purge_enabled BOOLEAN NOT NULL DEFAULT true,
    tenant_id INTEGER NOT NULL,
    created_by_user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    retention_policy_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            retention_policy_id::text || policy_name || sla_tier || 
            retention_period_days::text || purge_method || 
            industry_overlay || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- ================================
-- Section 18.3: Residency Enforcement Tables
-- ================================

-- Task 18.3.1: Tenant residency tags table
CREATE TABLE IF NOT EXISTS residency_policy_tenant (
    residency_tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL UNIQUE,
    primary_region VARCHAR(10) NOT NULL,
    allowed_regions JSONB NOT NULL DEFAULT '[]',
    restricted_regions JSONB NOT NULL DEFAULT '[]',
    data_classification_rules JSONB NOT NULL DEFAULT '{}',
    cross_border_restrictions JSONB NOT NULL DEFAULT '{}',
    industry_overlay VARCHAR(50) NOT NULL CHECK (industry_overlay IN ('saas', 'banking', 'insurance', 'global')),
    compliance_frameworks JSONB NOT NULL DEFAULT '[]',
    failover_allowed BOOLEAN NOT NULL DEFAULT false,
    backup_region_restricted BOOLEAN NOT NULL DEFAULT true,
    compute_region_restricted BOOLEAN NOT NULL DEFAULT true,
    model_inference_region_restricted BOOLEAN NOT NULL DEFAULT true,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_by_user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    residency_tenant_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            residency_tenant_id::text || tenant_id::text || 
            primary_region || allowed_regions::text || 
            restricted_regions::text || industry_overlay || 
            created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Task 18.3.2: Residency enforcement policies table
CREATE TABLE IF NOT EXISTS residency_policy_rule (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_name VARCHAR(255) NOT NULL,
    rule_type VARCHAR(50) NOT NULL CHECK (rule_type IN ('data_movement', 'compute_placement', 'backup_location', 'model_inference')),
    source_region VARCHAR(10) NOT NULL,
    target_region VARCHAR(10) NOT NULL,
    data_classifications JSONB NOT NULL DEFAULT '[]',
    enforcement_action VARCHAR(50) NOT NULL CHECK (enforcement_action IN ('allow', 'deny', 'require_approval', 'log_only')),
    violation_severity VARCHAR(20) NOT NULL CHECK (violation_severity IN ('low', 'medium', 'high', 'critical')),
    industry_overlay VARCHAR(50) NOT NULL CHECK (industry_overlay IN ('saas', 'banking', 'insurance', 'global')),
    compliance_frameworks JSONB NOT NULL DEFAULT '[]',
    exception_conditions JSONB DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT true,
    tenant_id INTEGER,
    created_by_user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    residency_rule_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            rule_id::text || rule_name || rule_type || 
            source_region || target_region || 
            data_classifications::text || enforcement_action || 
            industry_overlay || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- ================================
-- Section 18.4: Penetration Testing Tables
-- ================================

-- Task 18.4.1: Pen-test configuration and scope table
CREATE TABLE IF NOT EXISTS pentest_policy_config (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_name VARCHAR(255) NOT NULL,
    test_scope VARCHAR(50) NOT NULL CHECK (test_scope IN ('infra', 'application', 'database', 'container', 'orchestrator', 'tenant_isolation')),
    industry_overlay VARCHAR(50) NOT NULL CHECK (industry_overlay IN ('saas', 'banking', 'insurance', 'global')),
    test_types JSONB NOT NULL DEFAULT '[]',
    target_systems JSONB NOT NULL DEFAULT '[]',
    attack_vectors JSONB NOT NULL DEFAULT '[]',
    test_schedule VARCHAR(100) NOT NULL,
    automation_enabled BOOLEAN NOT NULL DEFAULT true,
    ci_cd_integration BOOLEAN NOT NULL DEFAULT false,
    compliance_frameworks JSONB NOT NULL DEFAULT '[]',
    severity_thresholds JSONB NOT NULL DEFAULT '{}',
    notification_rules JSONB NOT NULL DEFAULT '{}',
    tenant_id INTEGER NOT NULL,
    created_by_user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    pentest_config_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            config_id::text || config_name || test_scope || 
            industry_overlay || test_types::text || 
            target_systems::text || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Task 18.4.2: Pen-test results and evidence table
CREATE TABLE IF NOT EXISTS pentest_policy_result (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_id UUID NOT NULL REFERENCES pentest_policy_config(config_id),
    test_run_id VARCHAR(255) NOT NULL,
    test_type VARCHAR(100) NOT NULL,
    attack_vector VARCHAR(255) NOT NULL,
    target_system VARCHAR(255) NOT NULL,
    test_status VARCHAR(20) NOT NULL CHECK (test_status IN ('running', 'completed', 'failed', 'cancelled')),
    result_status VARCHAR(20) NOT NULL CHECK (result_status IN ('success', 'failed', 'blocked', 'detected', 'partial')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info')),
    vulnerability_details JSONB NOT NULL DEFAULT '{}',
    evidence_data JSONB NOT NULL DEFAULT '{}',
    remediation_status VARCHAR(50) NOT NULL DEFAULT 'open',
    remediation_plan TEXT,
    remediation_priority INTEGER CHECK (remediation_priority BETWEEN 1 AND 5),
    false_positive BOOLEAN NOT NULL DEFAULT false,
    cvss_score DECIMAL(3,1) CHECK (cvss_score BETWEEN 0.0 AND 10.0),
    cve_references JSONB DEFAULT '[]',
    digital_signature VARCHAR(500),
    tenant_id INTEGER NOT NULL,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Immutable audit hash
    pentest_result_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            result_id::text || test_run_id || test_type || 
            attack_vector || target_system || result_status || 
            severity || executed_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- ================================
-- Row Level Security (RLS) Policies
-- ================================

-- Enable RLS on all tables
ALTER TABLE authz_policy_role ENABLE ROW LEVEL SECURITY;
ALTER TABLE authz_policy_attr ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_policy_class ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_policy_mask ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_policy_retention ENABLE ROW LEVEL SECURITY;
ALTER TABLE residency_policy_tenant ENABLE ROW LEVEL SECURITY;
ALTER TABLE residency_policy_rule ENABLE ROW LEVEL SECURITY;
ALTER TABLE pentest_policy_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE pentest_policy_result ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation
CREATE POLICY authz_policy_role_rls_policy ON authz_policy_role
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY authz_policy_attr_rls_policy ON authz_policy_attr
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY pii_policy_class_rls_policy ON pii_policy_class
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY pii_policy_mask_rls_policy ON pii_policy_mask
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY pii_policy_retention_rls_policy ON pii_policy_retention
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY residency_policy_tenant_rls_policy ON residency_policy_tenant
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY residency_policy_rule_rls_policy ON residency_policy_rule
    FOR ALL USING (tenant_id IS NULL OR tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY pentest_policy_config_rls_policy ON pentest_policy_config
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY pentest_policy_result_rls_policy ON pentest_policy_result
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ================================
-- Performance Indexes
-- ================================

-- RBAC/ABAC indexes
CREATE INDEX IF NOT EXISTS idx_authz_policy_role_tenant ON authz_policy_role(tenant_id);
CREATE INDEX IF NOT EXISTS idx_authz_policy_role_level ON authz_policy_role(role_level);
CREATE INDEX IF NOT EXISTS idx_authz_policy_role_industry ON authz_policy_role(industry_overlay);
CREATE INDEX IF NOT EXISTS idx_authz_policy_role_parent ON authz_policy_role(parent_role_id);
CREATE INDEX IF NOT EXISTS idx_authz_policy_role_active ON authz_policy_role(is_active);

CREATE INDEX IF NOT EXISTS idx_authz_policy_attr_tenant ON authz_policy_attr(tenant_id);
CREATE INDEX IF NOT EXISTS idx_authz_policy_attr_type ON authz_policy_attr(policy_type);
CREATE INDEX IF NOT EXISTS idx_authz_policy_attr_industry ON authz_policy_attr(industry_overlay);
CREATE INDEX IF NOT EXISTS idx_authz_policy_attr_active ON authz_policy_attr(is_active);

-- PII handling indexes
CREATE INDEX IF NOT EXISTS idx_pii_policy_class_tenant ON pii_policy_class(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pii_policy_class_industry ON pii_policy_class(industry_overlay);
CREATE INDEX IF NOT EXISTS idx_pii_policy_class_sensitivity ON pii_policy_class(sensitivity_level);

CREATE INDEX IF NOT EXISTS idx_pii_policy_mask_tenant ON pii_policy_mask(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pii_policy_mask_class ON pii_policy_mask(class_id);
CREATE INDEX IF NOT EXISTS idx_pii_policy_mask_type ON pii_policy_mask(masking_type);
CREATE INDEX IF NOT EXISTS idx_pii_policy_mask_industry ON pii_policy_mask(industry_overlay);
CREATE INDEX IF NOT EXISTS idx_pii_policy_mask_active ON pii_policy_mask(is_active);

CREATE INDEX IF NOT EXISTS idx_pii_policy_retention_tenant ON pii_policy_retention(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pii_policy_retention_class ON pii_policy_retention(class_id);
CREATE INDEX IF NOT EXISTS idx_pii_policy_retention_sla ON pii_policy_retention(sla_tier);
CREATE INDEX IF NOT EXISTS idx_pii_policy_retention_industry ON pii_policy_retention(industry_overlay);
CREATE INDEX IF NOT EXISTS idx_pii_policy_retention_auto_purge ON pii_policy_retention(auto_purge_enabled);

-- Residency enforcement indexes
CREATE INDEX IF NOT EXISTS idx_residency_policy_tenant_tenant ON residency_policy_tenant(tenant_id);
CREATE INDEX IF NOT EXISTS idx_residency_policy_tenant_region ON residency_policy_tenant(primary_region);
CREATE INDEX IF NOT EXISTS idx_residency_policy_tenant_industry ON residency_policy_tenant(industry_overlay);
CREATE INDEX IF NOT EXISTS idx_residency_policy_tenant_active ON residency_policy_tenant(is_active);

CREATE INDEX IF NOT EXISTS idx_residency_policy_rule_tenant ON residency_policy_rule(tenant_id);
CREATE INDEX IF NOT EXISTS idx_residency_policy_rule_type ON residency_policy_rule(rule_type);
CREATE INDEX IF NOT EXISTS idx_residency_policy_rule_source ON residency_policy_rule(source_region);
CREATE INDEX IF NOT EXISTS idx_residency_policy_rule_target ON residency_policy_rule(target_region);
CREATE INDEX IF NOT EXISTS idx_residency_policy_rule_action ON residency_policy_rule(enforcement_action);
CREATE INDEX IF NOT EXISTS idx_residency_policy_rule_severity ON residency_policy_rule(violation_severity);
CREATE INDEX IF NOT EXISTS idx_residency_policy_rule_industry ON residency_policy_rule(industry_overlay);
CREATE INDEX IF NOT EXISTS idx_residency_policy_rule_active ON residency_policy_rule(is_active);

-- Penetration testing indexes
CREATE INDEX IF NOT EXISTS idx_pentest_policy_config_tenant ON pentest_policy_config(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_config_scope ON pentest_policy_config(test_scope);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_config_industry ON pentest_policy_config(industry_overlay);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_config_automation ON pentest_policy_config(automation_enabled);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_config_cicd ON pentest_policy_config(ci_cd_integration);

CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_tenant ON pentest_policy_result(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_config ON pentest_policy_result(config_id);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_run ON pentest_policy_result(test_run_id);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_type ON pentest_policy_result(test_type);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_status ON pentest_policy_result(test_status);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_result ON pentest_policy_result(result_status);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_severity ON pentest_policy_result(severity);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_remediation ON pentest_policy_result(remediation_status);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_executed ON pentest_policy_result(executed_at);
CREATE INDEX IF NOT EXISTS idx_pentest_policy_result_cvss ON pentest_policy_result(cvss_score);

-- ================================
-- Comments for Documentation
-- ================================

COMMENT ON TABLE authz_policy_role IS 'RBAC role hierarchies with industry overlays and compliance frameworks (Task 18.1.1)';
COMMENT ON TABLE authz_policy_attr IS 'ABAC attribute policies for contextual access control (Task 18.1.2)';
COMMENT ON TABLE pii_policy_class IS 'PII classification categories by industry and sensitivity level (Task 18.2.1)';
COMMENT ON TABLE pii_policy_mask IS 'PII masking and anonymization rules with compliance frameworks (Task 18.2.2)';
COMMENT ON TABLE pii_policy_retention IS 'PII retention policies with SLA-based purge schedules (Task 18.2.3)';
COMMENT ON TABLE residency_policy_tenant IS 'Tenant-specific data residency tags and restrictions (Task 18.3.1)';
COMMENT ON TABLE residency_policy_rule IS 'Data residency enforcement policies and rules (Task 18.3.2)';
COMMENT ON TABLE pentest_policy_config IS 'Penetration testing configuration and scope definitions (Task 18.4.1)';
COMMENT ON TABLE pentest_policy_result IS 'Penetration testing results and vulnerability evidence (Task 18.4.2)';
