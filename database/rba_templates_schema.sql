-- ===============================================================================
-- Chapter 19.1: RBA Templates Database Schema Extensions
-- Tasks 19.1.1, 19.1.3, 19.1.6: Extend existing dsl_capability_registry for RBA templates
-- ===============================================================================

-- Task 19.1.1: Extend existing dsl_capability_registry for RBA template metadata
-- Note: Using existing dsl_capability_registry table and adding RBA-specific extensions

-- Task 19.1.35: RBA Template lifecycle tracking (extends existing capability lifecycle)
CREATE TABLE IF NOT EXISTS rba_template_lifecycle (
    lifecycle_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    capability_id UUID NOT NULL REFERENCES dsl_capability_registry(capability_id),
    lifecycle_state VARCHAR(50) NOT NULL DEFAULT 'draft', -- 'draft', 'published', 'promoted', 'deprecated', 'retired'
    state_metadata JSONB NOT NULL DEFAULT '{}',
    transition_history JSONB NOT NULL DEFAULT '[]',
    
    -- Task 19.1.3: Industry, SLA, jurisdiction, trust metadata
    industry_overlays TEXT[] DEFAULT '{}', -- ['SaaS', 'Banking', 'Insurance']
    sla_requirements JSONB DEFAULT '{}', -- SLA tier requirements
    jurisdiction_compliance JSONB DEFAULT '{}', -- Jurisdiction-specific compliance
    
    tenant_id INTEGER REFERENCES tenants(id),
    created_by_user_id INTEGER REFERENCES users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_lifecycle_state CHECK (lifecycle_state IN ('draft', 'published', 'promoted', 'deprecated', 'retired'))
);

-- Task 19.1.6: RBA Template contracts (extends existing capability contracts)
CREATE TABLE IF NOT EXISTS rba_template_contracts (
    contract_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    capability_id UUID NOT NULL REFERENCES dsl_capability_registry(capability_id),
    contract_type VARCHAR(50) NOT NULL, -- 'input', 'output', 'sla', 'governance'
    contract_schema JSONB NOT NULL, -- JSON Schema definition
    contract_version VARCHAR(20) NOT NULL DEFAULT '1.0',
    
    -- Dynamic contract parameters based on industry/tenant
    parameter_overrides JSONB DEFAULT '{}',
    validation_rules JSONB DEFAULT '{}',
    
    tenant_id INTEGER REFERENCES tenants(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE(capability_id, contract_type, tenant_id)
);

-- Task 19.1.9-19.1.34: Industry-specific template metadata
CREATE TABLE IF NOT EXISTS rba_industry_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    capability_id UUID NOT NULL REFERENCES dsl_capability_registry(capability_id),
    template_name VARCHAR(200) NOT NULL,
    industry_code VARCHAR(50) NOT NULL, -- 'SaaS', 'Banking', 'Insurance'
    template_category VARCHAR(100) NOT NULL, -- 'arr_rollup', 'churn_alerts', 'npa_detection', etc.
    
    -- Dynamic template configuration (not hardcoded)
    template_config JSONB NOT NULL DEFAULT '{}', -- Dynamic configuration parameters
    parameter_schema JSONB NOT NULL DEFAULT '{}', -- Schema for configurable parameters
    default_parameters JSONB NOT NULL DEFAULT '{}', -- Default parameter values
    
    -- Policy and compliance (dynamic)
    policy_pack_refs TEXT[] DEFAULT '{}', -- References to policy packs
    compliance_frameworks TEXT[] DEFAULT '{}', -- Dynamic compliance framework list
    
    -- Template source and loading
    template_source VARCHAR(100) NOT NULL DEFAULT 'yaml', -- 'yaml', 'json', 'generated'
    source_path TEXT, -- Path to YAML/JSON file or generator class
    loader_class TEXT, -- Python class for dynamic loading
    
    tenant_id INTEGER REFERENCES tenants(id),
    created_by_user_id INTEGER REFERENCES users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE(capability_id, industry_code, template_category)
);

-- Task 19.1.37: Tenant-specific version pins (extends existing capability versioning)
CREATE TABLE IF NOT EXISTS rba_template_tenant_pins (
    pin_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    capability_id UUID NOT NULL REFERENCES dsl_capability_registry(capability_id),
    pinned_version VARCHAR(50) NOT NULL,
    pin_reason TEXT,
    pinned_by_user_id INTEGER REFERENCES users(user_id),
    pinned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ, -- Optional expiry
    
    UNIQUE(tenant_id, capability_id) -- One pin per tenant per capability
);

-- Task 19.1.36: Promotion manifests (signed) - extends existing capability promotions
CREATE TABLE IF NOT EXISTS rba_template_promotions (
    promotion_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    capability_id UUID NOT NULL REFERENCES dsl_capability_registry(capability_id),
    from_state VARCHAR(50) NOT NULL,
    to_state VARCHAR(50) NOT NULL,
    promotion_manifest JSONB NOT NULL, -- Immutable promotion proof
    manifest_hash VARCHAR(64) NOT NULL, -- SHA256 hash
    digital_signature VARCHAR(500), -- Cryptographic signature
    evidence_pack_id UUID, -- Link to evidence service
    
    promoted_by_user_id INTEGER REFERENCES users(user_id),
    promoted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Audit trail
    approval_chain JSONB DEFAULT '[]', -- Array of approver user_ids
    approval_metadata JSONB DEFAULT '{}'
);

-- Task 19.1.12, 19.1.16, 19.1.19, 19.1.23, 19.1.27, 19.1.31, 19.1.34: Evidence packs for templates
CREATE TABLE IF NOT EXISTS rba_template_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    capability_id UUID NOT NULL REFERENCES dsl_capability_registry(capability_id),
    evidence_type VARCHAR(100) NOT NULL, -- 'validation', 'execution', 'compliance_test', 'promotion'
    evidence_data JSONB NOT NULL,
    evidence_hash VARCHAR(64) NOT NULL, -- SHA256 for immutability
    
    -- Compliance and audit
    compliance_frameworks TEXT[] DEFAULT '{}',
    regulator_visible BOOLEAN DEFAULT TRUE,
    
    tenant_id INTEGER REFERENCES tenants(id),
    created_by_user_id INTEGER REFERENCES users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_rba_lifecycle_capability ON rba_template_lifecycle(capability_id);
CREATE INDEX IF NOT EXISTS idx_rba_lifecycle_state ON rba_template_lifecycle(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_rba_lifecycle_tenant ON rba_template_lifecycle(tenant_id);

CREATE INDEX IF NOT EXISTS idx_rba_contracts_capability ON rba_template_contracts(capability_id);
CREATE INDEX IF NOT EXISTS idx_rba_contracts_type ON rba_template_contracts(contract_type);
CREATE INDEX IF NOT EXISTS idx_rba_contracts_tenant ON rba_template_contracts(tenant_id);

CREATE INDEX IF NOT EXISTS idx_rba_industry_templates_industry ON rba_industry_templates(industry_code);
CREATE INDEX IF NOT EXISTS idx_rba_industry_templates_category ON rba_industry_templates(template_category);
CREATE INDEX IF NOT EXISTS idx_rba_industry_templates_capability ON rba_industry_templates(capability_id);

CREATE INDEX IF NOT EXISTS idx_rba_tenant_pins_tenant ON rba_template_tenant_pins(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rba_promotions_capability ON rba_template_promotions(capability_id);
CREATE INDEX IF NOT EXISTS idx_rba_evidence_capability ON rba_template_evidence(capability_id);
CREATE INDEX IF NOT EXISTS idx_rba_evidence_type ON rba_template_evidence(evidence_type);

-- Task 19.1.4: RBAC/ABAC on registry tables (RLS policies)
ALTER TABLE rba_template_lifecycle ENABLE ROW LEVEL SECURITY;
ALTER TABLE rba_template_contracts ENABLE ROW LEVEL SECURITY;
ALTER TABLE rba_industry_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE rba_template_tenant_pins ENABLE ROW LEVEL SECURITY;
ALTER TABLE rba_template_promotions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rba_template_evidence ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only access templates for their tenant
CREATE POLICY rba_template_lifecycle_tenant_isolation ON rba_template_lifecycle
    FOR ALL TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY rba_template_contracts_tenant_isolation ON rba_template_contracts
    FOR ALL TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY rba_industry_templates_tenant_isolation ON rba_industry_templates
    FOR ALL TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY rba_template_pins_tenant_isolation ON rba_template_tenant_pins
    FOR ALL TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

CREATE POLICY rba_template_promotions_tenant_isolation ON rba_template_promotions
    FOR ALL TO authenticated_users
    USING (
        capability_id IN (
            SELECT capability_id FROM rba_template_lifecycle 
            WHERE tenant_id = current_setting('app.current_tenant_id')::INTEGER
        )
    );

CREATE POLICY rba_template_evidence_tenant_isolation ON rba_template_evidence
    FOR ALL TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- Additional RLS for regulators (read-only access to evidence)
CREATE POLICY rba_template_evidence_regulator_access ON rba_template_evidence
    FOR SELECT TO regulator_users
    USING (regulator_visible = TRUE);