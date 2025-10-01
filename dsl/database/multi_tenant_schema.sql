-- Multi-Tenant Database Schema Implementation
-- Chapter 9.4: Multi-tenant enforcement validation
-- Tasks 9.4.1-9.4.42 implementation

-- ============================================================================
-- SECTION 1: CORE TENANT MANAGEMENT (Tasks 9.4.1-9.4.8)
-- ============================================================================

-- Task 9.4.2: tenant_id as mandatory governance field
-- Task 9.4.8: Build tenant metadata service
CREATE TABLE IF NOT EXISTS tenant_metadata (
    tenant_id INTEGER PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    tenant_type VARCHAR(50) NOT NULL DEFAULT 'standard', -- standard, enterprise, regulated
    industry_code VARCHAR(10) NOT NULL, -- SaaS, BANK, INSUR, ECOMM, FS, IT
    region_code VARCHAR(10) NOT NULL, -- US, EU, IN, APAC
    compliance_requirements JSONB NOT NULL DEFAULT '[]', -- ["SOX", "GDPR", "HIPAA", "RBI"]
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, suspended, offboarding
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- Enable RLS on tenant_metadata
ALTER TABLE tenant_metadata ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to own tenant data
CREATE POLICY tenant_metadata_rls_policy ON tenant_metadata
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 2: DSL WORKFLOWS WITH MULTI-TENANT ENFORCEMENT (Tasks 9.4.4-9.4.6)
-- ============================================================================

-- Task 9.4.4: Schema partitioning per tenant
-- Enhanced DSL workflows table with tenant isolation
CREATE TABLE IF NOT EXISTS dsl_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    workflow_name VARCHAR(255) NOT NULL,
    workflow_type VARCHAR(50) NOT NULL, -- RBA, RBIA, AALA
    industry_overlay VARCHAR(10), -- SaaS, BANK, INSUR
    workflow_definition JSONB NOT NULL,
    policy_pack_id UUID,
    evidence_pack_id UUID,
    status VARCHAR(20) NOT NULL DEFAULT 'draft', -- draft, published, promoted, retired
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    blob_storage_url TEXT,
    trust_score DECIMAL(5,2) DEFAULT 1.0,
    execution_count INTEGER DEFAULT 0,
    last_executed_at TIMESTAMPTZ,
    
    -- Multi-tenant constraints
    CONSTRAINT fk_dsl_workflows_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    CONSTRAINT unique_workflow_per_tenant UNIQUE (tenant_id, workflow_name, version)
);

-- Enable RLS on dsl_workflows
ALTER TABLE dsl_workflows ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to workflows within the same tenant
CREATE POLICY dsl_workflows_rls_policy ON dsl_workflows
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 3: POLICY PACKS WITH COMPLIANCE FRAMEWORKS (Tasks 16.1-16.3)
-- ============================================================================

-- Task 16.1: Policy pack management (SOX, GDPR, HIPAA, RBI overlays)
CREATE TABLE IF NOT EXISTS dsl_policy_packs (
    policy_pack_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    pack_name VARCHAR(255) NOT NULL,
    pack_type VARCHAR(50) NOT NULL, -- SOX, GDPR, HIPAA, RBI, DPDP, NAIC
    industry_code VARCHAR(10) NOT NULL, -- SaaS, BANK, INSUR
    region_code VARCHAR(10) NOT NULL, -- US, EU, IN
    policy_rules JSONB NOT NULL,
    enforcement_level VARCHAR(20) NOT NULL DEFAULT 'strict', -- strict, advisory, disabled
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    
    -- Multi-tenant constraints
    CONSTRAINT fk_policy_packs_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    CONSTRAINT unique_policy_pack_per_tenant UNIQUE (tenant_id, pack_name, version)
);

-- Enable RLS on dsl_policy_packs
ALTER TABLE dsl_policy_packs ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to policy packs within the same tenant
CREATE POLICY dsl_policy_packs_rls_policy ON dsl_policy_packs
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 4: EXECUTION TRACES WITH TENANT ISOLATION (Tasks 8.1-8.2)
-- ============================================================================

-- Enhanced execution traces with multi-tenant enforcement
CREATE TABLE IF NOT EXISTS dsl_execution_traces (
    trace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    workflow_id UUID NOT NULL,
    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    agent_name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(50) NOT NULL, -- RBA, RBIA, AALA
    user_id INTEGER NOT NULL,
    execution_source VARCHAR(50) DEFAULT 'api', -- api, scheduler, webhook
    
    -- Input/Output data
    inputs JSONB,
    outputs JSONB,
    configuration_used JSONB,
    
    -- Governance metadata
    governance_metadata JSONB DEFAULT '{}',
    policy_pack_id UUID,
    evidence_pack_id UUID,
    override_count INTEGER DEFAULT 0,
    
    -- Performance metrics
    execution_time_ms INTEGER DEFAULT 0,
    memory_usage_mb DECIMAL(10,2) DEFAULT 0,
    trust_score DECIMAL(5,2) DEFAULT 1.0,
    
    -- Business impact
    opportunities_processed INTEGER DEFAULT 0,
    flagged_opportunities INTEGER DEFAULT 0,
    pipeline_value_analyzed DECIMAL(15,2) DEFAULT 0,
    
    -- Audit trail
    entities_affected TEXT[],
    relationships_created TEXT[],
    
    -- Timestamps
    execution_start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    execution_end_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Multi-tenant constraints
    CONSTRAINT fk_execution_traces_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    CONSTRAINT fk_execution_traces_workflow FOREIGN KEY (workflow_id) REFERENCES dsl_workflows(workflow_id)
);

-- Enable RLS on dsl_execution_traces
ALTER TABLE dsl_execution_traces ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to execution traces within the same tenant
CREATE POLICY dsl_execution_traces_rls_policy ON dsl_execution_traces
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 5: OVERRIDE LEDGER (Tasks 16.2)
-- ============================================================================

-- Task 16.2: Override ledger implementation
CREATE TABLE IF NOT EXISTS dsl_override_ledger (
    override_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    workflow_id UUID,
    execution_trace_id UUID,
    user_id INTEGER NOT NULL,
    
    -- Override details
    override_type VARCHAR(50) NOT NULL, -- policy_bypass, manual_approval, emergency_override
    component_affected VARCHAR(100) NOT NULL, -- workflow_name, policy_name, agent_name
    original_value JSONB,
    overridden_value JSONB,
    override_reason TEXT NOT NULL,
    
    -- Approval workflow
    approval_required BOOLEAN DEFAULT true,
    approved_by_user_id INTEGER,
    approved_at TIMESTAMPTZ,
    approval_justification TEXT,
    
    -- Risk assessment
    risk_level VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
    business_impact TEXT,
    
    -- Audit trail
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active', -- active, expired, revoked
    
    -- Governance
    policy_pack_id UUID,
    evidence_pack_id UUID,
    compliance_flags JSONB DEFAULT '{}',
    
    -- Multi-tenant constraints
    CONSTRAINT fk_override_ledger_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    CONSTRAINT fk_override_ledger_workflow FOREIGN KEY (workflow_id) REFERENCES dsl_workflows(workflow_id),
    CONSTRAINT fk_override_ledger_trace FOREIGN KEY (execution_trace_id) REFERENCES dsl_execution_traces(trace_id)
);

-- Enable RLS on dsl_override_ledger
ALTER TABLE dsl_override_ledger ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to override ledger within the same tenant
CREATE POLICY dsl_override_ledger_rls_policy ON dsl_override_ledger
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 6: EVIDENCE PACKS (Tasks 16.4)
-- ============================================================================

-- Task 16.4: Evidence pack service
CREATE TABLE IF NOT EXISTS dsl_evidence_packs (
    evidence_pack_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    pack_name VARCHAR(255) NOT NULL,
    pack_type VARCHAR(50) NOT NULL, -- execution_evidence, compliance_evidence, audit_evidence
    
    -- Evidence content
    evidence_data JSONB NOT NULL,
    evidence_hash VARCHAR(64) NOT NULL, -- SHA-256 hash for tamper detection
    digital_signature TEXT, -- Optional cryptographic signature
    
    -- Associated entities
    workflow_id UUID,
    execution_trace_id UUID,
    policy_pack_id UUID,
    override_id UUID,
    
    -- Compliance metadata
    compliance_framework VARCHAR(50), -- SOX, GDPR, HIPAA, RBI
    retention_period_days INTEGER DEFAULT 2555, -- 7 years default
    immutable BOOLEAN DEFAULT true,
    
    -- Storage details
    blob_storage_url TEXT,
    file_format VARCHAR(20) DEFAULT 'json', -- json, pdf, zip
    file_size_bytes BIGINT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    
    -- Multi-tenant constraints
    CONSTRAINT fk_evidence_packs_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    CONSTRAINT fk_evidence_packs_workflow FOREIGN KEY (workflow_id) REFERENCES dsl_workflows(workflow_id),
    CONSTRAINT fk_evidence_packs_trace FOREIGN KEY (execution_trace_id) REFERENCES dsl_execution_traces(trace_id),
    CONSTRAINT fk_evidence_packs_policy FOREIGN KEY (policy_pack_id) REFERENCES dsl_policy_packs(policy_pack_id),
    CONSTRAINT fk_evidence_packs_override FOREIGN KEY (override_id) REFERENCES dsl_override_ledger(override_id)
);

-- Enable RLS on dsl_evidence_packs
ALTER TABLE dsl_evidence_packs ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to evidence packs within the same tenant
CREATE POLICY dsl_evidence_packs_rls_policy ON dsl_evidence_packs
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 7: WORKFLOW TEMPLATES (Tasks 14.1)
-- ============================================================================

-- Enhanced workflow templates with multi-tenant support
CREATE TABLE IF NOT EXISTS dsl_workflow_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    template_name VARCHAR(255) NOT NULL,
    template_type VARCHAR(50) NOT NULL, -- RBA, RBIA, AALA
    industry_overlay VARCHAR(10) NOT NULL, -- SaaS, BANK, INSUR
    category VARCHAR(100) NOT NULL, -- data_quality, risk_analysis, velocity_analysis
    
    -- Template definition
    template_definition JSONB NOT NULL,
    ui_schema JSONB, -- For drag-and-drop designer
    parameter_schema JSONB, -- Configuration parameters
    
    -- Metadata
    description TEXT,
    tags TEXT[],
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    
    -- Usage stats
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    
    -- Governance
    policy_requirements JSONB DEFAULT '[]',
    compliance_frameworks TEXT[], -- ["SOX", "GDPR"]
    trust_score DECIMAL(5,2) DEFAULT 1.0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    
    -- Multi-tenant constraints
    CONSTRAINT fk_workflow_templates_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    CONSTRAINT unique_template_per_tenant UNIQUE (tenant_id, template_name, version)
);

-- Enable RLS on dsl_workflow_templates
ALTER TABLE dsl_workflow_templates ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to workflow templates within the same tenant
CREATE POLICY dsl_workflow_templates_rls_policy ON dsl_workflow_templates
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 8: CAPABILITY REGISTRY (Tasks 15.3)
-- ============================================================================

-- Enhanced capability registry with multi-tenant support
CREATE TABLE IF NOT EXISTS dsl_capability_registry (
    capability_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    capability_name VARCHAR(255) NOT NULL,
    capability_type VARCHAR(50) NOT NULL, -- query, decision, ml_decision, agent_call, notify, governance
    
    -- Capability definition
    operator_definition JSONB NOT NULL,
    input_schema JSONB NOT NULL,
    output_schema JSONB NOT NULL,
    
    -- Industry and compliance
    industry_overlays TEXT[], -- ["SaaS", "BANK", "INSUR"]
    compliance_requirements TEXT[], -- ["SOX", "GDPR", "HIPAA"]
    
    -- Metadata
    description TEXT,
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    
    -- Performance characteristics
    avg_execution_time_ms INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 100.0,
    trust_score DECIMAL(5,2) DEFAULT 1.0,
    
    -- Usage tracking
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    
    -- Multi-tenant constraints
    CONSTRAINT fk_capability_registry_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    CONSTRAINT unique_schema_capability_per_tenant UNIQUE (tenant_id, capability_name, version)
);

-- Enable RLS on dsl_capability_registry
ALTER TABLE dsl_capability_registry ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to capabilities within the same tenant
CREATE POLICY dsl_capability_registry_rls_policy ON dsl_capability_registry
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 9: KNOWLEDGE GRAPH TRACES (Tasks 8.1-8.2)
-- ============================================================================

-- Enhanced Knowledge Graph execution traces (already referenced above)
CREATE TABLE IF NOT EXISTS kg_execution_traces (
    trace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER NOT NULL,
    workflow_id VARCHAR(255) NOT NULL,
    run_id UUID NOT NULL,
    
    -- Trace data
    inputs JSONB,
    outputs JSONB,
    governance_metadata JSONB DEFAULT '{}',
    
    -- Performance metrics
    execution_time_ms INTEGER DEFAULT 0,
    trust_score DECIMAL(5,2) DEFAULT 1.0,
    entities_affected TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Multi-tenant constraints
    CONSTRAINT fk_kg_execution_traces_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Enable RLS on kg_execution_traces
ALTER TABLE kg_execution_traces ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Only allow access to KG traces within the same tenant
CREATE POLICY kg_execution_traces_rls_policy ON kg_execution_traces
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 10: INDEXES FOR PERFORMANCE
-- ============================================================================

-- Tenant-specific indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_dsl_workflows_tenant_status ON dsl_workflows(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_dsl_workflows_tenant_type ON dsl_workflows(tenant_id, workflow_type);
CREATE INDEX IF NOT EXISTS idx_dsl_policy_packs_tenant_type ON dsl_policy_packs(tenant_id, pack_type);
CREATE INDEX IF NOT EXISTS idx_dsl_execution_traces_tenant_workflow ON dsl_execution_traces(tenant_id, workflow_id);
CREATE INDEX IF NOT EXISTS idx_dsl_execution_traces_tenant_created ON dsl_execution_traces(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dsl_override_ledger_tenant_created ON dsl_override_ledger(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dsl_evidence_packs_tenant_type ON dsl_evidence_packs(tenant_id, pack_type);
CREATE INDEX IF NOT EXISTS idx_dsl_workflow_templates_tenant_industry ON dsl_workflow_templates(tenant_id, industry_overlay);
CREATE INDEX IF NOT EXISTS idx_dsl_capability_registry_tenant_type ON dsl_capability_registry(tenant_id, capability_type);
CREATE INDEX IF NOT EXISTS idx_kg_execution_traces_tenant_created ON kg_execution_traces(tenant_id, created_at DESC);

-- ============================================================================
-- SECTION 11: INITIAL DATA SEEDING
-- ============================================================================

-- Seed default tenant (existing tenant 1300)
INSERT INTO tenant_metadata (tenant_id, tenant_name, tenant_type, industry_code, region_code, compliance_requirements)
VALUES (1300, 'Default Tenant', 'enterprise', 'SaaS', 'US', '["SOX", "GDPR"]')
ON CONFLICT (tenant_id) DO UPDATE SET
    updated_at = NOW(),
    tenant_type = EXCLUDED.tenant_type,
    industry_code = EXCLUDED.industry_code,
    region_code = EXCLUDED.region_code,
    compliance_requirements = EXCLUDED.compliance_requirements;

-- Seed SaaS policy pack for tenant 1300
INSERT INTO dsl_policy_packs (tenant_id, pack_name, pack_type, industry_code, region_code, policy_rules, created_by_user_id)
VALUES (
    1300,
    'SaaS SOX Compliance Pack',
    'SOX',
    'SaaS',
    'US',
    '{
        "financial_controls": {
            "revenue_recognition": {"enabled": true, "enforcement": "strict"},
            "deal_approval_thresholds": {"high_value": 250000, "mega_deal": 1000000},
            "segregation_of_duties": {"enabled": true, "maker_checker": true}
        },
        "audit_requirements": {
            "execution_logging": {"enabled": true, "retention_days": 2555},
            "override_justification": {"required": true, "approval_required": true},
            "evidence_generation": {"enabled": true, "immutable": true}
        }
    }',
    1319
) ON CONFLICT (tenant_id, pack_name, version) DO NOTHING;

-- Seed GDPR policy pack for tenant 1300
INSERT INTO dsl_policy_packs (tenant_id, pack_name, pack_type, industry_code, region_code, policy_rules, created_by_user_id)
VALUES (
    1300,
    'SaaS GDPR Privacy Pack',
    'GDPR',
    'SaaS',
    'EU',
    '{
        "data_protection": {
            "consent_management": {"enabled": true, "explicit_consent": true},
            "right_to_erasure": {"enabled": true, "retention_override": false},
            "data_minimization": {"enabled": true, "purpose_limitation": true}
        },
        "privacy_controls": {
            "pii_classification": {"enabled": true, "auto_detection": true},
            "cross_border_transfer": {"restricted": true, "adequacy_required": true},
            "breach_notification": {"enabled": true, "notification_hours": 72}
        }
    }',
    1319
) ON CONFLICT (tenant_id, pack_name, version) DO NOTHING;

-- ============================================================================
-- SECTION 12: FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to automatically set updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
CREATE TRIGGER update_tenant_metadata_updated_at BEFORE UPDATE ON tenant_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_workflows_updated_at BEFORE UPDATE ON dsl_workflows FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_policy_packs_updated_at BEFORE UPDATE ON dsl_policy_packs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_workflow_templates_updated_at BEFORE UPDATE ON dsl_workflow_templates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_capability_registry_updated_at BEFORE UPDATE ON dsl_capability_registry FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to validate tenant context
CREATE OR REPLACE FUNCTION validate_tenant_context()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure tenant_id is set in session context
    IF current_setting('app.current_tenant_id', true) IS NULL OR 
       current_setting('app.current_tenant_id', true) = '' THEN
        RAISE EXCEPTION 'Tenant context not set. All operations must specify tenant_id.';
    END IF;
    
    -- Validate tenant exists and is active
    IF NOT EXISTS (
        SELECT 1 FROM tenant_metadata 
        WHERE tenant_id = NEW.tenant_id 
        AND status = 'active'
    ) THEN
        RAISE EXCEPTION 'Invalid or inactive tenant_id: %', NEW.tenant_id;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply tenant validation triggers
CREATE TRIGGER validate_tenant_dsl_workflows BEFORE INSERT OR UPDATE ON dsl_workflows FOR EACH ROW EXECUTE FUNCTION validate_tenant_context();
CREATE TRIGGER validate_tenant_dsl_policy_packs BEFORE INSERT OR UPDATE ON dsl_policy_packs FOR EACH ROW EXECUTE FUNCTION validate_tenant_context();
CREATE TRIGGER validate_tenant_dsl_execution_traces BEFORE INSERT OR UPDATE ON dsl_execution_traces FOR EACH ROW EXECUTE FUNCTION validate_tenant_context();
CREATE TRIGGER validate_tenant_dsl_override_ledger BEFORE INSERT OR UPDATE ON dsl_override_ledger FOR EACH ROW EXECUTE FUNCTION validate_tenant_context();
CREATE TRIGGER validate_tenant_dsl_evidence_packs BEFORE INSERT OR UPDATE ON dsl_evidence_packs FOR EACH ROW EXECUTE FUNCTION validate_tenant_context();

-- ============================================================================
-- SECTION 13: VIEWS FOR EASY ACCESS
-- ============================================================================

-- Tenant summary view
CREATE OR REPLACE VIEW tenant_summary AS
SELECT 
    tm.tenant_id,
    tm.tenant_name,
    tm.industry_code,
    tm.region_code,
    tm.compliance_requirements,
    tm.status,
    COUNT(DISTINCT dw.workflow_id) as total_workflows,
    COUNT(DISTINCT dpp.policy_pack_id) as total_policy_packs,
    COUNT(DISTINCT det.trace_id) as total_executions,
    MAX(det.created_at) as last_execution
FROM tenant_metadata tm
LEFT JOIN dsl_workflows dw ON tm.tenant_id = dw.tenant_id
LEFT JOIN dsl_policy_packs dpp ON tm.tenant_id = dpp.tenant_id
LEFT JOIN dsl_execution_traces det ON tm.tenant_id = det.tenant_id
GROUP BY tm.tenant_id, tm.tenant_name, tm.industry_code, tm.region_code, tm.compliance_requirements, tm.status;

-- Compliance dashboard view
CREATE OR REPLACE VIEW compliance_dashboard AS
SELECT 
    det.tenant_id,
    det.workflow_id,
    dw.workflow_name,
    det.agent_name,
    det.policy_pack_id,
    dpp.pack_type as compliance_framework,
    det.trust_score,
    det.override_count,
    CASE 
        WHEN det.override_count = 0 THEN 'COMPLIANT'
        WHEN det.override_count <= 2 THEN 'MINOR_ISSUES'
        ELSE 'NON_COMPLIANT'
    END as compliance_status,
    det.execution_start_time,
    det.created_at
FROM dsl_execution_traces det
JOIN dsl_workflows dw ON det.workflow_id = dw.workflow_id
LEFT JOIN dsl_policy_packs dpp ON det.policy_pack_id = dpp.policy_pack_id
WHERE det.tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1);

-- ============================================================================
-- SECTION 14: GRANT PERMISSIONS
-- ============================================================================

-- Grant appropriate permissions (assuming application role exists)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO application_role;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO application_role;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO application_role;

-- ============================================================================
-- SCHEMA VALIDATION
-- ============================================================================

-- Validate schema creation
DO $$
BEGIN
    -- Check if all critical tables exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tenant_metadata') THEN
        RAISE EXCEPTION 'tenant_metadata table not created';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'dsl_workflows') THEN
        RAISE EXCEPTION 'dsl_workflows table not created';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'dsl_policy_packs') THEN
        RAISE EXCEPTION 'dsl_policy_packs table not created';
    END IF;
    
    RAISE NOTICE '✅ Multi-tenant database schema created successfully!';
    RAISE NOTICE '📊 Tables created: tenant_metadata, dsl_workflows, dsl_policy_packs, dsl_execution_traces, dsl_override_ledger, dsl_evidence_packs, dsl_workflow_templates, dsl_capability_registry, kg_execution_traces';
    RAISE NOTICE '🔒 RLS policies enabled on all tables';
    RAISE NOTICE '📈 Performance indexes created';
    RAISE NOTICE '🎯 Default tenant (1300) seeded with SOX and GDPR policy packs';
END $$;
