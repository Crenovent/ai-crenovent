-- RBA Core Database Schemas
-- Implements foundational database structure for RBA system
-- Includes traces, evidence, overrides, workflows, and governance tables
-- All tables include tenant isolation via RLS (Row Level Security)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- TENANT METADATA TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS tenant_metadata (
    tenant_id INTEGER PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    tenant_type VARCHAR(50) NOT NULL DEFAULT 'standard',
    industry_code VARCHAR(10) NOT NULL,
    region_code VARCHAR(10) NOT NULL,
    compliance_requirements JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT tenant_status_check CHECK (status IN ('active', 'suspended', 'terminated')),
    CONSTRAINT tenant_type_check CHECK (tenant_type IN ('standard', 'enterprise', 'trial'))
);

-- Enable RLS on tenant_metadata
ALTER TABLE tenant_metadata ENABLE ROW LEVEL SECURITY;

-- RLS Policy for tenant_metadata
CREATE POLICY tenant_metadata_rls_policy ON tenant_metadata
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- DSL WORKFLOWS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS dsl_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    workflow_name VARCHAR(255) NOT NULL,
    workflow_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    workflow_description TEXT,
    workflow_status VARCHAR(20) NOT NULL DEFAULT 'draft',
    
    -- DSL Content
    dsl_content JSONB NOT NULL,
    compiled_plan JSONB,
    plan_hash VARCHAR(64) NOT NULL,
    
    -- Governance
    policy_pack_id VARCHAR(255) NOT NULL,
    compliance_tags JSONB DEFAULT '[]',
    industry_overlay VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    approved_by_user_id INTEGER,
    approved_at TIMESTAMPTZ,
    
    -- Storage
    blob_storage_url TEXT,
    
    CONSTRAINT workflow_status_check CHECK (workflow_status IN ('draft', 'review', 'approved', 'published', 'deprecated', 'archived')),
    CONSTRAINT unique_workflow_name_version UNIQUE (tenant_id, workflow_name, workflow_version),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Enable RLS on dsl_workflows
ALTER TABLE dsl_workflows ENABLE ROW LEVEL SECURITY;

-- RLS Policy for dsl_workflows
CREATE POLICY dsl_workflows_rls_policy ON dsl_workflows
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- DSL POLICY PACKS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS dsl_policy_packs (
    policy_pack_id VARCHAR(255) PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    policy_name VARCHAR(255) NOT NULL,
    policy_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    policy_description TEXT,
    
    -- Policy Content
    policy_rules JSONB NOT NULL,
    industry_overlay VARCHAR(50),
    compliance_framework VARCHAR(100),
    
    -- Governance
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    effective_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expiration_date TIMESTAMPTZ,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    
    CONSTRAINT policy_status_check CHECK (status IN ('draft', 'active', 'deprecated', 'expired')),
    CONSTRAINT unique_policy_name_version UNIQUE (tenant_id, policy_name, policy_version),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Enable RLS on dsl_policy_packs
ALTER TABLE dsl_policy_packs ENABLE ROW LEVEL SECURITY;

-- RLS Policy for dsl_policy_packs
CREATE POLICY dsl_policy_packs_rls_policy ON dsl_policy_packs
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- DSL EXECUTION TRACES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS dsl_execution_traces (
    trace_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL,
    workflow_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Execution Details
    execution_status VARCHAR(20) NOT NULL DEFAULT 'running',
    plan_hash VARCHAR(64) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Context
    user_id INTEGER NOT NULL,
    execution_context JSONB DEFAULT '{}',
    
    -- Trace Data
    trace_data JSONB NOT NULL,
    step_count INTEGER DEFAULT 0,
    error_message TEXT,
    
    -- Governance
    policy_pack_id VARCHAR(255),
    compliance_tags JSONB DEFAULT '[]',
    evidence_generated BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT execution_status_check CHECK (execution_status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    FOREIGN KEY (workflow_id) REFERENCES dsl_workflows(workflow_id),
    FOREIGN KEY (policy_pack_id) REFERENCES dsl_policy_packs(policy_pack_id)
);

-- Enable RLS on dsl_execution_traces
ALTER TABLE dsl_execution_traces ENABLE ROW LEVEL SECURITY;

-- RLS Policy for dsl_execution_traces
CREATE POLICY dsl_execution_traces_rls_policy ON dsl_execution_traces
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- DSL EVIDENCE PACKS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS dsl_evidence_packs (
    evidence_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL,
    node_id VARCHAR(255),
    tenant_id INTEGER NOT NULL,
    
    -- Evidence Content
    evidence_type VARCHAR(50) NOT NULL,
    evidence_data JSONB NOT NULL,
    evidence_hash VARCHAR(64) NOT NULL,
    
    -- Digital Signature
    signature_algorithm VARCHAR(50) DEFAULT 'SHA256withRSA',
    digital_signature TEXT,
    signature_timestamp TIMESTAMPTZ,
    
    -- Governance
    policy_pack_id VARCHAR(255),
    compliance_framework VARCHAR(100),
    retention_period_days INTEGER DEFAULT 2555, -- 7 years default
    
    -- Blockchain Anchoring (Optional)
    blockchain_anchor_hash VARCHAR(64),
    blockchain_transaction_id VARCHAR(255),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    immutable_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT evidence_type_check CHECK (evidence_type IN ('workflow_execution', 'step_execution', 'policy_enforcement', 'override_approval', 'audit_event')),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    FOREIGN KEY (execution_id) REFERENCES dsl_execution_traces(execution_id),
    FOREIGN KEY (policy_pack_id) REFERENCES dsl_policy_packs(policy_pack_id)
);

-- Enable RLS on dsl_evidence_packs
ALTER TABLE dsl_evidence_packs ENABLE ROW LEVEL SECURITY;

-- RLS Policy for dsl_evidence_packs
CREATE POLICY dsl_evidence_packs_rls_policy ON dsl_evidence_packs
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- DSL OVERRIDE LEDGER TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS dsl_override_ledger (
    override_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL,
    node_id VARCHAR(255),
    tenant_id INTEGER NOT NULL,
    
    -- Override Details
    override_type VARCHAR(50) NOT NULL,
    override_reason TEXT NOT NULL,
    original_value JSONB,
    override_value JSONB,
    
    -- Approval Workflow
    requested_by_user_id INTEGER NOT NULL,
    approved_by_user_id INTEGER,
    approval_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    
    -- Segregation of Duties
    sod_validation JSONB,
    conflict_of_interest_check BOOLEAN DEFAULT FALSE,
    
    -- Governance
    policy_pack_id VARCHAR(255),
    justification_category VARCHAR(100),
    business_impact_assessment JSONB,
    
    -- Digital Signature
    approval_signature TEXT,
    signature_timestamp TIMESTAMPTZ,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    
    CONSTRAINT override_type_check CHECK (override_type IN ('policy_exception', 'data_override', 'workflow_bypass', 'manual_intervention')),
    CONSTRAINT approval_status_check CHECK (approval_status IN ('pending', 'approved', 'rejected', 'expired', 'revoked')),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    FOREIGN KEY (execution_id) REFERENCES dsl_execution_traces(execution_id),
    FOREIGN KEY (policy_pack_id) REFERENCES dsl_policy_packs(policy_pack_id)
);

-- Enable RLS on dsl_override_ledger
ALTER TABLE dsl_override_ledger ENABLE ROW LEVEL SECURITY;

-- RLS Policy for dsl_override_ledger
CREATE POLICY dsl_override_ledger_rls_policy ON dsl_override_ledger
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- DSL WORKFLOW TEMPLATES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS dsl_workflow_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    template_name VARCHAR(255) NOT NULL,
    template_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    template_description TEXT,
    
    -- Template Content
    template_dsl JSONB NOT NULL,
    parameter_schema JSONB DEFAULT '{}',
    
    -- Classification
    industry_vertical VARCHAR(50),
    use_case_category VARCHAR(100),
    complexity_level VARCHAR(20) DEFAULT 'medium',
    
    -- Governance
    policy_requirements JSONB DEFAULT '[]',
    compliance_tags JSONB DEFAULT '[]',
    
    -- Publishing
    is_public BOOLEAN DEFAULT FALSE,
    is_verified BOOLEAN DEFAULT FALSE,
    download_count INTEGER DEFAULT 0,
    rating_average DECIMAL(3,2) DEFAULT 0.0,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    
    CONSTRAINT complexity_level_check CHECK (complexity_level IN ('simple', 'medium', 'complex', 'expert')),
    CONSTRAINT rating_check CHECK (rating_average >= 0.0 AND rating_average <= 5.0),
    CONSTRAINT unique_template_name_version UNIQUE (tenant_id, template_name, template_version),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Enable RLS on dsl_workflow_templates
ALTER TABLE dsl_workflow_templates ENABLE ROW LEVEL SECURITY;

-- RLS Policy for dsl_workflow_templates
CREATE POLICY dsl_workflow_templates_rls_policy ON dsl_workflow_templates
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1) OR is_public = TRUE);

-- ============================================================================
-- DSL CAPABILITY REGISTRY TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS dsl_capability_registry (
    capability_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    capability_name VARCHAR(255) NOT NULL,
    capability_type VARCHAR(50) NOT NULL,
    capability_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    
    -- Capability Definition
    operator_definition JSONB NOT NULL,
    input_schema JSONB NOT NULL DEFAULT '{}',
    output_schema JSONB NOT NULL DEFAULT '{}',
    
    -- Performance Metrics
    avg_execution_time_ms INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 1.0,
    usage_count INTEGER DEFAULT 0,
    
    -- Governance
    policy_requirements JSONB DEFAULT '[]',
    compliance_tags JSONB DEFAULT '[]',
    trust_score DECIMAL(3,2) DEFAULT 1.0,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    created_by_user_id INTEGER,
    
    CONSTRAINT capability_type_check CHECK (capability_type IN ('query', 'decision', 'ml_decision', 'agent_call', 'notify', 'governance')),
    CONSTRAINT success_rate_check CHECK (success_rate >= 0 AND success_rate <= 1),
    CONSTRAINT trust_score_check CHECK (trust_score >= 0.0 AND trust_score <= 1.0),
    CONSTRAINT unique_capability_name_version UNIQUE (tenant_id, capability_name, capability_version),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Enable RLS on dsl_capability_registry
ALTER TABLE dsl_capability_registry ENABLE ROW LEVEL SECURITY;

-- RLS Policy for dsl_capability_registry
CREATE POLICY dsl_capability_registry_rls_policy ON dsl_capability_registry
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- KNOWLEDGE GRAPH EXECUTION TRACES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS kg_execution_traces (
    kg_trace_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Graph Data
    entities JSONB DEFAULT '[]',
    relationships JSONB DEFAULT '[]',
    
    -- Enrichment
    enrichment_status VARCHAR(20) DEFAULT 'pending',
    enrichment_timestamp TIMESTAMPTZ,
    
    -- Analytics
    trace_embedding VECTOR(1536), -- For similarity search (if pgvector available)
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT enrichment_status_check CHECK (enrichment_status IN ('pending', 'processing', 'completed', 'failed')),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    FOREIGN KEY (execution_id) REFERENCES dsl_execution_traces(execution_id)
);

-- Enable RLS on kg_execution_traces
ALTER TABLE kg_execution_traces ENABLE ROW LEVEL SECURITY;

-- RLS Policy for kg_execution_traces
CREATE POLICY kg_execution_traces_rls_policy ON kg_execution_traces
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- IDEMPOTENCY RECORDS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS idempotency_records (
    idempotency_key VARCHAR(64) PRIMARY KEY,
    execution_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    workflow_id UUID NOT NULL,
    
    -- Status
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    
    -- Data
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    CONSTRAINT idempotency_status_check CHECK (status IN ('new', 'duplicate', 'in_progress', 'completed', 'failed', 'expired')),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    FOREIGN KEY (workflow_id) REFERENCES dsl_workflows(workflow_id),
    FOREIGN KEY (execution_id) REFERENCES dsl_execution_traces(execution_id)
);

-- Enable RLS on idempotency_records
ALTER TABLE idempotency_records ENABLE ROW LEVEL SECURITY;

-- RLS Policy for idempotency_records
CREATE POLICY idempotency_records_rls_policy ON idempotency_records
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- DSL Workflows Indexes
CREATE INDEX IF NOT EXISTS idx_dsl_workflows_tenant_status ON dsl_workflows(tenant_id, workflow_status);
CREATE INDEX IF NOT EXISTS idx_dsl_workflows_name_version ON dsl_workflows(workflow_name, workflow_version);
CREATE INDEX IF NOT EXISTS idx_dsl_workflows_created_at ON dsl_workflows(created_at);
CREATE INDEX IF NOT EXISTS idx_dsl_workflows_plan_hash ON dsl_workflows(plan_hash);

-- DSL Execution Traces Indexes
CREATE INDEX IF NOT EXISTS idx_dsl_execution_traces_tenant_workflow ON dsl_execution_traces(tenant_id, workflow_id);
CREATE INDEX IF NOT EXISTS idx_dsl_execution_traces_status ON dsl_execution_traces(execution_status);
CREATE INDEX IF NOT EXISTS idx_dsl_execution_traces_started_at ON dsl_execution_traces(started_at);
CREATE INDEX IF NOT EXISTS idx_dsl_execution_traces_execution_id ON dsl_execution_traces(execution_id);

-- DSL Evidence Packs Indexes
CREATE INDEX IF NOT EXISTS idx_dsl_evidence_packs_tenant_execution ON dsl_evidence_packs(tenant_id, execution_id);
CREATE INDEX IF NOT EXISTS idx_dsl_evidence_packs_type ON dsl_evidence_packs(evidence_type);
CREATE INDEX IF NOT EXISTS idx_dsl_evidence_packs_created_at ON dsl_evidence_packs(created_at);
CREATE INDEX IF NOT EXISTS idx_dsl_evidence_packs_hash ON dsl_evidence_packs(evidence_hash);

-- DSL Override Ledger Indexes
CREATE INDEX IF NOT EXISTS idx_dsl_override_ledger_tenant_execution ON dsl_override_ledger(tenant_id, execution_id);
CREATE INDEX IF NOT EXISTS idx_dsl_override_ledger_status ON dsl_override_ledger(approval_status);
CREATE INDEX IF NOT EXISTS idx_dsl_override_ledger_requested_at ON dsl_override_ledger(requested_at);
CREATE INDEX IF NOT EXISTS idx_dsl_override_ledger_approved_by ON dsl_override_ledger(approved_by_user_id);

-- DSL Policy Packs Indexes
CREATE INDEX IF NOT EXISTS idx_dsl_policy_packs_tenant_status ON dsl_policy_packs(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_dsl_policy_packs_effective_date ON dsl_policy_packs(effective_date);
CREATE INDEX IF NOT EXISTS idx_dsl_policy_packs_compliance ON dsl_policy_packs(compliance_framework);

-- DSL Capability Registry Indexes
CREATE INDEX IF NOT EXISTS idx_dsl_capability_registry_tenant_type ON dsl_capability_registry(tenant_id, capability_type);
CREATE INDEX IF NOT EXISTS idx_dsl_capability_registry_usage ON dsl_capability_registry(usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_dsl_capability_registry_trust_score ON dsl_capability_registry(trust_score DESC);

-- Idempotency Records Indexes
CREATE INDEX IF NOT EXISTS idx_idempotency_records_tenant_workflow ON idempotency_records(tenant_id, workflow_id);
CREATE INDEX IF NOT EXISTS idx_idempotency_records_status_expires ON idempotency_records(status, expires_at);
CREATE INDEX IF NOT EXISTS idx_idempotency_records_created_at ON idempotency_records(created_at);

-- Knowledge Graph Indexes
CREATE INDEX IF NOT EXISTS idx_kg_execution_traces_tenant_execution ON kg_execution_traces(tenant_id, execution_id);
CREATE INDEX IF NOT EXISTS idx_kg_execution_traces_enrichment_status ON kg_execution_traces(enrichment_status);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMPS
-- ============================================================================

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_tenant_metadata_updated_at BEFORE UPDATE ON tenant_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_workflows_updated_at BEFORE UPDATE ON dsl_workflows FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_policy_packs_updated_at BEFORE UPDATE ON dsl_policy_packs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_execution_traces_updated_at BEFORE UPDATE ON dsl_execution_traces FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_override_ledger_updated_at BEFORE UPDATE ON dsl_override_ledger FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_workflow_templates_updated_at BEFORE UPDATE ON dsl_workflow_templates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dsl_capability_registry_updated_at BEFORE UPDATE ON dsl_capability_registry FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_kg_execution_traces_updated_at BEFORE UPDATE ON kg_execution_traces FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SAMPLE DATA FOR TESTING
-- ============================================================================

-- Insert sample tenant
INSERT INTO tenant_metadata (tenant_id, tenant_name, tenant_type, industry_code, region_code, compliance_requirements, created_by_user_id) 
VALUES (1300, 'Sample SaaS Company', 'enterprise', 'SAAS', 'US-EAST-1', '["SOX", "GDPR"]', 1323)
ON CONFLICT (tenant_id) DO NOTHING;

-- Insert sample policy pack
INSERT INTO dsl_policy_packs (policy_pack_id, tenant_id, policy_name, policy_version, policy_description, policy_rules, industry_overlay, compliance_framework, created_by_user_id)
VALUES (
    'saas_pipeline_policy_v1',
    1300,
    'SaaS Pipeline Hygiene Policy',
    '1.0.0',
    'Standard policy for SaaS pipeline hygiene workflows',
    '{"max_execution_time": 300, "required_approvals": 1, "data_retention_days": 90}',
    'SAAS',
    'SOX',
    1323
) ON CONFLICT (policy_pack_id) DO NOTHING;

-- Insert sample capability
INSERT INTO dsl_capability_registry (capability_id, tenant_id, capability_name, capability_type, operator_definition, input_schema, output_schema, created_by_user_id)
VALUES (
    uuid_generate_v4(),
    1300,
    'salesforce_query',
    'query',
    '{"type": "query", "source": "salesforce", "supports_soql": true}',
    '{"type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object"}}}',
    '{"type": "object", "properties": {"results": {"type": "array"}, "count": {"type": "integer"}}}',
    1323
) ON CONFLICT (capability_id) DO NOTHING;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE tenant_metadata IS 'Core tenant information with compliance and regional settings';
COMMENT ON TABLE dsl_workflows IS 'Workflow definitions with DSL content and governance metadata';
COMMENT ON TABLE dsl_policy_packs IS 'Policy definitions for governance and compliance enforcement';
COMMENT ON TABLE dsl_execution_traces IS 'Immutable execution traces for audit and analytics';
COMMENT ON TABLE dsl_evidence_packs IS 'Cryptographically signed evidence for regulatory compliance';
COMMENT ON TABLE dsl_override_ledger IS 'Tamper-evident ledger of all governance overrides';
COMMENT ON TABLE dsl_workflow_templates IS 'Reusable workflow templates with industry overlays';
COMMENT ON TABLE dsl_capability_registry IS 'Registry of available operators and capabilities';
COMMENT ON TABLE kg_execution_traces IS 'Knowledge graph enrichment of execution traces';
COMMENT ON TABLE idempotency_records IS 'Idempotency tracking to prevent duplicate executions';

-- Grant permissions (adjust as needed for your security model)
-- These would typically be more restrictive in production
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rba_application_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rba_application_role;

-- ============================================================================
-- VALIDATION QUERIES
-- ============================================================================

-- Verify table creation
SELECT 
    schemaname,
    tablename,
    tableowner,
    hasindexes,
    hasrules,
    hastriggers
FROM pg_tables 
WHERE tablename LIKE 'dsl_%' OR tablename IN ('tenant_metadata', 'kg_execution_traces', 'idempotency_records')
ORDER BY tablename;

-- Verify RLS is enabled
SELECT 
    schemaname,
    tablename,
    rowsecurity
FROM pg_tables 
WHERE tablename LIKE 'dsl_%' OR tablename IN ('tenant_metadata', 'kg_execution_traces', 'idempotency_records')
ORDER BY tablename;

-- Verify indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename LIKE 'dsl_%' OR tablename IN ('tenant_metadata', 'kg_execution_traces', 'idempotency_records')
ORDER BY tablename, indexname;
