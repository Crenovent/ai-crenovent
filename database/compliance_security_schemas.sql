-- Chapter 17: Compliance & Security Validation Database Schemas
-- ============================================================
-- Tasks 17.1-17.4: Complete database schemas for compliance and security validation

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- SECTION 1: POLICY PACK VALIDATION (Chapter 17.1)
-- ============================================================================

-- Task 17.1-T03: Policy pack schema and registry
CREATE TABLE IF NOT EXISTS policy_packs (
    id VARCHAR(255) PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    rules JSONB NOT NULL,
    scope JSONB NOT NULL,
    industry VARCHAR(50) NOT NULL,
    tenant_id INTEGER REFERENCES tenant_metadata(tenant_id),
    compliance_frameworks TEXT[] DEFAULT '{}',
    precedence_level INTEGER NOT NULL DEFAULT 2 CHECK (precedence_level BETWEEN 1 AND 3),
    enforcement_level VARCHAR(20) NOT NULL DEFAULT 'STRICT' CHECK (enforcement_level IN ('STRICT', 'ADVISORY', 'DISABLED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT' CHECK (status IN ('DRAFT', 'PUBLISHED', 'RETIRED')),
    metadata JSONB DEFAULT '{}',
    
    -- Unique constraint for tenant-specific policy packs
    UNIQUE(id, tenant_id)
);

-- Enable RLS on policy_packs
ALTER TABLE policy_packs ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only access policy packs for their tenant or global policies
CREATE POLICY policy_packs_rls_policy ON policy_packs
    FOR ALL USING (
        tenant_id IS NULL OR 
        tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1)
    );

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_policy_packs_tenant_industry ON policy_packs(tenant_id, industry);
CREATE INDEX IF NOT EXISTS idx_policy_packs_status_enforcement ON policy_packs(status, enforcement_level);
CREATE INDEX IF NOT EXISTS idx_policy_packs_precedence ON policy_packs(precedence_level);
CREATE INDEX IF NOT EXISTS idx_policy_packs_frameworks ON policy_packs USING GIN(compliance_frameworks);

-- Task 17.1-T14: Policy violation evidence schema
CREATE TABLE IF NOT EXISTS policy_violation_evidence (
    violation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_id VARCHAR(255) NOT NULL REFERENCES policy_packs(id),
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255),
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    user_id VARCHAR(255) NOT NULL,
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')),
    reason TEXT NOT NULL,
    evidence_data JSONB NOT NULL,
    digital_signature VARCHAR(500),
    hash_chain_ref VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Audit trail fields
    immutable_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            violation_id::text || policy_id || workflow_id || 
            tenant_id::text || user_id || violation_type || 
            severity || reason || evidence_data::text || 
            created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Enable RLS on policy_violation_evidence
ALTER TABLE policy_violation_evidence ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for violation evidence
CREATE POLICY policy_violation_evidence_rls_policy ON policy_violation_evidence
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Indexes for violation evidence
CREATE INDEX IF NOT EXISTS idx_violation_evidence_tenant_policy ON policy_violation_evidence(tenant_id, policy_id);
CREATE INDEX IF NOT EXISTS idx_violation_evidence_workflow ON policy_violation_evidence(workflow_id);
CREATE INDEX IF NOT EXISTS idx_violation_evidence_severity ON policy_violation_evidence(severity);
CREATE INDEX IF NOT EXISTS idx_violation_evidence_created_at ON policy_violation_evidence(created_at);

-- Task 17.1-T17: Policy override schema
CREATE TABLE IF NOT EXISTS policy_override_ledger (
    override_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    violation_id UUID REFERENCES policy_violation_evidence(violation_id),
    policy_id VARCHAR(255) NOT NULL REFERENCES policy_packs(id),
    approver_id VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    risk_assessment TEXT NOT NULL,
    expiration_time TIMESTAMPTZ,
    approval_chain TEXT[] DEFAULT '{}',
    business_impact TEXT NOT NULL,
    mitigation_plan TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED', 'EXPIRED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    
    -- Immutable audit hash
    ledger_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            override_id::text || violation_id::text || policy_id || 
            approver_id || reason || risk_assessment || 
            business_impact || mitigation_plan || status || 
            created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Enable RLS on policy_override_ledger
ALTER TABLE policy_override_ledger ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for override ledger
CREATE POLICY policy_override_ledger_rls_policy ON policy_override_ledger
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Indexes for override ledger
CREATE INDEX IF NOT EXISTS idx_override_ledger_tenant_policy ON policy_override_ledger(tenant_id, policy_id);
CREATE INDEX IF NOT EXISTS idx_override_ledger_status ON policy_override_ledger(status);
CREATE INDEX IF NOT EXISTS idx_override_ledger_approver ON policy_override_ledger(approver_id);
CREATE INDEX IF NOT EXISTS idx_override_ledger_expiration ON policy_override_ledger(expiration_time) WHERE expiration_time IS NOT NULL;

-- Policy analysis results storage
CREATE TABLE IF NOT EXISTS policy_analysis_results (
    analysis_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_pack_id VARCHAR(255) NOT NULL REFERENCES policy_packs(id),
    analysis_type VARCHAR(50) NOT NULL CHECK (analysis_type IN ('EXPLAIN', 'IMPACT', 'CONFLICTS', 'RECOMMENDATIONS')),
    user_id VARCHAR(255) NOT NULL,
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    results JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Enable RLS on policy_analysis_results
ALTER TABLE policy_analysis_results ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for analysis results
CREATE POLICY policy_analysis_results_rls_policy ON policy_analysis_results
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SECTION 2: TENANT & INDUSTRY ISOLATION (Chapter 17.2)
-- ============================================================================

-- Task 17.2-T25: Isolation evidence schema
CREATE TABLE IF NOT EXISTS tenant_isolation_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    region VARCHAR(10) NOT NULL,
    access_attempt JSONB NOT NULL,
    decision VARCHAR(20) NOT NULL CHECK (decision IN ('ALLOWED', 'DENIED', 'ESCALATED')),
    isolation_type VARCHAR(50) NOT NULL,
    evidence_data JSONB NOT NULL,
    digital_signature VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    isolation_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            evidence_id::text || tenant_id::text || region || 
            access_attempt::text || decision || isolation_type || 
            evidence_data::text || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Enable RLS on tenant_isolation_evidence
ALTER TABLE tenant_isolation_evidence ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for isolation evidence
CREATE POLICY tenant_isolation_evidence_rls_policy ON tenant_isolation_evidence
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Indexes for isolation evidence
CREATE INDEX IF NOT EXISTS idx_isolation_evidence_tenant_region ON tenant_isolation_evidence(tenant_id, region);
CREATE INDEX IF NOT EXISTS idx_isolation_evidence_decision ON tenant_isolation_evidence(decision);
CREATE INDEX IF NOT EXISTS idx_isolation_evidence_type ON tenant_isolation_evidence(isolation_type);
CREATE INDEX IF NOT EXISTS idx_isolation_evidence_created_at ON tenant_isolation_evidence(created_at);

-- Cross-tenant access attempts log
CREATE TABLE IF NOT EXISTS cross_tenant_access_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    target_tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    user_id VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    access_type VARCHAR(50) NOT NULL,
    result VARCHAR(20) NOT NULL CHECK (result IN ('ALLOWED', 'DENIED', 'BLOCKED')),
    reason TEXT,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Ensure source and target tenants are different for cross-tenant access
    CHECK (source_tenant_id != target_tenant_id)
);

-- Enable RLS on cross_tenant_access_log
ALTER TABLE cross_tenant_access_log ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only see logs involving their tenant
CREATE POLICY cross_tenant_access_log_rls_policy ON cross_tenant_access_log
    FOR ALL USING (
        source_tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1) OR
        target_tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1)
    );

-- Indexes for cross-tenant access log
CREATE INDEX IF NOT EXISTS idx_cross_tenant_log_source ON cross_tenant_access_log(source_tenant_id);
CREATE INDEX IF NOT EXISTS idx_cross_tenant_log_target ON cross_tenant_access_log(target_tenant_id);
CREATE INDEX IF NOT EXISTS idx_cross_tenant_log_result ON cross_tenant_access_log(result);
CREATE INDEX IF NOT EXISTS idx_cross_tenant_log_created_at ON cross_tenant_access_log(created_at);

-- ============================================================================
-- SECTION 3: PENETRATION TESTING & VULNERABILITY SCANS (Chapter 17.3)
-- ============================================================================

-- Task 17.3-T43: Pentest evidence schema
CREATE TABLE IF NOT EXISTS penetration_test_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    attack_id VARCHAR(255) NOT NULL,
    test_type VARCHAR(100) NOT NULL,
    vector VARCHAR(255) NOT NULL,
    target_system VARCHAR(255) NOT NULL,
    result VARCHAR(20) NOT NULL CHECK (result IN ('SUCCESS', 'FAILED', 'BLOCKED', 'DETECTED')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')),
    evidence_data JSONB NOT NULL,
    remediation_status VARCHAR(50) DEFAULT 'OPEN',
    remediation_plan TEXT,
    digital_signature VARCHAR(500),
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    pentest_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            evidence_id::text || attack_id || test_type || 
            vector || target_system || result || severity || 
            evidence_data::text || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Enable RLS on penetration_test_evidence
ALTER TABLE penetration_test_evidence ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for pentest evidence
CREATE POLICY penetration_test_evidence_rls_policy ON penetration_test_evidence
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Indexes for pentest evidence
CREATE INDEX IF NOT EXISTS idx_pentest_evidence_tenant ON penetration_test_evidence(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pentest_evidence_severity ON penetration_test_evidence(severity);
CREATE INDEX IF NOT EXISTS idx_pentest_evidence_result ON penetration_test_evidence(result);
CREATE INDEX IF NOT EXISTS idx_pentest_evidence_test_type ON penetration_test_evidence(test_type);
CREATE INDEX IF NOT EXISTS idx_pentest_evidence_created_at ON penetration_test_evidence(created_at);

-- Vulnerability scan results
CREATE TABLE IF NOT EXISTS vulnerability_scan_results (
    scan_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_type VARCHAR(100) NOT NULL,
    target_component VARCHAR(255) NOT NULL,
    vulnerability_id VARCHAR(255),
    cve_id VARCHAR(50),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')),
    description TEXT NOT NULL,
    remediation_advice TEXT,
    scan_tool VARCHAR(100) NOT NULL,
    scan_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    remediation_status VARCHAR(50) DEFAULT 'OPEN',
    false_positive BOOLEAN DEFAULT FALSE,
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    
    -- Evidence pack reference
    evidence_pack_id UUID,
    digital_signature VARCHAR(500)
);

-- Enable RLS on vulnerability_scan_results
ALTER TABLE vulnerability_scan_results ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for vulnerability scans
CREATE POLICY vulnerability_scan_results_rls_policy ON vulnerability_scan_results
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Indexes for vulnerability scan results
CREATE INDEX IF NOT EXISTS idx_vuln_scan_tenant ON vulnerability_scan_results(tenant_id);
CREATE INDEX IF NOT EXISTS idx_vuln_scan_severity ON vulnerability_scan_results(severity);
CREATE INDEX IF NOT EXISTS idx_vuln_scan_status ON vulnerability_scan_results(remediation_status);
CREATE INDEX IF NOT EXISTS idx_vuln_scan_cve ON vulnerability_scan_results(cve_id) WHERE cve_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_vuln_scan_timestamp ON vulnerability_scan_results(scan_timestamp);

-- ============================================================================
-- SECTION 4: RISK SIMULATION & STRESS TESTING (Chapter 17.4)
-- ============================================================================

-- Task 17.4-T05: Risk scenarios schema
CREATE TABLE IF NOT EXISTS risk_scenarios (
    scenario_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    scenario_type VARCHAR(50) NOT NULL CHECK (scenario_type IN ('infrastructure', 'compliance', 'operational', 'financial', 'security', 'industry_specific')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    industry VARCHAR(50),
    compliance_frameworks TEXT[] DEFAULT '{}',
    simulation_parameters JSONB NOT NULL,
    expected_duration_minutes INTEGER NOT NULL DEFAULT 30,
    success_criteria JSONB NOT NULL,
    failure_conditions JSONB NOT NULL,
    recovery_procedures TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE', 'DEPRECATED'))
);

-- Indexes for risk scenarios
CREATE INDEX IF NOT EXISTS idx_risk_scenarios_type ON risk_scenarios(scenario_type);
CREATE INDEX IF NOT EXISTS idx_risk_scenarios_severity ON risk_scenarios(severity);
CREATE INDEX IF NOT EXISTS idx_risk_scenarios_industry ON risk_scenarios(industry) WHERE industry IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_risk_scenarios_frameworks ON risk_scenarios USING GIN(compliance_frameworks);
CREATE INDEX IF NOT EXISTS idx_risk_scenarios_status ON risk_scenarios(status);

-- Task 17.4-T48: Risk simulation evidence schema
CREATE TABLE IF NOT EXISTS risk_simulation_evidence (
    evidence_pack_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    simulation_id VARCHAR(255) NOT NULL,
    scenario_id VARCHAR(255) NOT NULL REFERENCES risk_scenarios(scenario_id),
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    industry VARCHAR(50),
    simulation_result VARCHAR(20) NOT NULL CHECK (simulation_result IN ('SUCCESS', 'FAILURE', 'PARTIAL', 'CANCELLED')),
    impact_assessment JSONB NOT NULL,
    metrics_collected JSONB NOT NULL,
    evidence_data JSONB NOT NULL,
    digital_signature VARCHAR(500),
    hash_chain_ref VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Immutable audit hash
    simulation_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(digest(
            evidence_pack_id::text || simulation_id || scenario_id || 
            tenant_id::text || simulation_result || 
            impact_assessment::text || metrics_collected::text || 
            evidence_data::text || created_at::text, 'sha256'
        ), 'hex')
    ) STORED
);

-- Enable RLS on risk_simulation_evidence
ALTER TABLE risk_simulation_evidence ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for simulation evidence
CREATE POLICY risk_simulation_evidence_rls_policy ON risk_simulation_evidence
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Indexes for risk simulation evidence
CREATE INDEX IF NOT EXISTS idx_risk_sim_evidence_tenant ON risk_simulation_evidence(tenant_id);
CREATE INDEX IF NOT EXISTS idx_risk_sim_evidence_scenario ON risk_simulation_evidence(scenario_id);
CREATE INDEX IF NOT EXISTS idx_risk_sim_evidence_result ON risk_simulation_evidence(simulation_result);
CREATE INDEX IF NOT EXISTS idx_risk_sim_evidence_created_at ON risk_simulation_evidence(created_at);

-- Risk simulation execution results
CREATE TABLE IF NOT EXISTS risk_simulation_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    simulation_id VARCHAR(255) NOT NULL,
    scenario_id VARCHAR(255) NOT NULL REFERENCES risk_scenarios(scenario_id),
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    duration_seconds NUMERIC(10,3),
    success BOOLEAN NOT NULL,
    metrics JSONB NOT NULL,
    evidence_pack_id UUID REFERENCES risk_simulation_evidence(evidence_pack_id),
    trust_score_impact NUMERIC(5,3),
    risk_register_entries TEXT[] DEFAULT '{}',
    recovery_actions TEXT[] DEFAULT '{}',
    lessons_learned TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Enable RLS on risk_simulation_results
ALTER TABLE risk_simulation_results ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for simulation results
CREATE POLICY risk_simulation_results_rls_policy ON risk_simulation_results
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Indexes for risk simulation results
CREATE INDEX IF NOT EXISTS idx_risk_sim_results_tenant ON risk_simulation_results(tenant_id);
CREATE INDEX IF NOT EXISTS idx_risk_sim_results_simulation ON risk_simulation_results(simulation_id);
CREATE INDEX IF NOT EXISTS idx_risk_sim_results_status ON risk_simulation_results(status);
CREATE INDEX IF NOT EXISTS idx_risk_sim_results_success ON risk_simulation_results(success);
CREATE INDEX IF NOT EXISTS idx_risk_sim_results_start_time ON risk_simulation_results(start_time);

-- Risk register for tracking identified risks
CREATE TABLE IF NOT EXISTS risk_register (
    risk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL REFERENCES tenant_metadata(tenant_id),
    risk_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')),
    description TEXT NOT NULL,
    impact TEXT NOT NULL,
    likelihood VARCHAR(20) NOT NULL CHECK (likelihood IN ('VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')),
    mitigation_plan TEXT,
    owner_id VARCHAR(255),
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'IN_PROGRESS', 'MITIGATED', 'CLOSED', 'ACCEPTED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    due_date TIMESTAMPTZ,
    
    -- Link to simulation that identified this risk
    source_simulation_id VARCHAR(255),
    source_evidence_pack_id UUID REFERENCES risk_simulation_evidence(evidence_pack_id)
);

-- Enable RLS on risk_register
ALTER TABLE risk_register ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Tenant isolation for risk register
CREATE POLICY risk_register_rls_policy ON risk_register
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Indexes for risk register
CREATE INDEX IF NOT EXISTS idx_risk_register_tenant ON risk_register(tenant_id);
CREATE INDEX IF NOT EXISTS idx_risk_register_severity ON risk_register(severity);
CREATE INDEX IF NOT EXISTS idx_risk_register_status ON risk_register(status);
CREATE INDEX IF NOT EXISTS idx_risk_register_owner ON risk_register(owner_id) WHERE owner_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_risk_register_due_date ON risk_register(due_date) WHERE due_date IS NOT NULL;

-- ============================================================================
-- SECTION 5: AUTOMATED TRIGGERS AND FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
CREATE TRIGGER update_policy_packs_updated_at BEFORE UPDATE ON policy_packs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_risk_register_updated_at BEFORE UPDATE ON risk_register FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically expire policy overrides
CREATE OR REPLACE FUNCTION expire_policy_overrides()
RETURNS void AS $$
BEGIN
    UPDATE policy_override_ledger 
    SET status = 'EXPIRED'
    WHERE status = 'APPROVED' 
    AND expiration_time IS NOT NULL 
    AND expiration_time < NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to validate policy pack rules JSON schema
CREATE OR REPLACE FUNCTION validate_policy_pack_rules(rules_json JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Basic validation - ensure rules is an array
    IF jsonb_typeof(rules_json) != 'array' THEN
        RETURN FALSE;
    END IF;
    
    -- Ensure each rule has required fields
    IF EXISTS (
        SELECT 1 FROM jsonb_array_elements(rules_json) AS rule
        WHERE NOT (rule ? 'type' AND rule ? 'name')
    ) THEN
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to generate hash chain for evidence integrity
CREATE OR REPLACE FUNCTION generate_evidence_hash_chain(
    evidence_type VARCHAR(50),
    evidence_data JSONB,
    previous_hash VARCHAR(64) DEFAULT NULL
)
RETURNS VARCHAR(64) AS $$
DECLARE
    current_hash VARCHAR(64);
    chain_input TEXT;
BEGIN
    -- Create input for hash chain
    chain_input := evidence_type || '|' || evidence_data::text || '|' || 
                   COALESCE(previous_hash, '') || '|' || 
                   extract(epoch from NOW())::text;
    
    -- Generate SHA-256 hash
    current_hash := encode(digest(chain_input, 'sha256'), 'hex');
    
    RETURN current_hash;
END;
$$ LANGUAGE plpgsql;

-- Function to check cross-tenant access violations
CREATE OR REPLACE FUNCTION log_cross_tenant_access(
    source_tenant INTEGER,
    target_tenant INTEGER,
    user_id VARCHAR(255),
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    access_type VARCHAR(50)
)
RETURNS BOOLEAN AS $$
DECLARE
    access_allowed BOOLEAN := FALSE;
BEGIN
    -- Log the access attempt
    INSERT INTO cross_tenant_access_log (
        source_tenant_id, target_tenant_id, user_id, 
        resource_type, resource_id, access_type, result
    ) VALUES (
        source_tenant, target_tenant, user_id,
        resource_type, resource_id, access_type, 
        CASE WHEN access_allowed THEN 'ALLOWED' ELSE 'DENIED' END
    );
    
    RETURN access_allowed;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECTION 6: COMPLIANCE FRAMEWORK TEMPLATES
-- ============================================================================

-- Insert default policy pack templates for major compliance frameworks
INSERT INTO policy_packs (
    id, version, name, description, rules, scope, industry, 
    compliance_frameworks, precedence_level, enforcement_level, 
    created_by, status
) VALUES 
-- SOX Compliance Template
(
    'SOX_SAAS_TEMPLATE', '1.0.0', 'SOX Compliance for SaaS', 
    'Sarbanes-Oxley compliance template for SaaS companies',
    '[
        {
            "name": "financial_data_validation",
            "type": "data_validation",
            "required_fields": ["revenue", "customer_id", "transaction_date"],
            "validation_rules": "revenue > 0 AND customer_id IS NOT NULL"
        },
        {
            "name": "segregation_of_duties",
            "type": "sod_check",
            "maker_checker_required": true,
            "approval_chain": ["finance_manager", "cfo"]
        }
    ]'::jsonb,
    '{"tenant_scope": "all", "workflow_types": ["financial", "revenue"]}'::jsonb,
    'SaaS', '{"SOX"}', 1, 'STRICT', 'system', 'PUBLISHED'
),
-- GDPR Compliance Template
(
    'GDPR_SAAS_TEMPLATE', '1.0.0', 'GDPR Compliance for SaaS',
    'GDPR data protection template for SaaS companies',
    '[
        {
            "name": "data_consent_validation",
            "type": "compliance_check",
            "required_frameworks": ["GDPR"],
            "consent_required": true
        },
        {
            "name": "data_retention_check",
            "type": "retention_policy",
            "max_retention_days": 2555,
            "auto_purge": true
        }
    ]'::jsonb,
    '{"tenant_scope": "all", "data_types": ["personal_data", "customer_data"]}'::jsonb,
    'SaaS', '{"GDPR"}', 1, 'STRICT', 'system', 'PUBLISHED'
),
-- RBI Banking Template
(
    'RBI_BANKING_TEMPLATE', '1.0.0', 'RBI Banking Compliance',
    'Reserve Bank of India compliance template for banking',
    '[
        {
            "name": "kyc_validation",
            "type": "compliance_check",
            "required_frameworks": ["RBI", "KYC"],
            "kyc_documents_required": true
        },
        {
            "name": "loan_approval_limits",
            "type": "financial_limit",
            "max_loan_amount": 10000000,
            "approval_required": true
        }
    ]'::jsonb,
    '{"tenant_scope": "banking", "transaction_types": ["loan", "deposit", "transfer"]}'::jsonb,
    'BFSI', '{"RBI", "BASEL_III"}', 1, 'STRICT', 'system', 'PUBLISHED'
)
ON CONFLICT (id, tenant_id) DO NOTHING;

-- Insert default risk scenarios
INSERT INTO risk_scenarios (
    scenario_id, name, description, scenario_type, severity,
    industry, compliance_frameworks, simulation_parameters,
    success_criteria, failure_conditions, recovery_procedures,
    created_by
) VALUES 
(
    'NETWORK_PARTITION_BASIC', 'Basic Network Partition Test',
    'Simulate network partition between core services',
    'infrastructure', 'high', NULL, '{}',
    '{"type": "network_partition", "duration_minutes": 5, "affected_services": ["orchestrator", "runtime"]}'::jsonb,
    '{"system_recovery": true, "data_consistency": true, "max_downtime_seconds": 300}'::jsonb,
    '{"data_loss": true, "service_unavailable_minutes": 10}'::jsonb,
    '{"Restore network connectivity", "Verify data consistency", "Check service health"}',
    'system'
),
(
    'GDPR_MASS_PURGE', 'GDPR Mass Data Purge Simulation',
    'Simulate mass GDPR right-to-forget requests',
    'compliance', 'medium', 'SaaS', '{"GDPR"}',
    '{"type": "gdpr_mass_purge", "user_count": 1000, "concurrent_requests": 50}'::jsonb,
    '{"purge_completion_rate": 0.95, "audit_trail_complete": true, "max_processing_hours": 24}'::jsonb,
    '{"purge_failure_rate": 0.1, "audit_gaps": true, "processing_timeout": true}'::jsonb,
    '{"Retry failed purges", "Generate audit report", "Notify data protection officer"}',
    'system'
),
(
    'SOX_FINANCIAL_STRESS', 'SOX Financial Controls Stress Test',
    'Stress test SOX financial controls under high load',
    'compliance', 'high', 'SaaS', '{"SOX"}',
    '{"type": "sox_stress", "transaction_volume": 100000, "concurrent_users": 500}'::jsonb,
    '{"control_effectiveness": 0.99, "audit_trail_complete": true, "sod_violations": 0}'::jsonb,
    '{"control_failure_rate": 0.05, "audit_gaps": true, "sod_violations": 1}'::jsonb,
    '{"Review failed controls", "Generate compliance report", "Escalate to audit committee"}',
    'system'
)
ON CONFLICT (scenario_id) DO NOTHING;

-- ============================================================================
-- SECTION 7: PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Partitioning for large evidence tables (by month)
-- This would be implemented for production systems with high volume

-- Materialized view for compliance dashboard metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS compliance_metrics_summary AS
SELECT 
    tenant_id,
    DATE_TRUNC('day', created_at) as metric_date,
    COUNT(*) FILTER (WHERE severity = 'CRITICAL') as critical_violations,
    COUNT(*) FILTER (WHERE severity = 'HIGH') as high_violations,
    COUNT(*) FILTER (WHERE severity = 'MEDIUM') as medium_violations,
    COUNT(*) FILTER (WHERE severity = 'LOW') as low_violations,
    COUNT(*) as total_violations
FROM policy_violation_evidence
GROUP BY tenant_id, DATE_TRUNC('day', created_at);

-- Index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_compliance_metrics_summary_unique 
ON compliance_metrics_summary(tenant_id, metric_date);

-- Refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_compliance_metrics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY compliance_metrics_summary;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECTION 8: AUDIT AND MONITORING VIEWS
-- ============================================================================

-- View for policy compliance overview
CREATE OR REPLACE VIEW policy_compliance_overview AS
SELECT 
    pp.id as policy_pack_id,
    pp.name as policy_pack_name,
    pp.industry,
    pp.enforcement_level,
    COUNT(pve.violation_id) as total_violations,
    COUNT(pve.violation_id) FILTER (WHERE pve.severity = 'CRITICAL') as critical_violations,
    COUNT(pol.override_id) as total_overrides,
    COUNT(pol.override_id) FILTER (WHERE pol.status = 'APPROVED') as approved_overrides
FROM policy_packs pp
LEFT JOIN policy_violation_evidence pve ON pp.id = pve.policy_id
LEFT JOIN policy_override_ledger pol ON pve.violation_id = pol.violation_id
WHERE pp.status = 'PUBLISHED'
GROUP BY pp.id, pp.name, pp.industry, pp.enforcement_level;

-- View for tenant isolation health
CREATE OR REPLACE VIEW tenant_isolation_health AS
SELECT 
    tm.tenant_id,
    tm.tenant_name,
    tm.industry_code,
    COUNT(tie.evidence_id) as isolation_events,
    COUNT(tie.evidence_id) FILTER (WHERE tie.decision = 'DENIED') as access_denials,
    COUNT(ctal.log_id) as cross_tenant_attempts,
    COUNT(ctal.log_id) FILTER (WHERE ctal.result = 'BLOCKED') as blocked_attempts
FROM tenant_metadata tm
LEFT JOIN tenant_isolation_evidence tie ON tm.tenant_id = tie.tenant_id
LEFT JOIN cross_tenant_access_log ctal ON tm.tenant_id = ctal.source_tenant_id OR tm.tenant_id = ctal.target_tenant_id
GROUP BY tm.tenant_id, tm.tenant_name, tm.industry_code;

-- View for risk simulation summary
CREATE OR REPLACE VIEW risk_simulation_summary AS
SELECT 
    rs.scenario_id,
    rs.name as scenario_name,
    rs.scenario_type,
    rs.severity,
    COUNT(rsr.result_id) as total_executions,
    COUNT(rsr.result_id) FILTER (WHERE rsr.success = true) as successful_executions,
    AVG(rsr.duration_seconds) as avg_duration_seconds,
    COUNT(DISTINCT rsr.tenant_id) as tenants_tested
FROM risk_scenarios rs
LEFT JOIN risk_simulation_results rsr ON rs.scenario_id = rsr.scenario_id
GROUP BY rs.scenario_id, rs.name, rs.scenario_type, rs.severity;

-- Grant appropriate permissions
GRANT SELECT ON compliance_metrics_summary TO PUBLIC;
GRANT SELECT ON policy_compliance_overview TO PUBLIC;
GRANT SELECT ON tenant_isolation_health TO PUBLIC;
GRANT SELECT ON risk_simulation_summary TO PUBLIC;

-- Final comment
COMMENT ON SCHEMA public IS 'Chapter 17 Compliance & Security Validation - Complete database schema implementation with RLS, audit trails, and performance optimization';
