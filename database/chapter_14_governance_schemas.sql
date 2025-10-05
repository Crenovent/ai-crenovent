-- Chapter 14 Governance Controls Database Schemas
-- ================================================
-- Tasks 14.1-T63, 14.1-T64: Approval & Override Ledger schemas
-- Tasks 14.2-T48: Trust scoring storage
-- Tasks 14.3-T03: Evidence pack schema
-- Tasks 14.4-T03: Audit pack schema

-- Enable RLS and required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- 14.1 APPROVAL WORKFLOWS & OVERRIDE LEDGER SCHEMAS
-- =====================================================

-- Task 14.1-T63: Approval Ledger Schema
CREATE TABLE IF NOT EXISTS approval_ledger (
    approval_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL,
    workflow_execution_id UUID,
    tenant_id INTEGER NOT NULL,
    
    -- Approval Request Details
    requested_by_user_id INTEGER NOT NULL,
    request_type VARCHAR(100) NOT NULL, -- 'workflow_approval', 'policy_exception', 'override_request'
    request_reason TEXT NOT NULL,
    business_justification TEXT,
    risk_assessment VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    
    -- Approval Chain
    approval_chain JSONB NOT NULL, -- Array of approver roles/users
    current_approver_index INTEGER DEFAULT 0,
    
    -- Status & Timing
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected', 'expired', 'escalated'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Approvals Received
    approvals JSONB DEFAULT '[]'::jsonb, -- Array of approval records
    
    -- Evidence & Compliance
    evidence_pack_id UUID,
    policy_pack_refs JSONB, -- References to applicable policy packs
    compliance_frameworks TEXT[], -- ['SOX', 'GDPR', 'RBI', etc.]
    
    -- Audit Trail
    digital_signature TEXT, -- Cryptographic signature
    hash_chain_ref TEXT, -- Reference to hash chain for immutability
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_approval_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'approved', 'rejected', 'expired', 'escalated')),
    CONSTRAINT valid_risk CHECK (risk_assessment IN ('low', 'medium', 'high', 'critical'))
);

-- Task 14.1-T64: Override Ledger Schema  
CREATE TABLE IF NOT EXISTS override_ledger (
    override_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL,
    workflow_execution_id UUID,
    tenant_id INTEGER NOT NULL,
    
    -- Override Details
    override_type VARCHAR(100) NOT NULL, -- 'policy_bypass', 'compliance_exception', 'emergency_override'
    reason_code VARCHAR(50) NOT NULL, -- Standardized reason codes
    reason_description TEXT NOT NULL,
    business_impact_assessment TEXT,
    
    -- Authorization
    requested_by_user_id INTEGER NOT NULL,
    approved_by_user_id INTEGER,
    approval_id UUID, -- Link to approval_ledger if approval was required
    
    -- Policy Context
    bypassed_policies JSONB, -- Array of policy IDs that were bypassed
    compliance_frameworks_affected TEXT[], -- Which frameworks were impacted
    
    -- Risk & Impact
    risk_level VARCHAR(20) NOT NULL,
    estimated_business_impact DECIMAL(15,2), -- Financial impact estimate
    actual_business_impact DECIMAL(15,2), -- Actual impact (filled later)
    
    -- Timing & Expiration
    override_start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    override_end_time TIMESTAMP WITH TIME ZONE, -- When override expires
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'expired', 'revoked', 'resolved'
    
    -- Evidence & Audit
    evidence_pack_id UUID,
    digital_signature TEXT,
    hash_chain_ref TEXT,
    
    -- Remediation
    remediation_plan TEXT,
    remediation_completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_override_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT fk_override_approval FOREIGN KEY (approval_id) REFERENCES approval_ledger(approval_id),
    CONSTRAINT valid_override_status CHECK (status IN ('active', 'expired', 'revoked', 'resolved')),
    CONSTRAINT valid_override_risk CHECK (risk_level IN ('low', 'medium', 'high', 'critical'))
);

-- =====================================================
-- 14.2 TRUST SCORING SCHEMAS
-- =====================================================

-- Task 14.2-T48: Trust Scores Storage
CREATE TABLE IF NOT EXISTS trust_scores (
    trust_score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Scoring Target
    target_type VARCHAR(50) NOT NULL, -- 'workflow', 'capability', 'user', 'tenant'
    target_id VARCHAR(255) NOT NULL, -- ID of the target being scored
    
    -- Trust Score Components
    overall_score DECIMAL(5,3) NOT NULL, -- 0.000 to 1.000
    trust_level VARCHAR(20) NOT NULL, -- 'excellent', 'good', 'acceptable', 'concerning', 'poor'
    
    -- Dimension Scores
    policy_compliance_score DECIMAL(5,3),
    override_frequency_score DECIMAL(5,3),
    sla_adherence_score DECIMAL(5,3),
    evidence_completeness_score DECIMAL(5,3),
    risk_score DECIMAL(5,3),
    
    -- Scoring Context
    industry_code VARCHAR(20) DEFAULT 'SaaS',
    tenant_tier VARCHAR(10) DEFAULT 'T2',
    scoring_algorithm_version VARCHAR(10) DEFAULT 'v1.0',
    
    -- Calculation Details
    sample_size INTEGER DEFAULT 0,
    confidence_interval DECIMAL(5,3),
    calculation_factors JSONB, -- Detailed factor breakdown
    
    -- Timing
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    valid_until TIMESTAMP WITH TIME ZONE,
    lookback_period_days INTEGER DEFAULT 30,
    
    -- Recommendations
    recommendations JSONB DEFAULT '[]'::jsonb,
    
    -- Audit
    digital_signature TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_trust_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT valid_trust_level CHECK (trust_level IN ('excellent', 'good', 'acceptable', 'concerning', 'poor')),
    CONSTRAINT valid_score_range CHECK (overall_score >= 0.000 AND overall_score <= 1.000)
);

-- =====================================================
-- 14.3 EVIDENCE PACK SCHEMAS
-- =====================================================

-- Task 14.3-T03: Evidence Pack Schema
CREATE TABLE IF NOT EXISTS evidence_packs (
    evidence_pack_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Pack Identification
    pack_type VARCHAR(50) NOT NULL, -- 'workflow_execution', 'approval_decision', 'override_event', 'compliance_audit'
    pack_name VARCHAR(255) NOT NULL,
    pack_version VARCHAR(10) DEFAULT 'v1.0',
    
    -- Source Context
    workflow_id UUID,
    workflow_execution_id UUID,
    approval_id UUID,
    override_id UUID,
    
    -- Evidence Content
    evidence_data JSONB NOT NULL, -- Core evidence payload
    evidence_hash TEXT NOT NULL, -- SHA-256 hash of evidence_data
    evidence_size_bytes BIGINT,
    
    -- Compliance Context
    compliance_frameworks TEXT[], -- ['SOX', 'GDPR', 'RBI', etc.]
    policy_pack_refs JSONB, -- References to policy packs
    
    -- WORM Storage
    storage_location TEXT NOT NULL, -- Azure Blob Storage path
    storage_tier VARCHAR(20) DEFAULT 'hot', -- 'hot', 'cool', 'archive'
    immutable_until TIMESTAMP WITH TIME ZONE, -- WORM retention period
    
    -- Digital Signature & Integrity
    digital_signature TEXT NOT NULL,
    signature_algorithm VARCHAR(50) DEFAULT 'RSA-SHA256',
    hash_chain_ref TEXT, -- Reference to blockchain anchor
    
    -- Lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    retention_period_years INTEGER DEFAULT 7, -- SOX default
    
    -- Export & Access
    export_count INTEGER DEFAULT 0,
    last_exported_at TIMESTAMP WITH TIME ZONE,
    access_log JSONB DEFAULT '[]'::jsonb,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'archived', 'deleted'
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_evidence_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT fk_evidence_approval FOREIGN KEY (approval_id) REFERENCES approval_ledger(approval_id),
    CONSTRAINT fk_evidence_override FOREIGN KEY (override_id) REFERENCES override_ledger(override_id),
    CONSTRAINT valid_evidence_status CHECK (status IN ('active', 'archived', 'deleted'))
);

-- =====================================================
-- 14.4 AUDIT PACK SCHEMAS  
-- =====================================================

-- Task 14.4-T03: Audit Pack Schema
CREATE TABLE IF NOT EXISTS audit_packs (
    audit_pack_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Pack Identification
    pack_name VARCHAR(255) NOT NULL,
    pack_version VARCHAR(10) DEFAULT 'v1.0',
    compliance_framework VARCHAR(50) NOT NULL, -- 'SOX', 'RBI', 'IRDAI', 'GDPR', 'HIPAA', 'PCI_DSS'
    industry_code VARCHAR(20) NOT NULL, -- 'SaaS', 'Banking', 'Insurance', etc.
    
    -- Audit Scope
    audit_period_start DATE NOT NULL,
    audit_period_end DATE NOT NULL,
    workflow_ids UUID[], -- Workflows included in this audit pack
    
    -- Evidence References
    evidence_pack_refs UUID[], -- References to evidence_packs
    approval_refs UUID[], -- References to approval_ledger
    override_refs UUID[], -- References to override_ledger
    risk_register_refs UUID[], -- References to risk entries
    
    -- Pack Content
    audit_summary JSONB NOT NULL, -- Executive summary
    compliance_assessment JSONB, -- Compliance status per requirement
    findings JSONB DEFAULT '[]'::jsonb, -- Audit findings
    recommendations JSONB DEFAULT '[]'::jsonb,
    
    -- Storage & Export
    storage_location TEXT NOT NULL, -- Azure Blob Storage path
    pack_size_bytes BIGINT,
    export_formats TEXT[] DEFAULT ARRAY['PDF', 'CSV'], -- Available export formats
    
    -- Digital Signature & Integrity
    digital_signature TEXT NOT NULL,
    hash_chain_ref TEXT,
    
    -- Lifecycle
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    approved_by_user_id INTEGER, -- Compliance officer approval
    approved_at TIMESTAMP WITH TIME ZONE,
    
    -- Regulator Submission
    submitted_to_regulator BOOLEAN DEFAULT FALSE,
    submission_date TIMESTAMP WITH TIME ZONE,
    regulator_reference VARCHAR(255),
    
    -- Status
    status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'review', 'approved', 'submitted', 'archived'
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_audit_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT valid_audit_status CHECK (status IN ('draft', 'review', 'approved', 'submitted', 'archived')),
    CONSTRAINT valid_framework CHECK (compliance_framework IN ('SOX', 'RBI', 'IRDAI', 'GDPR', 'HIPAA', 'PCI_DSS'))
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Approval Ledger Indexes (Task 14.1-T65)
CREATE INDEX IF NOT EXISTS idx_approval_ledger_tenant_status ON approval_ledger(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_approval_ledger_workflow ON approval_ledger(workflow_id);
CREATE INDEX IF NOT EXISTS idx_approval_ledger_created_at ON approval_ledger(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_approval_ledger_expires_at ON approval_ledger(expires_at) WHERE status = 'pending';

-- Override Ledger Indexes
CREATE INDEX IF NOT EXISTS idx_override_ledger_tenant_status ON override_ledger(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_override_ledger_workflow ON override_ledger(workflow_id);
CREATE INDEX IF NOT EXISTS idx_override_ledger_created_at ON override_ledger(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_override_ledger_risk_level ON override_ledger(risk_level);

-- Trust Scores Indexes
CREATE INDEX IF NOT EXISTS idx_trust_scores_tenant_target ON trust_scores(tenant_id, target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_trust_scores_calculated_at ON trust_scores(calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_trust_scores_valid_until ON trust_scores(valid_until) WHERE valid_until > NOW();

-- Evidence Packs Indexes
CREATE INDEX IF NOT EXISTS idx_evidence_packs_tenant_type ON evidence_packs(tenant_id, pack_type);
CREATE INDEX IF NOT EXISTS idx_evidence_packs_workflow ON evidence_packs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_evidence_packs_created_at ON evidence_packs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_packs_compliance ON evidence_packs USING GIN(compliance_frameworks);

-- Audit Packs Indexes
CREATE INDEX IF NOT EXISTS idx_audit_packs_tenant_framework ON audit_packs(tenant_id, compliance_framework);
CREATE INDEX IF NOT EXISTS idx_audit_packs_period ON audit_packs(audit_period_start, audit_period_end);
CREATE INDEX IF NOT EXISTS idx_audit_packs_status ON audit_packs(status);

-- =====================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE approval_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE override_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE trust_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence_packs ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_packs ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation
CREATE POLICY approval_ledger_tenant_isolation ON approval_ledger
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('app.current_tenant_id')::integer);

CREATE POLICY override_ledger_tenant_isolation ON override_ledger
    FOR ALL TO authenticated  
    USING (tenant_id = current_setting('app.current_tenant_id')::integer);

CREATE POLICY trust_scores_tenant_isolation ON trust_scores
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('app.current_tenant_id')::integer);

CREATE POLICY evidence_packs_tenant_isolation ON evidence_packs
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('app.current_tenant_id')::integer);

CREATE POLICY audit_packs_tenant_isolation ON audit_packs
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- =====================================================
-- SAMPLE DATA FOR TESTING
-- =====================================================

-- Insert sample approval request
INSERT INTO approval_ledger (
    workflow_id, tenant_id, requested_by_user_id, request_type, 
    request_reason, business_justification, risk_assessment,
    approval_chain, compliance_frameworks
) VALUES (
    uuid_generate_v4(), 1300, 1323, 'workflow_approval',
    'SaaS pipeline hygiene workflow requires approval for production deployment',
    'Critical for Q4 revenue forecasting accuracy and SOX compliance',
    'medium',
    '["sales_manager", "compliance_officer", "cfo"]'::jsonb,
    ARRAY['SOX', 'GDPR']
) ON CONFLICT DO NOTHING;

-- Insert sample trust score
INSERT INTO trust_scores (
    tenant_id, target_type, target_id, overall_score, trust_level,
    policy_compliance_score, sla_adherence_score, evidence_completeness_score,
    industry_code, tenant_tier, recommendations
) VALUES (
    1300, 'workflow', 'pipeline_hygiene_v1', 0.875, 'good',
    0.90, 0.85, 0.88,
    'SaaS', 'T2', 
    '["Improve SLA adherence for better trust score", "Ensure complete evidence capture"]'::jsonb
) ON CONFLICT DO NOTHING;

COMMENT ON TABLE approval_ledger IS 'Task 14.1-T63: Approval workflow ledger with maker-checker enforcement';
COMMENT ON TABLE override_ledger IS 'Task 14.1-T64: Override ledger for compliance bypass tracking';
COMMENT ON TABLE trust_scores IS 'Task 14.2-T48: Trust scoring storage with multi-dimensional scoring';
COMMENT ON TABLE evidence_packs IS 'Task 14.3-T03: Evidence pack schema with WORM storage and digital signatures';
COMMENT ON TABLE audit_packs IS 'Task 14.4-T03: Industry-specific audit pack schema for regulator readiness';
