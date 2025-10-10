-- ============================================================================
-- CHAPTER 16.1: POLICY ORCHESTRATION DATABASE SCHEMA
-- ============================================================================
-- Tasks 16.1.1-16.1.3: Core policy orchestration infrastructure
-- Implements policy metadata, versioning, and lifecycle management

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- TASK 16.1.1: Provision policy_orch_policy table
-- ============================================================================

-- Store policy metadata (name, industry, SLA)
CREATE TABLE IF NOT EXISTS policy_orch_policy (
    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Policy Identification
    policy_name VARCHAR(255) NOT NULL,
    policy_description TEXT,
    policy_type VARCHAR(50) NOT NULL, -- 'compliance', 'business_rule', 'security', 'governance'
    
    -- Industry & Context
    industry_code VARCHAR(20) NOT NULL, -- 'SaaS', 'Banking', 'Insurance'
    compliance_framework VARCHAR(50) NOT NULL, -- 'SOX', 'GDPR', 'RBI', 'HIPAA', 'NAIC'
    jurisdiction VARCHAR(10) NOT NULL DEFAULT 'US', -- 'US', 'EU', 'IN'
    
    -- SLA & Tenant Context
    sla_tier VARCHAR(20) NOT NULL DEFAULT 'standard', -- 'basic', 'standard', 'premium', 'enterprise'
    target_automation_types TEXT[] DEFAULT ARRAY['RBA', 'RBIA', 'AALA'], -- Which automation types this policy applies to
    
    -- Policy Content
    policy_content JSONB NOT NULL, -- OPA/Rego policy rules
    policy_metadata JSONB DEFAULT '{}', -- Additional metadata
    
    -- Enforcement Configuration
    enforcement_mode VARCHAR(20) NOT NULL DEFAULT 'enforcing', -- 'enforcing', 'permissive', 'disabled'
    fail_closed BOOLEAN DEFAULT true, -- Fail-closed enforcement
    override_allowed BOOLEAN DEFAULT true, -- Allow overrides with justification
    
    -- Lifecycle
    status VARCHAR(20) NOT NULL DEFAULT 'draft', -- 'draft', 'published', 'promoted', 'deprecated', 'retired'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT valid_policy_type CHECK (policy_type IN ('compliance', 'business_rule', 'security', 'governance')),
    CONSTRAINT valid_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'FinTech', 'E-commerce', 'IT_Services')),
    CONSTRAINT valid_compliance_framework CHECK (compliance_framework IN ('SOX', 'GDPR', 'RBI', 'DPDP', 'HIPAA', 'NAIC', 'AML_KYC', 'PCI_DSS', 'SAAS_BUSINESS_RULES')),
    CONSTRAINT valid_sla_tier CHECK (sla_tier IN ('basic', 'standard', 'premium', 'enterprise')),
    CONSTRAINT valid_enforcement_mode CHECK (enforcement_mode IN ('enforcing', 'permissive', 'disabled')),
    CONSTRAINT valid_status CHECK (status IN ('draft', 'published', 'promoted', 'deprecated', 'retired')),
    
    -- Foreign Keys
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_policy_orch_policy_tenant_id ON policy_orch_policy(tenant_id);
CREATE INDEX IF NOT EXISTS idx_policy_orch_policy_industry ON policy_orch_policy(industry_code);
CREATE INDEX IF NOT EXISTS idx_policy_orch_policy_compliance ON policy_orch_policy(compliance_framework);
CREATE INDEX IF NOT EXISTS idx_policy_orch_policy_status ON policy_orch_policy(status);
CREATE INDEX IF NOT EXISTS idx_policy_orch_policy_sla ON policy_orch_policy(sla_tier);
CREATE INDEX IF NOT EXISTS idx_policy_orch_policy_enforcement ON policy_orch_policy(enforcement_mode);

-- Row Level Security
ALTER TABLE policy_orch_policy ENABLE ROW LEVEL SECURITY;
CREATE POLICY policy_orch_policy_rls_policy ON policy_orch_policy
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- TASK 16.1.2: Provision policy_orch_version table
-- ============================================================================

-- Track policy versions and states
CREATE TABLE IF NOT EXISTS policy_orch_version (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Version Information
    version_number VARCHAR(50) NOT NULL, -- Semantic versioning: MAJOR.MINOR.PATCH
    version_hash VARCHAR(64) NOT NULL, -- SHA256 hash of policy content
    
    -- Version Content
    policy_content JSONB NOT NULL, -- Versioned policy rules
    policy_metadata JSONB DEFAULT '{}',
    changelog TEXT, -- What changed in this version
    breaking_changes BOOLEAN DEFAULT false,
    
    -- Lifecycle State
    lifecycle_state VARCHAR(20) NOT NULL DEFAULT 'draft',
    
    -- Validation & Testing
    validation_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'passed', 'failed'
    validation_results JSONB DEFAULT '{}',
    test_coverage_percent DECIMAL(5,2) DEFAULT 0.0,
    
    -- Promotion & Deployment
    promoted_at TIMESTAMPTZ,
    promoted_by_user_id INTEGER,
    deployment_targets TEXT[] DEFAULT ARRAY[], -- Which environments this version is deployed to
    
    -- Approval Workflow
    approval_required BOOLEAN DEFAULT true,
    approved_by_user_id INTEGER,
    approved_at TIMESTAMPTZ,
    approval_notes TEXT,
    
    -- Lifecycle Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    
    -- Audit Trail
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT valid_lifecycle_state CHECK (lifecycle_state IN ('draft', 'testing', 'validated', 'published', 'promoted', 'deprecated', 'retired')),
    CONSTRAINT valid_validation_status CHECK (validation_status IN ('pending', 'passed', 'failed')),
    
    -- Foreign Keys
    FOREIGN KEY (policy_id) REFERENCES policy_orch_policy(policy_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    
    -- Unique constraint
    UNIQUE (policy_id, version_number)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_policy_orch_version_policy_id ON policy_orch_version(policy_id);
CREATE INDEX IF NOT EXISTS idx_policy_orch_version_tenant_id ON policy_orch_version(tenant_id);
CREATE INDEX IF NOT EXISTS idx_policy_orch_version_lifecycle ON policy_orch_version(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_policy_orch_version_validation ON policy_orch_version(validation_status);
CREATE INDEX IF NOT EXISTS idx_policy_orch_version_hash ON policy_orch_version(version_hash);

-- Row Level Security
ALTER TABLE policy_orch_version ENABLE ROW LEVEL SECURITY;
CREATE POLICY policy_orch_version_rls_policy ON policy_orch_version
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- TASK 16.1.3: Provision policy_orch_lifecycle table
-- ============================================================================

-- Store Draft → Published → Promoted → Retired lifecycle events
CREATE TABLE IF NOT EXISTS policy_orch_lifecycle (
    lifecycle_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Lifecycle Event
    event_type VARCHAR(20) NOT NULL, -- 'created', 'tested', 'validated', 'published', 'promoted', 'deprecated', 'retired'
    from_state VARCHAR(20),
    to_state VARCHAR(20) NOT NULL,
    
    -- Event Context
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    triggered_by VARCHAR(20) NOT NULL, -- 'user', 'system', 'automation', 'schedule'
    triggered_by_user_id INTEGER,
    
    -- Event Details
    event_reason TEXT,
    event_metadata JSONB DEFAULT '{}',
    
    -- Validation & Approval
    validation_required BOOLEAN DEFAULT false,
    validation_passed BOOLEAN,
    validation_results JSONB DEFAULT '{}',
    
    approval_required BOOLEAN DEFAULT false,
    approval_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    approved_by_user_id INTEGER,
    approved_at TIMESTAMPTZ,
    approval_notes TEXT,
    
    -- Evidence & Audit
    evidence_pack_id UUID, -- Link to evidence pack
    audit_trail JSONB DEFAULT '{}',
    
    -- Rollback Information
    rollback_version_id UUID, -- If this is a rollback, which version to rollback to
    rollback_reason TEXT,
    
    -- Constraints
    CONSTRAINT valid_event_type CHECK (event_type IN ('created', 'tested', 'validated', 'published', 'promoted', 'deprecated', 'retired', 'rollback')),
    CONSTRAINT valid_to_state CHECK (to_state IN ('draft', 'testing', 'validated', 'published', 'promoted', 'deprecated', 'retired')),
    CONSTRAINT valid_triggered_by CHECK (triggered_by IN ('user', 'system', 'automation', 'schedule')),
    CONSTRAINT valid_approval_status CHECK (approval_status IN ('pending', 'approved', 'rejected')),
    
    -- Foreign Keys
    FOREIGN KEY (version_id) REFERENCES policy_orch_version(version_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    FOREIGN KEY (rollback_version_id) REFERENCES policy_orch_version(version_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_policy_orch_lifecycle_version_id ON policy_orch_lifecycle(version_id);
CREATE INDEX IF NOT EXISTS idx_policy_orch_lifecycle_tenant_id ON policy_orch_lifecycle(tenant_id);
CREATE INDEX IF NOT EXISTS idx_policy_orch_lifecycle_event_type ON policy_orch_lifecycle(event_type);
CREATE INDEX IF NOT EXISTS idx_policy_orch_lifecycle_to_state ON policy_orch_lifecycle(to_state);
CREATE INDEX IF NOT EXISTS idx_policy_orch_lifecycle_timestamp ON policy_orch_lifecycle(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_policy_orch_lifecycle_approval ON policy_orch_lifecycle(approval_status);

-- Row Level Security
ALTER TABLE policy_orch_lifecycle ENABLE ROW LEVEL SECURITY;
CREATE POLICY policy_orch_lifecycle_rls_policy ON policy_orch_lifecycle
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- ADDITIONAL SUPPORTING TABLES
-- ============================================================================

-- Policy enforcement logs (for Task 16.1.18: Evidence logging)
CREATE TABLE IF NOT EXISTS policy_orch_enforcement_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    policy_id UUID NOT NULL,
    version_id UUID NOT NULL,
    
    -- Enforcement Context
    workflow_id UUID,
    execution_id UUID,
    automation_type VARCHAR(10), -- 'RBA', 'RBIA', 'AALA'
    
    -- Enforcement Result
    enforcement_result VARCHAR(20) NOT NULL, -- 'allowed', 'denied', 'overridden'
    policy_decision JSONB NOT NULL, -- OPA decision result
    
    -- Override Information (if applicable)
    override_id UUID,
    override_reason TEXT,
    override_approved_by INTEGER,
    
    -- Evidence & Audit
    evidence_pack_id UUID,
    execution_time_ms INTEGER,
    
    -- Timestamps
    enforced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_enforcement_result CHECK (enforcement_result IN ('allowed', 'denied', 'overridden')),
    CONSTRAINT valid_automation_type CHECK (automation_type IN ('RBA', 'RBIA', 'AALA')),
    
    -- Foreign Keys
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    FOREIGN KEY (policy_id) REFERENCES policy_orch_policy(policy_id),
    FOREIGN KEY (version_id) REFERENCES policy_orch_version(version_id)
);

-- Indexes for enforcement log
CREATE INDEX IF NOT EXISTS idx_policy_orch_enforcement_tenant ON policy_orch_enforcement_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_policy_orch_enforcement_policy ON policy_orch_enforcement_log(policy_id);
CREATE INDEX IF NOT EXISTS idx_policy_orch_enforcement_result ON policy_orch_enforcement_log(enforcement_result);
CREATE INDEX IF NOT EXISTS idx_policy_orch_enforcement_timestamp ON policy_orch_enforcement_log(enforced_at);

-- Row Level Security for enforcement log
ALTER TABLE policy_orch_enforcement_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY policy_orch_enforcement_log_rls_policy ON policy_orch_enforcement_log
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SAMPLE DATA INSERTION (SaaS Industry Focus)
-- ============================================================================

-- Insert sample SaaS policy for tenant 1300
INSERT INTO policy_orch_policy (
    tenant_id, policy_name, policy_description, policy_type, 
    industry_code, compliance_framework, jurisdiction, sla_tier,
    policy_content, enforcement_mode, fail_closed
) VALUES (
    1300,
    'SaaS SOX Revenue Recognition Policy',
    'Ensures SOX compliance for SaaS revenue recognition workflows',
    'compliance',
    'SaaS',
    'SOX',
    'US',
    'enterprise',
    '{
        "package": "saas.sox.revenue",
        "rules": {
            "require_dual_approval": {
                "condition": "input.revenue_amount > 10000",
                "action": "require_approval",
                "approver_roles": ["finance_manager", "cfo"]
            },
            "segregation_of_duties": {
                "condition": "true",
                "action": "enforce_sod",
                "incompatible_roles": [["revenue_calculator", "revenue_approver"]]
            },
            "audit_trail_required": {
                "condition": "true",
                "action": "log_evidence",
                "retention_years": 7
            }
        }
    }',
    'enforcing',
    true
) ON CONFLICT DO NOTHING;

-- Insert sample GDPR policy for SaaS
INSERT INTO policy_orch_policy (
    tenant_id, policy_name, policy_description, policy_type,
    industry_code, compliance_framework, jurisdiction, sla_tier,
    policy_content, enforcement_mode, fail_closed
) VALUES (
    1300,
    'SaaS GDPR Data Privacy Policy',
    'Ensures GDPR compliance for SaaS customer data processing',
    'compliance',
    'SaaS',
    'GDPR',
    'EU',
    'enterprise',
    '{
        "package": "saas.gdpr.privacy",
        "rules": {
            "consent_required": {
                "condition": "input.data_type == \"personal_data\"",
                "action": "require_consent",
                "consent_types": ["explicit", "legitimate_interest"]
            },
            "data_minimization": {
                "condition": "true",
                "action": "validate_necessity",
                "max_retention_days": 1095
            },
            "right_to_erasure": {
                "condition": "input.erasure_request == true",
                "action": "delete_data",
                "confirmation_required": true
            }
        }
    }',
    'enforcing',
    true
) ON CONFLICT DO NOTHING;

-- Insert sample SaaS business rules policy
INSERT INTO policy_orch_policy (
    tenant_id, policy_name, policy_description, policy_type,
    industry_code, compliance_framework, jurisdiction, sla_tier,
    policy_content, enforcement_mode, fail_closed
) VALUES (
    1300,
    'SaaS Subscription Lifecycle Policy',
    'Business rules for SaaS subscription lifecycle management',
    'business_rule',
    'SaaS',
    'SAAS_BUSINESS_RULES',
    'US',
    'standard',
    '{
        "package": "saas.business.subscription",
        "rules": {
            "churn_prevention": {
                "condition": "input.churn_risk_score > 0.8",
                "action": "trigger_retention_workflow",
                "escalation_required": true
            },
            "usage_validation": {
                "condition": "input.usage_overage > input.plan_limit",
                "action": "notify_billing",
                "auto_upgrade_threshold": 1.5
            },
            "contract_renewal": {
                "condition": "input.days_to_renewal <= 30",
                "action": "trigger_renewal_workflow",
                "early_renewal_discount": 0.05
            }
        }
    }',
    'enforcing',
    true
) ON CONFLICT DO NOTHING;

-- Add comments for documentation
COMMENT ON TABLE policy_orch_policy IS 'Task 16.1.1: Core policy metadata storage with industry and SLA context';
COMMENT ON TABLE policy_orch_version IS 'Task 16.1.2: Policy version tracking with lifecycle management';
COMMENT ON TABLE policy_orch_lifecycle IS 'Task 16.1.3: Policy lifecycle state machine (Draft → Published → Promoted → Retired)';
COMMENT ON TABLE policy_orch_enforcement_log IS 'Task 16.1.18: Evidence logging for policy enforcement decisions';
