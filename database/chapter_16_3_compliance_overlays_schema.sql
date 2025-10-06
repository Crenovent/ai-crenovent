-- ============================================================================
-- CHAPTER 16.3: COMPLIANCE OVERLAYS DATABASE SCHEMA
-- ============================================================================
-- Tasks 16.3.1-16.3.11: Compliance overlay infrastructure for multi-industry governance
-- Implements SOX, GDPR, RBI, HIPAA, NAIC compliance overlays with automated enforcement

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- TASK 16.3.1: Provision compliance_overlay table
-- ============================================================================

-- Core compliance overlay definitions
CREATE TABLE IF NOT EXISTS compliance_overlay (
    overlay_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Overlay Identification
    overlay_name VARCHAR(255) NOT NULL,
    overlay_description TEXT,
    compliance_framework VARCHAR(50) NOT NULL, -- 'SOX', 'GDPR', 'RBI', 'HIPAA', 'NAIC'
    
    -- Industry & Jurisdiction
    industry_code VARCHAR(20) NOT NULL, -- 'SaaS', 'Banking', 'Insurance'
    jurisdiction VARCHAR(10) NOT NULL DEFAULT 'US', -- 'US', 'EU', 'IN'
    regulatory_authority VARCHAR(100), -- 'SEC', 'GDPR_DPA', 'RBI', 'HHS', 'NAIC'
    
    -- Overlay Configuration
    overlay_version VARCHAR(20) NOT NULL DEFAULT '1.0',
    overlay_config JSONB NOT NULL, -- Compliance rules and configurations
    enforcement_rules JSONB NOT NULL, -- How to enforce compliance
    
    -- Applicability
    applicable_workflows TEXT[] DEFAULT ARRAY[], -- Which workflows this applies to
    applicable_automation_types TEXT[] DEFAULT ARRAY['RBA', 'RBIA', 'AALA'],
    
    -- Enforcement Settings
    enforcement_mode VARCHAR(20) NOT NULL DEFAULT 'enforcing', -- 'enforcing', 'monitoring', 'disabled'
    fail_closed BOOLEAN DEFAULT true,
    override_allowed BOOLEAN DEFAULT false, -- Most compliance overlays don't allow overrides
    
    -- Lifecycle
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- 'active', 'deprecated', 'retired'
    effective_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    effective_until TIMESTAMPTZ,
    
    -- Audit & Governance
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER,
    approved_by_user_id INTEGER,
    approval_required BOOLEAN DEFAULT true,
    
    -- Constraints
    CONSTRAINT valid_compliance_framework CHECK (compliance_framework IN ('SOX', 'GDPR', 'RBI', 'DPDP', 'HIPAA', 'NAIC', 'AML_KYC', 'PCI_DSS')),
    CONSTRAINT valid_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'FinTech', 'E-commerce', 'IT_Services', 'Healthcare')),
    CONSTRAINT valid_enforcement_mode CHECK (enforcement_mode IN ('enforcing', 'monitoring', 'disabled')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'deprecated', 'retired')),
    
    -- Foreign Keys
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    
    -- Unique constraint
    UNIQUE (tenant_id, overlay_name, overlay_version)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_compliance_overlay_tenant_id ON compliance_overlay(tenant_id);
CREATE INDEX IF NOT EXISTS idx_compliance_overlay_framework ON compliance_overlay(compliance_framework);
CREATE INDEX IF NOT EXISTS idx_compliance_overlay_industry ON compliance_overlay(industry_code);
CREATE INDEX IF NOT EXISTS idx_compliance_overlay_status ON compliance_overlay(status);
CREATE INDEX IF NOT EXISTS idx_compliance_overlay_effective ON compliance_overlay(effective_from, effective_until);

-- Row Level Security
ALTER TABLE compliance_overlay ENABLE ROW LEVEL SECURITY;
CREATE POLICY compliance_overlay_rls_policy ON compliance_overlay
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- TASK 16.3.2: Provision compliance_overlay_rule table
-- ============================================================================

-- Individual compliance rules within overlays
CREATE TABLE IF NOT EXISTS compliance_overlay_rule (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    overlay_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Rule Identification
    rule_name VARCHAR(255) NOT NULL,
    rule_description TEXT,
    rule_category VARCHAR(100), -- 'data_protection', 'financial_reporting', 'audit_trail', 'access_control'
    
    -- Regulatory Reference
    regulatory_section VARCHAR(100), -- e.g., 'SOX-404', 'GDPR-Article-6', 'RBI-Digital-Lending'
    regulatory_requirement TEXT,
    
    -- Rule Configuration
    rule_condition JSONB NOT NULL, -- When this rule applies
    rule_action JSONB NOT NULL, -- What action to take
    rule_parameters JSONB DEFAULT '{}', -- Rule-specific parameters
    
    -- Enforcement
    enforcement_level VARCHAR(20) NOT NULL DEFAULT 'mandatory', -- 'mandatory', 'recommended', 'optional'
    violation_severity VARCHAR(20) NOT NULL DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    
    -- Automation Integration
    automation_hook VARCHAR(100), -- Which automation point this hooks into
    execution_order INTEGER DEFAULT 100, -- Order of execution (lower = earlier)
    
    -- Rule Lifecycle
    rule_status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_enforcement_level CHECK (enforcement_level IN ('mandatory', 'recommended', 'optional')),
    CONSTRAINT valid_violation_severity CHECK (violation_severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT valid_rule_status CHECK (rule_status IN ('active', 'inactive', 'deprecated')),
    
    -- Foreign Keys
    FOREIGN KEY (overlay_id) REFERENCES compliance_overlay(overlay_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id),
    
    -- Unique constraint
    UNIQUE (overlay_id, rule_name)
);

-- Indexes for compliance rules
CREATE INDEX IF NOT EXISTS idx_compliance_rule_overlay_id ON compliance_overlay_rule(overlay_id);
CREATE INDEX IF NOT EXISTS idx_compliance_rule_tenant_id ON compliance_overlay_rule(tenant_id);
CREATE INDEX IF NOT EXISTS idx_compliance_rule_category ON compliance_overlay_rule(rule_category);
CREATE INDEX IF NOT EXISTS idx_compliance_rule_enforcement ON compliance_overlay_rule(enforcement_level);
CREATE INDEX IF NOT EXISTS idx_compliance_rule_severity ON compliance_overlay_rule(violation_severity);
CREATE INDEX IF NOT EXISTS idx_compliance_rule_order ON compliance_overlay_rule(execution_order);

-- Row Level Security
ALTER TABLE compliance_overlay_rule ENABLE ROW LEVEL SECURITY;
CREATE POLICY compliance_overlay_rule_rls_policy ON compliance_overlay_rule
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- TASK 16.3.3: Provision compliance_overlay_enforcement table
-- ============================================================================

-- Track compliance overlay enforcement events
CREATE TABLE IF NOT EXISTS compliance_overlay_enforcement (
    enforcement_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    overlay_id UUID NOT NULL,
    rule_id UUID,
    tenant_id INTEGER NOT NULL,
    
    -- Enforcement Context
    workflow_id UUID,
    execution_id UUID,
    automation_type VARCHAR(10), -- 'RBA', 'RBIA', 'AALA'
    
    -- Enforcement Result
    enforcement_result VARCHAR(20) NOT NULL, -- 'compliant', 'violation', 'warning', 'exception'
    violation_details JSONB DEFAULT '{}',
    compliance_score DECIMAL(5,2), -- 0.00 to 100.00
    
    -- Violation Information
    violation_count INTEGER DEFAULT 0,
    critical_violations INTEGER DEFAULT 0,
    high_violations INTEGER DEFAULT 0,
    medium_violations INTEGER DEFAULT 0,
    low_violations INTEGER DEFAULT 0,
    
    -- Remediation
    remediation_required BOOLEAN DEFAULT false,
    remediation_actions JSONB DEFAULT '[]',
    remediation_deadline TIMESTAMPTZ,
    remediation_status VARCHAR(20) DEFAULT 'pending',
    
    -- Regulatory Reporting
    regulatory_notification_required BOOLEAN DEFAULT false,
    regulatory_notification_sent BOOLEAN DEFAULT false,
    regulatory_notification_date TIMESTAMPTZ,
    regulatory_reference_number VARCHAR(100),
    
    -- Performance Metrics
    enforcement_time_ms INTEGER,
    rules_evaluated INTEGER DEFAULT 0,
    rules_passed INTEGER DEFAULT 0,
    rules_failed INTEGER DEFAULT 0,
    
    -- Evidence & Audit
    evidence_pack_id UUID,
    audit_trail JSONB DEFAULT '{}',
    
    -- Timestamps
    enforced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_enforcement_result CHECK (enforcement_result IN ('compliant', 'violation', 'warning', 'exception')),
    CONSTRAINT valid_automation_type CHECK (automation_type IN ('RBA', 'RBIA', 'AALA')),
    CONSTRAINT valid_remediation_status CHECK (remediation_status IN ('pending', 'in_progress', 'completed', 'overdue')),
    
    -- Foreign Keys
    FOREIGN KEY (overlay_id) REFERENCES compliance_overlay(overlay_id),
    FOREIGN KEY (rule_id) REFERENCES compliance_overlay_rule(rule_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Indexes for enforcement tracking
CREATE INDEX IF NOT EXISTS idx_compliance_enforcement_overlay ON compliance_overlay_enforcement(overlay_id);
CREATE INDEX IF NOT EXISTS idx_compliance_enforcement_tenant ON compliance_overlay_enforcement(tenant_id);
CREATE INDEX IF NOT EXISTS idx_compliance_enforcement_result ON compliance_overlay_enforcement(enforcement_result);
CREATE INDEX IF NOT EXISTS idx_compliance_enforcement_timestamp ON compliance_overlay_enforcement(enforced_at);
CREATE INDEX IF NOT EXISTS idx_compliance_enforcement_workflow ON compliance_overlay_enforcement(workflow_id);

-- Row Level Security
ALTER TABLE compliance_overlay_enforcement ENABLE ROW LEVEL SECURITY;
CREATE POLICY compliance_overlay_enforcement_rls_policy ON compliance_overlay_enforcement
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- TASK 16.3.4: Provision compliance_overlay_exception table
-- ============================================================================

-- Track compliance exceptions and waivers
CREATE TABLE IF NOT EXISTS compliance_overlay_exception (
    exception_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    overlay_id UUID NOT NULL,
    rule_id UUID,
    tenant_id INTEGER NOT NULL,
    
    -- Exception Details
    exception_type VARCHAR(50) NOT NULL, -- 'temporary_waiver', 'permanent_exception', 'emergency_bypass'
    exception_reason TEXT NOT NULL,
    business_justification TEXT NOT NULL,
    
    -- Exception Scope
    workflow_ids UUID[], -- Specific workflows this exception applies to
    user_ids INTEGER[], -- Specific users this exception applies to
    exception_conditions JSONB DEFAULT '{}', -- Conditions under which exception applies
    
    -- Approval Workflow
    approval_required BOOLEAN DEFAULT true,
    approval_status VARCHAR(20) DEFAULT 'pending',
    approved_by_user_id INTEGER,
    approved_at TIMESTAMPTZ,
    approval_notes TEXT,
    
    -- Exception Lifecycle
    effective_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    effective_until TIMESTAMPTZ,
    auto_expire BOOLEAN DEFAULT true,
    
    -- Risk Assessment
    risk_level VARCHAR(20) NOT NULL DEFAULT 'medium',
    risk_assessment TEXT,
    mitigation_controls JSONB DEFAULT '[]',
    
    -- Monitoring & Review
    review_required BOOLEAN DEFAULT true,
    review_frequency_days INTEGER DEFAULT 30,
    last_reviewed_at TIMESTAMPTZ,
    next_review_due TIMESTAMPTZ,
    
    -- Usage Tracking
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    
    -- Exception Status
    exception_status VARCHAR(20) NOT NULL DEFAULT 'active',
    revoked_at TIMESTAMPTZ,
    revoked_by_user_id INTEGER,
    revocation_reason TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_exception_type CHECK (exception_type IN ('temporary_waiver', 'permanent_exception', 'emergency_bypass')),
    CONSTRAINT valid_exception_approval_status CHECK (approval_status IN ('pending', 'approved', 'rejected')),
    CONSTRAINT valid_exception_risk_level CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT valid_exception_status CHECK (exception_status IN ('active', 'expired', 'revoked', 'suspended')),
    
    -- Foreign Keys
    FOREIGN KEY (overlay_id) REFERENCES compliance_overlay(overlay_id),
    FOREIGN KEY (rule_id) REFERENCES compliance_overlay_rule(rule_id),
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Indexes for exception tracking
CREATE INDEX IF NOT EXISTS idx_compliance_exception_overlay ON compliance_overlay_exception(overlay_id);
CREATE INDEX IF NOT EXISTS idx_compliance_exception_tenant ON compliance_overlay_exception(tenant_id);
CREATE INDEX IF NOT EXISTS idx_compliance_exception_status ON compliance_overlay_exception(exception_status);
CREATE INDEX IF NOT EXISTS idx_compliance_exception_effective ON compliance_overlay_exception(effective_from, effective_until);
CREATE INDEX IF NOT EXISTS idx_compliance_exception_review ON compliance_overlay_exception(next_review_due);

-- Row Level Security
ALTER TABLE compliance_overlay_exception ENABLE ROW LEVEL SECURITY;
CREATE POLICY compliance_overlay_exception_rls_policy ON compliance_overlay_exception
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- TASK 16.3.5: Provision compliance_overlay_report table
-- ============================================================================

-- Compliance reporting and analytics
CREATE TABLE IF NOT EXISTS compliance_overlay_report (
    report_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Report Configuration
    report_name VARCHAR(255) NOT NULL,
    report_type VARCHAR(50) NOT NULL, -- 'regulatory_filing', 'internal_audit', 'executive_summary', 'violation_report'
    compliance_framework VARCHAR(50) NOT NULL,
    
    -- Report Period
    report_period_start TIMESTAMPTZ NOT NULL,
    report_period_end TIMESTAMPTZ NOT NULL,
    
    -- Report Content
    report_data JSONB NOT NULL, -- Structured report data
    report_summary JSONB DEFAULT '{}', -- Executive summary
    
    -- Compliance Metrics
    total_evaluations INTEGER DEFAULT 0,
    compliant_evaluations INTEGER DEFAULT 0,
    violation_count INTEGER DEFAULT 0,
    exception_count INTEGER DEFAULT 0,
    compliance_rate DECIMAL(5,2), -- Percentage
    
    -- Violation Breakdown
    critical_violations INTEGER DEFAULT 0,
    high_violations INTEGER DEFAULT 0,
    medium_violations INTEGER DEFAULT 0,
    low_violations INTEGER DEFAULT 0,
    
    -- Regulatory Information
    regulatory_filing_required BOOLEAN DEFAULT false,
    regulatory_filing_deadline TIMESTAMPTZ,
    regulatory_filing_status VARCHAR(20) DEFAULT 'pending',
    regulatory_reference_number VARCHAR(100),
    
    -- Report Status
    report_status VARCHAR(20) NOT NULL DEFAULT 'draft', -- 'draft', 'review', 'approved', 'filed', 'archived'
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    generated_by_user_id INTEGER,
    approved_by_user_id INTEGER,
    approved_at TIMESTAMPTZ,
    
    -- File References
    report_file_path VARCHAR(500), -- Path to generated report file
    report_file_format VARCHAR(20) DEFAULT 'pdf', -- 'pdf', 'excel', 'json'
    
    -- Constraints
    CONSTRAINT valid_report_type CHECK (report_type IN ('regulatory_filing', 'internal_audit', 'executive_summary', 'violation_report', 'exception_report')),
    CONSTRAINT valid_report_filing_status CHECK (regulatory_filing_status IN ('pending', 'filed', 'accepted', 'rejected')),
    CONSTRAINT valid_report_status CHECK (report_status IN ('draft', 'review', 'approved', 'filed', 'archived')),
    
    -- Foreign Keys
    FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
);

-- Indexes for reporting
CREATE INDEX IF NOT EXISTS idx_compliance_report_tenant ON compliance_overlay_report(tenant_id);
CREATE INDEX IF NOT EXISTS idx_compliance_report_framework ON compliance_overlay_report(compliance_framework);
CREATE INDEX IF NOT EXISTS idx_compliance_report_period ON compliance_overlay_report(report_period_start, report_period_end);
CREATE INDEX IF NOT EXISTS idx_compliance_report_status ON compliance_overlay_report(report_status);
CREATE INDEX IF NOT EXISTS idx_compliance_report_filing ON compliance_overlay_report(regulatory_filing_deadline);

-- Row Level Security
ALTER TABLE compliance_overlay_report ENABLE ROW LEVEL SECURITY;
CREATE POLICY compliance_overlay_report_rls_policy ON compliance_overlay_report
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- ============================================================================
-- SAMPLE DATA INSERTION (SaaS Industry Focus)
-- ============================================================================

-- Insert SOX compliance overlay for SaaS
INSERT INTO compliance_overlay (
    tenant_id, overlay_name, overlay_description, compliance_framework,
    industry_code, jurisdiction, regulatory_authority, overlay_config, enforcement_rules
) VALUES (
    1300,
    'SOX SaaS Revenue Recognition Overlay',
    'Sarbanes-Oxley compliance overlay for SaaS revenue recognition workflows',
    'SOX',
    'SaaS',
    'US',
    'SEC',
    '{
        "sox_sections": ["302", "404", "409"],
        "materiality_threshold": 100000,
        "dual_approval_required": true,
        "segregation_of_duties": true,
        "audit_trail_retention_years": 7,
        "quarterly_certification_required": true
    }',
    '{
        "enforcement_points": ["workflow_start", "decision_points", "workflow_completion"],
        "violation_actions": ["block_execution", "require_approval", "log_violation"],
        "escalation_rules": {
            "critical_violations": "immediate_escalation",
            "material_amounts": "cfo_approval_required"
        }
    }'
) ON CONFLICT DO NOTHING;

-- Insert GDPR compliance overlay for SaaS
INSERT INTO compliance_overlay (
    tenant_id, overlay_name, overlay_description, compliance_framework,
    industry_code, jurisdiction, regulatory_authority, overlay_config, enforcement_rules
) VALUES (
    1300,
    'GDPR SaaS Data Privacy Overlay',
    'GDPR compliance overlay for SaaS customer data processing',
    'GDPR',
    'SaaS',
    'EU',
    'GDPR_DPA',
    '{
        "gdpr_articles": ["6", "7", "17", "20", "25"],
        "consent_management": true,
        "data_minimization": true,
        "retention_limits": {
            "personal_data": 1095,
            "marketing_data": 730,
            "analytics_data": 365
        },
        "right_to_erasure": true,
        "data_portability": true,
        "privacy_by_design": true
    }',
    '{
        "enforcement_points": ["data_collection", "data_processing", "data_storage", "data_deletion"],
        "violation_actions": ["block_processing", "require_consent", "anonymize_data"],
        "breach_notification": {
            "dpa_notification_hours": 72,
            "individual_notification_required": true
        }
    }'
) ON CONFLICT DO NOTHING;

-- Insert sample SOX rules
INSERT INTO compliance_overlay_rule (
    overlay_id, tenant_id, rule_name, rule_description, rule_category,
    regulatory_section, rule_condition, rule_action, enforcement_level, violation_severity
) VALUES (
    (SELECT overlay_id FROM compliance_overlay WHERE overlay_name = 'SOX SaaS Revenue Recognition Overlay' AND tenant_id = 1300),
    1300,
    'Dual Approval for Material Revenue',
    'Require dual approval for revenue recognition over materiality threshold',
    'financial_reporting',
    'SOX-404',
    '{"condition": "revenue_amount > 100000"}',
    '{"action": "require_dual_approval", "approvers": ["finance_manager", "cfo"]}',
    'mandatory',
    'high'
) ON CONFLICT DO NOTHING;

INSERT INTO compliance_overlay_rule (
    overlay_id, tenant_id, rule_name, rule_description, rule_category,
    regulatory_section, rule_condition, rule_action, enforcement_level, violation_severity
) VALUES (
    (SELECT overlay_id FROM compliance_overlay WHERE overlay_name = 'SOX SaaS Revenue Recognition Overlay' AND tenant_id = 1300),
    1300,
    'Segregation of Duties',
    'Enforce segregation of duties between revenue calculation and approval',
    'access_control',
    'SOX-404',
    '{"condition": "true"}',
    '{"action": "enforce_sod", "incompatible_roles": [["revenue_calculator", "revenue_approver"]]}',
    'mandatory',
    'critical'
) ON CONFLICT DO NOTHING;

-- Insert sample GDPR rules
INSERT INTO compliance_overlay_rule (
    overlay_id, tenant_id, rule_name, rule_description, rule_category,
    regulatory_section, rule_condition, rule_action, enforcement_level, violation_severity
) VALUES (
    (SELECT overlay_id FROM compliance_overlay WHERE overlay_name = 'GDPR SaaS Data Privacy Overlay' AND tenant_id = 1300),
    1300,
    'Consent Validation',
    'Validate consent before processing personal data',
    'data_protection',
    'GDPR-Article-6',
    '{"condition": "data_type == \"personal_data\""}',
    '{"action": "validate_consent", "consent_types": ["explicit", "legitimate_interest"]}',
    'mandatory',
    'high'
) ON CONFLICT DO NOTHING;

INSERT INTO compliance_overlay_rule (
    overlay_id, tenant_id, rule_name, rule_description, rule_category,
    regulatory_section, rule_condition, rule_action, enforcement_level, violation_severity
) VALUES (
    (SELECT overlay_id FROM compliance_overlay WHERE overlay_name = 'GDPR SaaS Data Privacy Overlay' AND tenant_id = 1300),
    1300,
    'Data Retention Limits',
    'Enforce data retention limits based on data type',
    'data_protection',
    'GDPR-Article-5',
    '{"condition": "data_age_days > retention_limit"}',
    '{"action": "delete_data", "confirmation_required": true}',
    'mandatory',
    'medium'
) ON CONFLICT DO NOTHING;

-- Add comments for documentation
COMMENT ON TABLE compliance_overlay IS 'Task 16.3.1: Core compliance overlay definitions for multi-industry governance';
COMMENT ON TABLE compliance_overlay_rule IS 'Task 16.3.2: Individual compliance rules within overlays';
COMMENT ON TABLE compliance_overlay_enforcement IS 'Task 16.3.3: Compliance overlay enforcement tracking';
COMMENT ON TABLE compliance_overlay_exception IS 'Task 16.3.4: Compliance exceptions and waivers management';
COMMENT ON TABLE compliance_overlay_report IS 'Task 16.3.5: Compliance reporting and regulatory filing';
