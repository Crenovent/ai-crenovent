-- Approval System Database Schemas - Chapter 14.1
-- ================================================
-- Tasks 14.1-T12, 14.1-T13: Approval quorum rules and time-bound approvals
-- Enhanced schemas for approval state machine and policy enforcement

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- APPROVAL CHAIN CONFIGURATIONS
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_chain_configs (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chain_id VARCHAR(100) UNIQUE NOT NULL,
    tenant_id INTEGER,
    
    -- Configuration Details
    name VARCHAR(255) NOT NULL,
    description TEXT,
    industry_code VARCHAR(20) DEFAULT 'SaaS',
    compliance_frameworks TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Chain Configuration (JSON)
    config_data JSONB NOT NULL,
    
    -- Status & Lifecycle
    active BOOLEAN DEFAULT true,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_chain_config_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT valid_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'IT_Services', 'FinTech'))
);

-- =====================================================
-- ENHANCED APPROVAL REQUESTS
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_requests (
    request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Request Details
    requested_by_user_id INTEGER NOT NULL,
    approval_type VARCHAR(100) NOT NULL, -- 'workflow_approval', 'policy_exception', 'override_request', 'emergency_approval'
    business_justification TEXT NOT NULL,
    risk_level VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    financial_amount DECIMAL(15,2),
    
    -- Chain Configuration
    chain_config_id VARCHAR(100) NOT NULL,
    
    -- Context & Requirements
    context_data JSONB DEFAULT '{}'::jsonb,
    required_evidence TEXT[] DEFAULT ARRAY[]::TEXT[],
    compliance_frameworks TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- State Management
    approval_state JSONB NOT NULL, -- Current state, level, decisions, quorum status
    assigned_approvers JSONB DEFAULT '[]'::jsonb, -- Currently assigned approvers
    
    -- Timing & SLA
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Evidence & Compliance
    evidence_pack_id UUID,
    digital_signature TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_approval_request_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT fk_approval_request_user FOREIGN KEY (requested_by_user_id) REFERENCES users(user_id),
    CONSTRAINT valid_risk_level CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT valid_approval_type CHECK (approval_type IN ('workflow_approval', 'policy_exception', 'override_request', 'emergency_approval'))
);

-- =====================================================
-- APPROVAL DECISIONS
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_decisions (
    decision_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Approver Details
    approver_user_id INTEGER NOT NULL,
    approver_role VARCHAR(100) NOT NULL,
    approval_level VARCHAR(50) NOT NULL, -- 'l1_manager', 'l2_senior_manager', etc.
    
    -- Decision Details
    decision VARCHAR(20) NOT NULL, -- 'approve', 'reject', 'delegate', 'escalate'
    comments TEXT,
    conditions TEXT[] DEFAULT ARRAY[]::TEXT[],
    evidence_provided TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Timing
    decision_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Security & Audit
    digital_signature TEXT,
    ip_address INET,
    user_agent TEXT,
    
    -- Delegation (if applicable)
    delegated_to_user_id INTEGER,
    delegation_reason TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_decision_request FOREIGN KEY (request_id) REFERENCES approval_requests(request_id),
    CONSTRAINT fk_decision_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT fk_decision_approver FOREIGN KEY (approver_user_id) REFERENCES users(user_id),
    CONSTRAINT fk_decision_delegate FOREIGN KEY (delegated_to_user_id) REFERENCES users(user_id),
    CONSTRAINT valid_decision CHECK (decision IN ('approve', 'reject', 'delegate', 'escalate')),
    CONSTRAINT valid_approval_level CHECK (approval_level IN ('l1_manager', 'l2_senior_manager', 'l3_director', 'l4_vp', 'l5_c_suite'))
);

-- =====================================================
-- POLICY ENFORCEMENT LOG
-- =====================================================

CREATE TABLE IF NOT EXISTS policy_enforcement_log (
    enforcement_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    
    -- Enforcement Details
    approval_type VARCHAR(100) NOT NULL,
    result VARCHAR(20) NOT NULL, -- 'approved', 'denied', 'requires_override', 'escalated'
    violations JSONB DEFAULT '[]'::jsonb,
    
    -- Residency & Compliance
    residency_region VARCHAR(10) DEFAULT 'GLOBAL',
    compliance_frameworks TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Policy Context
    policy_pack_refs JSONB DEFAULT '[]'::jsonb,
    override_options JSONB DEFAULT '[]'::jsonb,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_policy_enforcement_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT fk_policy_enforcement_user FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT valid_enforcement_result CHECK (result IN ('approved', 'denied', 'requires_override', 'escalated')),
    CONSTRAINT valid_residency_region CHECK (residency_region IN ('EU', 'IN', 'US', 'APAC', 'GLOBAL'))
);

-- =====================================================
-- APPROVAL SLA MONITORING
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_sla_monitoring (
    monitor_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- SLA Configuration
    sla_hours INTEGER NOT NULL,
    escalation_threshold DECIMAL(3,2) DEFAULT 0.75, -- 75%
    
    -- Timing Tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    first_escalation_at TIMESTAMP WITH TIME ZONE,
    final_escalation_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Status Tracking
    current_elapsed_percentage DECIMAL(5,2) DEFAULT 0.00,
    escalation_count INTEGER DEFAULT 0,
    sla_breached BOOLEAN DEFAULT false,
    
    -- Notifications Sent
    notifications_sent JSONB DEFAULT '[]'::jsonb,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_sla_monitor_request FOREIGN KEY (request_id) REFERENCES approval_requests(request_id),
    CONSTRAINT fk_sla_monitor_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
);

-- =====================================================
-- APPROVAL DELEGATION RULES
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_delegation_rules (
    delegation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Delegation Configuration
    delegator_user_id INTEGER NOT NULL,
    delegate_user_id INTEGER NOT NULL,
    
    -- Scope & Permissions
    approval_types TEXT[] DEFAULT ARRAY[]::TEXT[],
    max_financial_amount DECIMAL(15,2),
    approval_levels TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Timing & Conditions
    effective_from TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    effective_until TIMESTAMP WITH TIME ZONE,
    conditions JSONB DEFAULT '{}'::jsonb,
    
    -- Status
    active BOOLEAN DEFAULT true,
    reason TEXT,
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_delegation_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT fk_delegation_delegator FOREIGN KEY (delegator_user_id) REFERENCES users(user_id),
    CONSTRAINT fk_delegation_delegate FOREIGN KEY (delegate_user_id) REFERENCES users(user_id),
    CONSTRAINT fk_delegation_creator FOREIGN KEY (created_by_user_id) REFERENCES users(user_id),
    CONSTRAINT valid_delegation_period CHECK (effective_until > effective_from)
);

-- =====================================================
-- APPROVAL WORKLOAD TRACKING
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_workload_tracking (
    tracking_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Workload Metrics
    pending_approvals INTEGER DEFAULT 0,
    completed_approvals_today INTEGER DEFAULT 0,
    completed_approvals_week INTEGER DEFAULT 0,
    avg_decision_time_hours DECIMAL(8,2) DEFAULT 0.00,
    
    -- Performance Metrics
    approval_rate DECIMAL(5,2) DEFAULT 0.00, -- Percentage of approvals vs rejections
    sla_compliance_rate DECIMAL(5,2) DEFAULT 100.00,
    escalation_rate DECIMAL(5,2) DEFAULT 0.00,
    
    -- Capacity & Availability
    max_concurrent_approvals INTEGER DEFAULT 10,
    availability_status VARCHAR(20) DEFAULT 'available', -- 'available', 'busy', 'unavailable'
    out_of_office_until TIMESTAMP WITH TIME ZONE,
    
    -- Tracking Period
    tracking_date DATE DEFAULT CURRENT_DATE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT fk_workload_user FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_workload_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
    CONSTRAINT valid_availability_status CHECK (availability_status IN ('available', 'busy', 'unavailable')),
    UNIQUE(user_id, tenant_id, tracking_date)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Approval Requests Indexes
CREATE INDEX IF NOT EXISTS idx_approval_requests_tenant_status ON approval_requests(tenant_id, (approval_state->>'current_state'));
CREATE INDEX IF NOT EXISTS idx_approval_requests_expires_at ON approval_requests(expires_at) WHERE (approval_state->>'current_state') IN ('pending', 'in_review');
CREATE INDEX IF NOT EXISTS idx_approval_requests_workflow ON approval_requests(workflow_id);
CREATE INDEX IF NOT EXISTS idx_approval_requests_requester ON approval_requests(requested_by_user_id, tenant_id);
CREATE INDEX IF NOT EXISTS idx_approval_requests_risk_level ON approval_requests(risk_level, tenant_id);

-- Approval Decisions Indexes
CREATE INDEX IF NOT EXISTS idx_approval_decisions_request ON approval_decisions(request_id);
CREATE INDEX IF NOT EXISTS idx_approval_decisions_approver ON approval_decisions(approver_user_id, tenant_id);
CREATE INDEX IF NOT EXISTS idx_approval_decisions_timestamp ON approval_decisions(decision_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_approval_decisions_level ON approval_decisions(approval_level, tenant_id);

-- Policy Enforcement Log Indexes
CREATE INDEX IF NOT EXISTS idx_policy_enforcement_workflow ON policy_enforcement_log(workflow_id);
CREATE INDEX IF NOT EXISTS idx_policy_enforcement_tenant_result ON policy_enforcement_log(tenant_id, result);
CREATE INDEX IF NOT EXISTS idx_policy_enforcement_created_at ON policy_enforcement_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_policy_enforcement_frameworks ON policy_enforcement_log USING GIN(compliance_frameworks);

-- SLA Monitoring Indexes
CREATE INDEX IF NOT EXISTS idx_sla_monitoring_request ON approval_sla_monitoring(request_id);
CREATE INDEX IF NOT EXISTS idx_sla_monitoring_tenant_breached ON approval_sla_monitoring(tenant_id, sla_breached);
CREATE INDEX IF NOT EXISTS idx_sla_monitoring_escalation ON approval_sla_monitoring(first_escalation_at) WHERE first_escalation_at IS NOT NULL;

-- Delegation Rules Indexes
CREATE INDEX IF NOT EXISTS idx_delegation_rules_delegator ON approval_delegation_rules(delegator_user_id, tenant_id) WHERE active = true;
CREATE INDEX IF NOT EXISTS idx_delegation_rules_delegate ON approval_delegation_rules(delegate_user_id, tenant_id) WHERE active = true;
CREATE INDEX IF NOT EXISTS idx_delegation_rules_effective ON approval_delegation_rules(effective_from, effective_until) WHERE active = true;

-- Workload Tracking Indexes
CREATE INDEX IF NOT EXISTS idx_workload_tracking_user_date ON approval_workload_tracking(user_id, tracking_date);
CREATE INDEX IF NOT EXISTS idx_workload_tracking_tenant_availability ON approval_workload_tracking(tenant_id, availability_status);
CREATE INDEX IF NOT EXISTS idx_workload_tracking_pending ON approval_workload_tracking(pending_approvals DESC) WHERE availability_status = 'available';

-- =====================================================
-- ROW LEVEL SECURITY POLICIES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE approval_chain_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy_enforcement_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_sla_monitoring ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_delegation_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE approval_workload_tracking ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation

-- Approval Chain Configs
CREATE POLICY approval_chain_configs_tenant_isolation ON approval_chain_configs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER OR tenant_id IS NULL);

-- Approval Requests
CREATE POLICY approval_requests_tenant_isolation ON approval_requests
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- Approval Decisions
CREATE POLICY approval_decisions_tenant_isolation ON approval_decisions
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- Policy Enforcement Log
CREATE POLICY policy_enforcement_log_tenant_isolation ON policy_enforcement_log
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- SLA Monitoring
CREATE POLICY approval_sla_monitoring_tenant_isolation ON approval_sla_monitoring
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- Delegation Rules
CREATE POLICY approval_delegation_rules_tenant_isolation ON approval_delegation_rules
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- Workload Tracking
CREATE POLICY approval_workload_tracking_tenant_isolation ON approval_workload_tracking
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- =====================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================

-- Update last_updated timestamp on approval_requests
CREATE OR REPLACE FUNCTION update_approval_request_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_approval_request_timestamp
    BEFORE UPDATE ON approval_requests
    FOR EACH ROW
    EXECUTE FUNCTION update_approval_request_timestamp();

-- Update workload tracking when decisions are made
CREATE OR REPLACE FUNCTION update_workload_on_decision()
RETURNS TRIGGER AS $$
BEGIN
    -- Update workload tracking for the approver
    INSERT INTO approval_workload_tracking (
        user_id, tenant_id, tracking_date, completed_approvals_today, last_updated
    ) VALUES (
        NEW.approver_user_id, NEW.tenant_id, CURRENT_DATE, 1, NOW()
    )
    ON CONFLICT (user_id, tenant_id, tracking_date)
    DO UPDATE SET
        completed_approvals_today = approval_workload_tracking.completed_approvals_today + 1,
        last_updated = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_workload_on_decision
    AFTER INSERT ON approval_decisions
    FOR EACH ROW
    EXECUTE FUNCTION update_workload_on_decision();

-- =====================================================
-- INITIAL DATA SEEDING
-- =====================================================

-- Insert default approval chain configurations
INSERT INTO approval_chain_configs (chain_id, name, description, config_data, active) VALUES
('standard_workflow', 'Standard Workflow Approval', 'Default approval chain for regular workflows', 
 '{"levels": [{"level": "l1_manager", "roles": ["sales_manager", "revops_manager"], "quorum_type": "minimum_count", "quorum_value": 1, "sla_hours": 24}], "quorum_rules": {"overall_type": "simple_majority", "level_progression": "sequential"}, "sla_hours": 72, "auto_escalate": true}', 
 true),

('high_value_financial', 'High Value Financial Approval', 'Approval chain for high-value financial transactions',
 '{"levels": [{"level": "l2_senior_manager", "roles": ["finance_manager"], "quorum_type": "minimum_count", "quorum_value": 1, "sla_hours": 12}, {"level": "l3_director", "roles": ["finance_director"], "quorum_type": "unanimous", "quorum_value": 1.0, "sla_hours": 24}, {"level": "l4_vp", "roles": ["cfo"], "quorum_type": "minimum_count", "quorum_value": 1, "sla_hours": 48}], "quorum_rules": {"overall_type": "unanimous", "level_progression": "sequential", "require_all_levels": true}, "sla_hours": 84, "auto_escalate": true}',
 true),

('regulatory_compliance', 'Regulatory Compliance Approval', 'Approval chain for regulatory compliance matters',
 '{"levels": [{"level": "l2_senior_manager", "roles": ["compliance_officer"], "quorum_type": "super_majority", "quorum_value": 0.67, "sla_hours": 8}, {"level": "l3_director", "roles": ["compliance_director", "legal_director"], "quorum_type": "unanimous", "quorum_value": 1.0, "sla_hours": 16}, {"level": "l5_c_suite", "roles": ["ceo"], "quorum_type": "minimum_count", "quorum_value": 1, "sla_hours": 24}], "quorum_rules": {"overall_type": "unanimous", "level_progression": "sequential", "require_all_levels": true}, "sla_hours": 48, "auto_escalate": true}',
 true)

ON CONFLICT (chain_id) DO NOTHING;

-- Create indexes on JSONB fields for better performance
CREATE INDEX IF NOT EXISTS idx_approval_requests_state_gin ON approval_requests USING GIN(approval_state);
CREATE INDEX IF NOT EXISTS idx_approval_requests_context_gin ON approval_requests USING GIN(context_data);
CREATE INDEX IF NOT EXISTS idx_policy_enforcement_violations_gin ON policy_enforcement_log USING GIN(violations);
CREATE INDEX IF NOT EXISTS idx_chain_configs_data_gin ON approval_chain_configs USING GIN(config_data);
