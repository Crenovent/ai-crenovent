-- Chapter 15.2 Fallback & Escalation Database Schemas
-- ====================================================
-- Tasks 15.2-T21, T22: Fallback and escalation evidence schemas

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- FALLBACK POLICIES (Task 15.2-T09)
-- =====================================================

CREATE TABLE IF NOT EXISTS fallback_policies (
    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Policy configuration
    industry_code VARCHAR(50) NOT NULL DEFAULT 'SaaS',
    
    -- Fallback rules
    fallback_rules JSONB NOT NULL DEFAULT '{}',
    data_source_priorities JSONB NOT NULL DEFAULT '[]',
    cache_retention_hours INTEGER NOT NULL DEFAULT 24,
    
    -- Policy metadata
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech')),
    CONSTRAINT chk_cache_retention CHECK (cache_retention_hours > 0 AND cache_retention_hours <= 168) -- Max 1 week
);

-- RLS for fallback policies
ALTER TABLE fallback_policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY fallback_policies_tenant_isolation ON fallback_policies
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_fallback_policies_tenant_active ON fallback_policies(tenant_id) WHERE active = true;
CREATE INDEX idx_fallback_policies_industry ON fallback_policies(tenant_id, industry_code);

-- =====================================================
-- ESCALATION POLICIES (Task 15.2-T10, T17, T18, T19)
-- =====================================================

CREATE TABLE IF NOT EXISTS escalation_policies (
    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Policy configuration
    industry_code VARCHAR(50) NOT NULL DEFAULT 'SaaS',
    
    -- Escalation configuration
    escalation_chains JSONB NOT NULL DEFAULT '{}',
    sla_timers JSONB NOT NULL DEFAULT '{}',
    
    -- Residency configuration (Task 15.2-T20)
    residency_rules JSONB DEFAULT '{}',
    cross_border_restrictions JSONB DEFAULT '{}',
    
    -- Policy metadata
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_escalation_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech'))
);

-- RLS for escalation policies
ALTER TABLE escalation_policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY escalation_policies_tenant_isolation ON escalation_policies
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_escalation_policies_tenant_active ON escalation_policies(tenant_id) WHERE active = true;
CREATE INDEX idx_escalation_policies_industry ON escalation_policies(tenant_id, industry_code);

-- =====================================================
-- FALLBACK REQUESTS (Task 15.2-T21)
-- =====================================================

CREATE TABLE IF NOT EXISTS fallback_requests (
    fallback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Request context
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255) NOT NULL,
    user_id INTEGER NOT NULL,
    
    -- Fallback details
    fallback_type VARCHAR(50) NOT NULL,
    trigger_reason TEXT NOT NULL,
    error_context JSONB DEFAULT '{}',
    
    -- Fallback configuration
    alternate_sources JSONB DEFAULT '[]',
    cache_key VARCHAR(255),
    approval_chain JSONB DEFAULT '[]',
    
    -- Fallback result
    fallback_result JSONB,
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    
    -- Evidence tracking
    evidence_pack_id UUID,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_fallback_type CHECK (fallback_type IN ('data_fallback', 'system_fallback', 'manual_fallback', 'cached_fallback', 'suspension_fallback', 'termination_fallback')),
    CONSTRAINT chk_fallback_status CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled'))
);

-- RLS for fallback requests
ALTER TABLE fallback_requests ENABLE ROW LEVEL SECURITY;
CREATE POLICY fallback_requests_tenant_isolation ON fallback_requests
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_fallback_requests_tenant_workflow ON fallback_requests(tenant_id, workflow_id);
CREATE INDEX idx_fallback_requests_tenant_status ON fallback_requests(tenant_id, status);
CREATE INDEX idx_fallback_requests_type ON fallback_requests(tenant_id, fallback_type);
CREATE INDEX idx_fallback_requests_created ON fallback_requests(tenant_id, created_at DESC);

-- =====================================================
-- ESCALATION REQUESTS (Task 15.2-T22)
-- =====================================================

CREATE TABLE IF NOT EXISTS escalation_requests (
    escalation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Request context
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255),
    user_id INTEGER NOT NULL,
    
    -- Escalation details
    severity VARCHAR(10) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    error_context JSONB DEFAULT '{}',
    business_impact TEXT,
    
    -- Routing
    target_persona VARCHAR(50),
    escalation_chain JSONB NOT NULL DEFAULT '[]',
    assigned_to VARCHAR(255),
    
    -- SLA configuration
    sla_hours INTEGER NOT NULL DEFAULT 24,
    sla_deadline TIMESTAMP WITH TIME ZONE NOT NULL,
    auto_escalate BOOLEAN NOT NULL DEFAULT true,
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    resolved_by INTEGER,
    resolution_notes TEXT,
    
    -- Evidence tracking
    evidence_pack_id UUID,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_escalation_severity CHECK (severity IN ('P0', 'P1', 'P2', 'P3')),
    CONSTRAINT chk_escalation_status CHECK (status IN ('open', 'acknowledged', 'in_progress', 'resolved', 'closed', 'escalated')),
    CONSTRAINT chk_escalation_persona CHECK (target_persona IS NULL OR target_persona IN ('ops', 'compliance', 'finance', 'executive', 'legal', 'security')),
    CONSTRAINT chk_sla_hours CHECK (sla_hours > 0 AND sla_hours <= 168), -- Max 1 week
    CONSTRAINT chk_sla_deadline CHECK (sla_deadline > created_at)
);

-- RLS for escalation requests
ALTER TABLE escalation_requests ENABLE ROW LEVEL SECURITY;
CREATE POLICY escalation_requests_tenant_isolation ON escalation_requests
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_escalation_requests_tenant_workflow ON escalation_requests(tenant_id, workflow_id);
CREATE INDEX idx_escalation_requests_tenant_status ON escalation_requests(tenant_id, status);
CREATE INDEX idx_escalation_requests_severity ON escalation_requests(tenant_id, severity);
CREATE INDEX idx_escalation_requests_assigned ON escalation_requests(tenant_id, assigned_to) WHERE assigned_to IS NOT NULL;
CREATE INDEX idx_escalation_requests_sla_deadline ON escalation_requests(tenant_id, sla_deadline) WHERE status NOT IN ('resolved', 'closed');
CREATE INDEX idx_escalation_requests_created ON escalation_requests(tenant_id, created_at DESC);

-- =====================================================
-- ESCALATION STATE TRANSITIONS (Task 15.2-T16)
-- =====================================================

CREATE TABLE IF NOT EXISTS escalation_state_transitions (
    transition_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    escalation_id UUID NOT NULL REFERENCES escalation_requests(escalation_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Transition details
    from_status VARCHAR(20) NOT NULL,
    to_status VARCHAR(20) NOT NULL,
    
    -- Change tracking
    changed_by INTEGER NOT NULL,
    change_reason TEXT,
    
    -- Metadata
    transition_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_transition_from_status CHECK (from_status IN ('open', 'acknowledged', 'in_progress', 'resolved', 'closed', 'escalated')),
    CONSTRAINT chk_transition_to_status CHECK (to_status IN ('open', 'acknowledged', 'in_progress', 'resolved', 'closed', 'escalated'))
);

-- RLS for state transitions
ALTER TABLE escalation_state_transitions ENABLE ROW LEVEL SECURITY;
CREATE POLICY escalation_state_transitions_tenant_isolation ON escalation_state_transitions
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_escalation_transitions_escalation_id ON escalation_state_transitions(escalation_id, created_at DESC);
CREATE INDEX idx_escalation_transitions_tenant_user ON escalation_state_transitions(tenant_id, changed_by);
CREATE INDEX idx_escalation_transitions_status ON escalation_state_transitions(tenant_id, from_status, to_status);

-- =====================================================
-- WORKFLOW SUSPENSIONS (Task 15.2-T14)
-- =====================================================

CREATE TABLE IF NOT EXISTS workflow_suspensions (
    suspension_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255) NOT NULL,
    tenant_id INTEGER NOT NULL,
    fallback_id UUID REFERENCES fallback_requests(fallback_id) ON DELETE CASCADE,
    
    -- Suspension details
    suspension_reason TEXT NOT NULL,
    suspension_context JSONB DEFAULT '{}',
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'suspended',
    resumed_by INTEGER,
    resume_reason TEXT,
    
    -- Timestamps
    suspended_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    suspended_by INTEGER NOT NULL,
    resumed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_suspension_status CHECK (status IN ('suspended', 'resumed', 'cancelled'))
);

-- RLS for workflow suspensions
ALTER TABLE workflow_suspensions ENABLE ROW LEVEL SECURITY;
CREATE POLICY workflow_suspensions_tenant_isolation ON workflow_suspensions
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_workflow_suspensions_tenant_workflow ON workflow_suspensions(tenant_id, workflow_id);
CREATE INDEX idx_workflow_suspensions_status ON workflow_suspensions(tenant_id, status);
CREATE INDEX idx_workflow_suspensions_suspended ON workflow_suspensions(tenant_id, suspended_at DESC);

-- =====================================================
-- WORKFLOW TERMINATIONS (Task 15.2-T15)
-- =====================================================

CREATE TABLE IF NOT EXISTS workflow_terminations (
    termination_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255) NOT NULL,
    tenant_id INTEGER NOT NULL,
    fallback_id UUID REFERENCES fallback_requests(fallback_id) ON DELETE CASCADE,
    
    -- Termination details
    termination_reason TEXT NOT NULL,
    termination_context JSONB DEFAULT '{}',
    cleanup_result JSONB DEFAULT '{}',
    
    -- Timestamps
    terminated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    terminated_by INTEGER NOT NULL,
    
    -- Constraints
    CONSTRAINT chk_termination_cleanup CHECK (cleanup_result IS NOT NULL)
);

-- RLS for workflow terminations
ALTER TABLE workflow_terminations ENABLE ROW LEVEL SECURITY;
CREATE POLICY workflow_terminations_tenant_isolation ON workflow_terminations
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_workflow_terminations_tenant_workflow ON workflow_terminations(tenant_id, workflow_id);
CREATE INDEX idx_workflow_terminations_terminated ON workflow_terminations(tenant_id, terminated_at DESC);

-- =====================================================
-- FALLBACK EVIDENCE PACKS (Task 15.2-T23)
-- =====================================================

CREATE TABLE IF NOT EXISTS fallback_evidence_packs (
    evidence_pack_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fallback_id UUID NOT NULL REFERENCES fallback_requests(fallback_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Evidence details
    event_type VARCHAR(50) NOT NULL,
    evidence_data JSONB NOT NULL,
    
    -- Digital signature (Task 15.2-T25)
    digital_signature VARCHAR(255) NOT NULL,
    hash_algorithm VARCHAR(50) NOT NULL DEFAULT 'SHA256',
    
    -- Storage information
    storage_location VARCHAR(500),
    compressed BOOLEAN DEFAULT false,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    retention_until DATE,
    
    -- Constraints
    CONSTRAINT chk_fallback_event_type CHECK (event_type IN ('fallback_created', 'fallback_executed', 'fallback_completed', 'fallback_failed')),
    CONSTRAINT chk_fallback_hash_algorithm CHECK (hash_algorithm IN ('SHA256', 'SHA512', 'BLAKE2b'))
);

-- RLS for fallback evidence packs
ALTER TABLE fallback_evidence_packs ENABLE ROW LEVEL SECURITY;
CREATE POLICY fallback_evidence_packs_tenant_isolation ON fallback_evidence_packs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_fallback_evidence_fallback_id ON fallback_evidence_packs(fallback_id, created_at DESC);
CREATE INDEX idx_fallback_evidence_tenant_type ON fallback_evidence_packs(tenant_id, event_type);
CREATE INDEX idx_fallback_evidence_created ON fallback_evidence_packs(tenant_id, created_at DESC);
CREATE INDEX idx_fallback_evidence_retention ON fallback_evidence_packs(retention_until) WHERE retention_until IS NOT NULL;

-- =====================================================
-- ESCALATION EVIDENCE PACKS (Task 15.2-T24)
-- =====================================================

CREATE TABLE IF NOT EXISTS escalation_evidence_packs (
    evidence_pack_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    escalation_id UUID NOT NULL REFERENCES escalation_requests(escalation_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Evidence details
    event_type VARCHAR(50) NOT NULL,
    evidence_data JSONB NOT NULL,
    
    -- Digital signature (Task 15.2-T25)
    digital_signature VARCHAR(255) NOT NULL,
    hash_algorithm VARCHAR(50) NOT NULL DEFAULT 'SHA256',
    
    -- Storage information
    storage_location VARCHAR(500),
    compressed BOOLEAN DEFAULT false,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    retention_until DATE,
    
    -- Constraints
    CONSTRAINT chk_escalation_event_type CHECK (event_type IN ('escalation_created', 'escalation_assigned', 'escalation_status_changed', 'escalation_resolved', 'escalation_escalated')),
    CONSTRAINT chk_escalation_hash_algorithm CHECK (hash_algorithm IN ('SHA256', 'SHA512', 'BLAKE2b'))
);

-- RLS for escalation evidence packs
ALTER TABLE escalation_evidence_packs ENABLE ROW LEVEL SECURITY;
CREATE POLICY escalation_evidence_packs_tenant_isolation ON escalation_evidence_packs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_escalation_evidence_escalation_id ON escalation_evidence_packs(escalation_id, created_at DESC);
CREATE INDEX idx_escalation_evidence_tenant_type ON escalation_evidence_packs(tenant_id, event_type);
CREATE INDEX idx_escalation_evidence_created ON escalation_evidence_packs(tenant_id, created_at DESC);
CREATE INDEX idx_escalation_evidence_retention ON escalation_evidence_packs(retention_until) WHERE retention_until IS NOT NULL;

-- =====================================================
-- ESCALATION CHAIN REGISTRY (Task 15.2-T17)
-- =====================================================

CREATE TABLE IF NOT EXISTS escalation_chain_registry (
    chain_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Chain configuration
    chain_name VARCHAR(255) NOT NULL,
    industry_code VARCHAR(50) NOT NULL,
    persona VARCHAR(50) NOT NULL,
    
    -- Chain definition
    chain_members JSONB NOT NULL DEFAULT '[]',
    escalation_rules JSONB DEFAULT '{}',
    sla_overrides JSONB DEFAULT '{}',
    
    -- Versioning
    version INTEGER NOT NULL DEFAULT 1,
    parent_chain_id UUID,
    
    -- Status
    active BOOLEAN NOT NULL DEFAULT true,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_chain_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech')),
    CONSTRAINT chk_chain_persona CHECK (persona IN ('ops', 'compliance', 'finance', 'executive', 'legal', 'security')),
    CONSTRAINT chk_chain_version CHECK (version > 0),
    CONSTRAINT chk_chain_name_unique UNIQUE (tenant_id, chain_name, version)
);

-- RLS for escalation chain registry
ALTER TABLE escalation_chain_registry ENABLE ROW LEVEL SECURITY;
CREATE POLICY escalation_chain_registry_tenant_isolation ON escalation_chain_registry
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_escalation_chains_tenant_industry ON escalation_chain_registry(tenant_id, industry_code);
CREATE INDEX idx_escalation_chains_persona ON escalation_chain_registry(tenant_id, persona);
CREATE INDEX idx_escalation_chains_active ON escalation_chain_registry(tenant_id, active) WHERE active = true;
CREATE INDEX idx_escalation_chains_parent ON escalation_chain_registry(parent_chain_id) WHERE parent_chain_id IS NOT NULL;

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to auto-escalate overdue escalations
CREATE OR REPLACE FUNCTION auto_escalate_overdue_escalations()
RETURNS INTEGER AS $$
DECLARE
    escalated_count INTEGER;
BEGIN
    -- Find overdue escalations that should auto-escalate
    UPDATE escalation_requests
    SET status = 'escalated',
        updated_at = NOW()
    WHERE status IN ('open', 'acknowledged', 'in_progress')
    AND auto_escalate = true
    AND sla_deadline <= NOW()
    AND tenant_id = current_setting('app.current_tenant_id')::integer;
    
    GET DIAGNOSTICS escalated_count = ROW_COUNT;
    
    -- Log escalations
    INSERT INTO escalation_state_transitions (
        transition_id, escalation_id, tenant_id, from_status, to_status,
        changed_by, change_reason, created_at
    )
    SELECT 
        uuid_generate_v4(),
        escalation_id,
        tenant_id,
        status,
        'escalated',
        0, -- System user
        'Auto-escalated due to SLA breach',
        NOW()
    FROM escalation_requests
    WHERE status = 'escalated'
    AND updated_at >= NOW() - INTERVAL '1 minute';
    
    RETURN escalated_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update escalation SLA status
CREATE OR REPLACE FUNCTION update_escalation_sla_status()
RETURNS TRIGGER AS $$
BEGIN
    -- Update SLA breach flag when deadline is passed
    IF NEW.sla_deadline <= NOW() AND OLD.sla_deadline > NOW() THEN
        -- Log SLA breach
        INSERT INTO escalation_state_transitions (
            transition_id, escalation_id, tenant_id, from_status, to_status,
            changed_by, change_reason, created_at
        ) VALUES (
            uuid_generate_v4(),
            NEW.escalation_id,
            NEW.tenant_id,
            NEW.status,
            NEW.status,
            0, -- System user
            'SLA deadline breached',
            NOW()
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for SLA monitoring
CREATE TRIGGER trigger_escalation_sla_monitoring
    AFTER UPDATE ON escalation_requests
    FOR EACH ROW
    WHEN (OLD.sla_deadline IS DISTINCT FROM NEW.sla_deadline OR OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION update_escalation_sla_status();

-- Function to clean up old fallback/escalation data
CREATE OR REPLACE FUNCTION cleanup_old_fallback_escalation_data()
RETURNS INTEGER AS $$
DECLARE
    cleanup_count INTEGER;
BEGIN
    -- Clean up old fallback requests (keep for 90 days)
    DELETE FROM fallback_requests 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS cleanup_count = ROW_COUNT;
    
    -- Clean up old escalation requests (keep for 1 year)
    DELETE FROM escalation_requests 
    WHERE created_at < NOW() - INTERVAL '1 year'
    AND status IN ('resolved', 'closed');
    
    -- Clean up old evidence packs based on retention policy
    DELETE FROM fallback_evidence_packs 
    WHERE retention_until IS NOT NULL 
    AND retention_until < CURRENT_DATE;
    
    DELETE FROM escalation_evidence_packs 
    WHERE retention_until IS NOT NULL 
    AND retention_until < CURRENT_DATE;
    
    RETURN cleanup_count;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE fallback_policies IS 'Fallback policies per tenant with industry-specific configurations (Task 15.2-T09)';
COMMENT ON TABLE escalation_policies IS 'Escalation policies with industry-specific chains and SLA timers (Task 15.2-T10, T17-T20)';
COMMENT ON TABLE fallback_requests IS 'Fallback requests with evidence tracking (Task 15.2-T21)';
COMMENT ON TABLE escalation_requests IS 'Manual escalation requests with state machine (Task 15.2-T22, T16)';
COMMENT ON TABLE escalation_state_transitions IS 'Escalation state transition audit trail (Task 15.2-T16)';
COMMENT ON TABLE workflow_suspensions IS 'Workflow suspension tracking (Task 15.2-T14)';
COMMENT ON TABLE workflow_terminations IS 'Workflow termination tracking (Task 15.2-T15)';
COMMENT ON TABLE fallback_evidence_packs IS 'Evidence packs for fallback events (Task 15.2-T23)';
COMMENT ON TABLE escalation_evidence_packs IS 'Evidence packs for escalation events (Task 15.2-T24)';
COMMENT ON TABLE escalation_chain_registry IS 'Versioned escalation chain definitions (Task 15.2-T17)';

COMMENT ON FUNCTION auto_escalate_overdue_escalations() IS 'Automatically escalate overdue escalations based on SLA deadlines';
COMMENT ON FUNCTION update_escalation_sla_status() IS 'Monitor and log SLA breaches for escalations';
COMMENT ON FUNCTION cleanup_old_fallback_escalation_data() IS 'Clean up old fallback and escalation data based on retention policies';
