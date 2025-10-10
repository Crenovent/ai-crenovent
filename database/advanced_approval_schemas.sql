-- Advanced Approval System Database Schemas - Chapter 14.1 Advanced Features
-- ==========================================================================
-- Tasks 14.1-T14 to T42: Advanced approval features and AI assistance

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- DELEGATION AND PROXY APPROVALS (Task 14.1-T35)
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_delegations (
    delegation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Delegation participants
    delegator_id INTEGER NOT NULL,
    delegate_id INTEGER NOT NULL,
    
    -- Delegation configuration
    delegation_type VARCHAR(50) NOT NULL DEFAULT 'temporary',
    scope_filters JSONB DEFAULT '{}',
    conditions JSONB DEFAULT '[]',
    
    -- Time bounds
    start_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    end_date TIMESTAMP WITH TIME ZONE,
    
    -- Status
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_delegation_type CHECK (delegation_type IN ('temporary', 'permanent', 'conditional', 'emergency_only')),
    CONSTRAINT chk_delegation_dates CHECK (end_date IS NULL OR end_date > start_date),
    CONSTRAINT chk_different_users CHECK (delegator_id != delegate_id)
);

-- RLS for delegation rules
ALTER TABLE approval_delegations ENABLE ROW LEVEL SECURITY;

CREATE POLICY approval_delegations_tenant_isolation ON approval_delegations
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes for delegation queries
CREATE INDEX idx_approval_delegations_tenant_delegator ON approval_delegations(tenant_id, delegator_id);
CREATE INDEX idx_approval_delegations_tenant_delegate ON approval_delegations(tenant_id, delegate_id);
CREATE INDEX idx_approval_delegations_active ON approval_delegations(tenant_id, active) WHERE active = true;

-- =====================================================
-- BULK APPROVAL OPERATIONS (Task 14.1-T38)
-- =====================================================

CREATE TABLE IF NOT EXISTS bulk_approval_operations (
    batch_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Operation details
    approver_id INTEGER NOT NULL,
    operation_type VARCHAR(50) NOT NULL DEFAULT 'bulk_approve',
    
    -- Batch statistics
    total_requests INTEGER NOT NULL DEFAULT 0,
    approved_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    
    -- Operation results
    operation_results JSONB DEFAULT '{}',
    
    -- Timing
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    
    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'processing',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_bulk_operation_type CHECK (operation_type IN ('bulk_approve', 'bulk_reject', 'bulk_delegate')),
    CONSTRAINT chk_bulk_status CHECK (status IN ('processing', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_bulk_counts CHECK (
        total_requests >= 0 AND 
        approved_count >= 0 AND 
        rejected_count >= 0 AND 
        error_count >= 0 AND
        (approved_count + rejected_count + error_count) <= total_requests
    )
);

-- RLS for bulk operations
ALTER TABLE bulk_approval_operations ENABLE ROW LEVEL SECURITY;

CREATE POLICY bulk_approval_operations_tenant_isolation ON bulk_approval_operations
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes for bulk operations
CREATE INDEX idx_bulk_operations_tenant_approver ON bulk_approval_operations(tenant_id, approver_id);
CREATE INDEX idx_bulk_operations_status ON bulk_approval_operations(tenant_id, status);
CREATE INDEX idx_bulk_operations_created ON bulk_approval_operations(tenant_id, created_at DESC);

-- =====================================================
-- CROSS-MODULE APPROVAL FLOWS (Task 14.1-T32)
-- =====================================================

CREATE TABLE IF NOT EXISTS cross_module_approval_flows (
    flow_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Flow configuration
    flow_name VARCHAR(255) NOT NULL,
    modules JSONB NOT NULL DEFAULT '[]',
    dependencies JSONB NOT NULL DEFAULT '{}',
    
    -- Approval sequence
    approval_sequence JSONB NOT NULL DEFAULT '[]',
    parallel_approvals BOOLEAN NOT NULL DEFAULT false,
    
    -- Flow configuration
    flow_config JSONB DEFAULT '{}',
    
    -- Status
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER
);

-- RLS for cross-module flows
ALTER TABLE cross_module_approval_flows ENABLE ROW LEVEL SECURITY;

CREATE POLICY cross_module_flows_tenant_isolation ON cross_module_approval_flows
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes for cross-module flows
CREATE INDEX idx_cross_module_flows_tenant ON cross_module_approval_flows(tenant_id);
CREATE INDEX idx_cross_module_flows_active ON cross_module_approval_flows(tenant_id, active) WHERE active = true;

-- =====================================================
-- AI APPROVAL RECOMMENDATIONS (Task 14.1-T40)
-- =====================================================

CREATE TABLE IF NOT EXISTS ai_approval_recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Associated approval
    approval_id UUID NOT NULL,
    workflow_id UUID,
    
    -- AI recommendation
    recommendation VARCHAR(50) NOT NULL,
    confidence DECIMAL(4,3) NOT NULL DEFAULT 0.500,
    reasoning JSONB DEFAULT '[]',
    
    -- Supporting data
    trust_score DECIMAL(4,3),
    risk_factors JSONB DEFAULT '[]',
    historical_context JSONB DEFAULT '{}',
    
    -- Model information
    model_version VARCHAR(50) DEFAULT 'v1.0',
    model_features JSONB DEFAULT '{}',
    
    -- Outcome tracking
    human_decision VARCHAR(50),
    recommendation_accuracy BOOLEAN,
    feedback_score INTEGER,
    
    -- Timing
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    decision_made_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_ai_recommendation CHECK (recommendation IN ('auto_approve', 'recommend_approve', 'manual_review', 'recommend_reject', 'reject')),
    CONSTRAINT chk_ai_confidence CHECK (confidence >= 0.000 AND confidence <= 1.000),
    CONSTRAINT chk_ai_trust_score CHECK (trust_score IS NULL OR (trust_score >= 0.000 AND trust_score <= 1.000)),
    CONSTRAINT chk_ai_feedback CHECK (feedback_score IS NULL OR (feedback_score >= 1 AND feedback_score <= 5))
);

-- RLS for AI recommendations
ALTER TABLE ai_approval_recommendations ENABLE ROW LEVEL SECURITY;

CREATE POLICY ai_recommendations_tenant_isolation ON ai_approval_recommendations
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes for AI recommendations
CREATE INDEX idx_ai_recommendations_tenant_approval ON ai_approval_recommendations(tenant_id, approval_id);
CREATE INDEX idx_ai_recommendations_workflow ON ai_approval_recommendations(tenant_id, workflow_id);
CREATE INDEX idx_ai_recommendations_generated ON ai_approval_recommendations(tenant_id, generated_at DESC);
CREATE INDEX idx_ai_recommendations_accuracy ON ai_approval_recommendations(tenant_id, recommendation_accuracy) WHERE recommendation_accuracy IS NOT NULL;

-- =====================================================
-- APPROVER WORKLOAD TRACKING (Task 14.1-T14)
-- =====================================================

CREATE TABLE IF NOT EXISTS approver_workload_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Approver information
    approver_id INTEGER NOT NULL,
    
    -- Workload metrics
    pending_approvals INTEGER NOT NULL DEFAULT 0,
    avg_response_time_hours DECIMAL(8,2) DEFAULT 24.00,
    approval_rate DECIMAL(4,3) DEFAULT 0.500,
    current_availability BOOLEAN NOT NULL DEFAULT true,
    expertise_score DECIMAL(4,3) DEFAULT 0.500,
    workload_score DECIMAL(4,3) DEFAULT 0.500,
    
    -- Performance metrics
    total_approvals_30d INTEGER DEFAULT 0,
    approved_count_30d INTEGER DEFAULT 0,
    rejected_count_30d INTEGER DEFAULT 0,
    escalated_count_30d INTEGER DEFAULT 0,
    
    -- Availability information
    availability_status VARCHAR(50) DEFAULT 'available',
    out_of_office_until TIMESTAMP WITH TIME ZONE,
    
    -- Calculation metadata
    calculated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    calculation_period_days INTEGER DEFAULT 30,
    
    -- Constraints
    CONSTRAINT chk_workload_metrics_rates CHECK (
        approval_rate >= 0.000 AND approval_rate <= 1.000 AND
        expertise_score >= 0.000 AND expertise_score <= 1.000 AND
        workload_score >= 0.000 AND workload_score <= 1.000
    ),
    CONSTRAINT chk_workload_counts CHECK (
        pending_approvals >= 0 AND
        total_approvals_30d >= 0 AND
        approved_count_30d >= 0 AND
        rejected_count_30d >= 0 AND
        escalated_count_30d >= 0
    ),
    CONSTRAINT chk_availability_status CHECK (availability_status IN ('available', 'busy', 'out_of_office', 'unavailable'))
);

-- RLS for workload metrics
ALTER TABLE approver_workload_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY workload_metrics_tenant_isolation ON approver_workload_metrics
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes for workload metrics
CREATE INDEX idx_workload_metrics_tenant_approver ON approver_workload_metrics(tenant_id, approver_id);
CREATE INDEX idx_workload_metrics_availability ON approver_workload_metrics(tenant_id, current_availability);
CREATE INDEX idx_workload_metrics_calculated ON approver_workload_metrics(tenant_id, calculated_at DESC);

-- Unique constraint to prevent duplicate metrics for same approver
CREATE UNIQUE INDEX idx_workload_metrics_unique ON approver_workload_metrics(tenant_id, approver_id, calculated_at::date);

-- =====================================================
-- APPROVAL ANALYTICS AND REPORTING (Task 14.1-T33)
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_analytics_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Snapshot metadata
    snapshot_date DATE NOT NULL,
    snapshot_type VARCHAR(50) NOT NULL DEFAULT 'daily',
    
    -- Approval metrics
    total_approvals INTEGER DEFAULT 0,
    approved_count INTEGER DEFAULT 0,
    rejected_count INTEGER DEFAULT 0,
    pending_count INTEGER DEFAULT 0,
    escalated_count INTEGER DEFAULT 0,
    expired_count INTEGER DEFAULT 0,
    
    -- Performance metrics
    avg_approval_time_hours DECIMAL(8,2) DEFAULT 24.00,
    median_approval_time_hours DECIMAL(8,2) DEFAULT 24.00,
    sla_compliance_rate DECIMAL(4,3) DEFAULT 1.000,
    
    -- Risk metrics
    high_risk_approvals INTEGER DEFAULT 0,
    policy_violations INTEGER DEFAULT 0,
    override_requests INTEGER DEFAULT 0,
    
    -- Efficiency metrics
    auto_approved_count INTEGER DEFAULT 0,
    bulk_operation_count INTEGER DEFAULT 0,
    delegation_usage_count INTEGER DEFAULT 0,
    
    -- Detailed breakdown
    approval_breakdown_by_type JSONB DEFAULT '{}',
    approval_breakdown_by_risk JSONB DEFAULT '{}',
    approver_performance JSONB DEFAULT '{}',
    
    -- Calculation metadata
    calculated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    calculation_duration_ms INTEGER,
    
    -- Constraints
    CONSTRAINT chk_analytics_snapshot_type CHECK (snapshot_type IN ('hourly', 'daily', 'weekly', 'monthly')),
    CONSTRAINT chk_analytics_counts CHECK (
        total_approvals >= 0 AND
        approved_count >= 0 AND
        rejected_count >= 0 AND
        pending_count >= 0 AND
        escalated_count >= 0 AND
        expired_count >= 0
    ),
    CONSTRAINT chk_analytics_sla_rate CHECK (sla_compliance_rate >= 0.000 AND sla_compliance_rate <= 1.000)
);

-- RLS for analytics snapshots
ALTER TABLE approval_analytics_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY analytics_snapshots_tenant_isolation ON approval_analytics_snapshots
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes for analytics snapshots
CREATE INDEX idx_analytics_snapshots_tenant_date ON approval_analytics_snapshots(tenant_id, snapshot_date DESC);
CREATE INDEX idx_analytics_snapshots_type ON approval_analytics_snapshots(tenant_id, snapshot_type, snapshot_date DESC);

-- Unique constraint to prevent duplicate snapshots
CREATE UNIQUE INDEX idx_analytics_snapshots_unique ON approval_analytics_snapshots(tenant_id, snapshot_date, snapshot_type);

-- =====================================================
-- DIGITAL SIGNATURES FOR EVIDENCE PACKS (Task 14.3-T07)
-- =====================================================

CREATE TABLE IF NOT EXISTS evidence_digital_signatures (
    signature_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_pack_id UUID NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Signature data
    signature_data JSONB NOT NULL,
    data_hash VARCHAR(64) NOT NULL,
    signature_hash VARCHAR(64) NOT NULL,
    algorithm VARCHAR(50) NOT NULL DEFAULT 'ECDSA_P256_SHA256',
    
    -- Certificate information
    certificate_chain JSONB DEFAULT '[]',
    certificate_fingerprint VARCHAR(64),
    
    -- Timestamp information
    timestamp_token TEXT,
    timestamp_authority VARCHAR(100) DEFAULT 'internal',
    
    -- Verification status
    verification_status VARCHAR(50) NOT NULL DEFAULT 'signed',
    last_verified_at TIMESTAMP WITH TIME ZONE,
    
    -- Creation metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_signature_algorithm CHECK (algorithm IN ('ECDSA_P256_SHA256', 'RSA_PSS_SHA256', 'EdDSA')),
    CONSTRAINT chk_verification_status CHECK (verification_status IN ('signed', 'verified', 'invalid', 'expired', 'revoked'))
);

-- RLS for digital signatures
ALTER TABLE evidence_digital_signatures ENABLE ROW LEVEL SECURITY;

CREATE POLICY evidence_signatures_tenant_isolation ON evidence_digital_signatures
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes for digital signatures
CREATE INDEX idx_evidence_signatures_pack ON evidence_digital_signatures(tenant_id, evidence_pack_id);
CREATE INDEX idx_evidence_signatures_hash ON evidence_digital_signatures(tenant_id, data_hash);
CREATE INDEX idx_evidence_signatures_created ON evidence_digital_signatures(tenant_id, created_at DESC);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update workload metrics
CREATE OR REPLACE FUNCTION update_approver_workload_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update workload metrics when approval status changes
    IF TG_OP = 'UPDATE' AND OLD.status != NEW.status THEN
        -- Recalculate metrics for affected approvers
        -- This would be implemented as a background job in production
        NULL;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Trigger to update workload metrics
CREATE TRIGGER trigger_update_workload_metrics
    AFTER UPDATE ON approval_ledger
    FOR EACH ROW
    EXECUTE FUNCTION update_approver_workload_metrics();

-- Function to validate delegation rules
CREATE OR REPLACE FUNCTION validate_delegation_rule()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure delegation end date is in the future
    IF NEW.end_date IS NOT NULL AND NEW.end_date <= NOW() THEN
        RAISE EXCEPTION 'Delegation end date must be in the future';
    END IF;
    
    -- Ensure delegator and delegate are different
    IF NEW.delegator_id = NEW.delegate_id THEN
        RAISE EXCEPTION 'Delegator and delegate must be different users';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to validate delegation rules
CREATE TRIGGER trigger_validate_delegation
    BEFORE INSERT OR UPDATE ON approval_delegations
    FOR EACH ROW
    EXECUTE FUNCTION validate_delegation_rule();

-- Function to auto-expire delegations
CREATE OR REPLACE FUNCTION auto_expire_delegations()
RETURNS INTEGER AS $$
DECLARE
    expired_count INTEGER;
BEGIN
    UPDATE approval_delegations
    SET active = false, updated_at = NOW()
    WHERE active = true 
    AND end_date IS NOT NULL 
    AND end_date <= NOW();
    
    GET DIAGNOSTICS expired_count = ROW_COUNT;
    RETURN expired_count;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE approval_delegations IS 'Delegation rules for proxy approvals (Task 14.1-T35)';
COMMENT ON TABLE bulk_approval_operations IS 'Bulk approval operation tracking (Task 14.1-T38)';
COMMENT ON TABLE cross_module_approval_flows IS 'Cross-module approval flow definitions (Task 14.1-T32)';
COMMENT ON TABLE ai_approval_recommendations IS 'AI-powered approval recommendations (Task 14.1-T40)';
COMMENT ON TABLE approver_workload_metrics IS 'Approver workload tracking for dynamic selection (Task 14.1-T14)';
COMMENT ON TABLE approval_analytics_snapshots IS 'Approval analytics and reporting snapshots (Task 14.1-T33)';
COMMENT ON TABLE evidence_digital_signatures IS 'Digital signatures for evidence packs (Task 14.3-T07)';

COMMENT ON FUNCTION auto_expire_delegations() IS 'Function to automatically expire delegation rules past their end date';
COMMENT ON FUNCTION validate_delegation_rule() IS 'Function to validate delegation rule constraints';
COMMENT ON FUNCTION update_approver_workload_metrics() IS 'Function to update approver workload metrics on approval changes';
