-- Chapter 14 Complete Database Schemas
-- =====================================
-- All remaining Chapter 14 database schemas for backend features

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- INDUSTRY TEMPLATES (Tasks 14.1-T29, T30)
-- =====================================================

CREATE TABLE IF NOT EXISTS industry_approval_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Template metadata
    template_name VARCHAR(255) NOT NULL,
    industry_code VARCHAR(50) NOT NULL,
    template_type VARCHAR(50) NOT NULL,
    description TEXT,
    
    -- Approval configuration
    approval_thresholds JSONB NOT NULL DEFAULT '[]',
    compliance_frameworks JSONB NOT NULL DEFAULT '[]',
    policy_pack_ids JSONB NOT NULL DEFAULT '[]',
    workflow_config JSONB NOT NULL DEFAULT '{}',
    
    -- Status and versioning
    version INTEGER DEFAULT 1,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech')),
    CONSTRAINT chk_template_type CHECK (template_type IN ('approval', 'override', 'cross_module', 'escalation'))
);

-- RLS for industry templates
ALTER TABLE industry_approval_templates ENABLE ROW LEVEL SECURITY;
CREATE POLICY industry_templates_tenant_isolation ON industry_approval_templates
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_industry_templates_tenant_industry ON industry_approval_templates(tenant_id, industry_code);
CREATE INDEX idx_industry_templates_type ON industry_approval_templates(tenant_id, template_type);
CREATE INDEX idx_industry_templates_active ON industry_approval_templates(tenant_id, active) WHERE active = true;

-- Override templates table
CREATE TABLE IF NOT EXISTS industry_override_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Override configuration
    industry_code VARCHAR(50) NOT NULL,
    override_type VARCHAR(100) NOT NULL,
    reason_codes JSONB NOT NULL DEFAULT '[]',
    required_approvers JSONB NOT NULL DEFAULT '[]',
    compliance_frameworks JSONB NOT NULL DEFAULT '[]',
    evidence_requirements JSONB NOT NULL DEFAULT '[]',
    
    -- Template configuration
    template_config JSONB NOT NULL DEFAULT '{}',
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_override_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech'))
);

-- RLS for override templates
ALTER TABLE industry_override_templates ENABLE ROW LEVEL SECURITY;
CREATE POLICY override_templates_tenant_isolation ON industry_override_templates
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_override_templates_tenant_industry ON industry_override_templates(tenant_id, industry_code);
CREATE INDEX idx_override_templates_type ON industry_override_templates(tenant_id, override_type);

-- =====================================================
-- CROSS-MODULE FLOWS (Task 14.1-T31)
-- =====================================================

CREATE TABLE IF NOT EXISTS cross_module_approval_flows (
    flow_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Flow configuration
    flow_name VARCHAR(255) NOT NULL,
    source_module VARCHAR(100) NOT NULL,
    target_modules JSONB NOT NULL DEFAULT '[]',
    
    -- Flow logic
    trigger_conditions JSONB NOT NULL DEFAULT '{}',
    approval_sequence JSONB NOT NULL DEFAULT '[]',
    parallel_approvals BOOLEAN DEFAULT false,
    
    -- Industry specificity
    industry_specific BOOLEAN DEFAULT false,
    industry_codes JSONB DEFAULT '[]',
    
    -- Configuration and status
    flow_config JSONB DEFAULT '{}',
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_source_module CHECK (source_module IN ('pipeline', 'forecasting', 'revenue', 'planning', 'compensation', 'cruxx')),
    CONSTRAINT chk_flow_name_unique UNIQUE (tenant_id, flow_name)
);

-- RLS for cross-module flows
ALTER TABLE cross_module_approval_flows ENABLE ROW LEVEL SECURITY;
CREATE POLICY cross_module_flows_tenant_isolation ON cross_module_approval_flows
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_cross_module_flows_tenant_source ON cross_module_approval_flows(tenant_id, source_module);
CREATE INDEX idx_cross_module_flows_industry ON cross_module_approval_flows(tenant_id, industry_specific);
CREATE INDEX idx_cross_module_flows_active ON cross_module_approval_flows(tenant_id, active) WHERE active = true;

-- =====================================================
-- AI APPROVAL ASSISTANT (Tasks 14.1-T40, T41)
-- =====================================================

CREATE TABLE IF NOT EXISTS ai_approver_suggestions (
    suggestion_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Request context
    workflow_id UUID,
    request_type VARCHAR(100) NOT NULL,
    financial_amount DECIMAL(15,2),
    urgency_level VARCHAR(20) DEFAULT 'medium',
    industry_context VARCHAR(50) DEFAULT 'SaaS',
    
    -- AI suggestions
    suggested_approvers JSONB NOT NULL DEFAULT '[]',
    suggestion_reasoning JSONB NOT NULL DEFAULT '{}',
    confidence_score DECIMAL(5,3),
    model_version VARCHAR(50) DEFAULT 'v1.0',
    
    -- Outcome tracking
    selected_approver_id INTEGER,
    suggestion_accepted BOOLEAN,
    actual_approval_time_hours DECIMAL(8,2),
    feedback_score INTEGER,
    
    -- Metadata
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days'),
    
    -- Constraints
    CONSTRAINT chk_urgency_level CHECK (urgency_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.000 AND confidence_score <= 1.000),
    CONSTRAINT chk_feedback_score CHECK (feedback_score IS NULL OR (feedback_score >= 1 AND feedback_score <= 5))
);

-- RLS for AI suggestions
ALTER TABLE ai_approver_suggestions ENABLE ROW LEVEL SECURITY;
CREATE POLICY ai_suggestions_tenant_isolation ON ai_approver_suggestions
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_ai_suggestions_tenant_workflow ON ai_approver_suggestions(tenant_id, workflow_id);
CREATE INDEX idx_ai_suggestions_request_type ON ai_approver_suggestions(tenant_id, request_type);
CREATE INDEX idx_ai_suggestions_generated ON ai_approver_suggestions(tenant_id, generated_at DESC);
CREATE INDEX idx_ai_suggestions_feedback ON ai_approver_suggestions(tenant_id, suggestion_accepted) WHERE suggestion_accepted IS NOT NULL;

-- AI override risk assessments
CREATE TABLE IF NOT EXISTS ai_override_risk_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Override context
    override_id UUID,
    workflow_id UUID,
    override_reason TEXT,
    reason_code VARCHAR(100),
    requested_by_user_id INTEGER NOT NULL,
    financial_impact DECIMAL(15,2),
    compliance_impact JSONB DEFAULT '[]',
    
    -- AI assessment
    risk_level VARCHAR(20) NOT NULL,
    risk_score DECIMAL(5,3) NOT NULL,
    risk_factors JSONB NOT NULL DEFAULT '[]',
    compliance_violations JSONB DEFAULT '[]',
    recommended_actions JSONB DEFAULT '[]',
    requires_escalation BOOLEAN DEFAULT false,
    escalation_path JSONB DEFAULT '[]',
    
    -- Model information
    model_version VARCHAR(50) DEFAULT 'v1.0',
    assessment_confidence DECIMAL(5,3),
    
    -- Outcome tracking
    human_decision VARCHAR(50),
    assessment_accuracy BOOLEAN,
    actual_outcome VARCHAR(50),
    
    -- Metadata
    assessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_risk_level CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_risk_score CHECK (risk_score >= 0.000 AND risk_score <= 1.000),
    CONSTRAINT chk_assessment_confidence CHECK (assessment_confidence IS NULL OR (assessment_confidence >= 0.000 AND assessment_confidence <= 1.000))
);

-- RLS for risk assessments
ALTER TABLE ai_override_risk_assessments ENABLE ROW LEVEL SECURITY;
CREATE POLICY ai_risk_assessments_tenant_isolation ON ai_override_risk_assessments
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_ai_risk_assessments_tenant_override ON ai_override_risk_assessments(tenant_id, override_id);
CREATE INDEX idx_ai_risk_assessments_user ON ai_override_risk_assessments(tenant_id, requested_by_user_id);
CREATE INDEX idx_ai_risk_assessments_risk_level ON ai_override_risk_assessments(tenant_id, risk_level);
CREATE INDEX idx_ai_risk_assessments_assessed ON ai_override_risk_assessments(tenant_id, assessed_at DESC);

-- =====================================================
-- APPROVAL DELEGATION (Task 14.1-T33)
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_delegations (
    delegation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Delegation participants
    delegator_id INTEGER NOT NULL,
    delegate_id INTEGER NOT NULL,
    
    -- Delegation scope
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
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_delegation_type CHECK (delegation_type IN ('temporary', 'permanent', 'conditional', 'emergency_only')),
    CONSTRAINT chk_delegation_dates CHECK (end_date IS NULL OR end_date > start_date),
    CONSTRAINT chk_different_users CHECK (delegator_id != delegate_id)
);

-- RLS for delegations
ALTER TABLE approval_delegations ENABLE ROW LEVEL SECURITY;
CREATE POLICY approval_delegations_tenant_isolation ON approval_delegations
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_approval_delegations_tenant_delegator ON approval_delegations(tenant_id, delegator_id);
CREATE INDEX idx_approval_delegations_tenant_delegate ON approval_delegations(tenant_id, delegate_id);
CREATE INDEX idx_approval_delegations_active ON approval_delegations(tenant_id, active) WHERE active = true;
CREATE INDEX idx_approval_delegations_dates ON approval_delegations(tenant_id, start_date, end_date);

-- =====================================================
-- BULK OPERATIONS (Task 14.1-T34)
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
CREATE POLICY bulk_operations_tenant_isolation ON bulk_approval_operations
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_bulk_operations_tenant_approver ON bulk_approval_operations(tenant_id, approver_id);
CREATE INDEX idx_bulk_operations_status ON bulk_approval_operations(tenant_id, status);
CREATE INDEX idx_bulk_operations_started ON bulk_approval_operations(tenant_id, started_at DESC);

-- =====================================================
-- APPROVAL COMMENTS & ANNOTATIONS (Task 14.1-T35)
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_comments (
    comment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Associated approval
    approval_id UUID NOT NULL,
    parent_comment_id UUID, -- For threaded comments
    
    -- Comment details
    comment_text TEXT NOT NULL,
    comment_type VARCHAR(50) DEFAULT 'general',
    visibility VARCHAR(50) DEFAULT 'all_approvers',
    
    -- Annotations
    annotations JSONB DEFAULT '{}',
    attachments JSONB DEFAULT '[]',
    
    -- Author information
    author_id INTEGER NOT NULL,
    author_role VARCHAR(100),
    
    -- Status
    edited BOOLEAN DEFAULT false,
    deleted BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_comment_type CHECK (comment_type IN ('general', 'approval_reason', 'rejection_reason', 'question', 'clarification')),
    CONSTRAINT chk_visibility CHECK (visibility IN ('all_approvers', 'senior_approvers', 'compliance_only', 'private'))
);

-- RLS for comments
ALTER TABLE approval_comments ENABLE ROW LEVEL SECURITY;
CREATE POLICY approval_comments_tenant_isolation ON approval_comments
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_approval_comments_tenant_approval ON approval_comments(tenant_id, approval_id);
CREATE INDEX idx_approval_comments_author ON approval_comments(tenant_id, author_id);
CREATE INDEX idx_approval_comments_created ON approval_comments(tenant_id, created_at DESC);
CREATE INDEX idx_approval_comments_parent ON approval_comments(tenant_id, parent_comment_id) WHERE parent_comment_id IS NOT NULL;

-- =====================================================
-- NOTIFICATION HOOKS (Task 14.1-T36)
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_notification_hooks (
    hook_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Hook configuration
    hook_name VARCHAR(255) NOT NULL,
    hook_type VARCHAR(50) NOT NULL,
    trigger_events JSONB NOT NULL DEFAULT '[]',
    
    -- Notification channels
    notification_channels JSONB NOT NULL DEFAULT '{}',
    
    -- Targeting
    target_roles JSONB DEFAULT '[]',
    target_users JSONB DEFAULT '[]',
    conditions JSONB DEFAULT '{}',
    
    -- Template and formatting
    message_template TEXT,
    template_variables JSONB DEFAULT '{}',
    
    -- Status and timing
    active BOOLEAN DEFAULT true,
    throttle_minutes INTEGER DEFAULT 0,
    last_triggered_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_hook_type CHECK (hook_type IN ('approval_request', 'approval_decision', 'sla_breach', 'escalation', 'override_event')),
    CONSTRAINT chk_throttle_minutes CHECK (throttle_minutes >= 0)
);

-- RLS for notification hooks
ALTER TABLE approval_notification_hooks ENABLE ROW LEVEL SECURITY;
CREATE POLICY notification_hooks_tenant_isolation ON approval_notification_hooks
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_notification_hooks_tenant_type ON approval_notification_hooks(tenant_id, hook_type);
CREATE INDEX idx_notification_hooks_active ON approval_notification_hooks(tenant_id, active) WHERE active = true;
CREATE INDEX idx_notification_hooks_last_triggered ON approval_notification_hooks(tenant_id, last_triggered_at);

-- =====================================================
-- APPROVAL QUOTAS (Task 14.1-T39)
-- =====================================================

CREATE TABLE IF NOT EXISTS approval_quotas (
    quota_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Quota scope
    quota_name VARCHAR(255) NOT NULL,
    quota_type VARCHAR(50) NOT NULL,
    scope_type VARCHAR(50) NOT NULL,
    scope_value VARCHAR(255) NOT NULL,
    
    -- Quota limits
    max_pending_approvals INTEGER NOT NULL,
    max_daily_approvals INTEGER,
    max_weekly_approvals INTEGER,
    max_monthly_approvals INTEGER,
    
    -- Time window
    time_window_hours INTEGER DEFAULT 24,
    reset_schedule VARCHAR(50) DEFAULT 'daily',
    
    -- Current usage
    current_pending INTEGER DEFAULT 0,
    current_daily INTEGER DEFAULT 0,
    current_weekly INTEGER DEFAULT 0,
    current_monthly INTEGER DEFAULT 0,
    
    -- Status
    active BOOLEAN DEFAULT true,
    enforcement_level VARCHAR(50) DEFAULT 'warn',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_reset_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_quota_type CHECK (quota_type IN ('user_quota', 'role_quota', 'department_quota', 'global_quota')),
    CONSTRAINT chk_scope_type CHECK (scope_type IN ('user_id', 'role_name', 'department', 'tenant')),
    CONSTRAINT chk_enforcement_level CHECK (enforcement_level IN ('warn', 'block', 'escalate')),
    CONSTRAINT chk_reset_schedule CHECK (reset_schedule IN ('hourly', 'daily', 'weekly', 'monthly')),
    CONSTRAINT chk_quota_limits CHECK (max_pending_approvals > 0)
);

-- RLS for quotas
ALTER TABLE approval_quotas ENABLE ROW LEVEL SECURITY;
CREATE POLICY approval_quotas_tenant_isolation ON approval_quotas
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_approval_quotas_tenant_scope ON approval_quotas(tenant_id, scope_type, scope_value);
CREATE INDEX idx_approval_quotas_type ON approval_quotas(tenant_id, quota_type);
CREATE INDEX idx_approval_quotas_active ON approval_quotas(tenant_id, active) WHERE active = true;

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

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

-- Function to update quota usage
CREATE OR REPLACE FUNCTION update_approval_quota_usage()
RETURNS TRIGGER AS $$
BEGIN
    -- Update quota usage when approval status changes
    IF TG_OP = 'INSERT' AND NEW.status = 'pending' THEN
        -- Increment pending count
        UPDATE approval_quotas
        SET current_pending = current_pending + 1,
            updated_at = NOW()
        WHERE tenant_id = NEW.tenant_id
        AND scope_type = 'user_id'
        AND scope_value = NEW.requested_by_user_id::text
        AND active = true;
        
    ELSIF TG_OP = 'UPDATE' AND OLD.status = 'pending' AND NEW.status != 'pending' THEN
        -- Decrement pending count, increment daily count
        UPDATE approval_quotas
        SET current_pending = GREATEST(0, current_pending - 1),
            current_daily = current_daily + 1,
            updated_at = NOW()
        WHERE tenant_id = NEW.tenant_id
        AND scope_type = 'user_id'
        AND scope_value = NEW.requested_by_user_id::text
        AND active = true;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Trigger for quota usage updates
CREATE TRIGGER trigger_update_quota_usage
    AFTER INSERT OR UPDATE ON approval_ledger
    FOR EACH ROW
    EXECUTE FUNCTION update_approval_quota_usage();

-- Function to reset quota counters
CREATE OR REPLACE FUNCTION reset_quota_counters()
RETURNS INTEGER AS $$
DECLARE
    reset_count INTEGER;
BEGIN
    -- Reset daily counters
    UPDATE approval_quotas
    SET current_daily = 0,
        last_reset_at = NOW(),
        updated_at = NOW()
    WHERE reset_schedule = 'daily'
    AND last_reset_at < CURRENT_DATE;
    
    GET DIAGNOSTICS reset_count = ROW_COUNT;
    
    -- Reset weekly counters
    UPDATE approval_quotas
    SET current_weekly = 0,
        last_reset_at = NOW(),
        updated_at = NOW()
    WHERE reset_schedule = 'weekly'
    AND last_reset_at < DATE_TRUNC('week', NOW());
    
    -- Reset monthly counters
    UPDATE approval_quotas
    SET current_monthly = 0,
        last_reset_at = NOW(),
        updated_at = NOW()
    WHERE reset_schedule = 'monthly'
    AND last_reset_at < DATE_TRUNC('month', NOW());
    
    RETURN reset_count;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE industry_approval_templates IS 'Industry-specific approval templates (Task 14.1-T29)';
COMMENT ON TABLE industry_override_templates IS 'Industry-specific override templates (Task 14.1-T30)';
COMMENT ON TABLE cross_module_approval_flows IS 'Cross-module approval flow definitions (Task 14.1-T31)';
COMMENT ON TABLE ai_approver_suggestions IS 'AI-powered approver suggestions (Task 14.1-T40)';
COMMENT ON TABLE ai_override_risk_assessments IS 'AI-powered override risk assessments (Task 14.1-T41)';
COMMENT ON TABLE approval_delegations IS 'Approval delegation rules (Task 14.1-T33)';
COMMENT ON TABLE bulk_approval_operations IS 'Bulk approval operation tracking (Task 14.1-T34)';
COMMENT ON TABLE approval_comments IS 'Approval comments and annotations (Task 14.1-T35)';
COMMENT ON TABLE approval_notification_hooks IS 'Notification hooks for approval events (Task 14.1-T36)';
COMMENT ON TABLE approval_quotas IS 'Approval quota management (Task 14.1-T39)';

COMMENT ON FUNCTION auto_expire_delegations() IS 'Automatically expire delegation rules past their end date';
COMMENT ON FUNCTION update_approval_quota_usage() IS 'Update quota usage counters on approval status changes';
COMMENT ON FUNCTION reset_quota_counters() IS 'Reset quota counters based on schedule';
