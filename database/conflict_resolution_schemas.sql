-- Chapter 15.3 Conflict Resolution Database Schemas
-- =================================================
-- Tasks 15.3-T04, T15, T16: Conflict schema, ledger, and override integration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- CONFLICT REQUESTS (Task 15.3-T04)
-- =====================================================

CREATE TABLE IF NOT EXISTS conflict_requests (
    conflict_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Request context
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255),
    user_id INTEGER NOT NULL,
    
    -- Conflict details
    conflict_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    
    -- Conflicting elements
    conflicting_rules JSONB NOT NULL DEFAULT '[]',
    affected_modules JSONB DEFAULT '[]',
    
    -- Context
    business_context JSONB DEFAULT '{}',
    compliance_context JSONB DEFAULT '{}',
    
    -- Resolution preferences
    preferred_hierarchy VARCHAR(50),
    auto_resolve BOOLEAN NOT NULL DEFAULT true,
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'detected',
    
    -- Evidence tracking
    evidence_pack_id UUID,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_conflict_type CHECK (conflict_type IN ('policy_vs_policy', 'data_vs_policy', 'sla_vs_policy', 'persona_vs_persona', 'compliance_vs_business', 'cross_module')),
    CONSTRAINT chk_conflict_severity CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_conflict_status CHECK (status IN ('detected', 'analyzing', 'resolved', 'escalated', 'overridden')),
    CONSTRAINT chk_preferred_hierarchy CHECK (preferred_hierarchy IS NULL OR preferred_hierarchy IN ('compliance', 'finance', 'operations', 'business'))
);

-- RLS for conflict requests
ALTER TABLE conflict_requests ENABLE ROW LEVEL SECURITY;
CREATE POLICY conflict_requests_tenant_isolation ON conflict_requests
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_conflict_requests_tenant_workflow ON conflict_requests(tenant_id, workflow_id);
CREATE INDEX idx_conflict_requests_tenant_status ON conflict_requests(tenant_id, status);
CREATE INDEX idx_conflict_requests_type ON conflict_requests(tenant_id, conflict_type);
CREATE INDEX idx_conflict_requests_severity ON conflict_requests(tenant_id, severity);
CREATE INDEX idx_conflict_requests_created ON conflict_requests(tenant_id, created_at DESC);

-- =====================================================
-- CONFLICT RESOLUTIONS
-- =====================================================

CREATE TABLE IF NOT EXISTS conflict_resolutions (
    resolution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_id UUID NOT NULL REFERENCES conflict_requests(conflict_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Resolution details
    resolution_hierarchy VARCHAR(50) NOT NULL,
    winning_rule JSONB NOT NULL,
    losing_rules JSONB NOT NULL DEFAULT '[]',
    
    -- Decision rationale
    decision_rationale TEXT NOT NULL,
    compliance_impact TEXT,
    business_impact TEXT,
    
    -- Approver information
    resolved_by INTEGER,
    approval_required BOOLEAN NOT NULL DEFAULT false,
    
    -- Evidence tracking
    evidence_pack_id UUID,
    
    -- Metadata
    resolution_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_resolution_hierarchy CHECK (resolution_hierarchy IN ('compliance', 'finance', 'operations', 'business'))
);

-- RLS for conflict resolutions
ALTER TABLE conflict_resolutions ENABLE ROW LEVEL SECURITY;
CREATE POLICY conflict_resolutions_tenant_isolation ON conflict_resolutions
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_conflict_resolutions_conflict_id ON conflict_resolutions(conflict_id);
CREATE INDEX idx_conflict_resolutions_tenant_hierarchy ON conflict_resolutions(tenant_id, resolution_hierarchy);
CREATE INDEX idx_conflict_resolutions_resolved_by ON conflict_resolutions(tenant_id, resolved_by) WHERE resolved_by IS NOT NULL;
CREATE INDEX idx_conflict_resolutions_created ON conflict_resolutions(tenant_id, created_at DESC);

-- =====================================================
-- CONFLICT LEDGER (Task 15.3-T15)
-- =====================================================

CREATE TABLE IF NOT EXISTS conflict_ledger (
    ledger_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_id UUID NOT NULL REFERENCES conflict_requests(conflict_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL,
    event_description TEXT NOT NULL,
    
    -- Resolution information
    resolution_hierarchy VARCHAR(50),
    approver_id INTEGER,
    
    -- Decision context
    decision_context JSONB DEFAULT '{}',
    override_reason TEXT,
    
    -- Metadata
    ledger_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_ledger_event_type CHECK (event_type IN ('conflict_detected', 'conflict_analyzed', 'conflict_resolved', 'conflict_escalated', 'conflict_overridden', 'status_change')),
    CONSTRAINT chk_ledger_hierarchy CHECK (resolution_hierarchy IS NULL OR resolution_hierarchy IN ('compliance', 'finance', 'operations', 'business'))
);

-- RLS for conflict ledger
ALTER TABLE conflict_ledger ENABLE ROW LEVEL SECURITY;
CREATE POLICY conflict_ledger_tenant_isolation ON conflict_ledger
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_conflict_ledger_conflict_id ON conflict_ledger(conflict_id, created_at DESC);
CREATE INDEX idx_conflict_ledger_tenant_event ON conflict_ledger(tenant_id, event_type);
CREATE INDEX idx_conflict_ledger_approver ON conflict_ledger(tenant_id, approver_id) WHERE approver_id IS NOT NULL;
CREATE INDEX idx_conflict_ledger_hierarchy ON conflict_ledger(tenant_id, resolution_hierarchy) WHERE resolution_hierarchy IS NOT NULL;

-- =====================================================
-- CONFLICT EVIDENCE PACKS (Task 15.3-T13, T14)
-- =====================================================

CREATE TABLE IF NOT EXISTS conflict_evidence_packs (
    evidence_pack_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_id UUID NOT NULL REFERENCES conflict_requests(conflict_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Evidence details
    event_type VARCHAR(50) NOT NULL,
    evidence_data JSONB NOT NULL,
    
    -- Digital signature (Task 15.3-T14)
    digital_signature VARCHAR(255) NOT NULL,
    hash_algorithm VARCHAR(50) NOT NULL DEFAULT 'SHA256',
    
    -- Storage information
    storage_location VARCHAR(500),
    compressed BOOLEAN DEFAULT false,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    retention_until DATE,
    
    -- Constraints
    CONSTRAINT chk_conflict_evidence_event_type CHECK (event_type IN ('conflict_detected', 'conflict_analyzed', 'conflict_resolved_automatically', 'conflict_resolved_manually', 'conflict_escalated')),
    CONSTRAINT chk_conflict_evidence_hash_algorithm CHECK (hash_algorithm IN ('SHA256', 'SHA512', 'BLAKE2b'))
);

-- RLS for conflict evidence packs
ALTER TABLE conflict_evidence_packs ENABLE ROW LEVEL SECURITY;
CREATE POLICY conflict_evidence_packs_tenant_isolation ON conflict_evidence_packs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_conflict_evidence_conflict_id ON conflict_evidence_packs(conflict_id, created_at DESC);
CREATE INDEX idx_conflict_evidence_tenant_type ON conflict_evidence_packs(tenant_id, event_type);
CREATE INDEX idx_conflict_evidence_created ON conflict_evidence_packs(tenant_id, created_at DESC);
CREATE INDEX idx_conflict_evidence_retention ON conflict_evidence_packs(retention_until) WHERE retention_until IS NOT NULL;

-- =====================================================
-- CONFLICT HIERARCHY RULES (Task 15.3-T02, T11)
-- =====================================================

CREATE TABLE IF NOT EXISTS conflict_hierarchy_rules (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Rule configuration
    rule_name VARCHAR(255) NOT NULL,
    industry_code VARCHAR(50) NOT NULL,
    
    -- Hierarchy definition
    hierarchy_order JSONB NOT NULL DEFAULT '[]', -- ['compliance', 'finance', 'operations', 'business']
    
    -- Industry-specific overrides (Task 15.3-T11)
    framework_conflicts JSONB DEFAULT '{}', -- RBI vs GDPR, etc.
    escalation_patterns JSONB DEFAULT '[]',
    
    -- Cross-module rules (Task 15.3-T12)
    cross_module_rules JSONB DEFAULT '{}',
    
    -- Rule metadata
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_hierarchy_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech')),
    CONSTRAINT chk_hierarchy_rule_name_unique UNIQUE (tenant_id, rule_name, industry_code)
);

-- RLS for hierarchy rules
ALTER TABLE conflict_hierarchy_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY conflict_hierarchy_rules_tenant_isolation ON conflict_hierarchy_rules
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_conflict_hierarchy_rules_tenant_industry ON conflict_hierarchy_rules(tenant_id, industry_code);
CREATE INDEX idx_conflict_hierarchy_rules_active ON conflict_hierarchy_rules(tenant_id, active) WHERE active = true;

-- =====================================================
-- CONFLICT OVERRIDE LINKS (Task 15.3-T16)
-- =====================================================

CREATE TABLE IF NOT EXISTS conflict_override_links (
    link_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_id UUID NOT NULL REFERENCES conflict_requests(conflict_id) ON DELETE CASCADE,
    resolution_id UUID NOT NULL REFERENCES conflict_resolutions(resolution_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Override details
    override_type VARCHAR(50) NOT NULL DEFAULT 'conflict_resolution',
    override_reason TEXT,
    
    -- Approver information
    override_by INTEGER NOT NULL,
    approval_chain JSONB DEFAULT '[]',
    
    -- Risk assessment
    risk_level VARCHAR(20) DEFAULT 'medium',
    risk_justification TEXT,
    
    -- Compliance tracking
    compliance_frameworks JSONB DEFAULT '[]',
    regulatory_impact TEXT,
    
    -- Metadata
    override_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_override_type CHECK (override_type IN ('conflict_resolution', 'hierarchy_override', 'emergency_override', 'compliance_exception')),
    CONSTRAINT chk_risk_level CHECK (risk_level IN ('low', 'medium', 'high', 'critical'))
);

-- RLS for override links
ALTER TABLE conflict_override_links ENABLE ROW LEVEL SECURITY;
CREATE POLICY conflict_override_links_tenant_isolation ON conflict_override_links
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_conflict_override_links_conflict_id ON conflict_override_links(conflict_id);
CREATE INDEX idx_conflict_override_links_resolution_id ON conflict_override_links(resolution_id);
CREATE INDEX idx_conflict_override_links_tenant_type ON conflict_override_links(tenant_id, override_type);
CREATE INDEX idx_conflict_override_links_override_by ON conflict_override_links(tenant_id, override_by);
CREATE INDEX idx_conflict_override_links_risk ON conflict_override_links(tenant_id, risk_level);

-- =====================================================
-- CONFLICT METRICS (Task 15.3-T18, T19, T20, T21)
-- =====================================================

CREATE TABLE IF NOT EXISTS conflict_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Time window
    metric_date DATE NOT NULL DEFAULT CURRENT_DATE,
    hour_bucket INTEGER, -- 0-23 for hourly metrics
    
    -- Conflict metrics
    total_conflicts INTEGER NOT NULL DEFAULT 0,
    resolved_conflicts INTEGER NOT NULL DEFAULT 0,
    escalated_conflicts INTEGER NOT NULL DEFAULT 0,
    overridden_conflicts INTEGER NOT NULL DEFAULT 0,
    
    -- Resolution metrics by hierarchy
    compliance_resolutions INTEGER NOT NULL DEFAULT 0,
    finance_resolutions INTEGER NOT NULL DEFAULT 0,
    operations_resolutions INTEGER NOT NULL DEFAULT 0,
    business_resolutions INTEGER NOT NULL DEFAULT 0,
    
    -- Timing metrics
    avg_resolution_time_hours DECIMAL(8,2),
    min_resolution_time_hours DECIMAL(8,2),
    max_resolution_time_hours DECIMAL(8,2),
    
    -- Conflict types
    conflict_types JSONB DEFAULT '{}',
    
    -- Success metrics
    auto_resolution_rate DECIMAL(5,4), -- 0.0000 to 1.0000
    hierarchy_compliance_rate DECIMAL(5,4),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_conflict_metrics_hour_bucket CHECK (hour_bucket IS NULL OR (hour_bucket >= 0 AND hour_bucket <= 23)),
    CONSTRAINT chk_conflict_metrics_counts CHECK (
        total_conflicts >= 0 AND 
        resolved_conflicts >= 0 AND 
        escalated_conflicts >= 0 AND 
        overridden_conflicts >= 0 AND
        compliance_resolutions >= 0 AND 
        finance_resolutions >= 0 AND 
        operations_resolutions >= 0 AND 
        business_resolutions >= 0
    ),
    CONSTRAINT chk_auto_resolution_rate CHECK (auto_resolution_rate IS NULL OR (auto_resolution_rate >= 0.0 AND auto_resolution_rate <= 1.0)),
    CONSTRAINT chk_hierarchy_compliance_rate CHECK (hierarchy_compliance_rate IS NULL OR (hierarchy_compliance_rate >= 0.0 AND hierarchy_compliance_rate <= 1.0))
);

-- RLS for conflict metrics
ALTER TABLE conflict_metrics ENABLE ROW LEVEL SECURITY;
CREATE POLICY conflict_metrics_tenant_isolation ON conflict_metrics
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_conflict_metrics_tenant_date_hour ON conflict_metrics(tenant_id, metric_date, hour_bucket);
CREATE INDEX idx_conflict_metrics_date ON conflict_metrics(tenant_id, metric_date DESC);

-- =====================================================
-- CONFLICT SLA TIMERS (Task 15.3-T41)
-- =====================================================

CREATE TABLE IF NOT EXISTS conflict_sla_timers (
    timer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_id UUID NOT NULL REFERENCES conflict_requests(conflict_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- SLA configuration
    severity VARCHAR(20) NOT NULL,
    sla_hours INTEGER NOT NULL,
    sla_deadline TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Timer status
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    breach_detected BOOLEAN DEFAULT false,
    breach_time TIMESTAMP WITH TIME ZONE,
    
    -- Escalation tracking
    escalation_triggered BOOLEAN DEFAULT false,
    escalation_time TIMESTAMP WITH TIME ZONE,
    escalation_target VARCHAR(255),
    
    -- Metadata
    timer_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_sla_severity CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_sla_status CHECK (status IN ('active', 'completed', 'breached', 'cancelled')),
    CONSTRAINT chk_sla_hours CHECK (sla_hours > 0 AND sla_hours <= 168), -- Max 1 week
    CONSTRAINT chk_sla_deadline CHECK (sla_deadline > created_at)
);

-- RLS for SLA timers
ALTER TABLE conflict_sla_timers ENABLE ROW LEVEL SECURITY;
CREATE POLICY conflict_sla_timers_tenant_isolation ON conflict_sla_timers
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_conflict_sla_timers_conflict_id ON conflict_sla_timers(conflict_id);
CREATE INDEX idx_conflict_sla_timers_tenant_status ON conflict_sla_timers(tenant_id, status);
CREATE INDEX idx_conflict_sla_timers_deadline ON conflict_sla_timers(tenant_id, sla_deadline) WHERE status = 'active';
CREATE INDEX idx_conflict_sla_timers_breach ON conflict_sla_timers(tenant_id, breach_detected) WHERE breach_detected = true;

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to monitor conflict SLA breaches
CREATE OR REPLACE FUNCTION monitor_conflict_sla_breaches()
RETURNS INTEGER AS $$
DECLARE
    breach_count INTEGER;
BEGIN
    -- Detect SLA breaches
    UPDATE conflict_sla_timers
    SET breach_detected = true,
        breach_time = NOW(),
        status = 'breached',
        updated_at = NOW()
    WHERE status = 'active'
    AND sla_deadline <= NOW()
    AND breach_detected = false
    AND tenant_id = current_setting('app.current_tenant_id')::integer;
    
    GET DIAGNOSTICS breach_count = ROW_COUNT;
    
    -- Trigger escalations for breached SLAs
    UPDATE conflict_sla_timers
    SET escalation_triggered = true,
        escalation_time = NOW(),
        escalation_target = 'compliance_team',
        updated_at = NOW()
    WHERE breach_detected = true
    AND escalation_triggered = false
    AND severity IN ('high', 'critical')
    AND tenant_id = current_setting('app.current_tenant_id')::integer;
    
    RETURN breach_count;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate conflict metrics
CREATE OR REPLACE FUNCTION aggregate_conflict_metrics()
RETURNS INTEGER AS $$
DECLARE
    aggregated_count INTEGER;
BEGIN
    -- Aggregate daily metrics from conflicts
    INSERT INTO conflict_metrics (
        tenant_id, metric_date,
        total_conflicts, resolved_conflicts, escalated_conflicts, overridden_conflicts,
        compliance_resolutions, finance_resolutions, operations_resolutions, business_resolutions,
        avg_resolution_time_hours, auto_resolution_rate, hierarchy_compliance_rate
    )
    SELECT 
        cr.tenant_id,
        DATE(cr.created_at) as metric_date,
        COUNT(*) as total_conflicts,
        COUNT(CASE WHEN cr.status = 'resolved' THEN 1 END) as resolved_conflicts,
        COUNT(CASE WHEN cr.status = 'escalated' THEN 1 END) as escalated_conflicts,
        COUNT(CASE WHEN cr.status = 'overridden' THEN 1 END) as overridden_conflicts,
        COUNT(CASE WHEN cres.resolution_hierarchy = 'compliance' THEN 1 END) as compliance_resolutions,
        COUNT(CASE WHEN cres.resolution_hierarchy = 'finance' THEN 1 END) as finance_resolutions,
        COUNT(CASE WHEN cres.resolution_hierarchy = 'operations' THEN 1 END) as operations_resolutions,
        COUNT(CASE WHEN cres.resolution_hierarchy = 'business' THEN 1 END) as business_resolutions,
        AVG(EXTRACT(EPOCH FROM (cr.resolved_at - cr.created_at))/3600) as avg_resolution_time_hours,
        CASE 
            WHEN COUNT(*) > 0 THEN COUNT(CASE WHEN cres.resolved_by IS NULL THEN 1 END)::DECIMAL / COUNT(*)
            ELSE 0.0
        END as auto_resolution_rate,
        CASE 
            WHEN COUNT(CASE WHEN cres.resolution_hierarchy IS NOT NULL THEN 1 END) > 0 
            THEN COUNT(CASE WHEN cres.resolution_hierarchy = 'compliance' THEN 1 END)::DECIMAL / COUNT(CASE WHEN cres.resolution_hierarchy IS NOT NULL THEN 1 END)
            ELSE 0.0
        END as hierarchy_compliance_rate
    FROM conflict_requests cr
    LEFT JOIN conflict_resolutions cres ON cr.conflict_id = cres.conflict_id
    WHERE DATE(cr.created_at) = CURRENT_DATE - INTERVAL '1 day'
    AND NOT EXISTS (
        SELECT 1 FROM conflict_metrics cm 
        WHERE cm.tenant_id = cr.tenant_id 
        AND cm.metric_date = DATE(cr.created_at)
        AND cm.hour_bucket IS NULL
    )
    GROUP BY cr.tenant_id, DATE(cr.created_at)
    ON CONFLICT (tenant_id, metric_date, hour_bucket) DO UPDATE SET
        total_conflicts = EXCLUDED.total_conflicts,
        resolved_conflicts = EXCLUDED.resolved_conflicts,
        escalated_conflicts = EXCLUDED.escalated_conflicts,
        overridden_conflicts = EXCLUDED.overridden_conflicts,
        compliance_resolutions = EXCLUDED.compliance_resolutions,
        finance_resolutions = EXCLUDED.finance_resolutions,
        operations_resolutions = EXCLUDED.operations_resolutions,
        business_resolutions = EXCLUDED.business_resolutions,
        avg_resolution_time_hours = EXCLUDED.avg_resolution_time_hours,
        auto_resolution_rate = EXCLUDED.auto_resolution_rate,
        hierarchy_compliance_rate = EXCLUDED.hierarchy_compliance_rate,
        updated_at = NOW();
    
    GET DIAGNOSTICS aggregated_count = ROW_COUNT;
    RETURN aggregated_count;
END;
$$ LANGUAGE plpgsql;

-- Function to create SLA timer for new conflicts
CREATE OR REPLACE FUNCTION create_conflict_sla_timer()
RETURNS TRIGGER AS $$
DECLARE
    sla_hours INTEGER;
BEGIN
    -- Determine SLA hours based on severity
    CASE NEW.severity
        WHEN 'critical' THEN sla_hours := 2;
        WHEN 'high' THEN sla_hours := 8;
        WHEN 'medium' THEN sla_hours := 24;
        WHEN 'low' THEN sla_hours := 72;
        ELSE sla_hours := 24;
    END CASE;
    
    -- Create SLA timer
    INSERT INTO conflict_sla_timers (
        timer_id, conflict_id, tenant_id, severity, sla_hours, sla_deadline
    ) VALUES (
        uuid_generate_v4(),
        NEW.conflict_id,
        NEW.tenant_id,
        NEW.severity,
        sla_hours,
        NEW.created_at + (sla_hours || ' hours')::INTERVAL
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic SLA timer creation
CREATE TRIGGER trigger_create_conflict_sla_timer
    AFTER INSERT ON conflict_requests
    FOR EACH ROW
    EXECUTE FUNCTION create_conflict_sla_timer();

-- Function to update SLA timer on conflict resolution
CREATE OR REPLACE FUNCTION update_conflict_sla_timer()
RETURNS TRIGGER AS $$
BEGIN
    -- Update SLA timer when conflict is resolved
    IF NEW.status IN ('resolved', 'overridden') AND OLD.status != NEW.status THEN
        UPDATE conflict_sla_timers
        SET status = 'completed',
            updated_at = NOW()
        WHERE conflict_id = NEW.conflict_id
        AND tenant_id = NEW.tenant_id
        AND status = 'active';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for SLA timer updates
CREATE TRIGGER trigger_update_conflict_sla_timer
    AFTER UPDATE ON conflict_requests
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION update_conflict_sla_timer();

-- Function to clean up old conflict data
CREATE OR REPLACE FUNCTION cleanup_old_conflict_data()
RETURNS INTEGER AS $$
DECLARE
    cleanup_count INTEGER;
BEGIN
    -- Clean up old conflict requests (keep for 2 years)
    DELETE FROM conflict_requests 
    WHERE created_at < NOW() - INTERVAL '2 years'
    AND status IN ('resolved', 'overridden');
    
    GET DIAGNOSTICS cleanup_count = ROW_COUNT;
    
    -- Clean up old evidence packs based on retention policy
    DELETE FROM conflict_evidence_packs 
    WHERE retention_until IS NOT NULL 
    AND retention_until < CURRENT_DATE;
    
    -- Clean up old metrics (keep for 3 years)
    DELETE FROM conflict_metrics 
    WHERE metric_date < CURRENT_DATE - INTERVAL '3 years';
    
    -- Clean up old SLA timers (keep for 1 year)
    DELETE FROM conflict_sla_timers 
    WHERE created_at < NOW() - INTERVAL '1 year'
    AND status IN ('completed', 'cancelled');
    
    RETURN cleanup_count;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE conflict_requests IS 'Conflict requests with compliance-first resolution (Task 15.3-T04)';
COMMENT ON TABLE conflict_resolutions IS 'Conflict resolutions with hierarchy enforcement (Task 15.3-T08)';
COMMENT ON TABLE conflict_ledger IS 'Immutable conflict ledger for audit trail (Task 15.3-T15)';
COMMENT ON TABLE conflict_evidence_packs IS 'Evidence packs for conflict events with digital signatures (Task 15.3-T13, T14)';
COMMENT ON TABLE conflict_hierarchy_rules IS 'Industry-specific conflict hierarchy rules (Task 15.3-T02, T11)';
COMMENT ON TABLE conflict_override_links IS 'Links to override ledger for manual resolutions (Task 15.3-T16)';
COMMENT ON TABLE conflict_metrics IS 'Aggregated conflict metrics for dashboards (Task 15.3-T18-T21)';
COMMENT ON TABLE conflict_sla_timers IS 'SLA monitoring for conflict resolution (Task 15.3-T41)';

COMMENT ON FUNCTION monitor_conflict_sla_breaches() IS 'Monitor and detect conflict SLA breaches with automatic escalation';
COMMENT ON FUNCTION aggregate_conflict_metrics() IS 'Aggregate conflict metrics from requests for daily reporting';
COMMENT ON FUNCTION create_conflict_sla_timer() IS 'Automatically create SLA timer for new conflicts based on severity';
COMMENT ON FUNCTION update_conflict_sla_timer() IS 'Update SLA timer status when conflicts are resolved';
COMMENT ON FUNCTION cleanup_old_conflict_data() IS 'Clean up old conflict data based on retention policies';
