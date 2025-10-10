-- Chapter 15.4 Resilience Logging Database Schemas
-- =================================================
-- Tasks 15.4-T03, T17, T18: Resilience log schema and tenant isolation

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- RESILIENCE LOGGING POLICIES (Task 15.4-T17)
-- =====================================================

CREATE TABLE IF NOT EXISTS resilience_logging_policies (
    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Policy configuration
    industry_code VARCHAR(50) NOT NULL DEFAULT 'SaaS',
    
    -- Logging rules
    logging_rules JSONB NOT NULL DEFAULT '{}',
    retention_days INTEGER NOT NULL DEFAULT 90,
    
    -- Industry-specific overlays (Task 15.4-T19)
    compliance_frameworks JSONB DEFAULT '[]',
    required_fields JSONB DEFAULT '[]',
    
    -- Policy metadata
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_logging_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech')),
    CONSTRAINT chk_retention_days CHECK (retention_days > 0 AND retention_days <= 3650) -- Max 10 years
);

-- RLS for logging policies
ALTER TABLE resilience_logging_policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY resilience_logging_policies_tenant_isolation ON resilience_logging_policies
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_resilience_logging_policies_tenant_active ON resilience_logging_policies(tenant_id) WHERE active = true;
CREATE INDEX idx_resilience_logging_policies_industry ON resilience_logging_policies(tenant_id, industry_code);

-- =====================================================
-- RESILIENCE ALERT POLICIES (Task 15.4-T22, T25)
-- =====================================================

CREATE TABLE IF NOT EXISTS resilience_alert_policies (
    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Alert configuration
    alert_rules JSONB NOT NULL DEFAULT '{}',
    suppression_rules JSONB NOT NULL DEFAULT '{}',
    routing_overrides JSONB DEFAULT '{}',
    
    -- Thresholds
    error_rate_threshold DECIMAL(5,4) DEFAULT 0.05,
    sla_breach_threshold DECIMAL(5,4) DEFAULT 0.02,
    
    -- Policy metadata
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_error_rate_threshold CHECK (error_rate_threshold >= 0.0 AND error_rate_threshold <= 1.0),
    CONSTRAINT chk_sla_breach_threshold CHECK (sla_breach_threshold >= 0.0 AND sla_breach_threshold <= 1.0)
);

-- RLS for alert policies
ALTER TABLE resilience_alert_policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY resilience_alert_policies_tenant_isolation ON resilience_alert_policies
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_resilience_alert_policies_tenant_active ON resilience_alert_policies(tenant_id) WHERE active = true;

-- =====================================================
-- RESILIENCE LOGS (Task 15.4-T03)
-- =====================================================

CREATE TABLE IF NOT EXISTS resilience_logs (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Event context
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255),
    
    -- Event details
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    event_message TEXT NOT NULL,
    
    -- Context and action
    error_context JSONB DEFAULT '{}',
    action_taken TEXT,
    
    -- Correlation (Task 15.4-T07)
    correlation_id VARCHAR(255),
    parent_event_id UUID,
    
    -- Evidence reference
    evidence_ref VARCHAR(255),
    
    -- Digital signature (Task 15.4-T08)
    digital_signature VARCHAR(255) NOT NULL,
    hash_algorithm VARCHAR(50) NOT NULL DEFAULT 'SHA256',
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Storage information (Task 15.4-T09)
    storage_location VARCHAR(500),
    archived BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    retention_until DATE,
    
    -- Constraints
    CONSTRAINT chk_event_type CHECK (event_type IN ('retry_event', 'fallback_event', 'conflict_event', 'sla_breach', 'escalation_event', 'system_error', 'performance_degradation')),
    CONSTRAINT chk_severity CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    CONSTRAINT chk_hash_algorithm CHECK (hash_algorithm IN ('SHA256', 'SHA512', 'BLAKE2b'))
);

-- RLS for resilience logs (Task 15.4-T18)
ALTER TABLE resilience_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY resilience_logs_tenant_isolation ON resilience_logs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_resilience_logs_tenant_workflow ON resilience_logs(tenant_id, workflow_id);
CREATE INDEX idx_resilience_logs_tenant_event_type ON resilience_logs(tenant_id, event_type);
CREATE INDEX idx_resilience_logs_tenant_severity ON resilience_logs(tenant_id, severity);
CREATE INDEX idx_resilience_logs_correlation ON resilience_logs(tenant_id, correlation_id) WHERE correlation_id IS NOT NULL;
CREATE INDEX idx_resilience_logs_parent ON resilience_logs(parent_event_id) WHERE parent_event_id IS NOT NULL;
CREATE INDEX idx_resilience_logs_created ON resilience_logs(tenant_id, created_at DESC);
CREATE INDEX idx_resilience_logs_retention ON resilience_logs(retention_until) WHERE retention_until IS NOT NULL;

-- =====================================================
-- RESILIENCE ALERTS (Task 15.4-T21)
-- =====================================================

CREATE TABLE IF NOT EXISTS resilience_alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Alert details
    alert_type VARCHAR(50) NOT NULL,
    priority VARCHAR(10) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    
    -- Source information
    source_event_ids JSONB DEFAULT '[]',
    affected_workflows JSONB DEFAULT '[]',
    
    -- Routing (Task 15.4-T24)
    target_personas JSONB DEFAULT '[]',
    notification_channels JSONB DEFAULT '[]',
    routing_config JSONB DEFAULT '{}',
    
    -- SLA configuration (Task 15.4-T23)
    sla_minutes INTEGER NOT NULL DEFAULT 60,
    sla_deadline TIMESTAMP WITH TIME ZONE NOT NULL,
    auto_escalate BOOLEAN NOT NULL DEFAULT true,
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    acknowledged_by INTEGER,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_by INTEGER,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_alert_priority CHECK (priority IN ('P0', 'P1', 'P2', 'P3')),
    CONSTRAINT chk_alert_status CHECK (status IN ('open', 'acknowledged', 'resolved', 'closed', 'escalated')),
    CONSTRAINT chk_sla_minutes CHECK (sla_minutes > 0 AND sla_minutes <= 10080), -- Max 1 week
    CONSTRAINT chk_sla_deadline CHECK (sla_deadline > created_at)
);

-- RLS for resilience alerts
ALTER TABLE resilience_alerts ENABLE ROW LEVEL SECURITY;
CREATE POLICY resilience_alerts_tenant_isolation ON resilience_alerts
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_resilience_alerts_tenant_priority ON resilience_alerts(tenant_id, priority);
CREATE INDEX idx_resilience_alerts_tenant_status ON resilience_alerts(tenant_id, status);
CREATE INDEX idx_resilience_alerts_sla_deadline ON resilience_alerts(tenant_id, sla_deadline) WHERE status NOT IN ('resolved', 'closed');
CREATE INDEX idx_resilience_alerts_created ON resilience_alerts(tenant_id, created_at DESC);
CREATE INDEX idx_resilience_alerts_acknowledged ON resilience_alerts(tenant_id, acknowledged_by) WHERE acknowledged_by IS NOT NULL;

-- =====================================================
-- RESILIENCE METRICS (Task 15.4-T48)
-- =====================================================

CREATE TABLE IF NOT EXISTS resilience_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Time window
    metric_date DATE NOT NULL DEFAULT CURRENT_DATE,
    hour_bucket INTEGER, -- 0-23 for hourly metrics
    
    -- Retry metrics
    total_retries INTEGER NOT NULL DEFAULT 0,
    successful_retries INTEGER NOT NULL DEFAULT 0,
    failed_retries INTEGER NOT NULL DEFAULT 0,
    avg_retry_time_seconds DECIMAL(10,2),
    
    -- Fallback metrics
    total_fallbacks INTEGER NOT NULL DEFAULT 0,
    successful_fallbacks INTEGER NOT NULL DEFAULT 0,
    fallback_types JSONB DEFAULT '{}',
    
    -- Escalation metrics
    total_escalations INTEGER NOT NULL DEFAULT 0,
    resolved_escalations INTEGER NOT NULL DEFAULT 0,
    avg_resolution_time_hours DECIMAL(8,2),
    
    -- SLA metrics
    sla_breaches INTEGER NOT NULL DEFAULT 0,
    mttr_hours DECIMAL(8,2), -- Mean Time To Recovery
    mttd_hours DECIMAL(8,2), -- Mean Time To Detection
    
    -- Error metrics
    error_count_by_type JSONB DEFAULT '{}',
    critical_errors INTEGER NOT NULL DEFAULT 0,
    
    -- Performance metrics
    system_availability DECIMAL(5,4), -- 0.0000 to 1.0000
    error_rate DECIMAL(5,4),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_metrics_hour_bucket CHECK (hour_bucket IS NULL OR (hour_bucket >= 0 AND hour_bucket <= 23)),
    CONSTRAINT chk_metrics_counts CHECK (
        total_retries >= 0 AND 
        successful_retries >= 0 AND 
        failed_retries >= 0 AND 
        total_fallbacks >= 0 AND 
        successful_fallbacks >= 0 AND
        total_escalations >= 0 AND 
        resolved_escalations >= 0 AND
        sla_breaches >= 0 AND 
        critical_errors >= 0
    ),
    CONSTRAINT chk_system_availability CHECK (system_availability IS NULL OR (system_availability >= 0.0 AND system_availability <= 1.0)),
    CONSTRAINT chk_error_rate CHECK (error_rate IS NULL OR (error_rate >= 0.0 AND error_rate <= 1.0))
);

-- RLS for resilience metrics
ALTER TABLE resilience_metrics ENABLE ROW LEVEL SECURITY;
CREATE POLICY resilience_metrics_tenant_isolation ON resilience_metrics
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_resilience_metrics_tenant_date_hour ON resilience_metrics(tenant_id, metric_date, hour_bucket);
CREATE INDEX idx_resilience_metrics_date ON resilience_metrics(tenant_id, metric_date DESC);

-- =====================================================
-- EVENT CORRELATIONS (Task 15.4-T07)
-- =====================================================

CREATE TABLE IF NOT EXISTS resilience_event_correlations (
    correlation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Correlation details
    correlation_key VARCHAR(255) NOT NULL,
    correlation_value VARCHAR(255) NOT NULL,
    correlation_type VARCHAR(50) NOT NULL,
    
    -- Events in correlation
    event_ids JSONB NOT NULL DEFAULT '[]',
    correlation_strength DECIMAL(5,3) NOT NULL DEFAULT 0.0,
    
    -- Time window
    time_window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    time_window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Pattern information
    pattern_type VARCHAR(50),
    pattern_metadata JSONB DEFAULT '{}',
    
    -- Status
    active BOOLEAN NOT NULL DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_correlation_type CHECK (correlation_type IN ('retry_chain', 'escalation_chain', 'sla_breach_pattern', 'error_cascade')),
    CONSTRAINT chk_correlation_strength CHECK (correlation_strength >= 0.0 AND correlation_strength <= 1.0),
    CONSTRAINT chk_time_window CHECK (time_window_end >= time_window_start)
);

-- RLS for event correlations
ALTER TABLE resilience_event_correlations ENABLE ROW LEVEL SECURITY;
CREATE POLICY resilience_event_correlations_tenant_isolation ON resilience_event_correlations
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_event_correlations_tenant_key ON resilience_event_correlations(tenant_id, correlation_key, correlation_value);
CREATE INDEX idx_event_correlations_type ON resilience_event_correlations(tenant_id, correlation_type);
CREATE INDEX idx_event_correlations_time_window ON resilience_event_correlations(tenant_id, time_window_start, time_window_end);
CREATE INDEX idx_event_correlations_active ON resilience_event_correlations(tenant_id, active) WHERE active = true;

-- =====================================================
-- ALERT NOTIFICATIONS (Task 15.4-T24)
-- =====================================================

CREATE TABLE IF NOT EXISTS resilience_alert_notifications (
    notification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id UUID NOT NULL REFERENCES resilience_alerts(alert_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Notification details
    notification_channel VARCHAR(50) NOT NULL,
    target_persona VARCHAR(50) NOT NULL,
    recipient_identifier VARCHAR(255) NOT NULL,
    
    -- Message details
    message_title VARCHAR(500) NOT NULL,
    message_body TEXT NOT NULL,
    message_priority VARCHAR(10) NOT NULL,
    
    -- Delivery tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    sent_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    
    -- Retry information
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    next_retry_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    notification_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_notification_channel CHECK (notification_channel IN ('email', 'slack', 'sms', 'pagerduty', 'webhook')),
    CONSTRAINT chk_notification_status CHECK (status IN ('pending', 'sent', 'delivered', 'acknowledged', 'failed', 'cancelled')),
    CONSTRAINT chk_notification_priority CHECK (message_priority IN ('P0', 'P1', 'P2', 'P3')),
    CONSTRAINT chk_retry_count CHECK (retry_count >= 0 AND retry_count <= max_retries)
);

-- RLS for alert notifications
ALTER TABLE resilience_alert_notifications ENABLE ROW LEVEL SECURITY;
CREATE POLICY resilience_alert_notifications_tenant_isolation ON resilience_alert_notifications
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_alert_notifications_alert_id ON resilience_alert_notifications(alert_id, created_at DESC);
CREATE INDEX idx_alert_notifications_tenant_status ON resilience_alert_notifications(tenant_id, status);
CREATE INDEX idx_alert_notifications_channel ON resilience_alert_notifications(tenant_id, notification_channel);
CREATE INDEX idx_alert_notifications_retry ON resilience_alert_notifications(tenant_id, next_retry_at) WHERE status = 'pending' AND next_retry_at IS NOT NULL;

-- =====================================================
-- LOG RETENTION JOBS (Task 15.4-T20)
-- =====================================================

CREATE TABLE IF NOT EXISTS resilience_log_retention_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Job configuration
    retention_policy VARCHAR(50) NOT NULL,
    retention_days INTEGER NOT NULL,
    
    -- Execution details
    job_type VARCHAR(50) NOT NULL DEFAULT 'cleanup',
    target_tables JSONB NOT NULL DEFAULT '[]',
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'scheduled',
    scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Results
    records_processed INTEGER DEFAULT 0,
    records_deleted INTEGER DEFAULT 0,
    records_archived INTEGER DEFAULT 0,
    
    -- Error handling
    error_message TEXT,
    
    -- Metadata
    job_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_retention_policy CHECK (retention_policy IN ('bronze_90d', 'silver_400d', 'gold_2y', 'sox_7y', 'rbi_10y', 'custom')),
    CONSTRAINT chk_job_type CHECK (job_type IN ('cleanup', 'archive', 'migrate')),
    CONSTRAINT chk_job_status CHECK (status IN ('scheduled', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_retention_days CHECK (retention_days > 0),
    CONSTRAINT chk_records_counts CHECK (
        records_processed >= 0 AND 
        records_deleted >= 0 AND 
        records_archived >= 0
    )
);

-- RLS for retention jobs
ALTER TABLE resilience_log_retention_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY resilience_log_retention_jobs_tenant_isolation ON resilience_log_retention_jobs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_log_retention_jobs_tenant_status ON resilience_log_retention_jobs(tenant_id, status);
CREATE INDEX idx_log_retention_jobs_scheduled ON resilience_log_retention_jobs(tenant_id, scheduled_at);
CREATE INDEX idx_log_retention_jobs_policy ON resilience_log_retention_jobs(tenant_id, retention_policy);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to auto-escalate overdue alerts
CREATE OR REPLACE FUNCTION auto_escalate_overdue_alerts()
RETURNS INTEGER AS $$
DECLARE
    escalated_count INTEGER;
BEGIN
    -- Find overdue alerts that should auto-escalate
    UPDATE resilience_alerts
    SET status = 'escalated',
        updated_at = NOW()
    WHERE status IN ('open', 'acknowledged')
    AND auto_escalate = true
    AND sla_deadline <= NOW()
    AND tenant_id = current_setting('app.current_tenant_id')::integer;
    
    GET DIAGNOSTICS escalated_count = ROW_COUNT;
    
    RETURN escalated_count;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate resilience metrics
CREATE OR REPLACE FUNCTION aggregate_resilience_metrics()
RETURNS INTEGER AS $$
DECLARE
    aggregated_count INTEGER;
BEGIN
    -- Aggregate daily metrics from logs
    INSERT INTO resilience_metrics (
        tenant_id, metric_date,
        total_retries, successful_retries, failed_retries,
        total_fallbacks, successful_fallbacks,
        total_escalations, resolved_escalations,
        sla_breaches, critical_errors
    )
    SELECT 
        rl.tenant_id,
        DATE(rl.created_at) as metric_date,
        COUNT(CASE WHEN rl.event_type = 'retry_event' THEN 1 END) as total_retries,
        COUNT(CASE WHEN rl.event_type = 'retry_event' AND rl.action_taken LIKE '%success%' THEN 1 END) as successful_retries,
        COUNT(CASE WHEN rl.event_type = 'retry_event' AND rl.action_taken LIKE '%failed%' THEN 1 END) as failed_retries,
        COUNT(CASE WHEN rl.event_type = 'fallback_event' THEN 1 END) as total_fallbacks,
        COUNT(CASE WHEN rl.event_type = 'fallback_event' AND rl.action_taken LIKE '%success%' THEN 1 END) as successful_fallbacks,
        COUNT(CASE WHEN rl.event_type = 'escalation_event' THEN 1 END) as total_escalations,
        COUNT(CASE WHEN rl.event_type = 'escalation_event' AND rl.action_taken LIKE '%resolved%' THEN 1 END) as resolved_escalations,
        COUNT(CASE WHEN rl.event_type = 'sla_breach' THEN 1 END) as sla_breaches,
        COUNT(CASE WHEN rl.severity = 'critical' THEN 1 END) as critical_errors
    FROM resilience_logs rl
    WHERE DATE(rl.created_at) = CURRENT_DATE - INTERVAL '1 day'
    AND NOT EXISTS (
        SELECT 1 FROM resilience_metrics rm 
        WHERE rm.tenant_id = rl.tenant_id 
        AND rm.metric_date = DATE(rl.created_at)
        AND rm.hour_bucket IS NULL
    )
    GROUP BY rl.tenant_id, DATE(rl.created_at)
    ON CONFLICT (tenant_id, metric_date, hour_bucket) DO UPDATE SET
        total_retries = EXCLUDED.total_retries,
        successful_retries = EXCLUDED.successful_retries,
        failed_retries = EXCLUDED.failed_retries,
        total_fallbacks = EXCLUDED.total_fallbacks,
        successful_fallbacks = EXCLUDED.successful_fallbacks,
        total_escalations = EXCLUDED.total_escalations,
        resolved_escalations = EXCLUDED.resolved_escalations,
        sla_breaches = EXCLUDED.sla_breaches,
        critical_errors = EXCLUDED.critical_errors,
        updated_at = NOW();
    
    GET DIAGNOSTICS aggregated_count = ROW_COUNT;
    RETURN aggregated_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old resilience data
CREATE OR REPLACE FUNCTION cleanup_old_resilience_data()
RETURNS INTEGER AS $$
DECLARE
    cleanup_count INTEGER;
BEGIN
    -- Clean up old logs based on retention policies
    DELETE FROM resilience_logs 
    WHERE retention_until IS NOT NULL 
    AND retention_until < CURRENT_DATE;
    
    GET DIAGNOSTICS cleanup_count = ROW_COUNT;
    
    -- Clean up old alerts (keep for 1 year)
    DELETE FROM resilience_alerts 
    WHERE created_at < NOW() - INTERVAL '1 year'
    AND status IN ('resolved', 'closed');
    
    -- Clean up old metrics (keep for 2 years)
    DELETE FROM resilience_metrics 
    WHERE metric_date < CURRENT_DATE - INTERVAL '2 years';
    
    -- Clean up old correlations (keep for 90 days)
    DELETE FROM resilience_event_correlations 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    RETURN cleanup_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update log retention dates
CREATE OR REPLACE FUNCTION update_log_retention_dates()
RETURNS TRIGGER AS $$
DECLARE
    retention_days INTEGER;
    industry_code VARCHAR(50);
BEGIN
    -- Get retention policy for tenant
    SELECT 
        rlp.retention_days,
        rlp.industry_code
    INTO retention_days, industry_code
    FROM resilience_logging_policies rlp
    WHERE rlp.tenant_id = NEW.tenant_id 
    AND rlp.active = true
    ORDER BY rlp.created_at DESC 
    LIMIT 1;
    
    -- Set retention date based on policy
    IF retention_days IS NOT NULL THEN
        NEW.retention_until := CURRENT_DATE + (retention_days || ' days')::INTERVAL;
    ELSE
        -- Default retention based on severity
        CASE NEW.severity
            WHEN 'critical' THEN NEW.retention_until := CURRENT_DATE + INTERVAL '2 years';
            WHEN 'error' THEN NEW.retention_until := CURRENT_DATE + INTERVAL '1 year';
            ELSE NEW.retention_until := CURRENT_DATE + INTERVAL '90 days';
        END CASE;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic retention date setting
CREATE TRIGGER trigger_update_log_retention_dates
    BEFORE INSERT ON resilience_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_log_retention_dates();

-- Comments for documentation
COMMENT ON TABLE resilience_logging_policies IS 'Resilience logging policies with industry-specific configurations (Task 15.4-T17, T19)';
COMMENT ON TABLE resilience_alert_policies IS 'Resilience alert policies with suppression and routing rules (Task 15.4-T22, T25)';
COMMENT ON TABLE resilience_logs IS 'Centralized resilience event logs with digital signatures (Task 15.4-T03, T08, T18)';
COMMENT ON TABLE resilience_alerts IS 'Resilience alerts with persona-based routing (Task 15.4-T21, T24)';
COMMENT ON TABLE resilience_metrics IS 'Aggregated resilience metrics for dashboards (Task 15.4-T48)';
COMMENT ON TABLE resilience_event_correlations IS 'Event correlation tracking for pattern detection (Task 15.4-T07)';
COMMENT ON TABLE resilience_alert_notifications IS 'Alert notification delivery tracking (Task 15.4-T24)';
COMMENT ON TABLE resilience_log_retention_jobs IS 'Log retention and cleanup job tracking (Task 15.4-T20)';

COMMENT ON FUNCTION auto_escalate_overdue_alerts() IS 'Automatically escalate overdue alerts based on SLA deadlines';
COMMENT ON FUNCTION aggregate_resilience_metrics() IS 'Aggregate resilience metrics from logs for daily reporting';
COMMENT ON FUNCTION cleanup_old_resilience_data() IS 'Clean up old resilience data based on retention policies';
COMMENT ON FUNCTION update_log_retention_dates() IS 'Automatically set retention dates for new log entries based on policy';
