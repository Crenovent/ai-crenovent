-- Chapter 15.1 Retry Engine Database Schemas
-- ============================================
-- Tasks 15.1-T33, T56: Retry logs storage and persistence

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- RETRY POLICIES (Task 15.1-T11)
-- =====================================================

CREATE TABLE IF NOT EXISTS retry_policies (
    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Policy configuration
    tenant_tier VARCHAR(10) NOT NULL DEFAULT 'T1',
    industry_code VARCHAR(50) NOT NULL DEFAULT 'SaaS',
    
    -- Retry limits
    max_attempts INTEGER NOT NULL DEFAULT 5,
    base_backoff_seconds DECIMAL(10,2) NOT NULL DEFAULT 30.0,
    max_backoff_seconds DECIMAL(10,2) NOT NULL DEFAULT 300.0,
    backoff_curve VARCHAR(20) NOT NULL DEFAULT 'exponential',
    backoff_factor DECIMAL(5,2) NOT NULL DEFAULT 2.0,
    
    -- Jitter configuration
    jitter_enabled BOOLEAN NOT NULL DEFAULT true,
    jitter_factor DECIMAL(5,3) NOT NULL DEFAULT 0.1,
    
    -- SLA configuration
    sla_window_hours INTEGER NOT NULL DEFAULT 24,
    escalation_threshold INTEGER NOT NULL DEFAULT 3,
    
    -- Suppression rules
    suppress_permanent_errors BOOLEAN NOT NULL DEFAULT true,
    suppress_compliance_errors BOOLEAN NOT NULL DEFAULT true,
    
    -- Policy metadata
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by_user_id INTEGER,
    
    -- Constraints
    CONSTRAINT chk_tenant_tier CHECK (tenant_tier IN ('T0', 'T1', 'T2')),
    CONSTRAINT chk_industry_code CHECK (industry_code IN ('SaaS', 'Banking', 'Insurance', 'Healthcare', 'E-commerce', 'FinTech')),
    CONSTRAINT chk_backoff_curve CHECK (backoff_curve IN ('linear', 'exponential', 'capped')),
    CONSTRAINT chk_max_attempts CHECK (max_attempts > 0 AND max_attempts <= 20),
    CONSTRAINT chk_backoff_seconds CHECK (base_backoff_seconds > 0 AND max_backoff_seconds >= base_backoff_seconds),
    CONSTRAINT chk_backoff_factor CHECK (backoff_factor >= 1.0 AND backoff_factor <= 10.0),
    CONSTRAINT chk_jitter_factor CHECK (jitter_factor >= 0.0 AND jitter_factor <= 1.0),
    CONSTRAINT chk_sla_window CHECK (sla_window_hours > 0 AND sla_window_hours <= 168), -- Max 1 week
    CONSTRAINT chk_escalation_threshold CHECK (escalation_threshold > 0)
);

-- RLS for retry policies
ALTER TABLE retry_policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY retry_policies_tenant_isolation ON retry_policies
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_retry_policies_tenant_active ON retry_policies(tenant_id) WHERE active = true;
CREATE INDEX idx_retry_policies_industry ON retry_policies(tenant_id, industry_code);
CREATE INDEX idx_retry_policies_tier ON retry_policies(tenant_id, tenant_tier);

-- =====================================================
-- RETRY REQUESTS (Task 15.1-T56)
-- =====================================================

CREATE TABLE IF NOT EXISTS retry_requests (
    retry_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Request context
    workflow_id VARCHAR(255) NOT NULL,
    step_id VARCHAR(255) NOT NULL,
    user_id INTEGER NOT NULL,
    
    -- Error information
    error_type VARCHAR(20) NOT NULL,
    error_message TEXT NOT NULL,
    error_code VARCHAR(100),
    
    -- Idempotency (Task 15.1-T06)
    idempotency_key VARCHAR(255) NOT NULL,
    
    -- Retry configuration
    max_attempts INTEGER NOT NULL,
    backoff_factor DECIMAL(5,2) NOT NULL DEFAULT 2.0,
    jitter_enabled BOOLEAN NOT NULL DEFAULT true,
    
    -- Retry schedule (pre-calculated)
    retry_schedule JSONB NOT NULL DEFAULT '[]',
    
    -- Operation data
    operation_data JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    current_attempt INTEGER NOT NULL DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_error_type CHECK (error_type IN ('transient', 'permanent', 'compliance', 'manual')),
    CONSTRAINT chk_retry_status CHECK (status IN ('pending', 'in_progress', 'success', 'failed', 'exhausted', 'escalated')),
    CONSTRAINT chk_max_attempts_limit CHECK (max_attempts > 0 AND max_attempts <= 20),
    CONSTRAINT chk_current_attempt CHECK (current_attempt >= 0 AND current_attempt <= max_attempts)
);

-- RLS for retry requests
ALTER TABLE retry_requests ENABLE ROW LEVEL SECURITY;
CREATE POLICY retry_requests_tenant_isolation ON retry_requests
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_retry_requests_tenant_workflow ON retry_requests(tenant_id, workflow_id);
CREATE INDEX idx_retry_requests_tenant_status ON retry_requests(tenant_id, status);
CREATE INDEX idx_retry_requests_idempotency ON retry_requests(tenant_id, idempotency_key);
CREATE INDEX idx_retry_requests_created ON retry_requests(tenant_id, created_at DESC);
CREATE INDEX idx_retry_requests_error_type ON retry_requests(tenant_id, error_type);

-- =====================================================
-- RETRY ATTEMPTS (Task 15.1-T56)
-- =====================================================

CREATE TABLE IF NOT EXISTS retry_attempts (
    attempt_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    retry_id UUID NOT NULL REFERENCES retry_requests(retry_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Attempt details
    attempt_number INTEGER NOT NULL,
    scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
    executed_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Backoff information
    backoff_seconds DECIMAL(10,2) NOT NULL,
    jitter_applied DECIMAL(10,2) DEFAULT 0.0,
    
    -- Attempt result
    status VARCHAR(20) NOT NULL DEFAULT 'scheduled',
    error_message TEXT,
    result_data JSONB,
    
    -- Evidence tracking (Task 15.1-T14)
    evidence_pack_id UUID,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_attempt_status CHECK (status IN ('scheduled', 'in_progress', 'success', 'failed', 'skipped')),
    CONSTRAINT chk_attempt_number CHECK (attempt_number > 0),
    CONSTRAINT chk_backoff_seconds CHECK (backoff_seconds >= 0)
);

-- RLS for retry attempts
ALTER TABLE retry_attempts ENABLE ROW LEVEL SECURITY;
CREATE POLICY retry_attempts_tenant_isolation ON retry_attempts
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_retry_attempts_retry_id ON retry_attempts(retry_id, attempt_number);
CREATE INDEX idx_retry_attempts_tenant_status ON retry_attempts(tenant_id, status);
CREATE INDEX idx_retry_attempts_scheduled ON retry_attempts(tenant_id, scheduled_at);
CREATE INDEX idx_retry_attempts_executed ON retry_attempts(tenant_id, executed_at) WHERE executed_at IS NOT NULL;

-- =====================================================
-- RETRY SCHEDULED ATTEMPTS (Task 15.1-T08)
-- =====================================================

CREATE TABLE IF NOT EXISTS retry_scheduled_attempts (
    schedule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    retry_id UUID NOT NULL REFERENCES retry_requests(retry_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Schedule details
    attempt_number INTEGER NOT NULL,
    scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Backoff configuration
    backoff_seconds DECIMAL(10,2) NOT NULL,
    jitter_applied DECIMAL(10,2) DEFAULT 0.0,
    
    -- Schedule status
    status VARCHAR(20) NOT NULL DEFAULT 'scheduled',
    processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_schedule_status CHECK (status IN ('scheduled', 'processing', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_schedule_attempt_number CHECK (attempt_number > 0)
);

-- RLS for scheduled attempts
ALTER TABLE retry_scheduled_attempts ENABLE ROW LEVEL SECURITY;
CREATE POLICY retry_scheduled_attempts_tenant_isolation ON retry_scheduled_attempts
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_retry_scheduled_retry_id ON retry_scheduled_attempts(retry_id, attempt_number);
CREATE INDEX idx_retry_scheduled_tenant_status ON retry_scheduled_attempts(tenant_id, status);
CREATE INDEX idx_retry_scheduled_due ON retry_scheduled_attempts(tenant_id, scheduled_at) WHERE status = 'scheduled';

-- =====================================================
-- RETRY EVIDENCE PACKS (Task 15.1-T13, T14, T15)
-- =====================================================

CREATE TABLE IF NOT EXISTS retry_evidence_packs (
    evidence_pack_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    retry_id UUID NOT NULL REFERENCES retry_requests(retry_id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    
    -- Evidence details
    event_type VARCHAR(50) NOT NULL,
    evidence_data JSONB NOT NULL,
    
    -- Digital signature (Task 15.1-T15)
    digital_signature VARCHAR(255) NOT NULL,
    hash_algorithm VARCHAR(50) NOT NULL DEFAULT 'SHA256',
    
    -- Storage information
    storage_location VARCHAR(500),
    compressed BOOLEAN DEFAULT false,
    compression_algorithm VARCHAR(20),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    retention_until DATE,
    
    -- Constraints
    CONSTRAINT chk_event_type CHECK (event_type IN ('retry_created', 'retry_attempt', 'retry_success', 'retry_failed', 'retry_exhausted', 'retry_escalated')),
    CONSTRAINT chk_hash_algorithm CHECK (hash_algorithm IN ('SHA256', 'SHA512', 'BLAKE2b'))
);

-- RLS for evidence packs
ALTER TABLE retry_evidence_packs ENABLE ROW LEVEL SECURITY;
CREATE POLICY retry_evidence_packs_tenant_isolation ON retry_evidence_packs
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE INDEX idx_retry_evidence_retry_id ON retry_evidence_packs(retry_id, created_at DESC);
CREATE INDEX idx_retry_evidence_tenant_type ON retry_evidence_packs(tenant_id, event_type);
CREATE INDEX idx_retry_evidence_created ON retry_evidence_packs(tenant_id, created_at DESC);
CREATE INDEX idx_retry_evidence_retention ON retry_evidence_packs(retention_until) WHERE retention_until IS NOT NULL;

-- =====================================================
-- RETRY QUOTAS (Task 15.1-T24)
-- =====================================================

CREATE TABLE IF NOT EXISTS retry_quotas (
    quota_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Quota configuration
    quota_type VARCHAR(50) NOT NULL DEFAULT 'daily',
    quota_period VARCHAR(20) NOT NULL DEFAULT '24h',
    
    -- Limits
    max_retries_per_period INTEGER NOT NULL DEFAULT 1000,
    max_retries_per_workflow INTEGER NOT NULL DEFAULT 50,
    max_concurrent_retries INTEGER NOT NULL DEFAULT 10,
    
    -- Current usage
    current_period_retries INTEGER NOT NULL DEFAULT 0,
    current_concurrent_retries INTEGER NOT NULL DEFAULT 0,
    
    -- Period tracking
    period_start TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    period_end TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() + INTERVAL '24 hours'),
    
    -- Status
    active BOOLEAN NOT NULL DEFAULT true,
    enforcement_level VARCHAR(20) NOT NULL DEFAULT 'warn',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_quota_type CHECK (quota_type IN ('daily', 'hourly', 'weekly', 'monthly')),
    CONSTRAINT chk_quota_period CHECK (quota_period IN ('1h', '24h', '7d', '30d')),
    CONSTRAINT chk_enforcement_level CHECK (enforcement_level IN ('warn', 'throttle', 'block')),
    CONSTRAINT chk_quota_limits CHECK (
        max_retries_per_period > 0 AND 
        max_retries_per_workflow > 0 AND 
        max_concurrent_retries > 0
    ),
    CONSTRAINT chk_current_usage CHECK (
        current_period_retries >= 0 AND 
        current_concurrent_retries >= 0
    )
);

-- RLS for retry quotas
ALTER TABLE retry_quotas ENABLE ROW LEVEL SECURITY;
CREATE POLICY retry_quotas_tenant_isolation ON retry_quotas
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_retry_quotas_tenant_type ON retry_quotas(tenant_id, quota_type) WHERE active = true;
CREATE INDEX idx_retry_quotas_period ON retry_quotas(tenant_id, period_start, period_end);
CREATE INDEX idx_retry_quotas_enforcement ON retry_quotas(tenant_id, enforcement_level);

-- =====================================================
-- RETRY METRICS (Task 15.1-T34)
-- =====================================================

CREATE TABLE IF NOT EXISTS retry_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    
    -- Metric context
    workflow_id VARCHAR(255),
    step_id VARCHAR(255),
    error_type VARCHAR(20),
    
    -- Time window
    metric_date DATE NOT NULL DEFAULT CURRENT_DATE,
    hour_bucket INTEGER, -- 0-23 for hourly metrics
    
    -- Retry metrics
    total_retries INTEGER NOT NULL DEFAULT 0,
    successful_retries INTEGER NOT NULL DEFAULT 0,
    failed_retries INTEGER NOT NULL DEFAULT 0,
    exhausted_retries INTEGER NOT NULL DEFAULT 0,
    
    -- Timing metrics
    avg_backoff_seconds DECIMAL(10,2),
    min_backoff_seconds DECIMAL(10,2),
    max_backoff_seconds DECIMAL(10,2),
    avg_attempts_to_success DECIMAL(5,2),
    
    -- Success rate
    success_rate DECIMAL(5,4), -- 0.0000 to 1.0000
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_hour_bucket CHECK (hour_bucket IS NULL OR (hour_bucket >= 0 AND hour_bucket <= 23)),
    CONSTRAINT chk_retry_counts CHECK (
        total_retries >= 0 AND 
        successful_retries >= 0 AND 
        failed_retries >= 0 AND 
        exhausted_retries >= 0 AND
        (successful_retries + failed_retries + exhausted_retries) <= total_retries
    ),
    CONSTRAINT chk_success_rate CHECK (success_rate IS NULL OR (success_rate >= 0.0 AND success_rate <= 1.0))
);

-- RLS for retry metrics
ALTER TABLE retry_metrics ENABLE ROW LEVEL SECURITY;
CREATE POLICY retry_metrics_tenant_isolation ON retry_metrics
    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id')::integer);

-- Indexes
CREATE UNIQUE INDEX idx_retry_metrics_tenant_date_hour ON retry_metrics(tenant_id, metric_date, hour_bucket, workflow_id, error_type);
CREATE INDEX idx_retry_metrics_date ON retry_metrics(tenant_id, metric_date DESC);
CREATE INDEX idx_retry_metrics_workflow ON retry_metrics(tenant_id, workflow_id, metric_date DESC);
CREATE INDEX idx_retry_metrics_error_type ON retry_metrics(tenant_id, error_type, metric_date DESC);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update retry quotas
CREATE OR REPLACE FUNCTION update_retry_quota_usage()
RETURNS TRIGGER AS $$
BEGIN
    -- Update quota usage when retry is created
    IF TG_OP = 'INSERT' THEN
        UPDATE retry_quotas
        SET current_period_retries = current_period_retries + 1,
            updated_at = NOW()
        WHERE tenant_id = NEW.tenant_id
        AND active = true
        AND period_start <= NOW()
        AND period_end > NOW();
        
        RETURN NEW;
    END IF;
    
    -- Update concurrent retries when status changes
    IF TG_OP = 'UPDATE' AND OLD.status != NEW.status THEN
        IF NEW.status = 'in_progress' THEN
            UPDATE retry_quotas
            SET current_concurrent_retries = current_concurrent_retries + 1,
                updated_at = NOW()
            WHERE tenant_id = NEW.tenant_id AND active = true;
        ELSIF OLD.status = 'in_progress' AND NEW.status IN ('success', 'failed', 'exhausted', 'escalated') THEN
            UPDATE retry_quotas
            SET current_concurrent_retries = GREATEST(0, current_concurrent_retries - 1),
                updated_at = NOW()
            WHERE tenant_id = NEW.tenant_id AND active = true;
        END IF;
        
        RETURN NEW;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Trigger for quota usage updates
CREATE TRIGGER trigger_update_retry_quota_usage
    AFTER INSERT OR UPDATE ON retry_requests
    FOR EACH ROW
    EXECUTE FUNCTION update_retry_quota_usage();

-- Function to reset quota counters
CREATE OR REPLACE FUNCTION reset_retry_quota_counters()
RETURNS INTEGER AS $$
DECLARE
    reset_count INTEGER;
BEGIN
    -- Reset daily quotas
    UPDATE retry_quotas
    SET current_period_retries = 0,
        period_start = CURRENT_DATE,
        period_end = CURRENT_DATE + INTERVAL '24 hours',
        updated_at = NOW()
    WHERE quota_type = 'daily'
    AND period_end <= NOW();
    
    GET DIAGNOSTICS reset_count = ROW_COUNT;
    
    -- Reset hourly quotas
    UPDATE retry_quotas
    SET current_period_retries = 0,
        period_start = DATE_TRUNC('hour', NOW()),
        period_end = DATE_TRUNC('hour', NOW()) + INTERVAL '1 hour',
        updated_at = NOW()
    WHERE quota_type = 'hourly'
    AND period_end <= NOW();
    
    -- Reset weekly quotas
    UPDATE retry_quotas
    SET current_period_retries = 0,
        period_start = DATE_TRUNC('week', NOW()),
        period_end = DATE_TRUNC('week', NOW()) + INTERVAL '7 days',
        updated_at = NOW()
    WHERE quota_type = 'weekly'
    AND period_end <= NOW();
    
    -- Reset monthly quotas
    UPDATE retry_quotas
    SET current_period_retries = 0,
        period_start = DATE_TRUNC('month', NOW()),
        period_end = DATE_TRUNC('month', NOW()) + INTERVAL '1 month',
        updated_at = NOW()
    WHERE quota_type = 'monthly'
    AND period_end <= NOW();
    
    RETURN reset_count;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate retry metrics
CREATE OR REPLACE FUNCTION aggregate_retry_metrics()
RETURNS INTEGER AS $$
DECLARE
    aggregated_count INTEGER;
BEGIN
    -- Aggregate daily metrics
    INSERT INTO retry_metrics (
        tenant_id, workflow_id, step_id, error_type, metric_date,
        total_retries, successful_retries, failed_retries, exhausted_retries,
        avg_backoff_seconds, min_backoff_seconds, max_backoff_seconds,
        avg_attempts_to_success, success_rate
    )
    SELECT 
        rr.tenant_id,
        rr.workflow_id,
        rr.step_id,
        rr.error_type,
        DATE(rr.created_at) as metric_date,
        COUNT(*) as total_retries,
        COUNT(CASE WHEN rr.status = 'success' THEN 1 END) as successful_retries,
        COUNT(CASE WHEN rr.status = 'failed' THEN 1 END) as failed_retries,
        COUNT(CASE WHEN rr.status = 'exhausted' THEN 1 END) as exhausted_retries,
        AVG(ra.backoff_seconds) as avg_backoff_seconds,
        MIN(ra.backoff_seconds) as min_backoff_seconds,
        MAX(ra.backoff_seconds) as max_backoff_seconds,
        AVG(rr.current_attempt) as avg_attempts_to_success,
        CASE 
            WHEN COUNT(*) > 0 THEN COUNT(CASE WHEN rr.status = 'success' THEN 1 END)::DECIMAL / COUNT(*)
            ELSE 0.0
        END as success_rate
    FROM retry_requests rr
    LEFT JOIN retry_attempts ra ON rr.retry_id = ra.retry_id
    WHERE DATE(rr.created_at) = CURRENT_DATE - INTERVAL '1 day'
    AND NOT EXISTS (
        SELECT 1 FROM retry_metrics rm 
        WHERE rm.tenant_id = rr.tenant_id 
        AND rm.workflow_id = rr.workflow_id 
        AND rm.error_type = rr.error_type
        AND rm.metric_date = DATE(rr.created_at)
        AND rm.hour_bucket IS NULL
    )
    GROUP BY rr.tenant_id, rr.workflow_id, rr.step_id, rr.error_type, DATE(rr.created_at)
    ON CONFLICT (tenant_id, metric_date, hour_bucket, workflow_id, error_type) DO UPDATE SET
        total_retries = EXCLUDED.total_retries,
        successful_retries = EXCLUDED.successful_retries,
        failed_retries = EXCLUDED.failed_retries,
        exhausted_retries = EXCLUDED.exhausted_retries,
        avg_backoff_seconds = EXCLUDED.avg_backoff_seconds,
        min_backoff_seconds = EXCLUDED.min_backoff_seconds,
        max_backoff_seconds = EXCLUDED.max_backoff_seconds,
        avg_attempts_to_success = EXCLUDED.avg_attempts_to_success,
        success_rate = EXCLUDED.success_rate,
        updated_at = NOW();
    
    GET DIAGNOSTICS aggregated_count = ROW_COUNT;
    RETURN aggregated_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old retry data
CREATE OR REPLACE FUNCTION cleanup_old_retry_data()
RETURNS INTEGER AS $$
DECLARE
    cleanup_count INTEGER;
BEGIN
    -- Clean up old retry requests (keep for 90 days)
    DELETE FROM retry_requests 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS cleanup_count = ROW_COUNT;
    
    -- Clean up old evidence packs based on retention policy
    DELETE FROM retry_evidence_packs 
    WHERE retention_until IS NOT NULL 
    AND retention_until < CURRENT_DATE;
    
    -- Clean up old metrics (keep for 2 years)
    DELETE FROM retry_metrics 
    WHERE metric_date < CURRENT_DATE - INTERVAL '2 years';
    
    RETURN cleanup_count;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE retry_policies IS 'Retry policies per tenant with industry-specific configurations (Task 15.1-T11)';
COMMENT ON TABLE retry_requests IS 'Retry requests with idempotency keys and policy enforcement (Task 15.1-T56)';
COMMENT ON TABLE retry_attempts IS 'Individual retry attempts with backoff and evidence tracking (Task 15.1-T56)';
COMMENT ON TABLE retry_scheduled_attempts IS 'Scheduled retry attempts for job queue processing (Task 15.1-T08)';
COMMENT ON TABLE retry_evidence_packs IS 'Evidence packs for retry events with digital signatures (Task 15.1-T13, T14, T15)';
COMMENT ON TABLE retry_quotas IS 'Tenant-level retry quotas and usage tracking (Task 15.1-T24)';
COMMENT ON TABLE retry_metrics IS 'Aggregated retry metrics for monitoring and analytics (Task 15.1-T34)';

COMMENT ON FUNCTION update_retry_quota_usage() IS 'Update retry quota usage counters on retry creation and status changes';
COMMENT ON FUNCTION reset_retry_quota_counters() IS 'Reset quota counters based on period configuration';
COMMENT ON FUNCTION aggregate_retry_metrics() IS 'Aggregate retry metrics for daily reporting';
COMMENT ON FUNCTION cleanup_old_retry_data() IS 'Clean up old retry data based on retention policies';
