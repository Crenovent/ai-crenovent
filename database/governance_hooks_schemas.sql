-- Chapter 8.5 Governance Hooks Database Schemas
-- Tasks 8.5.1-8.5.42: Comprehensive governance enforcement infrastructure

-- Enable Row Level Security and required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================
-- Core Governance Event Logging (Task 8.5.6, 8.5.7)
-- =============================================

-- Governance event logs for audit trail
CREATE TABLE IF NOT EXISTS governance_event_logs (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    plane VARCHAR(50) NOT NULL, -- control, execution, data, governance
    hook_type VARCHAR(50) NOT NULL, -- policy, consent, sla, finops, lineage, trust, sod, residency, anomaly
    decision VARCHAR(50) NOT NULL, -- allow, deny, override_required, escalate
    workflow_id VARCHAR(255) NOT NULL,
    execution_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    user_role VARCHAR(100) NOT NULL,
    violations_count INTEGER DEFAULT 0,
    execution_time_ms DECIMAL(10,2) DEFAULT 0,
    trust_score_impact DECIMAL(5,3) DEFAULT 0,
    evidence_pack_id UUID,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Indexes for performance
    CONSTRAINT governance_event_logs_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Indexes for governance event logs
CREATE INDEX IF NOT EXISTS idx_governance_event_logs_tenant_id ON governance_event_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_event_logs_timestamp ON governance_event_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_governance_event_logs_plane_hook ON governance_event_logs(plane, hook_type);
CREATE INDEX IF NOT EXISTS idx_governance_event_logs_decision ON governance_event_logs(decision);
CREATE INDEX IF NOT EXISTS idx_governance_event_logs_workflow ON governance_event_logs(workflow_id, execution_id);

-- Row Level Security for governance event logs
ALTER TABLE governance_event_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_event_logs_tenant_isolation ON governance_event_logs
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- Evidence Pack Storage (Task 8.5.7)
-- =============================================

-- Evidence packs for immutable compliance artifacts
CREATE TABLE IF NOT EXISTS governance_evidence_packs (
    evidence_pack_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    plane VARCHAR(50) NOT NULL,
    hook_type VARCHAR(50) NOT NULL,
    decision VARCHAR(50) NOT NULL,
    violations_count INTEGER DEFAULT 0,
    evidence_data JSONB NOT NULL,
    digital_signature TEXT,
    hash_chain_previous VARCHAR(64),
    hash_chain_current VARCHAR(64),
    worm_storage_path TEXT,
    retention_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT governance_evidence_packs_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Indexes for evidence packs
CREATE INDEX IF NOT EXISTS idx_governance_evidence_packs_tenant_id ON governance_evidence_packs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_evidence_packs_created_at ON governance_evidence_packs(created_at);
CREATE INDEX IF NOT EXISTS idx_governance_evidence_packs_plane_hook ON governance_evidence_packs(plane, hook_type);
CREATE INDEX IF NOT EXISTS idx_governance_evidence_packs_violations ON governance_evidence_packs(violations_count);

-- GIN index for JSONB evidence data
CREATE INDEX IF NOT EXISTS idx_governance_evidence_packs_data ON governance_evidence_packs USING GIN (evidence_data);

-- Row Level Security for evidence packs
ALTER TABLE governance_evidence_packs ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_evidence_packs_tenant_isolation ON governance_evidence_packs
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- Override Ledger Integration (Task 8.5.6)
-- =============================================

-- Governance override ledger for human-in-loop interventions
CREATE TABLE IF NOT EXISTS governance_override_ledger (
    override_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    workflow_id VARCHAR(255) NOT NULL,
    execution_id VARCHAR(255) NOT NULL,
    governance_hook VARCHAR(50) NOT NULL,
    violations JSONB NOT NULL,
    required_approver_role VARCHAR(100) NOT NULL,
    approver_user_id VARCHAR(255),
    approval_timestamp TIMESTAMPTZ,
    status VARCHAR(50) NOT NULL DEFAULT 'pending_approval', -- pending_approval, approved, denied, expired
    justification TEXT,
    evidence_pack_id UUID,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT governance_override_ledger_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    CONSTRAINT governance_override_ledger_evidence_pack_fkey FOREIGN KEY (evidence_pack_id) REFERENCES governance_evidence_packs(evidence_pack_id)
);

-- Indexes for override ledger
CREATE INDEX IF NOT EXISTS idx_governance_override_ledger_tenant_id ON governance_override_ledger(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_override_ledger_status ON governance_override_ledger(status);
CREATE INDEX IF NOT EXISTS idx_governance_override_ledger_workflow ON governance_override_ledger(workflow_id, execution_id);
CREATE INDEX IF NOT EXISTS idx_governance_override_ledger_hook ON governance_override_ledger(governance_hook);
CREATE INDEX IF NOT EXISTS idx_governance_override_ledger_expires_at ON governance_override_ledger(expires_at);

-- Row Level Security for override ledger
ALTER TABLE governance_override_ledger ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_override_ledger_tenant_isolation ON governance_override_ledger
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- SLA and Error Budget Tracking (Task 8.5.8)
-- =============================================

-- SLA monitoring for governance hooks
CREATE TABLE IF NOT EXISTS governance_sla_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    sla_tier VARCHAR(50) NOT NULL, -- enterprise, professional, standard
    plane VARCHAR(50) NOT NULL,
    hook_type VARCHAR(50) NOT NULL,
    target_uptime_percentage DECIMAL(5,3) NOT NULL,
    target_response_time_ms INTEGER NOT NULL,
    actual_uptime_percentage DECIMAL(5,3),
    actual_response_time_ms DECIMAL(10,2),
    error_budget_remaining DECIMAL(5,3),
    sla_breach_count INTEGER DEFAULT 0,
    measurement_window_start TIMESTAMPTZ NOT NULL,
    measurement_window_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT governance_sla_metrics_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Indexes for SLA metrics
CREATE INDEX IF NOT EXISTS idx_governance_sla_metrics_tenant_id ON governance_sla_metrics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_sla_metrics_sla_tier ON governance_sla_metrics(sla_tier);
CREATE INDEX IF NOT EXISTS idx_governance_sla_metrics_plane_hook ON governance_sla_metrics(plane, hook_type);
CREATE INDEX IF NOT EXISTS idx_governance_sla_metrics_window ON governance_sla_metrics(measurement_window_start, measurement_window_end);

-- Row Level Security for SLA metrics
ALTER TABLE governance_sla_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_sla_metrics_tenant_isolation ON governance_sla_metrics
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- FinOps Tracking (Task 8.5.10)
-- =============================================

-- FinOps cost attribution for governance operations
CREATE TABLE IF NOT EXISTS governance_finops_tracking (
    finops_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    workflow_id VARCHAR(255) NOT NULL,
    execution_id VARCHAR(255) NOT NULL,
    plane VARCHAR(50) NOT NULL,
    hook_type VARCHAR(50) NOT NULL,
    cost_center VARCHAR(100),
    region_id VARCHAR(50),
    environment VARCHAR(50), -- dev, staging, prod
    estimated_cost_usd DECIMAL(10,4) DEFAULT 0,
    actual_cost_usd DECIMAL(10,4),
    cost_attribution_tags JSONB,
    billing_period VARCHAR(20), -- YYYY-MM format
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT governance_finops_tracking_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Indexes for FinOps tracking
CREATE INDEX IF NOT EXISTS idx_governance_finops_tracking_tenant_id ON governance_finops_tracking(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_finops_tracking_cost_center ON governance_finops_tracking(cost_center);
CREATE INDEX IF NOT EXISTS idx_governance_finops_tracking_billing_period ON governance_finops_tracking(billing_period);
CREATE INDEX IF NOT EXISTS idx_governance_finops_tracking_plane_hook ON governance_finops_tracking(plane, hook_type);

-- GIN index for cost attribution tags
CREATE INDEX IF NOT EXISTS idx_governance_finops_tracking_tags ON governance_finops_tracking USING GIN (cost_attribution_tags);

-- Row Level Security for FinOps tracking
ALTER TABLE governance_finops_tracking ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_finops_tracking_tenant_isolation ON governance_finops_tracking
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- Lineage Tracking (Task 8.5.13)
-- =============================================

-- Data lineage tracking through governance hooks
CREATE TABLE IF NOT EXISTS governance_lineage_tracking (
    lineage_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    workflow_id VARCHAR(255) NOT NULL,
    execution_id VARCHAR(255) NOT NULL,
    source_system VARCHAR(100),
    target_system VARCHAR(100),
    data_classification VARCHAR(50), -- public, internal, confidential, restricted
    processing_purpose VARCHAR(100),
    lineage_tags JSONB,
    parent_lineage_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT governance_lineage_tracking_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    CONSTRAINT governance_lineage_tracking_parent_fkey FOREIGN KEY (parent_lineage_id) REFERENCES governance_lineage_tracking(lineage_id)
);

-- Indexes for lineage tracking
CREATE INDEX IF NOT EXISTS idx_governance_lineage_tracking_tenant_id ON governance_lineage_tracking(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_lineage_tracking_workflow ON governance_lineage_tracking(workflow_id, execution_id);
CREATE INDEX IF NOT EXISTS idx_governance_lineage_tracking_source_target ON governance_lineage_tracking(source_system, target_system);
CREATE INDEX IF NOT EXISTS idx_governance_lineage_tracking_classification ON governance_lineage_tracking(data_classification);
CREATE INDEX IF NOT EXISTS idx_governance_lineage_tracking_parent ON governance_lineage_tracking(parent_lineage_id);

-- GIN index for lineage tags
CREATE INDEX IF NOT EXISTS idx_governance_lineage_tracking_tags ON governance_lineage_tracking USING GIN (lineage_tags);

-- Row Level Security for lineage tracking
ALTER TABLE governance_lineage_tracking ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_lineage_tracking_tenant_isolation ON governance_lineage_tracking
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- Anomaly Detection (Task 8.5.15)
-- =============================================

-- Anomaly detection results for governance checkpoints
CREATE TABLE IF NOT EXISTS governance_anomaly_detection (
    anomaly_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    detection_model VARCHAR(100) NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL, -- usage_spike, off_hours_access, role_escalation, data_exfiltration
    severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
    confidence_score DECIMAL(5,3) NOT NULL,
    affected_workflows JSONB,
    anomaly_details JSONB,
    baseline_metrics JSONB,
    current_metrics JSONB,
    investigation_status VARCHAR(50) DEFAULT 'open', -- open, investigating, resolved, false_positive
    resolution_notes TEXT,
    detected_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT governance_anomaly_detection_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Indexes for anomaly detection
CREATE INDEX IF NOT EXISTS idx_governance_anomaly_detection_tenant_id ON governance_anomaly_detection(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_anomaly_detection_severity ON governance_anomaly_detection(severity);
CREATE INDEX IF NOT EXISTS idx_governance_anomaly_detection_status ON governance_anomaly_detection(investigation_status);
CREATE INDEX IF NOT EXISTS idx_governance_anomaly_detection_detected_at ON governance_anomaly_detection(detected_at);
CREATE INDEX IF NOT EXISTS idx_governance_anomaly_detection_type ON governance_anomaly_detection(anomaly_type);

-- GIN indexes for JSONB fields
CREATE INDEX IF NOT EXISTS idx_governance_anomaly_detection_workflows ON governance_anomaly_detection USING GIN (affected_workflows);
CREATE INDEX IF NOT EXISTS idx_governance_anomaly_detection_details ON governance_anomaly_detection USING GIN (anomaly_details);

-- Row Level Security for anomaly detection
ALTER TABLE governance_anomaly_detection ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_anomaly_detection_tenant_isolation ON governance_anomaly_detection
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- Regulator Notifications (Task 8.5.19)
-- =============================================

-- Regulator notification tracking
CREATE TABLE IF NOT EXISTS governance_regulator_notifications (
    notification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    regulator_type VARCHAR(50) NOT NULL, -- SEC, GDPR_DPA, RBI, IRDAI, etc.
    notification_type VARCHAR(50) NOT NULL, -- violation, breach, incident, audit_request
    severity VARCHAR(20) NOT NULL,
    governance_event_id UUID,
    evidence_pack_id UUID,
    notification_content JSONB NOT NULL,
    delivery_method VARCHAR(50) NOT NULL, -- email, api, portal
    delivery_status VARCHAR(50) DEFAULT 'pending', -- pending, sent, delivered, failed
    delivery_attempts INTEGER DEFAULT 0,
    acknowledgment_required BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledgment_reference VARCHAR(255),
    legal_hold BOOLEAN DEFAULT FALSE,
    retention_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT governance_regulator_notifications_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    CONSTRAINT governance_regulator_notifications_event_fkey FOREIGN KEY (governance_event_id) REFERENCES governance_event_logs(event_id),
    CONSTRAINT governance_regulator_notifications_evidence_fkey FOREIGN KEY (evidence_pack_id) REFERENCES governance_evidence_packs(evidence_pack_id)
);

-- Indexes for regulator notifications
CREATE INDEX IF NOT EXISTS idx_governance_regulator_notifications_tenant_id ON governance_regulator_notifications(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_regulator_notifications_regulator ON governance_regulator_notifications(regulator_type);
CREATE INDEX IF NOT EXISTS idx_governance_regulator_notifications_status ON governance_regulator_notifications(delivery_status);
CREATE INDEX IF NOT EXISTS idx_governance_regulator_notifications_severity ON governance_regulator_notifications(severity);
CREATE INDEX IF NOT EXISTS idx_governance_regulator_notifications_legal_hold ON governance_regulator_notifications(legal_hold);

-- Row Level Security for regulator notifications
ALTER TABLE governance_regulator_notifications ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_regulator_notifications_tenant_isolation ON governance_regulator_notifications
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- Resilience and Recovery Tracking (Task 8.5.20, 8.5.39)
-- =============================================

-- Governance resilience metrics
CREATE TABLE IF NOT EXISTS governance_resilience_metrics (
    resilience_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id INTEGER NOT NULL,
    plane VARCHAR(50) NOT NULL,
    hook_type VARCHAR(50) NOT NULL,
    failure_type VARCHAR(50), -- timeout, service_unavailable, policy_error, validation_error
    failure_count INTEGER DEFAULT 0,
    recovery_time_seconds DECIMAL(10,2),
    fallback_activated BOOLEAN DEFAULT FALSE,
    fallback_type VARCHAR(50), -- cached_decision, manual_override, bypass, escalation
    sla_impact BOOLEAN DEFAULT FALSE,
    measurement_window_start TIMESTAMPTZ NOT NULL,
    measurement_window_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT governance_resilience_metrics_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Indexes for resilience metrics
CREATE INDEX IF NOT EXISTS idx_governance_resilience_metrics_tenant_id ON governance_resilience_metrics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_governance_resilience_metrics_plane_hook ON governance_resilience_metrics(plane, hook_type);
CREATE INDEX IF NOT EXISTS idx_governance_resilience_metrics_failure_type ON governance_resilience_metrics(failure_type);
CREATE INDEX IF NOT EXISTS idx_governance_resilience_metrics_window ON governance_resilience_metrics(measurement_window_start, measurement_window_end);

-- Row Level Security for resilience metrics
ALTER TABLE governance_resilience_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY governance_resilience_metrics_tenant_isolation ON governance_resilience_metrics
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('rls.tenant_id')::INTEGER);

-- =============================================
-- Automated Functions and Triggers
-- =============================================

-- Function to update evidence pack hash chain (Task 8.5.32)
CREATE OR REPLACE FUNCTION update_evidence_pack_hash_chain()
RETURNS TRIGGER AS $$
DECLARE
    previous_hash VARCHAR(64);
    current_content TEXT;
    current_hash VARCHAR(64);
BEGIN
    -- Get the previous hash from the most recent evidence pack
    SELECT hash_chain_current INTO previous_hash
    FROM governance_evidence_packs
    WHERE tenant_id = NEW.tenant_id
      AND created_at < NEW.created_at
    ORDER BY created_at DESC
    LIMIT 1;
    
    -- Set previous hash (NULL for first evidence pack)
    NEW.hash_chain_previous := COALESCE(previous_hash, '');
    
    -- Calculate current hash
    current_content := NEW.evidence_pack_id::TEXT || 
                      NEW.tenant_id::TEXT || 
                      NEW.evidence_data::TEXT || 
                      NEW.created_at::TEXT ||
                      COALESCE(NEW.hash_chain_previous, '');
    
    NEW.hash_chain_current := encode(digest(current_content, 'sha256'), 'hex');
    
    -- Set retention period (7 years for compliance)
    NEW.retention_until := NEW.created_at + INTERVAL '7 years';
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for evidence pack hash chain
DROP TRIGGER IF EXISTS trigger_evidence_pack_hash_chain ON governance_evidence_packs;
CREATE TRIGGER trigger_evidence_pack_hash_chain
    BEFORE INSERT ON governance_evidence_packs
    FOR EACH ROW
    EXECUTE FUNCTION update_evidence_pack_hash_chain();

-- Function to auto-expire override requests
CREATE OR REPLACE FUNCTION auto_expire_override_requests()
RETURNS TRIGGER AS $$
BEGIN
    -- Set expiration time (24 hours for governance overrides)
    NEW.expires_at := NEW.created_at + INTERVAL '24 hours';
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for override request expiration
DROP TRIGGER IF EXISTS trigger_auto_expire_override_requests ON governance_override_ledger;
CREATE TRIGGER trigger_auto_expire_override_requests
    BEFORE INSERT ON governance_override_ledger
    FOR EACH ROW
    EXECUTE FUNCTION auto_expire_override_requests();

-- Function to update override ledger timestamps
CREATE OR REPLACE FUNCTION update_override_ledger_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    
    -- Auto-expire if past expiration time
    IF NEW.expires_at < NOW() AND NEW.status = 'pending_approval' THEN
        NEW.status := 'expired';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for override ledger timestamp updates
DROP TRIGGER IF EXISTS trigger_update_override_ledger_timestamp ON governance_override_ledger;
CREATE TRIGGER trigger_update_override_ledger_timestamp
    BEFORE UPDATE ON governance_override_ledger
    FOR EACH ROW
    EXECUTE FUNCTION update_override_ledger_timestamp();

-- =============================================
-- Materialized Views for Dashboard Performance
-- =============================================

-- Governance summary materialized view for dashboard performance
CREATE MATERIALIZED VIEW IF NOT EXISTS governance_summary_hourly AS
SELECT 
    tenant_id,
    DATE_TRUNC('hour', timestamp) as hour,
    plane,
    hook_type,
    decision,
    COUNT(*) as event_count,
    SUM(violations_count) as total_violations,
    AVG(execution_time_ms) as avg_execution_time,
    COUNT(CASE WHEN evidence_pack_id IS NOT NULL THEN 1 END) as evidence_packs_generated
FROM governance_event_logs
GROUP BY tenant_id, DATE_TRUNC('hour', timestamp), plane, hook_type, decision;

-- Index for materialized view
CREATE INDEX IF NOT EXISTS idx_governance_summary_hourly_tenant_hour ON governance_summary_hourly(tenant_id, hour);

-- Refresh materialized view function
CREATE OR REPLACE FUNCTION refresh_governance_summary_hourly()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY governance_summary_hourly;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- Partitioning for Large Tables (Performance Optimization)
-- =============================================

-- Partition governance_event_logs by month for better performance
-- Note: This would be implemented in production for high-volume tenants

-- =============================================
-- Sample Data for Testing (SaaS Scenarios)
-- =============================================

-- Insert sample SLA tiers for SaaS tenants
INSERT INTO governance_sla_metrics (tenant_id, sla_tier, plane, hook_type, target_uptime_percentage, target_response_time_ms, measurement_window_start, measurement_window_end)
VALUES 
    (1300, 'enterprise', 'control', 'policy', 99.99, 100, NOW() - INTERVAL '1 hour', NOW()),
    (1300, 'professional', 'execution', 'trust', 99.9, 200, NOW() - INTERVAL '1 hour', NOW()),
    (1300, 'standard', 'data', 'consent', 99.5, 500, NOW() - INTERVAL '1 hour', NOW())
ON CONFLICT DO NOTHING;

-- Insert sample FinOps cost centers for SaaS
INSERT INTO governance_finops_tracking (tenant_id, workflow_id, execution_id, plane, hook_type, cost_center, region_id, environment, estimated_cost_usd, billing_period)
VALUES 
    (1300, 'saas_pipeline_health', 'exec_001', 'execution', 'trust', 'saas_operations', 'us-east-1', 'prod', 0.05, '2024-01'),
    (1300, 'saas_churn_prediction', 'exec_002', 'data', 'lineage', 'saas_analytics', 'us-west-2', 'prod', 0.12, '2024-01')
ON CONFLICT DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE governance_event_logs IS 'Core governance event logging for all planes and hook types (Task 8.5.6)';
COMMENT ON TABLE governance_evidence_packs IS 'Immutable evidence packs with WORM storage and digital signatures (Task 8.5.7)';
COMMENT ON TABLE governance_override_ledger IS 'Override ledger integration for human-in-loop interventions (Task 8.5.6)';
COMMENT ON TABLE governance_sla_metrics IS 'SLA monitoring and error budget validation (Task 8.5.8)';
COMMENT ON TABLE governance_finops_tracking IS 'FinOps cost attribution and budget enforcement (Task 8.5.10)';
COMMENT ON TABLE governance_lineage_tracking IS 'Data lineage tracking through governance hooks (Task 8.5.13)';
COMMENT ON TABLE governance_anomaly_detection IS 'Anomaly detection at governance checkpoints (Task 8.5.15)';
COMMENT ON TABLE governance_regulator_notifications IS 'Regulator notification tracking for compliance (Task 8.5.19)';
COMMENT ON TABLE governance_resilience_metrics IS 'Resilience and recovery metrics for governance (Task 8.5.39)';

-- Grant appropriate permissions
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;
