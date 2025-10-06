-- Task 3.2.8: Kill Switch Database Schema
-- Kill switch per model/node/workflow with global intelligence off fallback

-- Kill switches table
CREATE TABLE IF NOT EXISTS kill_switches (
    switch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope VARCHAR(20) NOT NULL CHECK (scope IN ('model', 'node', 'workflow', 'tenant', 'global')),
    target_id VARCHAR(255) NOT NULL,
    target_name VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'killed', 'maintenance')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    
    -- Unique constraint per scope and target
    UNIQUE(scope, target_id, tenant_id)
);

-- Kill switch activations table
CREATE TABLE IF NOT EXISTS kill_switch_activations (
    activation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    switch_id UUID NOT NULL REFERENCES kill_switches(switch_id),
    trigger_type VARCHAR(30) NOT NULL CHECK (trigger_type IN (
        'manual', 'automatic', 'drift_detected', 'bias_detected', 
        'performance_degradation', 'security_incident', 'compliance_violation'
    )),
    triggered_by VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    evidence JSONB DEFAULT '{}',
    activated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deactivated_at TIMESTAMP WITH TIME ZONE,
    deactivated_by VARCHAR(255),
    deactivation_reason TEXT,
    auto_revert_at TIMESTAMP WITH TIME ZONE,
    
    -- Index for querying recent activations
    INDEX idx_activations_switch_activated (switch_id, activated_at),
    INDEX idx_activations_trigger_type (trigger_type, activated_at)
);

-- Fallback configurations table
CREATE TABLE IF NOT EXISTS fallback_configurations (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scope VARCHAR(20) NOT NULL CHECK (scope IN ('model', 'node', 'workflow', 'tenant', 'global')),
    target_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    fallback_type VARCHAR(50) NOT NULL, -- 'RBA', 'baseline_model', 'static_rules', etc.
    fallback_config JSONB NOT NULL DEFAULT '{}',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint per scope and target
    UNIQUE(scope, target_id, tenant_id)
);

-- Kill switch audit log
CREATE TABLE IF NOT EXISTS kill_switch_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event VARCHAR(50) NOT NULL,
    switch_id UUID REFERENCES kill_switches(switch_id),
    activation_id UUID REFERENCES kill_switch_activations(activation_id),
    scope VARCHAR(20),
    target_id VARCHAR(255),
    tenant_id VARCHAR(255),
    actor_id VARCHAR(255) NOT NULL, -- User or system that performed action
    event_data JSONB DEFAULT '{}',
    severity VARCHAR(20) DEFAULT 'INFO' CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for audit queries
    INDEX idx_audit_timestamp (timestamp),
    INDEX idx_audit_switch_id (switch_id, timestamp),
    INDEX idx_audit_tenant_id (tenant_id, timestamp),
    INDEX idx_audit_severity (severity, timestamp)
);

-- Global platform state table
CREATE TABLE IF NOT EXISTS platform_intelligence_state (
    state_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    global_intelligence_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    changed_by VARCHAR(255) NOT NULL,
    change_reason TEXT,
    emergency_mode BOOLEAN DEFAULT FALSE,
    
    -- Only one row should exist
    CONSTRAINT single_row CHECK (state_id = '00000000-0000-0000-0000-000000000001')
);

-- Insert initial platform state
INSERT INTO platform_intelligence_state (
    state_id, global_intelligence_enabled, changed_by, change_reason
) VALUES (
    '00000000-0000-0000-0000-000000000001', 
    TRUE, 
    'system', 
    'Initial platform state'
) ON CONFLICT (state_id) DO NOTHING;

-- Enable Row Level Security
ALTER TABLE kill_switches ENABLE ROW LEVEL SECURITY;
ALTER TABLE kill_switch_activations ENABLE ROW LEVEL SECURITY;
ALTER TABLE fallback_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE kill_switch_audit_log ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation
CREATE POLICY tenant_isolation_kill_switches ON kill_switches
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true) OR scope = 'global');

CREATE POLICY tenant_isolation_activations ON kill_switch_activations
    FOR ALL TO rbia_users
    USING (EXISTS (
        SELECT 1 FROM kill_switches ks 
        WHERE ks.switch_id = kill_switch_activations.switch_id 
        AND (ks.tenant_id = current_setting('rbia.current_tenant_id', true) OR ks.scope = 'global')
    ));

CREATE POLICY tenant_isolation_fallback_configs ON fallback_configurations
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true) OR scope = 'global');

CREATE POLICY tenant_isolation_audit_log ON kill_switch_audit_log
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true) OR tenant_id = 'system');

-- Performance indexes
CREATE INDEX idx_kill_switches_tenant_status ON kill_switches(tenant_id, status);
CREATE INDEX idx_kill_switches_scope_target ON kill_switches(scope, target_id);
CREATE INDEX idx_fallback_configs_scope_target ON fallback_configurations(scope, target_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_kill_switch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_kill_switches_updated_at
    BEFORE UPDATE ON kill_switches
    FOR EACH ROW
    EXECUTE FUNCTION update_kill_switch_updated_at();

CREATE TRIGGER update_fallback_configs_updated_at
    BEFORE UPDATE ON fallback_configurations
    FOR EACH ROW
    EXECUTE FUNCTION update_kill_switch_updated_at();

-- Function to automatically log kill switch events
CREATE OR REPLACE FUNCTION log_kill_switch_event()
RETURNS TRIGGER AS $$
DECLARE
    event_name VARCHAR(50);
    severity_level VARCHAR(20) := 'INFO';
BEGIN
    -- Determine event name based on operation and status change
    IF TG_OP = 'INSERT' THEN
        event_name := 'kill_switch_created';
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.status != NEW.status THEN
            IF NEW.status = 'killed' THEN
                event_name := 'kill_switch_activated';
                severity_level := 'WARNING';
            ELSIF NEW.status = 'active' THEN
                event_name := 'kill_switch_deactivated';
                severity_level := 'INFO';
            ELSE
                event_name := 'kill_switch_status_changed';
            END IF;
        ELSE
            event_name := 'kill_switch_updated';
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        event_name := 'kill_switch_deleted';
        severity_level := 'WARNING';
    END IF;
    
    -- Log the event
    INSERT INTO kill_switch_audit_log (
        event, switch_id, scope, target_id, tenant_id, actor_id, 
        event_data, severity
    ) VALUES (
        event_name,
        COALESCE(NEW.switch_id, OLD.switch_id),
        COALESCE(NEW.scope, OLD.scope),
        COALESCE(NEW.target_id, OLD.target_id),
        COALESCE(NEW.tenant_id, OLD.tenant_id),
        COALESCE(NEW.created_by, OLD.created_by, 'system'),
        jsonb_build_object(
            'old_status', CASE WHEN TG_OP != 'INSERT' THEN OLD.status END,
            'new_status', CASE WHEN TG_OP != 'DELETE' THEN NEW.status END,
            'operation', TG_OP
        ),
        severity_level
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Audit trigger for kill switches
CREATE TRIGGER audit_kill_switches
    AFTER INSERT OR UPDATE OR DELETE ON kill_switches
    FOR EACH ROW
    EXECUTE FUNCTION log_kill_switch_event();

-- Function to log activation events
CREATE OR REPLACE FUNCTION log_activation_event()
RETURNS TRIGGER AS $$
DECLARE
    switch_record kill_switches%ROWTYPE;
    event_name VARCHAR(50);
    severity_level VARCHAR(20) := 'WARNING';
BEGIN
    -- Get switch details
    SELECT * INTO switch_record FROM kill_switches WHERE switch_id = NEW.switch_id;
    
    IF TG_OP = 'INSERT' THEN
        event_name := 'kill_switch_activation_created';
        IF NEW.trigger_type IN ('security_incident', 'compliance_violation') THEN
            severity_level := 'CRITICAL';
        END IF;
    ELSIF TG_OP = 'UPDATE' AND OLD.deactivated_at IS NULL AND NEW.deactivated_at IS NOT NULL THEN
        event_name := 'kill_switch_activation_deactivated';
        severity_level := 'INFO';
    END IF;
    
    -- Log the activation event
    INSERT INTO kill_switch_audit_log (
        event, switch_id, activation_id, scope, target_id, tenant_id, 
        actor_id, event_data, severity
    ) VALUES (
        event_name,
        NEW.switch_id,
        NEW.activation_id,
        switch_record.scope,
        switch_record.target_id,
        switch_record.tenant_id,
        NEW.triggered_by,
        jsonb_build_object(
            'trigger_type', NEW.trigger_type,
            'reason', NEW.reason,
            'auto_revert_at', NEW.auto_revert_at,
            'deactivated_by', NEW.deactivated_by
        ),
        severity_level
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Audit trigger for activations
CREATE TRIGGER audit_kill_switch_activations
    AFTER INSERT OR UPDATE ON kill_switch_activations
    FOR EACH ROW
    EXECUTE FUNCTION log_activation_event();

-- Function to update global platform state
CREATE OR REPLACE FUNCTION update_platform_intelligence_state(
    p_enabled BOOLEAN,
    p_changed_by VARCHAR(255),
    p_reason TEXT,
    p_emergency_mode BOOLEAN DEFAULT FALSE
) RETURNS VOID AS $$
BEGIN
    UPDATE platform_intelligence_state 
    SET 
        global_intelligence_enabled = p_enabled,
        last_changed_at = CURRENT_TIMESTAMP,
        changed_by = p_changed_by,
        change_reason = p_reason,
        emergency_mode = p_emergency_mode
    WHERE state_id = '00000000-0000-0000-0000-000000000001';
    
    -- Log the platform state change
    INSERT INTO kill_switch_audit_log (
        event, scope, target_id, tenant_id, actor_id, event_data, severity
    ) VALUES (
        'platform_intelligence_state_changed',
        'global',
        'platform',
        'system',
        p_changed_by,
        jsonb_build_object(
            'intelligence_enabled', p_enabled,
            'emergency_mode', p_emergency_mode,
            'reason', p_reason
        ),
        CASE WHEN NOT p_enabled THEN 'CRITICAL' ELSE 'INFO' END
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get current platform intelligence state
CREATE OR REPLACE FUNCTION get_platform_intelligence_state()
RETURNS TABLE(
    intelligence_enabled BOOLEAN,
    emergency_mode BOOLEAN,
    last_changed_at TIMESTAMP WITH TIME ZONE,
    changed_by VARCHAR(255)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pis.global_intelligence_enabled,
        pis.emergency_mode,
        pis.last_changed_at,
        pis.changed_by
    FROM platform_intelligence_state pis
    WHERE pis.state_id = '00000000-0000-0000-0000-000000000001';
END;
$$ LANGUAGE plpgsql;

-- Function to check if intelligence is enabled for specific target
CREATE OR REPLACE FUNCTION is_intelligence_enabled(
    p_scope VARCHAR(20),
    p_target_id VARCHAR(255),
    p_tenant_id VARCHAR(255)
) RETURNS BOOLEAN AS $$
DECLARE
    global_enabled BOOLEAN;
    switch_killed BOOLEAN := FALSE;
BEGIN
    -- Check global state first
    SELECT global_intelligence_enabled INTO global_enabled
    FROM platform_intelligence_state
    WHERE state_id = '00000000-0000-0000-0000-000000000001';
    
    IF NOT global_enabled THEN
        RETURN FALSE;
    END IF;
    
    -- Check specific kill switches (more specific scopes take precedence)
    SELECT EXISTS(
        SELECT 1 FROM kill_switches 
        WHERE scope = p_scope 
        AND target_id = p_target_id 
        AND tenant_id = p_tenant_id 
        AND status = 'killed'
    ) INTO switch_killed;
    
    IF switch_killed THEN
        RETURN FALSE;
    END IF;
    
    -- Check tenant-level kill switch
    SELECT EXISTS(
        SELECT 1 FROM kill_switches 
        WHERE scope = 'tenant' 
        AND target_id = p_tenant_id 
        AND status = 'killed'
    ) INTO switch_killed;
    
    RETURN NOT switch_killed;
END;
$$ LANGUAGE plpgsql;

-- Sample data for testing
INSERT INTO kill_switches (scope, target_id, target_name, tenant_id, created_by) VALUES
('model', 'churn_model_v2', 'Churn Prediction Model v2', 'tenant_saas_001', 'admin'),
('workflow', 'lead_scoring_workflow', 'Lead Scoring Workflow', 'tenant_saas_001', 'admin'),
('model', 'fraud_detection_v3', 'Fraud Detection Model v3', 'tenant_banking_001', 'admin'),
('global', 'platform', 'Global Platform Intelligence', 'system', 'system');

INSERT INTO fallback_configurations (scope, target_id, tenant_id, fallback_type, fallback_config) VALUES
('model', 'churn_model_v2', 'tenant_saas_001', 'RBA', 
 '{"mode": "deterministic_rules", "use_baseline_scoring": true, "confidence_threshold": 0.0}'::jsonb),
('workflow', 'lead_scoring_workflow', 'tenant_saas_001', 'static_rules',
 '{"scoring_method": "rule_based", "default_score": 0.5, "escalation_threshold": 0.8}'::jsonb),
('global', 'platform', 'system', 'RBA',
 '{"mode": "full_deterministic", "disable_all_ml": true, "fallback_message": "Intelligence disabled - using rule-based automation"}'::jsonb);

COMMENT ON TABLE kill_switches IS 'Task 3.2.8: Kill switches for models, nodes, workflows, and global intelligence';
COMMENT ON TABLE kill_switch_activations IS 'Task 3.2.8: Kill switch activation history with triggers and reasons';
COMMENT ON TABLE fallback_configurations IS 'Task 3.2.8: Fallback configurations for when intelligence is disabled';
COMMENT ON TABLE kill_switch_audit_log IS 'Task 3.2.8: Comprehensive audit log for all kill switch activities';
COMMENT ON TABLE platform_intelligence_state IS 'Task 3.2.8: Global platform intelligence state management';
