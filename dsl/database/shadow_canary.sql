-- Task 3.2.7: Shadow Mode & Canary Execution Database Schema
-- Safe rollout with measurable lift and auto rollback on regressions

-- Model deployments table
CREATE TABLE IF NOT EXISTS model_deployments (
    deployment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    baseline_model_id VARCHAR(255) NOT NULL,
    deployment_mode VARCHAR(20) NOT NULL CHECK (deployment_mode IN ('shadow', 'canary', 'full')),
    traffic_percentage DECIMAL(5,2) DEFAULT 0.0 CHECK (traffic_percentage >= 0.0 AND traffic_percentage <= 100.0),
    shadow_percentage DECIMAL(5,2) DEFAULT 100.0 CHECK (shadow_percentage >= 0.0 AND shadow_percentage <= 100.0),
    auto_rollback_enabled BOOLEAN DEFAULT TRUE,
    performance_thresholds JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'rolled_back', 'completed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Tenant isolation
    CONSTRAINT tenant_isolation CHECK (tenant_id IS NOT NULL)
);

-- Model executions table (for both baseline and candidate)
CREATE TABLE IF NOT EXISTS model_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID NOT NULL REFERENCES model_deployments(deployment_id),
    model_id VARCHAR(255) NOT NULL,
    input_data JSONB NOT NULL,
    output_data JSONB NOT NULL,
    execution_time_ms DECIMAL(10,3) NOT NULL,
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    is_baseline BOOLEAN NOT NULL DEFAULT FALSE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Performance indexes
    INDEX idx_executions_deployment_timestamp (deployment_id, timestamp),
    INDEX idx_executions_baseline (deployment_id, is_baseline, timestamp)
);

-- Performance comparison results table
CREATE TABLE IF NOT EXISTS performance_comparisons (
    comparison_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID NOT NULL REFERENCES model_deployments(deployment_id),
    metric_name VARCHAR(100) NOT NULL,
    baseline_value DECIMAL(15,6) NOT NULL,
    candidate_value DECIMAL(15,6) NOT NULL,
    improvement_percentage DECIMAL(8,4) NOT NULL,
    threshold_met BOOLEAN NOT NULL,
    threshold_value DECIMAL(8,4),
    sample_size INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite index for querying comparisons
    INDEX idx_comparisons_deployment_metric (deployment_id, metric_name, timestamp)
);

-- Rollback decisions table
CREATE TABLE IF NOT EXISTS rollback_decisions (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID NOT NULL REFERENCES model_deployments(deployment_id),
    trigger_reason TEXT NOT NULL,
    failed_metrics TEXT[] NOT NULL,
    auto_triggered BOOLEAN NOT NULL DEFAULT TRUE,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    rollback_completed BOOLEAN DEFAULT FALSE,
    rollback_completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Only one rollback decision per deployment
    UNIQUE(deployment_id)
);

-- Canary traffic configuration table
CREATE TABLE IF NOT EXISTS canary_traffic_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID NOT NULL REFERENCES model_deployments(deployment_id),
    traffic_percentage DECIMAL(5,2) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    success_criteria JSONB,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Shadow execution audit log
CREATE TABLE IF NOT EXISTS shadow_execution_audit (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID NOT NULL REFERENCES model_deployments(deployment_id),
    execution_id UUID REFERENCES model_executions(execution_id),
    event_type VARCHAR(50) NOT NULL, -- 'execution', 'comparison', 'rollback', 'promotion'
    event_data JSONB NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Enable Row Level Security
ALTER TABLE model_deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_comparisons ENABLE ROW LEVEL SECURITY;
ALTER TABLE rollback_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE canary_traffic_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE shadow_execution_audit ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation
CREATE POLICY tenant_isolation_deployments ON model_deployments
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_executions ON model_executions
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_comparisons ON performance_comparisons
    FOR ALL TO rbia_users
    USING (EXISTS (
        SELECT 1 FROM model_deployments md 
        WHERE md.deployment_id = performance_comparisons.deployment_id 
        AND md.tenant_id = current_setting('rbia.current_tenant_id', true)
    ));

CREATE POLICY tenant_isolation_rollbacks ON rollback_decisions
    FOR ALL TO rbia_users
    USING (EXISTS (
        SELECT 1 FROM model_deployments md 
        WHERE md.deployment_id = rollback_decisions.deployment_id 
        AND md.tenant_id = current_setting('rbia.current_tenant_id', true)
    ));

CREATE POLICY tenant_isolation_canary_config ON canary_traffic_config
    FOR ALL TO rbia_users
    USING (EXISTS (
        SELECT 1 FROM model_deployments md 
        WHERE md.deployment_id = canary_traffic_config.deployment_id 
        AND md.tenant_id = current_setting('rbia.current_tenant_id', true)
    ));

CREATE POLICY tenant_isolation_shadow_audit ON shadow_execution_audit
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

-- Performance indexes
CREATE INDEX idx_deployments_tenant_status ON model_deployments(tenant_id, status);
CREATE INDEX idx_deployments_mode ON model_deployments(deployment_mode);
CREATE INDEX idx_executions_tenant_timestamp ON model_executions(tenant_id, timestamp);
CREATE INDEX idx_comparisons_timestamp ON performance_comparisons(timestamp);
CREATE INDEX idx_rollbacks_executed_at ON rollback_decisions(executed_at);
CREATE INDEX idx_shadow_audit_tenant_timestamp ON shadow_execution_audit(tenant_id, timestamp);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for updated_at
CREATE TRIGGER update_deployments_updated_at
    BEFORE UPDATE ON model_deployments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to validate deployment mode transitions
CREATE OR REPLACE FUNCTION validate_deployment_transition()
RETURNS TRIGGER AS $$
BEGIN
    -- Prevent invalid transitions
    IF OLD.deployment_mode = 'full' AND NEW.deployment_mode IN ('shadow', 'canary') THEN
        -- Allow rollback from full to shadow/canary
        IF NOT EXISTS (SELECT 1 FROM rollback_decisions WHERE deployment_id = NEW.deployment_id) THEN
            RAISE EXCEPTION 'Cannot transition from full deployment without rollback decision';
        END IF;
    END IF;
    
    -- Validate traffic percentage based on mode
    IF NEW.deployment_mode = 'shadow' AND NEW.traffic_percentage > 0 THEN
        RAISE EXCEPTION 'Shadow mode cannot have production traffic percentage > 0';
    END IF;
    
    IF NEW.deployment_mode = 'full' AND NEW.traffic_percentage != 100.0 THEN
        NEW.traffic_percentage = 100.0;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for deployment validation
CREATE TRIGGER validate_deployment_mode_transition
    BEFORE UPDATE ON model_deployments
    FOR EACH ROW
    EXECUTE FUNCTION validate_deployment_transition();

-- Function to auto-create performance comparison
CREATE OR REPLACE FUNCTION create_performance_comparison(
    p_deployment_id UUID,
    p_metric_name VARCHAR,
    p_baseline_value DECIMAL,
    p_candidate_value DECIMAL,
    p_threshold_value DECIMAL,
    p_sample_size INTEGER
) RETURNS UUID AS $$
DECLARE
    comparison_id UUID;
    improvement_pct DECIMAL;
    threshold_met BOOLEAN;
BEGIN
    -- Calculate improvement percentage
    improvement_pct := CASE 
        WHEN p_baseline_value = 0 THEN 0
        ELSE ((p_candidate_value - p_baseline_value) / p_baseline_value) * 100
    END;
    
    -- Check if threshold is met (depends on metric type)
    threshold_met := CASE 
        WHEN p_metric_name LIKE '%_improvement' THEN improvement_pct >= (p_threshold_value * 100)
        WHEN p_metric_name LIKE '%_regression' THEN improvement_pct >= (p_threshold_value * 100)
        ELSE improvement_pct >= 0
    END;
    
    -- Insert comparison result
    INSERT INTO performance_comparisons (
        deployment_id, metric_name, baseline_value, candidate_value,
        improvement_percentage, threshold_met, threshold_value, sample_size
    ) VALUES (
        p_deployment_id, p_metric_name, p_baseline_value, p_candidate_value,
        improvement_pct, threshold_met, p_threshold_value, p_sample_size
    ) RETURNING comparison_id INTO comparison_id;
    
    RETURN comparison_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check if rollback is needed
CREATE OR REPLACE FUNCTION check_rollback_needed(p_deployment_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    failed_count INTEGER;
    auto_rollback_enabled BOOLEAN;
BEGIN
    -- Get auto rollback setting
    SELECT md.auto_rollback_enabled INTO auto_rollback_enabled
    FROM model_deployments md
    WHERE md.deployment_id = p_deployment_id;
    
    IF NOT auto_rollback_enabled THEN
        RETURN FALSE;
    END IF;
    
    -- Count failed comparisons in last hour
    SELECT COUNT(*) INTO failed_count
    FROM performance_comparisons pc
    WHERE pc.deployment_id = p_deployment_id
    AND pc.timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
    AND NOT pc.threshold_met;
    
    -- Trigger rollback if any critical metrics failed
    RETURN failed_count > 0;
END;
$$ LANGUAGE plpgsql;

-- Audit trigger function
CREATE OR REPLACE FUNCTION audit_shadow_execution()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO shadow_execution_audit (deployment_id, execution_id, event_type, event_data, tenant_id)
    VALUES (
        NEW.deployment_id,
        NEW.execution_id,
        'execution',
        jsonb_build_object(
            'model_id', NEW.model_id,
            'is_baseline', NEW.is_baseline,
            'confidence_score', NEW.confidence_score,
            'execution_time_ms', NEW.execution_time_ms
        ),
        NEW.tenant_id
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Audit trigger
CREATE TRIGGER audit_model_executions
    AFTER INSERT ON model_executions
    FOR EACH ROW
    EXECUTE FUNCTION audit_shadow_execution();

-- Sample data for testing
INSERT INTO model_deployments (
    model_id, model_version, tenant_id, baseline_model_id, 
    deployment_mode, performance_thresholds
) VALUES 
(
    'churn_model_v2', '2.1.0', 'tenant_saas_001', 'churn_model_v1',
    'shadow', 
    '{"accuracy_improvement": 0.02, "latency_regression": -0.10, "confidence_improvement": 0.01}'::jsonb
),
(
    'fraud_model_v3', '3.0.0', 'tenant_banking_001', 'fraud_model_v2',
    'canary',
    '{"precision_improvement": 0.05, "recall_improvement": 0.03, "latency_regression": -0.15}'::jsonb
);

COMMENT ON TABLE model_deployments IS 'Task 3.2.7: Shadow and canary model deployments';
COMMENT ON TABLE model_executions IS 'Task 3.2.7: Model execution results for comparison';
COMMENT ON TABLE performance_comparisons IS 'Task 3.2.7: Performance comparison results';
COMMENT ON TABLE rollback_decisions IS 'Task 3.2.7: Automatic and manual rollback decisions';
COMMENT ON TABLE canary_traffic_config IS 'Task 3.2.7: Canary traffic configuration and progression';
COMMENT ON TABLE shadow_execution_audit IS 'Task 3.2.7: Audit log for shadow mode executions';
