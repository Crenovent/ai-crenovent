-- Task 3.2.9: Trust Score Database Schema
-- Trust score computation (quality × explainability × drift × bias × ops SLO)

-- Trust scores table
CREATE TABLE IF NOT EXISTS trust_scores (
    trust_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_type VARCHAR(50) NOT NULL, -- 'model', 'node', 'workflow'
    target_id VARCHAR(255) NOT NULL,
    target_name VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    overall_score DECIMAL(5,4) NOT NULL CHECK (overall_score >= 0.0 AND overall_score <= 1.0),
    trust_level VARCHAR(20) NOT NULL CHECK (trust_level IN ('excellent', 'good', 'acceptable', 'poor', 'critical')),
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    
    -- Unique constraint per target per tenant
    UNIQUE(target_type, target_id, tenant_id),
    
    -- Index for queries
    INDEX idx_trust_scores_tenant_type (tenant_id, target_type),
    INDEX idx_trust_scores_level (trust_level, overall_score),
    INDEX idx_trust_scores_computed_at (computed_at)
);

-- Trust score components table
CREATE TABLE IF NOT EXISTS trust_score_components (
    component_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trust_id UUID NOT NULL REFERENCES trust_scores(trust_id) ON DELETE CASCADE,
    component VARCHAR(20) NOT NULL CHECK (component IN ('quality', 'explainability', 'drift', 'bias', 'ops_slo')),
    score DECIMAL(5,4) NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    weight DECIMAL(5,4) NOT NULL DEFAULT 1.0 CHECK (weight >= 0.0 AND weight <= 1.0),
    confidence DECIMAL(5,4) NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    metrics JSONB NOT NULL DEFAULT '{}',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint per component per trust score
    UNIQUE(trust_id, component),
    
    -- Index for component queries
    INDEX idx_components_trust_component (trust_id, component),
    INDEX idx_components_score (component, score)
);

-- Trust gates table (minimum requirements)
CREATE TABLE IF NOT EXISTS trust_gates (
    gate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_type VARCHAR(50) NOT NULL,
    target_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    gate_type VARCHAR(50) NOT NULL, -- 'deployment', 'execution', 'promotion', etc.
    minimum_trust_score DECIMAL(5,4) NOT NULL CHECK (minimum_trust_score >= 0.0 AND minimum_trust_score <= 1.0),
    required_components JSONB DEFAULT '{}', -- Component-specific minimum scores
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    
    -- Unique constraint per target per gate type
    UNIQUE(target_type, target_id, tenant_id, gate_type),
    
    -- Index for gate checks
    INDEX idx_trust_gates_target (target_type, target_id, tenant_id, active)
);

-- Trust score history table (for tracking changes over time)
CREATE TABLE IF NOT EXISTS trust_score_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_type VARCHAR(50) NOT NULL,
    target_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    overall_score DECIMAL(5,4) NOT NULL,
    trust_level VARCHAR(20) NOT NULL,
    component_scores JSONB NOT NULL, -- Snapshot of all component scores
    computed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    change_reason VARCHAR(255), -- What triggered the recomputation
    
    -- Partitioning by date for performance
    INDEX idx_history_target_date (target_type, target_id, tenant_id, computed_at),
    INDEX idx_history_tenant_date (tenant_id, computed_at)
);

-- Trust gate violations table (audit log)
CREATE TABLE IF NOT EXISTS trust_gate_violations (
    violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gate_id UUID NOT NULL REFERENCES trust_gates(gate_id),
    trust_id UUID REFERENCES trust_scores(trust_id),
    target_type VARCHAR(50) NOT NULL,
    target_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    violation_type VARCHAR(50) NOT NULL, -- 'overall_score', 'component_score'
    required_score DECIMAL(5,4) NOT NULL,
    actual_score DECIMAL(5,4) NOT NULL,
    component VARCHAR(20), -- If component-specific violation
    action_taken VARCHAR(100), -- 'blocked', 'warned', 'overridden'
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    
    -- Index for violation queries
    INDEX idx_violations_gate_date (gate_id, detected_at),
    INDEX idx_violations_target_date (target_type, target_id, tenant_id, detected_at)
);

-- Component weight configurations table
CREATE TABLE IF NOT EXISTS component_weight_configs (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255) NOT NULL,
    target_type VARCHAR(50), -- NULL means applies to all target types
    industry VARCHAR(50), -- Industry-specific weights
    config_name VARCHAR(255) NOT NULL,
    weights JSONB NOT NULL, -- Component weights
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    
    -- Unique constraint per tenant per config name
    UNIQUE(tenant_id, config_name),
    
    INDEX idx_weight_configs_tenant_industry (tenant_id, industry, active)
);

-- Trust score alerts table
CREATE TABLE IF NOT EXISTS trust_score_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_type VARCHAR(50) NOT NULL,
    target_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    alert_type VARCHAR(50) NOT NULL, -- 'score_drop', 'component_failure', 'gate_violation'
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    current_score DECIMAL(5,4),
    previous_score DECIMAL(5,4),
    threshold_breached DECIMAL(5,4),
    component VARCHAR(20), -- If component-specific alert
    message TEXT NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- Index for alert queries
    INDEX idx_alerts_tenant_severity (tenant_id, severity, triggered_at),
    INDEX idx_alerts_target_type (target_type, triggered_at),
    INDEX idx_alerts_unresolved (resolved_at) WHERE resolved_at IS NULL
);

-- Enable Row Level Security
ALTER TABLE trust_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE trust_score_components ENABLE ROW LEVEL SECURITY;
ALTER TABLE trust_gates ENABLE ROW LEVEL SECURITY;
ALTER TABLE trust_score_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE trust_gate_violations ENABLE ROW LEVEL SECURITY;
ALTER TABLE component_weight_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE trust_score_alerts ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation
CREATE POLICY tenant_isolation_trust_scores ON trust_scores
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_trust_components ON trust_score_components
    FOR ALL TO rbia_users
    USING (EXISTS (
        SELECT 1 FROM trust_scores ts 
        WHERE ts.trust_id = trust_score_components.trust_id 
        AND ts.tenant_id = current_setting('rbia.current_tenant_id', true)
    ));

CREATE POLICY tenant_isolation_trust_gates ON trust_gates
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_trust_history ON trust_score_history
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_gate_violations ON trust_gate_violations
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_weight_configs ON component_weight_configs
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_trust_alerts ON trust_score_alerts
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

-- Function to compute overall trust score
CREATE OR REPLACE FUNCTION compute_trust_score(
    p_trust_id UUID,
    p_weights JSONB DEFAULT NULL
) RETURNS DECIMAL(5,4) AS $$
DECLARE
    component_record RECORD;
    weighted_sum DECIMAL(10,6) := 0.0;
    total_weight DECIMAL(10,6) := 0.0;
    default_weights JSONB := '{"quality": 0.25, "explainability": 0.20, "drift": 0.20, "bias": 0.20, "ops_slo": 0.15}';
    weights JSONB;
BEGIN
    -- Use provided weights or defaults
    weights := COALESCE(p_weights, default_weights);
    
    -- Calculate weighted sum
    FOR component_record IN 
        SELECT component, score, weight 
        FROM trust_score_components 
        WHERE trust_id = p_trust_id
    LOOP
        weighted_sum := weighted_sum + (component_record.score * COALESCE((weights->>component_record.component)::DECIMAL, component_record.weight));
        total_weight := total_weight + COALESCE((weights->>component_record.component)::DECIMAL, component_record.weight);
    END LOOP;
    
    -- Return weighted average
    IF total_weight > 0 THEN
        RETURN LEAST(1.0, GREATEST(0.0, weighted_sum / total_weight));
    ELSE
        RETURN 0.0;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to determine trust level from score
CREATE OR REPLACE FUNCTION get_trust_level(score DECIMAL(5,4))
RETURNS VARCHAR(20) AS $$
BEGIN
    IF score >= 0.9 THEN
        RETURN 'excellent';
    ELSIF score >= 0.7 THEN
        RETURN 'good';
    ELSIF score >= 0.5 THEN
        RETURN 'acceptable';
    ELSIF score >= 0.3 THEN
        RETURN 'poor';
    ELSE
        RETURN 'critical';
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to check trust gates
CREATE OR REPLACE FUNCTION check_trust_gates(
    p_target_type VARCHAR(50),
    p_target_id VARCHAR(255),
    p_tenant_id VARCHAR(255),
    p_trust_score DECIMAL(5,4),
    p_component_scores JSONB
) RETURNS TABLE(
    gate_id UUID,
    gate_type VARCHAR(50),
    passed BOOLEAN,
    violation_reason TEXT
) AS $$
DECLARE
    gate_record RECORD;
    component_key TEXT;
    required_score DECIMAL(5,4);
    actual_score DECIMAL(5,4);
BEGIN
    FOR gate_record IN 
        SELECT * FROM trust_gates 
        WHERE target_type = p_target_type 
        AND target_id = p_target_id 
        AND tenant_id = p_tenant_id 
        AND active = TRUE
    LOOP
        -- Check overall score
        IF p_trust_score < gate_record.minimum_trust_score THEN
            gate_id := gate_record.gate_id;
            gate_type := gate_record.gate_type;
            passed := FALSE;
            violation_reason := format('Overall trust score %s < required %s', 
                                     p_trust_score, gate_record.minimum_trust_score);
            RETURN NEXT;
            CONTINUE;
        END IF;
        
        -- Check component requirements
        FOR component_key IN SELECT jsonb_object_keys(gate_record.required_components)
        LOOP
            required_score := (gate_record.required_components->>component_key)::DECIMAL;
            actual_score := (p_component_scores->>component_key)::DECIMAL;
            
            IF actual_score < required_score THEN
                gate_id := gate_record.gate_id;
                gate_type := gate_record.gate_type;
                passed := FALSE;
                violation_reason := format('Component %s score %s < required %s', 
                                         component_key, actual_score, required_score);
                RETURN NEXT;
                CONTINUE;
            END IF;
        END LOOP;
        
        -- If we get here, gate passed
        gate_id := gate_record.gate_id;
        gate_type := gate_record.gate_type;
        passed := TRUE;
        violation_reason := NULL;
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to archive old trust score history
CREATE OR REPLACE FUNCTION archive_old_trust_history()
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
    cutoff_date TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Archive history older than 90 days
    cutoff_date := CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Move to archive table (not implemented here, but would be in real system)
    DELETE FROM trust_score_history 
    WHERE computed_at < cutoff_date;
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update trust score history
CREATE OR REPLACE FUNCTION update_trust_score_history()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert into history table
    INSERT INTO trust_score_history (
        target_type, target_id, tenant_id, overall_score, trust_level,
        component_scores, computed_at, change_reason
    ) VALUES (
        NEW.target_type, NEW.target_id, NEW.tenant_id, NEW.overall_score, NEW.trust_level,
        (SELECT jsonb_object_agg(component, jsonb_build_object('score', score, 'weight', weight, 'confidence', confidence))
         FROM trust_score_components WHERE trust_id = NEW.trust_id),
        NEW.computed_at,
        CASE WHEN TG_OP = 'INSERT' THEN 'initial_computation' ELSE 'score_update' END
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for trust score history
CREATE TRIGGER update_trust_history
    AFTER INSERT OR UPDATE ON trust_scores
    FOR EACH ROW
    EXECUTE FUNCTION update_trust_score_history();

-- Sample data for testing
INSERT INTO trust_scores (target_type, target_id, target_name, tenant_id, overall_score, trust_level, expires_at) VALUES
('model', 'churn_model_v2', 'Churn Prediction Model v2', 'tenant_saas_001', 0.85, 'good', CURRENT_TIMESTAMP + INTERVAL '1 hour'),
('model', 'fraud_detection_v3', 'Fraud Detection Model v3', 'tenant_banking_001', 0.92, 'excellent', CURRENT_TIMESTAMP + INTERVAL '1 hour'),
('workflow', 'lead_scoring_workflow', 'Lead Scoring Workflow', 'tenant_saas_001', 0.78, 'good', CURRENT_TIMESTAMP + INTERVAL '1 hour');

-- Sample component scores
INSERT INTO trust_score_components (trust_id, component, score, weight, confidence, metrics) 
SELECT 
    ts.trust_id,
    'quality',
    0.88,
    0.25,
    0.95,
    '{"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85}'::jsonb
FROM trust_scores ts WHERE ts.target_id = 'churn_model_v2';

INSERT INTO trust_score_components (trust_id, component, score, weight, confidence, metrics) 
SELECT 
    ts.trust_id,
    'explainability',
    0.82,
    0.20,
    0.90,
    '{"shap_available": 1.0, "explanation_coverage": 0.92, "human_interpretability": 0.80}'::jsonb
FROM trust_scores ts WHERE ts.target_id = 'churn_model_v2';

-- Sample trust gates
INSERT INTO trust_gates (target_type, target_id, tenant_id, gate_type, minimum_trust_score, required_components, created_by) VALUES
('model', 'churn_model_v2', 'tenant_saas_001', 'deployment', 0.75, '{"quality": 0.8, "bias": 0.7}'::jsonb, 'admin'),
('model', 'fraud_detection_v3', 'tenant_banking_001', 'deployment', 0.90, '{"quality": 0.85, "bias": 0.85, "explainability": 0.80}'::jsonb, 'admin');

-- Sample component weight configurations
INSERT INTO component_weight_configs (tenant_id, target_type, industry, config_name, weights, created_by) VALUES
('tenant_saas_001', 'model', 'SaaS', 'SaaS Model Weights', 
 '{"quality": 0.30, "explainability": 0.15, "drift": 0.20, "bias": 0.20, "ops_slo": 0.15}'::jsonb, 'admin'),
('tenant_banking_001', 'model', 'Banking', 'Banking Model Weights',
 '{"quality": 0.20, "explainability": 0.25, "drift": 0.15, "bias": 0.30, "ops_slo": 0.10}'::jsonb, 'admin');

COMMENT ON TABLE trust_scores IS 'Task 3.2.9: Overall trust scores for models, nodes, and workflows';
COMMENT ON TABLE trust_score_components IS 'Task 3.2.9: Individual component scores that make up trust score';
COMMENT ON TABLE trust_gates IS 'Task 3.2.9: Minimum trust score requirements for various operations';
COMMENT ON TABLE trust_score_history IS 'Task 3.2.9: Historical trust score changes for trend analysis';
COMMENT ON TABLE trust_gate_violations IS 'Task 3.2.9: Audit log of trust gate violations';
COMMENT ON TABLE component_weight_configs IS 'Task 3.2.9: Configurable weights for trust score components';
COMMENT ON TABLE trust_score_alerts IS 'Task 3.2.9: Alerts for trust score degradation and violations';
