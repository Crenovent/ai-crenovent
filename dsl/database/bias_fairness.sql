-- Task 3.2.10: Bias/Fairness Library Database Schema

-- Bias metrics definitions table
CREATE TABLE IF NOT EXISTS bias_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(20) NOT NULL CHECK (metric_name IN ('EQOD', 'DI', 'TPR_gap')),
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Industry-specific thresholds table
CREATE TABLE IF NOT EXISTS industry_bias_thresholds (
    threshold_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    industry VARCHAR(20) NOT NULL CHECK (industry IN ('SaaS', 'Banking', 'Insurance', 'FS', 'E-commerce')),
    metric_name VARCHAR(20) NOT NULL CHECK (metric_name IN ('EQOD', 'DI', 'TPR_gap')),
    threshold_value DECIMAL(5,4) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(industry, metric_name, tenant_id)
);

-- Bias check results table (evidence persistence)
CREATE TABLE IF NOT EXISTS bias_check_results (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id VARCHAR(255) NOT NULL,
    industry VARCHAR(20) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    metric_results JSONB NOT NULL,
    check_passed BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_bias_results_model (model_id, tenant_id),
    INDEX idx_bias_results_industry (industry, created_at)
);

-- Enable Row Level Security
ALTER TABLE industry_bias_thresholds ENABLE ROW LEVEL SECURITY;
ALTER TABLE bias_check_results ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY tenant_isolation_bias_thresholds ON industry_bias_thresholds
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

CREATE POLICY tenant_isolation_bias_results ON bias_check_results
    FOR ALL TO rbia_users
    USING (tenant_id = current_setting('rbia.current_tenant_id', true));

-- Sample data
INSERT INTO bias_metrics (metric_name, display_name, description) VALUES
('EQOD', 'Equalized Odds', 'Measures fairness by equalizing true positive and false positive rates across groups'),
('DI', 'Disparate Impact', 'Measures the ratio of positive outcomes between protected and unprotected groups'),
('TPR_gap', 'True Positive Rate Gap', 'Measures the difference in true positive rates between groups');

INSERT INTO industry_bias_thresholds (industry, metric_name, threshold_value, tenant_id) VALUES
('SaaS', 'EQOD', 0.10, 'tenant_saas_001'),
('SaaS', 'DI', 0.80, 'tenant_saas_001'),
('SaaS', 'TPR_gap', 0.05, 'tenant_saas_001'),
('Banking', 'EQOD', 0.05, 'tenant_banking_001'),
('Banking', 'DI', 0.80, 'tenant_banking_001'),
('Banking', 'TPR_gap', 0.03, 'tenant_banking_001');

COMMENT ON TABLE bias_metrics IS 'Task 3.2.10: Bias/fairness metric definitions';
COMMENT ON TABLE industry_bias_thresholds IS 'Task 3.2.10: Industry-specific bias metric thresholds';
COMMENT ON TABLE bias_check_results IS 'Task 3.2.10: Bias check results for evidence persistence';