-- Task 3.4.18: Differentiation KPIs Database Schema
-- Metrics pipeline service for tracking adoption rates, override reductions
-- Database schema for KPI storage and historical tracking

-- KPI metrics table
CREATE TABLE IF NOT EXISTS kpi_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(255), -- NULL for platform-wide metrics
    
    -- KPI identification
    kpi_type VARCHAR(100) NOT NULL CHECK (kpi_type IN (
        'user_adoption_rate', 'feature_adoption_rate', 'tenant_onboarding_time', 'workflow_creation_rate',
        'override_reduction_rate', 'policy_compliance_rate', 'evidence_pack_generation_rate', 'audit_readiness_score',
        'trust_score_improvement', 'explainability_usage_rate', 'shadow_mode_success_rate',
        'workflow_execution_time', 'false_positive_reduction', 'manual_intervention_reduction',
        'regulator_approval_rate', 'compliance_violation_reduction', 'data_residency_compliance_rate',
        'revenue_impact_positive', 'cost_savings_realized', 'deal_velocity_improvement', 'customer_satisfaction_score'
    )),
    kpi_category VARCHAR(50) NOT NULL CHECK (kpi_category IN (
        'adoption', 'governance', 'trust', 'efficiency', 'compliance', 'business_impact'
    )),
    metric_name VARCHAR(255) NOT NULL,
    metric_description TEXT,
    
    -- Metric value
    current_value DECIMAL(15,4) NOT NULL,
    previous_value DECIMAL(15,4),
    target_value DECIMAL(15,4),
    baseline_value DECIMAL(15,4),
    
    -- Units and context
    unit VARCHAR(50) NOT NULL, -- percentage, count, minutes, score
    measurement_period VARCHAR(20) NOT NULL, -- daily, weekly, monthly, quarterly
    measurement_date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Analysis
    trend VARCHAR(20) DEFAULT 'stable' CHECK (trend IN ('improving', 'stable', 'declining', 'volatile')),
    status VARCHAR(30) DEFAULT 'acceptable' CHECK (status IN ('excellent', 'good', 'acceptable', 'needs_improvement', 'critical')),
    variance_from_target DECIMAL(10,4),
    percentile_rank DECIMAL(5,2) CHECK (percentile_rank >= 0.0 AND percentile_rank <= 100.0),
    
    -- Metadata
    data_sources JSONB DEFAULT '[]',
    calculation_method TEXT,
    confidence_level DECIMAL(3,2) DEFAULT 0.8 CHECK (confidence_level >= 0.0 AND confidence_level <= 1.0),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- KPI historical data table
CREATE TABLE IF NOT EXISTS kpi_historical_data (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_id UUID NOT NULL REFERENCES kpi_metrics(metric_id) ON DELETE CASCADE,
    
    -- Historical values
    recorded_value DECIMAL(15,4) NOT NULL,
    recorded_date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Context at time of recording
    trend_at_time VARCHAR(20),
    status_at_time VARCHAR(30),
    target_at_time DECIMAL(15,4),
    
    -- Metadata
    data_source VARCHAR(255),
    calculation_context JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- KPI dashboards table
CREATE TABLE IF NOT EXISTS kpi_dashboards (
    dashboard_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dashboard_name VARCHAR(255) NOT NULL,
    dashboard_type VARCHAR(50) NOT NULL, -- executive, operational, detailed
    
    -- Configuration
    included_kpis JSONB DEFAULT '[]',
    included_categories JSONB DEFAULT '[]',
    tenant_filter VARCHAR(255),
    time_range_days INTEGER DEFAULT 30,
    
    -- Visualization settings
    chart_types JSONB DEFAULT '{}', -- kpi_type -> chart_type mapping
    refresh_interval_minutes INTEGER DEFAULT 15,
    
    -- Alerts
    alert_thresholds JSONB DEFAULT '{}',
    alert_recipients JSONB DEFAULT '[]',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_refreshed TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- KPI alerts table
CREATE TABLE IF NOT EXISTS kpi_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kpi_type VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(255),
    
    -- Alert details
    alert_type VARCHAR(50) NOT NULL, -- threshold_breach, trend_change, anomaly
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    message TEXT NOT NULL,
    
    -- Context
    current_value DECIMAL(15,4) NOT NULL,
    threshold_value DECIMAL(15,4),
    previous_value DECIMAL(15,4),
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- KPI reports table
CREATE TABLE IF NOT EXISTS kpi_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(50) NOT NULL, -- weekly, monthly, quarterly, annual
    report_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    report_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Summary metrics
    kpis_tracked INTEGER NOT NULL,
    kpis_improving INTEGER NOT NULL,
    kpis_declining INTEGER NOT NULL,
    overall_differentiation_score DECIMAL(5,2) NOT NULL,
    
    -- Category performance
    category_scores JSONB DEFAULT '{}',
    
    -- Key insights
    top_performing_kpis JSONB DEFAULT '[]',
    underperforming_kpis JSONB DEFAULT '[]',
    trend_analysis JSONB DEFAULT '{}',
    
    -- Recommendations
    improvement_recommendations JSONB DEFAULT '[]',
    strategic_priorities JSONB DEFAULT '[]',
    
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    generated_by VARCHAR(255) DEFAULT 'system'
);

-- KPI targets table
CREATE TABLE IF NOT EXISTS kpi_targets (
    target_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kpi_type VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(255), -- NULL for platform-wide targets
    
    -- Target details
    target_value DECIMAL(15,4) NOT NULL,
    baseline_value DECIMAL(15,4),
    target_period VARCHAR(20) NOT NULL, -- monthly, quarterly, annual
    
    -- Thresholds
    excellent_threshold DECIMAL(15,4),
    good_threshold DECIMAL(15,4),
    acceptable_threshold DECIMAL(15,4),
    critical_threshold DECIMAL(15,4),
    
    -- Validity
    effective_from TIMESTAMP WITH TIME ZONE NOT NULL,
    effective_until TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    set_by VARCHAR(255) NOT NULL,
    justification TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_kpi_metrics_type ON kpi_metrics(kpi_type);
CREATE INDEX idx_kpi_metrics_category ON kpi_metrics(kpi_category);
CREATE INDEX idx_kpi_metrics_tenant ON kpi_metrics(tenant_id);
CREATE INDEX idx_kpi_metrics_date ON kpi_metrics(measurement_date);
CREATE INDEX idx_kpi_metrics_status ON kpi_metrics(status);
CREATE INDEX idx_kpi_metrics_trend ON kpi_metrics(trend);

CREATE INDEX idx_kpi_historical_metric ON kpi_historical_data(metric_id);
CREATE INDEX idx_kpi_historical_date ON kpi_historical_data(recorded_date);

CREATE INDEX idx_kpi_dashboards_type ON kpi_dashboards(dashboard_type);
CREATE INDEX idx_kpi_dashboards_tenant ON kpi_dashboards(tenant_filter);

CREATE INDEX idx_kpi_alerts_type ON kpi_alerts(kpi_type);
CREATE INDEX idx_kpi_alerts_severity ON kpi_alerts(severity);
CREATE INDEX idx_kpi_alerts_active ON kpi_alerts(is_active);
CREATE INDEX idx_kpi_alerts_created ON kpi_alerts(created_at);

CREATE INDEX idx_kpi_reports_type ON kpi_reports(report_type);
CREATE INDEX idx_kpi_reports_period ON kpi_reports(report_period_start, report_period_end);

CREATE INDEX idx_kpi_targets_type ON kpi_targets(kpi_type);
CREATE INDEX idx_kpi_targets_tenant ON kpi_targets(tenant_id);
CREATE INDEX idx_kpi_targets_effective ON kpi_targets(effective_from, effective_until);

-- Functions for automatic updates
CREATE OR REPLACE FUNCTION update_kpi_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_kpi_metrics_updated_at
    BEFORE UPDATE ON kpi_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_kpi_updated_at();

CREATE TRIGGER update_kpi_dashboards_updated_at
    BEFORE UPDATE ON kpi_dashboards
    FOR EACH ROW
    EXECUTE FUNCTION update_kpi_updated_at();

-- Function to automatically create historical records
CREATE OR REPLACE FUNCTION create_kpi_historical_record()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO kpi_historical_data (
        metric_id, recorded_value, recorded_date, trend_at_time, 
        status_at_time, target_at_time
    ) VALUES (
        NEW.metric_id, NEW.current_value, NEW.measurement_date, 
        NEW.trend, NEW.status, NEW.target_value
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to create historical records
CREATE TRIGGER create_kpi_historical_record_trigger
    AFTER INSERT OR UPDATE ON kpi_metrics
    FOR EACH ROW
    EXECUTE FUNCTION create_kpi_historical_record();

-- Function to calculate KPI trend
CREATE OR REPLACE FUNCTION calculate_kpi_trend(
    p_metric_id UUID,
    p_lookback_days INTEGER DEFAULT 30
) RETURNS VARCHAR AS $$
DECLARE
    recent_values DECIMAL[];
    trend_result VARCHAR;
    slope DECIMAL;
BEGIN
    -- Get recent values
    SELECT ARRAY_AGG(recorded_value ORDER BY recorded_date)
    INTO recent_values
    FROM kpi_historical_data
    WHERE metric_id = p_metric_id
    AND recorded_date >= CURRENT_TIMESTAMP - INTERVAL '1 day' * p_lookback_days
    AND recorded_date <= CURRENT_TIMESTAMP;
    
    -- Calculate simple trend
    IF array_length(recent_values, 1) < 2 THEN
        RETURN 'stable';
    END IF;
    
    -- Simple slope calculation (last value - first value)
    slope := recent_values[array_length(recent_values, 1)] - recent_values[1];
    
    IF slope > 0.05 THEN
        trend_result := 'improving';
    ELSIF slope < -0.05 THEN
        trend_result := 'declining';
    ELSE
        trend_result := 'stable';
    END IF;
    
    RETURN trend_result;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate differentiation score
CREATE OR REPLACE FUNCTION calculate_differentiation_score(
    p_tenant_id VARCHAR DEFAULT NULL,
    p_category VARCHAR DEFAULT NULL
) RETURNS DECIMAL AS $$
DECLARE
    category_weights JSONB;
    weighted_score DECIMAL := 0;
    total_weight DECIMAL := 0;
    category_score DECIMAL;
    weight DECIMAL;
BEGIN
    -- Define category weights
    category_weights := '{
        "adoption": 0.20,
        "governance": 0.25,
        "trust": 0.20,
        "efficiency": 0.15,
        "compliance": 0.15,
        "business_impact": 0.05
    }'::jsonb;
    
    -- Calculate weighted score for each category
    FOR category_score, weight IN
        SELECT 
            AVG(CASE 
                WHEN status = 'excellent' THEN 95.0
                WHEN status = 'good' THEN 80.0
                WHEN status = 'acceptable' THEN 65.0
                WHEN status = 'needs_improvement' THEN 40.0
                WHEN status = 'critical' THEN 20.0
                ELSE 50.0
            END) as avg_score,
            (category_weights->>kpi_category)::decimal as cat_weight
        FROM kpi_metrics
        WHERE (p_tenant_id IS NULL OR tenant_id = p_tenant_id)
        AND (p_category IS NULL OR kpi_category = p_category)
        AND measurement_date >= CURRENT_TIMESTAMP - INTERVAL '30 days'
        GROUP BY kpi_category
    LOOP
        weighted_score := weighted_score + (category_score * weight);
        total_weight := total_weight + weight;
    END LOOP;
    
    RETURN CASE 
        WHEN total_weight > 0 THEN weighted_score / total_weight
        ELSE 0
    END;
END;
$$ LANGUAGE plpgsql;

-- Function to generate KPI alert
CREATE OR REPLACE FUNCTION check_kpi_threshold_breach(
    p_metric_id UUID
) RETURNS BOOLEAN AS $$
DECLARE
    metric_record RECORD;
    target_record RECORD;
    alert_needed BOOLEAN := FALSE;
BEGIN
    -- Get metric details
    SELECT * INTO metric_record
    FROM kpi_metrics
    WHERE metric_id = p_metric_id;
    
    -- Get current target
    SELECT * INTO target_record
    FROM kpi_targets
    WHERE kpi_type = metric_record.kpi_type
    AND (tenant_id IS NULL OR tenant_id = metric_record.tenant_id)
    AND effective_from <= CURRENT_TIMESTAMP
    AND (effective_until IS NULL OR effective_until > CURRENT_TIMESTAMP)
    ORDER BY effective_from DESC
    LIMIT 1;
    
    -- Check for threshold breach
    IF target_record.target_id IS NOT NULL THEN
        IF metric_record.current_value < target_record.critical_threshold THEN
            INSERT INTO kpi_alerts (
                kpi_type, tenant_id, alert_type, severity, message,
                current_value, threshold_value
            ) VALUES (
                metric_record.kpi_type, metric_record.tenant_id, 'threshold_breach', 'critical',
                format('%s is critically below threshold: %s %s (threshold: %s %s)',
                    metric_record.metric_name, metric_record.current_value, metric_record.unit,
                    target_record.critical_threshold, metric_record.unit),
                metric_record.current_value, target_record.critical_threshold
            );
            alert_needed := TRUE;
        END IF;
    END IF;
    
    RETURN alert_needed;
END;
$$ LANGUAGE plpgsql;

-- Sample data for testing
INSERT INTO kpi_metrics (kpi_type, kpi_category, metric_name, metric_description, current_value, target_value, baseline_value, unit, measurement_period, calculation_method) VALUES
('user_adoption_rate', 'adoption', 'User Adoption Rate', 'Percentage of users actively using RBIA features', 78.5, 85.0, 65.0, 'percentage', 'monthly', '(active_users / total_users) * 100'),
('override_reduction_rate', 'governance', 'Override Reduction Rate', 'Reduction in manual overrides due to improved automation', 42.3, 50.0, 0.0, 'percentage', 'monthly', '((baseline_overrides - current_overrides) / baseline_overrides) * 100'),
('trust_score_improvement', 'trust', 'Trust Score Improvement', 'Average improvement in trust scores across all models', 15.7, 20.0, 0.0, 'percentage', 'monthly', '((current_avg_trust - baseline_avg_trust) / baseline_avg_trust) * 100'),
('regulator_approval_rate', 'compliance', 'Regulator Approval Rate', 'Percentage of regulatory submissions approved without issues', 94.2, 95.0, 85.0, 'percentage', 'quarterly', '(approved_submissions / total_submissions) * 100');

INSERT INTO kpi_targets (kpi_type, target_value, baseline_value, target_period, excellent_threshold, good_threshold, acceptable_threshold, critical_threshold, effective_from, set_by) VALUES
('user_adoption_rate', 85.0, 65.0, 'monthly', 90.0, 80.0, 70.0, 50.0, CURRENT_TIMESTAMP, 'system'),
('override_reduction_rate', 50.0, 0.0, 'monthly', 60.0, 45.0, 30.0, 15.0, CURRENT_TIMESTAMP, 'system'),
('trust_score_improvement', 20.0, 0.0, 'monthly', 25.0, 18.0, 12.0, 5.0, CURRENT_TIMESTAMP, 'system'),
('regulator_approval_rate', 95.0, 85.0, 'quarterly', 98.0, 93.0, 88.0, 80.0, CURRENT_TIMESTAMP, 'system');

INSERT INTO kpi_dashboards (dashboard_name, dashboard_type, included_kpis, included_categories, chart_types) VALUES
('Executive KPI Overview', 'executive', '["user_adoption_rate", "override_reduction_rate", "trust_score_improvement", "regulator_approval_rate"]', '["adoption", "governance", "trust", "compliance"]', '{"user_adoption_rate": "gauge", "override_reduction_rate": "line", "trust_score_improvement": "bar", "regulator_approval_rate": "scorecard"}'),
('Operational Metrics Dashboard', 'operational', '["workflow_execution_time", "false_positive_reduction", "manual_intervention_reduction"]', '["efficiency"]', '{"workflow_execution_time": "line", "false_positive_reduction": "area", "manual_intervention_reduction": "bar"}');
