-- Task 3.4.15: Competitive Feature Benchmark Database Schema
-- Backend API service for competitive benchmarking
-- Database schema for storing competitor feature comparisons

-- Competitors table
CREATE TABLE IF NOT EXISTS competitors (
    competitor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    competitor_type VARCHAR(50) NOT NULL CHECK (competitor_type IN ('rba_only', 'ai_first_blackbox', 'hybrid', 'enterprise_platform')),
    description TEXT,
    website_url VARCHAR(500),
    
    -- Market position
    market_cap_tier VARCHAR(50) DEFAULT 'unknown',
    target_industries JSONB DEFAULT '[]',
    primary_personas JSONB DEFAULT '[]',
    
    -- Business model
    pricing_model VARCHAR(100) DEFAULT 'unknown',
    deployment_options JSONB DEFAULT '[]',
    
    -- Metadata
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    data_sources JSONB DEFAULT '[]',
    confidence_score DECIMAL(3,2) DEFAULT 0.7 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Feature benchmarks table
CREATE TABLE IF NOT EXISTS feature_benchmarks (
    benchmark_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_id UUID NOT NULL REFERENCES competitors(competitor_id) ON DELETE CASCADE,
    
    -- Feature details
    feature_category VARCHAR(50) NOT NULL CHECK (feature_category IN (
        'governance', 'explainability', 'multi_tenancy', 'industry_overlays', 
        'ux_modes', 'trust_scoring', 'compliance', 'auditability', 
        'reversibility', 'cost_transparency'
    )),
    feature_name VARCHAR(255) NOT NULL,
    feature_description TEXT,
    
    -- Support assessment
    support_level VARCHAR(50) NOT NULL CHECK (support_level IN (
        'full_support', 'partial_support', 'limited_support', 'no_support', 'unknown'
    )),
    implementation_notes TEXT,
    evidence_sources JSONB DEFAULT '[]',
    
    -- Scoring (1-10 scale)
    functionality_score DECIMAL(3,1) DEFAULT 0.0 CHECK (functionality_score >= 0.0 AND functionality_score <= 10.0),
    usability_score DECIMAL(3,1) DEFAULT 0.0 CHECK (usability_score >= 0.0 AND usability_score <= 10.0),
    enterprise_readiness_score DECIMAL(3,1) DEFAULT 0.0 CHECK (enterprise_readiness_score >= 0.0 AND enterprise_readiness_score <= 10.0),
    compliance_score DECIMAL(3,1) DEFAULT 0.0 CHECK (compliance_score >= 0.0 AND compliance_score <= 10.0),
    
    -- RBIA comparison
    rbia_advantage TEXT,
    competitive_gap TEXT,
    
    -- Metadata
    assessed_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assessed_by VARCHAR(255) DEFAULT 'system',
    confidence_level DECIMAL(3,2) DEFAULT 0.7 CHECK (confidence_level >= 0.0 AND confidence_level <= 1.0),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint to prevent duplicate benchmarks
    UNIQUE(competitor_id, feature_category, feature_name)
);

-- Competitive dashboards table
CREATE TABLE IF NOT EXISTS competitive_dashboards (
    dashboard_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dashboard_name VARCHAR(255) NOT NULL,
    target_audience VARCHAR(100) NOT NULL, -- sales, marketing, product, executives
    
    -- Dashboard configuration
    included_competitors JSONB DEFAULT '[]',
    included_categories JSONB DEFAULT '[]',
    benchmark_metrics JSONB DEFAULT '[]',
    
    -- Visualization settings
    chart_types JSONB DEFAULT '[]',
    refresh_interval_minutes INTEGER DEFAULT 60,
    
    -- Access control
    allowed_roles JSONB DEFAULT '[]',
    tenant_id VARCHAR(255),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_refreshed TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Benchmark reports table
CREATE TABLE IF NOT EXISTS benchmark_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(100) NOT NULL, -- executive_summary, detailed_analysis, sales_battlecard
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Report content
    executive_summary JSONB DEFAULT '{}',
    competitive_positioning JSONB DEFAULT '{}',
    feature_comparison_matrix JSONB DEFAULT '[]',
    rbia_advantages JSONB DEFAULT '[]',
    competitive_gaps JSONB DEFAULT '[]',
    
    -- Recommendations
    sales_talking_points JSONB DEFAULT '[]',
    product_priorities JSONB DEFAULT '[]',
    marketing_messages JSONB DEFAULT '[]',
    
    -- Metadata
    included_competitors JSONB DEFAULT '[]',
    target_audience VARCHAR(100),
    generated_by VARCHAR(255) DEFAULT 'system',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Competitive metrics tracking table
CREATE TABLE IF NOT EXISTS competitive_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_id UUID NOT NULL REFERENCES competitors(competitor_id) ON DELETE CASCADE,
    
    -- Metric details
    metric_type VARCHAR(100) NOT NULL, -- feature_coverage, market_adoption, customer_satisfaction
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    metric_unit VARCHAR(50), -- percentage, score, count
    
    -- Context
    measurement_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    data_source VARCHAR(255),
    confidence_score DECIMAL(3,2) DEFAULT 0.7,
    
    -- Comparison context
    industry_benchmark DECIMAL(10,4),
    rbia_comparison_value DECIMAL(10,4),
    competitive_advantage BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_competitors_type ON competitors(competitor_type);
CREATE INDEX idx_competitors_updated ON competitors(last_updated);

CREATE INDEX idx_benchmarks_competitor ON feature_benchmarks(competitor_id);
CREATE INDEX idx_benchmarks_category ON feature_benchmarks(feature_category);
CREATE INDEX idx_benchmarks_support ON feature_benchmarks(support_level);
CREATE INDEX idx_benchmarks_scores ON feature_benchmarks(functionality_score, enterprise_readiness_score);
CREATE INDEX idx_benchmarks_assessed ON feature_benchmarks(assessed_date);

CREATE INDEX idx_dashboards_audience ON competitive_dashboards(target_audience);
CREATE INDEX idx_dashboards_tenant ON competitive_dashboards(tenant_id);
CREATE INDEX idx_dashboards_refreshed ON competitive_dashboards(last_refreshed);

CREATE INDEX idx_reports_type ON benchmark_reports(report_type);
CREATE INDEX idx_reports_generated ON benchmark_reports(generated_at);

CREATE INDEX idx_metrics_competitor ON competitive_metrics(competitor_id);
CREATE INDEX idx_metrics_type ON competitive_metrics(metric_type);
CREATE INDEX idx_metrics_date ON competitive_metrics(measurement_date);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_competitive_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_competitors_updated_at
    BEFORE UPDATE ON competitors
    FOR EACH ROW
    EXECUTE FUNCTION update_competitive_updated_at();

CREATE TRIGGER update_benchmarks_updated_at
    BEFORE UPDATE ON feature_benchmarks
    FOR EACH ROW
    EXECUTE FUNCTION update_competitive_updated_at();

CREATE TRIGGER update_dashboards_updated_at
    BEFORE UPDATE ON competitive_dashboards
    FOR EACH ROW
    EXECUTE FUNCTION update_competitive_updated_at();

-- Function to calculate competitive advantage score
CREATE OR REPLACE FUNCTION calculate_competitive_advantage_score(
    p_competitor_id UUID,
    p_feature_category VARCHAR DEFAULT NULL
) RETURNS DECIMAL AS $$
DECLARE
    avg_score DECIMAL;
BEGIN
    SELECT AVG((functionality_score + usability_score + enterprise_readiness_score + compliance_score) / 4.0)
    INTO avg_score
    FROM feature_benchmarks
    WHERE competitor_id = p_competitor_id
    AND (p_feature_category IS NULL OR feature_category = p_feature_category);
    
    RETURN COALESCE(avg_score, 0.0);
END;
$$ LANGUAGE plpgsql;

-- Function to get RBIA advantages count
CREATE OR REPLACE FUNCTION get_rbia_advantages_count(
    p_competitor_id UUID DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
    advantage_count INTEGER;
BEGIN
    SELECT COUNT(*)
    INTO advantage_count
    FROM feature_benchmarks
    WHERE (p_competitor_id IS NULL OR competitor_id = p_competitor_id)
    AND rbia_advantage IS NOT NULL
    AND rbia_advantage != '';
    
    RETURN advantage_count;
END;
$$ LANGUAGE plpgsql;

-- Sample data for testing
INSERT INTO competitors (competitor_id, name, competitor_type, description, website_url, market_cap_tier, target_industries, primary_personas) VALUES
('clari-001', 'Clari', 'rba_only', 'Revenue operations platform with rule-based automation', 'https://clari.com', 'public', '["SaaS", "Technology"]', '["Revenue Operations", "Sales Leadership"]'),
('gong-001', 'Gong', 'ai_first_blackbox', 'AI-first conversation analytics with black-box models', 'https://gong.io', 'public', '["SaaS", "Technology", "Financial Services"]', '["Sales Teams", "Revenue Operations"]'),
('salesforce-001', 'Salesforce Einstein', 'enterprise_platform', 'Enterprise CRM with AI features but limited governance', 'https://salesforce.com', 'public', '["All"]', '["Sales Teams", "Marketing", "Service"]');

INSERT INTO feature_benchmarks (competitor_id, feature_category, feature_name, feature_description, support_level, functionality_score, usability_score, enterprise_readiness_score, compliance_score, rbia_advantage, competitive_gap) VALUES
('clari-001', 'governance', 'Policy Enforcement', 'Ability to enforce business policies and compliance rules', 'limited_support', 4.0, 6.0, 5.0, 4.0, 'Dynamic policy packs with industry-specific overlays', 'No real-time policy adaptation or compliance frameworks'),
('gong-001', 'explainability', 'AI Explainability', 'Ability to explain AI model decisions and predictions', 'no_support', 1.0, 2.0, 2.0, 1.0, 'Full SHAP/LIME explainability with inline explanations', 'Complete lack of model transparency and explainability'),
('salesforce-001', 'multi_tenancy', 'Multi-Tenant Isolation', 'Tenant-level data isolation and governance', 'partial_support', 6.0, 7.0, 7.0, 5.0, 'RLS-enforced tenant isolation with governance metadata', 'No tenant-specific compliance overlays or evidence packs');

INSERT INTO competitive_dashboards (dashboard_name, target_audience, included_competitors, included_categories, benchmark_metrics) VALUES
('Executive Competitive Overview', 'executives', '["clari-001", "gong-001", "salesforce-001"]', '["governance", "explainability", "multi_tenancy"]', '["feature_coverage", "implementation_quality", "regulatory_compliance"]'),
('Sales Battlecard Dashboard', 'sales', '["clari-001", "gong-001"]', '["governance", "explainability", "trust_scoring"]', '["feature_coverage", "customer_satisfaction"]');
