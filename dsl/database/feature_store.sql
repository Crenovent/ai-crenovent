-- Feature Store Database Schema - Task 6.1.14
-- ============================================
-- Supports offline (training/batch) and online (real-time) feature storage

-- Feature metadata registry
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_id VARCHAR(255) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    feature_type VARCHAR(50) NOT NULL CHECK (feature_type IN ('numeric', 'categorical', 'boolean', 'text', 'embedding', 'timestamp')),
    description TEXT,
    
    -- Versioning
    version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Ownership
    owner VARCHAR(255) NOT NULL DEFAULT 'system',
    tenant_id INTEGER NOT NULL,
    
    -- Lineage
    source_dataset VARCHAR(255),
    transformation_logic TEXT,
    dependencies TEXT[],  -- Array of dependent feature_ids
    
    -- Schema
    data_type VARCHAR(50) NOT NULL DEFAULT 'float',
    allowed_values JSONB,  -- For categorical features
    min_value DECIMAL(20,6),
    max_value DECIMAL(20,6),
    
    -- Quality
    null_allowed BOOLEAN DEFAULT FALSE,
    importance_score DECIMAL(5,4) DEFAULT 0.5,
    
    -- Tags
    tags TEXT[],
    
    -- Storage configuration
    storage_backend VARCHAR(20) NOT NULL DEFAULT 'both' CHECK (storage_backend IN ('offline', 'online', 'both')),
    ttl_seconds INTEGER DEFAULT 3600,  -- TTL for online cache
    
    -- Primary key
    PRIMARY KEY (tenant_id, feature_id, version),
    
    -- Foreign key
    CONSTRAINT fk_feature_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id) ON DELETE CASCADE
);

-- Indexes for feature metadata
CREATE INDEX idx_feature_metadata_tenant ON feature_metadata(tenant_id);
CREATE INDEX idx_feature_metadata_name ON feature_metadata(feature_name);
CREATE INDEX idx_feature_metadata_type ON feature_metadata(feature_type);
CREATE INDEX idx_feature_metadata_tags ON feature_metadata USING GIN(tags);

-- Offline feature values (historical/training data)
CREATE TABLE IF NOT EXISTS feature_values_offline (
    value_id BIGSERIAL PRIMARY KEY,
    feature_id VARCHAR(255) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,  -- customer_id, opportunity_id, etc.
    value JSONB NOT NULL,  -- Feature value (supports complex types)
    
    -- Timestamps
    event_timestamp TIMESTAMPTZ NOT NULL,  -- When the event occurred
    ingestion_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- When ingested to feature store
    
    -- Metadata
    tenant_id INTEGER NOT NULL,
    version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    source VARCHAR(255) DEFAULT 'unknown',  -- Data source
    
    -- Partitioning key (for large-scale data)
    partition_date DATE GENERATED ALWAYS AS (DATE(event_timestamp)) STORED,
    
    -- Foreign keys
    CONSTRAINT fk_feature_value_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id) ON DELETE CASCADE
);

-- Indexes for offline feature values
CREATE INDEX idx_feature_values_offline_tenant ON feature_values_offline(tenant_id, feature_id);
CREATE INDEX idx_feature_values_offline_entity ON feature_values_offline(entity_id, feature_id);
CREATE INDEX idx_feature_values_offline_timestamp ON feature_values_offline(event_timestamp DESC);
CREATE INDEX idx_feature_values_offline_partition ON feature_values_offline(partition_date DESC);

-- Partitioning for offline features (optional, for scale)
-- CREATE TABLE feature_values_offline_2024_01 PARTITION OF feature_values_offline
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Feature vector snapshots (pre-computed feature vectors for entities)
CREATE TABLE IF NOT EXISTS feature_vectors (
    vector_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,  -- customer, opportunity, account
    features JSONB NOT NULL,  -- Complete feature vector
    
    -- Metadata
    tenant_id INTEGER NOT NULL,
    version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    
    -- Quality indicators
    completeness DECIMAL(5,4) DEFAULT 1.0,  -- % of features present
    freshness_seconds INTEGER DEFAULT 0,  -- Age of oldest feature
    
    -- Timestamps
    snapshot_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Foreign key
    CONSTRAINT fk_feature_vector_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id) ON DELETE CASCADE
);

-- Indexes for feature vectors
CREATE INDEX idx_feature_vectors_entity ON feature_vectors(tenant_id, entity_id, entity_type);
CREATE INDEX idx_feature_vectors_timestamp ON feature_vectors(snapshot_timestamp DESC);

-- Feature usage tracking (for monitoring and optimization)
CREATE TABLE IF NOT EXISTS feature_usage_stats (
    stat_id BIGSERIAL PRIMARY KEY,
    feature_id VARCHAR(255) NOT NULL,
    tenant_id INTEGER NOT NULL,
    
    -- Usage metrics
    read_count BIGINT DEFAULT 0,
    write_count BIGINT DEFAULT 0,
    cache_hits BIGINT DEFAULT 0,
    cache_misses BIGINT DEFAULT 0,
    
    -- Performance metrics
    avg_read_latency_ms DECIMAL(10,2) DEFAULT 0,
    avg_write_latency_ms DECIMAL(10,2) DEFAULT 0,
    
    -- Quality metrics
    null_count BIGINT DEFAULT 0,
    out_of_range_count BIGINT DEFAULT 0,
    
    -- Time period
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    
    -- Foreign key
    CONSTRAINT fk_feature_usage_tenant FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id) ON DELETE CASCADE
);

-- Indexes for feature usage stats
CREATE INDEX idx_feature_usage_stats_feature ON feature_usage_stats(tenant_id, feature_id);
CREATE INDEX idx_feature_usage_stats_period ON feature_usage_stats(period_start DESC);

-- Enable Row Level Security
ALTER TABLE feature_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_values_offline ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_vectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_usage_stats ENABLE ROW LEVEL SECURITY;

-- RLS Policies for tenant isolation
CREATE POLICY feature_metadata_tenant_isolation ON feature_metadata
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY feature_values_offline_tenant_isolation ON feature_values_offline
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY feature_vectors_tenant_isolation ON feature_vectors
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

CREATE POLICY feature_usage_stats_tenant_isolation ON feature_usage_stats
    FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));

-- Utility functions

-- Get latest feature value for an entity
CREATE OR REPLACE FUNCTION get_latest_feature_value(
    p_tenant_id INTEGER,
    p_feature_id VARCHAR,
    p_entity_id VARCHAR,
    p_version VARCHAR DEFAULT '1.0.0'
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT value INTO result
    FROM feature_values_offline
    WHERE tenant_id = p_tenant_id
      AND feature_id = p_feature_id
      AND entity_id = p_entity_id
      AND version = p_version
    ORDER BY event_timestamp DESC
    LIMIT 1;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get feature vector for an entity
CREATE OR REPLACE FUNCTION get_feature_vector(
    p_tenant_id INTEGER,
    p_entity_id VARCHAR,
    p_feature_ids VARCHAR[],
    p_version VARCHAR DEFAULT '1.0.0'
) RETURNS JSONB AS $$
DECLARE
    feature_vector JSONB := '{}'::JSONB;
    feature_id VARCHAR;
    feature_value JSONB;
BEGIN
    FOREACH feature_id IN ARRAY p_feature_ids
    LOOP
        feature_value := get_latest_feature_value(p_tenant_id, feature_id, p_entity_id, p_version);
        IF feature_value IS NOT NULL THEN
            feature_vector := jsonb_set(feature_vector, ARRAY[feature_id], feature_value);
        END IF;
    END LOOP;
    
    RETURN feature_vector;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Track feature usage
CREATE OR REPLACE FUNCTION track_feature_usage(
    p_tenant_id INTEGER,
    p_feature_id VARCHAR,
    p_operation VARCHAR,  -- 'read' or 'write'
    p_cache_hit BOOLEAN DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    -- In production, this would update stats table
    -- For now, just a placeholder
    NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Sample data for testing
INSERT INTO feature_metadata (feature_id, feature_name, feature_type, description, tenant_id, owner, source_dataset, data_type) VALUES
('customer_mrr', 'Monthly Recurring Revenue', 'numeric', 'Customer MRR in USD', 1300, 'data_team', 'billing_data', 'float'),
('customer_tenure_days', 'Customer Tenure', 'numeric', 'Days since customer signup', 1300, 'data_team', 'user_data', 'integer'),
('customer_engagement_score', 'Engagement Score', 'numeric', 'Customer engagement score (0-100)', 1300, 'ml_team', 'activity_data', 'float'),
('customer_industry', 'Customer Industry', 'categorical', 'Industry vertical', 1300, 'data_team', 'account_data', 'string'),
('opportunity_stage', 'Opportunity Stage', 'categorical', 'Sales stage', 1300, 'sales_team', 'crm_data', 'string')
ON CONFLICT (tenant_id, feature_id, version) DO NOTHING;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON feature_metadata TO rbia_users;
GRANT SELECT, INSERT ON feature_values_offline TO rbia_users;
GRANT SELECT, INSERT ON feature_vectors TO rbia_users;
GRANT SELECT, INSERT, UPDATE ON feature_usage_stats TO rbia_users;
GRANT USAGE ON SEQUENCE feature_values_offline_value_id_seq TO rbia_users;
GRANT USAGE ON SEQUENCE feature_usage_stats_stat_id_seq TO rbia_users;

