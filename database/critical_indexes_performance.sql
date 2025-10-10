-- Task 7.2-T30: Define indexes (btrees, gin) for critical queries
-- Performance-critical indexes based on RBA workload patterns
-- Version: 1.0.0
-- Date: 2024-10-08

-- ============================================================================
-- INDEX STRATEGY DOCUMENTATION
-- ============================================================================

/*
Index Strategy for RBA Workloads:

1. TENANT ISOLATION INDEXES
   - Every table needs btree(tenant_id) for RLS performance
   - Composite indexes with tenant_id as first column

2. TEMPORAL INDEXES  
   - created_at/updated_at for time-based queries
   - Descending order for "recent first" queries

3. STATUS INDEXES
   - Partial indexes on active/non-deleted records
   - Status-based filtering is common

4. FOREIGN KEY INDEXES
   - All foreign keys need indexes for join performance
   - Especially important for cascade operations

5. FULL-TEXT SEARCH INDEXES
   - GIN indexes for text search on names/descriptions
   - JSONB GIN indexes for metadata search

6. COMPOSITE INDEXES
   - Multi-column indexes for common query patterns
   - Order columns by selectivity (most selective first)
*/

-- ============================================================================
-- TENANT ISOLATION INDEXES (CRITICAL)
-- ============================================================================

-- Tenant metadata - primary lookup table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_metadata_tenant_id 
ON tenant_metadata (tenant_id) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tenant_metadata_industry_region 
ON tenant_metadata (industry_code, region, status) WHERE deleted_at IS NULL;

-- Users - frequent lookups by tenant and email
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_tenant_id 
ON users (tenant_id) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_tenant 
ON users (email, tenant_id) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_status_tenant 
ON users (tenant_id, status) WHERE deleted_at IS NULL;

-- User roles - authorization lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_role_tenant 
ON users_role (tenant_id, user_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_role_permissions 
ON users_role USING GIN (permissions) WHERE tenant_id IS NOT NULL;

-- ============================================================================
-- DSL WORKFLOW INDEXES (HIGH PRIORITY)
-- ============================================================================

-- DSL Workflows - core RBA entity
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflows_tenant_status 
ON dsl_workflows (tenant_id, status) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflows_type_automation 
ON dsl_workflows (tenant_id, workflow_type, automation_type) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflows_industry 
ON dsl_workflows (tenant_id, industry_overlay) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflows_created_at 
ON dsl_workflows (tenant_id, created_at DESC) WHERE deleted_at IS NULL;

-- Full-text search on workflow names and descriptions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflows_search 
ON dsl_workflows USING GIN (to_tsvector('english', workflow_name || ' ' || COALESCE(description, ''))) 
WHERE deleted_at IS NULL;

-- DSL Execution Traces - high-volume table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_execution_traces_workflow 
ON dsl_execution_traces (workflow_id, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_execution_traces_tenant_status 
ON dsl_execution_traces (tenant_id, status, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_execution_traces_user_activity 
ON dsl_execution_traces (user_id, created_at DESC) WHERE user_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_execution_traces_trust_score 
ON dsl_execution_traces (tenant_id, trust_score) WHERE trust_score IS NOT NULL;

-- Execution time performance analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_execution_traces_performance 
ON dsl_execution_traces (workflow_id, execution_time_ms) WHERE execution_time_ms > 0;

-- DSL Policy Packs - governance lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_policy_packs_tenant_status 
ON dsl_policy_packs (tenant_id, status) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_policy_packs_industry 
ON dsl_policy_packs (tenant_id, industry_overlay) WHERE deleted_at IS NULL;

-- Policy pack content search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_policy_packs_content 
ON dsl_policy_packs USING GIN (policy_content) WHERE deleted_at IS NULL;

-- ============================================================================
-- GOVERNANCE AND COMPLIANCE INDEXES
-- ============================================================================

-- DSL Evidence Packs - compliance queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_evidence_packs_tenant_type 
ON dsl_evidence_packs (tenant_id, pack_type, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_evidence_packs_workflow 
ON dsl_evidence_packs (workflow_id, created_at DESC) WHERE workflow_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_evidence_packs_compliance 
ON dsl_evidence_packs (tenant_id, compliance_framework) WHERE compliance_framework IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_evidence_packs_retention 
ON dsl_evidence_packs (expires_at) WHERE expires_at IS NOT NULL;

-- Evidence data search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_evidence_packs_data 
ON dsl_evidence_packs USING GIN (evidence_data);

-- DSL Override Ledger - governance violations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_override_ledger_tenant_risk 
ON dsl_override_ledger (tenant_id, risk_level, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_override_ledger_workflow 
ON dsl_override_ledger (workflow_id, created_at DESC) WHERE workflow_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_override_ledger_user 
ON dsl_override_ledger (user_id, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_override_ledger_approval 
ON dsl_override_ledger (approved_by_user_id, approved_at DESC) WHERE approved_by_user_id IS NOT NULL;

-- Pending approvals
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_override_ledger_pending 
ON dsl_override_ledger (tenant_id, created_at DESC) 
WHERE approval_required = true AND approved_at IS NULL;

-- DSL Workflow Templates - template discovery
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflow_templates_tenant 
ON dsl_workflow_templates (tenant_id, is_active) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflow_templates_industry 
ON dsl_workflow_templates (industry_overlay, is_active) WHERE deleted_at IS NULL;

-- Template search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflow_templates_search 
ON dsl_workflow_templates USING GIN (to_tsvector('english', template_name || ' ' || COALESCE(description, ''))) 
WHERE deleted_at IS NULL;

-- DSL Capability Registry - capability lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_capability_registry_tenant 
ON dsl_capability_registry (tenant_id, is_active) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_capability_registry_type 
ON dsl_capability_registry (capability_type, is_active) WHERE deleted_at IS NULL;

-- ============================================================================
-- STRATEGIC ACCOUNT PLANNING INDEXES
-- ============================================================================

-- Strategic Account Plans - business entity lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategic_account_plans_tenant 
ON strategic_account_plans (tenant_id, status) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategic_account_plans_account 
ON strategic_account_plans (account_id, status) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategic_account_plans_owner 
ON strategic_account_plans (created_by_user_id, created_at DESC) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategic_account_plans_template 
ON strategic_account_plans (template_id) WHERE template_id IS NOT NULL;

-- Plan search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategic_account_plans_search 
ON strategic_account_plans USING GIN (to_tsvector('english', plan_name || ' ' || COALESCE(short_term_goals, '') || ' ' || COALESCE(long_term_goals, ''))) 
WHERE deleted_at IS NULL;

-- Account Planning Templates
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_planning_templates_tenant 
ON account_planning_templates (tenant_id, is_active) WHERE deleted_at IS NULL;

-- Opportunity Insights - CRM integration
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_opportunity_insights_tenant 
ON opportunity_insights (tenant_id, created_at DESC) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_opportunity_insights_account 
ON opportunity_insights (account_id, created_at DESC) WHERE account_id IS NOT NULL;

-- Risk flags search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_opportunity_insights_risk 
ON opportunity_insights USING GIN (risk_flags) WHERE risk_flags IS NOT NULL;

-- Plan Stakeholders - relationship queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_stakeholders_plan 
ON plan_stakeholders (plan_id, influence_level) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_stakeholders_contact 
ON plan_stakeholders (contact_name, influence_level) WHERE deleted_at IS NULL;

-- Plan Activities - timeline queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_activities_plan_date 
ON plan_activities (plan_id, activity_date DESC) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_activities_owner_date 
ON plan_activities (owner_user_id, activity_date DESC) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_activities_status 
ON plan_activities (plan_id, status, activity_date DESC) WHERE deleted_at IS NULL;

-- Plan Collaborators - access control
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_collaborators_plan 
ON plan_collaborators (plan_id, permission_level) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_collaborators_user 
ON plan_collaborators (user_id, permission_level) WHERE deleted_at IS NULL;

-- Plan Comments - activity feed
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_comments_plan_date 
ON plan_comments (plan_id, created_at DESC) WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_comments_user 
ON plan_comments (user_id, created_at DESC) WHERE deleted_at IS NULL;

-- Plan Insight Attachments - document management
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_insight_attachments_plan 
ON plan_insight_attachments (plan_id, attachment_type) WHERE deleted_at IS NULL;

-- ============================================================================
-- COMPOSITE INDEXES FOR COMPLEX QUERIES
-- ============================================================================

-- Workflow execution analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_execution_analysis 
ON dsl_execution_traces (tenant_id, workflow_id, status, created_at DESC);

-- User workflow activity
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_workflow_activity 
ON dsl_execution_traces (user_id, workflow_id, created_at DESC) WHERE user_id IS NOT NULL;

-- Governance compliance tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_governance_compliance 
ON dsl_override_ledger (tenant_id, risk_level, status, created_at DESC);

-- Trust score monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trust_score_monitoring 
ON dsl_execution_traces (tenant_id, trust_score, created_at DESC) WHERE trust_score < 0.8;

-- Plan performance tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plan_performance 
ON strategic_account_plans (tenant_id, status, revenue_growth_target, created_at DESC) WHERE deleted_at IS NULL;

-- ============================================================================
-- PARTIAL INDEXES FOR SPECIFIC CONDITIONS
-- ============================================================================

-- Active workflows only
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_workflows 
ON dsl_workflows (tenant_id, workflow_type, created_at DESC) 
WHERE status = 'active' AND deleted_at IS NULL;

-- Failed executions for monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_failed_executions 
ON dsl_execution_traces (tenant_id, workflow_id, created_at DESC) 
WHERE status = 'failed';

-- High-risk overrides for alerts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_high_risk_overrides 
ON dsl_override_ledger (tenant_id, created_at DESC) 
WHERE risk_level IN ('high', 'critical');

-- Pending approvals for workflow
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pending_workflow_approvals 
ON dsl_override_ledger (tenant_id, workflow_id, created_at DESC) 
WHERE approval_required = true AND approved_at IS NULL;

-- Recent evidence packs for compliance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_evidence_packs 
ON dsl_evidence_packs (tenant_id, pack_type, created_at DESC) 
WHERE created_at >= NOW() - INTERVAL '90 days';

-- Active account plans
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_account_plans 
ON strategic_account_plans (tenant_id, account_id, created_at DESC) 
WHERE status IN ('draft', 'active') AND deleted_at IS NULL;

-- ============================================================================
-- JSONB INDEXES FOR METADATA QUERIES
-- ============================================================================

-- Workflow metadata search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dsl_workflows_metadata 
ON dsl_workflows USING GIN (metadata) WHERE metadata IS NOT NULL;

-- User profile search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_profile 
ON users USING GIN (profile) WHERE profile IS NOT NULL;

-- Policy pack content search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_policy_pack_content 
ON dsl_policy_packs USING GIN (policy_content) WHERE policy_content IS NOT NULL;

-- Evidence data search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_data_search 
ON dsl_evidence_packs USING GIN (evidence_data) WHERE evidence_data IS NOT NULL;

-- Opportunity risk flags
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_opportunity_risk_flags 
ON opportunity_insights USING GIN (risk_flags) WHERE risk_flags IS NOT NULL;

-- ============================================================================
-- EXPRESSION INDEXES FOR COMPUTED VALUES
-- ============================================================================

-- Lowercase email search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_lower 
ON users (tenant_id, LOWER(email)) WHERE deleted_at IS NULL;

-- Full name search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_full_name 
ON users USING GIN (to_tsvector('english', first_name || ' ' || last_name)) 
WHERE deleted_at IS NULL;

-- Workflow name search (case-insensitive)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_workflows_name_lower 
ON dsl_workflows (tenant_id, LOWER(workflow_name)) WHERE deleted_at IS NULL;

-- Plan name search (case-insensitive)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plans_name_lower 
ON strategic_account_plans (tenant_id, LOWER(plan_name)) WHERE deleted_at IS NULL;

-- ============================================================================
-- INDEX MONITORING AND MAINTENANCE
-- ============================================================================

-- Function to analyze index usage
CREATE OR REPLACE FUNCTION analyze_index_usage()
RETURNS TABLE(
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    num_rows BIGINT,
    table_size TEXT,
    index_size TEXT,
    unique_index BOOLEAN,
    number_of_scans BIGINT,
    tuples_read BIGINT,
    tuples_fetched BIGINT,
    usage_ratio NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::TEXT,
        s.tablename::TEXT,
        s.indexname::TEXT,
        pg_class.reltuples::BIGINT as num_rows,
        pg_size_pretty(pg_total_relation_size(s.schemaname||'.'||s.tablename)) as table_size,
        pg_size_pretty(pg_total_relation_size(s.schemaname||'.'||s.indexname)) as index_size,
        i.indisunique as unique_index,
        pg_stat_user_indexes.idx_scan as number_of_scans,
        pg_stat_user_indexes.idx_tup_read as tuples_read,
        pg_stat_user_indexes.idx_tup_fetch as tuples_fetched,
        CASE 
            WHEN pg_stat_user_indexes.idx_scan = 0 THEN 0
            ELSE ROUND((pg_stat_user_indexes.idx_tup_fetch::NUMERIC / pg_stat_user_indexes.idx_tup_read) * 100, 2)
        END as usage_ratio
    FROM pg_stat_user_indexes
    JOIN pg_statio_user_indexes s ON pg_stat_user_indexes.indexrelid = s.indexrelid
    JOIN pg_index i ON pg_stat_user_indexes.indexrelid = i.indexrelid
    JOIN pg_class ON i.indrelid = pg_class.oid
    WHERE s.schemaname = 'public'
    ORDER BY pg_stat_user_indexes.idx_scan DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to find unused indexes
CREATE OR REPLACE FUNCTION find_unused_indexes()
RETURNS TABLE(
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    index_size TEXT,
    scans BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::TEXT,
        s.tablename::TEXT,
        s.indexname::TEXT,
        pg_size_pretty(pg_total_relation_size(s.schemaname||'.'||s.indexname)) as index_size,
        pg_stat_user_indexes.idx_scan as scans
    FROM pg_stat_user_indexes
    JOIN pg_statio_user_indexes s ON pg_stat_user_indexes.indexrelid = s.indexrelid
    WHERE s.schemaname = 'public'
    AND pg_stat_user_indexes.idx_scan < 10  -- Less than 10 scans
    ORDER BY pg_total_relation_size(s.schemaname||'.'||s.indexname) DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get index bloat information
CREATE OR REPLACE FUNCTION get_index_bloat()
RETURNS TABLE(
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    bloat_ratio NUMERIC,
    waste_bytes BIGINT,
    waste_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::TEXT,
        s.tablename::TEXT,
        s.indexname::TEXT,
        CASE 
            WHEN pg_class.relpages = 0 THEN 0
            ELSE ROUND((pg_class.relpages - (pg_class.reltuples / 100))::NUMERIC / pg_class.relpages * 100, 2)
        END as bloat_ratio,
        (pg_class.relpages - (pg_class.reltuples / 100)::INTEGER) * 8192 as waste_bytes,
        pg_size_pretty((pg_class.relpages - (pg_class.reltuples / 100)::INTEGER) * 8192) as waste_size
    FROM pg_statio_user_indexes s
    JOIN pg_class ON s.indexrelid = pg_class.oid
    WHERE s.schemaname = 'public'
    AND pg_class.relpages > 100  -- Only consider larger indexes
    ORDER BY (pg_class.relpages - (pg_class.reltuples / 100)) DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INDEX CREATION VERIFICATION
-- ============================================================================

-- View to check index creation status
CREATE OR REPLACE VIEW index_creation_status AS
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef,
    CASE 
        WHEN indexdef LIKE '%CONCURRENTLY%' THEN 'CONCURRENT'
        ELSE 'STANDARD'
    END as creation_method,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||indexname)) as index_size
FROM pg_indexes 
WHERE schemaname = 'public'
AND indexname LIKE 'idx_%'
ORDER BY tablename, indexname;

-- ============================================================================
-- PERFORMANCE TESTING QUERIES
-- ============================================================================

-- Example queries to test index performance:

-- 1. Tenant workflow lookup (should use idx_dsl_workflows_tenant_status)
/*
EXPLAIN (ANALYZE, BUFFERS) 
SELECT workflow_id, workflow_name, status 
FROM dsl_workflows 
WHERE tenant_id = 1000 AND status = 'active' AND deleted_at IS NULL;
*/

-- 2. Recent execution traces (should use idx_dsl_execution_traces_workflow)
/*
EXPLAIN (ANALYZE, BUFFERS)
SELECT trace_id, status, trust_score, created_at 
FROM dsl_execution_traces 
WHERE workflow_id = 'some-uuid' 
ORDER BY created_at DESC 
LIMIT 50;
*/

-- 3. User activity lookup (should use idx_dsl_execution_traces_user_activity)
/*
EXPLAIN (ANALYZE, BUFFERS)
SELECT workflow_id, status, created_at 
FROM dsl_execution_traces 
WHERE user_id = 1234 
ORDER BY created_at DESC 
LIMIT 100;
*/

-- 4. Pending approvals (should use idx_dsl_override_ledger_pending)
/*
EXPLAIN (ANALYZE, BUFFERS)
SELECT override_id, override_reason, created_at 
FROM dsl_override_ledger 
WHERE tenant_id = 1000 
AND approval_required = true 
AND approved_at IS NULL 
ORDER BY created_at DESC;
*/

-- 5. Account plan search (should use idx_strategic_account_plans_search)
/*
EXPLAIN (ANALYZE, BUFFERS)
SELECT plan_id, plan_name, account_id 
FROM strategic_account_plans 
WHERE to_tsvector('english', plan_name || ' ' || COALESCE(short_term_goals, '')) @@ plainto_tsquery('growth strategy')
AND deleted_at IS NULL;
*/

-- ============================================================================
-- MAINTENANCE RECOMMENDATIONS
-- ============================================================================

/*
MAINTENANCE SCHEDULE:

1. DAILY:
   - Monitor index usage with analyze_index_usage()
   - Check for failed index builds
   - Monitor query performance

2. WEEKLY:
   - Run find_unused_indexes() and consider dropping unused indexes
   - Check index bloat with get_index_bloat()
   - Analyze query plans for new slow queries

3. MONTHLY:
   - REINDEX CONCURRENTLY on high-bloat indexes
   - Review and optimize index strategy based on usage patterns
   - Update statistics with ANALYZE on large tables

4. QUARTERLY:
   - Full index review and optimization
   - Consider new indexes based on query patterns
   - Remove obsolete indexes
*/

COMMIT;
