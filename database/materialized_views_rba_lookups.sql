-- Task 7.2-T29: Create Materialized Views for common RBA lookups
-- Materialized views for low-latency reads in RBA workflows
-- Version: 1.0.0
-- Date: 2024-10-08

-- ============================================================================
-- TENANT SUMMARY MATERIALIZED VIEW
-- ============================================================================

-- Drop existing view if it exists
DROP MATERIALIZED VIEW IF EXISTS mv_tenant_summary CASCADE;

CREATE MATERIALIZED VIEW mv_tenant_summary AS
SELECT 
    tm.tenant_id,
    tm.tenant_name,
    tm.tenant_code,
    tm.industry_code,
    tm.region,
    tm.status as tenant_status,
    tm.subscription_tier,
    tm.compliance_frameworks,
    
    -- User counts
    COALESCE(u.user_count, 0) as total_users,
    COALESCE(u.active_user_count, 0) as active_users,
    COALESCE(u.admin_count, 0) as admin_users,
    
    -- Workflow counts
    COALESCE(w.workflow_count, 0) as total_workflows,
    COALESCE(w.active_workflow_count, 0) as active_workflows,
    
    -- Execution stats
    COALESCE(e.total_executions, 0) as total_executions,
    COALESCE(e.successful_executions, 0) as successful_executions,
    COALESCE(e.failed_executions, 0) as failed_executions,
    COALESCE(e.avg_execution_time_ms, 0) as avg_execution_time_ms,
    
    -- Governance stats
    COALESCE(g.total_overrides, 0) as total_overrides,
    COALESCE(g.evidence_packs, 0) as evidence_packs,
    COALESCE(g.policy_violations, 0) as policy_violations,
    
    -- Timestamps
    tm.created_at as tenant_created_at,
    tm.updated_at as tenant_updated_at,
    NOW() as materialized_at
    
FROM tenant_metadata tm

-- User statistics
LEFT JOIN (
    SELECT 
        tenant_id,
        COUNT(*) as user_count,
        COUNT(CASE WHEN status = 'active' THEN 1 END) as active_user_count,
        COUNT(CASE WHEN profile->>'role' = 'admin' THEN 1 END) as admin_count
    FROM users 
    WHERE deleted_at IS NULL
    GROUP BY tenant_id
) u ON tm.tenant_id = u.tenant_id

-- Workflow statistics
LEFT JOIN (
    SELECT 
        tenant_id,
        COUNT(*) as workflow_count,
        COUNT(CASE WHEN status = 'active' THEN 1 END) as active_workflow_count
    FROM dsl_workflows 
    WHERE deleted_at IS NULL
    GROUP BY tenant_id
) w ON tm.tenant_id = w.tenant_id

-- Execution statistics
LEFT JOIN (
    SELECT 
        tenant_id,
        COUNT(*) as total_executions,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_executions,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_executions,
        AVG(CASE WHEN execution_time_ms > 0 THEN execution_time_ms END) as avg_execution_time_ms
    FROM dsl_execution_traces 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY tenant_id
) e ON tm.tenant_id = e.tenant_id

-- Governance statistics
LEFT JOIN (
    SELECT 
        o.tenant_id,
        COUNT(DISTINCT o.override_id) as total_overrides,
        COUNT(DISTINCT ep.evidence_pack_id) as evidence_packs,
        COUNT(CASE WHEN o.risk_level IN ('high', 'critical') THEN 1 END) as policy_violations
    FROM dsl_override_ledger o
    LEFT JOIN dsl_evidence_packs ep ON o.tenant_id = ep.tenant_id
    WHERE o.created_at >= NOW() - INTERVAL '30 days'
    GROUP BY o.tenant_id
) g ON tm.tenant_id = g.tenant_id

WHERE tm.deleted_at IS NULL;

-- Create indexes on materialized view
CREATE UNIQUE INDEX idx_mv_tenant_summary_tenant_id ON mv_tenant_summary (tenant_id);
CREATE INDEX idx_mv_tenant_summary_industry ON mv_tenant_summary (industry_code);
CREATE INDEX idx_mv_tenant_summary_region ON mv_tenant_summary (region);
CREATE INDEX idx_mv_tenant_summary_tier ON mv_tenant_summary (subscription_tier);

-- ============================================================================
-- WORKFLOW PERFORMANCE MATERIALIZED VIEW
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS mv_workflow_performance CASCADE;

CREATE MATERIALIZED VIEW mv_workflow_performance AS
SELECT 
    w.tenant_id,
    w.workflow_id,
    w.workflow_name,
    w.workflow_type,
    w.automation_type,
    w.industry_overlay,
    w.status as workflow_status,
    
    -- Execution metrics (last 30 days)
    COALESCE(e.execution_count, 0) as execution_count_30d,
    COALESCE(e.success_count, 0) as success_count_30d,
    COALESCE(e.failure_count, 0) as failure_count_30d,
    COALESCE(e.avg_execution_time_ms, 0) as avg_execution_time_ms,
    COALESCE(e.p95_execution_time_ms, 0) as p95_execution_time_ms,
    COALESCE(e.min_execution_time_ms, 0) as min_execution_time_ms,
    COALESCE(e.max_execution_time_ms, 0) as max_execution_time_ms,
    
    -- Success rate
    CASE 
        WHEN COALESCE(e.execution_count, 0) > 0 
        THEN ROUND((COALESCE(e.success_count, 0)::DECIMAL / e.execution_count) * 100, 2)
        ELSE 0 
    END as success_rate_pct,
    
    -- Trust and governance metrics
    COALESCE(t.avg_trust_score, 0) as avg_trust_score,
    COALESCE(t.min_trust_score, 1) as min_trust_score,
    COALESCE(o.override_count, 0) as override_count_30d,
    COALESCE(o.high_risk_overrides, 0) as high_risk_overrides_30d,
    
    -- Last execution info
    e.last_execution_at,
    e.last_execution_status,
    
    -- Timestamps
    w.created_at as workflow_created_at,
    w.updated_at as workflow_updated_at,
    NOW() as materialized_at
    
FROM dsl_workflows w

-- Execution statistics
LEFT JOIN (
    SELECT 
        workflow_id,
        COUNT(*) as execution_count,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as success_count,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failure_count,
        AVG(CASE WHEN execution_time_ms > 0 THEN execution_time_ms END) as avg_execution_time_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time_ms,
        MIN(CASE WHEN execution_time_ms > 0 THEN execution_time_ms END) as min_execution_time_ms,
        MAX(execution_time_ms) as max_execution_time_ms,
        MAX(created_at) as last_execution_at,
        (ARRAY_AGG(status ORDER BY created_at DESC))[1] as last_execution_status
    FROM dsl_execution_traces 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY workflow_id
) e ON w.workflow_id = e.workflow_id

-- Trust score statistics
LEFT JOIN (
    SELECT 
        workflow_id,
        AVG(trust_score) as avg_trust_score,
        MIN(trust_score) as min_trust_score
    FROM dsl_execution_traces 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    AND trust_score IS NOT NULL
    GROUP BY workflow_id
) t ON w.workflow_id = t.workflow_id

-- Override statistics
LEFT JOIN (
    SELECT 
        workflow_id,
        COUNT(*) as override_count,
        COUNT(CASE WHEN risk_level IN ('high', 'critical') THEN 1 END) as high_risk_overrides
    FROM dsl_override_ledger 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY workflow_id
) o ON w.workflow_id = o.workflow_id

WHERE w.deleted_at IS NULL;

-- Create indexes on materialized view
CREATE UNIQUE INDEX idx_mv_workflow_performance_workflow_id ON mv_workflow_performance (workflow_id);
CREATE INDEX idx_mv_workflow_performance_tenant ON mv_workflow_performance (tenant_id);
CREATE INDEX idx_mv_workflow_performance_type ON mv_workflow_performance (workflow_type);
CREATE INDEX idx_mv_workflow_performance_automation ON mv_workflow_performance (automation_type);
CREATE INDEX idx_mv_workflow_performance_success_rate ON mv_workflow_performance (success_rate_pct DESC);
CREATE INDEX idx_mv_workflow_performance_execution_count ON mv_workflow_performance (execution_count_30d DESC);

-- ============================================================================
-- USER ACTIVITY SUMMARY MATERIALIZED VIEW
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS mv_user_activity_summary CASCADE;

CREATE MATERIALIZED VIEW mv_user_activity_summary AS
SELECT 
    u.tenant_id,
    u.user_id,
    u.email,
    u.first_name,
    u.last_name,
    u.status as user_status,
    u.profile->>'role' as user_role,
    u.profile->>'department' as department,
    u.reports_to,
    
    -- Activity metrics (last 30 days)
    COALESCE(w.workflows_created, 0) as workflows_created_30d,
    COALESCE(e.workflows_executed, 0) as workflows_executed_30d,
    COALESCE(e.successful_executions, 0) as successful_executions_30d,
    COALESCE(e.failed_executions, 0) as failed_executions_30d,
    
    -- Success rate
    CASE 
        WHEN COALESCE(e.workflows_executed, 0) > 0 
        THEN ROUND((COALESCE(e.successful_executions, 0)::DECIMAL / e.workflows_executed) * 100, 2)
        ELSE 0 
    END as success_rate_pct,
    
    -- Governance activity
    COALESCE(o.overrides_created, 0) as overrides_created_30d,
    COALESCE(o.overrides_approved, 0) as overrides_approved_30d,
    COALESCE(a.approvals_given, 0) as approvals_given_30d,
    
    -- Planning activity
    COALESCE(p.plans_created, 0) as plans_created_30d,
    COALESCE(p.plans_updated, 0) as plans_updated_30d,
    
    -- Last activity
    GREATEST(
        u.last_login_at,
        e.last_execution_at,
        o.last_override_at,
        p.last_plan_activity_at
    ) as last_activity_at,
    
    -- Timestamps
    u.created_at as user_created_at,
    u.updated_at as user_updated_at,
    NOW() as materialized_at
    
FROM users u

-- Workflow creation activity
LEFT JOIN (
    SELECT 
        created_by,
        COUNT(*) as workflows_created
    FROM dsl_workflows 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    AND deleted_at IS NULL
    GROUP BY created_by
) w ON u.user_id = w.created_by

-- Workflow execution activity
LEFT JOIN (
    SELECT 
        user_id,
        COUNT(*) as workflows_executed,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_executions,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_executions,
        MAX(created_at) as last_execution_at
    FROM dsl_execution_traces 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY user_id
) e ON u.user_id = e.user_id

-- Override activity
LEFT JOIN (
    SELECT 
        user_id,
        COUNT(*) as overrides_created,
        COUNT(CASE WHEN approved_by_user_id = user_id THEN 1 END) as overrides_approved,
        MAX(created_at) as last_override_at
    FROM dsl_override_ledger 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY user_id
    
    UNION ALL
    
    SELECT 
        approved_by_user_id as user_id,
        0 as overrides_created,
        COUNT(*) as overrides_approved,
        MAX(approved_at) as last_override_at
    FROM dsl_override_ledger 
    WHERE approved_at >= NOW() - INTERVAL '30 days'
    AND approved_by_user_id IS NOT NULL
    GROUP BY approved_by_user_id
) o ON u.user_id = o.user_id

-- Approval activity (placeholder - would need approval tables)
LEFT JOIN (
    SELECT 
        user_id,
        0 as approvals_given -- Placeholder
    FROM users 
    WHERE 1=0 -- Disabled until approval tables exist
) a ON u.user_id = a.user_id

-- Planning activity
LEFT JOIN (
    SELECT 
        created_by_user_id as user_id,
        COUNT(CASE WHEN created_at >= NOW() - INTERVAL '30 days' THEN 1 END) as plans_created,
        COUNT(CASE WHEN updated_at >= NOW() - INTERVAL '30 days' AND updated_at > created_at THEN 1 END) as plans_updated,
        MAX(GREATEST(created_at, updated_at)) as last_plan_activity_at
    FROM strategic_account_plans 
    WHERE deleted_at IS NULL
    GROUP BY created_by_user_id
) p ON u.user_id = p.user_id

WHERE u.deleted_at IS NULL;

-- Create indexes on materialized view
CREATE UNIQUE INDEX idx_mv_user_activity_user_id ON mv_user_activity_summary (user_id);
CREATE INDEX idx_mv_user_activity_tenant ON mv_user_activity_summary (tenant_id);
CREATE INDEX idx_mv_user_activity_role ON mv_user_activity_summary (user_role);
CREATE INDEX idx_mv_user_activity_department ON mv_user_activity_summary (department);
CREATE INDEX idx_mv_user_activity_last_activity ON mv_user_activity_summary (last_activity_at DESC);
CREATE INDEX idx_mv_user_activity_success_rate ON mv_user_activity_summary (success_rate_pct DESC);

-- ============================================================================
-- GOVERNANCE COMPLIANCE MATERIALIZED VIEW
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS mv_governance_compliance CASCADE;

CREATE MATERIALIZED VIEW mv_governance_compliance AS
SELECT 
    tm.tenant_id,
    tm.tenant_name,
    tm.industry_code,
    tm.compliance_frameworks,
    
    -- Policy compliance metrics
    COALESCE(p.total_policies, 0) as total_policies,
    COALESCE(p.active_policies, 0) as active_policies,
    
    -- Override metrics (last 30 days)
    COALESCE(o.total_overrides, 0) as total_overrides_30d,
    COALESCE(o.emergency_overrides, 0) as emergency_overrides_30d,
    COALESCE(o.high_risk_overrides, 0) as high_risk_overrides_30d,
    COALESCE(o.pending_approvals, 0) as pending_approvals,
    COALESCE(o.avg_approval_time_hours, 0) as avg_approval_time_hours,
    
    -- Evidence pack metrics
    COALESCE(e.evidence_packs_generated, 0) as evidence_packs_30d,
    COALESCE(e.compliance_evidence_packs, 0) as compliance_evidence_30d,
    COALESCE(e.audit_evidence_packs, 0) as audit_evidence_30d,
    
    -- Trust score metrics
    COALESCE(t.avg_trust_score, 0) as avg_trust_score_30d,
    COALESCE(t.min_trust_score, 1) as min_trust_score_30d,
    COALESCE(t.trust_degradation_events, 0) as trust_degradation_events_30d,
    
    -- Compliance violations
    COALESCE(v.policy_violations, 0) as policy_violations_30d,
    COALESCE(v.sla_breaches, 0) as sla_breaches_30d,
    COALESCE(v.data_residency_violations, 0) as data_residency_violations_30d,
    
    -- Compliance score calculation
    CASE 
        WHEN COALESCE(o.total_overrides, 0) + COALESCE(v.policy_violations, 0) = 0 THEN 100
        ELSE GREATEST(0, 100 - (
            (COALESCE(o.high_risk_overrides, 0) * 10) +
            (COALESCE(v.policy_violations, 0) * 5) +
            (COALESCE(v.sla_breaches, 0) * 3)
        ))
    END as compliance_score,
    
    -- Risk level assessment
    CASE 
        WHEN COALESCE(o.high_risk_overrides, 0) > 5 OR COALESCE(v.policy_violations, 0) > 10 THEN 'HIGH'
        WHEN COALESCE(o.high_risk_overrides, 0) > 2 OR COALESCE(v.policy_violations, 0) > 5 THEN 'MEDIUM'
        ELSE 'LOW'
    END as risk_level,
    
    NOW() as materialized_at
    
FROM tenant_metadata tm

-- Policy statistics
LEFT JOIN (
    SELECT 
        tenant_id,
        COUNT(*) as total_policies,
        COUNT(CASE WHEN status = 'active' THEN 1 END) as active_policies
    FROM dsl_policy_packs 
    WHERE deleted_at IS NULL
    GROUP BY tenant_id
) p ON tm.tenant_id = p.tenant_id

-- Override statistics
LEFT JOIN (
    SELECT 
        tenant_id,
        COUNT(*) as total_overrides,
        COUNT(CASE WHEN override_type = 'emergency' THEN 1 END) as emergency_overrides,
        COUNT(CASE WHEN risk_level IN ('high', 'critical') THEN 1 END) as high_risk_overrides,
        COUNT(CASE WHEN approval_required = true AND approved_at IS NULL THEN 1 END) as pending_approvals,
        AVG(CASE 
            WHEN approved_at IS NOT NULL AND created_at IS NOT NULL 
            THEN EXTRACT(EPOCH FROM (approved_at - created_at)) / 3600 
        END) as avg_approval_time_hours
    FROM dsl_override_ledger 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY tenant_id
) o ON tm.tenant_id = o.tenant_id

-- Evidence pack statistics
LEFT JOIN (
    SELECT 
        tenant_id,
        COUNT(*) as evidence_packs_generated,
        COUNT(CASE WHEN pack_type = 'compliance_evidence' THEN 1 END) as compliance_evidence_packs,
        COUNT(CASE WHEN pack_type = 'audit_evidence' THEN 1 END) as audit_evidence_packs
    FROM dsl_evidence_packs 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY tenant_id
) e ON tm.tenant_id = e.tenant_id

-- Trust score statistics
LEFT JOIN (
    SELECT 
        tenant_id,
        AVG(trust_score) as avg_trust_score,
        MIN(trust_score) as min_trust_score,
        COUNT(CASE WHEN trust_score < 0.7 THEN 1 END) as trust_degradation_events
    FROM dsl_execution_traces 
    WHERE created_at >= NOW() - INTERVAL '30 days'
    AND trust_score IS NOT NULL
    GROUP BY tenant_id
) t ON tm.tenant_id = t.tenant_id

-- Violation statistics (placeholder - would need violation tracking)
LEFT JOIN (
    SELECT 
        tenant_id,
        0 as policy_violations, -- Placeholder
        0 as sla_breaches, -- Placeholder
        0 as data_residency_violations -- Placeholder
    FROM tenant_metadata 
    WHERE 1=0 -- Disabled until violation tables exist
) v ON tm.tenant_id = v.tenant_id

WHERE tm.deleted_at IS NULL;

-- Create indexes on materialized view
CREATE UNIQUE INDEX idx_mv_governance_compliance_tenant_id ON mv_governance_compliance (tenant_id);
CREATE INDEX idx_mv_governance_compliance_industry ON mv_governance_compliance (industry_code);
CREATE INDEX idx_mv_governance_compliance_score ON mv_governance_compliance (compliance_score DESC);
CREATE INDEX idx_mv_governance_compliance_risk ON mv_governance_compliance (risk_level);

-- ============================================================================
-- REFRESH FUNCTIONS AND SCHEDULING
-- ============================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_rba_materialized_views()
RETURNS VOID AS $$
BEGIN
    -- Refresh in dependency order
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_tenant_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_workflow_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_activity_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_governance_compliance;
    
    -- Log refresh completion
    INSERT INTO materialized_view_refresh_log (
        view_name, 
        refresh_started_at, 
        refresh_completed_at, 
        status
    ) VALUES 
    ('mv_tenant_summary', NOW(), NOW(), 'success'),
    ('mv_workflow_performance', NOW(), NOW(), 'success'),
    ('mv_user_activity_summary', NOW(), NOW(), 'success'),
    ('mv_governance_compliance', NOW(), NOW(), 'success');
    
    RAISE NOTICE 'All RBA materialized views refreshed successfully';
END;
$$ LANGUAGE plpgsql;

-- Create log table for materialized view refreshes
CREATE TABLE IF NOT EXISTS materialized_view_refresh_log (
    log_id SERIAL PRIMARY KEY,
    view_name VARCHAR(100) NOT NULL,
    refresh_started_at TIMESTAMPTZ NOT NULL,
    refresh_completed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'in_progress',
    error_message TEXT,
    rows_affected INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Function to get materialized view statistics
CREATE OR REPLACE FUNCTION get_materialized_view_stats()
RETURNS TABLE(
    view_name TEXT,
    row_count BIGINT,
    size_bytes BIGINT,
    last_refreshed TIMESTAMPTZ,
    refresh_duration INTERVAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname || '.' || matviewname as view_name,
        n_tup_ins as row_count,
        pg_total_relation_size(schemaname||'.'||matviewname) as size_bytes,
        (SELECT MAX(refresh_completed_at) 
         FROM materialized_view_refresh_log 
         WHERE view_name = matviewname) as last_refreshed,
        (SELECT MAX(refresh_completed_at - refresh_started_at) 
         FROM materialized_view_refresh_log 
         WHERE view_name = matviewname) as refresh_duration
    FROM pg_stat_user_tables 
    WHERE schemaname = 'public' 
    AND relname LIKE 'mv_%';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- USAGE EXAMPLES AND DOCUMENTATION
-- ============================================================================

-- Example queries using the materialized views:

-- 1. Get tenant performance overview
/*
SELECT 
    tenant_name,
    industry_code,
    total_users,
    active_workflows,
    total_executions,
    ROUND(successful_executions::DECIMAL / NULLIF(total_executions, 0) * 100, 2) as success_rate_pct
FROM mv_tenant_summary 
WHERE industry_code = 'SaaS'
ORDER BY total_executions DESC;
*/

-- 2. Find poorly performing workflows
/*
SELECT 
    workflow_name,
    automation_type,
    execution_count_30d,
    success_rate_pct,
    avg_execution_time_ms,
    override_count_30d
FROM mv_workflow_performance 
WHERE success_rate_pct < 95 
OR avg_execution_time_ms > 5000
ORDER BY success_rate_pct ASC, avg_execution_time_ms DESC;
*/

-- 3. Get user activity insights
/*
SELECT 
    email,
    user_role,
    workflows_executed_30d,
    success_rate_pct,
    overrides_created_30d,
    last_activity_at
FROM mv_user_activity_summary 
WHERE last_activity_at >= NOW() - INTERVAL '7 days'
ORDER BY workflows_executed_30d DESC;
*/

-- 4. Compliance dashboard data
/*
SELECT 
    tenant_name,
    industry_code,
    compliance_score,
    risk_level,
    total_overrides_30d,
    high_risk_overrides_30d,
    evidence_packs_30d
FROM mv_governance_compliance 
WHERE compliance_score < 90
ORDER BY compliance_score ASC;
*/

-- ============================================================================
-- MAINTENANCE AND MONITORING
-- ============================================================================

-- View to monitor materialized view freshness
CREATE OR REPLACE VIEW mv_freshness_monitor AS
SELECT 
    view_name,
    row_count,
    pg_size_pretty(size_bytes) as size,
    last_refreshed,
    CASE 
        WHEN last_refreshed IS NULL THEN 'NEVER REFRESHED'
        WHEN last_refreshed < NOW() - INTERVAL '1 hour' THEN 'STALE'
        ELSE 'FRESH'
    END as freshness_status,
    refresh_duration
FROM get_materialized_view_stats();

-- Initial refresh of all materialized views
SELECT refresh_rba_materialized_views();

-- Check materialized view status
SELECT * FROM mv_freshness_monitor;

COMMIT;
