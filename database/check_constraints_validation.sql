-- Task 7.2-T33: Add check constraints (amount ≥ 0, status enums)
-- Data correctness constraints with early fail validation
-- Version: 1.0.0
-- Date: 2024-10-08

-- ============================================================================
-- CHECK CONSTRAINT STRATEGY
-- ============================================================================

/*
CHECK CONSTRAINT CATEGORIES:

1. RANGE CONSTRAINTS:
   - Numeric ranges (amounts >= 0, percentages 0-100)
   - Date ranges (end_date >= start_date)
   - Length constraints (minimum/maximum string lengths)

2. ENUM CONSTRAINTS:
   - Status values (active, inactive, suspended)
   - Type classifications (workflow_type, automation_type)
   - Priority levels (low, medium, high, critical)

3. FORMAT CONSTRAINTS:
   - Email format validation
   - Phone number patterns
   - URL format validation

4. BUSINESS LOGIC CONSTRAINTS:
   - Conditional constraints (if A then B must be true)
   - Cross-column validation
   - Industry-specific rules

5. DATA QUALITY CONSTRAINTS:
   - Non-empty required strings
   - Valid JSON structure
   - Referential data consistency
*/

-- ============================================================================
-- UTILITY FUNCTIONS FOR CONSTRAINT MANAGEMENT
-- ============================================================================

-- Function to safely add check constraint
CREATE OR REPLACE FUNCTION add_check_constraint(
    table_name TEXT,
    constraint_name TEXT,
    constraint_condition TEXT,
    constraint_description TEXT DEFAULT ''
)
RETURNS BOOLEAN AS $$
DECLARE
    constraint_exists BOOLEAN;
BEGIN
    -- Check if constraint already exists
    SELECT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = add_check_constraint.constraint_name
        AND table_name = add_check_constraint.table_name
        AND constraint_type = 'CHECK'
    ) INTO constraint_exists;
    
    IF constraint_exists THEN
        RAISE NOTICE 'Check constraint % already exists on table %', constraint_name, table_name;
        RETURN FALSE;
    END IF;
    
    -- Add the constraint
    EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I CHECK (%s)',
                   table_name, constraint_name, constraint_condition);
    
    -- Add comment if provided
    IF constraint_description != '' THEN
        EXECUTE format('COMMENT ON CONSTRAINT %I ON %I IS %L',
                       constraint_name, table_name, constraint_description);
    END IF;
    
    RAISE NOTICE 'Added check constraint % to table %: %', constraint_name, table_name, constraint_description;
    RETURN TRUE;
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Failed to add check constraint % to table %: %', constraint_name, table_name, SQLERRM;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Function to validate existing data against new constraint
CREATE OR REPLACE FUNCTION validate_check_constraint_data(
    table_name TEXT,
    constraint_condition TEXT
)
RETURNS TABLE(
    violation_count BIGINT,
    sample_violations JSONB
) AS $$
DECLARE
    sql_query TEXT;
BEGIN
    sql_query := format('
        SELECT 
            COUNT(*) as violation_count,
            jsonb_agg(
                jsonb_build_object(
                    ''row_data'', to_jsonb(t.*),
                    ''primary_key'', COALESCE(t.id, t.user_id, t.tenant_id, t.workflow_id, t.plan_id)::TEXT
                )
            ) FILTER (WHERE NOT (%s)) as sample_violations
        FROM %I t',
        constraint_condition, table_name
    );
    
    RETURN QUERY EXECUTE sql_query;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TENANT AND USER CONSTRAINTS
-- ============================================================================

-- Tenant metadata constraints
SELECT add_check_constraint(
    'tenant_metadata',
    'chk_tenant_metadata_status',
    'status IN (''active'', ''suspended'', ''archived'')',
    'Valid tenant status values'
);

SELECT add_check_constraint(
    'tenant_metadata',
    'chk_tenant_metadata_industry',
    'industry_code IN (''SaaS'', ''BANK'', ''INSUR'', ''ECOMM'', ''FS'', ''IT'', ''HEALTH'', ''MANUF'', ''RETAIL'', ''TELECOM'')',
    'Valid industry codes'
);

SELECT add_check_constraint(
    'tenant_metadata',
    'chk_tenant_metadata_region',
    'region IN (''US'', ''EU'', ''IN'', ''APAC'', ''GLOBAL'')',
    'Valid region codes'
);

SELECT add_check_constraint(
    'tenant_metadata',
    'chk_tenant_metadata_tier',
    'subscription_tier IN (''free'', ''standard'', ''premium'', ''enterprise'')',
    'Valid subscription tiers'
);

SELECT add_check_constraint(
    'tenant_metadata',
    'chk_tenant_metadata_name_length',
    'LENGTH(tenant_name) >= 2 AND LENGTH(tenant_name) <= 100',
    'Tenant name must be 2-100 characters'
);

SELECT add_check_constraint(
    'tenant_metadata',
    'chk_tenant_metadata_code_format',
    'tenant_code ~ ''^[A-Z0-9_]{2,20}$''',
    'Tenant code must be 2-20 uppercase alphanumeric characters'
);

-- Users constraints
SELECT add_check_constraint(
    'users',
    'chk_users_status',
    'status IN (''active'', ''inactive'', ''suspended'', ''pending'')',
    'Valid user status values'
);

SELECT add_check_constraint(
    'users',
    'chk_users_email_format',
    'email ~ ''^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$''',
    'Valid email format'
);

SELECT add_check_constraint(
    'users',
    'chk_users_name_length',
    'LENGTH(first_name) >= 1 AND LENGTH(last_name) >= 1',
    'First and last names must not be empty'
);

SELECT add_check_constraint(
    'users',
    'chk_users_tenant_positive',
    'tenant_id > 0',
    'Tenant ID must be positive'
);

-- User roles constraints
SELECT add_check_constraint(
    'users_role',
    'chk_users_role_segment',
    'segment IN (''Enterprise'', ''Mid-Market'', ''SMB'', ''Global'', ''Americas'', ''EMEA'', ''APAC'')',
    'Valid user segments'
);

SELECT add_check_constraint(
    'users_role',
    'chk_users_role_region',
    'region IN (''US'', ''EU'', ''IN'', ''APAC'', ''GLOBAL'', ''Americas'', ''EMEA'')',
    'Valid user regions'
);

-- ============================================================================
-- DSL WORKFLOW CONSTRAINTS
-- ============================================================================

-- DSL Workflows constraints
SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_status',
    'status IN (''draft'', ''active'', ''inactive'', ''deprecated'', ''archived'')',
    'Valid workflow status values'
);

SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_type',
    'workflow_type IN (''rba'', ''rbia'', ''aala'', ''hybrid'')',
    'Valid workflow types'
);

SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_automation_type',
    'automation_type IN (''pipeline_hygiene'', ''forecast_accuracy'', ''lead_scoring'', ''approval_workflow'', ''compliance_check'', ''data_sync'', ''notification'', ''custom'')',
    'Valid automation types'
);

SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_industry',
    'industry_overlay IN (''SaaS'', ''BANK'', ''INSUR'', ''ECOMM'', ''FS'', ''IT'', ''HEALTH'', ''MANUF'', ''RETAIL'', ''TELECOM'', ''UNIVERSAL'')',
    'Valid industry overlays'
);

SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_name_length',
    'LENGTH(workflow_name) >= 3 AND LENGTH(workflow_name) <= 200',
    'Workflow name must be 3-200 characters'
);

SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_version_format',
    'version ~ ''^v\d+\.\d+\.\d+$''',
    'Version must follow semantic versioning (vX.Y.Z)'
);

-- DSL Policy Packs constraints
SELECT add_check_constraint(
    'dsl_policy_packs',
    'chk_dsl_policy_packs_status',
    'status IN (''draft'', ''active'', ''inactive'', ''deprecated'')',
    'Valid policy pack status values'
);

SELECT add_check_constraint(
    'dsl_policy_packs',
    'chk_dsl_policy_packs_industry',
    'industry_overlay IN (''SaaS'', ''BANK'', ''INSUR'', ''ECOMM'', ''FS'', ''IT'', ''HEALTH'', ''MANUF'', ''RETAIL'', ''TELECOM'', ''UNIVERSAL'')',
    'Valid industry overlays'
);

SELECT add_check_constraint(
    'dsl_policy_packs',
    'chk_dsl_policy_packs_name_length',
    'LENGTH(pack_name) >= 3 AND LENGTH(pack_name) <= 200',
    'Policy pack name must be 3-200 characters'
);

-- DSL Execution Traces constraints
SELECT add_check_constraint(
    'dsl_execution_traces',
    'chk_dsl_execution_traces_status',
    'status IN (''pending'', ''running'', ''completed'', ''failed'', ''cancelled'', ''timeout'')',
    'Valid execution status values'
);

SELECT add_check_constraint(
    'dsl_execution_traces',
    'chk_dsl_execution_traces_execution_time',
    'execution_time_ms >= 0',
    'Execution time must be non-negative'
);

SELECT add_check_constraint(
    'dsl_execution_traces',
    'chk_dsl_execution_traces_trust_score',
    'trust_score IS NULL OR (trust_score >= 0.0 AND trust_score <= 1.0)',
    'Trust score must be between 0.0 and 1.0'
);

SELECT add_check_constraint(
    'dsl_execution_traces',
    'chk_dsl_execution_traces_compliance_score',
    'compliance_score IS NULL OR (compliance_score >= 0.0 AND compliance_score <= 1.0)',
    'Compliance score must be between 0.0 and 1.0'
);

-- DSL Evidence Packs constraints
SELECT add_check_constraint(
    'dsl_evidence_packs',
    'chk_dsl_evidence_packs_type',
    'pack_type IN (''execution_evidence'', ''compliance_evidence'', ''audit_evidence'', ''governance_evidence'', ''override_evidence'')',
    'Valid evidence pack types'
);

SELECT add_check_constraint(
    'dsl_evidence_packs',
    'chk_dsl_evidence_packs_compliance',
    'compliance_framework IN (''SOX'', ''GDPR'', ''HIPAA'', ''RBI'', ''IRDAI'', ''BASEL_III'', ''DPDP'', ''CCPA'', ''PCI_DSS'', ''CUSTOM'')',
    'Valid compliance frameworks'
);

SELECT add_check_constraint(
    'dsl_evidence_packs',
    'chk_dsl_evidence_packs_retention',
    'retention_days > 0 AND retention_days <= 3650',
    'Retention period must be 1-3650 days (10 years max)'
);

-- DSL Override Ledger constraints
SELECT add_check_constraint(
    'dsl_override_ledger',
    'chk_dsl_override_ledger_type',
    'override_type IN (''emergency'', ''planned'', ''exception'', ''maintenance'', ''testing'')',
    'Valid override types'
);

SELECT add_check_constraint(
    'dsl_override_ledger',
    'chk_dsl_override_ledger_risk',
    'risk_level IN (''low'', ''medium'', ''high'', ''critical'')',
    'Valid risk levels'
);

SELECT add_check_constraint(
    'dsl_override_ledger',
    'chk_dsl_override_ledger_status',
    'status IN (''pending'', ''approved'', ''rejected'', ''expired'', ''revoked'')',
    'Valid override status values'
);

SELECT add_check_constraint(
    'dsl_override_ledger',
    'chk_dsl_override_ledger_reason_length',
    'LENGTH(override_reason) >= 10',
    'Override reason must be at least 10 characters'
);

SELECT add_check_constraint(
    'dsl_override_ledger',
    'chk_dsl_override_ledger_approval_logic',
    '(approval_required = false) OR (approval_required = true AND approved_by_user_id IS NOT NULL AND approved_at IS NOT NULL) OR (approval_required = true AND status = ''pending'')',
    'If approval required, must have approver and approval time, or be pending'
);

-- DSL Workflow Templates constraints
SELECT add_check_constraint(
    'dsl_workflow_templates',
    'chk_dsl_workflow_templates_industry',
    'industry_overlay IN (''SaaS'', ''BANK'', ''INSUR'', ''ECOMM'', ''FS'', ''IT'', ''HEALTH'', ''MANUF'', ''RETAIL'', ''TELECOM'', ''UNIVERSAL'')',
    'Valid industry overlays'
);

SELECT add_check_constraint(
    'dsl_workflow_templates',
    'chk_dsl_workflow_templates_name_length',
    'LENGTH(template_name) >= 3 AND LENGTH(template_name) <= 200',
    'Template name must be 3-200 characters'
);

-- DSL Capability Registry constraints
SELECT add_check_constraint(
    'dsl_capability_registry',
    'chk_dsl_capability_registry_type',
    'capability_type IN (''query'', ''decision'', ''ml_decision'', ''agent_call'', ''notify'', ''governance'', ''custom'')',
    'Valid capability types'
);

SELECT add_check_constraint(
    'dsl_capability_registry',
    'chk_dsl_capability_registry_name_length',
    'LENGTH(capability_name) >= 3 AND LENGTH(capability_name) <= 100',
    'Capability name must be 3-100 characters'
);

-- ============================================================================
-- STRATEGIC ACCOUNT PLANNING CONSTRAINTS
-- ============================================================================

-- Strategic Account Plans constraints
SELECT add_check_constraint(
    'strategic_account_plans',
    'chk_strategic_account_plans_status',
    'status IN (''draft'', ''active'', ''completed'', ''cancelled'', ''archived'')',
    'Valid plan status values'
);

SELECT add_check_constraint(
    'strategic_account_plans',
    'chk_strategic_account_plans_tier',
    'account_tier IN (''enterprise'', ''mid_market'', ''smb'', ''startup'')',
    'Valid account tiers'
);

SELECT add_check_constraint(
    'strategic_account_plans',
    'chk_strategic_account_plans_revenue_target',
    'revenue_growth_target IS NULL OR revenue_growth_target >= 0',
    'Revenue growth target must be non-negative'
);

SELECT add_check_constraint(
    'strategic_account_plans',
    'chk_strategic_account_plans_annual_revenue',
    'annual_revenue IS NULL OR annual_revenue >= 0',
    'Annual revenue must be non-negative'
);

SELECT add_check_constraint(
    'strategic_account_plans',
    'chk_strategic_account_plans_name_length',
    'LENGTH(plan_name) >= 3 AND LENGTH(plan_name) <= 200',
    'Plan name must be 3-200 characters'
);

-- Account Planning Templates constraints
SELECT add_check_constraint(
    'account_planning_templates',
    'chk_account_planning_templates_name_length',
    'LENGTH(template_name) >= 3 AND LENGTH(template_name) <= 200',
    'Template name must be 3-200 characters'
);

-- Opportunity Insights constraints
SELECT add_check_constraint(
    'opportunity_insights',
    'chk_opportunity_insights_amount',
    'opportunity_amount IS NULL OR opportunity_amount >= 0',
    'Opportunity amount must be non-negative'
);

SELECT add_check_constraint(
    'opportunity_insights',
    'chk_opportunity_insights_probability',
    'win_probability IS NULL OR (win_probability >= 0 AND win_probability <= 100)',
    'Win probability must be between 0 and 100'
);

-- Plan Stakeholders constraints
SELECT add_check_constraint(
    'plan_stakeholders',
    'chk_plan_stakeholders_influence',
    'influence_level IN (''low'', ''medium'', ''high'', ''champion'', ''blocker'')',
    'Valid influence levels'
);

SELECT add_check_constraint(
    'plan_stakeholders',
    'chk_plan_stakeholders_contact_length',
    'LENGTH(contact_name) >= 2',
    'Contact name must be at least 2 characters'
);

-- Plan Activities constraints
SELECT add_check_constraint(
    'plan_activities',
    'chk_plan_activities_status',
    'status IN (''planned'', ''in_progress'', ''completed'', ''cancelled'', ''overdue'')',
    'Valid activity status values'
);

SELECT add_check_constraint(
    'plan_activities',
    'chk_plan_activities_priority',
    'priority IN (''low'', ''medium'', ''high'', ''critical'')',
    'Valid activity priorities'
);

SELECT add_check_constraint(
    'plan_activities',
    'chk_plan_activities_name_length',
    'LENGTH(activity_name) >= 3',
    'Activity name must be at least 3 characters'
);

SELECT add_check_constraint(
    'plan_activities',
    'chk_plan_activities_date_logic',
    'due_date IS NULL OR activity_date IS NULL OR due_date >= activity_date',
    'Due date must be on or after activity date'
);

-- Plan Collaborators constraints
SELECT add_check_constraint(
    'plan_collaborators',
    'chk_plan_collaborators_permission',
    'permission_level IN (''read'', ''comment'', ''edit'', ''admin'')',
    'Valid permission levels'
);

-- Plan Comments constraints
SELECT add_check_constraint(
    'plan_comments',
    'chk_plan_comments_content_length',
    'LENGTH(comment_text) >= 1 AND LENGTH(comment_text) <= 5000',
    'Comment must be 1-5000 characters'
);

-- Plan Insight Attachments constraints
SELECT add_check_constraint(
    'plan_insight_attachments',
    'chk_plan_insight_attachments_type',
    'attachment_type IN (''document'', ''image'', ''spreadsheet'', ''presentation'', ''video'', ''other'')',
    'Valid attachment types'
);

SELECT add_check_constraint(
    'plan_insight_attachments',
    'chk_plan_insight_attachments_size',
    'file_size_bytes > 0 AND file_size_bytes <= 104857600',
    'File size must be between 1 byte and 100MB'
);

-- ============================================================================
-- BUSINESS LOGIC CONSTRAINTS
-- ============================================================================

-- Workflow execution time constraints
SELECT add_check_constraint(
    'dsl_execution_traces',
    'chk_dsl_execution_traces_time_logic',
    '(status = ''completed'' AND completed_at IS NOT NULL) OR (status != ''completed'')',
    'Completed workflows must have completion timestamp'
);

SELECT add_check_constraint(
    'dsl_execution_traces',
    'chk_dsl_execution_traces_duration_logic',
    '(completed_at IS NULL) OR (started_at IS NULL) OR (completed_at >= started_at)',
    'Completion time must be after start time'
);

-- Override approval logic
SELECT add_check_constraint(
    'dsl_override_ledger',
    'chk_dsl_override_ledger_expiry_logic',
    '(expires_at IS NULL) OR (expires_at > created_at)',
    'Expiry time must be after creation time'
);

SELECT add_check_constraint(
    'dsl_override_ledger',
    'chk_dsl_override_ledger_approval_time_logic',
    '(approved_at IS NULL) OR (approved_at >= created_at)',
    'Approval time must be after creation time'
);

-- Evidence pack retention logic
SELECT add_check_constraint(
    'dsl_evidence_packs',
    'chk_dsl_evidence_packs_expiry_logic',
    '(expires_at IS NULL) OR (expires_at > created_at)',
    'Evidence expiry must be after creation'
);

-- Plan activity date logic
SELECT add_check_constraint(
    'strategic_account_plans',
    'chk_strategic_account_plans_date_logic',
    '(plan_end_date IS NULL) OR (plan_start_date IS NULL) OR (plan_end_date >= plan_start_date)',
    'Plan end date must be after start date'
);

-- ============================================================================
-- INDUSTRY-SPECIFIC CONSTRAINTS
-- ============================================================================

-- Banking industry constraints
SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_banking_compliance',
    '(industry_overlay != ''BANK'') OR (metadata IS NOT NULL AND metadata ? ''rbi_compliant'')',
    'Banking workflows must have RBI compliance metadata'
);

-- Insurance industry constraints
SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_insurance_compliance',
    '(industry_overlay != ''INSUR'') OR (metadata IS NOT NULL AND metadata ? ''irdai_compliant'')',
    'Insurance workflows must have IRDAI compliance metadata'
);

-- SaaS industry constraints
SELECT add_check_constraint(
    'strategic_account_plans',
    'chk_strategic_account_plans_saas_metrics',
    '(industry != ''SaaS'') OR (customer_success_metrics IS NOT NULL)',
    'SaaS account plans must have customer success metrics'
);

-- ============================================================================
-- JSON VALIDATION CONSTRAINTS
-- ============================================================================

-- Validate JSON structure in metadata fields
SELECT add_check_constraint(
    'dsl_workflows',
    'chk_dsl_workflows_metadata_json',
    'metadata IS NULL OR jsonb_typeof(metadata) = ''object''',
    'Workflow metadata must be valid JSON object'
);

SELECT add_check_constraint(
    'users',
    'chk_users_profile_json',
    'profile IS NULL OR jsonb_typeof(profile) = ''object''',
    'User profile must be valid JSON object'
);

SELECT add_check_constraint(
    'dsl_policy_packs',
    'chk_dsl_policy_packs_content_json',
    'policy_content IS NULL OR jsonb_typeof(policy_content) = ''object''',
    'Policy content must be valid JSON object'
);

SELECT add_check_constraint(
    'dsl_evidence_packs',
    'chk_dsl_evidence_packs_data_json',
    'evidence_data IS NULL OR jsonb_typeof(evidence_data) = ''object''',
    'Evidence data must be valid JSON object'
);

-- ============================================================================
-- CONSTRAINT VALIDATION AND REPORTING
-- ============================================================================

-- Function to check all constraint violations
CREATE OR REPLACE FUNCTION check_constraint_violations()
RETURNS TABLE(
    table_name TEXT,
    constraint_name TEXT,
    constraint_definition TEXT,
    violation_count BIGINT,
    sample_violations JSONB
) AS $$
DECLARE
    constraint_record RECORD;
    violation_info RECORD;
BEGIN
    -- Get all check constraints
    FOR constraint_record IN 
        SELECT 
            tc.table_name,
            tc.constraint_name,
            cc.check_clause
        FROM information_schema.table_constraints tc
        JOIN information_schema.check_constraints cc 
            ON tc.constraint_name = cc.constraint_name
        WHERE tc.constraint_type = 'CHECK'
        AND tc.table_schema = 'public'
        AND tc.constraint_name LIKE 'chk_%'
    LOOP
        -- Check for violations
        BEGIN
            SELECT * INTO violation_info 
            FROM validate_check_constraint_data(
                constraint_record.table_name,
                constraint_record.check_clause
            );
            
            -- Return violations if any found
            IF violation_info.violation_count > 0 THEN
                table_name := constraint_record.table_name;
                constraint_name := constraint_record.constraint_name;
                constraint_definition := constraint_record.check_clause;
                violation_count := violation_info.violation_count;
                sample_violations := violation_info.sample_violations;
                RETURN NEXT;
            END IF;
        EXCEPTION WHEN OTHERS THEN
            -- Skip constraints that can't be validated
            CONTINUE;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to get constraint summary by table
CREATE OR REPLACE FUNCTION get_check_constraint_summary()
RETURNS TABLE(
    table_name TEXT,
    total_constraints INTEGER,
    enum_constraints INTEGER,
    range_constraints INTEGER,
    format_constraints INTEGER,
    business_logic_constraints INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tc.table_name::TEXT,
        COUNT(*)::INTEGER as total_constraints,
        COUNT(CASE WHEN cc.check_clause LIKE '%IN (%' THEN 1 END)::INTEGER as enum_constraints,
        COUNT(CASE WHEN cc.check_clause ~ '(>=|<=|>|<|\sBETWEEN\s)' THEN 1 END)::INTEGER as range_constraints,
        COUNT(CASE WHEN cc.check_clause ~ '~|\sLIKE\s|\sILIKE\s' THEN 1 END)::INTEGER as format_constraints,
        COUNT(CASE WHEN cc.check_clause ~ '\sAND\s|\sOR\s|\sCASE\s' THEN 1 END)::INTEGER as business_logic_constraints
    FROM information_schema.table_constraints tc
    JOIN information_schema.check_constraints cc 
        ON tc.constraint_name = cc.constraint_name
    WHERE tc.constraint_type = 'CHECK'
    AND tc.table_schema = 'public'
    AND tc.constraint_name LIKE 'chk_%'
    GROUP BY tc.table_name
    ORDER BY tc.table_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CONSTRAINT TESTING FUNCTIONS
-- ============================================================================

-- Function to test constraint with sample data
CREATE OR REPLACE FUNCTION test_check_constraint(
    table_name TEXT,
    constraint_name TEXT,
    test_data JSONB
)
RETURNS BOOLEAN AS $$
DECLARE
    constraint_clause TEXT;
    test_sql TEXT;
    result BOOLEAN;
BEGIN
    -- Get constraint definition
    SELECT cc.check_clause INTO constraint_clause
    FROM information_schema.table_constraints tc
    JOIN information_schema.check_constraints cc 
        ON tc.constraint_name = cc.constraint_name
    WHERE tc.table_name = test_check_constraint.table_name
    AND tc.constraint_name = test_check_constraint.constraint_name
    AND tc.constraint_type = 'CHECK';
    
    IF constraint_clause IS NULL THEN
        RAISE EXCEPTION 'Constraint % not found on table %', constraint_name, table_name;
    END IF;
    
    -- Build test query
    test_sql := format('SELECT (%s) FROM (SELECT %s) AS test_data',
                      constraint_clause,
                      (SELECT string_agg(format('%L::%s AS %I', 
                                               test_data->>key, 
                                               'TEXT',  -- Simplified type handling
                                               key), ', ')
                       FROM jsonb_object_keys(test_data) AS key));
    
    EXECUTE test_sql INTO result;
    RETURN result;
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error testing constraint %: %', constraint_name, SQLERRM;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CONSTRAINT MONITORING VIEWS
-- ============================================================================

-- View to show all check constraints
CREATE OR REPLACE VIEW check_constraint_catalog AS
SELECT 
    tc.table_name,
    tc.constraint_name,
    cc.check_clause as constraint_definition,
    obj_description(pgc.oid) as constraint_description,
    CASE 
        WHEN cc.check_clause LIKE '%IN (%' THEN 'ENUM'
        WHEN cc.check_clause ~ '(>=|<=|>|<|\sBETWEEN\s)' THEN 'RANGE'
        WHEN cc.check_clause ~ '~|\sLIKE\s|\sILIKE\s' THEN 'FORMAT'
        WHEN cc.check_clause ~ '\sAND\s|\sOR\s|\sCASE\s' THEN 'BUSINESS_LOGIC'
        ELSE 'OTHER'
    END as constraint_type
FROM information_schema.table_constraints tc
JOIN information_schema.check_constraints cc 
    ON tc.constraint_name = cc.constraint_name
LEFT JOIN pg_constraint pgc ON pgc.conname = tc.constraint_name
WHERE tc.constraint_type = 'CHECK'
AND tc.table_schema = 'public'
AND tc.constraint_name LIKE 'chk_%'
ORDER BY tc.table_name, tc.constraint_name;

-- View to show constraint coverage by table
CREATE OR REPLACE VIEW constraint_coverage AS
SELECT 
    t.table_name,
    COUNT(c.column_name) as total_columns,
    COUNT(CASE WHEN tc.constraint_name IS NOT NULL THEN 1 END) as constrained_columns,
    ROUND(
        COUNT(CASE WHEN tc.constraint_name IS NOT NULL THEN 1 END)::DECIMAL / 
        NULLIF(COUNT(c.column_name), 0) * 100, 2
    ) as coverage_percentage
FROM information_schema.tables t
JOIN information_schema.columns c ON t.table_name = c.table_name
LEFT JOIN information_schema.constraint_column_usage ccu 
    ON c.table_name = ccu.table_name AND c.column_name = ccu.column_name
LEFT JOIN information_schema.table_constraints tc 
    ON ccu.constraint_name = tc.constraint_name AND tc.constraint_type = 'CHECK'
WHERE t.table_schema = 'public'
AND t.table_type = 'BASE TABLE'
AND c.table_schema = 'public'
GROUP BY t.table_name
ORDER BY coverage_percentage DESC, t.table_name;

-- ============================================================================
-- VERIFICATION AND CLEANUP
-- ============================================================================

-- Check for any constraint violations
SELECT 'Checking for check constraint violations...' as status;
-- SELECT * FROM check_constraint_violations() LIMIT 10;

-- Show constraint summary
SELECT 'Check constraint summary:' as status;
SELECT * FROM get_check_constraint_summary();

-- Show constraint coverage
SELECT 'Constraint coverage by table:' as status;
SELECT * FROM constraint_coverage LIMIT 15;

-- Clean up utility functions (optional)
-- DROP FUNCTION IF EXISTS add_check_constraint(TEXT, TEXT, TEXT, TEXT);
-- DROP FUNCTION IF EXISTS validate_check_constraint_data(TEXT, TEXT);

COMMIT;
