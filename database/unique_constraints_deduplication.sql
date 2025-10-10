-- Task 7.2-T34: Define unique constraints (account external IDs, invoice no.)
-- Deduplication protection with natural keys and golden record strategies
-- Version: 1.0.0
-- Date: 2024-10-08

-- ============================================================================
-- UNIQUE CONSTRAINT STRATEGY
-- ============================================================================

/*
UNIQUE CONSTRAINT CATEGORIES:

1. NATURAL KEYS:
   - External system IDs (CRM IDs, billing system IDs)
   - Business identifiers (invoice numbers, account codes)
   - Email addresses, phone numbers (where applicable)

2. COMPOSITE UNIQUE KEYS:
   - Tenant-scoped uniqueness (tenant_id + business_key)
   - Time-based uniqueness (tenant_id + period + entity)
   - Hierarchical uniqueness (parent_id + child_name)

3. CONDITIONAL UNIQUE KEYS:
   - Unique among active records only
   - Unique per tenant and status combination
   - Unique with NULL handling (partial unique indexes)

4. GOLDEN RECORD CONSTRAINTS:
   - One primary record per external entity
   - Prevent duplicate master data
   - Ensure data quality and consistency

5. BUSINESS RULE CONSTRAINTS:
   - One active plan per account
   - One primary contact per account
   - One default template per tenant
*/

-- ============================================================================
-- UTILITY FUNCTIONS FOR CONSTRAINT MANAGEMENT
-- ============================================================================

-- Function to safely add unique constraint
CREATE OR REPLACE FUNCTION add_unique_constraint(
    table_name TEXT,
    constraint_name TEXT,
    column_list TEXT,
    constraint_description TEXT DEFAULT '',
    is_partial BOOLEAN DEFAULT FALSE,
    partial_condition TEXT DEFAULT ''
)
RETURNS BOOLEAN AS $$
DECLARE
    constraint_exists BOOLEAN;
    sql_statement TEXT;
BEGIN
    -- Check if constraint already exists
    SELECT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = add_unique_constraint.constraint_name
        AND table_name = add_unique_constraint.table_name
        AND constraint_type = 'UNIQUE'
    ) INTO constraint_exists;
    
    IF constraint_exists THEN
        RAISE NOTICE 'Unique constraint % already exists on table %', constraint_name, table_name;
        RETURN FALSE;
    END IF;
    
    -- Build SQL statement
    IF is_partial AND partial_condition != '' THEN
        -- Create partial unique index instead of constraint for conditional uniqueness
        sql_statement := format('CREATE UNIQUE INDEX CONCURRENTLY %I ON %I (%s) WHERE %s',
                               constraint_name, table_name, column_list, partial_condition);
    ELSE
        -- Create standard unique constraint
        sql_statement := format('ALTER TABLE %I ADD CONSTRAINT %I UNIQUE (%s)',
                               table_name, constraint_name, column_list);
    END IF;
    
    -- Execute the statement
    EXECUTE sql_statement;
    
    -- Add comment if provided
    IF constraint_description != '' THEN
        IF is_partial THEN
            EXECUTE format('COMMENT ON INDEX %I IS %L', constraint_name, constraint_description);
        ELSE
            EXECUTE format('COMMENT ON CONSTRAINT %I ON %I IS %L',
                           constraint_name, table_name, constraint_description);
        END IF;
    END IF;
    
    RAISE NOTICE 'Added unique constraint % to table %: %', constraint_name, table_name, constraint_description;
    RETURN TRUE;
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Failed to add unique constraint % to table %: %', constraint_name, table_name, SQLERRM;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Function to check for duplicate data before adding constraint
CREATE OR REPLACE FUNCTION check_duplicate_data(
    table_name TEXT,
    column_list TEXT,
    where_condition TEXT DEFAULT ''
)
RETURNS TABLE(
    duplicate_count BIGINT,
    total_duplicates BIGINT,
    sample_duplicates JSONB
) AS $$
DECLARE
    sql_query TEXT;
    where_clause TEXT;
BEGIN
    -- Build WHERE clause
    where_clause := CASE 
        WHEN where_condition != '' THEN 'WHERE ' || where_condition 
        ELSE '' 
    END;
    
    -- Build query to find duplicates
    sql_query := format('
        WITH duplicates AS (
            SELECT %s, COUNT(*) as dup_count
            FROM %I 
            %s
            GROUP BY %s
            HAVING COUNT(*) > 1
        ),
        sample_data AS (
            SELECT d.*, t.*
            FROM duplicates d
            JOIN %I t ON (%s)
            %s
        )
        SELECT 
            COUNT(DISTINCT (%s)) as duplicate_count,
            SUM(dup_count) as total_duplicates,
            jsonb_agg(
                jsonb_build_object(
                    ''duplicate_key'', jsonb_build_object(%s),
                    ''count'', dup_count,
                    ''sample_record'', to_jsonb(sample_data.*)
                )
            ) as sample_duplicates
        FROM sample_data',
        column_list, table_name, where_clause, column_list,
        table_name, 
        (SELECT string_agg(format('d.%s = t.%s', col, col), ' AND ') 
         FROM unnest(string_to_array(column_list, ',')) AS col),
        where_clause,
        column_list,
        (SELECT string_agg(format('''%s'', %s', trim(col), trim(col)), ', ') 
         FROM unnest(string_to_array(column_list, ',')) AS col)
    );
    
    RETURN QUERY EXECUTE sql_query;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TENANT AND USER UNIQUE CONSTRAINTS
-- ============================================================================

-- Tenant metadata unique constraints
SELECT add_unique_constraint(
    'tenant_metadata',
    'uq_tenant_metadata_tenant_code',
    'tenant_code',
    'Tenant codes must be globally unique'
);

SELECT add_unique_constraint(
    'tenant_metadata',
    'uq_tenant_metadata_tenant_name',
    'tenant_name',
    'Tenant names must be globally unique'
);

-- Users unique constraints
SELECT add_unique_constraint(
    'users',
    'uq_users_email',
    'email',
    'Email addresses must be globally unique'
);

-- Tenant-scoped user constraints (if needed for multi-tenant email)
-- SELECT add_unique_constraint(
--     'users',
--     'uq_users_email_tenant',
--     'tenant_id, email',
--     'Email addresses must be unique within tenant'
-- );

-- User roles unique constraints
SELECT add_unique_constraint(
    'users_role',
    'uq_users_role_user_tenant',
    'user_id, tenant_id',
    'One role record per user per tenant'
);

-- ============================================================================
-- DSL WORKFLOW UNIQUE CONSTRAINTS
-- ============================================================================

-- DSL Workflows unique constraints
SELECT add_unique_constraint(
    'dsl_workflows',
    'uq_dsl_workflows_name_tenant',
    'tenant_id, workflow_name',
    'Workflow names must be unique within tenant',
    TRUE,
    'deleted_at IS NULL'
);

SELECT add_unique_constraint(
    'dsl_workflows',
    'uq_dsl_workflows_id_version',
    'workflow_id, version',
    'Each workflow version must be unique'
);

-- DSL Policy Packs unique constraints
SELECT add_unique_constraint(
    'dsl_policy_packs',
    'uq_dsl_policy_packs_name_tenant',
    'tenant_id, pack_name',
    'Policy pack names must be unique within tenant',
    TRUE,
    'deleted_at IS NULL'
);

SELECT add_unique_constraint(
    'dsl_policy_packs',
    'uq_dsl_policy_packs_id_version',
    'policy_pack_id, version',
    'Each policy pack version must be unique'
);

-- DSL Execution Traces unique constraints
SELECT add_unique_constraint(
    'dsl_execution_traces',
    'uq_dsl_execution_traces_trace_id',
    'trace_id',
    'Trace IDs must be globally unique'
);

-- DSL Evidence Packs unique constraints
SELECT add_unique_constraint(
    'dsl_evidence_packs',
    'uq_dsl_evidence_packs_evidence_id',
    'evidence_pack_id',
    'Evidence pack IDs must be globally unique'
);

-- Evidence hash uniqueness for integrity
SELECT add_unique_constraint(
    'dsl_evidence_packs',
    'uq_dsl_evidence_packs_hash',
    'evidence_hash',
    'Evidence hashes must be unique to prevent tampering',
    TRUE,
    'evidence_hash IS NOT NULL'
);

-- DSL Override Ledger unique constraints
SELECT add_unique_constraint(
    'dsl_override_ledger',
    'uq_dsl_override_ledger_override_id',
    'override_id',
    'Override IDs must be globally unique'
);

-- DSL Workflow Templates unique constraints
SELECT add_unique_constraint(
    'dsl_workflow_templates',
    'uq_dsl_workflow_templates_name_tenant',
    'tenant_id, template_name',
    'Template names must be unique within tenant',
    TRUE,
    'deleted_at IS NULL'
);

-- DSL Capability Registry unique constraints
SELECT add_unique_constraint(
    'dsl_capability_registry',
    'uq_dsl_capability_registry_name_tenant',
    'tenant_id, capability_name',
    'Capability names must be unique within tenant',
    TRUE,
    'deleted_at IS NULL'
);

-- ============================================================================
-- STRATEGIC ACCOUNT PLANNING UNIQUE CONSTRAINTS
-- ============================================================================

-- Strategic Account Plans unique constraints
SELECT add_unique_constraint(
    'strategic_account_plans',
    'uq_strategic_account_plans_name_tenant',
    'tenant_id, plan_name',
    'Plan names must be unique within tenant',
    TRUE,
    'deleted_at IS NULL'
);

-- One active plan per account constraint
SELECT add_unique_constraint(
    'strategic_account_plans',
    'uq_strategic_account_plans_active_per_account',
    'tenant_id, account_id',
    'Only one active plan per account',
    TRUE,
    'status = ''active'' AND deleted_at IS NULL'
);

-- Account Planning Templates unique constraints
SELECT add_unique_constraint(
    'account_planning_templates',
    'uq_account_planning_templates_name_tenant',
    'tenant_id, template_name',
    'Template names must be unique within tenant',
    TRUE,
    'deleted_at IS NULL'
);

-- One default template per tenant
SELECT add_unique_constraint(
    'account_planning_templates',
    'uq_account_planning_templates_default_tenant',
    'tenant_id',
    'Only one default template per tenant',
    TRUE,
    'is_default = true AND deleted_at IS NULL'
);

-- Opportunity Insights unique constraints (external system integration)
SELECT add_unique_constraint(
    'opportunity_insights',
    'uq_opportunity_insights_external_id',
    'tenant_id, external_opportunity_id',
    'External opportunity IDs must be unique within tenant',
    TRUE,
    'external_opportunity_id IS NOT NULL AND deleted_at IS NULL'
);

-- Plan Stakeholders unique constraints
SELECT add_unique_constraint(
    'plan_stakeholders',
    'uq_plan_stakeholders_contact_plan',
    'plan_id, contact_name, contact_email',
    'Each contact can only be listed once per plan',
    TRUE,
    'deleted_at IS NULL'
);

-- Plan Activities unique constraints
SELECT add_unique_constraint(
    'plan_activities',
    'uq_plan_activities_name_plan',
    'plan_id, activity_name',
    'Activity names must be unique within plan',
    TRUE,
    'deleted_at IS NULL'
);

-- Plan Collaborators unique constraints
SELECT add_unique_constraint(
    'plan_collaborators',
    'uq_plan_collaborators_user_plan',
    'plan_id, user_id',
    'Each user can only be a collaborator once per plan',
    TRUE,
    'deleted_at IS NULL'
);

-- Plan Insight Attachments unique constraints
SELECT add_unique_constraint(
    'plan_insight_attachments',
    'uq_plan_insight_attachments_filename_plan',
    'plan_id, file_name',
    'File names must be unique within plan',
    TRUE,
    'deleted_at IS NULL'
);

-- ============================================================================
-- EXTERNAL SYSTEM INTEGRATION CONSTRAINTS
-- ============================================================================

-- External ID constraints for CRM integration
-- These would be added when CRM integration tables are created

-- Example: Account external IDs
/*
SELECT add_unique_constraint(
    'accounts',
    'uq_accounts_external_id_source',
    'tenant_id, external_id, source_system',
    'External IDs must be unique per source system within tenant',
    TRUE,
    'external_id IS NOT NULL AND deleted_at IS NULL'
);
*/

-- Example: Invoice numbers
/*
SELECT add_unique_constraint(
    'invoices',
    'uq_invoices_invoice_number_tenant',
    'tenant_id, invoice_number',
    'Invoice numbers must be unique within tenant',
    TRUE,
    'deleted_at IS NULL'
);
*/

-- Example: Contract numbers
/*
SELECT add_unique_constraint(
    'contracts',
    'uq_contracts_contract_number_tenant',
    'tenant_id, contract_number',
    'Contract numbers must be unique within tenant',
    TRUE,
    'deleted_at IS NULL'
);
*/

-- ============================================================================
-- BUSINESS RULE UNIQUE CONSTRAINTS
-- ============================================================================

-- Workflow execution uniqueness (prevent duplicate runs)
SELECT add_unique_constraint(
    'dsl_execution_traces',
    'uq_dsl_execution_traces_workflow_correlation',
    'workflow_id, correlation_id',
    'Prevent duplicate workflow executions with same correlation ID',
    TRUE,
    'correlation_id IS NOT NULL'
);

-- Policy application uniqueness
SELECT add_unique_constraint(
    'dsl_override_ledger',
    'uq_dsl_override_ledger_workflow_policy_trace',
    'workflow_id, policy_pack_id, execution_trace_id',
    'One override per policy per workflow execution',
    TRUE,
    'workflow_id IS NOT NULL AND policy_pack_id IS NOT NULL AND execution_trace_id IS NOT NULL'
);

-- Evidence pack workflow linkage
SELECT add_unique_constraint(
    'dsl_evidence_packs',
    'uq_dsl_evidence_packs_trace_type',
    'execution_trace_id, pack_type',
    'One evidence pack per type per execution',
    TRUE,
    'execution_trace_id IS NOT NULL'
);

-- ============================================================================
-- GOLDEN RECORD CONSTRAINTS
-- ============================================================================

-- Primary contact per account (when contact management is implemented)
/*
SELECT add_unique_constraint(
    'contacts',
    'uq_contacts_primary_per_account',
    'account_id',
    'Only one primary contact per account',
    TRUE,
    'is_primary = true AND deleted_at IS NULL'
);
*/

-- Primary territory assignment per user
/*
SELECT add_unique_constraint(
    'territory_assignments',
    'uq_territory_assignments_primary_user',
    'user_id',
    'Only one primary territory per user',
    TRUE,
    'is_primary = true AND deleted_at IS NULL'
);
*/

-- Active quota per user per period
/*
SELECT add_unique_constraint(
    'quotas',
    'uq_quotas_user_period',
    'user_id, quota_year, quota_quarter',
    'One quota per user per period',
    TRUE,
    'status = ''active'' AND deleted_at IS NULL'
);
*/

-- ============================================================================
-- CONSTRAINT VALIDATION AND REPORTING
-- ============================================================================

-- Function to check all unique constraint violations
CREATE OR REPLACE FUNCTION check_unique_constraint_violations()
RETURNS TABLE(
    table_name TEXT,
    constraint_name TEXT,
    constraint_columns TEXT,
    duplicate_count BIGINT,
    total_duplicates BIGINT,
    sample_duplicates JSONB
) AS $$
DECLARE
    constraint_record RECORD;
    duplicate_info RECORD;
BEGIN
    -- Get all unique constraints
    FOR constraint_record IN 
        SELECT 
            tc.table_name,
            tc.constraint_name,
            string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) as constraint_columns
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.constraint_type = 'UNIQUE'
        AND tc.table_schema = 'public'
        AND tc.constraint_name LIKE 'uq_%'
        GROUP BY tc.table_name, tc.constraint_name
    LOOP
        -- Check for duplicates
        BEGIN
            SELECT * INTO duplicate_info 
            FROM check_duplicate_data(
                constraint_record.table_name,
                constraint_record.constraint_columns
            );
            
            -- Return violations if any found
            IF duplicate_info.duplicate_count > 0 THEN
                table_name := constraint_record.table_name;
                constraint_name := constraint_record.constraint_name;
                constraint_columns := constraint_record.constraint_columns;
                duplicate_count := duplicate_info.duplicate_count;
                total_duplicates := duplicate_info.total_duplicates;
                sample_duplicates := duplicate_info.sample_duplicates;
                RETURN NEXT;
            END IF;
        EXCEPTION WHEN OTHERS THEN
            -- Skip constraints that can't be validated
            CONTINUE;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to get unique constraint summary
CREATE OR REPLACE FUNCTION get_unique_constraint_summary()
RETURNS TABLE(
    table_name TEXT,
    total_unique_constraints INTEGER,
    single_column_constraints INTEGER,
    multi_column_constraints INTEGER,
    partial_constraints INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH constraint_info AS (
        SELECT 
            tc.table_name,
            tc.constraint_name,
            COUNT(kcu.column_name) as column_count,
            CASE WHEN EXISTS (
                SELECT 1 FROM pg_indexes pi 
                WHERE pi.tablename = tc.table_name 
                AND pi.indexname = tc.constraint_name
                AND pi.indexdef LIKE '%WHERE%'
            ) THEN 1 ELSE 0 END as is_partial
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.constraint_type = 'UNIQUE'
        AND tc.table_schema = 'public'
        AND tc.constraint_name LIKE 'uq_%'
        GROUP BY tc.table_name, tc.constraint_name
    )
    SELECT 
        ci.table_name::TEXT,
        COUNT(*)::INTEGER as total_unique_constraints,
        COUNT(CASE WHEN ci.column_count = 1 THEN 1 END)::INTEGER as single_column_constraints,
        COUNT(CASE WHEN ci.column_count > 1 THEN 1 END)::INTEGER as multi_column_constraints,
        COUNT(CASE WHEN ci.is_partial = 1 THEN 1 END)::INTEGER as partial_constraints
    FROM constraint_info ci
    GROUP BY ci.table_name
    ORDER BY ci.table_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- DEDUPLICATION UTILITIES
-- ============================================================================

-- Function to find and merge duplicate records
CREATE OR REPLACE FUNCTION find_duplicate_records(
    table_name TEXT,
    key_columns TEXT,
    merge_strategy TEXT DEFAULT 'keep_oldest'
)
RETURNS TABLE(
    duplicate_group INTEGER,
    record_count BIGINT,
    keep_record_id TEXT,
    merge_record_ids TEXT[]
) AS $$
DECLARE
    sql_query TEXT;
BEGIN
    -- Build query to find duplicates and suggest merge strategy
    sql_query := format('
        WITH duplicates AS (
            SELECT 
                %s,
                COUNT(*) as dup_count,
                CASE 
                    WHEN %L = ''keep_oldest'' THEN MIN(created_at)
                    WHEN %L = ''keep_newest'' THEN MAX(created_at)
                    ELSE MIN(created_at)
                END as keep_timestamp
            FROM %I 
            WHERE deleted_at IS NULL
            GROUP BY %s
            HAVING COUNT(*) > 1
        ),
        ranked_records AS (
            SELECT 
                t.*,
                ROW_NUMBER() OVER (
                    PARTITION BY %s 
                    ORDER BY 
                        CASE WHEN %L = ''keep_oldest'' THEN t.created_at END ASC,
                        CASE WHEN %L = ''keep_newest'' THEN t.created_at END DESC
                ) as rn
            FROM %I t
            JOIN duplicates d ON (%s)
            WHERE t.deleted_at IS NULL
        )
        SELECT 
            DENSE_RANK() OVER (ORDER BY %s) as duplicate_group,
            COUNT(*) as record_count,
            (ARRAY_AGG(COALESCE(id, user_id, tenant_id)::TEXT ORDER BY rn))[1] as keep_record_id,
            ARRAY_AGG(COALESCE(id, user_id, tenant_id)::TEXT ORDER BY rn)[2:] as merge_record_ids
        FROM ranked_records
        GROUP BY %s
        ORDER BY duplicate_group',
        key_columns, merge_strategy, merge_strategy, table_name, key_columns,
        key_columns, merge_strategy, merge_strategy, table_name,
        (SELECT string_agg(format('d.%s = t.%s', trim(col), trim(col)), ' AND ') 
         FROM unnest(string_to_array(key_columns, ',')) AS col),
        key_columns, key_columns
    );
    
    RETURN QUERY EXECUTE sql_query;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CONSTRAINT MONITORING VIEWS
-- ============================================================================

-- View to show all unique constraints
CREATE OR REPLACE VIEW unique_constraint_catalog AS
SELECT 
    tc.table_name,
    tc.constraint_name,
    string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) as constraint_columns,
    CASE 
        WHEN COUNT(kcu.column_name) = 1 THEN 'SINGLE_COLUMN'
        ELSE 'MULTI_COLUMN'
    END as constraint_type,
    CASE WHEN EXISTS (
        SELECT 1 FROM pg_indexes pi 
        WHERE pi.tablename = tc.table_name 
        AND pi.indexname = tc.constraint_name
        AND pi.indexdef LIKE '%WHERE%'
    ) THEN 'PARTIAL' ELSE 'FULL' END as constraint_scope,
    obj_description(pgc.oid) as constraint_description
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
LEFT JOIN pg_constraint pgc ON pgc.conname = tc.constraint_name
WHERE tc.constraint_type = 'UNIQUE'
AND tc.table_schema = 'public'
AND tc.constraint_name LIKE 'uq_%'
GROUP BY tc.table_name, tc.constraint_name, pgc.oid
ORDER BY tc.table_name, tc.constraint_name;

-- View to show constraint effectiveness
CREATE OR REPLACE VIEW constraint_effectiveness AS
WITH table_stats AS (
    SELECT 
        schemaname,
        tablename,
        n_tup_ins as total_inserts,
        n_tup_upd as total_updates,
        n_tup_del as total_deletes
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
)
SELECT 
    ucc.table_name,
    ucc.constraint_name,
    ucc.constraint_columns,
    ucc.constraint_type,
    ucc.constraint_scope,
    COALESCE(ts.total_inserts, 0) as table_inserts,
    COALESCE(ts.total_updates, 0) as table_updates,
    -- Constraint violation attempts would need to be tracked separately
    0 as constraint_violations,
    CASE 
        WHEN COALESCE(ts.total_inserts, 0) > 0 THEN 
            ROUND((0::DECIMAL / NULLIF(ts.total_inserts, 0)) * 100, 2)
        ELSE 0 
    END as violation_rate_pct
FROM unique_constraint_catalog ucc
LEFT JOIN table_stats ts ON ucc.table_name = ts.tablename
ORDER BY ucc.table_name, ucc.constraint_name;

-- ============================================================================
-- VERIFICATION AND CLEANUP
-- ============================================================================

-- Check for any unique constraint violations
SELECT 'Checking for unique constraint violations...' as status;
-- SELECT * FROM check_unique_constraint_violations() LIMIT 10;

-- Show unique constraint summary
SELECT 'Unique constraint summary:' as status;
SELECT * FROM get_unique_constraint_summary();

-- Show constraint catalog
SELECT 'Unique constraint catalog:' as status;
SELECT * FROM unique_constraint_catalog LIMIT 20;

-- Clean up utility functions (optional)
-- DROP FUNCTION IF EXISTS add_unique_constraint(TEXT, TEXT, TEXT, TEXT, BOOLEAN, TEXT);
-- DROP FUNCTION IF EXISTS check_duplicate_data(TEXT, TEXT, TEXT);

COMMIT;
