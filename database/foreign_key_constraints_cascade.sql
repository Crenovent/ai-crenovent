-- Task 7.2-T32: Implement foreign key constraints and cascade rules
-- Referential integrity with appropriate cascade behaviors
-- Version: 1.0.0
-- Date: 2024-10-08

-- ============================================================================
-- FOREIGN KEY CONSTRAINT STRATEGY
-- ============================================================================

/*
CASCADE STRATEGY:

1. CASCADE DELETE:
   - Child records that are meaningless without parent
   - Examples: comments -> plans, activities -> plans, line_items -> invoices

2. RESTRICT DELETE:
   - Parent records that should not be deleted if children exist
   - Examples: accounts with opportunities, users with created workflows

3. SET NULL:
   - Optional relationships where child can exist without parent
   - Examples: optional manager relationships, optional territory assignments

4. SET DEFAULT:
   - Relationships with meaningful defaults
   - Examples: setting default territory when specific territory is deleted

5. NO ACTION:
   - Default behavior, similar to RESTRICT but checked at end of transaction
   - Used for complex business logic constraints
*/

-- ============================================================================
-- UTILITY FUNCTIONS FOR CONSTRAINT MANAGEMENT
-- ============================================================================

-- Function to safely add foreign key constraint
CREATE OR REPLACE FUNCTION add_foreign_key_constraint(
    table_name TEXT,
    constraint_name TEXT,
    column_name TEXT,
    referenced_table TEXT,
    referenced_column TEXT,
    on_delete_action TEXT DEFAULT 'RESTRICT',
    on_update_action TEXT DEFAULT 'CASCADE'
)
RETURNS BOOLEAN AS $$
DECLARE
    constraint_exists BOOLEAN;
BEGIN
    -- Check if constraint already exists
    SELECT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = add_foreign_key_constraint.constraint_name
        AND table_name = add_foreign_key_constraint.table_name
        AND constraint_type = 'FOREIGN KEY'
    ) INTO constraint_exists;
    
    IF constraint_exists THEN
        RAISE NOTICE 'Foreign key constraint % already exists on table %', constraint_name, table_name;
        RETURN FALSE;
    END IF;
    
    -- Add the constraint
    EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I 
                    FOREIGN KEY (%I) REFERENCES %I(%I) 
                    ON DELETE %s ON UPDATE %s',
                   table_name, constraint_name, column_name, 
                   referenced_table, referenced_column,
                   on_delete_action, on_update_action);
    
    RAISE NOTICE 'Added foreign key constraint % to table %', constraint_name, table_name;
    RETURN TRUE;
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Failed to add foreign key constraint % to table %: %', constraint_name, table_name, SQLERRM;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Function to validate foreign key relationships
CREATE OR REPLACE FUNCTION validate_foreign_key_data(
    table_name TEXT,
    column_name TEXT,
    referenced_table TEXT,
    referenced_column TEXT
)
RETURNS TABLE(
    invalid_count BIGINT,
    sample_invalid_values TEXT[]
) AS $$
DECLARE
    sql_query TEXT;
BEGIN
    sql_query := format('
        SELECT 
            COUNT(*) as invalid_count,
            ARRAY_AGG(DISTINCT %I::TEXT) as sample_invalid_values
        FROM %I t1
        WHERE %I IS NOT NULL 
        AND NOT EXISTS (
            SELECT 1 FROM %I t2 WHERE t2.%I = t1.%I
        )',
        column_name, table_name, column_name,
        referenced_table, referenced_column, column_name
    );
    
    RETURN QUERY EXECUTE sql_query;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TENANT AND USER CONSTRAINTS (FOUNDATION)
-- ============================================================================

-- Users reference tenant_metadata
SELECT add_foreign_key_constraint(
    'users', 
    'fk_users_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'RESTRICT',  -- Don't allow tenant deletion if users exist
    'CASCADE'    -- Update tenant_id if it changes
);

-- Users self-reference for manager relationship
SELECT add_foreign_key_constraint(
    'users', 
    'fk_users_reports_to', 
    'reports_to', 
    'users', 
    'user_id', 
    'SET NULL',  -- If manager is deleted, set to NULL
    'CASCADE'
);

-- User roles reference users and tenants
SELECT add_foreign_key_constraint(
    'users_role', 
    'fk_users_role_user_id', 
    'user_id', 
    'users', 
    'user_id', 
    'CASCADE',   -- Delete role when user is deleted
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'users_role', 
    'fk_users_role_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'CASCADE',   -- Delete role when tenant is deleted
    'CASCADE'
);

-- ============================================================================
-- DSL WORKFLOW CONSTRAINTS
-- ============================================================================

-- DSL Workflows reference tenant and creator
SELECT add_foreign_key_constraint(
    'dsl_workflows', 
    'fk_dsl_workflows_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'RESTRICT',  -- Don't allow tenant deletion if workflows exist
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_workflows', 
    'fk_dsl_workflows_created_by', 
    'created_by', 
    'users', 
    'user_id', 
    'SET NULL',  -- If creator is deleted, set to NULL
    'CASCADE'
);

-- DSL Policy Packs reference tenant
SELECT add_foreign_key_constraint(
    'dsl_policy_packs', 
    'fk_dsl_policy_packs_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'RESTRICT',  -- Don't allow tenant deletion if policies exist
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_policy_packs', 
    'fk_dsl_policy_packs_created_by', 
    'created_by', 
    'users', 
    'user_id', 
    'SET NULL',
    'CASCADE'
);

-- DSL Execution Traces reference workflow and tenant
SELECT add_foreign_key_constraint(
    'dsl_execution_traces', 
    'fk_dsl_execution_traces_workflow_id', 
    'workflow_id', 
    'dsl_workflows', 
    'workflow_id', 
    'CASCADE',   -- Delete traces when workflow is deleted
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_execution_traces', 
    'fk_dsl_execution_traces_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'CASCADE',   -- Delete traces when tenant is deleted
    'CASCADE'
);

-- DSL Evidence Packs reference tenant, workflow, and trace
SELECT add_foreign_key_constraint(
    'dsl_evidence_packs', 
    'fk_dsl_evidence_packs_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'CASCADE',
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_evidence_packs', 
    'fk_dsl_evidence_packs_workflow_id', 
    'workflow_id', 
    'dsl_workflows', 
    'workflow_id', 
    'SET NULL',  -- Evidence can exist without workflow
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_evidence_packs', 
    'fk_dsl_evidence_packs_trace_id', 
    'execution_trace_id', 
    'dsl_execution_traces', 
    'trace_id', 
    'SET NULL',  -- Evidence can exist without specific trace
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_evidence_packs', 
    'fk_dsl_evidence_packs_policy_id', 
    'policy_pack_id', 
    'dsl_policy_packs', 
    'policy_pack_id', 
    'SET NULL',
    'CASCADE'
);

-- DSL Override Ledger references
SELECT add_foreign_key_constraint(
    'dsl_override_ledger', 
    'fk_dsl_override_ledger_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'CASCADE',
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_override_ledger', 
    'fk_dsl_override_ledger_user_id', 
    'user_id', 
    'users', 
    'user_id', 
    'RESTRICT',  -- Don't delete user if they have overrides
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_override_ledger', 
    'fk_dsl_override_ledger_workflow_id', 
    'workflow_id', 
    'dsl_workflows', 
    'workflow_id', 
    'SET NULL',
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_override_ledger', 
    'fk_dsl_override_ledger_trace_id', 
    'execution_trace_id', 
    'dsl_execution_traces', 
    'trace_id', 
    'SET NULL',
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_override_ledger', 
    'fk_dsl_override_ledger_approved_by', 
    'approved_by_user_id', 
    'users', 
    'user_id', 
    'SET NULL',  -- If approver is deleted, keep override but clear approver
    'CASCADE'
);

-- DSL Workflow Templates reference tenant
SELECT add_foreign_key_constraint(
    'dsl_workflow_templates', 
    'fk_dsl_workflow_templates_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'CASCADE',
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_workflow_templates', 
    'fk_dsl_workflow_templates_created_by', 
    'created_by', 
    'users', 
    'user_id', 
    'SET NULL',
    'CASCADE'
);

-- DSL Capability Registry reference tenant
SELECT add_foreign_key_constraint(
    'dsl_capability_registry', 
    'fk_dsl_capability_registry_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'CASCADE',
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'dsl_capability_registry', 
    'fk_dsl_capability_registry_created_by', 
    'created_by', 
    'users', 
    'user_id', 
    'SET NULL',
    'CASCADE'
);

-- ============================================================================
-- STRATEGIC ACCOUNT PLANNING CONSTRAINTS
-- ============================================================================

-- Strategic Account Plans reference tenant, template, and creator
SELECT add_foreign_key_constraint(
    'strategic_account_plans', 
    'fk_strategic_account_plans_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'RESTRICT',  -- Don't allow tenant deletion if plans exist
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'strategic_account_plans', 
    'fk_strategic_account_plans_template_id', 
    'template_id', 
    'account_planning_templates', 
    'template_id', 
    'SET NULL',  -- Plan can exist without template
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'strategic_account_plans', 
    'fk_strategic_account_plans_created_by', 
    'created_by_user_id', 
    'users', 
    'user_id', 
    'RESTRICT',  -- Don't delete user if they have plans
    'CASCADE'
);

-- Account Planning Templates reference tenant
SELECT add_foreign_key_constraint(
    'account_planning_templates', 
    'fk_account_planning_templates_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'CASCADE',
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'account_planning_templates', 
    'fk_account_planning_templates_created_by', 
    'created_by_user_id', 
    'users', 
    'user_id', 
    'SET NULL',
    'CASCADE'
);

-- Opportunity Insights reference tenant
SELECT add_foreign_key_constraint(
    'opportunity_insights', 
    'fk_opportunity_insights_tenant_id', 
    'tenant_id', 
    'tenant_metadata', 
    'tenant_id', 
    'CASCADE',
    'CASCADE'
);

-- Plan Stakeholders reference plan (CASCADE DELETE)
SELECT add_foreign_key_constraint(
    'plan_stakeholders', 
    'fk_plan_stakeholders_plan_id', 
    'plan_id', 
    'strategic_account_plans', 
    'plan_id', 
    'CASCADE',   -- Delete stakeholders when plan is deleted
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'plan_stakeholders', 
    'fk_plan_stakeholders_created_by', 
    'created_by_user_id', 
    'users', 
    'user_id', 
    'SET NULL',
    'CASCADE'
);

-- Plan Activities reference plan (CASCADE DELETE)
SELECT add_foreign_key_constraint(
    'plan_activities', 
    'fk_plan_activities_plan_id', 
    'plan_id', 
    'strategic_account_plans', 
    'plan_id', 
    'CASCADE',   -- Delete activities when plan is deleted
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'plan_activities', 
    'fk_plan_activities_owner', 
    'owner_user_id', 
    'users', 
    'user_id', 
    'SET NULL',  -- Activity can exist without owner
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'plan_activities', 
    'fk_plan_activities_created_by', 
    'created_by_user_id', 
    'users', 
    'user_id', 
    'SET NULL',
    'CASCADE'
);

-- Plan Collaborators reference plan and user (CASCADE DELETE)
SELECT add_foreign_key_constraint(
    'plan_collaborators', 
    'fk_plan_collaborators_plan_id', 
    'plan_id', 
    'strategic_account_plans', 
    'plan_id', 
    'CASCADE',   -- Delete collaborators when plan is deleted
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'plan_collaborators', 
    'fk_plan_collaborators_user_id', 
    'user_id', 
    'users', 
    'user_id', 
    'CASCADE',   -- Delete collaboration when user is deleted
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'plan_collaborators', 
    'fk_plan_collaborators_added_by', 
    'added_by_user_id', 
    'users', 
    'user_id', 
    'SET NULL',
    'CASCADE'
);

-- Plan Comments reference plan (CASCADE DELETE)
SELECT add_foreign_key_constraint(
    'plan_comments', 
    'fk_plan_comments_plan_id', 
    'plan_id', 
    'strategic_account_plans', 
    'plan_id', 
    'CASCADE',   -- Delete comments when plan is deleted
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'plan_comments', 
    'fk_plan_comments_user_id', 
    'user_id', 
    'users', 
    'user_id', 
    'SET NULL',  -- Keep comment but clear user if user is deleted
    'CASCADE'
);

-- Plan Insight Attachments reference plan (CASCADE DELETE)
SELECT add_foreign_key_constraint(
    'plan_insight_attachments', 
    'fk_plan_insight_attachments_plan_id', 
    'plan_id', 
    'strategic_account_plans', 
    'plan_id', 
    'CASCADE',   -- Delete attachments when plan is deleted
    'CASCADE'
);

SELECT add_foreign_key_constraint(
    'plan_insight_attachments', 
    'fk_plan_insight_attachments_uploaded_by', 
    'uploaded_by_user_id', 
    'users', 
    'user_id', 
    'SET NULL',
    'CASCADE'
);

-- ============================================================================
-- CROSS-REFERENCE CONSTRAINTS (GOVERNANCE LINKAGES)
-- ============================================================================

-- Evidence packs can reference override ledger entries
SELECT add_foreign_key_constraint(
    'dsl_evidence_packs', 
    'fk_dsl_evidence_packs_override_id', 
    'override_id', 
    'dsl_override_ledger', 
    'override_id', 
    'SET NULL',  -- Evidence can exist without override
    'CASCADE'
);

-- Override ledger can reference policy packs
SELECT add_foreign_key_constraint(
    'dsl_override_ledger', 
    'fk_dsl_override_ledger_policy_pack_id', 
    'policy_pack_id', 
    'dsl_policy_packs', 
    'policy_pack_id', 
    'SET NULL',
    'CASCADE'
);

-- Override ledger can reference evidence packs
SELECT add_foreign_key_constraint(
    'dsl_override_ledger', 
    'fk_dsl_override_ledger_evidence_pack_id', 
    'evidence_pack_id', 
    'dsl_evidence_packs', 
    'evidence_pack_id', 
    'SET NULL',
    'CASCADE'
);

-- ============================================================================
-- CONSTRAINT VALIDATION AND REPORTING
-- ============================================================================

-- Function to check all foreign key constraint violations
CREATE OR REPLACE FUNCTION check_foreign_key_violations()
RETURNS TABLE(
    table_name TEXT,
    constraint_name TEXT,
    column_name TEXT,
    referenced_table TEXT,
    referenced_column TEXT,
    violation_count BIGINT,
    sample_violations TEXT[]
) AS $$
DECLARE
    fk_record RECORD;
    violation_info RECORD;
BEGIN
    -- Get all foreign key constraints
    FOR fk_record IN 
        SELECT 
            tc.table_name,
            tc.constraint_name,
            kcu.column_name,
            ccu.table_name AS referenced_table,
            ccu.column_name AS referenced_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu 
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = 'public'
    LOOP
        -- Check for violations
        SELECT * INTO violation_info 
        FROM validate_foreign_key_data(
            fk_record.table_name,
            fk_record.column_name,
            fk_record.referenced_table,
            fk_record.referenced_column
        );
        
        -- Return violations if any found
        IF violation_info.invalid_count > 0 THEN
            table_name := fk_record.table_name;
            constraint_name := fk_record.constraint_name;
            column_name := fk_record.column_name;
            referenced_table := fk_record.referenced_table;
            referenced_column := fk_record.referenced_column;
            violation_count := violation_info.invalid_count;
            sample_violations := violation_info.sample_invalid_values;
            RETURN NEXT;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to get foreign key constraint summary
CREATE OR REPLACE FUNCTION get_foreign_key_summary()
RETURNS TABLE(
    table_name TEXT,
    total_constraints INTEGER,
    cascade_delete INTEGER,
    restrict_delete INTEGER,
    set_null_delete INTEGER,
    no_action_delete INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tc.table_name::TEXT,
        COUNT(*)::INTEGER as total_constraints,
        COUNT(CASE WHEN rc.delete_rule = 'CASCADE' THEN 1 END)::INTEGER as cascade_delete,
        COUNT(CASE WHEN rc.delete_rule = 'RESTRICT' THEN 1 END)::INTEGER as restrict_delete,
        COUNT(CASE WHEN rc.delete_rule = 'SET NULL' THEN 1 END)::INTEGER as set_null_delete,
        COUNT(CASE WHEN rc.delete_rule = 'NO ACTION' THEN 1 END)::INTEGER as no_action_delete
    FROM information_schema.table_constraints tc
    JOIN information_schema.referential_constraints rc 
        ON tc.constraint_name = rc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = 'public'
    GROUP BY tc.table_name
    ORDER BY tc.table_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CONSTRAINT TESTING FUNCTIONS
-- ============================================================================

-- Function to test cascade delete behavior
CREATE OR REPLACE FUNCTION test_cascade_delete(
    parent_table TEXT,
    parent_id_column TEXT,
    parent_id_value TEXT,
    dry_run BOOLEAN DEFAULT TRUE
)
RETURNS TABLE(
    affected_table TEXT,
    affected_count BIGINT,
    cascade_action TEXT
) AS $$
DECLARE
    fk_record RECORD;
    count_result BIGINT;
    sql_query TEXT;
BEGIN
    -- Find all tables that reference the parent table
    FOR fk_record IN 
        SELECT 
            tc.table_name,
            kcu.column_name,
            rc.delete_rule
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu 
            ON ccu.constraint_name = tc.constraint_name
        JOIN information_schema.referential_constraints rc 
            ON tc.constraint_name = rc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = 'public'
        AND ccu.table_name = parent_table
        AND ccu.column_name = parent_id_column
    LOOP
        -- Count affected records
        sql_query := format('SELECT COUNT(*) FROM %I WHERE %I = %L',
                           fk_record.table_name, fk_record.column_name, parent_id_value);
        EXECUTE sql_query INTO count_result;
        
        IF count_result > 0 THEN
            affected_table := fk_record.table_name;
            affected_count := count_result;
            cascade_action := fk_record.delete_rule;
            RETURN NEXT;
        END IF;
    END LOOP;
    
    -- Show what would happen to parent record
    sql_query := format('SELECT COUNT(*) FROM %I WHERE %I = %L',
                       parent_table, parent_id_column, parent_id_value);
    EXECUTE sql_query INTO count_result;
    
    IF count_result > 0 THEN
        affected_table := parent_table || ' (PARENT)';
        affected_count := count_result;
        cascade_action := 'DELETE';
        RETURN NEXT;
    END IF;
    
    IF NOT dry_run THEN
        RAISE NOTICE 'DRY RUN DISABLED - This would actually delete records!';
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CONSTRAINT MONITORING VIEWS
-- ============================================================================

-- View to monitor foreign key constraint status
CREATE OR REPLACE VIEW foreign_key_status AS
SELECT 
    tc.table_name,
    tc.constraint_name,
    kcu.column_name,
    ccu.table_name as referenced_table,
    ccu.column_name as referenced_column,
    rc.delete_rule,
    rc.update_rule,
    CASE 
        WHEN tc.is_deferrable = 'YES' THEN 'DEFERRABLE'
        ELSE 'IMMEDIATE'
    END as constraint_timing
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu 
    ON ccu.constraint_name = tc.constraint_name
JOIN information_schema.referential_constraints rc 
    ON tc.constraint_name = rc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_schema = 'public'
ORDER BY tc.table_name, tc.constraint_name;

-- View to show constraint dependency tree
CREATE OR REPLACE VIEW constraint_dependency_tree AS
WITH RECURSIVE constraint_tree AS (
    -- Base case: tables with no foreign key dependencies
    SELECT 
        table_name,
        0 as level,
        ARRAY[table_name] as path
    FROM information_schema.tables t
    WHERE table_schema = 'public'
    AND table_type = 'BASE TABLE'
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints tc
        WHERE tc.table_name = t.table_name
        AND tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = 'public'
    )
    
    UNION ALL
    
    -- Recursive case: tables that depend on tables already in the tree
    SELECT 
        tc.table_name,
        ct.level + 1,
        ct.path || tc.table_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.constraint_column_usage ccu 
        ON tc.constraint_name = ccu.constraint_name
    JOIN constraint_tree ct ON ccu.table_name = ct.table_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = 'public'
    AND NOT (tc.table_name = ANY(ct.path))  -- Prevent cycles
)
SELECT 
    table_name,
    level,
    path,
    REPEAT('  ', level) || table_name as indented_name
FROM constraint_tree
ORDER BY level, table_name;

-- ============================================================================
-- VERIFICATION AND CLEANUP
-- ============================================================================

-- Check for any foreign key violations
SELECT 'Checking for foreign key violations...' as status;
SELECT * FROM check_foreign_key_violations();

-- Show foreign key constraint summary
SELECT 'Foreign key constraint summary:' as status;
SELECT * FROM get_foreign_key_summary();

-- Show constraint dependency levels
SELECT 'Constraint dependency tree:' as status;
SELECT table_name, level, indented_name FROM constraint_dependency_tree LIMIT 20;

-- Clean up utility functions (optional)
-- DROP FUNCTION IF EXISTS add_foreign_key_constraint(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT);
-- DROP FUNCTION IF EXISTS validate_foreign_key_data(TEXT, TEXT, TEXT, TEXT);

COMMIT;
