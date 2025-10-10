-- Task 7.2-T18: Add audit columns (created_at/by, updated_at/by, deleted_at)
-- Migration to add audit columns to all existing tables
-- Version: 1.0.0
-- Date: 2024-10-08

-- ============================================================================
-- AUDIT COLUMNS MIGRATION
-- ============================================================================

-- Function to add audit columns to a table
CREATE OR REPLACE FUNCTION add_audit_columns(table_name TEXT, schema_name TEXT DEFAULT 'public')
RETURNS VOID AS $$
BEGIN
    -- Add created_at if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schema_name 
        AND table_name = table_name 
        AND column_name = 'created_at'
    ) THEN
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()', schema_name, table_name);
    END IF;
    
    -- Add created_by if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schema_name 
        AND table_name = table_name 
        AND column_name = 'created_by'
    ) THEN
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN created_by INTEGER', schema_name, table_name);
    END IF;
    
    -- Add updated_at if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schema_name 
        AND table_name = table_name 
        AND column_name = 'updated_at'
    ) THEN
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()', schema_name, table_name);
    END IF;
    
    -- Add updated_by if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schema_name 
        AND table_name = table_name 
        AND column_name = 'updated_by'
    ) THEN
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN updated_by INTEGER', schema_name, table_name);
    END IF;
    
    -- Add deleted_at if it doesn't exist (for soft delete)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schema_name 
        AND table_name = table_name 
        AND column_name = 'deleted_at'
    ) THEN
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN deleted_at TIMESTAMPTZ', schema_name, table_name);
    END IF;
    
    RAISE NOTICE 'Added audit columns to %.%', schema_name, table_name;
END;
$$ LANGUAGE plpgsql;

-- Function to create update trigger for updated_at column
CREATE OR REPLACE FUNCTION create_updated_at_trigger(table_name TEXT, schema_name TEXT DEFAULT 'public')
RETURNS VOID AS $$
BEGIN
    -- Create trigger function if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column') THEN
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $trigger$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $trigger$ LANGUAGE plpgsql;
    END IF;
    
    -- Drop existing trigger if it exists
    EXECUTE format('DROP TRIGGER IF EXISTS update_%I_updated_at ON %I.%I', table_name, schema_name, table_name);
    
    -- Create new trigger
    EXECUTE format('CREATE TRIGGER update_%I_updated_at 
                    BEFORE UPDATE ON %I.%I 
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()', 
                   table_name, schema_name, table_name);
    
    RAISE NOTICE 'Created updated_at trigger for %.%', schema_name, table_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- APPLY AUDIT COLUMNS TO EXISTING TABLES
-- ============================================================================

-- Core tenant and user tables
SELECT add_audit_columns('tenant_metadata');
SELECT create_updated_at_trigger('tenant_metadata');

SELECT add_audit_columns('users');
SELECT create_updated_at_trigger('users');

SELECT add_audit_columns('users_role');
SELECT create_updated_at_trigger('users_role');

-- DSL workflow tables
SELECT add_audit_columns('dsl_workflows');
SELECT create_updated_at_trigger('dsl_workflows');

SELECT add_audit_columns('dsl_policy_packs');
SELECT create_updated_at_trigger('dsl_policy_packs');

SELECT add_audit_columns('dsl_execution_traces');
SELECT create_updated_at_trigger('dsl_execution_traces');

SELECT add_audit_columns('dsl_evidence_packs');
SELECT create_updated_at_trigger('dsl_evidence_packs');

SELECT add_audit_columns('dsl_override_ledger');
SELECT create_updated_at_trigger('dsl_override_ledger');

SELECT add_audit_columns('dsl_workflow_templates');
SELECT create_updated_at_trigger('dsl_workflow_templates');

SELECT add_audit_columns('dsl_capability_registry');
SELECT create_updated_at_trigger('dsl_capability_registry');

-- Strategic account planning tables
SELECT add_audit_columns('strategic_account_plans');
SELECT create_updated_at_trigger('strategic_account_plans');

SELECT add_audit_columns('account_planning_templates');
SELECT create_updated_at_trigger('account_planning_templates');

SELECT add_audit_columns('opportunity_insights');
SELECT create_updated_at_trigger('opportunity_insights');

SELECT add_audit_columns('plan_stakeholders');
SELECT create_updated_at_trigger('plan_stakeholders');

SELECT add_audit_columns('plan_activities');
SELECT create_updated_at_trigger('plan_activities');

SELECT add_audit_columns('plan_collaborators');
SELECT create_updated_at_trigger('plan_collaborators');

SELECT add_audit_columns('plan_comments');
SELECT create_updated_at_trigger('plan_comments');

SELECT add_audit_columns('plan_insight_attachments');
SELECT create_updated_at_trigger('plan_insight_attachments');

-- ============================================================================
-- ADD FOREIGN KEY CONSTRAINTS FOR AUDIT COLUMNS
-- ============================================================================

-- Function to add foreign key constraints for audit columns
CREATE OR REPLACE FUNCTION add_audit_foreign_keys(table_name TEXT, schema_name TEXT DEFAULT 'public')
RETURNS VOID AS $$
BEGIN
    -- Add foreign key for created_by if column exists and constraint doesn't exist
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schema_name 
        AND table_name = table_name 
        AND column_name = 'created_by'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_schema = schema_name 
        AND tc.table_name = table_name 
        AND kcu.column_name = 'created_by'
        AND tc.constraint_type = 'FOREIGN KEY'
    ) THEN
        BEGIN
            EXECUTE format('ALTER TABLE %I.%I ADD CONSTRAINT fk_%I_created_by 
                           FOREIGN KEY (created_by) REFERENCES users(user_id)', 
                          schema_name, table_name, table_name);
            RAISE NOTICE 'Added created_by foreign key to %.%', schema_name, table_name;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Could not add created_by foreign key to %.%: %', schema_name, table_name, SQLERRM;
        END;
    END IF;
    
    -- Add foreign key for updated_by if column exists and constraint doesn't exist
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schema_name 
        AND table_name = table_name 
        AND column_name = 'updated_by'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_schema = schema_name 
        AND tc.table_name = table_name 
        AND kcu.column_name = 'updated_by'
        AND tc.constraint_type = 'FOREIGN KEY'
    ) THEN
        BEGIN
            EXECUTE format('ALTER TABLE %I.%I ADD CONSTRAINT fk_%I_updated_by 
                           FOREIGN KEY (updated_by) REFERENCES users(user_id)', 
                          schema_name, table_name, table_name);
            RAISE NOTICE 'Added updated_by foreign key to %.%', schema_name, table_name;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Could not add updated_by foreign key to %.%: %', schema_name, table_name, SQLERRM;
        END;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Apply foreign key constraints to tables (where users table exists)
SELECT add_audit_foreign_keys('tenant_metadata');
SELECT add_audit_foreign_keys('users_role');
SELECT add_audit_foreign_keys('dsl_workflows');
SELECT add_audit_foreign_keys('dsl_policy_packs');
SELECT add_audit_foreign_keys('dsl_execution_traces');
SELECT add_audit_foreign_keys('dsl_evidence_packs');
SELECT add_audit_foreign_keys('dsl_override_ledger');
SELECT add_audit_foreign_keys('dsl_workflow_templates');
SELECT add_audit_foreign_keys('dsl_capability_registry');
SELECT add_audit_foreign_keys('strategic_account_plans');
SELECT add_audit_foreign_keys('account_planning_templates');
SELECT add_audit_foreign_keys('opportunity_insights');
SELECT add_audit_foreign_keys('plan_stakeholders');
SELECT add_audit_foreign_keys('plan_activities');
SELECT add_audit_foreign_keys('plan_collaborators');
SELECT add_audit_foreign_keys('plan_comments');
SELECT add_audit_foreign_keys('plan_insight_attachments');

-- ============================================================================
-- CREATE INDEXES FOR AUDIT COLUMNS
-- ============================================================================

-- Function to create indexes for audit columns
CREATE OR REPLACE FUNCTION create_audit_indexes(table_name TEXT, schema_name TEXT DEFAULT 'public')
RETURNS VOID AS $$
BEGIN
    -- Index on created_at for temporal queries
    BEGIN
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_created_at ON %I.%I (created_at DESC)', 
                      table_name, schema_name, table_name);
        RAISE NOTICE 'Created created_at index for %.%', schema_name, table_name;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Could not create created_at index for %.%: %', schema_name, table_name, SQLERRM;
    END;
    
    -- Index on updated_at for change tracking
    BEGIN
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_updated_at ON %I.%I (updated_at DESC)', 
                      table_name, schema_name, table_name);
        RAISE NOTICE 'Created updated_at index for %.%', schema_name, table_name;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Could not create updated_at index for %.%: %', schema_name, table_name, SQLERRM;
    END;
    
    -- Index on created_by for user activity tracking
    BEGIN
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_created_by ON %I.%I (created_by)', 
                      table_name, schema_name, table_name);
        RAISE NOTICE 'Created created_by index for %.%', schema_name, table_name;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Could not create created_by index for %.%: %', schema_name, table_name, SQLERRM;
    END;
    
    -- Partial index on deleted_at for soft delete queries
    BEGIN
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_deleted_at ON %I.%I (deleted_at) WHERE deleted_at IS NOT NULL', 
                      table_name, schema_name, table_name);
        RAISE NOTICE 'Created deleted_at index for %.%', schema_name, table_name;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Could not create deleted_at index for %.%: %', schema_name, table_name, SQLERRM;
    END;
    
    -- Composite index for active records (not deleted)
    BEGIN
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_active_records ON %I.%I (tenant_id, created_at DESC) WHERE deleted_at IS NULL', 
                      table_name, schema_name, table_name);
        RAISE NOTICE 'Created active records index for %.%', schema_name, table_name;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Could not create active records index for %.%: %', schema_name, table_name, SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql;

-- Apply indexes to all tables
SELECT create_audit_indexes('tenant_metadata');
SELECT create_audit_indexes('users');
SELECT create_audit_indexes('users_role');
SELECT create_audit_indexes('dsl_workflows');
SELECT create_audit_indexes('dsl_policy_packs');
SELECT create_audit_indexes('dsl_execution_traces');
SELECT create_audit_indexes('dsl_evidence_packs');
SELECT create_audit_indexes('dsl_override_ledger');
SELECT create_audit_indexes('dsl_workflow_templates');
SELECT create_audit_indexes('dsl_capability_registry');
SELECT create_audit_indexes('strategic_account_plans');
SELECT create_audit_indexes('account_planning_templates');
SELECT create_audit_indexes('opportunity_insights');
SELECT create_audit_indexes('plan_stakeholders');
SELECT create_audit_indexes('plan_activities');
SELECT create_audit_indexes('plan_collaborators');
SELECT create_audit_indexes('plan_comments');
SELECT create_audit_indexes('plan_insight_attachments');

-- ============================================================================
-- AUDIT COLUMN USAGE GUIDELINES
-- ============================================================================

-- Create view to show audit column usage across all tables
CREATE OR REPLACE VIEW audit_columns_status AS
SELECT 
    schemaname,
    tablename,
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schemaname 
        AND table_name = tablename 
        AND column_name = 'created_at'
    ) THEN 'YES' ELSE 'NO' END as has_created_at,
    
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schemaname 
        AND table_name = tablename 
        AND column_name = 'created_by'
    ) THEN 'YES' ELSE 'NO' END as has_created_by,
    
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schemaname 
        AND table_name = tablename 
        AND column_name = 'updated_at'
    ) THEN 'YES' ELSE 'NO' END as has_updated_at,
    
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schemaname 
        AND table_name = tablename 
        AND column_name = 'updated_by'
    ) THEN 'YES' ELSE 'NO' END as has_updated_by,
    
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = schemaname 
        AND table_name = tablename 
        AND column_name = 'deleted_at'
    ) THEN 'YES' ELSE 'NO' END as has_deleted_at,
    
    CASE WHEN EXISTS (
        SELECT 1 FROM pg_trigger t
        JOIN pg_class c ON t.tgrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = schemaname
        AND c.relname = tablename
        AND t.tgname LIKE '%updated_at%'
    ) THEN 'YES' ELSE 'NO' END as has_update_trigger
    
FROM pg_tables 
WHERE schemaname = 'public'
AND tablename NOT LIKE 'pg_%'
AND tablename NOT LIKE 'sql_%'
ORDER BY tablename;

-- Function to get audit trail for a specific record
CREATE OR REPLACE FUNCTION get_audit_trail(
    p_table_name TEXT,
    p_record_id TEXT,
    p_id_column TEXT DEFAULT 'id'
)
RETURNS TABLE(
    action TEXT,
    timestamp TIMESTAMPTZ,
    user_id INTEGER,
    details JSONB
) AS $$
DECLARE
    sql_query TEXT;
BEGIN
    -- Build dynamic query to get audit information
    sql_query := format('
        SELECT 
            CASE 
                WHEN deleted_at IS NOT NULL THEN ''DELETE''
                WHEN created_at = updated_at THEN ''CREATE''
                ELSE ''UPDATE''
            END as action,
            COALESCE(updated_at, created_at) as timestamp,
            COALESCE(updated_by, created_by) as user_id,
            jsonb_build_object(
                ''created_at'', created_at,
                ''created_by'', created_by,
                ''updated_at'', updated_at,
                ''updated_by'', updated_by,
                ''deleted_at'', deleted_at
            ) as details
        FROM %I 
        WHERE %I = $1
        ORDER BY COALESCE(updated_at, created_at) DESC',
        p_table_name, p_id_column
    );
    
    RETURN QUERY EXECUTE sql_query USING p_record_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CLEANUP FUNCTIONS
-- ============================================================================

-- Drop the helper functions (optional - keep for future use)
-- DROP FUNCTION IF EXISTS add_audit_columns(TEXT, TEXT);
-- DROP FUNCTION IF EXISTS create_updated_at_trigger(TEXT, TEXT);
-- DROP FUNCTION IF EXISTS add_audit_foreign_keys(TEXT, TEXT);
-- DROP FUNCTION IF EXISTS create_audit_indexes(TEXT, TEXT);

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check audit columns status
SELECT * FROM audit_columns_status;

-- Count tables with complete audit columns
SELECT 
    COUNT(*) as total_tables,
    COUNT(CASE WHEN has_created_at = 'YES' AND has_updated_at = 'YES' THEN 1 END) as tables_with_audit,
    COUNT(CASE WHEN has_update_trigger = 'YES' THEN 1 END) as tables_with_triggers
FROM audit_columns_status;

-- Show sample audit trail (replace with actual table and ID)
-- SELECT * FROM get_audit_trail('users', '1', 'user_id') LIMIT 5;

COMMIT;
