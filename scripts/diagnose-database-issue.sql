-- =====================================================
-- Diagnose Database Connection and Schema Issues
-- Find out why scripts can't see the users table
-- =====================================================

-- 1. Basic connection info
SELECT 
    '🔍 CONNECTION INFO' as section,
    current_database() as connected_database,
    current_user as connected_user,
    current_schema() as current_schema,
    version() as postgres_version;

-- 2. Show all schemas
SELECT 
    '📁 ALL SCHEMAS' as section,
    schema_name,
    CASE 
        WHEN schema_name = 'public' THEN '🏠 Default'
        WHEN schema_name LIKE 'pg_%' THEN '🔧 System'
        WHEN schema_name = 'information_schema' THEN '📊 Info'
        ELSE '📂 Custom'
    END as schema_type
FROM information_schema.schemata
ORDER BY schema_name;

-- 3. Search for users table in ALL schemas (case insensitive)
SELECT 
    '👥 USERS TABLE SEARCH' as section,
    table_schema,
    table_name,
    table_type,
    CASE 
        WHEN table_schema = 'public' THEN '✅ Found in public schema'
        ELSE '⚠️ Found in different schema'
    END as status
FROM information_schema.tables 
WHERE LOWER(table_name) LIKE '%user%'
ORDER BY table_schema, table_name;

-- 4. Show ALL tables in public schema
SELECT 
    '📋 PUBLIC SCHEMA TABLES' as section,
    table_name,
    table_type,
    CASE 
        WHEN table_type = 'BASE TABLE' THEN '📊'
        WHEN table_type = 'VIEW' THEN '👁️'
        ELSE '❓'
    END as icon
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name;

-- 5. Show ALL tables in ALL schemas (limited to first 20)
SELECT 
    '🗂️ ALL TABLES (FIRST 20)' as section,
    table_schema,
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
ORDER BY table_schema, table_name
LIMIT 20;

-- 6. Try direct table access with different case variations
DO $$
BEGIN
    -- Try lowercase 'users'
    BEGIN
        PERFORM 1 FROM users LIMIT 1;
        RAISE NOTICE '✅ Found table: users (lowercase)';
    EXCEPTION
        WHEN undefined_table THEN
            RAISE NOTICE '❌ Table not found: users (lowercase)';
    END;

    -- Try uppercase 'USERS'
    BEGIN
        PERFORM 1 FROM "USERS" LIMIT 1;
        RAISE NOTICE '✅ Found table: USERS (uppercase)';
    EXCEPTION
        WHEN undefined_table THEN
            RAISE NOTICE '❌ Table not found: USERS (uppercase)';
    END;

    -- Try mixed case 'Users'
    BEGIN
        PERFORM 1 FROM "Users" LIMIT 1;
        RAISE NOTICE '✅ Found table: Users (mixed case)';
    EXCEPTION
        WHEN undefined_table THEN
            RAISE NOTICE '❌ Table not found: Users (mixed case)';
    END;
END $$;

-- 7. Check current search_path
SELECT 
    '🛤️ SEARCH PATH' as section,
    current_setting('search_path') as search_path,
    'This determines which schemas are searched automatically' as explanation;

-- 8. Try to find the table using pg_class (system catalog)
SELECT 
    '🔍 SYSTEM CATALOG SEARCH' as section,
    n.nspname as schema_name,
    c.relname as table_name,
    CASE c.relkind 
        WHEN 'r' THEN '📊 Table'
        WHEN 'v' THEN '👁️ View'
        WHEN 'm' THEN '📈 Materialized View'
        ELSE '❓ Other'
    END as object_type
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE LOWER(c.relname) LIKE '%user%'
    AND c.relkind IN ('r', 'v', 'm')  -- tables, views, materialized views
ORDER BY n.nspname, c.relname;

-- 9. Check user permissions
SELECT 
    '🔐 USER PERMISSIONS' as section,
    has_database_privilege(current_user, current_database(), 'CONNECT') as can_connect,
    has_schema_privilege(current_user, 'public', 'USAGE') as can_use_public_schema,
    has_schema_privilege(current_user, 'public', 'CREATE') as can_create_in_public;

-- 10. Manual table check - try to describe what you see
SELECT 
    '📝 MANUAL CHECK INSTRUCTIONS' as section,
    'Please run these commands manually in Azure Data Studio:' as instruction_1,
    '1. SELECT * FROM information_schema.tables WHERE table_schema = ''public'';' as command_1,
    '2. \dt (if psql command line)' as command_2,
    '3. SELECT tablename FROM pg_tables WHERE schemaname = ''public'';' as command_3;

-- 11. Alternative table listing using pg_tables
SELECT 
    '📊 PG_TABLES VIEW' as section,
    schemaname,
    tablename,
    tableowner,
    hasindexes,
    hasrules,
    hastriggers
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

-- 12. Final diagnostic
SELECT 
    '🎯 DIAGNOSTIC SUMMARY' as section,
    'If you can see your users table in Azure Data Studio but scripts cannot find it:' as issue,
    '1. Check if table is in a different schema' as solution_1,
    '2. Check if table name has different capitalization' as solution_2,
    '3. Check if you are connected to the correct database' as solution_3,
    '4. Check user permissions' as solution_4;

SELECT '🔍 DIAGNOSTIC COMPLETE - Check the results above!' as final_message;