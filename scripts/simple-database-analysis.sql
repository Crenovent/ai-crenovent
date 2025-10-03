-- =====================================================
-- Simple PostgreSQL Database Analysis Script
-- Compatible with all PostgreSQL versions
-- =====================================================

-- Database Information
SELECT 
    '🗄️ DATABASE INFO' as section,
    current_database() as database_name,
    current_user as current_user;

-- =====================================================
-- 1. ALL TABLES
-- =====================================================
SELECT 
    '📋 ALL TABLES' as section,
    table_name,
    table_type,
    CASE 
        WHEN table_type = 'BASE TABLE' THEN '📊 Table'
        WHEN table_type = 'VIEW' THEN '👁️ View'
        ELSE '❓ Other'
    END as type_icon
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name;

-- =====================================================
-- 2. ALL COLUMNS FOR ALL TABLES
-- =====================================================
SELECT 
    '🏗️ ALL COLUMNS' as section,
    table_name,
    ordinal_position as pos,
    column_name,
    data_type,
    CASE 
        WHEN character_maximum_length IS NOT NULL 
        THEN data_type || '(' || character_maximum_length || ')'
        WHEN numeric_precision IS NOT NULL AND numeric_scale IS NOT NULL
        THEN data_type || '(' || numeric_precision || ',' || numeric_scale || ')'
        WHEN numeric_precision IS NOT NULL
        THEN data_type || '(' || numeric_precision || ')'
        ELSE data_type
    END as full_type,
    CASE 
        WHEN is_nullable = 'YES' THEN '✅ NULL'
        ELSE '❌ NOT NULL'
    END as nullable,
    COALESCE(column_default, '(none)') as default_value
FROM information_schema.columns 
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;

-- =====================================================
-- 3. PRIMARY KEYS
-- =====================================================
SELECT 
    '🔑 PRIMARY KEYS' as section,
    tc.table_name,
    tc.constraint_name,
    STRING_AGG(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) as primary_key_columns
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY' 
    AND tc.table_schema = 'public'
GROUP BY tc.table_name, tc.constraint_name
ORDER BY tc.table_name;

-- =====================================================
-- 4. FOREIGN KEYS
-- =====================================================
SELECT 
    '🔗 FOREIGN KEYS' as section,
    tc.table_name as from_table,
    kcu.column_name as from_column,
    ccu.table_name as to_table,
    ccu.column_name as to_column,
    tc.constraint_name
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' 
    AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;

-- =====================================================
-- 5. UNIQUE CONSTRAINTS
-- =====================================================
SELECT 
    '🆔 UNIQUE CONSTRAINTS' as section,
    tc.table_name,
    tc.constraint_name,
    STRING_AGG(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) as unique_columns
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'UNIQUE' 
    AND tc.table_schema = 'public'
GROUP BY tc.table_name, tc.constraint_name
ORDER BY tc.table_name;

-- =====================================================
-- 6. ALL INDEXES (Basic)
-- =====================================================
SELECT 
    '🔍 INDEXES' as section,
    tablename as table_name,
    indexname as index_name,
    CASE 
        WHEN indexdef LIKE '%UNIQUE%' THEN '🆔 UNIQUE'
        WHEN indexdef LIKE '%PRIMARY%' THEN '🔑 PRIMARY'
        WHEN indexdef LIKE '%gin%' THEN '🌲 GIN'
        WHEN indexdef LIKE '%gist%' THEN '🗺️ GIST'
        ELSE '📊 BTREE'
    END as index_type
FROM pg_indexes 
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- =====================================================
-- 7. USERS TABLE CHECK (if exists)
-- =====================================================
SELECT 
    '👥 USERS TABLE CHECK' as section,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public')
        THEN '✅ Users table exists'
        ELSE '❌ Users table missing'
    END as users_table_status;

-- =====================================================
-- 8. USERS TABLE COLUMNS (if table exists)
-- =====================================================
SELECT 
    '👥 USERS TABLE COLUMNS' as section,
    column_name,
    data_type,
    CASE 
        WHEN character_maximum_length IS NOT NULL 
        THEN data_type || '(' || character_maximum_length || ')'
        ELSE data_type
    END as full_type,
    CASE 
        WHEN is_nullable = 'YES' THEN '✅ NULL'
        ELSE '❌ NOT NULL'
    END as nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'users' 
    AND table_schema = 'public'
ORDER BY ordinal_position;

-- =====================================================
-- 9. ROW COUNTS (for existing tables)
-- =====================================================

-- Check users table count
SELECT 
    '📊 USERS COUNT' as section,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public')
        THEN (SELECT COUNT(*)::text FROM users)
        ELSE 'Table does not exist'
    END as user_count;

-- Check strategic_account_plans count
SELECT 
    '📊 PLANS COUNT' as section,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'strategic_account_plans' AND table_schema = 'public')
        THEN (SELECT COUNT(*)::text FROM strategic_account_plans)
        ELSE 'Table does not exist'
    END as plans_count;

-- =====================================================
-- 10. ONBOARDING REQUIREMENTS CHECK
-- =====================================================
SELECT 
    '🎯 ONBOARDING REQUIREMENTS' as section,
    'users' as required_table,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public')
        THEN '✅ EXISTS'
        ELSE '❌ MISSING'
    END as table_status,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'profile' AND table_schema = 'public')
        THEN '✅ Profile column exists'
        ELSE '❌ Profile column missing'
    END as profile_status,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'tenant_id' AND table_schema = 'public')
        THEN '✅ Multi-tenant ready'
        ELSE '❌ No tenant support'
    END as tenant_status;

-- =====================================================
-- 11. REQUIRED COLUMNS FOR ONBOARDING
-- =====================================================
WITH required_cols AS (
    SELECT 'user_id' as col_name, 'BIGINT' as expected_type
    UNION ALL SELECT 'username', 'VARCHAR'
    UNION ALL SELECT 'email', 'VARCHAR'
    UNION ALL SELECT 'tenant_id', 'INTEGER'
    UNION ALL SELECT 'profile', 'JSONB'
    UNION ALL SELECT 'reports_to', 'BIGINT'
    UNION ALL SELECT 'is_activated', 'BOOLEAN'
    UNION ALL SELECT 'password', 'VARCHAR'
    UNION ALL SELECT 'created_at', 'TIMESTAMP'
    UNION ALL SELECT 'updated_at', 'TIMESTAMP'
)
SELECT 
    '📋 REQUIRED COLUMNS' as section,
    rc.col_name as required_column,
    rc.expected_type,
    CASE 
        WHEN c.column_name IS NOT NULL THEN '✅ EXISTS'
        ELSE '❌ MISSING'
    END as status,
    COALESCE(c.data_type, 'N/A') as actual_type
FROM required_cols rc
LEFT JOIN information_schema.columns c 
    ON c.table_name = 'users' 
    AND c.column_name = rc.col_name 
    AND c.table_schema = 'public'
ORDER BY 
    CASE WHEN c.column_name IS NOT NULL THEN 0 ELSE 1 END,
    rc.col_name;

-- =====================================================
-- 12. SAMPLE DATA (if users table exists and has data)
-- =====================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public') THEN
        IF (SELECT COUNT(*) FROM users) > 0 THEN
            RAISE NOTICE '📄 Users table has data - sample available';
        ELSE
            RAISE NOTICE '📄 Users table exists but is empty';
        END IF;
    ELSE
        RAISE NOTICE '📄 Users table does not exist';
    END IF;
END $$;

-- =====================================================
-- 13. SUMMARY
-- =====================================================
SELECT 
    '📊 SUMMARY' as section,
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE') as total_tables,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'public') as total_columns,
    (SELECT COUNT(*) FROM information_schema.table_constraints WHERE table_schema = 'public') as total_constraints,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public')
        THEN '✅ Ready for Onboarding'
        ELSE '❌ Needs Users Table'
    END as onboarding_status;

-- =====================================================
-- END
-- =====================================================
SELECT '🎉 ANALYSIS COMPLETE!' as final_message;