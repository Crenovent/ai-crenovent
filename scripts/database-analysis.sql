-- =====================================================
-- PostgreSQL Database Analysis Script for Azure Data Studio
-- Shows all tables, columns, indexes, and relationships
-- =====================================================

-- Database Information
SELECT 
    '🗄️ DATABASE INFORMATION' as section,
    current_database() as database_name,
    current_user as current_user,
    version() as postgres_version;

-- =====================================================
-- 1. ALL TABLES OVERVIEW
-- =====================================================
SELECT 
    '📋 TABLES OVERVIEW' as section,
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

-- =====================================================
-- 2. DETAILED TABLE ANALYSIS WITH ROW COUNTS
-- =====================================================
SELECT 
    '📊 TABLE STATISTICS' as section,
    t.table_name,
    CASE 
        WHEN t.table_name = 'users' THEN 
            COALESCE((SELECT COUNT(*) FROM users), 0)
        WHEN t.table_name = 'strategic_account_plans' THEN 
            COALESCE((SELECT COUNT(*) FROM strategic_account_plans), 0)
        ELSE 0
    END as estimated_rows,
    CASE 
        WHEN t.table_name = 'users' THEN 
            CASE 
                WHEN COALESCE((SELECT COUNT(*) FROM users), 0) = 0 THEN '🔴 Empty'
                WHEN COALESCE((SELECT COUNT(*) FROM users), 0) < 100 THEN '🟡 Small'
                WHEN COALESCE((SELECT COUNT(*) FROM users), 0) < 10000 THEN '🟢 Medium'
                ELSE '🔵 Large'
            END
        WHEN t.table_name = 'strategic_account_plans' THEN 
            CASE 
                WHEN COALESCE((SELECT COUNT(*) FROM strategic_account_plans), 0) = 0 THEN '🔴 Empty'
                WHEN COALESCE((SELECT COUNT(*) FROM strategic_account_plans), 0) < 100 THEN '🟡 Small'
                WHEN COALESCE((SELECT COUNT(*) FROM strategic_account_plans), 0) < 10000 THEN '🟢 Medium'
                ELSE '🔵 Large'
            END
        ELSE '❓ Unknown'
    END as size_category
FROM information_schema.tables t
WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
ORDER BY t.table_name;

-- =====================================================
-- 3. COMPLETE COLUMN INFORMATION FOR ALL TABLES
-- =====================================================
SELECT 
    '🏗️ COLUMN DETAILS' as section,
    c.table_name,
    c.ordinal_position as pos,
    c.column_name,
    c.data_type,
    CASE 
        WHEN c.character_maximum_length IS NOT NULL 
        THEN c.data_type || '(' || c.character_maximum_length || ')'
        WHEN c.numeric_precision IS NOT NULL AND c.numeric_scale IS NOT NULL
        THEN c.data_type || '(' || c.numeric_precision || ',' || c.numeric_scale || ')'
        WHEN c.numeric_precision IS NOT NULL
        THEN c.data_type || '(' || c.numeric_precision || ')'
        ELSE c.data_type
    END as full_type,
    CASE 
        WHEN c.is_nullable = 'YES' THEN '✅ NULL'
        ELSE '❌ NOT NULL'
    END as nullable,
    COALESCE(c.column_default, '(none)') as default_value,
    CASE 
        WHEN pk.column_name IS NOT NULL THEN '🔑 PK'
        WHEN fk.column_name IS NOT NULL THEN '🔗 FK'
        WHEN uk.column_name IS NOT NULL THEN '🆔 UK'
        ELSE ''
    END as key_type
FROM information_schema.columns c
LEFT JOIN (
    -- Primary Keys
    SELECT kcu.table_name, kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu 
        ON tc.constraint_name = kcu.constraint_name
    WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = 'public'
) pk ON c.table_name = pk.table_name AND c.column_name = pk.column_name
LEFT JOIN (
    -- Foreign Keys
    SELECT kcu.table_name, kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu 
        ON tc.constraint_name = kcu.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
) fk ON c.table_name = fk.table_name AND c.column_name = fk.column_name
LEFT JOIN (
    -- Unique Keys
    SELECT kcu.table_name, kcu.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu 
        ON tc.constraint_name = kcu.constraint_name
    WHERE tc.constraint_type = 'UNIQUE' AND tc.table_schema = 'public'
) uk ON c.table_name = uk.table_name AND c.column_name = uk.column_name
WHERE c.table_schema = 'public'
ORDER BY c.table_name, c.ordinal_position;

-- =====================================================
-- 4. ALL INDEXES WITH DETAILED INFORMATION
-- =====================================================
SELECT 
    '🔍 INDEX ANALYSIS' as section,
    n.nspname as schema_name,
    t.relname as table_name,
    i.relname as index_name,
    CASE 
        WHEN ix.indisunique THEN '🆔 UNIQUE'
        WHEN ix.indisprimary THEN '🔑 PRIMARY'
        WHEN am.amname = 'gin' THEN '� GIN'
        WHEN am.amname = 'gist' THEN '🗺️ GIST'
        WHEN am.amname = 'hash' THEN '#️⃣ HASH'
        ELSE '📊 BTREE'
    END as index_type,
    pg_get_indexdef(i.oid) as index_definition
FROM pg_class t
JOIN pg_index ix ON t.oid = ix.indrelid
JOIN pg_class i ON i.oid = ix.indexrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
JOIN pg_am am ON i.relam = am.oid
WHERE n.nspname = 'public'
ORDER BY t.relname, i.relname;

-- =====================================================
-- 5. FOREIGN KEY RELATIONSHIPS
-- =====================================================
SELECT 
    '🔗 FOREIGN KEY RELATIONSHIPS' as section,
    tc.table_name as from_table,
    kcu.column_name as from_column,
    ccu.table_name as to_table,
    ccu.column_name as to_column,
    tc.constraint_name,
    CASE 
        WHEN rc.delete_rule = 'CASCADE' THEN '🗑️ CASCADE'
        WHEN rc.delete_rule = 'SET NULL' THEN '🔄 SET NULL'
        WHEN rc.delete_rule = 'RESTRICT' THEN '🚫 RESTRICT'
        ELSE rc.delete_rule
    END as on_delete,
    CASE 
        WHEN rc.update_rule = 'CASCADE' THEN '🔄 CASCADE'
        WHEN rc.update_rule = 'RESTRICT' THEN '🚫 RESTRICT'
        ELSE rc.update_rule
    END as on_update
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
LEFT JOIN information_schema.referential_constraints AS rc
    ON tc.constraint_name = rc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' 
    AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;

-- =====================================================
-- 6. TABLE CONSTRAINTS SUMMARY
-- =====================================================
SELECT 
    '⚖️ CONSTRAINTS SUMMARY' as section,
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    CASE 
        WHEN tc.constraint_type = 'PRIMARY KEY' THEN '🔑'
        WHEN tc.constraint_type = 'FOREIGN KEY' THEN '🔗'
        WHEN tc.constraint_type = 'UNIQUE' THEN '🆔'
        WHEN tc.constraint_type = 'CHECK' THEN '✅'
        ELSE '❓'
    END as icon,
    STRING_AGG(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) as columns
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
WHERE tc.table_schema = 'public'
GROUP BY tc.table_name, tc.constraint_name, tc.constraint_type
ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name;

-- =====================================================
-- 7. USERS TABLE SPECIFIC ANALYSIS (if exists)
-- =====================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public') THEN
        -- Users table exists, show detailed analysis
        RAISE NOTICE '👥 USERS TABLE ANALYSIS';
        
        -- Show users table structure
        PERFORM 1;
    ELSE
        RAISE NOTICE '❌ Users table does not exist';
    END IF;
END $$;

-- Users table detailed analysis (only if table exists)
SELECT 
    '👥 USERS TABLE STRUCTURE' as section,
    column_name,
    data_type,
    CASE 
        WHEN character_maximum_length IS NOT NULL 
        THEN data_type || '(' || character_maximum_length || ')'
        ELSE data_type
    END as full_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'users' AND table_schema = 'public'
ORDER BY ordinal_position;

-- =====================================================
-- 8. SAMPLE DATA FROM KEY TABLES (First 3 rows)
-- =====================================================

-- Check if users table has data
DO $$
DECLARE
    user_count INTEGER;
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public') THEN
        SELECT COUNT(*) INTO user_count FROM users;
        RAISE NOTICE '👥 Users table has % records', user_count;
        
        IF user_count > 0 THEN
            RAISE NOTICE '📄 Sample user data available';
        END IF;
    END IF;
END $$;

-- =====================================================
-- 9. DATABASE SIZE AND PERFORMANCE METRICS
-- =====================================================
SELECT 
    '💾 DATABASE SIZE ANALYSIS' as section,
    t.table_name,
    pg_size_pretty(pg_total_relation_size(quote_ident(t.table_name))) as total_size,
    pg_size_pretty(pg_relation_size(quote_ident(t.table_name))) as table_size,
    pg_size_pretty(pg_total_relation_size(quote_ident(t.table_name)) - pg_relation_size(quote_ident(t.table_name))) as index_size,
    ROUND(
        100.0 * (pg_total_relation_size(quote_ident(t.table_name)) - pg_relation_size(quote_ident(t.table_name))) 
        / NULLIF(pg_total_relation_size(quote_ident(t.table_name)), 0), 2
    ) as index_ratio_percent
FROM information_schema.tables t
WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
ORDER BY pg_total_relation_size(quote_ident(t.table_name)) DESC;

-- =====================================================
-- 10. ONBOARDING AGENT REQUIREMENTS CHECK
-- =====================================================
SELECT 
    '🎯 ONBOARDING REQUIREMENTS CHECK' as section,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public')
        THEN '✅ Users table exists'
        ELSE '❌ Users table missing'
    END as users_table_status,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'profile' AND table_schema = 'public')
        THEN '✅ Profile column exists'
        ELSE '❌ Profile column missing'
    END as profile_column_status,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'tenant_id' AND table_schema = 'public')
        THEN '✅ Multi-tenant support'
        ELSE '❌ No tenant support'
    END as tenant_support_status,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'reports_to' AND table_schema = 'public')
        THEN '✅ Hierarchy support'
        ELSE '❌ No hierarchy support'
    END as hierarchy_support_status;

-- =====================================================
-- 11. REQUIRED COLUMNS CHECK FOR USERS TABLE
-- =====================================================
WITH required_columns AS (
    SELECT unnest(ARRAY[
        'user_id', 'username', 'email', 'tenant_id', 'profile', 
        'reports_to', 'is_activated', 'password', 'access_token', 
        'refresh_token', 'expiration_date', 'created_at', 'updated_at'
    ]) as column_name
),
existing_columns AS (
    SELECT column_name
    FROM information_schema.columns 
    WHERE table_name = 'users' AND table_schema = 'public'
)
SELECT 
    '📋 REQUIRED COLUMNS STATUS' as section,
    rc.column_name,
    CASE 
        WHEN ec.column_name IS NOT NULL THEN '✅ EXISTS'
        ELSE '❌ MISSING'
    END as status,
    CASE 
        WHEN ec.column_name IS NOT NULL THEN 
            (SELECT data_type FROM information_schema.columns 
             WHERE table_name = 'users' AND column_name = rc.column_name AND table_schema = 'public')
        ELSE 'N/A'
    END as data_type
FROM required_columns rc
LEFT JOIN existing_columns ec ON rc.column_name = ec.column_name
ORDER BY 
    CASE WHEN ec.column_name IS NOT NULL THEN 0 ELSE 1 END,
    rc.column_name;

-- =====================================================
-- 12. SUMMARY REPORT
-- =====================================================
WITH summary_stats AS (
    SELECT 
        COUNT(*) as total_tables,
        COUNT(CASE WHEN table_type = 'BASE TABLE' THEN 1 END) as base_tables,
        COUNT(CASE WHEN table_type = 'VIEW' THEN 1 END) as views
    FROM information_schema.tables 
    WHERE table_schema = 'public'
),
index_stats AS (
    SELECT COUNT(*) as total_indexes
    FROM pg_indexes 
    WHERE schemaname = 'public'
),
constraint_stats AS (
    SELECT 
        COUNT(*) as total_constraints,
        COUNT(CASE WHEN constraint_type = 'PRIMARY KEY' THEN 1 END) as primary_keys,
        COUNT(CASE WHEN constraint_type = 'FOREIGN KEY' THEN 1 END) as foreign_keys,
        COUNT(CASE WHEN constraint_type = 'UNIQUE' THEN 1 END) as unique_constraints
    FROM information_schema.table_constraints
    WHERE table_schema = 'public'
)
SELECT 
    '📊 DATABASE SUMMARY' as section,
    ss.total_tables,
    ss.base_tables,
    ss.views,
    idx.total_indexes,
    cs.total_constraints,
    cs.primary_keys,
    cs.foreign_keys,
    cs.unique_constraints,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public')
        THEN '✅ Ready for Onboarding'
        ELSE '❌ Needs Setup'
    END as onboarding_ready
FROM summary_stats ss
CROSS JOIN index_stats idx
CROSS JOIN constraint_stats cs;

-- =====================================================
-- END OF ANALYSIS
-- =====================================================
SELECT '🎉 DATABASE ANALYSIS COMPLETE!' as final_message;