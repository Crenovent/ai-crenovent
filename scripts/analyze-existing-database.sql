-- =====================================================
-- Analyze Existing Database Structure
-- Check what already exists and what needs to be fixed
-- =====================================================

-- 1. Check if users table exists and show its structure
SELECT 
    '👥 USERS TABLE EXISTS' as status,
    'Checking structure...' as message;

-- Show users table columns
SELECT 
    '🏗️ USERS TABLE COLUMNS' as section,
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

-- Check existing constraints on users table
SELECT 
    '⚖️ USERS TABLE CONSTRAINTS' as section,
    constraint_name,
    constraint_type,
    CASE 
        WHEN constraint_type = 'PRIMARY KEY' THEN '🔑'
        WHEN constraint_type = 'FOREIGN KEY' THEN '🔗'
        WHEN constraint_type = 'UNIQUE' THEN '🆔'
        WHEN constraint_type = 'CHECK' THEN '✅'
        ELSE '❓'
    END as icon
FROM information_schema.table_constraints 
WHERE table_name = 'users' 
    AND table_schema = 'public'
ORDER BY constraint_type;

-- Check existing indexes on users table
SELECT 
    '🔍 USERS TABLE INDEXES' as section,
    indexname as index_name,
    CASE 
        WHEN indexdef LIKE '%UNIQUE%' THEN '🆔 UNIQUE'
        WHEN indexdef LIKE '%PRIMARY%' THEN '🔑 PRIMARY'
        WHEN indexdef LIKE '%gin%' THEN '🌲 GIN'
        ELSE '📊 BTREE'
    END as index_type
FROM pg_indexes 
WHERE tablename = 'users' 
    AND schemaname = 'public'
ORDER BY indexname;

-- Count rows in users table
SELECT 
    '📊 USERS TABLE DATA' as section,
    COUNT(*) as total_users,
    COUNT(CASE WHEN tenant_id IS NOT NULL THEN 1 END) as users_with_tenant,
    COUNT(CASE WHEN profile IS NOT NULL THEN 1 END) as users_with_profile,
    COUNT(CASE WHEN reports_to IS NOT NULL THEN 1 END) as users_with_manager
FROM users;

-- Check if other tables exist
SELECT 
    '📋 OTHER TABLES' as section,
    table_name,
    CASE 
        WHEN table_name = 'tenants' THEN '🏢'
        WHEN table_name = 'user_roles' THEN '👤'
        WHEN table_name = 'strategic_account_plans' THEN '📊'
        ELSE '📄'
    END as icon
FROM information_schema.tables 
WHERE table_schema = 'public' 
    AND table_type = 'BASE TABLE'
    AND table_name != 'users'
ORDER BY table_name;

-- Check sample user data (first 3 users)
SELECT 
    '👥 SAMPLE USERS' as section,
    user_id,
    username,
    email,
    tenant_id,
    reports_to,
    is_activated,
    CASE 
        WHEN profile IS NOT NULL THEN 'Has Profile'
        ELSE 'No Profile'
    END as profile_status,
    created_at
FROM users 
ORDER BY created_at DESC 
LIMIT 3;

-- Check for missing required columns
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
    '📋 COLUMN STATUS' as section,
    rc.column_name,
    CASE 
        WHEN ec.column_name IS NOT NULL THEN '✅ EXISTS'
        ELSE '❌ MISSING'
    END as status
FROM required_columns rc
LEFT JOIN existing_columns ec ON rc.column_name = ec.column_name
ORDER BY 
    CASE WHEN ec.column_name IS NOT NULL THEN 0 ELSE 1 END,
    rc.column_name;

-- Check for required indexes
WITH required_indexes AS (
    SELECT unnest(ARRAY[
        'idx_users_email', 'idx_users_tenant_id', 'idx_users_reports_to', 
        'idx_users_profile', 'idx_users_tenant_email'
    ]) as index_name
),
existing_indexes AS (
    SELECT indexname as index_name
    FROM pg_indexes 
    WHERE tablename = 'users' AND schemaname = 'public'
)
SELECT 
    '🔍 INDEX STATUS' as section,
    ri.index_name,
    CASE 
        WHEN ei.index_name IS NOT NULL THEN '✅ EXISTS'
        ELSE '❌ MISSING'
    END as status
FROM required_indexes ri
LEFT JOIN existing_indexes ei ON ri.index_name = ei.index_name
ORDER BY 
    CASE WHEN ei.index_name IS NOT NULL THEN 0 ELSE 1 END,
    ri.index_name;

-- Final assessment
SELECT 
    '🎯 ONBOARDING READINESS' as section,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'profile' AND table_schema = 'public')
        THEN '✅ Profile column exists'
        ELSE '❌ Profile column missing'
    END as profile_check,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'tenant_id' AND table_schema = 'public')
        THEN '✅ Multi-tenant ready'
        ELSE '❌ No tenant support'
    END as tenant_check,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'reports_to' AND table_schema = 'public')
        THEN '✅ Hierarchy support'
        ELSE '❌ No hierarchy support'
    END as hierarchy_check;

SELECT '🎉 ANALYSIS COMPLETE!' as final_message;