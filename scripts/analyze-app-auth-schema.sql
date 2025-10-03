-- =====================================================
-- Analyze app_auth Schema Database Structure
-- Corrected for your actual schema structure
-- =====================================================

-- 1. Database and schema info
SELECT 
    '🗄️ DATABASE INFO' as section,
    current_database() as database_name,
    current_user as current_user,
    'app_auth' as target_schema;

-- 2. All tables in app_auth schema
SELECT 
    '📋 APP_AUTH SCHEMA TABLES' as section,
    table_name,
    table_type,
    CASE 
        WHEN table_type = 'BASE TABLE' THEN '📊 Table'
        WHEN table_type = 'VIEW' THEN '👁️ View'
        ELSE '❓ Other'
    END as type_icon
FROM information_schema.tables 
WHERE table_schema = 'app_auth'
ORDER BY table_name;

-- 3. Users table structure
SELECT 
    '👥 USERS TABLE COLUMNS' as section,
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
    COALESCE(column_default, '(none)') as default_value,
    ordinal_position as pos
FROM information_schema.columns 
WHERE table_name = 'users' 
    AND table_schema = 'app_auth'
ORDER BY ordinal_position;

-- 4. Users table constraints
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
    AND table_schema = 'app_auth'
ORDER BY constraint_type;

-- 5. Users table indexes
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
    AND schemaname = 'app_auth'
ORDER BY indexname;

-- 6. Users table row count and sample data
SELECT 
    '📊 USERS TABLE DATA' as section,
    COUNT(*) as total_users,
    COUNT(CASE WHEN email IS NOT NULL THEN 1 END) as users_with_email,
    MIN(created_at) as oldest_user,
    MAX(created_at) as newest_user
FROM app_auth.users;

-- 7. Sample users (first 3)
SELECT 
    '👥 SAMPLE USERS' as section,
    user_id,
    username,
    email,
    CASE 
        WHEN created_at IS NOT NULL THEN created_at::text
        ELSE 'No timestamp'
    END as created_at
FROM app_auth.users 
ORDER BY 
    CASE WHEN created_at IS NOT NULL THEN created_at ELSE '1900-01-01'::timestamp END DESC
LIMIT 3;

-- 8. Foreign key relationships
SELECT 
    '🔗 FOREIGN KEY RELATIONSHIPS' as section,
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
    AND tc.table_schema = 'app_auth'
ORDER BY tc.table_name, kcu.column_name;

-- 9. Check for onboarding-required columns
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
    WHERE table_name = 'users' AND table_schema = 'app_auth'
)
SELECT 
    '📋 ONBOARDING COLUMNS CHECK' as section,
    rc.column_name as required_column,
    CASE 
        WHEN ec.column_name IS NOT NULL THEN '✅ EXISTS'
        ELSE '❌ MISSING'
    END as status,
    CASE 
        WHEN ec.column_name IS NOT NULL THEN 
            (SELECT data_type FROM information_schema.columns 
             WHERE table_name = 'users' AND column_name = rc.column_name AND table_schema = 'app_auth')
        ELSE 'N/A'
    END as data_type
FROM required_columns rc
LEFT JOIN existing_columns ec ON rc.column_name = ec.column_name
ORDER BY 
    CASE WHEN ec.column_name IS NOT NULL THEN 0 ELSE 1 END,
    rc.column_name;

-- 10. Related tables analysis
SELECT 
    '🔗 RELATED TABLES' as section,
    table_name,
    (SELECT COUNT(*) 
     FROM information_schema.columns 
     WHERE table_name = t.table_name AND table_schema = 'app_auth') as column_count,
    CASE 
        WHEN table_name = 'users' THEN '👥 Main users table'
        WHEN table_name = 'user_sessions' THEN '🔐 User sessions'
        WHEN table_name = 'users_license' THEN '📄 User licenses'
        WHEN table_name = 'users_role' THEN '👤 User roles'
        WHEN table_name = 'roles' THEN '🎭 Role definitions'
        WHEN table_name = 'tenant_registry' THEN '🏢 Tenant management'
        ELSE '📊 Other table'
    END as description
FROM information_schema.tables t
WHERE table_schema = 'app_auth' 
    AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- 11. Onboarding readiness assessment
SELECT 
    '🎯 ONBOARDING READINESS' as section,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'profile' AND table_schema = 'app_auth')
        THEN '✅ Profile column exists'
        ELSE '❌ Profile column missing'
    END as profile_status,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'tenant_id' AND table_schema = 'app_auth')
        THEN '✅ Multi-tenant ready'
        ELSE '❌ No tenant support'
    END as tenant_status,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'reports_to' AND table_schema = 'app_auth')
        THEN '✅ Hierarchy support'
        ELSE '❌ No hierarchy support'
    END as hierarchy_status,
    CASE 
        WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tenant_registry' AND table_schema = 'app_auth')
        THEN '✅ Tenant registry exists'
        ELSE '❌ No tenant registry'
    END as tenant_registry_status;

-- 12. Summary
SELECT 
    '📊 SUMMARY' as section,
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'app_auth' AND table_type = 'BASE TABLE') as total_tables,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'app_auth') as total_columns,
    (SELECT COUNT(*) FROM app_auth.users) as total_users,
    'app_auth schema' as schema_location;

SELECT '🎉 APP_AUTH SCHEMA ANALYSIS COMPLETE!' as final_message;