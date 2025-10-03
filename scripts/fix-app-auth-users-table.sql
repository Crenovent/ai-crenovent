-- =====================================================
-- Fix app_auth.users Table for Onboarding Agent
-- Adds missing columns and indexes to existing app_auth.users table
-- =====================================================

-- Add missing columns if they don't exist
DO $$
BEGIN
    -- Add profile column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'profile' AND table_schema = 'app_auth') THEN
        ALTER TABLE app_auth.users ADD COLUMN profile JSONB;
        RAISE NOTICE '✅ Added profile column to app_auth.users';
    ELSE
        RAISE NOTICE '✅ Profile column already exists in app_auth.users';
    END IF;

    -- Add tenant_id column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'tenant_id' AND table_schema = 'app_auth') THEN
        ALTER TABLE app_auth.users ADD COLUMN tenant_id INTEGER DEFAULT 1300;
        RAISE NOTICE '✅ Added tenant_id column to app_auth.users';
    ELSE
        RAISE NOTICE '✅ Tenant_id column already exists in app_auth.users';
    END IF;

    -- Add reports_to column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'reports_to' AND table_schema = 'app_auth') THEN
        ALTER TABLE app_auth.users ADD COLUMN reports_to BIGINT;
        RAISE NOTICE '✅ Added reports_to column to app_auth.users';
    ELSE
        RAISE NOTICE '✅ Reports_to column already exists in app_auth.users';
    END IF;

    -- Add is_activated column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'is_activated' AND table_schema = 'app_auth') THEN
        ALTER TABLE app_auth.users ADD COLUMN is_activated BOOLEAN DEFAULT TRUE;
        RAISE NOTICE '✅ Added is_activated column to app_auth.users';
    ELSE
        RAISE NOTICE '✅ Is_activated column already exists in app_auth.users';
    END IF;

    -- Add access_token column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'access_token' AND table_schema = 'app_auth') THEN
        ALTER TABLE app_auth.users ADD COLUMN access_token TEXT;
        RAISE NOTICE '✅ Added access_token column to app_auth.users';
    ELSE
        RAISE NOTICE '✅ Access_token column already exists in app_auth.users';
    END IF;

    -- Add refresh_token column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'refresh_token' AND table_schema = 'app_auth') THEN
        ALTER TABLE app_auth.users ADD COLUMN refresh_token TEXT;
        RAISE NOTICE '✅ Added refresh_token column to app_auth.users';
    ELSE
        RAISE NOTICE '✅ Refresh_token column already exists in app_auth.users';
    END IF;

    -- Add expiration_date column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'expiration_date' AND table_schema = 'app_auth') THEN
        ALTER TABLE app_auth.users ADD COLUMN expiration_date DATE;
        RAISE NOTICE '✅ Added expiration_date column to app_auth.users';
    ELSE
        RAISE NOTICE '✅ Expiration_date column already exists in app_auth.users';
    END IF;

    -- Add updated_at column if missing (created_at likely exists)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'updated_at' AND table_schema = 'app_auth') THEN
        ALTER TABLE app_auth.users ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        RAISE NOTICE '✅ Added updated_at column to app_auth.users';
    ELSE
        RAISE NOTICE '✅ Updated_at column already exists in app_auth.users';
    END IF;

END $$;

-- Create indexes if they don't exist
DO $$
BEGIN
    -- Index on email (might already exist)
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_app_auth_users_email' AND schemaname = 'app_auth') THEN
        CREATE INDEX idx_app_auth_users_email ON app_auth.users(email);
        RAISE NOTICE '✅ Created email index on app_auth.users';
    ELSE
        RAISE NOTICE '✅ Email index already exists on app_auth.users';
    END IF;

    -- Index on tenant_id
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_app_auth_users_tenant_id' AND schemaname = 'app_auth') THEN
        CREATE INDEX idx_app_auth_users_tenant_id ON app_auth.users(tenant_id);
        RAISE NOTICE '✅ Created tenant_id index on app_auth.users';
    ELSE
        RAISE NOTICE '✅ Tenant_id index already exists on app_auth.users';
    END IF;

    -- Index on reports_to
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_app_auth_users_reports_to' AND schemaname = 'app_auth') THEN
        CREATE INDEX idx_app_auth_users_reports_to ON app_auth.users(reports_to);
        RAISE NOTICE '✅ Created reports_to index on app_auth.users';
    ELSE
        RAISE NOTICE '✅ Reports_to index already exists on app_auth.users';
    END IF;

    -- GIN index on profile (only if profile column exists)
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'profile' AND table_schema = 'app_auth') THEN
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_app_auth_users_profile' AND schemaname = 'app_auth') THEN
            CREATE INDEX idx_app_auth_users_profile ON app_auth.users USING GIN(profile);
            RAISE NOTICE '✅ Created profile GIN index on app_auth.users';
        ELSE
            RAISE NOTICE '✅ Profile GIN index already exists on app_auth.users';
        END IF;
    END IF;

    -- Composite index on tenant_id and email
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_app_auth_users_tenant_email' AND schemaname = 'app_auth') THEN
        CREATE INDEX idx_app_auth_users_tenant_email ON app_auth.users(tenant_id, email);
        RAISE NOTICE '✅ Created tenant_email composite index on app_auth.users';
    ELSE
        RAISE NOTICE '✅ Tenant_email composite index already exists on app_auth.users';
    END IF;

END $$;

-- Add foreign key constraint for reports_to (if it doesn't exist)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name = 'users' 
        AND constraint_type = 'FOREIGN KEY' 
        AND constraint_name = 'fk_app_auth_users_reports_to'
        AND table_schema = 'app_auth'
    ) THEN
        -- Only add if reports_to column exists
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'reports_to' AND table_schema = 'app_auth') THEN
            ALTER TABLE app_auth.users ADD CONSTRAINT fk_app_auth_users_reports_to 
            FOREIGN KEY (reports_to) REFERENCES app_auth.users(user_id) ON DELETE SET NULL;
            RAISE NOTICE '✅ Added reports_to foreign key constraint to app_auth.users';
        END IF;
    ELSE
        RAISE NOTICE '✅ Reports_to foreign key constraint already exists on app_auth.users';
    END IF;
EXCEPTION
    WHEN others THEN
        RAISE NOTICE '⚠️ Could not add foreign key constraint (this is OK if there are data integrity issues)';
END $$;

-- Update existing users with default values for new columns
DO $$
BEGIN
    -- Set default tenant_id for users without one
    UPDATE app_auth.users SET tenant_id = 1300 WHERE tenant_id IS NULL;
    
    -- Set default is_activated for users without one
    UPDATE app_auth.users SET is_activated = TRUE WHERE is_activated IS NULL;
    
    -- Set default updated_at for users without one
    UPDATE app_auth.users SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL;
    
    RAISE NOTICE '✅ Updated existing users with default values';
END $$;

-- Create function for automatic timestamp updates if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for automatic timestamp updates if it doesn't exist
DROP TRIGGER IF EXISTS update_app_auth_users_updated_at ON app_auth.users;
CREATE TRIGGER update_app_auth_users_updated_at 
    BEFORE UPDATE ON app_auth.users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Verification
SELECT 
    '✅ APP_AUTH.USERS TABLE UPDATED' as status,
    COUNT(*) as total_users,
    COUNT(CASE WHEN profile IS NOT NULL THEN 1 END) as users_with_profile,
    COUNT(CASE WHEN tenant_id IS NOT NULL THEN 1 END) as users_with_tenant
FROM app_auth.users;

-- Show final column structure
SELECT 
    '🏗️ FINAL COLUMN STRUCTURE' as section,
    column_name,
    data_type,
    CASE 
        WHEN is_nullable = 'YES' THEN '✅ NULL'
        ELSE '❌ NOT NULL'
    END as nullable
FROM information_schema.columns 
WHERE table_name = 'users' 
    AND table_schema = 'app_auth'
ORDER BY ordinal_position;

SELECT '🎉 APP_AUTH.USERS TABLE READY FOR ONBOARDING AGENT!' as final_message;