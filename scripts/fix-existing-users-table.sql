-- =====================================================
-- Fix Existing Users Table for Onboarding Agent
-- Adds missing columns and indexes without breaking existing data
-- =====================================================

-- Add missing columns if they don't exist
DO $$
BEGIN
    -- Add profile column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'profile' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN profile JSONB;
        RAISE NOTICE '✅ Added profile column';
    ELSE
        RAISE NOTICE '✅ Profile column already exists';
    END IF;

    -- Add tenant_id column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'tenant_id' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN tenant_id INTEGER DEFAULT 1300;
        RAISE NOTICE '✅ Added tenant_id column';
    ELSE
        RAISE NOTICE '✅ Tenant_id column already exists';
    END IF;

    -- Add reports_to column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'reports_to' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN reports_to BIGINT;
        RAISE NOTICE '✅ Added reports_to column';
    ELSE
        RAISE NOTICE '✅ Reports_to column already exists';
    END IF;

    -- Add is_activated column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'is_activated' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN is_activated BOOLEAN DEFAULT TRUE;
        RAISE NOTICE '✅ Added is_activated column';
    ELSE
        RAISE NOTICE '✅ Is_activated column already exists';
    END IF;

    -- Add access_token column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'access_token' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN access_token TEXT;
        RAISE NOTICE '✅ Added access_token column';
    ELSE
        RAISE NOTICE '✅ Access_token column already exists';
    END IF;

    -- Add refresh_token column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'refresh_token' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN refresh_token TEXT;
        RAISE NOTICE '✅ Added refresh_token column';
    ELSE
        RAISE NOTICE '✅ Refresh_token column already exists';
    END IF;

    -- Add expiration_date column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'expiration_date' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN expiration_date DATE;
        RAISE NOTICE '✅ Added expiration_date column';
    ELSE
        RAISE NOTICE '✅ Expiration_date column already exists';
    END IF;

    -- Add created_at column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'created_at' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        RAISE NOTICE '✅ Added created_at column';
    ELSE
        RAISE NOTICE '✅ Created_at column already exists';
    END IF;

    -- Add updated_at column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'updated_at' AND table_schema = 'public') THEN
        ALTER TABLE users ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        RAISE NOTICE '✅ Added updated_at column';
    ELSE
        RAISE NOTICE '✅ Updated_at column already exists';
    END IF;

END $$;

-- Create indexes if they don't exist
DO $$
BEGIN
    -- Index on email
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_users_email') THEN
        CREATE INDEX idx_users_email ON users(email);
        RAISE NOTICE '✅ Created email index';
    ELSE
        RAISE NOTICE '✅ Email index already exists';
    END IF;

    -- Index on tenant_id
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_users_tenant_id') THEN
        CREATE INDEX idx_users_tenant_id ON users(tenant_id);
        RAISE NOTICE '✅ Created tenant_id index';
    ELSE
        RAISE NOTICE '✅ Tenant_id index already exists';
    END IF;

    -- Index on reports_to
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_users_reports_to') THEN
        CREATE INDEX idx_users_reports_to ON users(reports_to);
        RAISE NOTICE '✅ Created reports_to index';
    ELSE
        RAISE NOTICE '✅ Reports_to index already exists';
    END IF;

    -- GIN index on profile (only if profile column exists)
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'profile' AND table_schema = 'public') THEN
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_users_profile') THEN
            CREATE INDEX idx_users_profile ON users USING GIN(profile);
            RAISE NOTICE '✅ Created profile GIN index';
        ELSE
            RAISE NOTICE '✅ Profile GIN index already exists';
        END IF;
    END IF;

    -- Composite index on tenant_id and email
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_users_tenant_email') THEN
        CREATE INDEX idx_users_tenant_email ON users(tenant_id, email);
        RAISE NOTICE '✅ Created tenant_email composite index';
    ELSE
        RAISE NOTICE '✅ Tenant_email composite index already exists';
    END IF;

END $$;

-- Add foreign key constraint for reports_to (if it doesn't exist)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name = 'users' 
        AND constraint_type = 'FOREIGN KEY' 
        AND constraint_name = 'fk_users_reports_to'
        AND table_schema = 'public'
    ) THEN
        -- Only add if reports_to column exists
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'reports_to' AND table_schema = 'public') THEN
            ALTER TABLE users ADD CONSTRAINT fk_users_reports_to 
            FOREIGN KEY (reports_to) REFERENCES users(user_id) ON DELETE SET NULL;
            RAISE NOTICE '✅ Added reports_to foreign key constraint';
        END IF;
    ELSE
        RAISE NOTICE '✅ Reports_to foreign key constraint already exists';
    END IF;
EXCEPTION
    WHEN others THEN
        RAISE NOTICE '⚠️ Could not add foreign key constraint (this is OK if there are data integrity issues)';
END $$;

-- Update existing users with default values for new columns
DO $$
BEGIN
    -- Set default tenant_id for users without one
    UPDATE users SET tenant_id = 1300 WHERE tenant_id IS NULL;
    
    -- Set default is_activated for users without one
    UPDATE users SET is_activated = TRUE WHERE is_activated IS NULL;
    
    -- Set default created_at for users without one
    UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL;
    
    -- Set default updated_at for users without one
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL;
    
    RAISE NOTICE '✅ Updated existing users with default values';
END $$;

-- Create tenants table if it doesn't exist
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id INTEGER PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default tenant if it doesn't exist
INSERT INTO tenants (tenant_id, tenant_name, domain, settings, is_active)
VALUES (1300, 'Default Tenant', 'default.local', '{"modules": ["Forecasting", "Planning", "Pipeline", "Analytics"]}', TRUE)
ON CONFLICT (tenant_id) DO NOTHING;

-- Create function for automatic timestamp updates if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for automatic timestamp updates if it doesn't exist
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Verification
SELECT 
    '✅ USERS TABLE FIXED' as status,
    COUNT(*) as total_users,
    COUNT(CASE WHEN profile IS NOT NULL THEN 1 END) as users_with_profile,
    COUNT(CASE WHEN tenant_id IS NOT NULL THEN 1 END) as users_with_tenant
FROM users;

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
    AND table_schema = 'public'
ORDER BY ordinal_position;

SELECT '🎉 USERS TABLE READY FOR ONBOARDING AGENT!' as final_message;