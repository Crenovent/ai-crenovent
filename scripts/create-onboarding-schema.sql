-- =====================================================
-- Create Database Schema for Onboarding Agent
-- This script creates all required tables and indexes
-- =====================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- 1. CREATE USERS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    tenant_id INTEGER NOT NULL,
    reports_to BIGINT,
    is_activated BOOLEAN DEFAULT TRUE,
    profile JSONB,
    password VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    expiration_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint for self-referencing reports_to
    CONSTRAINT fk_users_reports_to FOREIGN KEY (reports_to) REFERENCES users(user_id) ON DELETE SET NULL
);

-- =====================================================
-- 2. CREATE INDEXES FOR PERFORMANCE
-- =====================================================

-- Index on email for fast lookups during login/authentication
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Index on tenant_id for multi-tenant queries
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);

-- Index on reports_to for hierarchy queries
CREATE INDEX IF NOT EXISTS idx_users_reports_to ON users(reports_to);

-- GIN index on profile JSONB for fast JSON queries
CREATE INDEX IF NOT EXISTS idx_users_profile ON users USING GIN(profile);

-- Composite index for tenant + email lookups
CREATE INDEX IF NOT EXISTS idx_users_tenant_email ON users(tenant_id, email);

-- Index on username for search functionality
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Index on is_activated for filtering active users
CREATE INDEX IF NOT EXISTS idx_users_activated ON users(is_activated);

-- =====================================================
-- 3. CREATE TENANTS TABLE (Optional but Recommended)
-- =====================================================
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id INTEGER PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index on tenant domain
CREATE INDEX IF NOT EXISTS idx_tenants_domain ON tenants(domain);

-- =====================================================
-- 4. CREATE USER ROLES TABLE (Optional)
-- =====================================================
CREATE TABLE IF NOT EXISTS user_roles (
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(100) NOT NULL,
    description TEXT,
    permissions JSONB DEFAULT '{}',
    tenant_id INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key to tenants table
    CONSTRAINT fk_user_roles_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    
    -- Unique constraint on role name per tenant
    CONSTRAINT uk_user_roles_name_tenant UNIQUE (role_name, tenant_id)
);

-- =====================================================
-- 5. CREATE USER ROLE ASSIGNMENTS TABLE (Optional)
-- =====================================================
CREATE TABLE IF NOT EXISTS user_role_assignments (
    assignment_id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    role_id INTEGER NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assigned_by BIGINT,
    
    -- Foreign keys
    CONSTRAINT fk_assignments_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT fk_assignments_role FOREIGN KEY (role_id) REFERENCES user_roles(role_id) ON DELETE CASCADE,
    CONSTRAINT fk_assignments_assigned_by FOREIGN KEY (assigned_by) REFERENCES users(user_id) ON DELETE SET NULL,
    
    -- Unique constraint to prevent duplicate assignments
    CONSTRAINT uk_user_role_assignments UNIQUE (user_id, role_id)
);

-- =====================================================
-- 6. CREATE STRATEGIC ACCOUNT PLANS TABLE (if not exists)
-- =====================================================
CREATE TABLE IF NOT EXISTS strategic_account_plans (
    plan_id SERIAL PRIMARY KEY,
    plan_name VARCHAR(255),
    account_id VARCHAR(255),
    annual_revenue DECIMAL(15,2),
    account_tier VARCHAR(50),
    short_term_goals TEXT,
    long_term_goals TEXT,
    key_opportunities TEXT,
    known_risks TEXT,
    stakeholders TEXT,
    activities TEXT,
    created_by BIGINT,
    tenant_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    CONSTRAINT fk_plans_created_by FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE SET NULL,
    CONSTRAINT fk_plans_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);

-- Indexes for strategic account plans
CREATE INDEX IF NOT EXISTS idx_plans_account_id ON strategic_account_plans(account_id);
CREATE INDEX IF NOT EXISTS idx_plans_tenant_id ON strategic_account_plans(tenant_id);
CREATE INDEX IF NOT EXISTS idx_plans_created_by ON strategic_account_plans(created_by);

-- =====================================================
-- 7. CREATE AUDIT LOG TABLE (Optional but Recommended)
-- =====================================================
CREATE TABLE IF NOT EXISTS audit_log (
    log_id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id VARCHAR(100) NOT NULL,
    action VARCHAR(20) NOT NULL, -- INSERT, UPDATE, DELETE
    old_values JSONB,
    new_values JSONB,
    changed_by BIGINT,
    tenant_id INTEGER,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    CONSTRAINT fk_audit_changed_by FOREIGN KEY (changed_by) REFERENCES users(user_id) ON DELETE SET NULL
);

-- Indexes for audit log
CREATE INDEX IF NOT EXISTS idx_audit_table_record ON audit_log(table_name, record_id);
CREATE INDEX IF NOT EXISTS idx_audit_changed_by ON audit_log(changed_by);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_id ON audit_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_changed_at ON audit_log(changed_at);

-- =====================================================
-- 8. INSERT DEFAULT TENANT (for testing)
-- =====================================================
INSERT INTO tenants (tenant_id, tenant_name, domain, settings, is_active)
VALUES (1300, 'Default Tenant', 'default.local', '{"modules": ["Forecasting", "Planning", "Pipeline", "Analytics"]}', TRUE)
ON CONFLICT (tenant_id) DO NOTHING;

-- =====================================================
-- 9. CREATE SAMPLE ADMIN USER (for testing)
-- =====================================================
INSERT INTO users (
    user_id, 
    username, 
    email, 
    tenant_id, 
    reports_to, 
    is_activated, 
    profile, 
    password, 
    access_token, 
    refresh_token, 
    expiration_date
) VALUES (
    1001,
    'System Administrator',
    'admin@crenovent.com',
    1300,
    NULL,
    TRUE,
    '{"role_title": "System Administrator", "department": "IT", "level": "Admin", "modules": "Forecasting,Planning,Pipeline,Analytics", "permissions": "admin", "region": "Global", "segment": "Internal"}',
    '$2b$12$LQv3c1yqBwlVHpPjrEyeye.svHgOcjsVmy/WSyUCLNQiOzHEvFa6i', -- password: admin123
    'temp_access_1001',
    'temp_refresh_1001',
    CURRENT_DATE + INTERVAL '1 year'
) ON CONFLICT (user_id) DO NOTHING;

-- =====================================================
-- 10. CREATE FUNCTIONS FOR AUTOMATIC TIMESTAMPS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tenants_updated_at 
    BEFORE UPDATE ON tenants 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_plans_updated_at 
    BEFORE UPDATE ON strategic_account_plans 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- 11. VERIFICATION QUERIES
-- =====================================================

-- Check if tables were created successfully
SELECT 
    '✅ SCHEMA CREATION COMPLETE' as status,
    COUNT(*) as tables_created
FROM information_schema.tables 
WHERE table_schema = 'public' 
    AND table_name IN ('users', 'tenants', 'user_roles', 'strategic_account_plans');

-- Check indexes
SELECT 
    '🔍 INDEXES CREATED' as status,
    COUNT(*) as indexes_created
FROM pg_indexes 
WHERE schemaname = 'public' 
    AND indexname LIKE 'idx_%';

-- Check sample data
SELECT 
    '👥 SAMPLE DATA' as status,
    (SELECT COUNT(*) FROM users) as user_count,
    (SELECT COUNT(*) FROM tenants) as tenant_count;

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================
SELECT '🎉 DATABASE SCHEMA CREATED SUCCESSFULLY!' as final_message,
       'You can now run the onboarding agent!' as next_step;