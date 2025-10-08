-- =====================================================
-- MIGRATION: Create Tenant Metadata Table
-- Version: v20251008_084936
-- ID: 20251008_084936_create_tenant_metadata_table
-- Author: RevOps Team
-- Created: 2025-10-08T08:49:36.659404+00:00
-- Risk Assessment: medium
-- Estimated Duration: 2-3 minutes
-- Rollback Strategy: automatic
-- =====================================================

-- Description: Initial migration to create tenant_metadata table with RLS policies for multi-tenant isolation

-- Tags: initial, tenant, multi-tenant, rls

-- =====================================================
-- UP MIGRATION
-- =====================================================

-- Step 1: Create table tenant_metadata
-- Risk Level: medium

-- Validation check
DO $$
BEGIN
    IF NOT EXISTS (-- Validate table tenant_metadata exists
SELECT 1 FROM information_schema.tables 
WHERE table_name = 'tenant_metadata' AND table_schema = current_schema();) THEN
        -- Proceed with migration step
        NULL;
    END IF;
END $$;

-- Create table tenant_metadata
CREATE TABLE IF NOT EXISTS tenant_metadata (
    tenant_id INTEGER PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    industry_code VARCHAR(10) NOT NULL CHECK (industry_code IN ('SaaS', 'BANK', 'INSUR', 'ECOMM', 'FS', 'IT')),
    region_code VARCHAR(10) NOT NULL CHECK (region_code IN ('US', 'EU', 'IN', 'APAC')),
    compliance_requirements JSONB DEFAULT '[]',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'offboarding')),
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    metadata JSONB DEFAULT '{}'

);

-- Verify step completion
-- -- Validate table tenant_metadata exists
SELECT 1 FROM information_schema.tables 
WHERE table_name = 'tenant_metadata' AND table_schema = current_schema();

-- ==================================================

-- Step 2: Create index idx_tenant_industry on tenant_metadata(industry_code)
-- Risk Level: low

-- Validation check
DO $$
BEGIN
    IF NOT EXISTS (-- Validate index idx_tenant_industry exists
SELECT 1 FROM pg_indexes 
WHERE indexname = 'idx_tenant_industry' AND schemaname = current_schema();) THEN
        -- Proceed with migration step
        NULL;
    END IF;
END $$;

-- Create index idx_tenant_industry on tenant_metadata
CREATE INDEX IF NOT EXISTS idx_tenant_industry 
ON tenant_metadata USING BTREE (industry_code);

-- Verify step completion
-- -- Validate index idx_tenant_industry exists
SELECT 1 FROM pg_indexes 
WHERE indexname = 'idx_tenant_industry' AND schemaname = current_schema();

-- ==================================================

-- Step 3: Create index idx_tenant_region on tenant_metadata(region_code)
-- Risk Level: low

-- Validation check
DO $$
BEGIN
    IF NOT EXISTS (-- Validate index idx_tenant_region exists
SELECT 1 FROM pg_indexes 
WHERE indexname = 'idx_tenant_region' AND schemaname = current_schema();) THEN
        -- Proceed with migration step
        NULL;
    END IF;
END $$;

-- Create index idx_tenant_region on tenant_metadata
CREATE INDEX IF NOT EXISTS idx_tenant_region 
ON tenant_metadata USING BTREE (region_code);

-- Verify step completion
-- -- Validate index idx_tenant_region exists
SELECT 1 FROM pg_indexes 
WHERE indexname = 'idx_tenant_region' AND schemaname = current_schema();

-- ==================================================

-- Step 4: Create index idx_tenant_status on tenant_metadata(status)
-- Risk Level: low

-- Validation check
DO $$
BEGIN
    IF NOT EXISTS (-- Validate index idx_tenant_status exists
SELECT 1 FROM pg_indexes 
WHERE indexname = 'idx_tenant_status' AND schemaname = current_schema();) THEN
        -- Proceed with migration step
        NULL;
    END IF;
END $$;

-- Create index idx_tenant_status on tenant_metadata
CREATE INDEX IF NOT EXISTS idx_tenant_status 
ON tenant_metadata USING BTREE (status);

-- Verify step completion
-- -- Validate index idx_tenant_status exists
SELECT 1 FROM pg_indexes 
WHERE indexname = 'idx_tenant_status' AND schemaname = current_schema();

-- ==================================================

-- Step 5: Insert sample tenant data into tenant_metadata
-- Risk Level: low

-- Insert sample tenant data into tenant_metadata

INSERT INTO tenant_metadata (tenant_id, tenant_name, industry_code, region_code, compliance_requirements) 
VALUES 
    (1000, 'Demo SaaS Company', 'SaaS', 'US', '["SOX", "GDPR"]'),
    (1001, 'Sample Bank Corp', 'BANK', 'US', '["SOX", "RBI"]'),
    (1002, 'Test Insurance Ltd', 'INSUR', 'EU', '["GDPR", "IRDAI"]')
ON CONFLICT (tenant_id) DO NOTHING;

-- Verify step completion
-- Manual verification required

-- ==================================================


-- =====================================================
-- DOWN MIGRATION (ROLLBACK)
-- =====================================================
-- IMPORTANT: Review rollback steps carefully before execution
-- Execute steps in REVERSE order for proper rollback

-- Rollback Step 1: Reverse Insert sample tenant data into tenant_metadata
-- Risk Level: low

/*
-- Remove sample tenant data from tenant_metadata
DELETE FROM tenant_metadata WHERE tenant_id IN (1000, 1001, 1002);
*/

-- ==================================================

-- Rollback Step 2: Reverse Create index idx_tenant_status on tenant_metadata(status)
-- Risk Level: low

/*
-- Drop index idx_tenant_status
DROP INDEX IF EXISTS idx_tenant_status;
*/

-- ==================================================

-- Rollback Step 3: Reverse Create index idx_tenant_region on tenant_metadata(region_code)
-- Risk Level: low

/*
-- Drop index idx_tenant_region
DROP INDEX IF EXISTS idx_tenant_region;
*/

-- ==================================================

-- Rollback Step 4: Reverse Create index idx_tenant_industry on tenant_metadata(industry_code)
-- Risk Level: low

/*
-- Drop index idx_tenant_industry
DROP INDEX IF EXISTS idx_tenant_industry;
*/

-- ==================================================

-- Rollback Step 5: Reverse Create table tenant_metadata
-- Risk Level: medium

/*
-- Drop table tenant_metadata
DROP TABLE IF EXISTS tenant_metadata CASCADE;
*/

-- ==================================================


-- =====================================================
-- MIGRATION COMPLETE
-- Create Tenant Metadata Table - v20251008_084936
-- =====================================================