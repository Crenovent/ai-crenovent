-- =====================================================
-- REVOPS ONTOLOGY DATABASE SCHEMA DDL
-- Generated on: 2025-10-08T08:46:23.812510+00:00
-- Schema: revops
-- Task 7.2-T44: Generate DDL (create table/index/rls)
-- =====================================================

-- Required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schema
CREATE SCHEMA IF NOT EXISTS revops;

-- =====================================================
-- TABLES
-- =====================================================

-- Multi-tenant isolation and metadata - Task 7.2-T04
CREATE TABLE IF NOT EXISTS revops.tenant_metadata (
    tenant_id INTEGER PRIMARY KEY NOT NULL
    tenant_name VARCHAR(255) NOT NULL
    industry_code VARCHAR(10) NOT NULL CHECK (industry_code IN ('SaaS', 'BANK', 'INSUR', 'ECOMM', 'FS', 'IT'))
    region_code VARCHAR(10) NOT NULL CHECK (region_code IN ('US', 'EU', 'IN', 'APAC'))
    compliance_requirements JSONB DEFAULT '[]'
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'offboarding'))
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    metadata JSONB DEFAULT '{}'
);

-- CRM canonical account entity - Task 7.2-T05
CREATE TABLE IF NOT EXISTS revops.accounts (
    account_id UUID PRIMARY KEY DEFAULT gen_random_uuid()
    tenant_id INTEGER NOT NULL
    external_id VARCHAR(255)
    account_name VARCHAR(255) NOT NULL
    account_type VARCHAR(50) CHECK (account_type IN ('prospect', 'customer', 'partner', 'competitor'))
    industry VARCHAR(100)
    annual_revenue DECIMAL(15,2)
    employee_count INTEGER
    website VARCHAR(255)
    billing_address JSONB
    shipping_address JSONB
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    created_by_user_id INTEGER
    is_deleted BOOLEAN DEFAULT false

    CONSTRAINT fk_accounts_tenant_id FOREIGN KEY (tenant_id) REFERENCES revops.tenant_metadata(tenant_id) ON DELETE CASCADE
);

-- Deal flow canonical entity - Task 7.2-T06
CREATE TABLE IF NOT EXISTS revops.opportunities (
    opportunity_id UUID PRIMARY KEY DEFAULT gen_random_uuid()
    tenant_id INTEGER NOT NULL
    account_id UUID NOT NULL
    external_id VARCHAR(255)
    opportunity_name VARCHAR(255) NOT NULL
    stage VARCHAR(100) NOT NULL
    amount DECIMAL(15,2)
    probability DECIMAL(5,2) CHECK (probability >= 0 AND probability <= 100)
    close_date DATE
    owner_user_id INTEGER
    lead_source VARCHAR(100)
    next_step TEXT
    description TEXT
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    is_deleted BOOLEAN DEFAULT false

    CONSTRAINT fk_opportunities_tenant_id FOREIGN KEY (tenant_id) REFERENCES revops.tenant_metadata(tenant_id) ON DELETE CASCADE
    CONSTRAINT fk_opportunities_account_id FOREIGN KEY (account_id) REFERENCES revops.accounts(account_id) ON DELETE CASCADE
);

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_tenant_industry ON revops.tenant_metadata (industry_code);

CREATE INDEX IF NOT EXISTS idx_tenant_region ON revops.tenant_metadata (region_code);

CREATE INDEX IF NOT EXISTS idx_tenant_status ON revops.tenant_metadata (status);

CREATE INDEX IF NOT EXISTS idx_accounts_tenant ON revops.accounts (tenant_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_accounts_external_id ON revops.accounts (tenant_id, external_id);

CREATE INDEX IF NOT EXISTS idx_accounts_name ON revops.accounts (account_name);

CREATE INDEX IF NOT EXISTS idx_accounts_type ON revops.accounts (account_type);

CREATE INDEX IF NOT EXISTS idx_accounts_created ON revops.accounts (created_at);

CREATE INDEX IF NOT EXISTS idx_opportunities_tenant ON revops.opportunities (tenant_id);

CREATE INDEX IF NOT EXISTS idx_opportunities_account ON revops.opportunities (account_id);

CREATE INDEX IF NOT EXISTS idx_opportunities_stage ON revops.opportunities (stage);

CREATE INDEX IF NOT EXISTS idx_opportunities_close_date ON revops.opportunities (close_date);

CREATE INDEX IF NOT EXISTS idx_opportunities_owner ON revops.opportunities (owner_user_id);

CREATE INDEX IF NOT EXISTS idx_opportunities_amount ON revops.opportunities (amount);

CREATE UNIQUE INDEX IF NOT EXISTS idx_opportunities_external_id ON revops.opportunities (tenant_id, external_id);

-- =====================================================
-- ROW LEVEL SECURITY POLICIES
-- =====================================================

-- Enforce tenant isolation for accounts
ALTER TABLE revops.accounts ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_accounts ON revops.accounts FOR ALL USING (tenant_id = current_setting('app.tenant_id')::integer);

ALTER TABLE revops.opportunities ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_opportunities ON revops.opportunities FOR ALL USING (tenant_id = current_setting('app.tenant_id')::integer);
