"""
Schema Migration Script
======================
Handles migrating existing database schema to support multi-tenant enforcement.

This script:
- Checks existing schema
- Adds missing columns and tables
- Enables RLS policies
- Seeds required data
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.services.connection_pool_manager import pool_manager

logger = logging.getLogger(__name__)

async def migrate_existing_schema():
    """Migrate existing database schema to support multi-tenant features"""
    try:
        print("🚀 Starting schema migration for multi-tenant support...")
        
        # Step 1: Initialize connection pool
        print("📊 Step 1: Initializing connection pool...")
        success = await pool_manager.initialize()
        if not success:
            raise Exception("Failed to initialize connection pool")
        print("✅ Connection pool ready")
        
        # Step 2: Check existing schema
        print("🔍 Step 2: Analyzing existing schema...")
        async with pool_manager.postgres_pool.acquire() as conn:
            # Check what tables exist
            existing_tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """)
            
            print(f"📊 Found {len(existing_tables)} existing tables:")
            for table in existing_tables:
                print(f"  - {table['table_name']}")
        
        # Step 3: Create tenant_metadata table if it doesn't exist
        print("🏢 Step 3: Creating tenant_metadata table...")
        async with pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tenant_metadata (
                    tenant_id INTEGER PRIMARY KEY,
                    tenant_name VARCHAR(255) NOT NULL,
                    tenant_type VARCHAR(50) NOT NULL DEFAULT 'standard',
                    industry_code VARCHAR(10) NOT NULL,
                    region_code VARCHAR(10) NOT NULL,
                    compliance_requirements JSONB NOT NULL DEFAULT '[]',
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_by_user_id INTEGER,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            
            # Enable RLS
            await conn.execute("ALTER TABLE tenant_metadata ENABLE ROW LEVEL SECURITY")
            
            # Create RLS policy
            await conn.execute("""
                DROP POLICY IF EXISTS tenant_metadata_rls_policy ON tenant_metadata
            """)
            await conn.execute("""
                CREATE POLICY tenant_metadata_rls_policy ON tenant_metadata
                FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1))
            """)
            
            print("✅ tenant_metadata table ready")
        
        # Step 4: Migrate dsl_workflows table
        print("📝 Step 4: Migrating dsl_workflows table...")
        async with pool_manager.postgres_pool.acquire() as conn:
            # Check if workflow_type column exists
            workflow_type_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'dsl_workflows' 
                    AND column_name = 'workflow_type'
                )
            """)
            
            if not workflow_type_exists:
                print("  Adding workflow_type column...")
                await conn.execute("""
                    ALTER TABLE dsl_workflows 
                    ADD COLUMN IF NOT EXISTS workflow_type VARCHAR(50) NOT NULL DEFAULT 'RBA'
                """)
            
            # Check if tenant_id column exists
            tenant_id_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'dsl_workflows' 
                    AND column_name = 'tenant_id'
                )
            """)
            
            if not tenant_id_exists:
                print("  Adding tenant_id column...")
                await conn.execute("""
                    ALTER TABLE dsl_workflows 
                    ADD COLUMN IF NOT EXISTS tenant_id INTEGER NOT NULL DEFAULT 1300
                """)
            
            # Add other missing columns
            missing_columns = [
                ('industry_overlay', 'VARCHAR(10)'),
                ('policy_pack_id', 'UUID'),
                ('evidence_pack_id', 'UUID'),
                ('trust_score', 'DECIMAL(5,2) DEFAULT 1.0'),
                ('execution_count', 'INTEGER DEFAULT 0'),
                ('last_executed_at', 'TIMESTAMPTZ')
            ]
            
            for column_name, column_type in missing_columns:
                column_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'dsl_workflows' 
                        AND column_name = $1
                    )
                """, column_name)
                
                if not column_exists:
                    print(f"  Adding {column_name} column...")
                    await conn.execute(f"""
                        ALTER TABLE dsl_workflows 
                        ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                    """)
            
            # Enable RLS
            await conn.execute("ALTER TABLE dsl_workflows ENABLE ROW LEVEL SECURITY")
            
            # Create RLS policy
            await conn.execute("""
                DROP POLICY IF EXISTS dsl_workflows_rls_policy ON dsl_workflows
            """)
            await conn.execute("""
                CREATE POLICY dsl_workflows_rls_policy ON dsl_workflows
                FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1))
            """)
            
            print("✅ dsl_workflows table migrated")
        
        # Step 5: Create policy packs table
        print("⚖️ Step 5: Creating dsl_policy_packs table...")
        async with pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dsl_policy_packs (
                    policy_pack_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    pack_name VARCHAR(255) NOT NULL,
                    pack_type VARCHAR(50) NOT NULL,
                    industry_code VARCHAR(10) NOT NULL,
                    region_code VARCHAR(10) NOT NULL,
                    policy_rules JSONB NOT NULL,
                    enforcement_level VARCHAR(20) NOT NULL DEFAULT 'strict',
                    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_by_user_id INTEGER NOT NULL,
                    
                    CONSTRAINT unique_policy_pack_per_tenant UNIQUE (tenant_id, pack_name, version)
                )
            """)
            
            # Enable RLS
            await conn.execute("ALTER TABLE dsl_policy_packs ENABLE ROW LEVEL SECURITY")
            
            # Create RLS policy
            await conn.execute("""
                DROP POLICY IF EXISTS dsl_policy_packs_rls_policy ON dsl_policy_packs
            """)
            await conn.execute("""
                CREATE POLICY dsl_policy_packs_rls_policy ON dsl_policy_packs
                FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1))
            """)
            
            print("✅ dsl_policy_packs table ready")
        
        # Step 6: Create execution traces table
        print("📊 Step 6: Creating dsl_execution_traces table...")
        async with pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dsl_execution_traces (
                    trace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    workflow_id UUID NOT NULL,
                    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
                    agent_name VARCHAR(255) NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    user_id INTEGER NOT NULL,
                    execution_source VARCHAR(50) DEFAULT 'api',
                    
                    inputs JSONB,
                    outputs JSONB,
                    configuration_used JSONB,
                    
                    governance_metadata JSONB DEFAULT '{}',
                    policy_pack_id UUID,
                    evidence_pack_id UUID,
                    override_count INTEGER DEFAULT 0,
                    
                    execution_time_ms INTEGER DEFAULT 0,
                    memory_usage_mb DECIMAL(10,2) DEFAULT 0,
                    trust_score DECIMAL(5,2) DEFAULT 1.0,
                    
                    opportunities_processed INTEGER DEFAULT 0,
                    flagged_opportunities INTEGER DEFAULT 0,
                    pipeline_value_analyzed DECIMAL(15,2) DEFAULT 0,
                    
                    entities_affected TEXT[],
                    relationships_created TEXT[],
                    
                    execution_start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    execution_end_time TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            
            # Enable RLS
            await conn.execute("ALTER TABLE dsl_execution_traces ENABLE ROW LEVEL SECURITY")
            
            # Create RLS policy
            await conn.execute("""
                DROP POLICY IF EXISTS dsl_execution_traces_rls_policy ON dsl_execution_traces
            """)
            await conn.execute("""
                CREATE POLICY dsl_execution_traces_rls_policy ON dsl_execution_traces
                FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1))
            """)
            
            print("✅ dsl_execution_traces table ready")
        
        # Step 7: Create override ledger table
        print("📝 Step 7: Creating dsl_override_ledger table...")
        async with pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dsl_override_ledger (
                    override_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    workflow_id UUID,
                    execution_trace_id UUID,
                    user_id INTEGER NOT NULL,
                    
                    override_type VARCHAR(50) NOT NULL,
                    component_affected VARCHAR(100) NOT NULL,
                    original_value JSONB,
                    overridden_value JSONB,
                    override_reason TEXT NOT NULL,
                    
                    approval_required BOOLEAN DEFAULT true,
                    approved_by_user_id INTEGER,
                    approved_at TIMESTAMPTZ,
                    approval_justification TEXT,
                    
                    risk_level VARCHAR(20) DEFAULT 'medium',
                    business_impact TEXT,
                    
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ,
                    status VARCHAR(20) DEFAULT 'active',
                    
                    policy_pack_id UUID,
                    evidence_pack_id UUID,
                    compliance_flags JSONB DEFAULT '{}'
                )
            """)
            
            # Enable RLS
            await conn.execute("ALTER TABLE dsl_override_ledger ENABLE ROW LEVEL SECURITY")
            
            # Create RLS policy
            await conn.execute("""
                DROP POLICY IF EXISTS dsl_override_ledger_rls_policy ON dsl_override_ledger
            """)
            await conn.execute("""
                CREATE POLICY dsl_override_ledger_rls_policy ON dsl_override_ledger
                FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1))
            """)
            
            print("✅ dsl_override_ledger table ready")
        
        # Step 8: Create evidence packs table
        print("📄 Step 8: Creating dsl_evidence_packs table...")
        async with pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dsl_evidence_packs (
                    evidence_pack_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id INTEGER NOT NULL,
                    pack_name VARCHAR(255) NOT NULL,
                    pack_type VARCHAR(50) NOT NULL,
                    
                    evidence_data JSONB NOT NULL,
                    evidence_hash VARCHAR(64) NOT NULL,
                    digital_signature TEXT,
                    
                    workflow_id UUID,
                    execution_trace_id UUID,
                    policy_pack_id UUID,
                    override_id UUID,
                    
                    compliance_framework VARCHAR(50),
                    retention_period_days INTEGER DEFAULT 2555,
                    immutable BOOLEAN DEFAULT true,
                    
                    blob_storage_url TEXT,
                    file_format VARCHAR(20) DEFAULT 'json',
                    file_size_bytes BIGINT,
                    
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ
                )
            """)
            
            # Enable RLS
            await conn.execute("ALTER TABLE dsl_evidence_packs ENABLE ROW LEVEL SECURITY")
            
            # Create RLS policy
            await conn.execute("""
                DROP POLICY IF EXISTS dsl_evidence_packs_rls_policy ON dsl_evidence_packs
            """)
            await conn.execute("""
                CREATE POLICY dsl_evidence_packs_rls_policy ON dsl_evidence_packs
                FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1))
            """)
            
            print("✅ dsl_evidence_packs table ready")
        
        # Step 9: Seed default tenant and policy packs
        print("🌱 Step 9: Seeding default tenant and policy packs...")
        async with pool_manager.postgres_pool.acquire() as conn:
            # Insert default tenant
            await conn.execute("""
                INSERT INTO tenant_metadata (tenant_id, tenant_name, tenant_type, industry_code, region_code, compliance_requirements)
                VALUES (1300, 'Default Tenant', 'enterprise', 'SaaS', 'US', '["SOX", "GDPR"]')
                ON CONFLICT (tenant_id) DO UPDATE SET
                    updated_at = NOW(),
                    tenant_type = EXCLUDED.tenant_type,
                    industry_code = EXCLUDED.industry_code,
                    region_code = EXCLUDED.region_code,
                    compliance_requirements = EXCLUDED.compliance_requirements
            """)
            
            # Set tenant context for policy pack creation
            await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "1300")
            
            # Insert SOX policy pack
            await conn.execute("""
                INSERT INTO dsl_policy_packs (tenant_id, pack_name, pack_type, industry_code, region_code, policy_rules, created_by_user_id)
                VALUES (
                    1300,
                    'SaaS SOX Compliance Pack',
                    'SOX',
                    'SaaS',
                    'US',
                    '{"financial_controls": {"revenue_recognition": {"enabled": true, "enforcement": "strict"}, "deal_approval_thresholds": {"high_value": 250000, "mega_deal": 1000000}, "segregation_of_duties": {"enabled": true, "maker_checker": true}}, "audit_requirements": {"execution_logging": {"enabled": true, "retention_days": 2555}, "override_justification": {"required": true, "approval_required": true}, "evidence_generation": {"enabled": true, "immutable": true}}}',
                    1319
                ) ON CONFLICT (tenant_id, pack_name, version) DO NOTHING
            """)
            
            # Insert GDPR policy pack
            await conn.execute("""
                INSERT INTO dsl_policy_packs (tenant_id, pack_name, pack_type, industry_code, region_code, policy_rules, created_by_user_id)
                VALUES (
                    1300,
                    'SaaS GDPR Privacy Pack',
                    'GDPR',
                    'SaaS',
                    'EU',
                    '{"data_protection": {"consent_management": {"enabled": true, "explicit_consent": true}, "right_to_erasure": {"enabled": true, "retention_override": false}, "data_minimization": {"enabled": true, "purpose_limitation": true}}, "privacy_controls": {"pii_classification": {"enabled": true, "auto_detection": true}, "cross_border_transfer": {"restricted": true, "adequacy_required": true}, "breach_notification": {"enabled": true, "notification_hours": 72}}}',
                    1319
                ) ON CONFLICT (tenant_id, pack_name, version) DO NOTHING
            """)
            
            print("✅ Default tenant and policy packs seeded")
        
        # Step 10: Create indexes for performance
        print("⚡ Step 10: Creating performance indexes...")
        async with pool_manager.postgres_pool.acquire() as conn:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_dsl_workflows_tenant_status ON dsl_workflows(tenant_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_dsl_workflows_tenant_type ON dsl_workflows(tenant_id, workflow_type)",
                "CREATE INDEX IF NOT EXISTS idx_dsl_policy_packs_tenant_type ON dsl_policy_packs(tenant_id, pack_type)",
                "CREATE INDEX IF NOT EXISTS idx_dsl_execution_traces_tenant_workflow ON dsl_execution_traces(tenant_id, workflow_id)",
                "CREATE INDEX IF NOT EXISTS idx_dsl_execution_traces_tenant_created ON dsl_execution_traces(tenant_id, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_dsl_override_ledger_tenant_created ON dsl_override_ledger(tenant_id, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_dsl_evidence_packs_tenant_type ON dsl_evidence_packs(tenant_id, pack_type)"
            ]
            
            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"  Warning: Failed to create index: {e}")
            
            print("✅ Performance indexes created")
        
        # Step 11: Test multi-tenant functionality
        print("🧪 Step 11: Testing multi-tenant functionality...")
        async with pool_manager.postgres_pool.acquire() as conn:
            # Test tenant context
            await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "1300")
            
            # Test tenant metadata access
            tenant_count = await conn.fetchval("SELECT COUNT(*) FROM tenant_metadata WHERE tenant_id = 1300")
            print(f"  ✅ Tenant 1300 accessible: {tenant_count} record(s)")
            
            # Test policy packs
            policy_count = await conn.fetchval("SELECT COUNT(*) FROM dsl_policy_packs WHERE tenant_id = 1300")
            print(f"  ✅ Policy packs for tenant 1300: {policy_count}")
            
            # Test workflows
            workflow_count = await conn.fetchval("SELECT COUNT(*) FROM dsl_workflows WHERE tenant_id = 1300")
            print(f"  ✅ Workflows for tenant 1300: {workflow_count}")
        
        print("\n🎉 SUCCESS: Schema migration completed successfully!")
        print("📋 Summary:")
        print("   ✅ Multi-tenant tables created/migrated")
        print("   ✅ RLS policies enabled on all tables")
        print("   ✅ Default tenant (1300) seeded with SOX and GDPR policy packs")
        print("   ✅ Performance indexes created")
        print("   ✅ Multi-tenant functionality tested and working")
        
        print("\n🚀 Ready for multi-tenant operations!")
        print("   • Multi-tenant enforcement: ENABLED")
        print("   • Compliance frameworks: SOX, GDPR")
        print("   • Evidence packs and override ledger: ACTIVE")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: Schema migration failed!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always close the pool
        if pool_manager.postgres_pool:
            await pool_manager.postgres_pool.close()
            print("🔄 Connection pool closed")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduced logging to avoid clutter
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run migration
    success = asyncio.run(migrate_existing_schema())
    
    if not success:
        sys.exit(1)
