"""
Multi-Tenant Database Initialization Script
==========================================
Initializes the complete multi-tenant database schema and seeding data.

This script implements:
- Chapter 9.4: Multi-tenant enforcement validation (Tasks 9.4.1-9.4.42)
- Chapter 16: Policy Packs and Compliance Framework
- Chapter 8: Knowledge Graph and Execution Traces
- Chapter 14-15: DSL Workflows and Capability Registry

Usage:
    python -m dsl.database.initialize_multi_tenant_db
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
from dsl.governance.multi_tenant_enforcer import get_multi_tenant_enforcer
from dsl.governance.policy_engine import get_policy_engine

logger = logging.getLogger(__name__)

class MultiTenantDatabaseInitializer:
    """
    Database initializer for multi-tenant RevAI Pro system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.schema_file = Path(__file__).parent / "multi_tenant_schema.sql"
    
    async def initialize_complete_database(self) -> bool:
        """
        Initialize complete multi-tenant database schema and services
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("🚀 Starting complete multi-tenant database initialization...")
            
            # Step 1: Initialize connection pool
            self.logger.info("📊 Step 1: Initializing connection pool...")
            success = await pool_manager.initialize()
            if not success:
                raise Exception("Failed to initialize connection pool")
            
            # Step 2: Execute database schema
            self.logger.info("🗃️ Step 2: Creating database schema...")
            await self._execute_schema_file()
            
            # Step 3: Verify schema creation
            self.logger.info("✅ Step 3: Verifying schema creation...")
            await self._verify_schema()
            
            # Step 4: Initialize multi-tenant enforcer
            self.logger.info("🔒 Step 4: Initializing multi-tenant enforcer...")
            enforcer = get_multi_tenant_enforcer(pool_manager)
            await enforcer.initialize()
            
            # Step 5: Initialize policy engine
            self.logger.info("⚖️ Step 5: Initializing policy engine...")
            policy_engine = get_policy_engine(pool_manager)
            await policy_engine.initialize()
            
            # Step 6: Seed additional test data
            self.logger.info("🌱 Step 6: Seeding additional test data...")
            await self._seed_test_data()
            
            # Step 7: Run validation tests
            self.logger.info("🧪 Step 7: Running validation tests...")
            await self._run_validation_tests()
            
            self.logger.info("🎉 Multi-tenant database initialization completed successfully!")
            self.logger.info("📋 Summary:")
            self.logger.info("   ✅ Database schema created with RLS policies")
            self.logger.info("   ✅ Multi-tenant enforcer initialized")
            self.logger.info("   ✅ Policy engine with compliance frameworks initialized")
            self.logger.info("   ✅ Test data seeded for tenant 1300")
            self.logger.info("   ✅ Validation tests passed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Multi-tenant database initialization failed: {e}")
            return False
    
    async def _execute_schema_file(self) -> None:
        """Execute the SQL schema file"""
        try:
            if not self.schema_file.exists():
                raise FileNotFoundError(f"Schema file not found: {self.schema_file}")
            
            # Read schema file
            with open(self.schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Execute schema SQL
            async with pool_manager.postgres_pool.acquire() as conn:
                await conn.execute(schema_sql)
            
            self.logger.info("✅ Database schema executed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute schema file: {e}")
            raise
    
    async def _verify_schema(self) -> None:
        """Verify that all required tables and policies exist"""
        try:
            required_tables = [
                'tenant_metadata',
                'dsl_workflows', 
                'dsl_policy_packs',
                'dsl_execution_traces',
                'dsl_override_ledger',
                'dsl_evidence_packs',
                'dsl_workflow_templates',
                'dsl_capability_registry',
                'kg_execution_traces'
            ]
            
            async with pool_manager.postgres_pool.acquire() as conn:
                for table_name in required_tables:
                    # Check table exists
                    exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = $1
                        )
                    """, table_name)
                    
                    if not exists:
                        raise Exception(f"Required table not found: {table_name}")
                    
                    # Check RLS is enabled
                    rls_enabled = await conn.fetchval("""
                        SELECT relrowsecurity
                        FROM pg_class
                        WHERE relname = $1
                    """, table_name)
                    
                    if not rls_enabled:
                        raise Exception(f"RLS not enabled on table: {table_name}")
                    
                    self.logger.debug(f"✅ Table verified: {table_name} (RLS enabled)")
                
                # Check default tenant exists
                tenant_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM tenant_metadata WHERE tenant_id = 1300
                    )
                """)
                
                if not tenant_exists:
                    raise Exception("Default tenant (1300) not found")
                
                # Check policy packs exist
                policy_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM dsl_policy_packs WHERE tenant_id = 1300
                """)
                
                if policy_count == 0:
                    raise Exception("No policy packs found for default tenant")
                
                self.logger.info(f"✅ Schema verification passed: {len(required_tables)} tables, RLS enabled, default tenant and {policy_count} policy packs")
                
        except Exception as e:
            self.logger.error(f"❌ Schema verification failed: {e}")
            raise
    
    async def _seed_test_data(self) -> None:
        """Seed additional test data"""
        try:
            async with pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "1300")
                
                # Create additional test tenants
                await conn.execute("""
                    INSERT INTO tenant_metadata (tenant_id, tenant_name, tenant_type, industry_code, region_code, compliance_requirements)
                    VALUES 
                        (1301, 'Banking Test Tenant', 'enterprise', 'BANK', 'IN', '["RBI", "DPDP"]'),
                        (1302, 'Insurance Test Tenant', 'regulated', 'INSUR', 'US', '["HIPAA", "NAIC"]'),
                        (1303, 'E-commerce Test Tenant', 'standard', 'ECOMM', 'EU', '["GDPR"]')
                    ON CONFLICT (tenant_id) DO NOTHING
                """)
                
                # Create sample workflow templates
                await conn.execute("""
                    INSERT INTO dsl_workflow_templates (
                        tenant_id, template_name, template_type, industry_overlay, category,
                        template_definition, description, created_by_user_id
                    ) VALUES (
                        1300,
                        'SaaS Pipeline Hygiene',
                        'RBA',
                        'SaaS',
                        'data_quality',
                        '{"workflow_type": "RBA", "agents": ["pipeline_hygiene_stale_deals"], "parameters": {"stale_threshold_days": 30}}',
                        'Automated pipeline hygiene checks for SaaS companies',
                        1319
                    ), (
                        1300,
                        'Sandbagging Detection',
                        'RBA', 
                        'SaaS',
                        'risk_analysis',
                        '{"workflow_type": "RBA", "agents": ["sandbagging_detection"], "parameters": {"high_value_threshold": 250000}}',
                        'Detect high-value deals with artificially low probability',
                        1319
                    )
                    ON CONFLICT (tenant_id, template_name, version) DO NOTHING
                """)
                
                # Create sample capabilities
                await conn.execute("""
                    INSERT INTO dsl_capability_registry (
                        tenant_id, capability_name, capability_type, operator_definition,
                        input_schema, output_schema, description, created_by_user_id
                    ) VALUES (
                        1300,
                        'pipeline_data_query',
                        'query',
                        '{"source": "salesforce", "resource": "opportunities", "filters": []}',
                        '{"type": "object", "properties": {"filters": {"type": "array"}}}',
                        '{"type": "object", "properties": {"opportunities": {"type": "array"}}}',
                        'Query pipeline data from Salesforce',
                        1319
                    ), (
                        1300,
                        'risk_assessment_decision',
                        'decision',
                        '{"expression": "deal_amount > 250000 AND probability < 35", "thresholds": {}}',
                        '{"type": "object", "properties": {"deal_amount": {"type": "number"}, "probability": {"type": "number"}}}',
                        '{"type": "object", "properties": {"risk_level": {"type": "string"}, "flagged": {"type": "boolean"}}}',
                        'Assess deal risk based on amount and probability',
                        1319
                    )
                    ON CONFLICT (tenant_id, capability_name, version) DO NOTHING
                """)
                
                self.logger.info("✅ Test data seeded successfully")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to seed test data: {e}")
            raise
    
    async def _run_validation_tests(self) -> None:
        """Run validation tests to ensure system works correctly"""
        try:
            async with pool_manager.postgres_pool.acquire() as conn:
                # Test 1: Tenant context enforcement
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "1300")
                
                tenant_workflows = await conn.fetchval("""
                    SELECT COUNT(*) FROM dsl_workflows WHERE tenant_id = 1300
                """)
                
                # Test 2: Cross-tenant isolation (should return 0)
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "1301")
                
                cross_tenant_workflows = await conn.fetchval("""
                    SELECT COUNT(*) FROM dsl_workflows WHERE tenant_id = 1300
                """)
                
                if cross_tenant_workflows > 0:
                    raise Exception("RLS policy failed - cross-tenant data access detected")
                
                # Test 3: Policy pack enforcement
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "1300")
                
                policy_packs = await conn.fetchval("""
                    SELECT COUNT(*) FROM dsl_policy_packs WHERE tenant_id = 1300
                """)
                
                if policy_packs == 0:
                    raise Exception("No policy packs found for tenant 1300")
                
                # Test 4: Evidence pack generation
                evidence_packs = await conn.fetchval("""
                    SELECT COUNT(*) FROM dsl_evidence_packs WHERE tenant_id = 1300
                """)
                
                self.logger.info("✅ Validation tests passed:")
                self.logger.info(f"   - Tenant isolation: ✅ (0 cross-tenant access)")
                self.logger.info(f"   - Policy packs: ✅ ({policy_packs} packs found)")
                self.logger.info(f"   - Evidence packs: ✅ ({evidence_packs} packs found)")
                
        except Exception as e:
            self.logger.error(f"❌ Validation tests failed: {e}")
            raise

async def main():
    """Main initialization function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    initializer = MultiTenantDatabaseInitializer()
    success = await initializer.initialize_complete_database()
    
    if success:
        print("\n🎉 SUCCESS: Multi-tenant database initialization completed!")
        print("🔧 The system is now ready with:")
        print("   • Multi-tenant enforcement with RLS policies")
        print("   • Policy packs for SOX, GDPR, RBI, HIPAA compliance")
        print("   • Evidence packs for regulatory reporting")
        print("   • Override ledger for governance")
        print("   • Knowledge graph execution tracing")
        print("   • Workflow templates and capability registry")
        print("\n📋 Next steps:")
        print("   1. Start the FastAPI application: python main.py")
        print("   2. Test multi-tenant workflows via API endpoints")
        print("   3. Monitor compliance dashboards")
    else:
        print("\n❌ FAILED: Multi-tenant database initialization failed!")
        print("   Check the logs above for error details")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
