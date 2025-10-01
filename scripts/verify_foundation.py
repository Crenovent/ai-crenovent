#!/usr/bin/env python3
"""
Simple Foundation Verification Script
=====================================

Verifies that the foundation layer is working correctly after applying the database fixes.
"""

import asyncio
import asyncpg
import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def verify_foundation():
    """Verify foundation layer is working"""
    try:
        print("🔍 Verifying Foundation Layer...")
        print("=" * 50)
        
        # Get database URL
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL environment variable not set")
            return False
        
        # Connect to database
        conn = await asyncpg.connect(database_url)
        
        try:
            # Test 1: Basic connectivity
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                print("✅ Database connectivity: PASSED")
            else:
                print("❌ Database connectivity: FAILED")
                return False
            
            # Test 2: Check if validation functions exist
            functions_exist = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.routines 
                WHERE routine_name IN ('validate_tenant_context', 'set_tenant_context', 'test_multi_tenant_isolation')
            """)
            
            if functions_exist >= 3:
                print("✅ Validation functions: PASSED")
            else:
                print(f"⚠️ Validation functions: {functions_exist}/3 found")
            
            # Test 3: Test tenant isolation
            try:
                isolation_results = await conn.fetch("SELECT * FROM test_multi_tenant_isolation()")
                passed_tests = sum(1 for row in isolation_results if row['isolation_status'] == 'PASS')
                total_tests = len(isolation_results)
                
                if passed_tests == total_tests:
                    print(f"✅ Multi-tenant isolation: PASSED ({passed_tests}/{total_tests})")
                else:
                    print(f"⚠️ Multi-tenant isolation: {passed_tests}/{total_tests} tests passed")
                    
                # Show detailed results
                for row in isolation_results:
                    status_icon = "✅" if row['isolation_status'] == 'PASS' else "❌"
                    print(f"   {status_icon} {row['test_name']}: {row['isolation_status']}")
                    
            except Exception as e:
                print(f"⚠️ Multi-tenant isolation test: ERROR - {e}")
            
            # Test 4: Check table structures
            tables_to_check = ['kg_entities', 'kg_relationships', 'strategic_account_plans']
            existing_tables = []
            
            for table in tables_to_check:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = $1 AND table_schema = 'public'
                    )
                """, table)
                
                if exists:
                    existing_tables.append(table)
                    print(f"✅ Table {table}: EXISTS")
                else:
                    print(f"⚠️ Table {table}: MISSING")
            
            # Test 5: Check RLS policies
            rls_tables = await conn.fetch("""
                SELECT tablename, rowsecurity 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename IN ('kg_entities', 'kg_relationships', 'strategic_account_plans')
            """)
            
            rls_enabled = sum(1 for row in rls_tables if row['rowsecurity'])
            total_rls_tables = len(rls_tables)
            
            if rls_enabled > 0:
                print(f"✅ Row Level Security: {rls_enabled}/{total_rls_tables} tables enabled")
            else:
                print(f"⚠️ Row Level Security: No tables have RLS enabled")
            
            # Test 6: Performance test
            start_time = datetime.now()
            await conn.execute("SET app.current_tenant_id = '1300'")
            await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables")
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if query_time < 100:  # Less than 100ms
                print(f"✅ Query performance: {query_time:.2f}ms (GOOD)")
            elif query_time < 500:
                print(f"⚠️ Query performance: {query_time:.2f}ms (ACCEPTABLE)")
            else:
                print(f"❌ Query performance: {query_time:.2f}ms (SLOW)")
            
            print("\n" + "=" * 50)
            print("📊 VERIFICATION SUMMARY")
            print("=" * 50)
            
            # Calculate overall score
            scores = {
                'Database connectivity': True,
                'Validation functions': functions_exist >= 3,
                'Multi-tenant isolation': passed_tests == total_tests if 'passed_tests' in locals() else False,
                'Required tables': len(existing_tables) >= 2,
                'Row Level Security': rls_enabled > 0,
                'Query performance': query_time < 500
            }
            
            passed_checks = sum(1 for check in scores.values() if check)
            total_checks = len(scores)
            success_rate = (passed_checks / total_checks) * 100
            
            print(f"✅ Passed checks: {passed_checks}/{total_checks}")
            print(f"📈 Success rate: {success_rate:.1f}%")
            
            if success_rate >= 80:
                print("🎉 Foundation layer is READY for production!")
                return True
            elif success_rate >= 60:
                print("⚠️ Foundation layer needs some fixes before production")
                return False
            else:
                print("❌ Foundation layer requires significant fixes")
                return False
                
        finally:
            await conn.close()
            
    except Exception as e:
        print(f"💥 Verification failed: {e}")
        logger.error(f"Verification error: {e}", exc_info=True)
        return False

async def main():
    """Main verification function"""
    success = await verify_foundation()
    
    if success:
        print("\n🚀 Foundation layer verification completed successfully!")
        print("You can now proceed with deploying RBA agents.")
    else:
        print("\n🔧 Please fix the identified issues before proceeding.")
        print("Re-run the FIXED_AZURE_SCHEMA.sql script if needed.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
