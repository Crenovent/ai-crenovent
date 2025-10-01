#!/usr/bin/env python3
"""
Quick Database Status Checker
=============================

Simple script to check database connection and table status without full verification.
"""

import asyncio
import asyncpg
import os
import sys

async def check_db_status():
    """Quick database status check"""
    try:
        print("🔍 Checking Database Status...")
        print("=" * 40)
        
        # Get database URL from environment
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL not set")
            print("💡 Set it like: $env:DATABASE_URL='postgresql://user:pass@host:port/db'")
            return False
        
        # Test connection
        try:
            conn = await asyncpg.connect(database_url)
            print("✅ Database connection: SUCCESS")
        except Exception as e:
            print(f"❌ Database connection: FAILED - {e}")
            return False
        
        try:
            # Check tables
            tables = await conn.fetch("""
                SELECT table_name, 
                       CASE WHEN table_name IN (
                           SELECT tablename FROM pg_tables WHERE rowsecurity = true
                       ) THEN 'RLS Enabled' ELSE 'No RLS' END as rls_status
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            
            print(f"\n📊 Found {len(tables)} tables:")
            for table in tables:
                print(f"   • {table['table_name']:<30} | {table['rls_status']}")
            
            # Check indexes
            indexes = await conn.fetchval("""
                SELECT COUNT(*) FROM pg_indexes 
                WHERE schemaname = 'public'
            """)
            print(f"\n🔍 Found {indexes} indexes")
            
            # Check functions
            functions = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.routines 
                WHERE routine_schema = 'public'
                AND routine_type = 'FUNCTION'
            """)
            print(f"⚙️ Found {functions} functions")
            
            # Test basic query performance
            import time
            start = time.time()
            await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables")
            query_time = (time.time() - start) * 1000
            
            if query_time < 50:
                print(f"⚡ Query performance: {query_time:.1f}ms (EXCELLENT)")
            elif query_time < 200:
                print(f"✅ Query performance: {query_time:.1f}ms (GOOD)")
            else:
                print(f"⚠️ Query performance: {query_time:.1f}ms (SLOW)")
            
            print("\n" + "=" * 40)
            
            # Quick recommendations
            if len(tables) < 5:
                print("💡 Run the FIXED_AZURE_SCHEMA.sql to create missing tables")
            elif functions < 3:
                print("💡 Some validation functions may be missing")
            else:
                print("🎉 Database looks healthy! Ready for full verification.")
            
            return True
            
        finally:
            await conn.close()
            
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

async def main():
    """Main function"""
    success = await check_db_status()
    
    if success:
        print("\n🚀 Database status check completed!")
        print("📋 Next steps:")
        print("   1. Run full verification: python verify_foundation.py")
        print("   2. Test with real data")
    else:
        print("\n🔧 Fix database issues first:")
        print("   1. Set DATABASE_URL environment variable")
        print("   2. Run FIXED_AZURE_SCHEMA.sql in Azure Data Studio")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
