#!/usr/bin/env python3
"""
PostgreSQL Connection Diagnostic Tool
=====================================

This tool helps diagnose and fix PostgreSQL connection issues with Azure.
Run this before starting the main application to identify problems.
"""

import os
import asyncio
import asyncpg
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLDiagnostic:
    def __init__(self):
        self.config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'database': os.getenv('POSTGRES_DB'),
            'ssl': os.getenv('POSTGRES_SSL', 'require')
        }
    
    async def test_basic_connection(self):
        """Test basic connection to PostgreSQL"""
        print("🔍 Testing basic PostgreSQL connection...")
        
        try:
            conn = await asyncpg.connect(**self.config)
            result = await conn.fetchval("SELECT version()")
            await conn.close()
            
            print(f"✅ Connection successful!")
            print(f"📊 PostgreSQL Version: {result}")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def test_connection_pool(self):
        """Test connection pool creation"""
        print("\n🔍 Testing connection pool...")
        
        pool_config = {
            **self.config,
            'min_size': 2,
            'max_size': 5,
            'command_timeout': 30
        }
        
        try:
            pool = await asyncpg.create_pool(**pool_config)
            
            # Test acquiring connections
            connections = []
            for i in range(3):
                conn = await pool.acquire()
                result = await conn.fetchval("SELECT $1", f"test_{i}")
                connections.append(conn)
                print(f"✅ Connection {i+1} acquired and tested")
            
            # Release connections
            for conn in connections:
                await pool.release(conn)
            
            await pool.close()
            print("✅ Connection pool test successful!")
            return True
            
        except Exception as e:
            print(f"❌ Connection pool test failed: {e}")
            return False
    
    async def check_server_settings(self):
        """Check important server settings"""
        print("\n🔍 Checking server settings...")
        
        try:
            conn = await asyncpg.connect(**self.config)
            
            # Check connection limits
            max_conn = await conn.fetchval("SHOW max_connections")
            current_conn = await conn.fetchval("SELECT count(*) FROM pg_stat_activity")
            
            print(f"📊 Max connections: {max_conn}")
            print(f"📊 Current connections: {current_conn}")
            
            # Check other important settings
            settings_to_check = [
                'shared_preload_libraries',
                'log_connections',
                'log_disconnections',
                'statement_timeout',
                'idle_in_transaction_session_timeout'
            ]
            
            for setting in settings_to_check:
                try:
                    value = await conn.fetchval(f"SHOW {setting}")
                    print(f"📊 {setting}: {value}")
                except:
                    print(f"⚠️ Could not retrieve {setting}")
            
            await conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Server settings check failed: {e}")
            return False
    
    async def check_active_connections(self):
        """Check currently active connections"""
        print("\n🔍 Checking active connections...")
        
        try:
            conn = await asyncpg.connect(**self.config)
            
            query = """
            SELECT 
                datname,
                usename,
                application_name,
                client_addr,
                state,
                query_start,
                state_change,
                query
            FROM pg_stat_activity 
            WHERE state != 'idle'
            ORDER BY query_start DESC
            LIMIT 10
            """
            
            rows = await conn.fetch(query)
            
            if rows:
                print("📊 Active connections:")
                for row in rows:
                    print(f"  - DB: {row['datname']}, User: {row['usename']}, "
                          f"App: {row['application_name']}, State: {row['state']}")
            else:
                print("📊 No active connections found")
            
            await conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Active connections check failed: {e}")
            return False
    
    async def test_table_creation(self):
        """Test creating and dropping a test table"""
        print("\n🔍 Testing table operations...")
        
        try:
            conn = await asyncpg.connect(**self.config)
            
            # Create test table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS diagnostic_test (
                    id SERIAL PRIMARY KEY,
                    test_data TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            print("✅ Test table created")
            
            # Insert test data
            await conn.execute(
                "INSERT INTO diagnostic_test (test_data) VALUES ($1)",
                f"Test at {datetime.now()}"
            )
            print("✅ Test data inserted")
            
            # Query test data
            result = await conn.fetchval("SELECT COUNT(*) FROM diagnostic_test")
            print(f"✅ Test data queried: {result} rows")
            
            # Clean up
            await conn.execute("DROP TABLE diagnostic_test")
            print("✅ Test table cleaned up")
            
            await conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Table operations test failed: {e}")
            return False
    
    def print_recommendations(self):
        """Print recommendations based on configuration"""
        print("\n💡 RECOMMENDATIONS:")
        
        # Check pool size vs max connections
        max_pool = int(os.getenv('POSTGRES_POOL_MAX_SIZE', '5'))
        print(f"📊 Current pool max size: {max_pool}")
        
        if max_pool > 10:
            print("⚠️ Pool size might be too large for Azure Basic/Standard tiers")
            print("   Consider reducing POSTGRES_POOL_MAX_SIZE to 5-8")
        
        # Check SSL setting
        ssl_mode = os.getenv('POSTGRES_SSL', 'prefer')
        if ssl_mode != 'require':
            print("⚠️ SSL should be 'require' for Azure PostgreSQL")
            print("   Set POSTGRES_SSL=require in your environment")
        
        # Check timeout settings
        timeout = int(os.getenv('POSTGRES_COMMAND_TIMEOUT', '30'))
        if timeout < 60:
            print("💡 Consider increasing POSTGRES_COMMAND_TIMEOUT to 60+ seconds")
            print("   This helps with longer-running operations")
        
        print("\n🔧 AZURE PORTAL SETTINGS TO CHECK:")
        print("1. Server Parameters > max_connections (increase to 100+)")
        print("2. Connection Security > Firewall rules (allow your IP)")
        print("3. Connection Security > SSL enforcement (Enabled)")
        print("4. Monitoring > Metrics (check connection usage)")

async def main():
    """Run all diagnostic tests"""
    print("🚀 PostgreSQL Connection Diagnostic Tool")
    print("=" * 50)
    
    diagnostic = PostgreSQLDiagnostic()
    
    # Print configuration (without password)
    config_safe = {k: v for k, v in diagnostic.config.items() if k != 'password'}
    config_safe['password'] = '***' if diagnostic.config['password'] else 'NOT SET'
    print(f"📋 Configuration: {config_safe}")
    
    tests = [
        ("Basic Connection", diagnostic.test_basic_connection),
        ("Connection Pool", diagnostic.test_connection_pool),
        ("Server Settings", diagnostic.check_server_settings),
        ("Active Connections", diagnostic.check_active_connections),
        ("Table Operations", diagnostic.test_table_creation)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your PostgreSQL connection should work.")
    else:
        print("⚠️ Some tests failed. Check the errors above and Azure settings.")
    
    diagnostic.print_recommendations()

if __name__ == "__main__":
    asyncio.run(main())
