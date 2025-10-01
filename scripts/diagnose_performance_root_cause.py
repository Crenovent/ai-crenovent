"""
Deep Performance Diagnosis
Identify root cause of 2+ second response times
"""

import asyncio
import time
import psutil
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_database_connection_speed():
    """Test if database connection is the bottleneck"""
    print("🔍 TESTING DATABASE CONNECTION SPEED")
    print("=" * 50)
    
    try:
        import asyncpg
        
        # Test direct database connection
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ No DATABASE_URL found")
            return
        
        print(f"Database URL: {database_url[:50]}...")
        
        # Test connection time
        start_time = time.time()
        try:
            conn = await asyncpg.connect(database_url, ssl='require')
            connection_time = (time.time() - start_time) * 1000
            print(f"✅ Connection established: {connection_time:.0f}ms")
            
            # Test simple query
            start_time = time.time()
            result = await conn.fetchval("SELECT 1")
            query_time = (time.time() - start_time) * 1000
            print(f"✅ Simple query: {query_time:.0f}ms")
            
            # Test more complex query
            start_time = time.time()
            result = await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables")
            complex_query_time = (time.time() - start_time) * 1000
            print(f"✅ Complex query: {complex_query_time:.0f}ms")
            
            # Test pgvector
            start_time = time.time()
            try:
                result = await conn.fetchval("SELECT '[1,2,3]'::vector")
                vector_time = (time.time() - start_time) * 1000
                print(f"✅ Vector query: {vector_time:.0f}ms")
            except Exception as e:
                vector_time = (time.time() - start_time) * 1000
                print(f"⚠️  Vector query failed: {vector_time:.0f}ms - {e}")
            
            await conn.close()
            
            # Analyze results
            total_time = connection_time + query_time + complex_query_time
            print(f"\n📊 Database Analysis:")
            print(f"   Connection: {connection_time:.0f}ms")
            print(f"   Simple Query: {query_time:.0f}ms") 
            print(f"   Complex Query: {complex_query_time:.0f}ms")
            print(f"   Total DB Time: {total_time:.0f}ms")
            
            if total_time > 1500:
                print(f"   🚨 DATABASE IS THE BOTTLENECK!")
            elif total_time > 500:
                print(f"   ⚠️  Database is slow but not main issue")
            else:
                print(f"   ✅ Database performance is good")
                
        except Exception as e:
            connection_time = (time.time() - start_time) * 1000
            print(f"❌ Database connection failed: {connection_time:.0f}ms - {e}")
            
    except ImportError:
        print("❌ asyncpg not available")

def test_import_times():
    """Test if imports are causing delays"""
    print("\n🐍 TESTING IMPORT TIMES")
    print("=" * 50)
    
    imports_to_test = [
        "import asyncio",
        "import asyncpg",
        "from fastapi import FastAPI",
        "from dotenv import load_dotenv",
        "import psutil",
        "import pandas as pd",
        "from dsl import DSLParser",
        "from dsl.hub.execution_hub import ExecutionHub"
    ]
    
    for import_statement in imports_to_test:
        start_time = time.time()
        try:
            exec(import_statement)
            import_time = (time.time() - start_time) * 1000
            
            if import_time > 100:
                print(f"🐌 SLOW: {import_statement} - {import_time:.0f}ms")
            elif import_time > 50:
                print(f"⚠️  MODERATE: {import_statement} - {import_time:.0f}ms")
            else:
                print(f"✅ FAST: {import_statement} - {import_time:.0f}ms")
                
        except Exception as e:
            import_time = (time.time() - start_time) * 1000
            print(f"❌ FAILED: {import_statement} - {import_time:.0f}ms - {e}")

def test_system_resources():
    """Check system resources"""
    print("\n💻 SYSTEM RESOURCE ANALYSIS")
    print("=" * 50)
    
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        print(f"Memory Usage: {memory.percent}% ({memory.used / 1024 / 1024 / 1024:.1f}GB / {memory.total / 1024 / 1024 / 1024:.1f}GB)")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        print(f"Disk Usage: {disk.percent}% ({disk.used / 1024 / 1024 / 1024:.1f}GB / {disk.total / 1024 / 1024 / 1024:.1f}GB)")
        
        # Network
        network = psutil.net_io_counters()
        print(f"Network: Sent {network.bytes_sent / 1024 / 1024:.1f}MB, Recv {network.bytes_recv / 1024 / 1024:.1f}MB")
        
        # Check if resources are constrained
        issues = []
        if cpu_percent > 80:
            issues.append(f"High CPU usage: {cpu_percent}%")
        if memory.percent > 80:
            issues.append(f"High memory usage: {memory.percent}%")
        if disk.percent > 90:
            issues.append(f"High disk usage: {disk.percent}%")
        
        if issues:
            print(f"\n🚨 Resource Issues:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"\n✅ System resources look healthy")
            
    except Exception as e:
        print(f"❌ Error checking system resources: {e}")

def test_environment_loading():
    """Test environment variable loading time"""
    print("\n🔧 TESTING ENVIRONMENT LOADING")
    print("=" * 50)
    
    start_time = time.time()
    load_dotenv()
    env_load_time = (time.time() - start_time) * 1000
    print(f"Environment loading: {env_load_time:.0f}ms")
    
    # Check critical variables
    critical_vars = [
        'DATABASE_URL',
        'POSTGRES_HOST',
        'AZURE_OPENAI_API_KEY',
        'MAIN_TENANT_UUID'
    ]
    
    missing_vars = []
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set ({len(value)} chars)")
        else:
            print(f"❌ {var}: Missing")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing critical variables: {missing_vars}")
    else:
        print(f"\n✅ All critical environment variables present")

async def test_async_operations():
    """Test if async operations are working efficiently"""
    print("\n⚡ TESTING ASYNC OPERATIONS")
    print("=" * 50)
    
    # Test simple async operations
    async def simple_async_task(delay):
        await asyncio.sleep(delay)
        return f"Task completed after {delay}s"
    
    # Test concurrent async operations
    start_time = time.time()
    tasks = [simple_async_task(0.1) for _ in range(5)]
    results = await asyncio.gather(*tasks)
    concurrent_time = (time.time() - start_time) * 1000
    
    print(f"5 concurrent async tasks (0.1s each): {concurrent_time:.0f}ms")
    
    if concurrent_time > 500:  # Should be around 100ms for concurrent execution
        print(f"🚨 ASYNC OPERATIONS ARE SLOW!")
    elif concurrent_time > 200:
        print(f"⚠️  Async operations slightly slow")
    else:
        print(f"✅ Async operations working efficiently")

async def main():
    """Run all performance diagnostics"""
    print("🔬 DEEP PERFORMANCE DIAGNOSIS")
    print("=" * 60)
    print("Identifying root cause of 2+ second response times")
    print("=" * 60)
    
    # Run all diagnostic tests
    test_environment_loading()
    test_import_times()
    test_system_resources()
    await test_database_connection_speed()
    await test_async_operations()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("Review the results above to identify the bottleneck:")
    print("- Database connection > 1500ms = Database issue")
    print("- Import times > 100ms = Module loading issue") 
    print("- High CPU/Memory = Resource constraint")
    print("- Async operations > 200ms = Event loop issue")

if __name__ == "__main__":
    asyncio.run(main())
