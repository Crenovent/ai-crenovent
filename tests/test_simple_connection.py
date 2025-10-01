"""
Test simple database connection
"""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def test_simple_connection():
    try:
        print("🔍 Testing simple database connection...")
        
        # Try with minimal connection
        conn = await asyncpg.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB'),
            ssl='require'
        )
        
        # Test simple query
        result = await conn.fetchval("SELECT 1")
        print(f"✅ Simple connection works: {result}")
        
        # Check existing tables
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%tenant%' OR table_name LIKE '%dsl%'
            ORDER BY table_name
        """)
        
        print(f"📊 Found {len(tables)} existing tables:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_simple_connection())
