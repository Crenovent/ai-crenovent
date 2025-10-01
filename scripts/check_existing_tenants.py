import asyncio
from src.services.connection_pool_manager import ConnectionPoolManager
from dotenv import load_dotenv
import uuid

load_dotenv()

async def check_existing_tenants():
    pool = ConnectionPoolManager()
    await pool.initialize()
    
    async with pool.postgres_pool.acquire() as conn:
        # Check existing users table for tenant patterns
        print("🔍 Checking existing tenant patterns...")
        
        # Check users table
        users = await conn.fetch("SELECT DISTINCT tenant_id FROM users WHERE tenant_id IS NOT NULL LIMIT 10")
        print(f"\n📊 Users table tenant_id values:")
        for row in users:
            print(f"  - {row['tenant_id']} (type: {type(row['tenant_id'])})")
        
        # Check strategic_account_plans
        plans = await conn.fetch("SELECT DISTINCT tenant_id FROM strategic_account_plans WHERE tenant_id IS NOT NULL LIMIT 10")
        print(f"\n📊 Strategic plans tenant_id values:")
        for row in plans:
            print(f"  - {row['tenant_id']} (type: {type(row['tenant_id'])})")
        
        # Generate a UUID for tenant 1300
        tenant_uuid = str(uuid.uuid4())
        print(f"\n🎯 Suggested UUID for tenant 1300: {tenant_uuid}")
        
        # Check if we should map 1300 to a UUID
        print(f"\n💡 Solution: Map tenant '1300' to UUID '{tenant_uuid}'")
    
    await pool.close()

asyncio.run(check_existing_tenants())
