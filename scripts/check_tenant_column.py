import asyncio
from src.services.connection_pool_manager import ConnectionPoolManager
from dotenv import load_dotenv

load_dotenv()

async def check_tenant_column():
    pool = ConnectionPoolManager()
    await pool.initialize()
    
    async with pool.postgres_pool.acquire() as conn:
        # Check tenant_id column type in DSL tables
        result = await conn.fetch("""
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name LIKE 'dsl_%' 
            AND column_name = 'tenant_id'
            ORDER BY table_name
        """)
        
        print("tenant_id columns in DSL tables:")
        for row in result:
            print(f"  {row['table_name']}.{row['column_name']}: {row['data_type']}")
    
    await pool.close()

asyncio.run(check_tenant_column())
