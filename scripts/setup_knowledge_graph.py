#!/usr/bin/env python3
"""
Setup Knowledge Graph
====================

Production setup script for Knowledge Graph initialization
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.services.connection_pool_manager import ConnectionPoolManager
except ImportError:
    # Handle import path for standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.services.connection_pool_manager import ConnectionPoolManager

async def setup_knowledge_graph():
    """Setup Knowledge Graph with Azure-compatible schema"""
    load_dotenv()
    
    print("🚀 Setting up Knowledge Graph...")
    
    pool_manager = ConnectionPoolManager()
    await pool_manager.initialize()
    
    try:
        # Read and execute Azure-compatible schema
        schema_path = os.path.join(os.path.dirname(__file__), 'KNOWLEDGE_GRAPH_SCHEMA_AZURE.sql')
        
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        print("📊 Executing Knowledge Graph schema...")
        
        async with pool_manager.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)
        
        print("✅ Knowledge Graph schema created successfully!")
        
        # Validate setup
        async with pool_manager.postgres_pool.acquire() as conn:
            # Check tables exist
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'kg_%'
                ORDER BY table_name
            """)
            
            print(f"📋 Created {len(tables)} KG tables:")
            for table in tables:
                print(f"   ✅ {table['table_name']}")
            
            # Check RLS is enabled
            rls_tables = await conn.fetch("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename LIKE 'kg_%'
                AND rowsecurity = true
            """)
            
            print(f"🔒 RLS enabled on {len(rls_tables)} tables")
            
            # Check indexes
            indexes = await conn.fetch("""
                SELECT indexname FROM pg_indexes 
                WHERE schemaname = 'public' 
                AND indexname LIKE 'idx_kg_%'
                ORDER BY indexname
            """)
            
            print(f"🚀 Created {len(indexes)} optimized indexes")
        
        print("\n✅ Knowledge Graph setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge Graph setup failed: {e}")
        return False
    finally:
        await pool_manager.close()

if __name__ == "__main__":
    success = asyncio.run(setup_knowledge_graph())
    sys.exit(0 if success else 1)
