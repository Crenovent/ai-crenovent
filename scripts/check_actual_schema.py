#!/usr/bin/env python3
"""
Check the actual database schema to understand table structure
"""

import asyncio
import sys
import os
sys.path.append('src')

async def check_actual_schema():
    """Check the actual database schema"""
    try:
        from src.services.connection_pool_manager import pool_manager
        
        # Initialize pool manager
        await pool_manager.initialize()
        
        async with pool_manager.postgres_pool.acquire() as conn:
            # Get strategic_account_plans table schema
            print("📋 Strategic Account Plans Table Schema:")
            schema = await conn.fetch("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'strategic_account_plans'
                ORDER BY ordinal_position
            """)
            
            for col in schema:
                print(f"   {col['column_name']}: {col['data_type']} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
            
            # Count total plans
            total_plans = await conn.fetchval("SELECT COUNT(*) FROM strategic_account_plans")
            print(f"\n📊 Total Strategic Plans: {total_plans}")
            
            # Get sample data
            if total_plans > 0:
                sample = await conn.fetchrow("SELECT * FROM strategic_account_plans ORDER BY created_at DESC LIMIT 1")
                print(f"\n📄 Sample Plan Fields:")
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"   {key}: {str(value)[:100]}...")
                    else:
                        print(f"   {key}: {value}")
            
            # Check users table
            print(f"\n👥 Users Table Schema:")
            users_schema = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'users'
                ORDER BY ordinal_position
            """)
            
            for col in users_schema:
                print(f"   {col['column_name']}: {col['data_type']} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
        
        await pool_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(check_actual_schema())
