#!/usr/bin/env python3
"""
Check existing strategic plans in database
"""

import asyncio
import sys
import os
sys.path.append('src')

async def check_existing_plans():
    """Check how many strategic plans exist"""
    try:
        from src.services.connection_pool_manager import pool_manager
        
        # Initialize pool manager
        await pool_manager.initialize()
        
        # Get plan count
        async with pool_manager.postgres_pool.acquire() as conn:
            # Count total plans
            total_plans = await conn.fetchval("SELECT COUNT(*) FROM strategic_account_plans")
            print(f"📊 Total Strategic Plans: {total_plans}")
            
            # Get recent plans sample
            if total_plans > 0:
                recent_plans = await conn.fetch("""
                    SELECT plan_name, account_id, annual_revenue, account_tier, 
                           created_at, planning_period
                    FROM strategic_account_plans 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                
                print(f"\n📋 Recent Plans Sample:")
                for i, plan in enumerate(recent_plans, 1):
                    print(f"   {i}. {plan['plan_name']} - {plan['account_id']}")
                    print(f"      Revenue: ${plan['annual_revenue']}, Tier: {plan['account_tier']}")
                    print(f"      Period: {plan['planning_period']}, Created: {plan['created_at'].strftime('%Y-%m-%d')}")
                    print()
                
                # Get account patterns
                account_stats = await conn.fetch("""
                    SELECT account_tier, COUNT(*) as count,
                           AVG(CAST(annual_revenue AS BIGINT)) as avg_revenue
                    FROM strategic_account_plans 
                    WHERE annual_revenue IS NOT NULL AND annual_revenue ~ '^[0-9]+$'
                    GROUP BY account_tier
                    ORDER BY count DESC
                """)
                
                print(f"📈 Account Patterns:")
                for stat in account_stats:
                    avg_rev = int(stat['avg_revenue']) if stat['avg_revenue'] else 0
                    print(f"   {stat['account_tier']}: {stat['count']} plans, Avg Revenue: ${avg_rev:,}")
                
            else:
                print("❌ No strategic plans found in database")
        
        await pool_manager.close()
        return total_plans
        
    except Exception as e:
        print(f"❌ Error checking plans: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("🔍 Checking existing strategic plans...")
    plan_count = asyncio.run(check_existing_plans())
    print(f"\n✅ Found {plan_count} strategic plans in database")
