#!/usr/bin/env python3
"""
Check database schema and existing strategic plans
"""

import asyncio
import sys
import os
sys.path.append('src')

async def check_schema_and_plans():
    """Check database schema and existing plans"""
    try:
        from src.services.connection_pool_manager import pool_manager
        
        # Initialize pool manager
        await pool_manager.initialize()
        
        async with pool_manager.postgres_pool.acquire() as conn:
            # Get table schema
            schema = await conn.fetch("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'strategic_account_plans'
                ORDER BY ordinal_position
            """)
            
            print(f"📋 Strategic Account Plans Table Schema:")
            for col in schema:
                print(f"   {col['column_name']}: {col['data_type']} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
            
            # Count total plans
            total_plans = await conn.fetchval("SELECT COUNT(*) FROM strategic_account_plans")
            print(f"\n📊 Total Strategic Plans: {total_plans}")
            
            if total_plans > 0:
                # Get sample of existing plans with available columns
                recent_plans = await conn.fetch("""
                    SELECT plan_name, account_id, annual_revenue, account_tier, 
                           short_term_goals, long_term_goals, key_opportunities,
                           known_risks, stakeholders, activities, created_at
                    FROM strategic_account_plans 
                    ORDER BY created_at DESC 
                    LIMIT 3
                """)
                
                print(f"\n📋 Recent Plans Sample:")
                for i, plan in enumerate(recent_plans, 1):
                    print(f"   {i}. Plan: {plan['plan_name']}")
                    print(f"      Account: {plan['account_id']}")
                    print(f"      Revenue: {plan['annual_revenue']}")
                    print(f"      Tier: {plan['account_tier']}")
                    print(f"      Goals: {plan['short_term_goals'][:100] if plan['short_term_goals'] else 'None'}...")
                    print(f"      Created: {plan['created_at']}")
                    print()
                
                # Get field completion statistics
                completion_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(plan_name) as has_plan_name,
                        COUNT(account_id) as has_account_id,
                        COUNT(annual_revenue) as has_revenue,
                        COUNT(short_term_goals) as has_short_goals,
                        COUNT(long_term_goals) as has_long_goals,
                        COUNT(stakeholders) as has_stakeholders,
                        COUNT(activities) as has_activities
                    FROM strategic_account_plans
                """)
                
                print(f"📈 Field Completion Stats:")
                for field, count in completion_stats.items():
                    if field != 'total':
                        percentage = (count / completion_stats['total']) * 100
                        print(f"   {field}: {count}/{completion_stats['total']} ({percentage:.1f}%)")
                
                # Get most common patterns
                patterns = await conn.fetch("""
                    SELECT account_tier, COUNT(*) as count
                    FROM strategic_account_plans 
                    WHERE account_tier IS NOT NULL
                    GROUP BY account_tier
                    ORDER BY count DESC
                """)
                
                print(f"\n🎯 Account Tier Patterns:")
                for pattern in patterns:
                    print(f"   {pattern['account_tier']}: {pattern['count']} plans")
        
        await pool_manager.close()
        return total_plans
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("🔍 Checking database schema and strategic plans...")
    plan_count = asyncio.run(check_schema_and_plans())
    print(f"\n✅ Analysis complete - {plan_count} plans found")
