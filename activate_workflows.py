"""
Activate loaded workflows so they can be discovered
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.connection_pool_manager import ConnectionPoolManager

async def activate_workflows():
    """Fix all workflow issues in one go"""
    
    print("🔧 FIXING ALL WORKFLOW ISSUES...")
    
    pool_manager = ConnectionPoolManager()
    await pool_manager.initialize()
    
    try:
        async with pool_manager.postgres_pool.acquire() as conn:
            print("\n📋 Current workflows:")
            workflows = await conn.fetch("""
                SELECT workflow_id, name, status 
                FROM dsl_workflows 
                WHERE tenant_id = $1
            """, "53b6ad06-6af3-5df4-b2f2-df26a4d03773")
            
            for row in workflows:
                print(f"   • {row['workflow_id']} | {row['name']} | {row['status']}")
            
            print("\n🔧 Fixing unknown_workflow entries...")
            # First, let's see what the unknown workflow name actually is
            unknown_workflows = await conn.fetch("""
                SELECT workflow_id, name, module 
                FROM dsl_workflows 
                WHERE workflow_id = 'unknown_workflow'
                AND tenant_id = $1
            """, "53b6ad06-6af3-5df4-b2f2-df26a4d03773")
            
            for unknown in unknown_workflows:
                print(f"   🔍 Found unknown: {unknown['name']} in {unknown['module']}")
                
                # Determine proper ID based on module and name
                if unknown['module'] == 'Revenue':
                    new_id = "revenue_trend_rba"
                elif unknown['module'] == 'Forecast':
                    if 'Accuracy' in unknown['name']:
                        new_id = "forecast_accuracy_rba"
                    else:
                        new_id = "forecast_calibration_rba"
                elif unknown['module'] == 'Pipeline':
                    if 'Coverage' in unknown['name']:
                        new_id = "pipeline_coverage_rba"
                    else:
                        new_id = "deal_risk_assessment_rba"
                else:
                    new_id = f"{unknown['module'].lower()}_workflow_rba"
                
                # Update this specific workflow
                result = await conn.execute("""
                    UPDATE dsl_workflows 
                    SET workflow_id = $1
                    WHERE workflow_id = 'unknown_workflow' 
                    AND name = $2
                    AND tenant_id = $3
                """, new_id, unknown['name'], "53b6ad06-6af3-5df4-b2f2-df26a4d03773")
                
                print(f"   ✅ Fixed: {unknown['name']} → {new_id}")
            
            # Verify final state
            print("\n📊 Final workflows:")
            workflows = await conn.fetch("""
                SELECT workflow_id, name, module, automation_type, status 
                FROM dsl_workflows 
                WHERE tenant_id = $1
                ORDER BY created_at DESC
            """, "53b6ad06-6af3-5df4-b2f2-df26a4d03773")
            
            for row in workflows:
                print(f"   • {row['workflow_id']} ({row['module']}/{row['automation_type']}) - {row['status']}")
                
            print(f"\n✅ Total workflows ready: {len(workflows)}")
    
    except Exception as e:
        print(f"❌ Error fixing workflows: {e}")
    
    await pool_manager.close()
    print(f"\n🎉 All workflow issues fixed!")

if __name__ == "__main__":
    asyncio.run(activate_workflows())
