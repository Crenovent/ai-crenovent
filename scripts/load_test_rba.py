"""
Load the test RBA agent to verify the system works
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.connection_pool_manager import ConnectionPoolManager
from dsl.parser import DSLParser
from dsl.storage import WorkflowStorage

async def load_test_rba():
    """Load test RBA workflow"""
    
    print("🧪 Loading Test RBA Agent...")
    
    pool_manager = ConnectionPoolManager()
    await pool_manager.initialize()
    
    parser = DSLParser()
    storage = WorkflowStorage(pool_manager)
    
    try:
        # Parse the test workflow
        dsl_workflow = parser.parse_file("test_rba_agent.yml")
        
        # Convert to dict for storage  
        workflow_data = {
            'workflow_id': dsl_workflow.workflow_id,
            'name': dsl_workflow.name,
            'description': dsl_workflow.metadata.get('description', 'Test RBA Agent'),
            'version': dsl_workflow.version,
            'module': 'Forecast',
            'automation_type': 'RBA',
            'steps': [
                {
                    'id': step.id,
                    'type': step.type,
                    'params': step.params,
                    'outputs': step.outputs,
                    'governance': step.governance,
                    'next_steps': step.next_steps
                } for step in dsl_workflow.steps
            ],
            'triggers': dsl_workflow.triggers,
            'metadata': dsl_workflow.metadata
        }
        
        # Save to database
        workflow_id = await storage.save_workflow(
            workflow_data=workflow_data,
            tenant_id="53b6ad06-6af3-5df4-b2f2-df26a4d03773",
            created_by_user_id=1323,
            version=1
        )
        
        print(f"✅ Test RBA Agent loaded: {workflow_id}")
        
        # Verify it's in the database
        async with pool_manager.postgres_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT workflow_id, name, module, automation_type, status 
                FROM dsl_workflows 
                WHERE tenant_id = $1
                ORDER BY created_at DESC
            """, "53b6ad06-6af3-5df4-b2f2-df26a4d03773")
            
            print(f"\n📊 Total workflows in database: {len(result)}")
            for row in result:
                print(f"   • {row['name']} ({row['module']}/{row['automation_type']}) - {row['status']}")
        
    except Exception as e:
        print(f"❌ Failed to load test RBA: {e}")
    
    await pool_manager.close()

if __name__ == "__main__":
    asyncio.run(load_test_rba())
