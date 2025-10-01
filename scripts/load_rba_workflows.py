"""
Load all RBA workflow files into the database
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.connection_pool_manager import ConnectionPoolManager
from dsl.parser import DSLParser
from dsl.storage import WorkflowStorage

async def load_all_rba_workflows():
    """Load all RBA workflow YAML files into the database"""
    
    print("🚀 Loading All RBA Workflows...")
    
    # Initialize components
    pool_manager = ConnectionPoolManager()
    await pool_manager.initialize()
    
    parser = DSLParser()
    storage = WorkflowStorage(pool_manager)
    
    # Define workflow files and their metadata
    workflow_files = [
        {
            'file': 'forecast_calibration_rba.yml',
            'name': 'Forecast Calibration',
            'description': 'Automated forecast calibration and accuracy analysis',
            'module': 'Forecast',
            'automation_type': 'RBA'
        },
        {
            'file': 'forecast_accuracy_rba.yml', 
            'name': 'Forecast Accuracy Analysis',
            'description': 'Deep analysis of forecast accuracy patterns and trends',
            'module': 'Forecast',
            'automation_type': 'RBA'
        },
        {
            'file': 'pipeline_coverage_rba.yml',
            'name': 'Pipeline Coverage Analysis',
            'description': 'Automated pipeline coverage and gap analysis',
            'module': 'Pipeline',
            'automation_type': 'RBA'
        },
        {
            'file': 'deal_risk_assessment_rba.yml',
            'name': 'Deal Risk Assessment',
            'description': 'Automated risk scoring and assessment for deals',
            'module': 'Pipeline',
            'automation_type': 'RBA'
        },
        {
            'file': 'revenue_trend_rba.yml',
            'name': 'Revenue Trend Analysis',
            'description': 'Comprehensive revenue trend analysis and forecasting',
            'module': 'Revenue',
            'automation_type': 'RBA'
        }
    ]
    
    workflows_dir = 'dsl/workflows'
    loaded_count = 0
    
    for workflow_info in workflow_files:
        try:
            file_path = os.path.join(workflows_dir, workflow_info['file'])
            
            if os.path.exists(file_path):
                print(f"\n📋 Loading: {workflow_info['name']}")
                
                # Parse the workflow
                dsl_workflow = parser.parse_file(file_path)
                
                # Update the workflow with our metadata
                dsl_workflow.module = workflow_info['module']
                dsl_workflow.automation_type = workflow_info['automation_type']
                dsl_workflow.metadata['description'] = workflow_info['description']
                
                # Save to database
                success = await storage.save_workflow(
                    workflow=dsl_workflow,
                    tenant_id=1300,  # Main tenant ID
                    created_by_user_id=1323  # Test user
                )
                
                if success:
                    print(f"   ✅ Saved as: {dsl_workflow.workflow_id}")
                    loaded_count += 1
                else:
                    print(f"   ❌ Failed to save: {dsl_workflow.workflow_id}")
                
            else:
                print(f"   ❌ File not found: {file_path}")
                
        except Exception as e:
            print(f"   ❌ Failed to load {workflow_info['file']}: {e}")
    
    print(f"\n🎉 Successfully loaded {loaded_count}/{len(workflow_files)} RBA workflows!")
    
    # Verify by listing all workflows
    print(f"\n🔍 Verifying workflows in database...")
    
    try:
        async with pool_manager.postgres_pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT workflow_id, name, module, automation_type, status 
                FROM dsl_workflows 
                WHERE tenant_id = $1 
                ORDER BY created_at DESC
            """, "53b6ad06-6af3-5df4-b2f2-df26a4d03773")
        
        print(f"📊 Total workflows in database: {len(result)}")
        for row in result:
            print(f"   • {row['name']} ({row['module']}/{row['automation_type']}) - {row['status']}")
    
    except Exception as e:
        print(f"❌ Error verifying workflows: {e}")
    
    await pool_manager.close()
    print(f"\n✅ Workflow loading complete!")

if __name__ == "__main__":
    asyncio.run(load_all_rba_workflows())
