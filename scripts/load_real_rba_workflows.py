#!/usr/bin/env python3
"""
Load Real RBA Workflows
======================

Load the new real business logic RBA workflows into the database
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import yaml

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.services.connection_pool_manager import ConnectionPoolManager
    from dsl.parser import DSLParser, DSLWorkflow
    from dsl.storage import WorkflowStorage
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def load_real_rba_workflows():
    """Load the new real RBA workflows with business logic"""
    load_dotenv()
    
    print("🚀 Loading Real RBA Workflows...")
    
    pool_manager = ConnectionPoolManager()
    await pool_manager.initialize()
    
    try:
        storage = WorkflowStorage(pool_manager)
        parser = DSLParser()
        
        # Real RBA workflow files (updated with policy-aware endpoints)
        workflow_files = [
            'forecast_variance_alert_rba.yml',
            'pipeline_coverage_monitor_rba.yml', 
            'revenue_recognition_rba.yml'
        ]
        
        print("🔄 Loading updated workflows with real data integration...")
        
        tenant_id = int(os.getenv('MAIN_TENANT_ID', '1300'))
        loaded_count = 0
        
        for workflow_file in workflow_files:
            workflow_path = os.path.join('dsl', 'workflows', workflow_file)
            
            if not os.path.exists(workflow_path):
                print(f"⚠️ Workflow file not found: {workflow_path}")
                continue
            
            print(f"📋 Loading: {workflow_file}")
            
            try:
                # Parse workflow
                with open(workflow_path, 'r', encoding='utf-8') as f:
                    workflow_yaml = f.read()
                
                workflow = parser.parse_yaml(workflow_yaml)
                
                # Save to database (using valid user_id from your system)
                success = await storage.save_workflow(
                    workflow=workflow,
                    created_by_user_id=1323,  # Active user from your system
                    tenant_id=tenant_id
                )
                
                if success:
                    print(f"   ✅ Loaded: {workflow.name}")
                    loaded_count += 1
                else:
                    print(f"   ❌ Failed to save: {workflow.name}")
                
            except Exception as e:
                print(f"   ❌ Failed to load {workflow_file}: {e}")
        
        print(f"\n✅ Successfully loaded {loaded_count}/{len(workflow_files)} real RBA workflows!")
        
        # Verify loading
        await verify_loaded_workflows(storage, tenant_id)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load workflows: {e}")
        return False
    finally:
        await pool_manager.close()

async def verify_loaded_workflows(storage: WorkflowStorage, tenant_id: str):
    """Verify the workflows were loaded correctly"""
    print("\n🔍 Verifying loaded workflows...")
    
    try:
        workflows = await storage.list_workflows(tenant_id)
        
        print(f"📊 Total workflows in database: {len(workflows)}")
        
        for workflow in workflows:
            print(f"   • {workflow['name']} ({workflow['module']}) - {workflow['automation_type']}")
        
        # Count by module
        modules = {}
        for workflow in workflows:
            module = workflow['module']
            modules[module] = modules.get(module, 0) + 1
        
        print(f"\n📈 Workflows by module:")
        for module, count in modules.items():
            print(f"   {module}: {count} workflows")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    success = asyncio.run(load_real_rba_workflows())
    sys.exit(0 if success else 1)
