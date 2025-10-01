"""
DSL Setup Script
Sets up the DSL environment and loads initial workflows and policies
"""

import asyncio
import os
import sys
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# DSL imports
from dsl import DSLParser, WorkflowStorage, PolicyEngine
from src.services.connection_pool_manager import ConnectionPoolManager

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DSLSetup:
    """DSL setup and initialization"""
    
    def __init__(self):
        self.pool_manager = None
        self.parser = None
        self.storage = None
        self.policy_engine = None
    
    async def initialize(self):
        """Initialize DSL components"""
        try:
            # Initialize connection pool
            self.pool_manager = ConnectionPoolManager()
            await self.pool_manager.initialize()
            
            # Initialize DSL components
            self.parser = DSLParser()
            self.storage = WorkflowStorage(self.pool_manager)
            self.policy_engine = PolicyEngine(self.pool_manager)
            
            logger.info("✅ DSL components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize DSL: {e}")
            raise
    
    async def setup_policy_packs(self):
        """Set up initial policy packs"""
        logger.info("📋 Setting up policy packs...")
        
        # SOX Compliance Policy Pack
        sox_rules = [
            {
                "type": "evidence_required",
                "name": "SOX Evidence Requirement",
                "description": "All financial operations must generate audit evidence",
                "mandatory": True
            },
            {
                "type": "approval_required",
                "name": "SOX High Value Approval",
                "description": "Plans with revenue >$100K require manager approval",
                "condition": "annual_revenue > 100000",
                "approver_role": "Sales Manager",
                "mandatory": True
            },
            {
                "type": "data_retention",
                "name": "SOX Data Retention",
                "description": "Evidence must be retained for 7 years",
                "retention_days": 2555,
                "mandatory": True
            }
        ]
        
        await self.policy_engine.create_policy_pack(
            policy_pack_id="account_planning_sox",
            name="SOX Compliance for Account Planning",
            description="SOX compliance policies for account planning workflows",
            rules=sox_rules,
            tenant_id=int(os.getenv('MAIN_TENANT_ID', '1300')),
            created_by_user_id=1323,
            industry="SaaS",
            compliance_standards=["SOX"],
            is_global=False
        )
        
        # GDPR Compliance Policy Pack
        gdpr_rules = [
            {
                "type": "consent_required",
                "name": "GDPR Consent",
                "description": "User consent required for data processing",
                "mandatory": True
            },
            {
                "type": "data_minimization",
                "name": "GDPR Data Minimization",
                "description": "Only necessary fields should be processed",
                "allowed_fields": [
                    "plan_name", "account_id", "account_owner", "industry",
                    "annual_revenue", "account_tier", "region_territory",
                    "short_term_goals", "long_term_goals", "revenue_growth_target",
                    "key_opportunities", "known_risks", "communication_cadence"
                ],
                "mandatory": True
            },
            {
                "type": "field_required",
                "name": "GDPR Required Fields",
                "description": "Minimum required fields for account planning",
                "fields": ["plan_name", "account_id", "account_owner"],
                "mandatory": True
            }
        ]
        
        await self.policy_engine.create_policy_pack(
            policy_pack_id="gdpr_data_extraction",
            name="GDPR Data Processing Compliance",
            description="GDPR compliance for data extraction and processing",
            rules=gdpr_rules,
            tenant_id=int(os.getenv('MAIN_TENANT_ID', '1300')),
            created_by_user_id=1323,
            industry="SaaS",
            compliance_standards=["GDPR"],
            is_global=False
        )
        
        # Default Policy Pack
        default_rules = [
            {
                "type": "field_required",
                "name": "Basic Required Fields",
                "fields": ["plan_name", "account_id"],
                "mandatory": True
            },
            {
                "type": "evidence_required",
                "name": "Basic Evidence",
                "description": "Basic audit trail required",
                "mandatory": True
            }
        ]
        
        await self.policy_engine.create_policy_pack(
            policy_pack_id="default_policy",
            name="Default DSL Policy",
            description="Default policy pack for DSL workflows",
            rules=default_rules,
            tenant_id=int(os.getenv('MAIN_TENANT_ID', '1300')),
            created_by_user_id=1323,
            industry="SaaS",
            compliance_standards=[],
            is_global=True
        )
        
        logger.info("✅ Policy packs created successfully")
    
    async def setup_workflows(self):
        """Set up initial workflows"""
        logger.info("🔧 Setting up workflows...")
        
        # Load and save the account planning workflow
        workflow_file = "dsl/workflows/account_planning_rba.yml"
        
        if os.path.exists(workflow_file):
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow_yaml = f.read()
            
            try:
                workflow = self.parser.parse_yaml(workflow_yaml)
                
                success = await self.storage.save_workflow(
                    workflow=workflow,
                    created_by_user_id=1323,
                    tenant_id=int(os.getenv('MAIN_TENANT_ID', '1300'))
                )
                
                if success:
                    logger.info(f"✅ Saved workflow: {workflow.workflow_id}")
                else:
                    logger.error(f"❌ Failed to save workflow: {workflow.workflow_id}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing workflow {workflow_file}: {e}")
        else:
            logger.warning(f"⚠️ Workflow file not found: {workflow_file}")
    
    async def verify_setup(self):
        """Verify the setup was successful"""
        logger.info("🔍 Verifying DSL setup...")
        
        try:
            # Check workflows
            workflows = await self.storage.list_workflows(tenant_id=int(os.getenv('MAIN_TENANT_ID', '1300')))
            logger.info(f"📋 Found {len(workflows)} workflow(s)")
            
            for workflow in workflows:
                logger.info(f"  - {workflow['workflow_id']}: {workflow['name']}")
            
            # Check policies
            test_policy = await self.policy_engine.load_policy_pack(
                policy_pack_id="account_planning_sox",
                tenant_id=int(os.getenv('MAIN_TENANT_ID', '1300'))
            )
            
            if test_policy:
                logger.info("✅ Policy packs verified")
            else:
                logger.error("❌ Policy pack verification failed")
                return False
            
            logger.info("✅ DSL setup verification complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup verification failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.pool_manager:
                await self.pool_manager.close()
            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    async def run_setup(self):
        """Run complete setup process"""
        logger.info("🚀 Starting DSL Setup...")
        
        try:
            await self.initialize()
            await self.setup_policy_packs()
            await self.setup_workflows()
            
            success = await self.verify_setup()
            
            if success:
                logger.info("🎉 DSL setup completed successfully!")
                logger.info("")
                logger.info("Next steps:")
                logger.info("1. Run the DSL server: python dsl_server.py")
                logger.info("2. Test the system: python test_dsl_complete.py")
                logger.info("3. Try the API endpoints at http://localhost:8001/docs")
            else:
                logger.error("❌ DSL setup failed during verification")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ DSL setup failed: {e}")
            return False
            
        finally:
            await self.cleanup()

async def main():
    """Main setup execution"""
    setup = DSLSetup()
    success = await setup.run_setup()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
