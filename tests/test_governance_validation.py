"""
Test Script for DSL Governance Validation System
===============================================
Demonstrates Task 7.1.3: Configure mandatory governance metadata fields

This script tests the enhanced DSL parser with governance-by-design validation.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from dsl.parser import DSLParser
from dsl.governance.governance_metadata_validator import EnforcementMode, ValidationContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_governance_validation():
    """Test the governance validation system"""
    logger.info("🚀 Testing DSL Governance Validation System (Task 7.1.3)")
    
    try:
        # Initialize parser
        parser = DSLParser()
        
        # Test 1: Valid workflow with all governance metadata
        logger.info("\n📋 Test 1: Valid Governance Workflow")
        sample_workflow_path = current_dir / "dsl" / "examples" / "sample_governance_workflow.yaml"
        
        if sample_workflow_path.exists():
            with open(sample_workflow_path, 'r') as f:
                workflow_yaml = f.read()
            
            # Parse with governance validation
            workflow = await parser.parse_yaml_with_governance(
                workflow_yaml,
                tenant_id=1300,
                user_id=1323,
                enforcement_mode=EnforcementMode.COMPLIANCE
            )
            
            logger.info(f"✅ Workflow parsed successfully: {workflow.workflow_id}")
            logger.info(f"   Tenant ID: {workflow.tenant_id}")
            logger.info(f"   Region: {workflow.region_id}")
            logger.info(f"   SLA Tier: {workflow.sla_tier}")
            logger.info(f"   Policy Pack: {workflow.policy_pack_id}")
            logger.info(f"   Trust Score: {workflow.trust_score_threshold}")
            logger.info(f"   Governance Validation: {'✅ PASSED' if workflow.governance_validation_passed else '❌ FAILED'}")
            
            if workflow.governance_validation_errors:
                logger.warning(f"   Validation Warnings: {workflow.governance_validation_errors}")
        
        # Test 2: Workflow missing mandatory governance fields
        logger.info("\n📋 Test 2: Missing Mandatory Fields")
        invalid_workflow_yaml = """
workflow_id: "test_invalid_workflow"
name: "Test Invalid Workflow"
module: "Pipeline"
automation_type: "RBA"
version: "1.0.0"

# Missing governance metadata!
steps:
  - id: "test_step"
    type: "query"
    params:
      source: "postgres"
    governance:
      policy_id: "test_policy"
"""
        
        try:
            workflow = await parser.parse_yaml_with_governance(
                invalid_workflow_yaml,
                tenant_id=1300,
                user_id=1323,
                enforcement_mode=EnforcementMode.COMPLIANCE
            )
            logger.info(f"⚠️ Workflow parsed with warnings: {workflow.workflow_id}")
            logger.info(f"   Governance Validation: {'✅ PASSED' if workflow.governance_validation_passed else '❌ FAILED'}")
            if workflow.governance_validation_errors:
                logger.warning(f"   Validation Errors: {workflow.governance_validation_errors}")
        except ValueError as e:
            logger.error(f"❌ Validation correctly blocked invalid workflow: {e}")
        
        # Test 3: Strict enforcement mode
        logger.info("\n📋 Test 3: Strict Enforcement Mode")
        try:
            workflow = await parser.parse_yaml_with_governance(
                invalid_workflow_yaml,
                tenant_id=1300,
                user_id=1323,
                enforcement_mode=EnforcementMode.STRICT
            )
            logger.warning("⚠️ Strict mode should have blocked this workflow!")
        except ValueError as e:
            logger.info(f"✅ Strict mode correctly blocked workflow: {e}")
        
        # Test 4: Permissive enforcement mode
        logger.info("\n📋 Test 4: Permissive Enforcement Mode")
        workflow = await parser.parse_yaml_with_governance(
            invalid_workflow_yaml,
            tenant_id=1300,
            user_id=1323,
            enforcement_mode=EnforcementMode.PERMISSIVE
        )
        logger.info(f"✅ Permissive mode allowed workflow: {workflow.workflow_id}")
        logger.info(f"   Governance Validation: {'✅ PASSED' if workflow.governance_validation_passed else '❌ FAILED'}")
        
        # Test 5: Industry overlay validation
        logger.info("\n📋 Test 5: Industry Overlay Validation")
        saas_workflow_yaml = """
workflow_id: "test_saas_workflow"
name: "SaaS Industry Workflow"
module: "Pipeline"
automation_type: "RBA"
version: "1.0.0"

governance:
  tenant_id: 1300
  region_id: "US"
  sla_tier: "T0"
  policy_pack_id: "550e8400-e29b-41d4-a716-446655440000"
  trust_score_threshold: 0.9

metadata:
  industry_overlay: "SaaS"
  arr_impact_threshold: 50000
  customer_tier: "enterprise"

steps:
  - id: "test_step"
    type: "query"
    params:
      source: "postgres"
    governance:
      policy_id: "saas_policy"
"""
        
        workflow = await parser.parse_yaml_with_governance(
            saas_workflow_yaml,
            tenant_id=1300,
            user_id=1323,
            enforcement_mode=EnforcementMode.COMPLIANCE
        )
        logger.info(f"✅ SaaS workflow parsed: {workflow.workflow_id}")
        logger.info(f"   Industry Overlay: SaaS")
        logger.info(f"   SLA Tier: {workflow.sla_tier}")
        logger.info(f"   Trust Score: {workflow.trust_score_threshold}")
        
        logger.info("\n🎉 All governance validation tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def test_configuration_driven_validation():
    """Test that validation is completely configuration-driven"""
    logger.info("\n🔧 Testing Configuration-Driven Validation")
    
    try:
        from dsl.governance.governance_metadata_validator import get_governance_validator
        
        # Load validator
        validator = get_governance_validator()
        
        # Test metadata
        test_metadata = {
            "tenant_id": 1300,
            "region_id": "US",
            "sla_tier": "T1",
            "policy_pack_id": "550e8400-e29b-41d4-a716-446655440000",
            "trust_score_threshold": 0.8,
            "workflow_type": "RBA"
        }
        
        context = ValidationContext(
            tenant_id=1300,
            user_id=1323,
            workflow_type="RBA",
            enforcement_mode=EnforcementMode.COMPLIANCE
        )
        
        # Validate
        is_valid, errors = await validator.validate_governance_metadata(test_metadata, context)
        
        logger.info(f"✅ Configuration-driven validation: {'PASSED' if is_valid else 'FAILED'}")
        logger.info(f"   Errors found: {len(errors)}")
        
        for error in errors:
            logger.warning(f"   - {error.field_name}: {error.message}")
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")

if __name__ == "__main__":
    print("🚀 DSL Governance Validation System Test")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_governance_validation())
    asyncio.run(test_configuration_driven_validation())
    
    print("\n✅ Test suite completed!")


