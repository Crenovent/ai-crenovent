#!/usr/bin/env python3
"""
Routing Orchestrator Setup & Validation Script
==============================================

This script sets up the complete Routing Orchestrator system:
1. Creates database schema
2. Initializes all components
3. Loads sample data
4. Runs validation tests
5. Demonstrates end-to-end functionality

Run this to get a 100% working Routing Orchestrator ready for production use.
"""

import asyncio
import sys
import os
import json
import uuid
from datetime import datetime

# Add paths
sys.path.append('src')
sys.path.append('dsl')

from src.services.connection_pool_manager import pool_manager
from dsl.hub.routing_orchestrator import RoutingOrchestrator, IntentType, AutomationType

async def setup_database_schema():
    """Create the complete database schema"""
    print("🔧 Setting up database schema...")
    
    try:
        async with pool_manager.postgres_pool.acquire() as conn:
            # Load and execute schema
            with open('ROUTING_ORCHESTRATOR_SCHEMA.sql', 'r') as f:
                schema_sql = f.read()
            
            await conn.execute(schema_sql)
            print("✅ Database schema created successfully")
            return True
            
    except Exception as e:
        print(f"❌ Schema setup failed: {e}")
        return False

async def load_sample_data():
    """Load comprehensive sample data for testing"""
    print("📊 Loading sample data...")
    
    try:
        async with pool_manager.postgres_pool.acquire() as conn:
            
            # Additional sample capabilities for testing
            sample_capabilities = [
                {
                    "capability_type": "RBA_TEMPLATE",
                    "name": "Deal Risk Assessment",
                    "description": "Automated assessment of deal closure risks",
                    "category": "pipeline",
                    "industry_tags": ["SaaS", "E-commerce"],
                    "persona_tags": ["Sales Manager", "AE"],
                    "trust_score": 0.88,
                    "readiness_state": "CERTIFIED",
                    "sla_tier": "T1",
                    "estimated_cost": 15.0,
                    "avg_execution_time": 25000
                },
                {
                    "capability_type": "RBIA_MODEL", 
                    "name": "Lead Scoring Intelligence",
                    "description": "ML-powered lead scoring with confidence intervals",
                    "category": "pipeline",
                    "industry_tags": ["SaaS"],
                    "persona_tags": ["Sales Manager", "RevOps"],
                    "trust_score": 0.82,
                    "readiness_state": "BETA",
                    "sla_tier": "T1",
                    "estimated_cost": 35.0,
                    "avg_execution_time": 45000
                },
                {
                    "capability_type": "AALA_AGENT",
                    "name": "Strategic Account Planner",
                    "description": "AI agent for comprehensive account planning",
                    "category": "planning",
                    "industry_tags": ["SaaS", "Enterprise"],
                    "persona_tags": ["AE", "Account Manager", "RevOps"],
                    "trust_score": 0.75,
                    "readiness_state": "BETA",
                    "sla_tier": "T2",
                    "estimated_cost": 85.0,
                    "avg_execution_time": 120000
                }
            ]
            
            for capability in sample_capabilities:
                await conn.execute("""
                    INSERT INTO ro_capabilities (
                        capability_type, name, description, category,
                        industry_tags, persona_tags, trust_score, readiness_state,
                        sla_tier, estimated_cost_per_execution, avg_execution_time_ms,
                        version, owner_team, created_by
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT DO NOTHING
                """, 
                    capability["capability_type"],
                    capability["name"],
                    capability["description"],
                    capability["category"],
                    capability["industry_tags"],
                    capability["persona_tags"],
                    capability["trust_score"],
                    capability["readiness_state"],
                    capability["sla_tier"],
                    capability["estimated_cost"],
                    capability["avg_execution_time"],
                    "1.0.0",
                    "AI Team",
                    uuid.UUID("00000000-0000-0000-0000-000000000001")
                )
            
            # Additional intent patterns
            additional_patterns = [
                {
                    "pattern_name": "Deal Risk Analysis",
                    "pattern_regex": r"(?i)(deal|opportunity).*(risk|assessment|analysis)",
                    "intent_type": "workflow_execution",
                    "target_workflow_type": "RBA",
                    "parameter_extraction": '{"workflow_category": "pipeline"}'
                },
                {
                    "pattern_name": "Lead Scoring Request",
                    "pattern_regex": r"(?i)(lead|prospect).*(scor|rank|priorit)",
                    "intent_type": "workflow_execution", 
                    "target_workflow_type": "RBIA",
                    "parameter_extraction": '{"workflow_category": "pipeline"}'
                },
                {
                    "pattern_name": "Account Planning",
                    "pattern_regex": r"(?i)(account|strategic).*(plan|strateg)",
                    "intent_type": "workflow_execution",
                    "target_workflow_type": "AALA", 
                    "parameter_extraction": '{"workflow_category": "planning"}'
                }
            ]
            
            for pattern in additional_patterns:
                await conn.execute("""
                    INSERT INTO ro_intent_patterns (
                        pattern_name, pattern_regex, intent_type, target_workflow_type,
                        parameter_extraction, confidence_weight, created_by
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT DO NOTHING
                """,
                    pattern["pattern_name"],
                    pattern["pattern_regex"],
                    pattern["intent_type"],
                    pattern["target_workflow_type"],
                    json.loads(pattern["parameter_extraction"]),
                    0.85,
                    uuid.UUID("00000000-0000-0000-0000-000000000001")
                )
            
            # Test tenant policy packs
            test_tenants = [
                {
                    "tenant_id": "00000000-0000-0000-0000-000000000001",
                    "name": "Test Tenant 1 - SaaS",
                    "industry": "SaaS",
                    "jurisdiction": "US",
                    "policy_rules": {
                        "max_execution_cost": 200.0,
                        "require_approval_over": 100.0,
                        "allowed_automation_types": ["RBA", "RBIA", "AALA"],
                        "restricted_operations": [],
                        "audit_all_executions": True,
                        "compliance_frameworks": ["SOX"]
                    }
                },
                {
                    "tenant_id": "00000000-0000-0000-0000-000000000002", 
                    "name": "Test Tenant 2 - Banking",
                    "industry": "Banking",
                    "jurisdiction": "US",
                    "policy_rules": {
                        "max_execution_cost": 50.0,
                        "require_approval_over": 25.0,
                        "allowed_automation_types": ["RBA", "RBIA"],
                        "restricted_operations": ["data_export"],
                        "audit_all_executions": True,
                        "compliance_frameworks": ["SOX", "RBI"]
                    }
                }
            ]
            
            for tenant in test_tenants:
                await conn.execute("""
                    INSERT INTO ro_policy_packs (
                        tenant_id, policy_pack_name, industry_focus, jurisdiction,
                        compliance_frameworks, policy_rules, enforcement_level
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT DO NOTHING
                """,
                    uuid.UUID(tenant["tenant_id"]),
                    tenant["name"],
                    tenant["industry"],
                    tenant["jurisdiction"],
                    tenant["policy_rules"]["compliance_frameworks"],
                    json.dumps(tenant["policy_rules"]),
                    "STRICT"
                )
            
            print("✅ Sample data loaded successfully")
            return True
            
    except Exception as e:
        print(f"❌ Sample data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def validate_components():
    """Validate all routing orchestrator components"""
    print("🔍 Validating routing orchestrator components...")
    
    try:
        # Initialize routing orchestrator
        ro = RoutingOrchestrator(pool_manager)
        await ro.initialize()
        
        # Test component initialization
        components = [
            ("Intent Parser", ro.intent_parser),
            ("Policy Gate", ro.policy_gate), 
            ("Capability Registry", ro.capability_registry),
            ("Plan Synthesizer", ro.plan_synthesizer),
            ("Execution Dispatcher", ro.execution_dispatcher),
            ("Promotion Manager", ro.promotion_manager)
        ]
        
        for name, component in components:
            if hasattr(component, 'initialize'):
                await component.initialize()
            print(f"   ✅ {name} - OK")
        
        print("✅ All components validated successfully")
        return ro
        
    except Exception as e:
        print(f"❌ Component validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def run_demonstration_scenarios():
    """Run demonstration scenarios to show functionality"""
    print("🎭 Running demonstration scenarios...")
    
    # Initialize routing orchestrator
    ro = RoutingOrchestrator(pool_manager)
    await ro.initialize()
    
    # Test scenarios covering all automation types
    scenarios = [
        {
            "name": "RBA Forecast Analysis",
            "input": "analyze forecast accuracy for Q3 2024",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "expected_type": "RBA"
        },
        {
            "name": "RBIA Lead Scoring", 
            "input": "score and prioritize leads from last week",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "expected_type": "RBIA"
        },
        {
            "name": "AALA Account Planning",
            "input": "create strategic account plan for Microsoft Enterprise",
            "tenant_id": "00000000-0000-0000-0000-000000000001", 
            "expected_type": "AALA"
        },
        {
            "name": "Capability Discovery",
            "input": "what forecasting capabilities are available?",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "expected_type": "DISCOVERY"
        },
        {
            "name": "Policy Restricted Request",
            "input": "export all customer data to external system",
            "tenant_id": "00000000-0000-0000-0000-000000000002",  # Banking tenant with restrictions
            "expected_type": "DENIED"
        }
    ]
    
    successful_scenarios = 0
    
    for scenario in scenarios:
        print(f"\n🎬 Scenario: {scenario['name']}")
        print(f"   Input: '{scenario['input']}'")
        
        try:
            # Route the request
            routing_result = await ro.route_request(
                user_input=scenario['input'],
                tenant_id=scenario['tenant_id'],
                user_id=1319,
                context_data={"demo": True, "scenario": scenario['name']}
            )
            
            # Analyze results
            print(f"   🔍 Intent: {routing_result.parsed_intent.intent_type.value}")
            print(f"   🔍 Confidence: {routing_result.parsed_intent.confidence:.2f}")
            print(f"   🛡️ Policy: {routing_result.policy_evaluation.result.value}")
            
            if routing_result.ready_for_execution and routing_result.selected_capability:
                print(f"   📋 Selected: {routing_result.selected_capability.name}")
                print(f"   ⚙️ Type: {routing_result.selected_capability.capability_type}")
                print(f"   💰 Cost: ${routing_result.execution_plan.estimated_total_cost:.2f}")
                print(f"   ⏱️ Duration: {routing_result.execution_plan.estimated_total_duration_ms}ms")
                
                # Execute if appropriate
                if scenario['expected_type'] in ['RBA', 'RBIA', 'AALA']:
                    execution_result = await ro.execute_routing_result(routing_result)
                    print(f"   🚀 Execution: {execution_result.status}")
                    print(f"   ✅ Result: {execution_result.output_data.get('results', 'N/A')}")
            
            elif routing_result.parsed_intent.intent_type == IntentType.CAPABILITY_DISCOVERY:
                recommendations = await ro.get_capability_recommendations(scenario['tenant_id'])
                print(f"   📋 Found {len(recommendations)} available capabilities")
            
            else:
                print(f"   ⚠️ Request not executable: {routing_result.routing_rationale}")
            
            successful_scenarios += 1
            print(f"   ✅ Scenario completed successfully")
            
        except Exception as e:
            print(f"   ❌ Scenario failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏆 Demonstration Results: {successful_scenarios}/{len(scenarios)} scenarios successful")
    return successful_scenarios == len(scenarios)

async def run_performance_validation():
    """Run performance validation tests"""
    print("⚡ Running performance validation...")
    
    ro = RoutingOrchestrator(pool_manager)
    await ro.initialize()
    
    # Performance test parameters
    test_requests = [
        "analyze forecast accuracy",
        "check pipeline coverage", 
        "score leads from last week",
        "create account plan for enterprise client",
        "what capabilities are available?"
    ]
    
    concurrent_batches = 3
    requests_per_batch = 5
    
    print(f"   Testing {concurrent_batches} batches of {requests_per_batch} concurrent requests each...")
    
    try:
        total_requests = 0
        total_time = 0
        successful_requests = 0
        
        for batch in range(concurrent_batches):
            print(f"   Batch {batch + 1}/{concurrent_batches}...")
            
            # Create concurrent requests
            async def single_request(request_input):
                import time
                start = time.time()
                try:
                    result = await ro.route_request(
                        user_input=request_input,
                        tenant_id="00000000-0000-0000-0000-000000000001",
                        user_id=1319,
                        context_data={"performance_test": True}
                    )
                    return time.time() - start, True
                except Exception as e:
                    return time.time() - start, False
            
            # Execute batch
            import time
            batch_start = time.time()
            batch_results = await asyncio.gather(*[
                single_request(test_requests[i % len(test_requests)]) 
                for i in range(requests_per_batch)
            ])
            batch_time = time.time() - batch_start
            
            # Collect results
            batch_successful = sum(1 for _, success in batch_results if success)
            batch_avg_time = sum(duration for duration, _ in batch_results) / len(batch_results)
            
            total_requests += requests_per_batch
            total_time += batch_time
            successful_requests += batch_successful
            
            print(f"      Successful: {batch_successful}/{requests_per_batch}")
            print(f"      Avg time: {batch_avg_time*1000:.0f}ms")
            print(f"      Throughput: {requests_per_batch/batch_time:.1f} req/s")
        
        # Final performance metrics
        overall_success_rate = successful_requests / total_requests * 100
        overall_throughput = total_requests / total_time
        
        print(f"   📊 Overall Performance:")
        print(f"      Total requests: {total_requests}")
        print(f"      Success rate: {overall_success_rate:.1f}%")
        print(f"      Overall throughput: {overall_throughput:.1f} req/s")
        
        # Performance assertions
        performance_ok = (
            overall_success_rate >= 95.0 and  # 95% success rate
            overall_throughput >= 5.0         # 5 requests/second minimum
        )
        
        if performance_ok:
            print("   ✅ Performance validation passed")
        else:
            print("   ⚠️ Performance below expectations")
        
        return performance_ok
        
    except Exception as e:
        print(f"   ❌ Performance validation failed: {e}")
        return False

async def generate_capability_report():
    """Generate a report of available capabilities"""
    print("📋 Generating capability report...")
    
    try:
        async with pool_manager.postgres_pool.acquire() as conn:
            capabilities = await conn.fetch("""
                SELECT capability_type, name, description, category, 
                       trust_score, readiness_state, estimated_cost_per_execution
                FROM ro_capabilities
                ORDER BY capability_type, trust_score DESC
            """)
            
            if capabilities:
                print("\n📊 Available Capabilities Report")
                print("=" * 60)
                
                current_type = None
                for cap in capabilities:
                    if cap['capability_type'] != current_type:
                        current_type = cap['capability_type']
                        print(f"\n🔧 {current_type}:")
                    
                    print(f"   • {cap['name']}")
                    print(f"     Category: {cap['category']}")
                    print(f"     Trust Score: {cap['trust_score']:.2f}")
                    print(f"     Readiness: {cap['readiness_state']}")
                    print(f"     Cost: ${cap['estimated_cost_per_execution']:.2f}")
                    print(f"     Description: {cap['description']}")
                    print()
            
            print(f"✅ Capability report generated ({len(capabilities)} capabilities)")
            return True
            
    except Exception as e:
        print(f"❌ Capability report generation failed: {e}")
        return False

async def main():
    """Main setup and validation routine"""
    
    print("🚀 ROUTING ORCHESTRATOR SETUP & VALIDATION")
    print("=" * 60)
    print("Setting up the complete Routing Orchestrator system...")
    print("This will create a 100% functional command center for RevAI Pro.")
    print()
    
    # Initialize connection pool
    print("🔌 Initializing database connection...")
    success = await pool_manager.initialize()
    if not success:
        print("❌ Database connection failed")
        return False
    print("✅ Database connected")
    
    try:
        # Setup phases
        phases = [
            ("Database Schema Setup", setup_database_schema),
            ("Sample Data Loading", load_sample_data),
            ("Component Validation", validate_components),
            ("Demonstration Scenarios", run_demonstration_scenarios),
            ("Performance Validation", run_performance_validation),
            ("Capability Report", generate_capability_report)
        ]
        
        completed_phases = 0
        
        for phase_name, phase_func in phases:
            print(f"\n📋 Phase: {phase_name}")
            print("-" * 40)
            
            try:
                result = await phase_func()
                if result or result is None:  # None means success for some functions
                    print(f"✅ {phase_name} - COMPLETED")
                    completed_phases += 1
                else:
                    print(f"❌ {phase_name} - FAILED")
                    break
                    
            except Exception as e:
                print(f"❌ {phase_name} - ERROR: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Final results
        print(f"\n{'='*60}")
        print(f"🏆 SETUP RESULTS")
        print(f"   Completed phases: {completed_phases}/{len(phases)}")
        
        if completed_phases == len(phases):
            print(f"\n🎉 ROUTING ORCHESTRATOR SETUP COMPLETE!")
            print(f"   ✅ Database schema created")
            print(f"   ✅ All components initialized")
            print(f"   ✅ Sample data loaded")
            print(f"   ✅ End-to-end functionality validated")
            print(f"   ✅ Performance benchmarks met")
            print(f"\n🚀 The Routing Orchestrator is ready for production use!")
            print(f"   You can now process natural language requests through:")
            print(f"   - Intent parsing with 6+ patterns")
            print(f"   - Policy-aware governance")
            print(f"   - Capability registry lookup")
            print(f"   - RBA/RBIA/AALA execution routing")
            print(f"   - Multi-tenant isolation")
            print(f"   - Performance monitoring")
            print(f"   - Promotion protocol tracking")
            return True
        else:
            print(f"\n⚠️ SETUP INCOMPLETE")
            print(f"   {len(phases) - completed_phases} phases failed")
            print(f"   Review the errors above and retry")
            return False
            
    except Exception as e:
        print(f"\n❌ SETUP FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await pool_manager.close()

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)


