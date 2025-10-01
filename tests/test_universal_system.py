#!/usr/bin/env python3
"""
Test Universal RBA Parameter System & Knowledge Graph Integration
================================================================
Comprehensive test to verify the universal parameter system and KG tracing work correctly.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_universal_parameter_system():
    """Test the universal parameter system"""
    
    print("\n" + "="*80)
    print("🧪 TESTING UNIVERSAL RBA PARAMETER SYSTEM")
    print("="*80)
    
    try:
        from dsl.configuration.universal_parameter_manager import get_universal_parameter_manager
        
        # Initialize parameter manager
        param_manager = get_universal_parameter_manager()
        param_manager.load_parameters()
        
        print(f"✅ Universal Parameter Manager initialized")
        
        # Test 1: Get all parameters
        all_params = param_manager.get_all_parameters()
        print(f"📊 Total universal parameters: {len(all_params)}")
        
        # Test 2: Get statistics
        stats = param_manager.get_parameter_statistics()
        print(f"📈 Parameter Statistics:")
        print(f"   - Total Parameters: {stats['total_parameters']}")
        print(f"   - Parameter Types: {stats['parameter_types']}")
        print(f"   - UI Components: {stats['ui_components']}")
        print(f"   - Agents Supported: {stats['agents_supported']}")
        
        # Test 3: Get parameters for sandbagging agent
        sandbagging_params = param_manager.get_parameters_for_agent('sandbagging_detection')
        print(f"🎯 Sandbagging agent parameters: {len(sandbagging_params)}")
        
        # Test 4: Get default configuration
        default_config = param_manager.get_default_config_for_agent('sandbagging_detection')
        print(f"⚙️ Default config parameters: {len(default_config)}")
        print(f"   - High value threshold: ${default_config.get('high_value_threshold', 0):,}")
        print(f"   - Sandbagging threshold: {default_config.get('sandbagging_threshold', 0)}%")
        
        # Test 5: Validate configuration
        test_config = {
            'high_value_threshold': 500000,
            'sandbagging_threshold': 70,
            'confidence_threshold': 85
        }
        
        validated_config = param_manager.validate_config_for_agent('sandbagging_detection', test_config)
        print(f"✅ Configuration validation successful: {len(validated_config)} parameters")
        
        # Test 6: Generate UI schema
        ui_schema = param_manager.generate_ui_schema_for_agent('sandbagging_detection')
        print(f"🎨 UI Schema generated: {ui_schema['total_parameters']} parameters in {len(ui_schema['parameter_groups'])} groups")
        
        return True
        
    except Exception as e:
        print(f"❌ Universal parameter system test failed: {e}")
        return False

async def test_enhanced_rba_agent():
    """Test the enhanced RBA agent with universal parameters"""
    
    print("\n" + "="*80)
    print("🤖 TESTING ENHANCED RBA AGENT")
    print("="*80)
    
    try:
        from dsl.operators.rba.sandbagging_rba_agent import SandbaggingRBAAgent
        
        # Create agent instance
        agent = SandbaggingRBAAgent()
        print(f"✅ Enhanced Sandbagging RBA Agent created")
        print(f"   - Agent Name: {agent.AGENT_NAME}")
        print(f"   - Agent Description: {agent.AGENT_DESCRIPTION}")
        
        # Test universal parameters
        universal_params = agent.get_universal_parameters()
        print(f"📊 Universal parameters available: {len(universal_params)}")
        
        # Test default configuration
        default_config = agent.get_default_configuration()
        print(f"⚙️ Default configuration: {len(default_config)} parameters")
        
        # Test parameter value retrieval
        high_value_threshold = agent.get_parameter_value(default_config, 'high_value_threshold', 100000)
        print(f"💰 High value threshold: ${high_value_threshold:,}")
        
        # Test UI schema generation
        ui_schema = agent.get_ui_schema()
        print(f"🎨 UI Schema: {ui_schema['total_parameters']} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced RBA agent test failed: {e}")
        return False

async def test_rba_execution_with_tracing():
    """Test RBA execution with Knowledge Graph tracing"""
    
    print("\n" + "="*80)
    print("🧠 TESTING KNOWLEDGE GRAPH EXECUTION TRACING")
    print("="*80)
    
    try:
        # Mock KG store for testing
        class MockKGStore:
            def __init__(self):
                self.pool_manager = MockPoolManager()
        
        class MockPoolManager:
            def __init__(self):
                self.postgres_pool = MockPostgresPool()
        
        class MockPostgresPool:
            def acquire(self):
                return MockConnection()
        
        class MockConnection:
            async def execute(self, query, *args):
                print(f"📝 Mock DB query executed: {query[:50]}...")
                return None
            
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        
        # Test execution tracer
        from dsl.knowledge.rba_execution_tracer import RBAExecutionTracer
        
        kg_store = MockKGStore()
        tracer = RBAExecutionTracer(kg_store)
        print(f"✅ RBA Execution Tracer created")
        
        # Test trace creation
        trace = await tracer.start_trace(
            agent_name='sandbagging_detection',
            input_data={'opportunities': [{'Name': 'Test Deal', 'Amount': 100000}]},
            config={'high_value_threshold': 250000},
            context={'tenant_id': '1300', 'user_id': '1319'}
        )
        
        print(f"🚀 Execution trace started: {trace.execution_id}")
        print(f"   - Agent: {trace.agent_name}")
        print(f"   - Tenant: {trace.tenant_id}")
        print(f"   - User: {trace.user_id}")
        
        # Test trace completion
        mock_result = {
            'success': True,
            'total_opportunities': 1,
            'flagged_opportunities': 0,
            'summary_metrics': {
                'total_pipeline_value': 100000,
                'at_risk_pipeline_value': 0
            }
        }
        
        await tracer.complete_trace(trace, mock_result)
        print(f"✅ Execution trace completed: {trace.execution_duration_ms:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge Graph tracing test failed: {e}")
        return False

async def test_api_integration():
    """Test API integration with universal parameters"""
    
    print("\n" + "="*80)
    print("🌐 TESTING API INTEGRATION")
    print("="*80)
    
    try:
        # Test parameter manager API integration
        from dsl.configuration.universal_parameter_manager import get_universal_parameter_manager
        
        param_manager = get_universal_parameter_manager()
        
        # Simulate API request for sandbagging parameters
        agent_params = param_manager.get_parameters_for_agent('sandbagging_detection')
        ui_schema = param_manager.generate_ui_schema_for_agent('sandbagging_detection')
        
        print(f"🔌 API Integration Test:")
        print(f"   - Agent parameters: {len(agent_params)}")
        print(f"   - UI schema groups: {len(ui_schema['parameter_groups'])}")
        print(f"   - Total parameters in schema: {ui_schema['total_parameters']}")
        
        # Test configuration validation (like API would do)
        user_config = {
            'high_value_threshold': 300000,
            'sandbagging_threshold': 75,
            'confidence_threshold': 80
        }
        
        merged_config = param_manager.merge_configs('sandbagging_detection', user_config)
        print(f"✅ Configuration merge successful: {len(merged_config)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    
    print("🧪 UNIVERSAL RBA SYSTEM & KNOWLEDGE GRAPH INTEGRATION TESTS")
    print("=" * 100)
    
    tests = [
        ("Universal Parameter System", test_universal_parameter_system),
        ("Enhanced RBA Agent", test_enhanced_rba_agent),
        ("Knowledge Graph Tracing", test_rba_execution_with_tracing),
        ("API Integration", test_api_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*100)
    print("📋 TEST SUMMARY")
    print("="*100)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Universal RBA system is ready!")
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
