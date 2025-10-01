#!/usr/bin/env python3
"""
Test Dynamic Agent Mapping System
Verifies that all frontend agent types map correctly to available RBA agents
"""

import asyncio
import logging
from api.pipeline_agents import parse_analysis_type_from_input
from dsl.registry.rba_agent_registry import rba_registry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dynamic_mapping():
    """Test the dynamic mapping system with all frontend agent types"""
    
    # Initialize registry
    print("🔧 Initializing RBA Registry...")
    success = rba_registry.initialize()
    print(f"   Registry Status: {'✅ Success' if success else '❌ Failed'}")
    
    # Test cases from the frontend (all 17 agents)
    test_cases = [
        # Data Quality (4)
        'Pipeline Data Quality Audit',
        'Duplicate Deal Detection', 
        'Ownerless Deals Detection',
        'Activity Gap Analysis',
        
        # Risk Analysis (4)
        'Deal Risk Scoring',
        'Pipeline Hygiene Check',
        'Sandbagging Detection',
        'At-Risk Deals Analysis',
        
        # Velocity Analysis (4)
        'Probability Alignment Check',
        'Close Date Standardization',
        'Stage Velocity Analysis',
        'Conversion Rate Analysis',
        
        # Performance Analysis (4)
        'Pipeline Summary',
        'Coverage Ratio Enforcement',
        'Company-wide Health Check',
        'Forecast Comparison',
        
        # Additional test cases
        'activity_tracking_audit',  # From terminal error
        'Check deals stuck for more than 90 days',  # User prompt
        'Find duplicate opportunities',  # User prompt
    ]
    
    print(f'\n🧪 Testing Dynamic Agent Mapping ({len(test_cases)} test cases):')
    print('=' * 70)
    
    success_count = 0
    for i, test_input in enumerate(test_cases, 1):
        try:
            analysis_type = await parse_analysis_type_from_input(test_input)
            print(f'{i:2d}. ✅ "{test_input}" → {analysis_type}')
            success_count += 1
        except Exception as e:
            print(f'{i:2d}. ❌ "{test_input}" → ERROR: {e}')
    
    print(f'\n📊 Results: {success_count}/{len(test_cases)} successful mappings')
    
    # Show available agents
    available_agents = rba_registry.get_all_agents()
    available_types = rba_registry.get_supported_analysis_types()
    
    print(f'\n🎯 Registry Status:')
    print(f'   Available Agents: {len(available_agents)}')
    print(f'   Analysis Types: {len(available_types)}')
    
    print(f'\n📋 All Available Analysis Types:')
    for i, analysis_type in enumerate(sorted(available_types), 1):
        print(f'   {i:2d}. {analysis_type}')
    
    return success_count == len(test_cases)

if __name__ == "__main__":
    asyncio.run(test_dynamic_mapping())
