#!/usr/bin/env python3
"""
Test API Integration with Dynamic RBA Orchestrator
"""

import asyncio
import sys
sys.path.append('.')

from api.pipeline_agents import list_available_agents, get_supported_analysis_types

async def test_api_integration():
    print('🧪 Testing API Integration...')
    
    # Test 1: List agents
    try:
        agents_result = await list_available_agents()
        print(f'✅ Agents API: {agents_result["total_agents"]} agents found')
        print(f'   - RBA Agents: {len(agents_result["agents"]["rba_agents"])}')
    except Exception as e:
        print(f'❌ Agents API failed: {e}')
    
    # Test 2: Get analysis types
    try:
        types_result = await get_supported_analysis_types()
        print(f'✅ Analysis Types API: {types_result["total_analysis_types"]} types found')
        print(f'   - Categories: {list(types_result["categorized_types"].keys())}')
        
        # Show some examples
        for category, types in types_result["categorized_types"].items():
            if types:  # Only show categories that have types
                print(f'   - {category}: {len(types)} types')
                
    except Exception as e:
        print(f'❌ Analysis Types API failed: {e}')
    
    print('🎉 API Integration test complete!')

if __name__ == "__main__":
    asyncio.run(test_api_integration())
