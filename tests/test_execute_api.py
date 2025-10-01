#!/usr/bin/env python3
"""
Test Execute API with Dynamic RBA Orchestrator
"""

import asyncio
import sys
sys.path.append('.')

from api.pipeline_agents import execute_pipeline_agent
from api.models import PipelineAgentRequest

async def test_execute_api():
    print('🧪 Testing Execute API...')
    
    # Test cases
    test_cases = [
        {
            "user_input": "Check for sandbagging in my pipeline",
            "expected_analysis": "sandbagging_detection"
        },
        {
            "user_input": "Find stale deals that are stuck",
            "expected_analysis": "stale_deals"
        },
        {
            "user_input": "Analyze missing fields in opportunities",
            "expected_analysis": "missing_fields"
        },
        {
            "user_input": "Get pipeline summary",
            "expected_analysis": "pipeline_summary"
        },
        {
            "user_input": "Check data quality issues",
            "expected_analysis": "data_quality"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f'\n📋 Test {i}: "{test_case["user_input"]}"')
        
        try:
            # Create request
            request = PipelineAgentRequest(
                user_input=test_case["user_input"],
                context={"test_mode": True},
                tenant_id="1300",
                user_id="1319"
            )
            
            # Execute
            result = await execute_pipeline_agent(request)
            
            if result.get("success"):
                print(f'   ✅ Success: {result.get("analysis_type")} analysis')
                print(f'   🤖 Agent: {result.get("agent_used")}')
                print(f'   📊 Opportunities: {result.get("total_opportunities")} total, {result.get("flagged_opportunities")} flagged')
                print(f'   ⏱️  Time: {result.get("execution_time_ms", 0):.2f}ms')
            else:
                print(f'   ❌ Failed: {result.get("error", "Unknown error")}')
                
        except Exception as e:
            print(f'   ❌ Exception: {e}')
    
    print('\n🎉 Execute API test complete!')

if __name__ == "__main__":
    asyncio.run(test_execute_api())
