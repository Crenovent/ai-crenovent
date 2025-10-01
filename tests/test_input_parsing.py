#!/usr/bin/env python3
"""
Test the improved input parsing logic
"""

import asyncio
import sys
sys.path.append('.')

from api.pipeline_agents import parse_analysis_type_from_input

async def test_input_parsing():
    print('🧪 Testing Input Parsing Logic...')
    
    test_cases = [
        {
            "input": "Identify sandbagging or inflated deals (low probability but high value)",
            "expected": "sandbagging_detection"
        },
        {
            "input": "Check for sandbagging in my pipeline",
            "expected": "sandbagging_detection"
        },
        {
            "input": "Find deals at risk",
            "expected": "deals_at_risk"
        },
        {
            "input": "Review pipeline velocity by stage",
            "expected": "stage_velocity"
        },
        {
            "input": "Get pipeline summary",
            "expected": "pipeline_summary"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f'\n📋 Test {i}: "{test_case["input"]}"')
        
        result = await parse_analysis_type_from_input(test_case["input"])
        expected = test_case["expected"]
        
        if result == expected:
            print(f'   ✅ CORRECT: {result}')
        else:
            print(f'   ❌ WRONG: Got {result}, expected {expected}')
    
    print('\n🎉 Input parsing test complete!')

if __name__ == "__main__":
    asyncio.run(test_input_parsing())
