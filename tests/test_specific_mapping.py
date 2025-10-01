#!/usr/bin/env python3
"""Test specific mapping cases"""

import asyncio
from api.pipeline_agents import parse_analysis_type_from_input

async def test_specific_cases():
    test_cases = [
        'Sandbagging Detection',
        'Pipeline Summary', 
        'Activity Gap Analysis',
        'Deal Risk Scoring',
        'Stage Velocity Analysis'
    ]
    
    print("🧪 Testing Specific Mapping Cases:")
    print("=" * 50)
    
    for case in test_cases:
        result = await parse_analysis_type_from_input(case)
        print(f'✅ "{case}" → {result}')

if __name__ == "__main__":
    asyncio.run(test_specific_cases())
