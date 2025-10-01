#!/usr/bin/env python3
"""
End-to-end test of the dynamic RBA system
"""

import asyncio
from dsl.orchestration.dynamic_rba_orchestrator import dynamic_rba_orchestrator
from dsl.registry.rba_agent_registry import rba_registry
from api.csv_data_loader import crm_data_loader

async def test_full_system():
    print("🚀 Testing Full Dynamic RBA System...")
    
    # Test 1: Registry Discovery
    print("\n1️⃣ Testing Agent Registry...")
    agents = rba_registry.get_all_agents()
    print(f"   ✅ Found {len(agents)} RBA agents")
    for agent_name, info in agents.items():
        print(f"   📋 {agent_name}: {info.agent_description}")
    
    # Test 2: Load Data
    print("\n2️⃣ Testing Data Loading...")
    opportunities = crm_data_loader.get_opportunities(limit=50)
    print(f"   ✅ Loaded {len(opportunities)} opportunities")
    
    if opportunities:
        sample = opportunities[0]
        print(f"   📋 Sample: {sample['Name']} - ${sample['Amount']:,.0f}")
    
    # Test 3: Dynamic Orchestration
    print("\n3️⃣ Testing Dynamic Orchestration...")
    
    test_cases = [
        ("sandbagging_detection", "Sandbagging Detection"),
        ("stale_deals", "Stale Deals Detection"),
        ("missing_fields", "Missing Fields Audit"),
        ("data_quality", "Data Quality Analysis"),
        ("deals_at_risk", "Risk Assessment")
    ]
    
    for analysis_type, description in test_cases:
        print(f"\n   🎯 Testing {description}...")
        
        try:
            # Test with basic config
            config = {
                'high_value_threshold': 100000,
                'sandbagging_threshold': 70,
                'stale_threshold_days': 30
            }
            
            result = await dynamic_rba_orchestrator.execute_rba_analysis(
                analysis_type=analysis_type,
                opportunities=opportunities,
                config=config,
                tenant_id="1300",
                user_id="1319"
            )
            
            if result and result.get('success'):
                print(f"      ✅ Success: {result.get('flagged_opportunities', 0)} flagged")
                print(f"      📊 Total analyzed: {result.get('total_opportunities', 0)}")
                if 'compliance_score' in result:
                    print(f"      🎯 Compliance: {result['compliance_score']:.1f}%")
            else:
                print(f"      ⚠️ No results or failed")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print("\n✅ Full system test complete!")

if __name__ == "__main__":
    asyncio.run(test_full_system())
