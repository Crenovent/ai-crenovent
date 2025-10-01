#!/usr/bin/env python3
"""
Test full sandbagging detection pipeline
"""

import asyncio
from api.csv_data_loader import crm_data_loader
from dsl.rules.business_rules_engine import BusinessRulesEngine
from dsl.operators.rba.sandbagging_rba_agent import SandbaggingRBAAgent

async def test_full_sandbagging_pipeline():
    print("🧪 Testing Full Sandbagging Detection Pipeline...")
    
    # Step 1: Load CSV data
    print("\n📊 Step 1: Loading CSV data...")
    opportunities = crm_data_loader.get_opportunities(limit=50)
    print(f"✅ Loaded {len(opportunities)} opportunities")
    
    # Step 2: Test rules engine
    print("\n⚙️ Step 2: Testing rules engine...")
    rules_engine = BusinessRulesEngine()
    if "sandbagging_detection" in rules_engine.rules:
        print("✅ Sandbagging rules loaded")
        defaults = rules_engine.rules["sandbagging_detection"]["defaults"]
        print(f"   📋 High value threshold: ${defaults['high_value_threshold']:,}")
        print(f"   📋 Low probability threshold: {defaults['low_probability_threshold']}%")
    else:
        print("❌ Sandbagging rules missing")
        return
    
    # Step 3: Test sandbagging agent
    print("\n🎯 Step 3: Testing sandbagging agent...")
    agent = SandbaggingRBAAgent()
    
    # Prepare input data
    input_data = {
        'opportunities': opportunities,
        'config': defaults,
        'analysis_type': 'sandbagging_detection',
        'tenant_id': '1300',
        'user_id': '1319'
    }
    
    # Execute analysis
    result = await agent.execute(input_data)
    
    print(f"✅ Analysis complete: {result.get('success', False)}")
    print(f"📊 Total opportunities: {result.get('total_opportunities', 0)}")
    print(f"⚠️ Flagged opportunities: {result.get('flagged_opportunities', 0)}")
    
    # Show flagged deals
    flagged_deals = result.get('flagged_deals', [])
    if flagged_deals:
        print("\n🚨 Flagged Sandbagging Deals:")
        for i, deal in enumerate(flagged_deals[:3], 1):
            print(f"   {i}. {deal['opportunity_name']}")
            print(f"      💰 ${deal['amount']:,.0f} | 📊 {deal['probability']}% | 🎯 {deal['stage_name']}")
            print(f"      🏢 {deal['account_name']}")
            print(f"      ⚠️ Score: {deal['sandbagging_score']:.1f} | Risk: {deal['risk_level']}")
            print()
    else:
        print("ℹ️ No deals flagged (may need to adjust thresholds)")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(test_full_sandbagging_pipeline())

