#!/usr/bin/env python3
"""
Test the fixed sandbagging detection with direct scoring
"""

import asyncio
from dsl.orchestration.dynamic_rba_orchestrator import dynamic_rba_orchestrator
from api.csv_data_loader import crm_data_loader

async def test_fixed_sandbagging():
    print("🚀 Testing Fixed Sandbagging Detection...")
    
    # Load sample opportunities for quick test
    opportunities = crm_data_loader.get_opportunities(limit=100)
    print(f"✅ Loaded {len(opportunities)} opportunities for testing")
    
    # Test with lower threshold for detection
    config = {
        'high_value_threshold': 250000,
        'low_probability_threshold': 35,
        'sandbagging_threshold': 40,  # Lower threshold for practical detection
    }
    
    result = await dynamic_rba_orchestrator.execute_rba_analysis(
        analysis_type="sandbagging_detection",
        opportunities=opportunities,
        config=config,
        tenant_id="1300",
        user_id="1319"
    )
    
    if result and result.get('success'):
        print(f"\n📊 FIXED SANDBAGGING RESULTS:")
        print(f"   ✅ Total Opportunities: {result.get('total_opportunities', 0)}")
        print(f"   🚨 FLAGGED Deals: {result.get('flagged_opportunities', 0)}")
        
        # Show flagged deals
        if result.get('flagged_deals'):
            print(f"\n🚨 FLAGGED SANDBAGGING DEALS:")
            for i, deal in enumerate(result['flagged_deals'][:5]):
                print(f"   {i+1}. {deal['opportunity_name']}")
                print(f"      💰 ${deal['amount']:,.0f} | 📊 {deal['probability']}% | 🎯 Score: {deal['sandbagging_score']}")
                print(f"      🔥 Risk: {deal['risk_level']} | 📈 {deal['stage_name']}")
                print(f"      ⚠️ Factors: {', '.join(deal.get('risk_factors', []))}")
        
        # Show summary
        if 'summary_metrics' in result:
            metrics = result['summary_metrics']
            print(f"\n📈 SUMMARY:")
            print(f"   🎯 Sandbagging Rate: {metrics.get('sandbagging_rate', 0)}%")
            print(f"   💰 At-Risk Value: ${metrics.get('at_risk_pipeline_value', 0):,.0f}")
    else:
        print("❌ Analysis failed")
        if result:
            print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_fixed_sandbagging())

