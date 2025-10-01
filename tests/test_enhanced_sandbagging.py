#!/usr/bin/env python3
"""
Test the enhanced sandbagging detection with rich analytics
"""

import asyncio
from dsl.orchestration.dynamic_rba_orchestrator import dynamic_rba_orchestrator
from api.csv_data_loader import crm_data_loader

async def test_enhanced_sandbagging():
    print("🚀 Testing Enhanced Sandbagging Detection...")
    
    # Load all opportunities
    opportunities = crm_data_loader.get_opportunities(limit=5000)
    print(f"✅ Loaded {len(opportunities)} opportunities")
    
    # Test enhanced sandbagging detection
    config = {
        'high_value_threshold': 250000,
        'low_probability_threshold': 35,
        'sandbagging_threshold': 40,  # Lower threshold for detection
        'confidence_threshold': 20     # Lower confidence requirement
    }
    
    result = await dynamic_rba_orchestrator.execute_rba_analysis(
        analysis_type="sandbagging_detection",
        opportunities=opportunities,
        config=config,
        tenant_id="1300",
        user_id="1319"
    )
    
    if result and result.get('success'):
        print(f"\n📊 SANDBAGGING ANALYSIS RESULTS:")
        print(f"   ✅ Total Opportunities: {result.get('total_opportunities', 0):,}")
        print(f"   🚨 Flagged for Sandbagging: {result.get('flagged_opportunities', 0)}")
        
        # Summary metrics
        if 'summary_metrics' in result:
            metrics = result['summary_metrics']
            print(f"\n📈 SUMMARY METRICS:")
            print(f"   💰 Total Pipeline Value: ${metrics.get('total_pipeline_value', 0):,.0f}")
            print(f"   💰 At-Risk Pipeline Value: ${metrics.get('at_risk_pipeline_value', 0):,.0f}")
            print(f"   📊 Average Deal Amount: ${metrics.get('average_deal_amount', 0):,.0f}")
            print(f"   📊 Average Probability: {metrics.get('average_probability', 0)}%")
            print(f"   🔍 High-Value Deals: {metrics.get('high_value_deals_count', 0)} ({metrics.get('high_value_deals_percentage', 0)}%)")
            print(f"   ⚠️ Low-Probability Deals: {metrics.get('low_probability_deals_count', 0)} ({metrics.get('low_probability_deals_percentage', 0)}%)")
            print(f"   🎯 Potential Risk Deals: {metrics.get('potential_risk_deals_count', 0)}")
        
        # Risk distribution
        if 'risk_distribution' in result:
            dist = result['risk_distribution']
            print(f"\n🎯 RISK DISTRIBUTION:")
            print(f"   🔥 Critical Risk: {dist.get('critical_risk', 0)} deals")
            print(f"   ⚠️ High Risk: {dist.get('high_risk', 0)} deals")
            print(f"   📊 Medium Risk: {dist.get('medium_risk', 0)} deals")
            print(f"   ✅ Low Risk: {dist.get('low_risk', 0)} deals")
        
        # Top risk deals
        if 'top_risk_deals' in result and result['top_risk_deals']:
            print(f"\n🏆 TOP 5 DEALS AT RISK:")
            for i, deal in enumerate(result['top_risk_deals'][:5]):
                print(f"   {i+1}. {deal['name'][:50]}...")
                print(f"      💰 ${deal['amount']:,.0f} | 📊 {deal['probability']}% | 🎯 Risk: {deal['risk_score']}")
                print(f"      👤 {deal['owner']} | 📈 {deal['stage']}")
        
        # Insights
        if 'insights' in result:
            print(f"\n💡 KEY INSIGHTS:")
            for insight in result['insights']:
                print(f"   {insight}")
    else:
        print("❌ Analysis failed")
        if result:
            print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_sandbagging())

