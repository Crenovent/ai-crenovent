#!/usr/bin/env python3
"""
Test CSV Data Loader
"""

from api.csv_data_loader import crm_data_loader

def test_csv_loader():
    print("🧪 Testing CSV Data Loader...")
    
    # Test summary stats
    print("\n📊 Summary Statistics:")
    stats = crm_data_loader.get_summary_stats()
    print(f"   Total opportunities: {stats.get('total_opportunities', 'N/A')}")
    print(f"   Pipeline value: ${stats.get('total_pipeline_value', 0):,.0f}")
    print(f"   Average probability: {stats.get('average_probability', 0)}%")
    print(f"   Potential sandbagging deals: {stats.get('potential_sandbagging_deals', 'N/A')}")
    
    # Test opportunity loading
    print("\n🔍 Sample Opportunities:")
    opps = crm_data_loader.get_opportunities(limit=5)
    print(f"   Loaded {len(opps)} opportunities")
    
    for i, opp in enumerate(opps[:3], 1):
        print(f"   {i}. {opp.get('Name', 'Unknown')}")
        print(f"      Amount: ${opp.get('Amount', 0):,.0f}")
        print(f"      Probability: {opp.get('Probability', 0)}%")
        print(f"      Stage: {opp.get('StageName', 'Unknown')}")
        print(f"      Account: {opp.get('Account', {}).get('Name', 'Unknown')}")
        print()
    
    # Test sandbagging detection data
    print("🎯 Sandbagging Detection Test:")
    high_value_low_prob = [opp for opp in opps if opp.get('Amount', 0) > 1000000 and opp.get('Probability', 0) < 40]
    print(f"   High-value, low-probability deals: {len(high_value_low_prob)}")
    
    for opp in high_value_low_prob[:2]:
        print(f"   - {opp.get('Name')}: ${opp.get('Amount', 0):,.0f} at {opp.get('Probability', 0)}%")
    
    print("\n✅ CSV Data Loader test complete!")

if __name__ == "__main__":
    test_csv_loader()


