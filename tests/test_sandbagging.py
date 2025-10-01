#!/usr/bin/env python3
"""
Test sandbagging detection with real data
"""

from api.csv_data_loader import crm_data_loader

def find_sandbagging_candidates():
    print("🎯 Finding sandbagging candidates in CSV data...")
    
    # Load more opportunities to find sandbagging cases
    opps = crm_data_loader.get_opportunities(limit=100)
    print(f"📊 Analyzing {len(opps)} opportunities")
    
    # Find different sandbagging patterns
    patterns = {
        "high_value_low_prob": [],
        "advanced_stage_low_prob": [],
        "very_high_value": []
    }
    
    for opp in opps:
        amount = opp.get('Amount', 0)
        prob = opp.get('Probability', 0)
        stage = opp.get('StageName', '').lower()
        
        # Pattern 1: High value (>$250K) + Low probability (<35%)
        if amount > 250000 and prob < 35:
            patterns["high_value_low_prob"].append(opp)
        
        # Pattern 2: Advanced stage + Low probability
        advanced_stages = ['proposal', 'negotiation', 'contract']
        if any(adv in stage for adv in advanced_stages) and prob < 40:
            patterns["advanced_stage_low_prob"].append(opp)
            
        # Pattern 3: Very high value deals (potential for sandbagging)
        if amount > 1000000:
            patterns["very_high_value"].append(opp)
    
    # Report findings
    print("\n🔍 Sandbagging Analysis Results:")
    
    for pattern_name, deals in patterns.items():
        print(f"\n{pattern_name.replace('_', ' ').title()}: {len(deals)} deals")
        
        for i, deal in enumerate(deals[:3], 1):  # Show top 3
            print(f"   {i}. {deal.get('Name')}")
            print(f"      💰 ${deal.get('Amount', 0):,.0f} | 📊 {deal.get('Probability', 0)}% | 🎯 {deal.get('StageName')}")
            print(f"      🏢 {deal.get('Account', {}).get('Name', 'Unknown Account')}")
    
    # Overall stats
    total_suspicious = len(patterns["high_value_low_prob"]) + len(patterns["advanced_stage_low_prob"])
    print(f"\n📈 Summary:")
    print(f"   Total suspicious deals: {total_suspicious}")
    print(f"   Percentage of pipeline: {(total_suspicious/len(opps)*100):.1f}%")
    
    return patterns["high_value_low_prob"][:5]  # Return top 5 for testing

if __name__ == "__main__":
    sandbagging_deals = find_sandbagging_candidates()


