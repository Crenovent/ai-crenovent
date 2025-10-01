#!/usr/bin/env python3
"""
Test the dynamic business rules engine
"""

from dsl.rules.business_rules_engine import BusinessRulesEngine
from api.csv_data_loader import crm_data_loader

def test_dynamic_rules():
    print("🧪 Testing Dynamic Business Rules Engine...")
    
    # Load rules engine
    engine = BusinessRulesEngine()
    print(f"✅ Loaded {len(engine.rules)} rule sets")
    
    # Get sample opportunity data
    opportunities = crm_data_loader.get_opportunities(limit=10)
    if not opportunities:
        print("❌ No opportunities loaded")
        return
    
    sample_opp = opportunities[0]
    print(f"📋 Testing with: {sample_opp['Name']} - ${sample_opp['Amount']:,.0f} ({sample_opp['Probability']}%)")
    
    # Test different rule sets dynamically
    test_rule_sets = ['sandbagging_detection', 'stale_deals', 'missing_fields']
    
    for rule_set_name in test_rule_sets:
        if rule_set_name in engine.rules:
            print(f"\n🎯 Testing {rule_set_name}...")
            
            try:
                # Get defaults for this rule set
                rule_config = engine.rules[rule_set_name].get('defaults', {})
                
                # Evaluate using the dynamic method
                result = engine.evaluate_rules(rule_set_name, sample_opp, rule_config)
                
                print(f"   ✅ Score: {result.total_score:.1f}")
                print(f"   🎯 Risk: {result.risk_level}")
                print(f"   🔒 Confidence: {result.confidence:.1f}%")
                
                if result.rule_results:
                    print(f"   📊 Rules triggered: {len(result.rule_results)}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ⚠️ Rule set '{rule_set_name}' not found")
    
    print("\n✅ Dynamic rules engine test complete!")

if __name__ == "__main__":
    test_dynamic_rules()

