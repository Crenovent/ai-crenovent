#!/usr/bin/env python3
"""
Test business rules loading
"""

from dsl.rules.business_rules_engine import BusinessRulesEngine

def test_rules_loading():
    print("🧪 Testing business rules loading...")
    
    try:
        engine = BusinessRulesEngine()
        print(f"✅ Rules loaded successfully: {len(engine.rules)} rule sets")
        print(f"📋 Available rules: {list(engine.rules.keys())}")
        
        if "sandbagging_detection" in engine.rules:
            print("✅ Sandbagging rules found!")
            sandbagging_rules = engine.rules["sandbagging_detection"]
            print(f"   📊 Defaults: {sandbagging_rules.get('defaults', {})}")
        else:
            print("❌ Sandbagging rules NOT found!")
            
    except Exception as e:
        print(f"❌ Error loading rules: {e}")

if __name__ == "__main__":
    test_rules_loading()

