#!/usr/bin/env python3
"""
Detailed verification script to check if the AI is using your actual database patterns
"""

import requests
import json

def test_user_1319_specific_patterns():
    """Test User 1319's specific historical patterns from your database"""
    
    print("🔍 DETAILED PERSONALIZATION VERIFICATION")
    print("=" * 60)
    print("Based on your database analysis:")
    print("📊 User 1319 Patterns:")
    print("  - Industry: Technology (5 plans)")
    print("  - Avg Growth: 16% (range 10-20%)")  
    print("  - Cadence: Monthly (predominant)")
    print("  - Risks: 'Economic fluctuations in APAC, competitive pressures'")
    print("  - Account Tiers: Key, Strategic")
    print("=" * 60)
    
    # Test with minimal prompt to see if it uses historical data
    test_payload = {
        "message": "Create plan for Microsoft",  # Minimal prompt
        "user_context": {
            "user_id": 1319,  # Your main user
            "account_id": "Microsoft", 
            "industry": "Technology",
            "account_tier": "Enterprise"
            # Note: No annual_revenue provided to test intelligence
        }
    }
    
    print(f"\n🧪 TESTING with minimal prompt: '{test_payload['message']}'")
    print("💡 This should trigger User 1319's historical patterns...")
    
    try:
        response = requests.post(
            'http://localhost:8000/api/strategic-planning/chat',
            headers={'Content-Type': 'application/json'},
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            form_data = data.get('form_prefill', {})
            
            print("\n✅ FULL RESPONSE ANALYSIS:")
            print("-" * 40)
            
            # Analyze each field for personalization
            fields_to_check = [
                'short_term_goals', 'long_term_goals', 'known_risks', 
                'revenue_growth_target', 'communication_cadence',
                'account_tier', 'region_territory', 'key_opportunities'
            ]
            
            personalization_score = 0
            total_checks = 0
            
            for field in fields_to_check:
                value = form_data.get(field, 'Not set')
                print(f"📋 {field}: {value}")
                
                # Check for User 1319 specific patterns
                if field == 'revenue_growth_target':
                    total_checks += 1
                    try:
                        growth = int(float(value))
                        if 10 <= growth <= 20:  # User 1319's historical range
                            print(f"  ✅ Within User 1319's historical range (10-20%)")
                            personalization_score += 1
                        else:
                            print(f"  ⚠️ Outside User 1319's typical range")
                    except:
                        print(f"  ❌ Invalid growth value")
                
                elif field == 'communication_cadence':
                    total_checks += 1
                    if 'monthly' in value.lower():
                        print(f"  ✅ Matches User 1319's Monthly preference")
                        personalization_score += 1
                    else:
                        print(f"  ⚠️ Doesn't match User 1319's Monthly preference")
                
                elif field == 'known_risks':
                    total_checks += 1
                    risk_text = value.lower()
                    user_1319_risk_keywords = ['economic', 'apac', 'competitive', 'market', 'competition']
                    found_keywords = [kw for kw in user_1319_risk_keywords if kw in risk_text]
                    if found_keywords:
                        print(f"  ✅ Contains User 1319 risk themes: {found_keywords}")
                        personalization_score += 1
                    else:
                        print(f"  ⚠️ No User 1319 specific risk patterns detected")
                
                elif field == 'short_term_goals':
                    total_checks += 1
                    goals_text = value.lower()
                    tech_keywords = ['technology', 'tech', 'digital', 'ai', 'cloud', 'azure', 'microsoft']
                    found_keywords = [kw for kw in tech_keywords if kw in goals_text]
                    if found_keywords:
                        print(f"  ✅ Technology/Microsoft themes: {found_keywords}")
                        personalization_score += 1
                    else:
                        print(f"  ⚠️ Limited technology-specific content")
            
            print("\n📊 PERSONALIZATION SCORE:")
            percentage = (personalization_score / total_checks) * 100 if total_checks > 0 else 0
            print(f"🎯 {personalization_score}/{total_checks} checks passed ({percentage:.1f}%)")
            
            if percentage >= 75:
                print("🎉 EXCELLENT: High level of personalization detected!")
            elif percentage >= 50:
                print("👍 GOOD: Moderate personalization detected")
            elif percentage >= 25:
                print("⚠️ LIMITED: Some personalization, but could be improved")
            else:
                print("❌ POOR: Little to no personalization detected")
                
            # Check for LangChain SQL usage in logs
            print(f"\n🔍 LOOK FOR THESE LOG PATTERNS in the server console:")
            print("  - '🤖 Using LangChain SQL agent for short_term_goals analysis'")
            print("  - '✅ LangChain SQL analysis successful'")
            print("  - '🧠 Generated intelligent short-term goals'")
            print("  - PostgreSQL query logs")
            
        else:
            print(f"❌ API ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_database_fallback():
    """Test what happens with no historical data (new user)"""
    print(f"\n🧪 TESTING DATABASE FALLBACK (New User)")
    print("-" * 40)
    
    test_payload = {
        "message": "Create strategic plan for Apple with innovation focus",
        "user_context": {
            "user_id": 9999,  # Non-existent user
            "account_id": "Apple",
            "industry": "Technology", 
            "account_tier": "Enterprise",
            "annual_revenue": "300000000000"
        }
    }
    
    try:
        response = requests.post(
            'http://localhost:8000/api/strategic-planning/chat',
            headers={'Content-Type': 'application/json'},
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            form_data = data.get('form_prefill', {})
            print("✅ New user response (should use industry benchmarks):")
            print(f"📊 Goals: {form_data.get('short_term_goals', 'Not set')}")
            print(f"📈 Growth: {form_data.get('revenue_growth_target', 'Not set')}%") 
            print(f"📅 Cadence: {form_data.get('communication_cadence', 'Not set')}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_user_1319_specific_patterns()
    test_database_fallback()






























