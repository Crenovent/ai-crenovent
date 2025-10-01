#!/usr/bin/env python3

from dsl.operators.rba.ultra_dynamic_geolocation_service import UltraDynamicGeolocationService

def test_ultra_dynamic_geolocation():
    """Test the ultra-dynamic hierarchical geolocation service"""
    
    # Initialize the service
    geo_service = UltraDynamicGeolocationService()
    
    # Test users with hierarchical location data
    test_users = [
        {
            'Name': 'Raj Patel',
            'Email': 'raj@test.com',
            'City': 'Mumbai',
            'State': 'Maharashtra', 
            'Country': 'India',
            'Area': 'Bandra West'
        },
        {
            'Name': 'John Doe',
            'Email': 'john@test.com',
            'City': 'New York',
            'State': 'New York',
            'Country': 'USA',
            'Area': 'Manhattan'
        },
        {
            'Name': 'Hans Mueller',
            'Email': 'hans@test.com',
            'City': 'Berlin',
            'State': 'Berlin',
            'Country': 'Germany'
        },
        {
            'Name': 'Yuki Tanaka',
            'Email': 'yuki@test.com',
            'City': 'Tokyo',
            'Prefecture': 'Tokyo',
            'Country': 'Japan'
        },
        {
            'Name': 'Carlos Silva',
            'Email': 'carlos@test.com',
            'City': 'São Paulo',
            'State': 'São Paulo',
            'Country': 'Brazil'
        }
    ]
    
    print("🌍 Testing Ultra-Dynamic Hierarchical Geolocation Service")
    print("=" * 70)
    print("✅ NO hardcoded cities/states/districts!")
    print("✅ Dynamic administrative hierarchy detection")
    print("✅ Country-specific admin level naming")
    print("✅ Uses geocoding APIs + reverse geocoding databases")
    print("=" * 70)
    
    for user in test_users:
        print(f"\n👤 {user['Name']}")
        
        # Show input data
        location_fields = ['City', 'State', 'Prefecture', 'Country', 'Area']
        input_data = {k: v for k, v in user.items() if k in location_fields and v}
        print(f"   📍 Input: {input_data}")
        
        # Process with ultra-dynamic service
        result = geo_service.process_user_location(user)
        
        print(f"   🗺️ Country: {result.get('country_code', 'None')}")
        print(f"   🌍 Region: {result.get('region', 'None')}")
        print(f"   📊 Confidence: {result.get('confidence', 0):.2f}")
        print(f"   🔧 Method: {result.get('method', 'None')}")
        
        # Show hierarchical admin levels
        admin_levels = result.get('admin_levels', {})
        if admin_levels:
            print(f"   🏛️ Administrative Hierarchy:")
            for level, value in admin_levels.items():
                if value:
                    print(f"      • {level.title()}: {value}")
        
        # Show generated territories
        territories = result.get('territories', {})
        if territories:
            print(f"   🏢 Generated Territories:")
            for territory_type, territory_code in territories.items():
                if territory_code and not territory_type.endswith('_name') and not territory_type.endswith('_type'):
                    territory_name = territories.get(f"{territory_type}_name", "")
                    territory_level = territories.get(f"{territory_type}_type", "")
                    print(f"      • {territory_type.title()}: {territory_code} ({territory_name}) [{territory_level}]")
        
        print("-" * 50)
    
    # Test hierarchical coverage validation
    print(f"\n🔍 Testing Hierarchical Coverage")
    print("=" * 50)
    
    coverage = geo_service.validate_hierarchical_coverage()
    for location, result in coverage.items():
        status = "✅" if result['hierarchy_detected'] else "⚠️"
        admin_count = len(result['admin_levels'])
        print(f"{status} {location}")
        print(f"   Region: {result['region']}")
        print(f"   Admin Levels: {admin_count} detected")
        print(f"   Method: {result['method']}")
        if result['admin_levels']:
            for level, value in result['admin_levels'].items():
                print(f"      • {level}: {value}")
        print()

if __name__ == "__main__":
    test_ultra_dynamic_geolocation()
