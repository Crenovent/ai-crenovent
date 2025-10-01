#!/usr/bin/env python3

import requests
import json

def test_ultra_dynamic_integration():
    """Test the integrated ultra-dynamic geolocation in RBA workflow"""
    
    print("🌍 Testing Ultra-Dynamic Geolocation Integration")
    print("=" * 60)
    
    # Test CSV with hierarchical location data
    csv_content = '''Name,Email,Location,Job Title,Department,City,State,Country,Area
Raj Patel,raj@test.com,Mumbai Maharashtra India,Sales Manager,Sales,Mumbai,Maharashtra,India,Bandra West
John Doe,john@test.com,New York NY USA,Account Executive,Sales,New York,NY,USA,Manhattan
Jane Smith,jane@test.com,London UK,Sales Director,Sales,London,,UK,
Hans Mueller,hans@test.com,Berlin Germany,VP Sales,Sales,Berlin,,Germany,
Yuki Tanaka,yuki@test.com,Tokyo Japan,Account Manager,Sales,Tokyo,,Japan,
Ahmed Al-Rashid,ahmed@test.com,Dubai UAE,Regional Manager,Sales,Dubai,,UAE,
Carlos Silva,carlos@test.com,São Paulo Brazil,Sales Rep,Sales,São Paulo,São Paulo,Brazil,'''

    files = {'file': ('test_hierarchical.csv', csv_content, 'text/csv')}
    data = {
        'tenant_id': '1300',
        'uploaded_by_user_id': '1',
        'workflow_config': json.dumps({
            'enable_region_assignment': True,
            'enable_segment_assignment': True,
            'enable_smart_location_mapping': True,
            'region_distribution_strategy': 'smart_location',
            'create_hierarchical_structure': True
        }),
        'execution_mode': 'sync'
    }

    try:
        response = requests.post('http://localhost:8000/api/onboarding/execute-workflow', files=files, data=data)
        print(f'Status: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('results'):
                users = result['results']['processed_users']
                print(f"\n✅ Successfully processed {len(users)} users with ultra-dynamic geolocation:")
                print("=" * 80)
                
                for user in users:
                    print(f"\n👤 {user['Name']}")
                    print(f"   📍 Input Location: {user.get('Location', 'N/A')}")
                    print(f"   🌍 Region: {user.get('Region', 'N/A')}")
                    print(f"   🏢 Territory: {user.get('Territory', 'N/A')}")
                    print(f"   🏛️ Area: {user.get('Area', 'N/A')}")
                    print(f"   🏘️ District: {user.get('District', 'N/A')}")
                    
                    # Show hierarchical admin levels
                    if user.get('admin_levels'):
                        print(f"   📊 Admin Levels:")
                        for level, value in user['admin_levels'].items():
                            print(f"      • {level.title()}: {value}")
                    
                    # Show territory metadata
                    if user.get('territory_metadata'):
                        print(f"   🏷️ Territory Metadata:")
                        metadata = user['territory_metadata']
                        if metadata.get('territory_name'):
                            print(f"      • Territory: {metadata['territory_name']} ({metadata.get('territory_type', 'N/A')})")
                        if metadata.get('area_name'):
                            print(f"      • Area: {metadata['area_name']} ({metadata.get('area_type', 'N/A')})")
                    
                    # Show geolocation metadata
                    if user.get('geolocation_metadata'):
                        geo = user['geolocation_metadata']
                        print(f"   🔧 Detection: {geo.get('detection_method', 'N/A')} (Confidence: {geo.get('confidence', 0):.2f})")
                    
                    print("-" * 60)
                
                # Summary
                regions = [user.get('Region') for user in users]
                methods = [user.get('geolocation_metadata', {}).get('detection_method') for user in users]
                
                print(f"\n📊 SUMMARY:")
                print(f"   • Regions detected: {set(filter(None, regions))}")
                print(f"   • Detection methods: {set(filter(None, methods))}")
                print(f"   • Users with hierarchical data: {len([u for u in users if u.get('admin_levels')])}")
                print(f"   • Users with territory metadata: {len([u for u in users if u.get('territory_metadata')])}")
                
                # Validate expected results
                expected_regions = {
                    'Raj Patel': 'apac',      # India
                    'John Doe': 'north_america',  # USA
                    'Jane Smith': 'emea',     # UK
                    'Hans Mueller': 'emea',   # Germany
                    'Yuki Tanaka': 'apac',    # Japan
                    'Ahmed Al-Rashid': 'emea', # UAE (business override)
                    'Carlos Silva': 'latam'   # Brazil
                }
                
                print(f"\n✅ VALIDATION:")
                all_correct = True
                for user in users:
                    name = user['Name']
                    actual_region = user.get('Region')
                    expected_region = expected_regions.get(name)
                    
                    if actual_region == expected_region:
                        print(f"   ✅ {name}: {actual_region} (CORRECT)")
                    else:
                        print(f"   ❌ {name}: got {actual_region}, expected {expected_region}")
                        all_correct = False
                
                if all_correct:
                    print(f"\n🎉 ALL ASSIGNMENTS CORRECT! Ultra-dynamic geolocation working perfectly!")
                else:
                    print(f"\n⚠️ Some assignments incorrect - check logs for details")
                    
            else:
                print(f'❌ RBA Error: {result}')
        else:
            print(f'❌ HTTP Error: {response.text}')
            
    except Exception as e:
        print(f'❌ Request Error: {e}')

if __name__ == "__main__":
    test_ultra_dynamic_integration()
