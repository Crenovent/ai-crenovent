#!/usr/bin/env python3
import requests
import json
import time

# Real user data with full modules (simulating what frontend sends)
real_user_data = [
    {
        "Name": "Sarah Chen",
        "Email": "sarah.chen@acmecorp.com",
        "Role": "Sales Manager",
        "Team": "Enterprise Sales",
        "Department": "Sales",
        "Location": "San Francisco",
        "Modules": "Calendar,Lets Meet,Cruxx,RevAIPro Action Center,Rhythm of Business,JIT,Tell Me,Forecasting,Pipeline,Planning,DealDesk,My Customers,Partner Management,Customer Success,Sales Engineering,Marketing operations,Contract Management,Market Intelligence,Compensation,Enablement,Performance Management"
    },
    {
        "Name": "Michael Rodriguez", 
        "Email": "michael.rodriguez@acmecorp.com",
        "Role": "Account Executive",
        "Team": "Mid-Market Sales",
        "Department": "Sales",
        "Location": "New York",
        "Modules": "Calendar,Lets Meet,Cruxx,RevAIPro Action Center,Rhythm of Business,JIT,Tell Me,Forecasting,Pipeline,Planning,DealDesk,My Customers,Partner Management,Customer Success,Sales Engineering,Marketing operations,Contract Management,Market Intelligence,Compensation"
    },
    {
        "Name": "Jennifer Wang",
        "Email": "jennifer.wang@acmecorp.com", 
        "Role": "VP Sales",
        "Team": "Sales Leadership",
        "Department": "Sales",
        "Location": "London",
        "Modules": "Calendar,Lets Meet,Cruxx,RevAIPro Action Center,Rhythm of Business,JIT,Tell Me,Forecasting,Pipeline,Planning,DealDesk,My Customers,Partner Management,Customer Success,Sales Engineering,Marketing operations,Contract Management,Market Intelligence,Compensation,Enablement,Performance Management"
    }
]

def test_execute_workflow():
    """Test the execute-workflow endpoint (CSV upload simulation)"""
    print("🧪 TESTING EXECUTE-WORKFLOW ENDPOINT")
    print("=" * 60)
    
    # Create form data (simulating CSV upload)
    csv_content = "Name,Email,Role,Team,Department,Location\n"
    for user in real_user_data:
        csv_content += f"{user['Name']},{user['Email']},{user['Role']},{user['Team']},{user['Department']},{user['Location']}\n"
    
    files = {
        'file': ('test_users.csv', csv_content, 'text/csv')
    }
    
    data = {
        'tenant_id': '1342',
        'uploaded_by_user_id': '1370',  # Required field
        'save_to_database': 'false',  # Should be disabled now
        'workflow_config': json.dumps({
            "enable_module_assignment": True,
            "industry": "saas"
        })
    }
    
    try:
        print(f"📤 Calling execute-workflow...")
        response = requests.post(
            'http://localhost:8000/api/onboarding/execute-workflow',
            files=files,
            data=data
        )
        
        print(f"📥 Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Execute-workflow SUCCESS")
            print(f"📊 Processed users: {len(result.get('processed_users', []))}")
            
            if result.get('processed_users'):
                first_user = result['processed_users'][0]
                print(f"🎯 FIRST USER MODULES: {first_user.get('Modules', 'NOT FOUND')}")
                print(f"🎯 MODULES TYPE: {type(first_user.get('Modules', 'NOT FOUND'))}")
                
                # Count modules
                modules = first_user.get('Modules', '')
                if isinstance(modules, str):
                    module_count = len(modules.split(',')) if modules else 0
                elif isinstance(modules, list):
                    module_count = len(modules)
                else:
                    module_count = 0
                print(f"🎯 MODULE COUNT: {module_count}")
            
            return result.get('processed_users', [])
        else:
            print(f"❌ Execute-workflow FAILED: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Execute-workflow ERROR: {e}")
        return []

def test_complete_setup(processed_users):
    """Test the complete-setup endpoint"""
    print("\n🧪 TESTING COMPLETE-SETUP ENDPOINT") 
    print("=" * 60)
    
    if not processed_users:
        print("❌ No processed users to test with")
        return
    
    # Prepare form data exactly like frontend
    form_data = {
        'processed_users': json.dumps(processed_users),
        'tenant_id': '1342'
    }
    
    print(f"📤 Sending {len(processed_users)} users to complete-setup...")
    print(f"📤 First user modules preview: {processed_users[0].get('Modules', 'NOT FOUND')[:100]}...")
    
    try:
        response = requests.post(
            'http://localhost:8000/api/onboarding/complete-setup',
            data=form_data
        )
        
        print(f"📥 Response status: {response.status_code}")
        print(f"📥 Response body: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Complete-setup SUCCESS")
            print(f"🎯 RETURNED MODULES: {result.get('first_user_modules', 'NOT FOUND')}")
        else:
            print(f"❌ Complete-setup FAILED")
            
    except Exception as e:
        print(f"❌ Complete-setup ERROR: {e}")

def main():
    print("🚀 FULL FLOW TEST - SIMULATING FRONTEND BEHAVIOR")
    print("=" * 80)
    
    # Test server health first
    try:
        health_response = requests.get('http://localhost:8000/health')
        if health_response.status_code == 200:
            print("✅ Python server is running")
        else:
            print("❌ Python server health check failed")
            return
    except:
        print("❌ Python server is not accessible")
        return
    
    # SKIP execute-workflow due to auth issues
    # Instead, use the exact data format that frontend would send
    print("⚠️ SKIPPING execute-workflow (auth required)")
    print("🔄 USING SIMULATED PROCESSED DATA")
    
    # This simulates the data that would come from execute-workflow
    processed_users = real_user_data  # Use the real data directly
    
    # Step 2: Test complete setup with real data
    test_complete_setup(processed_users)
    
    print("\n🏁 FULL FLOW TEST COMPLETED")
    print("=" * 80)
    print("📝 CHECK PYTHON TERMINAL FOR DEBUG LOGS!")

if __name__ == "__main__":
    main()
