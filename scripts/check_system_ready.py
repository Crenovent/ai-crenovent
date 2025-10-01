#!/usr/bin/env python3
"""
Check if the corrected RBA system is ready for testing
"""

import requests
import time
import json

def check_system_status():
    print("🔍 Checking System Readiness...")
    
    base_url = "http://localhost:8000"
    
    # Check if service is responding
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Service is responding")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            
            components = health_data.get('components', {})
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}")
            
            return True
        else:
            print(f"❌ Service not healthy: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Service not accessible - still starting up")
        return False
    except Exception as e:
        print(f"❌ Error checking service: {e}")
        return False

def test_api_endpoints():
    print("\n🧪 Testing API Endpoints...")
    
    base_url = "http://localhost:8000"
    
    endpoints = [
        ("/api/builder/health", "Workflow Builder Health"),
        ("/api/builder/agents", "Available Agents"),
        ("/api/builder/templates", "Workflow Templates")
    ]
    
    results = {}
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"   ✅ {name}")
                results[endpoint] = "OK"
            else:
                print(f"   ❌ {name}: HTTP {response.status_code}")
                results[endpoint] = f"HTTP_{response.status_code}"
        except Exception as e:
            print(f"   ❌ {name}: {str(e)[:50]}...")
            results[endpoint] = "ERROR"
    
    return results

def main():
    print("🚀 SYSTEM READINESS CHECK")
    print("=" * 50)
    
    max_attempts = 10
    attempt = 1
    
    while attempt <= max_attempts:
        print(f"\n📊 Attempt {attempt}/{max_attempts}")
        
        if check_system_status():
            print("\n✅ System is responding!")
            
            # Test API endpoints
            api_results = test_api_endpoints()
            
            # Check if core endpoints are working
            core_working = all(
                result == "OK" for result in api_results.values()
            )
            
            if core_working:
                print("\n🎉 SYSTEM IS READY!")
                print("✅ All core components operational")
                print("✅ API endpoints responding")
                print("✅ Ready for RBA workflow testing")
                print(f"\n🔗 Access UI: http://localhost:8000/ui/workflow_builder_ui.html")
                print(f"📚 API Docs: http://localhost:8000/docs")
                return True
            else:
                print(f"\n⏳ System responding but APIs not fully ready")
                print("   Waiting for initialization to complete...")
        else:
            print(f"⏳ System still initializing...")
        
        if attempt < max_attempts:
            print(f"   Waiting 10 seconds before next check...")
            time.sleep(10)
        
        attempt += 1
    
    print(f"\n⚠️ System not ready after {max_attempts} attempts")
    print("   Check the service logs for initialization issues")
    return False

if __name__ == "__main__":
    main()
