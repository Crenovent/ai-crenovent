"""
Quick Setup Verification Script
Checks if all services and components are ready for testing
"""

import subprocess
import sys
import os
import time
import requests
from datetime import datetime

def check_service_status(name, url, timeout=5):
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name}: Running (Status: {response.status_code})")
            return True
        else:
            print(f"⚠️ {name}: Responding but with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: Not running or not accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"⏰ {name}: Timeout (service may be slow)")
        return False
    except Exception as e:
        print(f"❌ {name}: Error - {str(e)}")
        return False

def check_python_deps():
    """Check if required Python dependencies are installed"""
    required_deps = ['aiohttp', 'fastapi', 'uvicorn', 'psycopg2', 'azure-storage-blob']
    missing_deps = []
    
    for dep in required_deps:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ Python dependency: {dep}")
        except ImportError:
            print(f"❌ Missing Python dependency: {dep}")
            missing_deps.append(dep)
    
    return len(missing_deps) == 0, missing_deps

def check_env_variables():
    """Check if required environment variables are set"""
    required_vars = [
        'DATABASE_URL',
        'AZURE_OPENAI_ENDPOINT', 
        'AZURE_OPENAI_KEY',
        'AZURE_STORAGE_CONNECTION_STRING',
        'AZURE_STORAGE_CONTAINER_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ Environment variable: {var}")
        else:
            print(f"❌ Missing environment variable: {var}")
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars

def check_ports():
    """Check if required ports are available"""
    import socket
    
    ports_to_check = [
        (3001, "Node.js Backend"),
        (8001, "Python DSL Service"),
        (3000, "React Frontend (optional)")
    ]
    
    available_ports = []
    for port, service in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            print(f"✅ Port {port} available for {service}")
            available_ports.append(port)
        except OSError:
            print(f"⚠️ Port {port} in use ({service} may be running)")
        finally:
            sock.close()
    
    return available_ports

def main():
    print("🔍 CRENOVENT AI SERVICE SETUP VERIFICATION")
    print("=" * 50)
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check 1: Python Dependencies
    print("1️⃣ Checking Python Dependencies...")
    deps_ok, missing_deps = check_python_deps()
    print()
    
    # Check 2: Environment Variables  
    print("2️⃣ Checking Environment Variables...")
    env_ok, missing_vars = check_env_variables()
    print()
    
    # Check 3: Port Availability
    print("3️⃣ Checking Port Availability...")
    available_ports = check_ports()
    print()
    
    # Check 4: Service Status
    print("4️⃣ Checking Service Status...")
    backend_status = check_service_status("Node.js Backend", "http://localhost:3001/health")
    dsl_status = check_service_status("DSL Service", "http://localhost:8001/health")
    frontend_status = check_service_status("React Frontend", "http://localhost:3000")
    print()
    
    # Summary and Recommendations
    print("📋 SETUP SUMMARY")
    print("-" * 30)
    
    if deps_ok and env_ok:
        print("✅ Environment: Ready")
    else:
        print("❌ Environment: Issues detected")
        
    if backend_status and dsl_status:
        print("✅ Services: All critical services running")
    elif dsl_status:
        print("⚠️ Services: DSL service running, backend needs to be started")
    elif backend_status:
        print("⚠️ Services: Backend running, DSL service needs to be started")
    else:
        print("❌ Services: Both backend and DSL service need to be started")
    
    print()
    print("🚀 NEXT STEPS:")
    
    if missing_deps:
        print(f"   📦 Install missing dependencies: pip install {' '.join(missing_deps)}")
    
    if missing_vars:
        print(f"   🔧 Set missing environment variables: {', '.join(missing_vars)}")
    
    if not backend_status:
        print("   🟢 Start Node.js backend: cd ../crenovent-backend && npm start")
    
    if not dsl_status:
        print("   🐍 Start DSL service: python dsl_server.py")
    
    if backend_status and dsl_status:
        print("   🧪 Run integration tests: python test_policy_aware_integration.py")
        print("   🎯 Test specific features: python test_foundation_validation.py")
    
    print()
    print("📚 TESTING COMMANDS:")
    print("   • Complete integration test: python test_policy_aware_integration.py")
    print("   • Foundation validation: python test_foundation_validation.py")
    print("   • DSL functionality: python test_dsl_complete.py")
    print("   • Enterprise testing: python test_enterprise_system.py")

if __name__ == "__main__":
    main()
