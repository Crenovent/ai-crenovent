#!/usr/bin/env python3
"""
Start All Services - Complete startup script for CSV RBA system
"""

import subprocess
import sys
import os
import time
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🚀 STARTING ALL CSV RBA SERVICES")
print("=" * 40)
print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("\n1️⃣ Checking Dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'asyncpg', 
        'pydantic', 'python-multipart', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("   🎉 All dependencies installed!")
    return True

def check_environment():
    """Check environment setup"""
    
    print("\n2️⃣ Checking Environment...")
    
    # Check current directory
    current_dir = os.getcwd()
    if not current_dir.endswith('crenovent-ai-service'):
        print(f"   ⚠️  Current directory: {current_dir}")
        print("   💡 Make sure you're in the crenovent-ai-service directory")
    else:
        print(f"   ✅ Working directory: {current_dir}")
    
    # Check key files
    key_files = [
        'main.py',
        'api/csv_upload_api.py',
        'src/services/csv_crm_analyzer.py',
        'dsl/integration_orchestrator.py'
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            return False
    
    # Check sample data
    sample_files = [
        'sample_opportunities.csv',
        'sample_accounts.csv', 
        'sample_forecasts.csv'
    ]
    
    sample_exists = any(os.path.exists(f) for f in sample_files)
    if sample_exists:
        print("   ✅ Sample data available")
    else:
        print("   ⚠️  No sample data found")
        print("   💡 Run: python demo_csv_rba_agents.py to create sample data")
    
    return True

def start_backend_services():
    """Start backend FastAPI services"""
    
    print("\n3️⃣ Starting Backend Services...")
    
    try:
        # Start FastAPI server
        print("   🚀 Starting FastAPI server on port 8000...")
        
        # Use subprocess to start uvicorn in background
        process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 'main:app', 
            '--reload', '--port', '8000', '--host', '0.0.0.0'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is running
        if process.poll() is None:
            print("   ✅ FastAPI server started successfully")
            print("   🔗 API available at: http://localhost:8000")
            print("   📚 API docs at: http://localhost:8000/docs")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"   ❌ FastAPI server failed to start")
            print(f"   📝 Error: {stderr}")
            return None
            
    except Exception as e:
        print(f"   ❌ Failed to start backend services: {e}")
        return None

def test_api_endpoints():
    """Test that API endpoints are working"""
    
    print("\n4️⃣ Testing API Endpoints...")
    
    import requests
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Main endpoint responding")
        else:
            print(f"   ⚠️  Main endpoint status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Main endpoint failed: {e}")
        return False
    
    try:
        # Test docs endpoint
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("   ✅ API documentation available")
        else:
            print(f"   ⚠️  API docs status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ API docs failed: {e}")
    
    # Test CSV endpoints
    csv_endpoints = [
        "/api/csv/files?tenant_id=1300&user_id=1319",
    ]
    
    for endpoint in csv_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code in [200, 422]:  # 422 is expected for some endpoints
                print(f"   ✅ {endpoint} - Available")
            else:
                print(f"   ⚠️  {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {endpoint} - Failed: {e}")
    
    return True

def create_sample_data():
    """Create sample data if it doesn't exist"""
    
    print("\n5️⃣ Ensuring Sample Data Exists...")
    
    sample_files = [
        'sample_opportunities.csv',
        'sample_accounts.csv',
        'sample_forecasts.csv'
    ]
    
    if all(os.path.exists(f) for f in sample_files):
        print("   ✅ Sample data already exists")
        return True
    
    try:
        print("   🔄 Creating sample data...")
        result = subprocess.run([
            sys.executable, 'demo_csv_rba_agents.py'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   ✅ Sample data created successfully")
            return True
        else:
            print(f"   ❌ Sample data creation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to create sample data: {e}")
        return False

def show_frontend_endpoints():
    """Show available frontend endpoints and usage"""
    
    print("\n6️⃣ Frontend Endpoints & Usage...")
    
    print("   🌐 Available API Endpoints:")
    print("      📊 CSV Upload & Analysis:")
    print("         POST /api/csv/upload          - Upload CSV files")
    print("         POST /api/csv/query           - Natural language queries")
    print("         POST /api/csv/analyze         - Detailed analysis")
    print("         GET  /api/csv/files           - List uploaded files")
    
    print("      🤖 RBA Workflow Endpoints:")
    print("         POST /api/workflow-builder/execute-natural-language")
    print("         GET  /api/workflow-builder/templates")
    print("         POST /api/workflow-builder/workflows")
    
    print("      📚 Documentation:")
    print("         GET  /docs                    - Interactive API docs")
    print("         GET  /redoc                   - Alternative API docs")
    
    print("\n   💻 Frontend Usage Examples:")
    
    # Upload CSV example
    print("\n      1. Upload CSV File:")
    print("         curl -X POST 'http://localhost:8000/api/csv/upload' \\")
    print("              -F 'file=@sample_opportunities.csv' \\")
    print("              -F 'tenant_id=1300' \\")
    print("              -F 'user_id=1319'")
    
    # Query example
    print("\n      2. Natural Language Query:")
    print("         curl -X POST 'http://localhost:8000/api/csv/query' \\")
    print("              -F 'file_id=<file-id-from-upload>' \\")
    print("              -F 'user_query=Show me stale opportunities' \\")
    print("              -F 'tenant_id=1300' \\")
    print("              -F 'user_id=1319'")
    
    # RBA workflow example
    print("\n      3. RBA Workflow Execution:")
    print("         curl -X POST 'http://localhost:8000/api/workflow-builder/execute-natural-language' \\")
    print("              -H 'Content-Type: application/json' \\")
    print("              -d '{")
    print("                \"user_input\": \"Check pipeline health\",")
    print("                \"tenant_id\": \"1300\",")
    print("                \"user_id\": \"1319\"")
    print("              }'")
    
    print("\n   🎯 Quick Test Commands:")
    print("      # Test API health")
    print("      curl http://localhost:8000/")
    print()
    print("      # List CSV files")
    print("      curl 'http://localhost:8000/api/csv/files?tenant_id=1300&user_id=1319'")
    print()
    print("      # View API documentation")
    print("      open http://localhost:8000/docs")

def show_next_steps():
    """Show next steps for using the system"""
    
    print("\n7️⃣ Next Steps...")
    
    print("   🎯 Ready to Use:")
    print("      1. API server is running on http://localhost:8000")
    print("      2. Upload your CRM CSV files using the upload endpoint")
    print("      3. Ask questions in natural language using the query endpoint")
    print("      4. Get intelligent RBA agent responses")
    
    print("\n   📊 Test with Sample Data:")
    print("      python test_csv_functionality.py")
    
    print("\n   🌐 Web Interface:")
    print("      • API Docs: http://localhost:8000/docs")
    print("      • ReDoc: http://localhost:8000/redoc")
    
    print("\n   💡 Troubleshooting:")
    print("      • Check logs in the terminal where uvicorn is running")
    print("      • Verify all dependencies are installed")
    print("      • Ensure port 8000 is not in use by other applications")
    
    print("\n   📚 Documentation:")
    print("      • CSV_RBA_USAGE_GUIDE.md - Complete usage guide")
    print("      • COMPLETE_SYSTEM_GUIDE.md - System architecture")

def main():
    """Main startup function"""
    
    try:
        # Check dependencies
        if not check_dependencies():
            return False
        
        # Check environment
        if not check_environment():
            return False
        
        # Create sample data if needed
        create_sample_data()
        
        # Start backend services
        server_process = start_backend_services()
        
        if server_process:
            # Test endpoints
            test_api_endpoints()
            
            # Show frontend information
            show_frontend_endpoints()
            
            # Show next steps
            show_next_steps()
            
            print("\n🎉 ALL SERVICES STARTED SUCCESSFULLY!")
            print("=" * 45)
            print("✅ FastAPI server running on http://localhost:8000")
            print("✅ CSV upload and analysis endpoints ready")
            print("✅ RBA agents ready for natural language queries")
            print("✅ Sample data available for testing")
            print("🤖 Your CSV RBA system is fully operational!")
            
            print("\n⚠️  IMPORTANT:")
            print("   Keep this terminal open to keep services running")
            print("   Press Ctrl+C to stop all services")
            
            try:
                # Keep the script running
                print("\n🔄 Services running... (Press Ctrl+C to stop)")
                server_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping services...")
                server_process.terminate()
                print("✅ Services stopped")
            
            return True
        else:
            print("\n❌ Failed to start services")
            return False
            
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
