#!/usr/bin/env python3
"""
Production Test Runner
Tests the application with production URLs locally
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_production_test_environment():
    """Set production test environment variables"""
    print("☁️ Setting up production test environment...")
    
    # Set production environment variables
    prod_env = {
        'ENVIRONMENT': 'production',
        'SERVICE_HOST': '0.0.0.0',
        'SERVICE_PORT': '8000',
        'AI_SERVICE_URL': 'https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net',
        'NODEJS_BACKEND_URL': 'https://revai-api-mainv2.azurewebsites.net',
        'BACKEND_BASE_URL': 'https://revai-api-mainv2.azurewebsites.net',
    }
    
    # Update environment
    for key, value in prod_env.items():
        os.environ[key] = value
        print(f"  ✅ {key}={value}")
    
    return prod_env

def check_production_dependencies():
    """Check if production configuration is available"""
    print("🔍 Checking production dependencies...")
    
    # Check if .env.production exists
    prod_env_file = Path('.env.production')
    if prod_env_file.exists():
        print("✅ .env.production found")
        
        # Load production environment variables
        try:
            with open(prod_env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            print("✅ Production environment variables loaded")
        except Exception as e:
            print(f"⚠️  Failed to load .env.production: {e}")
    else:
        print("⚠️  .env.production not found - using environment variables only")
    
    # Check if basic .env exists for fallback
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        return False
    
    return True

def test_production_connectivity():
    """Test connectivity to production services"""
    print("🌐 Testing production service connectivity...")
    
    import requests
    
    services = {
        'Backend API': 'https://revai-api-mainv2.azurewebsites.net/health',
        'AI Service': 'https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net/health'
    }
    
    all_healthy = True
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"  ✅ {service_name}: Healthy")
            else:
                print(f"  ⚠️  {service_name}: Status {response.status_code}")
                all_healthy = False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ {service_name}: Connection failed - {e}")
            all_healthy = False
    
    return all_healthy

def start_application():
    """Start the FastAPI application with production configuration"""
    print("🚀 Starting application with production configuration...")
    
    try:
        # Change to project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        os.chdir(project_root)
        
        # Start the application
        subprocess.run([sys.executable, 'main.py'], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Application failed to start: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("☁️ RevAI Production Test Runner")
    print("=" * 40)
    
    # Check dependencies
    if not check_production_dependencies():
        print("❌ Production dependency check failed!")
        sys.exit(1)
    
    # Setup environment
    env_vars = setup_production_test_environment()
    
    print("\n📋 Production Test Configuration:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")
    
    # Test connectivity
    print("\n🌐 Testing Production Services:")
    connectivity_ok = test_production_connectivity()
    
    if not connectivity_ok:
        print("\n⚠️  Some production services are not accessible.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            sys.exit(1)
    
    print("\n💡 Tips:")
    print("  • This will connect to LIVE production services")
    print("  • Make sure you have proper authentication configured")
    print("  • Press Ctrl+C to stop the application")
    
    print("\n" + "=" * 40)
    
    # Start application
    success = start_application()
    
    if success:
        print("✅ Production test completed successfully!")
    else:
        print("❌ Production test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()