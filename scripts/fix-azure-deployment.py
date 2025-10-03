#!/usr/bin/env python3
"""
Azure Deployment Fix Script
Fixes common 503 deployment issues
"""
import subprocess
import sys
import time

def set_required_environment_variables():
    """Set required environment variables for Azure App Service"""
    print("🔧 Setting required environment variables...")
    
    app_name = "RevAI-AI-mainV2"
    resource_group = "rg-newCrenoApp"
    
    # Critical environment variables for Azure App Service
    env_vars = {
        'WEBSITES_PORT': '8000',
        'SERVICE_HOST': '0.0.0.0',
        'SERVICE_PORT': '8000',
        'ENVIRONMENT': 'production',
        'BACKEND_BASE_URL': 'https://revai-api-mainv2.azurewebsites.net',
        'NODEJS_BACKEND_URL': 'https://revai-api-mainv2.azurewebsites.net',
        'AI_SERVICE_URL': 'https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net'
    }
    
    # Build the az command
    settings_args = []
    for key, value in env_vars.items():
        settings_args.extend(['--settings', f'{key}={value}'])
    
    try:
        cmd = [
            'az', 'webapp', 'config', 'appsettings', 'set',
            '--resource-group', resource_group,
            '--name', app_name
        ] + settings_args
        
        print("Running command:")
        print(" ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Environment variables set successfully!")
            return True
        else:
            print(f"❌ Failed to set environment variables: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting environment variables: {e}")
        return False

def restart_app_service():
    """Restart the Azure App Service"""
    print("🔄 Restarting Azure App Service...")
    
    app_name = "RevAI-AI-mainV2"
    resource_group = "rg-newCrenoApp"
    
    try:
        result = subprocess.run([
            'az', 'webapp', 'restart',
            '--name', app_name,
            '--resource-group', resource_group
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ App Service restarted successfully!")
            return True
        else:
            print(f"❌ Failed to restart app service: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error restarting app service: {e}")
        return False

def wait_for_app_to_start():
    """Wait for the application to start and test health endpoint"""
    print("⏳ Waiting for application to start...")
    
    app_url = "https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net"
    health_url = f"{app_url}/health"
    
    import requests
    
    max_attempts = 20
    wait_time = 15  # seconds
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Attempt {attempt}/{max_attempts}: Testing {health_url}")
            response = requests.get(health_url, timeout=30)
            
            if response.status_code == 200:
                print("🎉 Application is now healthy!")
                print(f"✅ Health check passed: {response.status_code}")
                print(f"🌐 Application URL: {app_url}")
                return True
            else:
                print(f"⚠️  Health check returned: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
        
        if attempt < max_attempts:
            print(f"⏳ Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)
    
    print("❌ Application failed to start after all attempts")
    return False

def check_logs():
    """Check recent application logs"""
    print("📋 Checking recent application logs...")
    
    app_name = "RevAI-AI-mainV2"
    resource_group = "rg-newCrenoApp"
    
    try:
        # Get recent logs (last 100 lines)
        result = subprocess.run([
            'az', 'webapp', 'log', 'download',
            '--name', app_name,
            '--resource-group', resource_group,
            '--log-file', 'app-logs.zip'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Logs downloaded to app-logs.zip")
            print("💡 Extract and check the logs for detailed error information")
        else:
            print(f"⚠️  Could not download logs: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Log download timed out")
    except Exception as e:
        print(f"❌ Error downloading logs: {e}")

def main():
    """Main fix function"""
    print("🔧 Azure App Service 503 Fix Tool")
    print("=" * 50)
    
    # Check if Azure CLI is available
    try:
        subprocess.run(['az', '--version'], capture_output=True, check=True)
        print("✅ Azure CLI is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Azure CLI not found. Please install Azure CLI first.")
        sys.exit(1)
    
    # Step 1: Set environment variables
    if not set_required_environment_variables():
        print("❌ Failed to set environment variables")
        sys.exit(1)
    
    # Step 2: Restart app service
    if not restart_app_service():
        print("❌ Failed to restart app service")
        sys.exit(1)
    
    # Step 3: Wait for app to start
    if wait_for_app_to_start():
        print("🎉 Deployment fix successful!")
    else:
        print("❌ Application still not responding")
        print("📋 Checking logs for more information...")
        check_logs()
        
        print("\n🔍 Additional troubleshooting steps:")
        print("1. Check the downloaded logs (app-logs.zip)")
        print("2. Verify Docker image was built correctly")
        print("3. Check database connectivity")
        print("4. Verify all required environment variables are set")

if __name__ == "__main__":
    main()