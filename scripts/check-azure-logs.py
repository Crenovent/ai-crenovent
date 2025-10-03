#!/usr/bin/env python3
"""
Azure App Service Log Checker
Helps diagnose deployment issues by checking Azure logs
"""
import subprocess
import sys
import json
from datetime import datetime, timedelta

def check_app_service_logs():
    """Check Azure App Service logs for errors"""
    print("🔍 Checking Azure App Service logs...")
    
    app_name = "RevAI-AI-mainV2"
    resource_group = "rg-newCrenoApp"
    
    try:
        # Get recent logs
        print("📋 Fetching recent application logs...")
        result = subprocess.run([
            'az', 'webapp', 'log', 'tail',
            '--name', app_name,
            '--resource-group', resource_group,
            '--provider', 'application'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("📄 Recent Application Logs:")
            print("-" * 50)
            print(result.stdout)
        else:
            print(f"❌ Failed to get logs: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Log fetch timed out")
    except Exception as e:
        print(f"❌ Error fetching logs: {e}")

def check_app_service_status():
    """Check App Service status and configuration"""
    print("\n🔍 Checking App Service status...")
    
    app_name = "RevAI-AI-mainV2"
    resource_group = "rg-newCrenoApp"
    
    try:
        # Get app service details
        result = subprocess.run([
            'az', 'webapp', 'show',
            '--name', app_name,
            '--resource-group', resource_group,
            '--query', '{state:state,hostNames:hostNames,kind:kind,sku:sku}'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            app_info = json.loads(result.stdout)
            print("📊 App Service Status:")
            print(f"  State: {app_info.get('state', 'Unknown')}")
            print(f"  Host Names: {app_info.get('hostNames', [])}")
            print(f"  Kind: {app_info.get('kind', 'Unknown')}")
            print(f"  SKU: {app_info.get('sku', 'Unknown')}")
        else:
            print(f"❌ Failed to get app status: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")

def check_container_settings():
    """Check container configuration"""
    print("\n🐳 Checking container settings...")
    
    app_name = "RevAI-AI-mainV2"
    resource_group = "rg-newCrenoApp"
    
    try:
        # Get container settings
        result = subprocess.run([
            'az', 'webapp', 'config', 'container', 'show',
            '--name', app_name,
            '--resource-group', resource_group
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            container_info = json.loads(result.stdout)
            print("🐳 Container Configuration:")
            for key, value in container_info.items():
                if key in ['linuxFxVersion', 'dockerRegistryServerUrl', 'dockerRegistryServerUserName']:
                    print(f"  {key}: {value}")
        else:
            print(f"❌ Failed to get container settings: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error checking container settings: {e}")

def check_environment_variables():
    """Check if environment variables are set"""
    print("\n🔧 Checking environment variables...")
    
    app_name = "RevAI-AI-mainV2"
    resource_group = "rg-newCrenoApp"
    
    try:
        # Get app settings
        result = subprocess.run([
            'az', 'webapp', 'config', 'appsettings', 'list',
            '--name', app_name,
            '--resource-group', resource_group,
            '--query', '[].{name:name,value:value}'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            settings = json.loads(result.stdout)
            print("🔧 Environment Variables:")
            
            important_vars = [
                'WEBSITES_PORT', 'SERVICE_PORT', 'SERVICE_HOST',
                'BACKEND_BASE_URL', 'NODEJS_BACKEND_URL',
                'DATABASE_URL', 'AZURE_OPENAI_API_KEY'
            ]
            
            for var in important_vars:
                found = False
                for setting in settings:
                    if setting['name'] == var:
                        value = setting['value']
                        # Mask sensitive values
                        if 'key' in var.lower() or 'secret' in var.lower() or 'password' in var.lower():
                            value = value[:10] + "..." if len(value) > 10 else "***"
                        print(f"  ✅ {var}: {value}")
                        found = True
                        break
                if not found:
                    print(f"  ❌ {var}: NOT SET")
        else:
            print(f"❌ Failed to get app settings: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error checking environment variables: {e}")

def suggest_fixes():
    """Suggest common fixes for 503 errors"""
    print("\n🔧 Common Fixes for 503 Errors:")
    print("=" * 50)
    
    fixes = [
        "1. Set WEBSITES_PORT environment variable to 8000",
        "2. Ensure SERVICE_HOST is set to 0.0.0.0 (not localhost)",
        "3. Check if all required environment variables are set",
        "4. Verify the Docker image was built and pushed correctly",
        "5. Check application startup logs for Python errors",
        "6. Ensure the application listens on the correct port",
        "7. Verify database connectivity from Azure",
        "8. Check if the container registry credentials are correct"
    ]
    
    for fix in fixes:
        print(f"  {fix}")
    
    print("\n🚀 Quick Fix Commands:")
    print("# Set required environment variables")
    print("az webapp config appsettings set \\")
    print("  --resource-group rg-newCrenoApp \\")
    print("  --name RevAI-AI-mainV2 \\")
    print("  --settings \\")
    print("    WEBSITES_PORT=8000 \\")
    print("    SERVICE_HOST=0.0.0.0 \\")
    print("    SERVICE_PORT=8000")
    
    print("\n# Restart the app service")
    print("az webapp restart --name RevAI-AI-mainV2 --resource-group rg-newCrenoApp")

def main():
    """Main diagnostic function"""
    print("🔍 Azure App Service Diagnostic Tool")
    print("=" * 50)
    
    # Check if Azure CLI is available
    try:
        subprocess.run(['az', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Azure CLI not found. Please install Azure CLI first.")
        print("   Download from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        sys.exit(1)
    
    # Run diagnostics
    check_app_service_status()
    check_container_settings()
    check_environment_variables()
    check_app_service_logs()
    suggest_fixes()

if __name__ == "__main__":
    main()