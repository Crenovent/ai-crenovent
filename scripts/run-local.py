#!/usr/bin/env python3
"""
Local Development Runner
Sets up local environment and starts the application
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_local_environment():
    """Set local environment variables"""
    print("🏠 Setting up local development environment...")
    
    # Set local environment variables
    local_env = {
        'ENVIRONMENT': 'local',
        'SERVICE_HOST': '0.0.0.0',
        'SERVICE_PORT': '8000',
        'AI_SERVICE_URL': 'http://localhost:8000',
        'NODEJS_BACKEND_URL': 'http://localhost:3001',
        'BACKEND_BASE_URL': 'http://localhost:3001',
    }
    
    # Update environment
    for key, value in local_env.items():
        os.environ[key] = value
        print(f"  ✅ {key}={value}")
    
    return local_env

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from template...")
        template_file = Path('.env.template')
        if template_file.exists():
            import shutil
            shutil.copy(template_file, env_file)
            print("✅ Created .env from template")
        else:
            print("❌ .env.template not found!")
            return False
    
    return True

def start_application():
    """Start the FastAPI application"""
    print("🚀 Starting FastAPI application...")
    
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
    print("🔧 RevAI Local Development Runner")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed!")
        sys.exit(1)
    
    # Setup environment
    env_vars = setup_local_environment()
    
    print("\n📋 Local Development Configuration:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")
    
    print("\n💡 Tips:")
    print("  • Make sure your Node.js backend is running on port 3001")
    print("  • Check .env file for database and API key configuration")
    print("  • Press Ctrl+C to stop the application")
    
    print("\n" + "=" * 40)
    
    # Start application
    success = start_application()
    
    if success:
        print("✅ Application started successfully!")
    else:
        print("❌ Failed to start application!")
        sys.exit(1)

if __name__ == "__main__":
    main()