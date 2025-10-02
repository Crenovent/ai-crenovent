#!/usr/bin/env python3
"""
Environment Setup Script
Helps developers set up their local environment safely
"""
import os
import shutil
import sys
from pathlib import Path

def setup_local_environment():
    """Set up local development environment"""
    print("🔧 Setting up local development environment...")
    
    # Check if .env already exists
    env_file = Path('.env')
    if env_file.exists():
        response = input("⚠️  .env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled.")
            return False
    
    # Copy template to .env
    template_file = Path('.env.template')
    if not template_file.exists():
        print("❌ .env.template not found!")
        return False
    
    shutil.copy(template_file, env_file)
    print("✅ Created .env from template")
    
    # Provide guidance
    print("\n📝 Next steps:")
    print("1. Edit .env file with your actual values")
    print("2. Never commit .env to git")
    print("3. Use different secrets for each environment")
    
    return True

def setup_production_environment():
    """Set up production environment template"""
    print("🚀 Setting up production environment template...")
    
    # Check if .env.production already exists
    env_file = Path('.env.production')
    if env_file.exists():
        response = input("⚠️  .env.production file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled.")
            return False
    
    # Copy template to .env.production
    template_file = Path('.env.template')
    if not template_file.exists():
        print("❌ .env.template not found!")
        return False
    
    shutil.copy(template_file, env_file)
    print("✅ Created .env.production from template")
    
    # Provide guidance
    print("\n📝 Next steps:")
    print("1. Edit .env.production with production values")
    print("2. Use strong, unique secrets")
    print("3. Set up Azure App Service environment variables")
    print("4. Never commit .env.production to git")
    
    return True

def check_gitignore():
    """Check if .gitignore is properly configured"""
    print("🔍 Checking .gitignore configuration...")
    
    gitignore_file = Path('.gitignore')
    if not gitignore_file.exists():
        print("❌ .gitignore not found!")
        return False
    
    gitignore_content = gitignore_file.read_text()
    
    required_patterns = [
        '.env',
        '.env.*',
        '*.secrets',
        'azure-app-settings.json'
    ]
    
    missing_patterns = []
    for pattern in required_patterns:
        if pattern not in gitignore_content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print(f"⚠️  Missing patterns in .gitignore: {missing_patterns}")
        return False
    
    print("✅ .gitignore is properly configured")
    return True

def validate_environment():
    """Validate current environment setup"""
    print("🔍 Validating environment setup...")
    
    # Check for .env file
    if not Path('.env').exists():
        print("❌ .env file not found")
        return False
    
    # Check for secrets in git
    try:
        import subprocess
        result = subprocess.run(['git', 'ls-files', '.env*'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            tracked_files = result.stdout.strip().split('\n')
            dangerous_files = [f for f in tracked_files 
                             if f not in ['.env.template', '.env.example']]
            if dangerous_files:
                print(f"❌ Dangerous files tracked in git: {dangerous_files}")
                print("   Run: git rm --cached <filename> to untrack them")
                return False
    except FileNotFoundError:
        print("⚠️  Git not found, skipping git validation")
    
    print("✅ Environment validation passed")
    return True

def main():
    """Main setup function"""
    print("🔒 RevAI Environment Setup Tool")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/setup-environment.py local      # Set up local development")
        print("  python scripts/setup-environment.py production # Set up production template")
        print("  python scripts/setup-environment.py validate   # Validate current setup")
        print("  python scripts/setup-environment.py check      # Check .gitignore")
        return
    
    command = sys.argv[1].lower()
    
    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    if command == 'local':
        success = setup_local_environment()
    elif command == 'production':
        success = setup_production_environment()
    elif command == 'validate':
        success = validate_environment()
    elif command == 'check':
        success = check_gitignore()
    else:
        print(f"❌ Unknown command: {command}")
        return
    
    if success:
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()