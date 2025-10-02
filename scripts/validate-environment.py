#!/usr/bin/env python3
"""
Environment Validation Script
Validates that the environment is properly configured for deployment
"""
import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, List, Any

class EnvironmentValidator:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.project_root = Path(__file__).parent.parent
        
    def validate_environment_files(self) -> bool:
        """Validate environment file configuration"""
        print("📁 Validating environment files...")
        
        # Check required files
        required_files = {
            '.env.template': 'Template file for environment variables',
            '.gitignore': 'Git ignore file to protect secrets'
        }
        
        for file_name, description in required_files.items():
            file_path = self.project_root / file_name
            if not file_path.exists():
                self.issues.append(f"Missing {file_name} ({description})")
        
        # Check if .env exists for local development
        env_file = self.project_root / '.env'
        if not env_file.exists():
            self.warnings.append(".env file not found - run 'python scripts/setup-environment.py local' to create it")
        
        # Check if production file exists (should not be committed)
        prod_file = self.project_root / '.env.production'
        if prod_file.exists():
            self.warnings.append(".env.production exists - ensure it's not committed to git")
        
        return len(self.issues) == 0
    
    def validate_environment_variables(self) -> bool:
        """Validate required environment variables"""
        print("🔧 Validating environment variables...")
        
        # Load .env file if it exists
        env_file = self.project_root / '.env'
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            if key not in os.environ:
                                os.environ[key] = value
            except Exception as e:
                self.warnings.append(f"Failed to load .env file: {e}")
        
        # Required environment variables
        required_vars = {
            'DATABASE_URL': 'Database connection string',
            'AZURE_OPENAI_API_KEY': 'Azure OpenAI API key',
            'AZURE_OPENAI_ENDPOINT': 'Azure OpenAI endpoint URL'
        }
        
        # Optional but recommended variables
        recommended_vars = {
            'BACKEND_BASE_URL': 'Backend service URL',
            'SERVICE_HOST': 'Service host binding',
            'SERVICE_PORT': 'Service port number',
            'ENVIRONMENT': 'Environment identifier'
        }
        
        # Check required variables
        for var_name, description in required_vars.items():
            if not os.getenv(var_name):
                self.issues.append(f"Missing required environment variable: {var_name} ({description})")
        
        # Check recommended variables
        for var_name, description in recommended_vars.items():
            if not os.getenv(var_name):
                self.warnings.append(f"Missing recommended environment variable: {var_name} ({description})")
        
        return len(self.issues) == 0
    
    def validate_service_urls(self) -> bool:
        """Validate service URL configuration"""
        print("🌐 Validating service URLs...")
        
        # Get URLs from environment
        backend_url = os.getenv('BACKEND_BASE_URL', 'http://localhost:3001')
        ai_service_url = os.getenv('AI_SERVICE_URL', 'http://localhost:8000')
        
        # Validate URL format
        if not backend_url.startswith(('http://', 'https://')):
            self.issues.append(f"Invalid backend URL format: {backend_url}")
        
        if not ai_service_url.startswith(('http://', 'https://')):
            self.issues.append(f"Invalid AI service URL format: {ai_service_url}")
        
        # Test connectivity for remote URLs
        if backend_url.startswith('https://'):
            try:
                response = requests.get(f"{backend_url}/health", timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ Backend service accessible: {backend_url}")
                else:
                    self.warnings.append(f"Backend service returned status {response.status_code}: {backend_url}")
            except requests.exceptions.RequestException as e:
                self.warnings.append(f"Backend service not accessible: {backend_url} - {e}")
        
        return len(self.issues) == 0
    
    def validate_git_configuration(self) -> bool:
        """Validate git configuration for security"""
        print("🔒 Validating git security configuration...")
        
        try:
            import subprocess
            
            # Check if any secret files are tracked
            result = subprocess.run(
                ['git', 'ls-files', '.env*'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                tracked_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
                dangerous_files = [f for f in tracked_files 
                                 if f and not any(safe in f for safe in ['.template', '.example'])]
                
                if dangerous_files:
                    self.issues.append(f"Dangerous environment files tracked in git: {dangerous_files}")
            
        except FileNotFoundError:
            self.warnings.append("Git not found - skipping git validation")
        except Exception as e:
            self.warnings.append(f"Git validation failed: {e}")
        
        return len(self.issues) == 0
    
    def validate_azure_configuration(self) -> bool:
        """Validate Azure-specific configuration"""
        print("☁️ Validating Azure configuration...")
        
        azure_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_CLIENT_ID',
            'AZURE_CLIENT_SECRET',
            'AZURE_TENANT_ID'
        ]
        
        missing_azure_vars = []
        for var in azure_vars:
            if not os.getenv(var):
                missing_azure_vars.append(var)
        
        if missing_azure_vars:
            self.warnings.append(f"Missing Azure configuration: {missing_azure_vars}")
        
        # Check Azure OpenAI endpoint format
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if azure_endpoint and not azure_endpoint.endswith('.cognitiveservices.azure.com/'):
            self.warnings.append("Azure OpenAI endpoint should end with '.cognitiveservices.azure.com/'")
        
        return True
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        return {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'environment': os.getenv('ENVIRONMENT', 'unknown'),
            'total_issues': len(self.issues),
            'total_warnings': len(self.warnings),
            'issues': self.issues,
            'warnings': self.warnings,
            'status': 'PASS' if len(self.issues) == 0 else 'FAIL',
            'environment_variables': {
                'ENVIRONMENT': os.getenv('ENVIRONMENT', 'not_set'),
                'SERVICE_HOST': os.getenv('SERVICE_HOST', 'not_set'),
                'SERVICE_PORT': os.getenv('SERVICE_PORT', 'not_set'),
                'BACKEND_BASE_URL': os.getenv('BACKEND_BASE_URL', 'not_set'),
                'AI_SERVICE_URL': os.getenv('AI_SERVICE_URL', 'not_set'),
            }
        }
    
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        print("🔍 RevAI Environment Validation")
        print("=" * 40)
        
        validations = [
            self.validate_environment_files,
            self.validate_environment_variables,
            self.validate_service_urls,
            self.validate_git_configuration,
            self.validate_azure_configuration
        ]
        
        all_passed = True
        for validation in validations:
            try:
                if not validation():
                    all_passed = False
            except Exception as e:
                self.issues.append(f"Validation failed with error: {str(e)}")
                all_passed = False
        
        print("\n" + "=" * 40)
        
        # Show results
        if self.issues:
            print("❌ Issues found:")
            for issue in self.issues:
                print(f"  • {issue}")
        
        if self.warnings:
            print("⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if all_passed and not self.warnings:
            print("🎉 All validations passed!")
        elif all_passed:
            print("✅ Validations passed with warnings")
        else:
            print("❌ Validation failed!")
        
        return all_passed

def main():
    """Main function"""
    validator = EnvironmentValidator()
    success = validator.run_all_validations()
    
    # Generate report
    report = validator.generate_report()
    report_file = Path('environment-validation-report.json')
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report saved to: {report_file}")
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()