#!/usr/bin/env python3
"""
Security Check Script
Validates that no secrets are exposed in the codebase
"""
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any

class SecurityChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.issues = []
        
    def check_gitignore(self) -> bool:
        """Check if .gitignore properly protects secrets"""
        print("🔍 Checking .gitignore configuration...")
        
        gitignore_file = self.project_root / '.gitignore'
        if not gitignore_file.exists():
            self.issues.append("❌ .gitignore file not found")
            return False
        
        gitignore_content = gitignore_file.read_text()
        
        required_patterns = [
            '.env',
            '.env.*',
            '*.secrets',
            'azure-app-settings.json',
            'api-keys.txt',
            'tokens.txt',
            '*.key',
            '*.pem'
        ]
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            self.issues.append(f"❌ Missing .gitignore patterns: {missing_patterns}")
            return False
        
        print("✅ .gitignore is properly configured")
        return True
    
    def check_tracked_secrets(self) -> bool:
        """Check if any secret files are tracked in git"""
        print("🔍 Checking for tracked secret files...")
        
        try:
            # Get list of tracked files
            result = subprocess.run(
                ['git', 'ls-files'], 
                cwd=self.project_root,
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                self.issues.append("❌ Failed to check git tracked files")
                return False
            
            tracked_files = result.stdout.strip().split('\n')
            
            # Dangerous file patterns
            dangerous_patterns = [
                r'\.env$',
                r'\.env\.',
                r'.*secrets.*',
                r'.*api[_-]?key.*',
                r'.*password.*',
                r'.*token.*\.txt$',
                r'azure-app-settings\.json$'
            ]
            
            dangerous_files = []
            for file in tracked_files:
                for pattern in dangerous_patterns:
                    if re.search(pattern, file, re.IGNORECASE):
                        # Allow safe template files
                        if not any(safe in file for safe in ['.template', '.example', 'README', '.md']):
                            dangerous_files.append(file)
            
            if dangerous_files:
                self.issues.append(f"❌ Dangerous files tracked in git: {dangerous_files}")
                return False
            
            print("✅ No secret files are tracked in git")
            return True
            
        except FileNotFoundError:
            print("⚠️  Git not found, skipping git validation")
            return True
    
    def check_hardcoded_secrets(self) -> bool:
        """Check for hardcoded secrets in source code"""
        print("🔍 Checking for hardcoded secrets in source code...")
        
        # Patterns that might indicate hardcoded secrets
        secret_patterns = [
            r'api[_-]?key\s*=\s*["\'][^"\']{20,}["\']',
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'secret\s*=\s*["\'][^"\']{20,}["\']',
            r'token\s*=\s*["\'][^"\']{20,}["\']',
            r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']',  # Base64-like strings
        ]
        
        # Files to check
        source_files = []
        for ext in ['*.py', '*.js', '*.ts', '*.json', '*.yml', '*.yaml']:
            source_files.extend(self.project_root.rglob(ext))
        
        # Exclude certain directories
        exclude_dirs = ['venv', 'node_modules', '.git', '__pycache__', 'dsl_env']
        source_files = [f for f in source_files 
                       if not any(exclude in str(f) for exclude in exclude_dirs)]
        
        suspicious_files = []
        for file_path in source_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Filter out obvious false positives
                        real_matches = []
                        for match in matches:
                            if not any(fp in match.lower() for fp in [
                                'your_', 'example', 'placeholder', 'template',
                                'test_', 'dummy', 'fake', 'sample'
                            ]):
                                real_matches.append(match)
                        
                        if real_matches:
                            suspicious_files.append({
                                'file': str(file_path.relative_to(self.project_root)),
                                'matches': real_matches
                            })
            except Exception as e:
                continue
        
        if suspicious_files:
            self.issues.append(f"⚠️  Potentially hardcoded secrets found: {suspicious_files}")
            print("⚠️  Manual review required for potential hardcoded secrets")
            return False
        
        print("✅ No obvious hardcoded secrets found")
        return True
    
    def check_environment_files(self) -> bool:
        """Check environment file configuration"""
        print("🔍 Checking environment file configuration...")
        
        # Check if template exists
        template_file = self.project_root / '.env.template'
        if not template_file.exists():
            self.issues.append("❌ .env.template file not found")
            return False
        
        # Check if .env exists (should exist for local development)
        env_file = self.project_root / '.env'
        if not env_file.exists():
            print("⚠️  .env file not found - run setup script to create it")
        
        # Check if production file exists but warn if it does
        prod_file = self.project_root / '.env.production'
        if prod_file.exists():
            print("⚠️  .env.production exists - ensure it's not committed to git")
        
        print("✅ Environment file configuration looks good")
        return True
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate security check report"""
        return {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'total_issues': len(self.issues),
            'issues': self.issues,
            'status': 'PASS' if len(self.issues) == 0 else 'FAIL'
        }
    
    def run_all_checks(self) -> bool:
        """Run all security checks"""
        print("🔒 Running Security Checks")
        print("=" * 40)
        
        checks = [
            self.check_gitignore,
            self.check_tracked_secrets,
            self.check_hardcoded_secrets,
            self.check_environment_files
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.issues.append(f"❌ Check failed with error: {str(e)}")
                all_passed = False
        
        print("\n" + "=" * 40)
        if all_passed:
            print("🎉 All security checks passed!")
        else:
            print("❌ Security issues found:")
            for issue in self.issues:
                print(f"  {issue}")
        
        return all_passed

def main():
    """Main function"""
    checker = SecurityChecker()
    success = checker.run_all_checks()
    
    # Generate report
    report = checker.generate_report()
    report_file = Path('security-check-report.json')
    
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report saved to: {report_file}")
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()