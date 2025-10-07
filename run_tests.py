#!/usr/bin/env python3
"""
Test Runner for RevAI Pro Platform
Comprehensive test execution with reporting and CI/CD integration
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import asyncio

class TestRunner:
    """Test runner for the RevAI Pro platform"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.reports_dir = self.project_root / "reports"
        self.tests_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_command(self, command: List[str], cwd: str = None) -> subprocess.CompletedProcess:
        """Run a command and return the result"""
        print(f"Running: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            cwd=cwd or str(self.project_root),
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result
    
    def install_dependencies(self):
        """Install test dependencies"""
        print("Installing test dependencies...")
        
        # Install main dependencies
        result = self.run_command([
            sys.executable, "-m", "pip", "install", "-r", "requirements.microservices.txt"
        ])
        
        if result.returncode != 0:
            print("Failed to install main dependencies")
            return False
        
        # Install test dependencies
        result = self.run_command([
            sys.executable, "-m", "pip", "install", "-r", "requirements.test.txt"
        ])
        
        if result.returncode != 0:
            print("Failed to install test dependencies")
            return False
        
        print("Dependencies installed successfully")
        return True
    
    def run_linting(self) -> bool:
        """Run code linting"""
        print("Running code linting...")
        
        # Run black
        result = self.run_command([
            sys.executable, "-m", "black", "--check", "--diff", "src/"
        ])
        
        if result.returncode != 0:
            print("Black formatting issues found")
            return False
        
        # Run isort
        result = self.run_command([
            sys.executable, "-m", "isort", "--check-only", "--diff", "src/"
        ])
        
        if result.returncode != 0:
            print("Import sorting issues found")
            return False
        
        # Run flake8
        result = self.run_command([
            sys.executable, "-m", "flake8", "src/"
        ])
        
        if result.returncode != 0:
            print("Flake8 issues found")
            return False
        
        print("Linting passed")
        return True
    
    def run_type_checking(self) -> bool:
        """Run type checking with mypy"""
        print("Running type checking...")
        
        result = self.run_command([
            sys.executable, "-m", "mypy", "src/"
        ])
        
        if result.returncode != 0:
            print("Type checking issues found")
            return False
        
        print("Type checking passed")
        return True
    
    def run_security_scanning(self) -> bool:
        """Run security scanning"""
        print("Running security scanning...")
        
        # Run bandit
        result = self.run_command([
            sys.executable, "-m", "bandit", "-r", "src/", "-f", "json", "-o", "reports/bandit-report.json"
        ])
        
        if result.returncode != 0:
            print("Security issues found by bandit")
            return False
        
        # Run safety
        result = self.run_command([
            sys.executable, "-m", "safety", "check", "--json", "--output", "reports/safety-report.json"
        ])
        
        if result.returncode != 0:
            print("Security vulnerabilities found by safety")
            return False
        
        print("Security scanning passed")
        return True
    
    def run_unit_tests(self, parallel: bool = False) -> bool:
        """Run unit tests"""
        print("Running unit tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_microservices_unit.py",
            "-m", "unit",
            "--cov=src",
            "--cov-report=html:reports/coverage-unit",
            "--cov-report=xml:reports/coverage-unit.xml",
            "--html=reports/pytest-unit-report.html",
            "--self-contained-html",
            "--json-report",
            "--json-report-file=reports/pytest-unit-report.json"
        ]
        
        if parallel:
            command.extend(["-n", "auto"])
        
        result = self.run_command(command)
        
        if result.returncode != 0:
            print("Unit tests failed")
            return False
        
        print("Unit tests passed")
        return True
    
    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        print("Running integration tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_integration.py",
            "-m", "integration",
            "--cov=src",
            "--cov-report=html:reports/coverage-integration",
            "--cov-report=xml:reports/coverage-integration.xml",
            "--html=reports/pytest-integration-report.html",
            "--self-contained-html",
            "--json-report",
            "--json-report-file=reports/pytest-integration-report.json"
        ]
        
        result = self.run_command(command)
        
        if result.returncode != 0:
            print("Integration tests failed")
            return False
        
        print("Integration tests passed")
        return True
    
    def run_e2e_tests(self) -> bool:
        """Run end-to-end tests"""
        print("Running end-to-end tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_e2e_scenarios.py",
            "-m", "e2e",
            "--cov=src",
            "--cov-report=html:reports/coverage-e2e",
            "--cov-report=xml:reports/coverage-e2e.xml",
            "--html=reports/pytest-e2e-report.html",
            "--self-contained-html",
            "--json-report",
            "--json-report-file=reports/pytest-e2e-report.json"
        ]
        
        result = self.run_command(command)
        
        if result.returncode != 0:
            print("End-to-end tests failed")
            return False
        
        print("End-to-end tests passed")
        return True
    
    def run_performance_tests(self) -> bool:
        """Run performance tests"""
        print("Running performance tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_performance.py",
            "-m", "performance",
            "--benchmark-only",
            "--benchmark-save=performance-benchmark",
            "--benchmark-save-data",
            "--html=reports/pytest-performance-report.html",
            "--self-contained-html",
            "--json-report",
            "--json-report-file=reports/pytest-performance-report.json"
        ]
        
        result = self.run_command(command)
        
        if result.returncode != 0:
            print("Performance tests failed")
            return False
        
        print("Performance tests passed")
        return True
    
    def run_security_tests(self) -> bool:
        """Run security tests"""
        print("Running security tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/test_security.py",
            "-m", "security",
            "--html=reports/pytest-security-report.html",
            "--self-contained-html",
            "--json-report",
            "--json-report-file=reports/pytest-security-report.json"
        ]
        
        result = self.run_command(command)
        
        if result.returncode != 0:
            print("Security tests failed")
            return False
        
        print("Security tests passed")
        return True
    
    def run_smoke_tests(self) -> bool:
        """Run smoke tests"""
        print("Running smoke tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-m", "smoke",
            "--maxfail=3",
            "--html=reports/pytest-smoke-report.html",
            "--self-contained-html"
        ]
        
        result = self.run_command(command)
        
        if result.returncode != 0:
            print("Smoke tests failed")
            return False
        
        print("Smoke tests passed")
        return True
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("Generating test report...")
        
        report = {
            "timestamp": time.time(),
            "test_suites": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "coverage": 0
            }
        }
        
        # Collect test results from JSON reports
        json_reports = [
            "pytest-unit-report.json",
            "pytest-integration-report.json",
            "pytest-e2e-report.json",
            "pytest-performance-report.json",
            "pytest-security-report.json"
        ]
        
        for json_report in json_reports:
            report_path = self.reports_dir / json_report
            if report_path.exists():
                with open(report_path, 'r') as f:
                    data = json.load(f)
                    suite_name = json_report.replace("pytest-", "").replace("-report.json", "")
                    report["test_suites"][suite_name] = data
                    
                    # Update summary
                    if "summary" in data:
                        summary = data["summary"]
                        report["summary"]["total_tests"] += summary.get("total", 0)
                        report["summary"]["passed"] += summary.get("passed", 0)
                        report["summary"]["failed"] += summary.get("failed", 0)
                        report["summary"]["skipped"] += summary.get("skipped", 0)
        
        # Save comprehensive report
        with open(self.reports_dir / "comprehensive-test-report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Test report generated: {self.reports_dir / 'comprehensive-test-report.json'}")
    
    def run_all_tests(self, 
                     skip_linting: bool = False,
                     skip_type_checking: bool = False,
                     skip_security_scanning: bool = False,
                     parallel: bool = False,
                     test_types: List[str] = None) -> bool:
        """Run all tests"""
        
        if test_types is None:
            test_types = ["unit", "integration", "e2e", "performance", "security"]
        
        print("Starting comprehensive test suite...")
        start_time = time.time()
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Run code quality checks
        if not skip_linting:
            if not self.run_linting():
                return False
        
        if not skip_type_checking:
            if not self.run_type_checking():
                return False
        
        if not skip_security_scanning:
            if not self.run_security_scanning():
                return False
        
        # Run test suites
        test_results = {}
        
        if "unit" in test_types:
            test_results["unit"] = self.run_unit_tests(parallel=parallel)
        
        if "integration" in test_types:
            test_results["integration"] = self.run_integration_tests()
        
        if "e2e" in test_types:
            test_results["e2e"] = self.run_e2e_tests()
        
        if "performance" in test_types:
            test_results["performance"] = self.run_performance_tests()
        
        if "security" in test_types:
            test_results["security"] = self.run_security_tests()
        
        # Generate comprehensive report
        self.generate_test_report()
        
        # Check overall results
        all_passed = all(test_results.values())
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nTest suite completed in {duration:.2f} seconds")
        print(f"Results: {test_results}")
        
        if all_passed:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        return all_passed

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RevAI Pro Test Runner")
    
    parser.add_argument(
        "--test-types",
        nargs="+",
        choices=["unit", "integration", "e2e", "performance", "security", "smoke"],
        default=["unit", "integration", "e2e", "performance", "security"],
        help="Types of tests to run"
    )
    
    parser.add_argument(
        "--skip-linting",
        action="store_true",
        help="Skip code linting"
    )
    
    parser.add_argument(
        "--skip-type-checking",
        action="store_true",
        help="Skip type checking"
    )
    
    parser.add_argument(
        "--skip-security-scanning",
        action="store_true",
        help="Skip security scanning"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only smoke tests"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.smoke_only:
        success = runner.run_smoke_tests()
    else:
        success = runner.run_all_tests(
            skip_linting=args.skip_linting,
            skip_type_checking=args.skip_type_checking,
            skip_security_scanning=args.skip_security_scanning,
            parallel=args.parallel,
            test_types=args.test_types
        )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
