#!/usr/bin/env python3
"""
Task 4.3.25: Tenant Isolation Test Runner
Simple script to run tenant isolation verification tests
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_tenant_isolation import run_tenant_isolation_tests, generate_isolation_report

async def main():
    """Run tenant isolation tests and generate report"""
    print("🚀 Starting Tenant Isolation Verification...")
    print("=" * 60)
    
    try:
        # Run the test suite
        results = await run_tenant_isolation_tests()
        
        # Generate and save report
        report = generate_isolation_report(results)
        
        # Save to file
        report_file = "tenant_isolation_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: {report_file}")
        
        # Print summary
        success_rate = (results['passed'] / results['total_tests']) * 100 if results['total_tests'] > 0 else 0
        
        if success_rate >= 95:
            print("🎉 RESULT: Tenant isolation is EXCELLENT!")
            return 0
        elif success_rate >= 80:
            print("⚠️  RESULT: Tenant isolation needs minor improvements")
            return 1
        else:
            print("🚨 RESULT: CRITICAL tenant isolation issues detected!")
            return 2
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
