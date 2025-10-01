#!/usr/bin/env python3
"""
Script to add comprehensive null handling to RBA agents
"""

import os
import re

# List of failing RBA agent files that need null handling fixes
failing_agents = [
    'dsl/operators/rba/missing_fields_rba_agent.py',
    'dsl/operators/rba/ownerless_deals_rba_agent.py',
    'dsl/operators/rba/deal_risk_scoring_rba_agent.py',
    'dsl/operators/rba/pipeline_summary_rba_agent.py',
    'dsl/operators/rba/data_quality_rba_agent.py',
    'dsl/operators/rba/coverage_analysis_rba_agent.py'
]

def add_null_checks(file_path):
    """Add comprehensive null checks to a file"""
    
    print(f"Adding null checks to {file_path}...")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Add null check at the beginning of loops that iterate over opportunities
    patterns_to_fix = [
        # Pattern: for opp in opportunities:
        (
            r'(\s+)(for opp in opportunities:)',
            r'\1\2\n\1    if opp is None:\n\1        continue  # Skip None opportunities'
        ),
        
        # Pattern: for deal in deals:
        (
            r'(\s+)(for deal in [^:]+:)',
            r'\1\2\n\1    if deal is None:\n\1        continue  # Skip None deals'
        ),
        
        # Pattern: for opportunity in opportunities:
        (
            r'(\s+)(for opportunity in opportunities:)',
            r'\1\2\n\1    if opportunity is None:\n\1        continue  # Skip None opportunities'
        )
    ]
    
    for pattern, replacement in patterns_to_fix:
        content = re.sub(pattern, replacement, content)
    
    # Add safe dictionary access patterns
    dict_access_patterns = [
        # Pattern: opp.get('Account', {}).get('Name', 'Unknown')
        # This is already safe, but let's make sure Account is checked for None
        (
            r"opp\.get\('Account', \{\}\)\.get\('Name', '[^']*'\)",
            r"(opp.get('Account') or {}).get('Name', 'Unknown')"
        ),
        
        # Pattern: opp.get('Owner', {}).get('Name', 'Unassigned')
        (
            r"opp\.get\('Owner', \{\}\)\.get\('Name', '[^']*'\)",
            r"(opp.get('Owner') or {}).get('Name', 'Unassigned')"
        )
    ]
    
    for pattern, replacement in dict_access_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write the file back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Added null checks to {file_path}")
    else:
        print(f"⚪ No changes needed in {file_path}")

def main():
    print("🔧 Adding comprehensive null handling to failing RBA agents...")
    
    for file_path in failing_agents:
        if os.path.exists(file_path):
            add_null_checks(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    
    print("✅ All RBA agents updated with null handling!")

if __name__ == '__main__':
    main()
