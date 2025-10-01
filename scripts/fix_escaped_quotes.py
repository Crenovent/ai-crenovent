#!/usr/bin/env python3
"""
Script to fix escaped quotes in RBA agents
"""

import os
import re

# List of RBA agent files to update
rba_agent_files = [
    'dsl/operators/rba/forecast_alignment_rba_agent.py',
    'dsl/operators/rba/stale_deals_rba_agent.py',
    'dsl/operators/rba/activity_tracking_rba_agent.py',
    'dsl/operators/rba/duplicate_detection_rba_agent.py',
    'dsl/operators/rba/ownerless_deals_rba_agent.py',
    'dsl/operators/rba/deal_risk_scoring_rba_agent.py',
    'dsl/operators/rba/deals_at_risk_rba_agent.py',
    'dsl/operators/rba/stage_velocity_rba_agent.py',
    'dsl/operators/rba/pipeline_summary_rba_agent.py',
    'dsl/operators/rba/sandbagging_rba_agent.py',
    'dsl/operators/rba/health_overview_rba_agent.py',
    'dsl/operators/rba/coverage_analysis_rba_agent.py'
]

def fix_escaped_quotes(file_path):
    """Fix escaped quotes in a file"""
    
    print(f"Fixing {file_path}...")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace escaped quotes
    content = content.replace("\\'", "'")
    
    # Write the file back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed {file_path}")
    else:
        print(f"⚪ No changes needed in {file_path}")

def main():
    print("🔧 Fixing escaped quotes in RBA agents...")
    
    for file_path in rba_agent_files:
        if os.path.exists(file_path):
            fix_escaped_quotes(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    
    print("✅ All RBA agents fixed!")

if __name__ == '__main__':
    main()
