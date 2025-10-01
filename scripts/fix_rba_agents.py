#!/usr/bin/env python3
"""
Quick script to update all RBA agents to inherit from BaseRBAAgent
"""

import os
import re

# List of RBA agent files to update
rba_agent_files = [
    'dsl/operators/rba/stale_deals_rba_agent.py',
    'dsl/operators/rba/missing_fields_rba_agent.py',
    'dsl/operators/rba/activity_tracking_rba_agent.py',
    'dsl/operators/rba/duplicate_detection_rba_agent.py',
    'dsl/operators/rba/ownerless_deals_rba_agent.py',
    'dsl/operators/rba/quarter_end_dumping_rba_agent.py',
    'dsl/operators/rba/deal_risk_scoring_rba_agent.py',
    'dsl/operators/rba/deals_at_risk_rba_agent.py',
    'dsl/operators/rba/stage_velocity_rba_agent.py',
    'dsl/operators/rba/pipeline_summary_rba_agent.py',
    'dsl/operators/rba/data_quality_rba_agent.py',
    'dsl/operators/rba/forecast_alignment_rba_agent.py',
    'dsl/operators/rba/coverage_analysis_rba_agent.py',
    'dsl/operators/rba/health_overview_rba_agent.py'
]

def update_rba_agent_file(file_path):
    """Update a single RBA agent file"""
    
    print(f"Updating {file_path}...")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace import
    content = re.sub(
        r'from \.\.base import BaseOperator',
        'from .base_rba_agent import BaseRBAAgent',
        content
    )
    
    # Replace class inheritance
    content = re.sub(
        r'class (\w+)\(BaseOperator\):',
        r'class \1(BaseRBAAgent):',
        content
    )
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {file_path}")

def main():
    print("🔧 Fixing RBA agents to inherit from BaseRBAAgent...")
    
    for file_path in rba_agent_files:
        if os.path.exists(file_path):
            update_rba_agent_file(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    
    print("✅ All RBA agents updated!")

if __name__ == '__main__':
    main()
