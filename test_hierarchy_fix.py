#!/usr/bin/env python3
"""
Test script to verify the hierarchy analysis fix in OnboardingRBAAgent
"""

import asyncio
import os
import sys
from typing import Dict, List, Any

# Add the parent directory to the sys.path to allow importing from dsl.operators.rba
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'dsl')))

from dsl.operators.rba.onboarding_rba_agent import OnboardingRBAAgent

async def test_hierarchy_analysis():
    print("🧪 Testing Hierarchy Analysis Fix")
    print("==================================")

    # Mock CSV data with clear hierarchy
    mock_users_data = [
        {'Full Name': 'Sarah Chen', 'Email': 'sarah.chen@acmecorp.com', 'Manager Email': '', 'Job Title': 'Chief Revenue Officer'},
        {'Full Name': 'Michael Rodriguez', 'Email': 'michael.rodriguez@acmecorp.com', 'Manager Email': 'sarah.chen@acmecorp.com', 'Job Title': 'VP Sales'},
        {'Full Name': 'Lisa Thompson', 'Email': 'lisa.thompson@acmecorp.com', 'Manager Email': 'michael.rodriguez@acmecorp.com', 'Job Title': 'Sales Director'},
        {'Full Name': 'Mark Stevens', 'Email': 'mark.stevens@acmecorp.com', 'Manager Email': 'lisa.thompson@acmecorp.com', 'Job Title': 'Senior Account Executive'},
        {'Full Name': 'Alex Johnson', 'Email': 'alex.johnson@acmecorp.com', 'Manager Email': 'mark.stevens@acmecorp.com', 'Job Title': 'Junior Account Executive'},
    ]

    # Mock context and config
    mock_context = {
        'users_data': mock_users_data, 
        'workflow_config': {
            'industry': 'saas',
            'default_user_role': 'user'
        }
    }
    
    mock_config = {
        'enable_smart_location_mapping': False,
        'preserve_existing_data': True,
        'enable_module_assignment': True,
        'default_user_role': 'user'
    }

    # Create agent instance
    agent = OnboardingRBAAgent()
    
    # Manually set field_mappings for the test
    agent.field_mappings = agent._initialize_field_mappings(mock_users_data)

    print(f"📊 Processing {len(mock_users_data)} users...")
    
    # Execute the RBA logic
    result = await agent._execute_rba_logic(mock_context, mock_config)
    
    if result.get('success'):
        processed_users = result.get('processed_users', [])
        print(f"✅ Successfully processed {len(processed_users)} users")
        
        # Check hierarchy analysis results
        print("\n🔍 Hierarchy Analysis Results:")
        print("=" * 50)
        
        for user in processed_users:
            name = user.get('Full Name', 'Unknown')
            email = user.get('Email', 'Unknown')
            has_direct_reports = user.get('has_direct_reports', False)
            is_leaf_node = user.get('is_leaf_node', True)
            modules = user.get('Modules', 'None')
            
            print(f"👤 {name} ({email})")
            print(f"   📊 has_direct_reports: {has_direct_reports}")
            print(f"   📊 is_leaf_node: {is_leaf_node}")
            print(f"   📊 modules: {modules}")
            print()
        
        # Verify expected hierarchy
        print("🎯 Expected vs Actual Results:")
        print("=" * 50)
        
        expected_results = {
            'sarah.chen@acmecorp.com': {'has_direct_reports': True, 'is_leaf_node': False},  # CRO - should have direct reports
            'michael.rodriguez@acmecorp.com': {'has_direct_reports': True, 'is_leaf_node': False},  # VP - should have direct reports
            'lisa.thompson@acmecorp.com': {'has_direct_reports': True, 'is_leaf_node': False},  # Director - should have direct reports
            'mark.stevens@acmecorp.com': {'has_direct_reports': True, 'is_leaf_node': False},  # Senior AE - should have direct reports
            'alex.johnson@acmecorp.com': {'has_direct_reports': False, 'is_leaf_node': True},  # Junior AE - should be leaf node
        }
        
        all_correct = True
        for user in processed_users:
            email = user.get('Email', '')
            if email in expected_results:
                expected = expected_results[email]
                actual_has_reports = user.get('has_direct_reports', False)
                actual_is_leaf = user.get('is_leaf_node', True)
                
                has_reports_correct = actual_has_reports == expected['has_direct_reports']
                is_leaf_correct = actual_is_leaf == expected['is_leaf_node']
                
                status = "✅" if (has_reports_correct and is_leaf_correct) else "❌"
                print(f"{status} {user.get('Full Name', 'Unknown')}: has_reports={actual_has_reports} (expected {expected['has_direct_reports']}), is_leaf={actual_is_leaf} (expected {expected['is_leaf_node']})")
                
                if not (has_reports_correct and is_leaf_correct):
                    all_correct = False
        
        print(f"\n🎉 Overall Result: {'✅ ALL CORRECT' if all_correct else '❌ SOME INCORRECT'}")
        
    else:
        print(f"❌ RBA execution failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_hierarchy_analysis())
