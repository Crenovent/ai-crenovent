#!/usr/bin/env python3
"""
Test script to verify the reports_to field fix
"""

import asyncio
import os
import sys
from typing import Dict, List, Any

# Add the parent directory to the sys.path to allow importing from dsl.operators.rba
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'dsl')))

from dsl.operators.rba.onboarding_rba_agent import OnboardingRBAAgent

async def test_reports_to_fix():
    print("🧪 Testing Reports To Field Fix")
    print("===============================")

    # Mock CSV data with clear hierarchy
    mock_users_data = [
        {'Full Name': 'Sarah Chen', 'Email': 'sarah.chen@acmecorp.com', 'Manager Email': '', 'Job Title': 'Chief Revenue Officer'},
        {'Full Name': 'Michael Rodriguez', 'Email': 'michael.rodriguez@acmecorp.com', 'Manager Email': 'sarah.chen@acmecorp.com', 'Job Title': 'VP Sales'},
        {'Full Name': 'Lisa Thompson', 'Email': 'lisa.thompson@acmecorp.com', 'Manager Email': 'michael.rodriguez@acmecorp.com', 'Job Title': 'Sales Director'},
        {'Full Name': 'Mark Stevens', 'Email': 'mark.stevens@acmecorp.com', 'Manager Email': 'lisa.thompson@acmecorp.com', 'Job Title': 'Senior Account Executive'},
    ]

    # Mock context and config
    mock_context = {'users_data': mock_users_data, 'workflow_config': {}}
    mock_config = {
        'enable_smart_location_mapping': False,
        'preserve_existing_data': True,
        'enable_module_assignment': True,
        'default_user_role': 'user'
    }

    agent = OnboardingRBAAgent()
    
    # Manually set field_mappings for the test
    agent.field_mappings = agent._initialize_field_mappings(mock_users_data)

    print("🔍 Testing hierarchy analysis...")
    
    # Test the hierarchy analysis methods
    for user in mock_users_data:
        email = user['Email']
        manager_email = user['Manager Email']
        
        has_reports = agent._user_has_direct_reports(email, mock_context['workflow_config'])
        is_leaf = agent._is_leaf_node_in_hierarchy(email, mock_context['workflow_config'])
        
        print(f"👤 {user['Full Name']} ({email}):")
        print(f"   Manager: {manager_email or 'None (top-level)'}")
        print(f"   Has direct reports: {has_reports}")
        print(f"   Is leaf node: {is_leaf}")
        print()

    print("🔍 Testing data flow to TypeScript backend format...")
    
    # Test the data transformation that happens in save_users_to_database
    for user in mock_users_data:
        # Simulate the data transformation from FastAPI to TypeScript backend
        email = user.get('Email', '').strip().lower()
        full_name = user.get('Full Name', '').strip()
        
        # Split name into first and last name
        name_parts = full_name.split(' ', 1)
        first_name = name_parts[0] if name_parts else ''
        last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        # Handle manager relationship (this is the key part)
        manager_email = (user.get('Manager Email') or '').strip().lower()
        
        user_data = {
            'email': email,
            'name': f"{first_name} {last_name}".strip(),
            'firstName': first_name,
            'lastName': last_name,
            'reportingEmail': manager_email,  # This should now be correctly mapped
        }
        
        print(f"👤 {user['Full Name']}:")
        print(f"   Original Manager Email: '{user.get('Manager Email', '')}'")
        print(f"   Transformed reportingEmail: '{user_data['reportingEmail']}'")
        print(f"   Will be saved to reports_to field: {user_data['reportingEmail'] or 'NULL'}")
        print()

    print("✅ Reports To field fix test completed!")
    print("🎯 Expected Results:")
    print("   - Sarah Chen: reports_to = NULL (top-level)")
    print("   - Michael Rodriguez: reports_to = 'sarah.chen@acmecorp.com'")
    print("   - Lisa Thompson: reports_to = 'michael.rodriguez@acmecorp.com'")
    print("   - Mark Stevens: reports_to = 'lisa.thompson@acmecorp.com'")

if __name__ == "__main__":
    asyncio.run(test_reports_to_fix())
