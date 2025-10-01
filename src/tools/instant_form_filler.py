"""
Instant Form Filler - Immediate Form Population Tool
Fast, reliable form filling without complex dependencies
"""

import re
import json
from typing import Dict, Any
from datetime import datetime, timedelta

def fill_form_instantly(message: str) -> Dict[str, Any]:
    """
    Instantly extract and return form data from user message
    No async, no database dependencies - just fast pattern matching
    """
    form_data = {}
    message_clean = message.strip()
    
    # Extract account name (prioritize Microsoft, Apple, etc.)
    account_patterns = [
        r'plan for ([A-Za-z0-9\s&\.]+?)(?:\s+with|\s+account|\s*$)',
        r'(?:create|build).*?plan.*?for\s+([A-Za-z0-9\s&\.]+?)(?:\s+with|\s*$)',
        r'strategic.*?plan.*?([A-Za-z0-9\s&\.]+?)(?:\s+with|\s*$)'
    ]
    
    account_name = None
    for pattern in account_patterns:
        match = re.search(pattern, message_clean, re.IGNORECASE)
        if match:
            account_name = match.group(1).strip()
            break
    
    if account_name:
        form_data['plan_name'] = f"Strategic Plan - {account_name} - {datetime.now().strftime('%Y Q%m')}"
        form_data['account_id'] = account_name
    
    # Extract ARR/Revenue target
    arr_patterns = [
        r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:M|million)',
        r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:M|million)\s*ARR',
        r'ARR.*?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:M|million)?',
        r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*ARR'
    ]
    
    for pattern in arr_patterns:
        match = re.search(pattern, message_clean, re.IGNORECASE)
        if match:
            arr_str = match.group(1).replace(',', '')
            try:
                arr_value = float(arr_str)
                # Convert millions to actual number
                if 'M' in match.group(0) or 'million' in match.group(0).lower():
                    arr_value = arr_value * 1000000
                
                form_data['annual_revenue'] = str(int(arr_value))
                
                # Set account tier based on ARR
                if arr_value >= 1000000:
                    form_data['account_tier'] = 'Enterprise'
                elif arr_value >= 100000:
                    form_data['account_tier'] = 'Key'
                else:
                    form_data['account_tier'] = 'Growth'
                break
            except ValueError:
                pass
    
    # Extract growth target
    growth_patterns = [
        r'(\d+(?:\.\d+)?)\%?\s*growth',
        r'growth.*?(\d+(?:\.\d+)?)\%?',
        r'target.*?(\d+(?:\.\d+)?)\%'
    ]
    
    for pattern in growth_patterns:
        match = re.search(pattern, message_clean, re.IGNORECASE)
        if match:
            try:
                growth = int(float(match.group(1)))
                form_data['revenue_growth_target'] = growth
                break
            except ValueError:
                pass
    
    # Extract quarter
    quarter_match = re.search(r'(Q[1-4])\s*(FY\d{2,4})?', message_clean, re.IGNORECASE)
    if quarter_match:
        form_data['planning_period'] = quarter_match.group(1).upper()
    
    # Extract region
    region_patterns = [
        r'(APAC|EMEA|Americas?|North America|Europe|Asia)',
        r'region[:\s]*([A-Za-z\s]+)'
    ]
    
    for pattern in region_patterns:
        match = re.search(pattern, message_clean, re.IGNORECASE)
        if match:
            form_data['region_territory'] = match.group(1).strip()
            break
    
    # Extract focus areas
    focus_patterns = [
        r'focus on ([^,\n.]+)',
        r'expansion[:\s]*([^,\n.]+)',
        r'cloud ([^,\n.]+)'
    ]
    
    focus_areas = []
    for pattern in focus_patterns:
        match = re.search(pattern, message_clean, re.IGNORECASE)
        if match:
            focus_areas.append(match.group(1).strip())
    
    if focus_areas:
        form_data['short_term_goals'] = f"Focus on {', '.join(focus_areas)}"
        form_data['key_opportunities'] = f"{', '.join(focus_areas)} expansion and growth opportunities"
    
    # NRR target
    nrr_match = re.search(r'NRR\s*[>]?\s*(\d+)\%?', message_clean, re.IGNORECASE)
    if nrr_match:
        form_data['customer_success_metrics'] = f"Maintain NRR > {nrr_match.group(1)}%"
    
    # Seasonal factor
    seasonal_match = re.search(r'(\d+(?:\.\d+)?)[x×]\s*seasonal', message_clean, re.IGNORECASE)
    if seasonal_match:
        factor = seasonal_match.group(1)
        form_data['planning_notes'] = f"Applied {factor}x seasonal adjustment factor"
    
    # Add smart defaults based on what we extracted
    if form_data.get('account_tier') == 'Enterprise':
        form_data['communication_cadence'] = 'Monthly strategic reviews with quarterly business reviews'
        form_data['stakeholders'] = [
            {"name": "CTO", "role": "Technical Decision Maker", "influence": "High", "relationship": "Positive"},
            {"name": "CFO", "role": "Budget Approver", "influence": "High", "relationship": "Neutral"},
            {"name": "VP Engineering", "role": "Technical Champion", "influence": "Medium", "relationship": "Strong"}
        ]
    elif form_data.get('account_tier') == 'Key':
        form_data['communication_cadence'] = 'Monthly check-ins with quarterly strategic reviews'
        form_data['stakeholders'] = [
            {"name": "VP Technology", "role": "Decision Maker", "influence": "High", "relationship": "Positive"},
            {"name": "Technical Lead", "role": "Champion", "influence": "Medium", "relationship": "Strong"}
        ]
    
    # Add renewal focus if mentioned
    if 'renewal' in message_clean.lower() or 'nrr' in message_clean.lower():
        form_data['long_term_goals'] = form_data.get('long_term_goals', '') + ' Optimize renewal rates and drive expansion revenue.'
        form_data['known_risks'] = 'Renewal timing, competitive pressure, and market conditions'
        form_data['risk_mitigation_strategies'] = 'Early renewal discussions, competitive analysis, and stakeholder engagement'
    
    # Add cloud specifics
    if 'cloud' in message_clean.lower():
        form_data['product_penetration_goals'] = 'Cloud platform adoption, migration acceleration, and usage optimization'
        form_data['long_term_goals'] = form_data.get('long_term_goals', '') + ' Drive cloud transformation and platform adoption.'
    
    # Add activities if quarter specified
    if form_data.get('planning_period'):
        quarter = form_data['planning_period']
        base_date = datetime.now()
        
        form_data['activities'] = [
            {
                "date": (base_date + timedelta(days=7)).strftime("%Y-%m-%d"),
                "title": f"{quarter} Planning Kickoff Meeting",
                "description": f"Initial {quarter} strategic planning session with key stakeholders"
            },
            {
                "date": (base_date + timedelta(days=30)).strftime("%Y-%m-%d"),
                "title": f"{quarter} Mid-Period Review",
                "description": f"Progress assessment and strategy adjustments for {quarter}"
            },
            {
                "date": (base_date + timedelta(days=60)).strftime("%Y-%m-%d"),
                "title": f"{quarter} Business Review",
                "description": f"Quarterly business review and performance analysis"
            }
        ]
    
    # Default customer since date
    if not form_data.get('customer_since'):
        form_data['customer_since'] = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    # Default account owner
    form_data['account_owner'] = 'Strategic Account Manager'
    
    # Set industry based on account name
    if account_name and account_name.lower() in ['microsoft', 'apple', 'google', 'amazon', 'salesforce']:
        form_data['industry'] = 'Technology'
    elif account_name and account_name.lower() in ['jpmorgan', 'goldman', 'morgan stanley', 'citi']:
        form_data['industry'] = 'Financial Services'
    else:
        form_data['industry'] = 'Enterprise'
    
    # Ensure we have a plan name
    if not form_data.get('plan_name'):
        form_data['plan_name'] = f"Strategic Account Plan - {datetime.now().strftime('%Y Q%m')}"
    
    return form_data
