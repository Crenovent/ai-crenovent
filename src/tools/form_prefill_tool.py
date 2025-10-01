"""
Form Prefill Tool for Strategic Account Planning
Extracts specific values from user messages and creates form data
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FormPrefillTool:
    """
    Intelligent form prefill tool that extracts structured data from natural language
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns for extracting form data
        self.patterns = {
            'account_name': [
                r'plan for ([A-Za-z\s&\.]+?)(?:\s+with|\s+account|\s*$)',
                r'account[:\s]+([A-Za-z\s&\.]+?)(?:\s+with|\s*$)',
                r'create.*plan.*for[:\s]+([A-Za-z\s&\.]+?)(?:\s+with|\s*$)'
            ],
            'arr_target': [
                r'(?:\$|USD\s*)?([\d,\.]+)(?:K|k|M|m)?\s*ARR',
                r'ARR.*?(?:\$|USD\s*)?([\d,\.]+)(?:K|k|M|m)?',
                r'revenue.*?(?:\$|USD\s*)?([\d,\.]+)(?:K|k|M|m)?'
            ],
            'growth_target': [
                r'(\d+(?:\.\d+)?)\%?\s*growth',
                r'growth.*?(\d+(?:\.\d+)?)\%?',
                r'increase.*?(\d+(?:\.\d+)?)\%?'
            ],
            'quarter': [
                r'(Q[1-4])\s*(FY\d{2,4})?',
                r'(quarter\s*[1-4])',
                r'(Q[1-4])'
            ],
            'focus_area': [
                r'focus on ([^,\n]+)',
                r'expansion[:\s]*([^,\n]+)',
                r'strategy[:\s]*([^,\n]+)'
            ],
            'nrr_target': [
                r'NRR\s*[>]?\s*(\d+)\%?',
                r'Net Revenue Retention.*?(\d+)\%?',
                r'retention.*?(\d+)\%?'
            ],
            'seasonal_factor': [
                r'(\d+(?:\.\d+)?)[x×]\s*seasonal',
                r'seasonal.*?(\d+(?:\.\d+)?)[x×]?',
                r'factor.*?(\d+(?:\.\d+)?)'
            ],
            'region': [
                r'(APAC|EMEA|Americas?|North America|Europe|Asia)',
                r'region[:\s]*([A-Za-z\s]+)',
                r'territory[:\s]*([A-Za-z\s]+)'
            ],
            'segment': [
                r'(Enterprise|Mid-Market|SMB|Corporate)',
                r'segment[:\s]*([A-Za-z\s-]+)',
                r'tier[:\s]*([A-Za-z\s-]+)'
            ]
        }
    
    def extract_form_data(self, message: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract form prefill data from user message
        """
        try:
            form_data = {}
            message = message.strip()
            
            # Extract account name
            account_name = self._extract_pattern(message, 'account_name')
            if account_name:
                form_data['plan_name'] = f"Strategic Plan - {account_name} - {datetime.now().strftime('%Y Q%m')}"
                form_data['account_id'] = account_name
            
            # Extract ARR target
            arr_target = self._extract_pattern(message, 'arr_target')
            if arr_target:
                # Convert to standard format
                arr_value = self._normalize_currency(arr_target)
                if arr_value:
                    form_data['annual_revenue'] = str(int(arr_value))
                    
                    # Determine account tier based on ARR
                    if arr_value >= 1000000:  # 1M+
                        form_data['account_tier'] = 'Enterprise'
                    elif arr_value >= 100000:  # 100K+
                        form_data['account_tier'] = 'Key'
                    else:
                        form_data['account_tier'] = 'Growth'
            
            # Extract growth target
            growth_target = self._extract_pattern(message, 'growth_target')
            if growth_target:
                form_data['revenue_growth_target'] = int(float(growth_target))
            
            # Extract quarter information
            quarter = self._extract_pattern(message, 'quarter')
            if quarter:
                form_data['planning_period'] = quarter.upper()
            
            # Extract focus area
            focus_area = self._extract_pattern(message, 'focus_area')
            if focus_area:
                form_data['short_term_goals'] = f"Focus on {focus_area.strip()}"
                form_data['key_opportunities'] = f"{focus_area.strip()} expansion opportunities"
            
            # Extract NRR target
            nrr_target = self._extract_pattern(message, 'nrr_target')
            if nrr_target:
                form_data['customer_success_metrics'] = f"Maintain NRR > {nrr_target}%"
            
            # Extract seasonal factor
            seasonal_factor = self._extract_pattern(message, 'seasonal_factor')
            if seasonal_factor:
                factor = float(seasonal_factor)
                form_data['planning_notes'] = f"Applied {factor}x seasonal adjustment factor"
            
            # Extract region
            region = self._extract_pattern(message, 'region')
            if region:
                form_data['region_territory'] = region
            
            # Extract segment
            segment = self._extract_pattern(message, 'segment')
            if segment:
                form_data['account_tier'] = segment
            
            # Add intelligent defaults based on extracted data
            self._add_intelligent_defaults(form_data, message, user_context)
            
            self.logger.info(f"✅ Extracted {len(form_data)} form fields from message")
            return form_data
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting form data: {e}")
            return {}
    
    def _extract_pattern(self, message: str, pattern_key: str) -> Optional[str]:
        """Extract value using pattern matching"""
        patterns = self.patterns.get(pattern_key, [])
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _normalize_currency(self, value_str: str) -> Optional[float]:
        """Convert currency string to numeric value"""
        try:
            # Remove common currency symbols and spaces
            cleaned = re.sub(r'[,$\s]', '', value_str)
            
            # Handle K/M suffixes
            if cleaned.upper().endswith('K'):
                return float(cleaned[:-1]) * 1000
            elif cleaned.upper().endswith('M'):
                return float(cleaned[:-1]) * 1000000
            else:
                return float(cleaned)
                
        except (ValueError, AttributeError):
            return None
    
    def _add_intelligent_defaults(self, form_data: Dict, message: str, user_context: Optional[Dict]):
        """Add intelligent defaults based on context"""
        message_lower = message.lower()
        
        # Default communication cadence based on tier
        if form_data.get('account_tier') == 'Enterprise':
            form_data['communication_cadence'] = 'Monthly strategic reviews with quarterly business reviews'
        elif form_data.get('account_tier') == 'Key':
            form_data['communication_cadence'] = 'Monthly check-ins with quarterly reviews'
        else:
            form_data['communication_cadence'] = 'Quarterly strategic reviews'
        
        # Add renewal focus if mentioned
        if 'renewal' in message_lower or 'nrr' in message_lower:
            form_data['long_term_goals'] = form_data.get('long_term_goals', '') + ' Optimize renewal rates and expansion opportunities.'
            form_data['known_risks'] = 'Renewal timing and competitive pressure'
        
        # Add cloud expansion specifics
        if 'cloud' in message_lower:
            form_data['product_penetration_goals'] = 'Cloud platform adoption and migration'
            form_data['success_metrics'] = ['Cloud adoption rate', 'Migration milestones', 'Usage growth']
        
        # Add default stakeholders based on tier
        if form_data.get('account_tier') == 'Enterprise':
            form_data['stakeholders'] = [
                {"name": "CTO", "role": "Technical Decision Maker", "influence": "High", "relationship": "Positive"},
                {"name": "CFO", "role": "Budget Approver", "influence": "High", "relationship": "Neutral"},
                {"name": "VP Engineering", "role": "Technical Champion", "influence": "Medium", "relationship": "Strong"}
            ]
        else:
            form_data['stakeholders'] = [
                {"name": "VP Technology", "role": "Decision Maker", "influence": "High", "relationship": "Positive"},
                {"name": "Technical Lead", "role": "Champion", "influence": "Medium", "relationship": "Strong"}
            ]
        
        # Add activities based on quarter
        if form_data.get('planning_period'):
            quarter = form_data['planning_period']
            base_date = datetime.now()
            
            form_data['activities'] = [
                {
                    "date": (base_date + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "title": f"{quarter} Planning Kickoff",
                    "description": f"Initial {quarter} strategic planning session"
                },
                {
                    "date": (base_date + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "title": f"{quarter} Progress Review",
                    "description": f"Mid-{quarter} progress assessment and adjustments"
                }
            ]
        
        # Add customer since date (default to 2 years ago for established accounts)
        if not form_data.get('customer_since'):
            form_data['customer_since'] = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        
        # Add risk mitigation
        if form_data.get('known_risks'):
            form_data['risk_mitigation_strategies'] = 'Regular stakeholder engagement, competitive analysis, and proactive issue resolution'

# Global instance
form_prefill_tool = FormPrefillTool()
