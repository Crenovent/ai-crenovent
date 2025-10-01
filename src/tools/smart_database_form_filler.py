"""
Smart Database-Driven Form Filler
Fetches user-specific historical data to intelligently prefill forms
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SmartDatabaseFormFiller:
    """
    Intelligent form filler that uses historical user data from PostgreSQL
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Patterns for extracting form data from messages
        self.patterns = {
            'account_name': [
                r'plan for ([A-Za-z0-9\s&\.]+?)(?:\s+with|\s+account|\s*$)',
                r'account[:\s]+([A-Za-z0-9\s&\.]+?)(?:\s+with|\s*$)',
                r'create.*plan.*for[:\s]+([A-Za-z0-9\s&\.]+?)(?:\s+with|\s*$)'
            ],
            'arr_target': [
                r'(?:\$|USD\s*)?([\d,\.]+)(?:K|k|M|m)?\s*ARR',
                r'ARR.*?(?:\$|USD\s*)?([\d,\.]+)(?:K|k|M|m)?',
                r'revenue.*?(?:\$|USD\s*)?([\d,\.]+)(?:K|k|M|m)?'
            ],
            'quarter': [
                r'(Q[1-4])\s*(FY\d{2,4})?',
                r'(quarter\s*[1-4])',
                r'(Q[1-4])'
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
    
    async def extract_form_data(self, message: str, user_context: Dict) -> Dict[str, Any]:
        """
        Extract form prefill data using message analysis + user historical data
        """
        try:
            user_id = user_context.get('user_id', '1319')
            tenant_id = user_context.get('tenant_id', '1300')
            
            # Convert to integers for database queries
            user_id_int = int(user_id) if isinstance(user_id, str) and user_id.isdigit() else int(user_id) if isinstance(user_id, int) else 1319
            tenant_id_int = int(tenant_id) if isinstance(tenant_id, str) and tenant_id.isdigit() else int(tenant_id) if isinstance(tenant_id, int) else 1300
            
            # Start with pattern extraction from message
            form_data = self._extract_from_message(message)
            
            # Enhance with user-specific historical data
            if self.pool_manager and self.pool_manager.postgres_pool:
                historical_data = await self._fetch_user_historical_data(user_id_int, tenant_id_int)
                form_data = self._merge_with_historical_data(form_data, historical_data, message)
            
            # Add intelligent defaults
            self._add_intelligent_defaults(form_data, message, user_context)
            
            self.logger.info(f"🎯 Generated smart form prefill with {len(form_data)} fields for user {user_id}")
            return form_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in smart form extraction: {e}")
            # Fallback to simple extraction
            return self._extract_from_message(message)
    
    def _extract_from_message(self, message: str) -> Dict[str, Any]:
        """Extract basic data from message using patterns"""
        form_data = {}
        message_clean = message.strip()
        
        # Extract account name
        account_name = self._extract_pattern(message_clean, 'account_name')
        if account_name:
            form_data['plan_name'] = f"Strategic Plan - {account_name} - {datetime.now().strftime('%Y Q%m')}"
            form_data['account_id'] = account_name
        
        # Extract ARR target
        arr_target = self._extract_pattern(message_clean, 'arr_target')
        if arr_target:
            arr_value = self._normalize_currency(arr_target)
            if arr_value:
                form_data['annual_revenue'] = str(int(arr_value))
                form_data['account_tier'] = 'Enterprise' if arr_value >= 1000000 else 'Key' if arr_value >= 100000 else 'Growth'
        
        # Extract quarter
        quarter = self._extract_pattern(message_clean, 'quarter')
        if quarter:
            form_data['planning_period'] = quarter.upper()
        
        # Extract region and segment
        region = self._extract_pattern(message_clean, 'region')
        if region:
            form_data['region_territory'] = region
            
        segment = self._extract_pattern(message_clean, 'segment')
        if segment:
            form_data['account_tier'] = segment
        
        return form_data
    
    async def _fetch_user_historical_data(self, user_id: int, tenant_id: int) -> Dict[str, Any]:
        """Fetch user-specific historical planning data"""
        try:
            # Use timeout and retry logic to handle connection issues
            conn = await asyncio.wait_for(
                self.pool_manager.postgres_pool.acquire(),
                timeout=3.0
            )
            
            try:
                # Get user's recent plans
                user_plans = await conn.fetch("""
                    SELECT plan_name, account_id, annual_revenue, account_tier,
                           short_term_goals, long_term_goals, key_opportunities,
                           known_risks, communication_cadence, region_territory,
                           customer_success_metrics, risk_mitigation_strategies,
                           product_penetration_goals, cross_sell_upsell_potential
                    FROM strategic_account_plans 
                    WHERE created_by_user_id = $1 AND tenant_id = $2
                    ORDER BY created_at DESC 
                    LIMIT 5
                """, user_id, tenant_id)
                
                # Get user profile and role info
                user_info = await conn.fetchrow("""
                    SELECT u.profile, ur.segment, ur.region, ur.role
                    FROM users u
                    LEFT JOIN users_role ur ON u.user_id = ur.user_id
                    WHERE u.user_id = $1 AND u.tenant_id = $2
                """, user_id, tenant_id)
                
                # Get common patterns for this user
                common_patterns = await conn.fetchrow("""
                    SELECT 
                        mode() WITHIN GROUP (ORDER BY account_tier) as common_tier,
                        mode() WITHIN GROUP (ORDER BY communication_cadence) as common_cadence,
                        AVG(annual_revenue) as avg_revenue
                    FROM strategic_account_plans 
                    WHERE created_by_user_id = $1 AND tenant_id = $2 
                      AND annual_revenue IS NOT NULL
                """, user_id, tenant_id)
                
                return {
                    'user_plans': [dict(plan) for plan in user_plans],
                    'user_info': dict(user_info) if user_info else {},
                    'patterns': dict(common_patterns) if common_patterns else {}
                }
                
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                
        except asyncio.TimeoutError:
            self.logger.warning("⚠️ Database query timeout - using fallback data")
            return {}
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical data: {e}")
            return {}
    
    def _merge_with_historical_data(self, form_data: Dict, historical_data: Dict, message: str) -> Dict[str, Any]:
        """Merge form data with user historical patterns"""
        user_plans = historical_data.get('user_plans', [])
        user_info = historical_data.get('user_info', {})
        patterns = historical_data.get('patterns', {})
        
        # Use user's segment/region if not extracted from message
        if not form_data.get('region_territory') and user_info.get('region'):
            form_data['region_territory'] = user_info['region']
            
        if not form_data.get('account_tier') and user_info.get('segment'):
            form_data['account_tier'] = user_info['segment']
        
        # Use historical patterns for missing fields
        if not form_data.get('account_tier') and patterns.get('common_tier'):
            form_data['account_tier'] = patterns['common_tier']
            
        if not form_data.get('communication_cadence') and patterns.get('common_cadence'):
            form_data['communication_cadence'] = patterns['common_cadence']
        
        # If no revenue specified, suggest based on historical average
        if not form_data.get('annual_revenue') and patterns.get('avg_revenue'):
            avg_revenue = int(patterns['avg_revenue'])
            form_data['suggested_annual_revenue'] = str(avg_revenue)
            form_data['annual_revenue'] = str(avg_revenue)
        
        # Extract common goals and strategies from recent plans
        if user_plans:
            recent_plan = user_plans[0]  # Most recent plan
            
            # Use recent goals as templates
            if recent_plan.get('short_term_goals') and 'goal' in message.lower():
                form_data['short_term_goals'] = recent_plan['short_term_goals']
                
            if recent_plan.get('long_term_goals'):
                form_data['long_term_goals'] = recent_plan['long_term_goals']
                
            if recent_plan.get('key_opportunities'):
                form_data['key_opportunities'] = recent_plan['key_opportunities']
                
            if recent_plan.get('known_risks'):
                form_data['known_risks'] = recent_plan['known_risks']
                
            if recent_plan.get('risk_mitigation_strategies'):
                form_data['risk_mitigation_strategies'] = recent_plan['risk_mitigation_strategies']
                
            if recent_plan.get('customer_success_metrics'):
                form_data['customer_success_metrics'] = recent_plan['customer_success_metrics']
            
            # Use product penetration goals and cross-sell potential
            if recent_plan.get('product_penetration_goals'):
                form_data['product_penetration_goals'] = recent_plan['product_penetration_goals']
                
            if recent_plan.get('cross_sell_upsell_potential'):
                form_data['cross_sell_upsell_potential'] = recent_plan['cross_sell_upsell_potential']
                
            # Use region from historical data
            if recent_plan.get('region_territory'):
                form_data['region_territory'] = recent_plan['region_territory']
        
        return form_data
    
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
            cleaned = re.sub(r'[,$\s]', '', value_str)
            
            if cleaned.upper().endswith('K'):
                return float(cleaned[:-1]) * 1000
            elif cleaned.upper().endswith('M'):
                return float(cleaned[:-1]) * 1000000
            else:
                return float(cleaned)
                
        except (ValueError, AttributeError):
            return None
    
    def _add_intelligent_defaults(self, form_data: Dict, message: str, user_context: Dict):
        """Add intelligent defaults based on context"""
        # Ensure we have a plan name
        if not form_data.get('plan_name'):
            form_data['plan_name'] = f"Strategic Account Plan - {datetime.now().strftime('%Y Q%m')}"
        
        # Add default customer since date
        if not form_data.get('customer_since'):
            form_data['customer_since'] = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Add default account owner
        if not form_data.get('account_owner'):
            form_data['account_owner'] = 'Strategic Account Manager'
        
        # Set industry based on account name
        account_name = form_data.get('account_id', '').lower()
        if account_name:
            if any(tech in account_name for tech in ['microsoft', 'apple', 'google', 'amazon', 'salesforce', 'oracle']):
                form_data['industry'] = 'Technology'
            elif any(finance in account_name for finance in ['jpmorgan', 'goldman', 'morgan stanley', 'citi', 'bank']):
                form_data['industry'] = 'Financial Services'
            else:
                form_data['industry'] = 'Enterprise'

# Global instance - will be initialized by the service
smart_form_filler = None

def get_smart_form_filler(pool_manager=None):
    """Get or create smart form filler instance"""
    global smart_form_filler
    if smart_form_filler is None:
        smart_form_filler = SmartDatabaseFormFiller(pool_manager)
    elif pool_manager and smart_form_filler.pool_manager is None:
        smart_form_filler.pool_manager = pool_manager
    return smart_form_filler
