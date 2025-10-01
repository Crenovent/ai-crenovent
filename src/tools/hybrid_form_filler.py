"""
Hybrid Form Filler - Combines instant pattern matching with database lookups
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import dynamic NLP extractor
def _get_dynamic_nlp_extractor():
    """Lazy import to avoid circular dependencies"""
    try:
        from .dynamic_nlp_extractor import get_dynamic_nlp_extractor
        return get_dynamic_nlp_extractor
    except ImportError:
        return None

class HybridFormFiller:
    """
    Hybrid form filler that combines instant pattern matching with database data
    Falls back gracefully if database is unavailable
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
    
    async def extract_form_data(self, message: str, user_context: Dict) -> Dict[str, Any]:
        """Extract form data using dynamic hybrid approach with LLM intelligence"""
        try:
            self.logger.info(f"🎯 Processing: '{message[:100]}...'")
            
            # Step 1: Try dynamic NLP extraction using LLM (most intelligent)
            form_data = await self._try_dynamic_nlp_extraction(message, user_context)
            
            # Step 2: If NLP didn't extract enough, add pattern matching
            if len(form_data) < 3:  # If we got minimal data, enhance with patterns
                pattern_data = self._extract_from_message(message)
                # Merge pattern data without overwriting NLP data
                for key, value in pattern_data.items():
                    if key not in form_data and value:
                        form_data[key] = value
                self.logger.info(f"🔄 Enhanced NLP data with {len(pattern_data)} pattern fields")
            else:
                self.logger.info(f"✅ Using rich NLP extraction ({len(form_data)} fields)")
            
            # Step 3: Try to enhance with database data (optional) - skip if conflicts
            if False:  # Temporarily disable database for robustness
                try:
                    db_data = await self._safe_fetch_db_data(user_context)
                    if db_data:
                        form_data = self._merge_database_data(form_data, db_data)
                        self.logger.info(f"✅ Enhanced form with database data")
                    else:
                        self.logger.info(f"📝 Using extracted data only")
                except Exception as e:
                    self.logger.warning(f"⚠️ Database enhancement failed, using extracted data: {e}")
            else:
                self.logger.info(f"📝 Using dynamic extraction (NLP + patterns)")
            
            # Add intelligent defaults
            self._add_smart_defaults(form_data, message, user_context)
            
            self.logger.info(f"🎯 Generated {len(form_data)} form fields")
            return form_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid form extraction: {e}")
            return {"error": f"Extraction failed: {str(e)}"}

    async def _try_dynamic_nlp_extraction(self, message: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Try dynamic NLP extraction using LLM for intelligent understanding"""
        try:
            nlp_extractor_func = _get_dynamic_nlp_extractor()
            if nlp_extractor_func and self.pool_manager:
                nlp_extractor = nlp_extractor_func(self.pool_manager)
                extracted_data = await nlp_extractor.extract_fields_dynamically(message, user_context)
                
                if extracted_data and len(extracted_data) > 0:
                    self.logger.info(f"🧠 Dynamic NLP extracted {len(extracted_data)} fields")
                    return extracted_data
                else:
                    self.logger.info("🔄 NLP extraction returned no data, will use pattern matching")
                    return {}
            else:
                self.logger.info("⚠️ NLP extractor not available, using pattern matching")
                return {}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Dynamic NLP extraction failed: {e}, falling back to patterns")
            return {}
    
    def _extract_from_message(self, message: str) -> Dict[str, Any]:
        """Extract basic data from message using patterns"""
        form_data = {}
        message_clean = message.strip()
        
        # Extract plan name (more specific patterns)
        plan_name_patterns = [
            r'plan named\s*["\']([^"\']+)["\']',
            r'plan called\s*["\']([^"\']+)["\']',
            r'create.*plan.*named\s*["\']?([A-Za-z0-9\s&\.]+?)["\']?(?:\s*,|\s*$)',
            r'plan for ([A-Za-z0-9\s&\.]+?)(?:\s+with|\s+account|\s*,)',
            r'account[:\s]+([A-Za-z0-9\s&\.]+?)(?:\s+with|\s*,)',
        ]
        
        for pattern in plan_name_patterns:
            match = re.search(pattern, message_clean, re.IGNORECASE)
            if match:
                plan_name = match.group(1).strip()
                form_data['plan_name'] = f"Strategic Plan - {plan_name}"
                form_data['account_id'] = plan_name
                break
        
        # Extract revenue (including thousands)
        revenue_patterns = [
            r'annual\s+revenue\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*(thousand|k|K)',
            r'revenue\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*(thousand|k|K)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(thousand|k|K)',
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:M|million)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:M|million)\s*ARR',
            r'ARR.*?\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:M|million|thousand|k|K)?'
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, message_clean, re.IGNORECASE)
            if match:
                try:
                    arr_str = match.group(1).replace(',', '')
                    arr_value = float(arr_str)
                    
                    # Handle different units
                    if len(match.groups()) > 1 and match.group(2):
                        unit = match.group(2).lower()
                        if unit in ['thousand', 'k']:
                            arr_value = arr_value * 1000
                        elif unit in ['million', 'm']:
                            arr_value = arr_value * 1000000
                    elif 'M' in match.group(0) or 'million' in match.group(0).lower():
                        arr_value = arr_value * 1000000
                    
                    form_data['annual_revenue'] = str(int(arr_value))
                    form_data['account_tier'] = 'Enterprise' if arr_value >= 1000000 else 'Key' if arr_value >= 100000 else 'Growth'
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract quarter
        quarter_match = re.search(r'(Q[1-4])\s*(FY\d{2,4})?', message_clean, re.IGNORECASE)
        if quarter_match:
            form_data['planning_period'] = quarter_match.group(1).upper()
        
        # Extract region
        region_patterns = [
            r'region\s+([A-Za-z\s]+?)(?:\s*,|\s+and|\s*$)',
            r'territory\s+([A-Za-z\s]+?)(?:\s*,|\s+and|\s*$)',
            r',\s*region\s+([A-Za-z\s]+?)(?:\s*,|\s+and|\s*$)',
        ]
        
        for pattern in region_patterns:
            match = re.search(pattern, message_clean, re.IGNORECASE)
            if match:
                region = match.group(1).strip()
                form_data['region_territory'] = region
                break

        # Extract customer since date
        customer_since_patterns = [
            r'customer since\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})',
            r'client since\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})',
            r'since\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})',
        ]
        
        month_names = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        for pattern in customer_since_patterns:
            match = re.search(pattern, message_clean, re.IGNORECASE)
            if match:
                day = match.group(1).zfill(2)
                month_name = match.group(2).lower()
                year = match.group(3)
                
                if month_name in month_names:
                    month = month_names[month_name]
                    form_data['customer_since'] = f"{year}-{month}-{day}"
                    break

        # Extract short term goals
        short_term_patterns = [
            r'short term goal is to\s+(.+?)(?:\s*,|\s*\.|$)',
            r'short term goal:\s*(.+?)(?:\s*,|\s*\.|$)',
            r'short goal is\s+(.+?)(?:\s*,|\s*\.|$)',
        ]
        
        for pattern in short_term_patterns:
            match = re.search(pattern, message_clean, re.IGNORECASE)
            if match:
                goal = match.group(1).strip()
                form_data['short_term_goals'] = goal
                break

        # Extract long term goals
        long_term_patterns = [
            r'long term goal is to\s+(.+?)(?:\s*,|\s*\.|$)',
            r'long term goal:\s*(.+?)(?:\s*,|\s*\.|$)',
            r'long goal is\s+(.+?)(?:\s*,|\s*\.|$)',
        ]
        
        for pattern in long_term_patterns:
            match = re.search(pattern, message_clean, re.IGNORECASE)
            if match:
                goal = match.group(1).strip()
                form_data['long_term_goals'] = goal
                break

        # Extract activities
        activity_patterns = [
            r'(\w+[\w\s]*meeting)\s+on\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})',
            r'schedule\s+(.+?)\s+on\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})',
            r'plan\s+(.+?)\s+on\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})',
        ]
        
        activities = []
        for pattern in activity_patterns:
            matches = re.finditer(pattern, message_clean, re.IGNORECASE)
            for match in matches:
                activity_title = match.group(1).strip()
                day = match.group(2).zfill(2)
                month_name = match.group(3).lower()
                year = match.group(4)
                
                if month_name in month_names:
                    month = month_names[month_name]
                    activity_date = f"{year}-{month}-{day}"
                    activities.append({
                        "title": activity_title.title(),
                        "date": activity_date,
                        "description": f"Scheduled {activity_title.lower()}"
                    })
        
        if activities:
            form_data['planned_activities'] = activities

        # Extract stakeholders
        stakeholder_patterns = [
            r'stakeholder\s+([^,]+?)\s+role\s+([^,]+?)(?:\s+influence\s+(high|medium|low))?',
            r'add stakeholder\s+([^,]+?)\s+as\s+([^,]+?)(?:\s+with\s+(high|medium|low)\s+influence)?',
        ]
        
        stakeholders = []
        for pattern in stakeholder_patterns:
            matches = re.finditer(pattern, message_clean, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                role = match.group(2).strip()
                influence = match.group(3).title() if match.group(3) else 'Medium'
                
                stakeholders.append({
                    "name": name.title(),
                    "role": role.title(),
                    "influence": influence,
                    "relationship": "Neutral"
                })
        
        if stakeholders:
            form_data['stakeholders'] = stakeholders
        
        # Check for "super important" or similar emphasis
        if any(word in message_clean.lower() for word in ['super important', 'very important', 'critical', 'urgent']):
            form_data['_high_priority'] = True  # Flag for enhanced filling
        
        return form_data
    
    async def _safe_fetch_db_data(self, user_context: Dict) -> Optional[Dict]:
        """Safely fetch database data with timeout"""
        try:
            user_id = int(user_context.get('user_id', 1319))
            tenant_id = int(user_context.get('tenant_id', 1300))
            
            # Quick timeout to avoid blocking
            conn = await asyncio.wait_for(
                self.pool_manager.postgres_pool.acquire(), 
                timeout=2.0
            )
            
            try:
                # Get user's most recent plan
                recent_plan = await conn.fetchrow("""
                    SELECT plan_name, account_id, annual_revenue, account_tier,
                           short_term_goals, long_term_goals, key_opportunities,
                           known_risks, communication_cadence, region_territory,
                           customer_success_metrics, risk_mitigation_strategies
                    FROM strategic_account_plans 
                    WHERE created_by_user_id = $1 AND tenant_id = $2
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, user_id, tenant_id)
                
                if recent_plan:
                    return dict(recent_plan)
                else:
                    return None
                    
            finally:
                await self.pool_manager.postgres_pool.release(conn)
                        
        except asyncio.TimeoutError:
            self.logger.warning("⚠️ Database query timeout")
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Database query failed: {e}")
            return None
    
    def _merge_database_data(self, form_data: Dict, db_data: Dict) -> Dict[str, Any]:
        """Merge database data with extracted form data"""
        # Only add database fields if not already extracted from message
        if not form_data.get('account_tier') and db_data.get('account_tier'):
            form_data['account_tier'] = db_data['account_tier']
            
        if not form_data.get('region_territory') and db_data.get('region_territory'):
            form_data['region_territory'] = db_data['region_territory']
            
        if not form_data.get('communication_cadence') and db_data.get('communication_cadence'):
            form_data['communication_cadence'] = db_data['communication_cadence']
        
        # Always add these from historical data if available
        if db_data.get('short_term_goals'):
            form_data['short_term_goals'] = db_data['short_term_goals']
            
        if db_data.get('long_term_goals'):
            form_data['long_term_goals'] = db_data['long_term_goals']
            
        if db_data.get('key_opportunities'):
            form_data['key_opportunities'] = db_data['key_opportunities']
            
        if db_data.get('known_risks'):
            form_data['known_risks'] = db_data['known_risks']
            
        if db_data.get('risk_mitigation_strategies'):
            form_data['risk_mitigation_strategies'] = db_data['risk_mitigation_strategies']
            
        if db_data.get('customer_success_metrics'):
            form_data['customer_success_metrics'] = db_data['customer_success_metrics']
        
        return form_data
    
    def _add_smart_defaults(self, form_data: Dict, message: str, user_context: Dict):
        """Add intelligent defaults"""
        # Ensure basic fields
        if not form_data.get('plan_name'):
            form_data['plan_name'] = f"Strategic Account Plan - {datetime.now().strftime('%Y Q%m')}"
            
        if not form_data.get('customer_since'):
            form_data['customer_since'] = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
        if not form_data.get('account_owner'):
            form_data['account_owner'] = 'Strategic Account Manager'
        
        # Industry based on account name
        account_name = form_data.get('account_id', '').lower()
        if not form_data.get('industry') and account_name:
            if any(tech in account_name for tech in ['microsoft', 'apple', 'google', 'amazon', 'salesforce']):
                form_data['industry'] = 'Technology'
            elif any(finance in account_name for finance in ['jpmorgan', 'goldman', 'citi', 'bank']):
                form_data['industry'] = 'Financial Services'
            else:
                form_data['industry'] = 'Enterprise'
        
        # Communication cadence based on tier
        if not form_data.get('communication_cadence'):
            tier = form_data.get('account_tier', '').lower()
            if tier == 'enterprise':
                form_data['communication_cadence'] = 'Monthly strategic reviews with quarterly business reviews'
            elif tier == 'key':
                form_data['communication_cadence'] = 'Quarterly strategic reviews with monthly check-ins'
            else:
                form_data['communication_cadence'] = 'Quarterly reviews'
        
        # Enhanced filling for high priority requests
        is_high_priority = form_data.pop('_high_priority', False)
        
        # Add defaults for common fields if missing
        if not form_data.get('short_term_goals'):
            if is_high_priority:
                form_data['short_term_goals'] = 'CRITICAL: Accelerate platform adoption, drive immediate user engagement, and establish strong product-market fit within 90 days'
            else:
                form_data['short_term_goals'] = 'Increase platform adoption and drive user engagement'
            
        if not form_data.get('long_term_goals'):
            if is_high_priority:
                form_data['long_term_goals'] = 'STRATEGIC PRIORITY: Establish long-term strategic partnership, maximize account lifetime value, and position as flagship customer within 12 months'
            else:
                form_data['long_term_goals'] = 'Establish strategic partnership and maximize account value'
            
        if not form_data.get('key_opportunities'):
            if is_high_priority:
                form_data['key_opportunities'] = 'HIGH-VALUE OPPORTUNITIES: Platform expansion across all business units, advanced feature adoption, deep system integration, and potential case study development'
            else:
                form_data['key_opportunities'] = 'Platform expansion, feature adoption, and integration opportunities'
            
        if not form_data.get('known_risks'):
            if is_high_priority:
                form_data['known_risks'] = 'CRITICAL RISKS: Competitive pressure from incumbent solutions, budget allocation challenges, change management resistance, and potential technology adoption delays'
            else:
                form_data['known_risks'] = 'Competitive pressure, budget constraints, technology adoption challenges'
            
        if not form_data.get('risk_mitigation_strategies'):
            if is_high_priority:
                form_data['risk_mitigation_strategies'] = 'PRIORITY MITIGATION: Weekly stakeholder engagement, competitive analysis with differentiation messaging, dedicated customer success support, and executive sponsorship program'
            else:
                form_data['risk_mitigation_strategies'] = 'Regular stakeholder engagement, competitive analysis, and proactive support'
        
        # Add more fields for high priority
        if is_high_priority:
            if not form_data.get('customer_success_metrics'):
                form_data['customer_success_metrics'] = 'CRITICAL METRICS: User adoption rate >80%, feature utilization >60%, customer satisfaction score >4.5/5, and renewal probability >95%'
                
            if not form_data.get('product_penetration_goals'):
                form_data['product_penetration_goals'] = 'STRATEGIC PENETRATION: Full platform deployment across all departments, advanced analytics adoption, and integration with core business systems'
                
            if not form_data.get('cross_sell_upsell_potential'):
                form_data['cross_sell_upsell_potential'] = 'HIGH POTENTIAL: Premium features upgrade, additional user licenses, professional services engagement, and enterprise add-on modules'

# Global instance
hybrid_form_filler = None

def get_hybrid_form_filler(pool_manager=None):
    """Get or create hybrid form filler instance"""
    global hybrid_form_filler
    if hybrid_form_filler is None:
        hybrid_form_filler = HybridFormFiller(pool_manager)
    elif pool_manager and hybrid_form_filler.pool_manager is None:
        hybrid_form_filler.pool_manager = pool_manager
    return hybrid_form_filler
