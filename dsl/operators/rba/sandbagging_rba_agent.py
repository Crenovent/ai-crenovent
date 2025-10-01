"""
Sandbagging RBA Agent
Single-purpose, focused RBA agent for sandbagging detection only

This agent ONLY handles sandbagging detection and nothing else.
Enhanced with universal parameter system and Knowledge Graph tracing.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent
from ...rules.business_rules_engine import BusinessRulesEngine

logger = logging.getLogger(__name__)

class SandbaggingRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for sandbagging detection
    
    Features:
    - ONLY handles sandbagging detection
    - Configuration-driven via YAML rules
    - Lightweight and focused
    - Industry-specific adjustments
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "sandbagging_detection"
    AGENT_DESCRIPTION = "Detect high-value deals with artificially low probability"
    SUPPORTED_ANALYSIS_TYPES = ["sandbagging_detection"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Initialize rules engine for sandbagging rules only
        self.rules_engine = BusinessRulesEngine()
        
        # Note: Default configuration now comes from universal parameter system
        # Agent-specific defaults are defined in universal_rba_parameters.yaml
    
    async def _validate_agent_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration for sandbagging detection"""
        errors = []
        
        # Validate sandbagging threshold
        threshold = config.get('sandbagging_threshold', 65)
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 100:
            errors.append("sandbagging_threshold must be a number between 0 and 100")
        
        # Validate high value threshold
        high_value = config.get('high_value_threshold', 250000)
        if not isinstance(high_value, (int, float)) or high_value <= 0:
            errors.append("high_value_threshold must be a positive number")
            
        # Validate probability thresholds
        low_prob = config.get('low_probability_threshold', 35)
        if not isinstance(low_prob, (int, float)) or low_prob < 0 or low_prob > 100:
            errors.append("low_probability_threshold must be between 0 and 100")
            
        advanced_prob = config.get('advanced_stage_probability_threshold', 40)
        if not isinstance(advanced_prob, (int, float)) or advanced_prob < 0 or advanced_prob > 100:
            errors.append("advanced_stage_probability_threshold must be between 0 and 100")
        
        # Validate close date range
        close_date_range = config.get('close_date_range_days', 90)
        if not isinstance(close_date_range, int) or close_date_range <= 0:
            errors.append("close_date_range_days must be a positive integer")
        
        return errors
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sandbagging detection analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            # Configuration is already validated and merged by enhanced base class
            config = input_data.get('config', {})
            
            logger.info(f"🎯 Sandbagging RBA: Analyzing {len(opportunities)} opportunities with universal parameters")
            logger.info(f"📋 Universal config values: high_value_threshold=${self.get_parameter_value(config, 'high_value_threshold', 250000):,}, "
                       f"sandbagging_threshold={self.get_parameter_value(config, 'sandbagging_threshold', 65)}%, "
                       f"low_prob_threshold={self.get_parameter_value(config, 'low_probability_threshold', 35)}%")
            logger.info(f"🎯 Total universal parameters available: {len(self.get_universal_parameters())}")
            
            # Use rules engine for sandbagging evaluation
            flagged_deals = []
            for opp in opportunities:
                if opp is None:
                    continue
                    
                opportunity_data = self._extract_opportunity_data(opp)
                industry = self._determine_industry(opp)
                
                # Direct sandbagging evaluation (more practical than rules engine)
                amount = self._safe_float(opp.get('Amount', 0))
                probability = self._safe_float(opp.get('Probability', 0))
                stage = opp.get('StageName', '').lower()
                
                # Calculate direct sandbagging score
                sandbagging_score = 0
                risk_factors = []
                
                # High value + low probability = primary indicator (universal parameters)
                high_value_threshold = self.get_parameter_value(config, 'high_value_threshold', 250000)
                low_prob_threshold = self.get_parameter_value(config, 'low_probability_threshold', 35)
                
                if amount >= high_value_threshold and probability <= low_prob_threshold:
                    sandbagging_score += 60
                    risk_factors.append(f"High value (${amount:,.0f}) with low probability ({probability}%)")
                
                # Advanced stage with low probability (universal parameters)
                advanced_stage_prob_threshold = self.get_parameter_value(config, 'proposal_stage_days', 40)
                advanced_stages = ['negotiation', 'proposal', 'contract', 'closing', 'quote']
                if any(stage_word in stage for stage_word in advanced_stages) and probability <= advanced_stage_prob_threshold:
                    sandbagging_score += 30
                    risk_factors.append(f"Advanced stage ({stage}) with low probability (<={advanced_stage_prob_threshold}%)")
                
                # Very high value deals with moderate probability (universal parameters)
                mega_deal_threshold = self.get_parameter_value(config, 'mega_deal_threshold', 1000000)
                mega_deal_prob_threshold = self.get_parameter_value(config, 'high_probability_threshold', 50)
                if amount >= mega_deal_threshold and probability <= mega_deal_prob_threshold:
                    sandbagging_score += 20
                    risk_factors.append(f"Mega deal (${amount:,.0f}) with moderate probability (<={mega_deal_prob_threshold}%)")
                
                # Check if meets flagging criteria (universal parameters)
                sandbagging_threshold = self.get_parameter_value(config, 'sandbagging_threshold', 65)
                
                if sandbagging_score >= sandbagging_threshold:
                    
                    logger.info(f"⚠️ SANDBAGGING DETECTED: {opp.get('Name', 'Unknown')} - Score: {sandbagging_score}")
                    
                    # Determine risk level based on score
                    if sandbagging_score >= 80:
                        risk_level = "CRITICAL"
                    elif sandbagging_score >= 60:
                        risk_level = "HIGH"
                    elif sandbagging_score >= 40:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "LOW"
                    
                    flagged_deals.append({
                        'opportunity_id': opp.get('Id'),
                        'opportunity_name': opp.get('Name'),
                        'amount': amount,
                        'probability': probability,
                        'stage_name': opp.get('StageName', ''),
                        'account_name': opp.get('Account', {}).get('Name', ''),
                        'owner_name': opp.get('Owner', {}).get('Name', ''),
                        'close_date': opp.get('CloseDate', ''),
                        'sandbagging_score': sandbagging_score,
                        'risk_level': risk_level,
                        'confidence': min(100, sandbagging_score + 20),  # Practical confidence
                        'risk_factors': risk_factors,
                        'issue_type': 'SANDBAGGING',
                        'flags': risk_factors
                    })
            
            # Calculate comprehensive analytics
            total_opps = len(opportunities)
            flagged_count = len(flagged_deals)
            
            # Analyze all deals for insights
            high_value_deals = [opp for opp in opportunities if self._safe_float(opp.get('Amount', 0)) >= config.get('high_value_threshold', 250000)]
            low_prob_deals = [opp for opp in opportunities if self._safe_float(opp.get('Probability', 0)) <= config.get('low_probability_threshold', 35)]
            
            # Risk distribution
            all_amounts = [self._safe_float(opp.get('Amount', 0)) for opp in opportunities]
            all_probs = [self._safe_float(opp.get('Probability', 0)) for opp in opportunities]
            
            # Top deals at risk (even if not flagged) - now fully configurable
            potential_risk_deals = []
            high_value_threshold = config.get('high_value_threshold', 250000)
            low_prob_threshold = config.get('low_probability_threshold', 35)
            mega_deal_threshold = config.get('mega_deal_threshold', 1000000)
            
            for opp in opportunities:
                amount = self._safe_float(opp.get('Amount', 0))
                prob = self._safe_float(opp.get('Probability', 0))
                if amount >= 100000:  # Any high-value deal (could make this configurable too)
                    risk_score = 0
                    # Use configured thresholds for risk scoring
                    if amount >= high_value_threshold and prob <= low_prob_threshold:
                        risk_score = 80
                    elif amount >= (high_value_threshold * 2) and prob <= 50:  # Double high-value threshold
                        risk_score = 60
                    elif amount >= mega_deal_threshold and prob <= 60:
                        risk_score = 40
                    
                    if risk_score > 0:
                        potential_risk_deals.append({
                            'name': opp.get('Name', 'Unknown'),
                            'amount': amount,
                            'probability': prob,
                            'stage': opp.get('StageName', ''),
                            'risk_score': risk_score,
                            'owner': opp.get('Owner', {}).get('Name', 'Unknown')
                        })
            
            # Sort by risk score and amount
            potential_risk_deals.sort(key=lambda x: (x['risk_score'], x['amount']), reverse=True)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'sandbagging_detection',
                'total_opportunities': total_opps,
                'flagged_opportunities': flagged_count,
                'flagged_deals': flagged_deals,
                
                # Rich Analytics
                'summary_metrics': {
                    'sandbagging_rate': round((flagged_count / total_opps * 100), 2) if total_opps > 0 else 0,
                    'high_value_deals_count': len(high_value_deals),
                    'high_value_deals_percentage': round((len(high_value_deals) / total_opps * 100), 1) if total_opps > 0 else 0,
                    'low_probability_deals_count': len(low_prob_deals),
                    'low_probability_deals_percentage': round((len(low_prob_deals) / total_opps * 100), 1) if total_opps > 0 else 0,
                    'potential_risk_deals_count': len(potential_risk_deals),
                    'average_deal_amount': round(sum(all_amounts) / len(all_amounts), 0) if all_amounts else 0,
                    'average_probability': round(sum(all_probs) / len(all_probs), 1) if all_probs else 0,
                    'total_pipeline_value': sum(all_amounts),
                    # FIXED: Value at risk should be sum of FLAGGED sandbagging deals, not potential deals
                    'at_risk_pipeline_value': sum([deal['amount'] for deal in flagged_deals])
                },
                
                # Top deals requiring attention
                'top_risk_deals': potential_risk_deals[:10],  # Top 10 deals at risk
                
                # Distribution insights
                'risk_distribution': {
                    'critical_risk': len([d for d in potential_risk_deals if d['risk_score'] >= 80]),
                    'high_risk': len([d for d in potential_risk_deals if 60 <= d['risk_score'] < 80]),
                    'medium_risk': len([d for d in potential_risk_deals if 40 <= d['risk_score'] < 60]),
                    'low_risk': len([d for d in potential_risk_deals if d['risk_score'] < 40])
                },
                
                # Actionable insights
                'insights': self._generate_sandbagging_insights(flagged_count, total_opps, flagged_deals, high_value_deals),
                
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Sandbagging RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _extract_opportunity_data(self, opp: Dict[str, Any]) -> Dict[str, Any]:
        """Extract opportunity data for sandbagging analysis"""
        return {
            'Id': opp.get('Id'),
            'Amount': self._safe_float(opp.get('Amount')),
            'Probability': self._safe_float(opp.get('Probability')),
            'StageName': opp.get('StageName', '').lower(),
            'CloseDate': opp.get('CloseDate'),
            'days_to_close': self._calculate_days_to_close(opp.get('CloseDate')),
            'days_since_activity': self._calculate_days_since_activity(opp.get('ActivityDate'))
        }
    
    def _determine_industry(self, opp: Dict[str, Any]) -> str:
        """Determine industry for adjustments"""
        return opp.get('Account', {}).get('Industry', 'Unknown')
    
    def _calculate_days_to_close(self, close_date_str: str) -> int:
        """Calculate days to close"""
        if not close_date_str:
            return 999
        try:
            close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
            return (close_date - datetime.now()).days
        except:
            return 999
    
    def _generate_sandbagging_insights(self, flagged_count: int, total_opps: int, flagged_deals: list, high_value_deals: list) -> list:
        """Generate actionable insights for sandbagging analysis"""
        insights = []
        
        if flagged_count > 0:
            insights.append(f"🚨 {flagged_count} deals confirmed as sandbagging cases requiring immediate review")
            
            # Calculate total value of flagged deals
            total_flagged_value = sum([deal['amount'] for deal in flagged_deals])
            insights.append(f"💰 ${total_flagged_value:,.0f} total value in flagged sandbagging deals")
            
            # Risk level breakdown of flagged deals
            critical_deals = [d for d in flagged_deals if d.get('risk_level') == 'CRITICAL']
            high_risk_deals = [d for d in flagged_deals if d.get('risk_level') == 'HIGH']
            
            if critical_deals:
                critical_value = sum([d['amount'] for d in critical_deals])
                insights.append(f"🔥 {len(critical_deals)} critical risk deals worth ${critical_value:,.0f}")
            
            if high_risk_deals:
                high_risk_value = sum([d['amount'] for d in high_risk_deals])  
                insights.append(f"⚠️ {len(high_risk_deals)} high risk deals worth ${high_risk_value:,.0f}")
        else:
            insights.append("✅ No sandbagging deals detected with current thresholds")
        
        if len(high_value_deals) > 0:
            high_value_pct = round(len(high_value_deals) / total_opps * 100, 1)
            insights.append(f"💎 {len(high_value_deals)} high-value deals ({high_value_pct}%) in pipeline")
        
        if flagged_count > 10:
            insights.append("📊 Consider pipeline review meeting to address probability alignment")
        
        if flagged_count == 0:
            insights.append("✅ No significant sandbagging patterns detected - healthy pipeline probability distribution")
        
        return insights
    
    def _calculate_days_since_activity(self, activity_date_str: str) -> int:
        """Calculate days since last activity"""
        if not activity_date_str:
            return 999
        try:
            activity_date = datetime.strptime(activity_date_str.split('T')[0], '%Y-%m-%d')
            return (datetime.now() - activity_date).days
        except:
            return 999
    
    @classmethod
    def get_agent_metadata(cls) -> Dict[str, Any]:
        """Get agent metadata for registry"""
        return {
            'agent_type': cls.AGENT_TYPE,
            'agent_name': cls.AGENT_NAME,
            'agent_description': cls.AGENT_DESCRIPTION,
            'supported_analysis_types': cls.SUPPORTED_ANALYSIS_TYPES,
            'class_name': cls.__name__,
            'module_path': cls.__module__
        }
