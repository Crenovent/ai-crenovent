"""
Deals at Risk RBA Agent
Single-purpose, focused RBA agent for deals at risk detection only

This agent ONLY handles deals that are slipping or need attention.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class DealsAtRiskRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for deals at risk detection
    
    Features:
    - ONLY handles deals at risk identification
    - Configuration-driven risk factors
    - Lightweight and focused
    - Customer disengagement detection
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "deals_at_risk"
    AGENT_DESCRIPTION = "Identify deals that are slipping or need immediate attention"
    SUPPORTED_ANALYSIS_TYPES = ["deals_at_risk", "slipping_deals"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Deals at risk specific defaults
        self.default_config = {
            'engagement_threshold_days': 30,
            'stage_velocity_threshold': 60,
            'probability_decline_threshold': 20,
            'competitive_pressure_threshold': 2,
            'risk_score_threshold': 70
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deals at risk analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Deals at Risk RBA: Analyzing {len(opportunities)} opportunities")
            
            # Filter for open deals only
            open_deals = [
                opp for opp in opportunities 
                if 'closed' not in opp.get('StageName', '').lower()
            ]
            
            # Identify deals at risk
            deals_at_risk = []
            
            for opp in open_deals:
                risk_assessment = self._assess_deal_risk_factors(opp, config)
                
                if risk_assessment['total_risk_score'] >= config.get('risk_score_threshold', 70):
                    deals_at_risk.append(
                        self._create_at_risk_record(opp, risk_assessment, config)
                    )
            
            # Calculate summary metrics
            summary_metrics = self._calculate_at_risk_summary_metrics(deals_at_risk, open_deals, config)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'deals_at_risk',
                'total_opportunities': len(opportunities),
                'open_deals_analyzed': len(open_deals),
                'flagged_opportunities': len(deals_at_risk),
                'deals_at_risk': deals_at_risk,
                'summary_metrics': summary_metrics,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Deals at Risk RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _assess_deal_risk_factors(self, opp: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk factors for a deal"""
        
        risk_factors = {}
        total_risk_score = 0
        
        # Factor 1: Customer Disengagement (30 points)
        engagement_threshold = config.get('engagement_threshold_days', 30)
        days_since_activity = self._calculate_days_since_activity(opp)
        
        if days_since_activity >= engagement_threshold:
            engagement_score = min(30, int(days_since_activity / engagement_threshold * 30))
            risk_factors['customer_disengagement'] = {
                'score': engagement_score,
                'days_since_activity': days_since_activity,
                'threshold': engagement_threshold
            }
            total_risk_score += engagement_score
        
        # Factor 2: Stage Stagnation (25 points)
        velocity_threshold = config.get('stage_velocity_threshold', 60)
        days_in_stage = self._calculate_days_in_stage(opp)
        
        if days_in_stage >= velocity_threshold:
            stagnation_score = min(25, int(days_in_stage / velocity_threshold * 25))
            risk_factors['stage_stagnation'] = {
                'score': stagnation_score,
                'days_in_stage': days_in_stage,
                'threshold': velocity_threshold
            }
            total_risk_score += stagnation_score
        
        # Factor 3: Probability Decline (20 points)
        # Note: This would require historical probability data in a real implementation
        probability = self._safe_float(opp.get('Probability'))
        if probability < 50:  # Simplified check for low probability
            decline_score = int((50 - probability) / 50 * 20)
            risk_factors['probability_decline'] = {
                'score': decline_score,
                'current_probability': probability,
                'threshold': 50
            }
            total_risk_score += decline_score
        
        # Factor 4: Competitive Pressure (15 points)
        competitor_count = len(opp.get('Competitors', []))
        competitive_threshold = config.get('competitive_pressure_threshold', 2)
        
        if competitor_count > competitive_threshold:
            competitive_score = min(15, (competitor_count - competitive_threshold) * 5)
            risk_factors['competitive_pressure'] = {
                'score': competitive_score,
                'competitor_count': competitor_count,
                'threshold': competitive_threshold
            }
            total_risk_score += competitive_score
        
        # Factor 5: Budget Concerns (10 points)
        budget_status = opp.get('BudgetStatus', '').lower()
        if budget_status in ['uncertain', 'reduced', 'frozen', 'no_budget']:
            budget_score = 10
            risk_factors['budget_concerns'] = {
                'score': budget_score,
                'budget_status': budget_status
            }
            total_risk_score += budget_score
        
        return {
            'total_risk_score': total_risk_score,
            'risk_factors': risk_factors,
            'risk_factor_count': len(risk_factors)
        }
    
    def _calculate_days_since_activity(self, opp: Dict[str, Any]) -> int:
        """Calculate days since last activity"""
        
        activity_date_str = opp.get('ActivityDate') or opp.get('LastActivityDate')
        
        if not activity_date_str:
            return 999  # Very stale if no activity date
        
        try:
            activity_date = datetime.strptime(activity_date_str.split('T')[0], '%Y-%m-%d')
            return (datetime.now() - activity_date).days
        except:
            return 999
    
    def _calculate_days_in_stage(self, opp: Dict[str, Any]) -> int:
        """Calculate days in current stage"""
        
        stage_date_str = opp.get('StageChangeDate') or opp.get('LastModifiedDate')
        
        if not stage_date_str:
            return 0
        
        try:
            stage_date = datetime.strptime(stage_date_str.split('T')[0], '%Y-%m-%d')
            return (datetime.now() - stage_date).days
        except:
            return 0
    
    def _create_at_risk_record(
        self, 
        opp: Dict[str, Any], 
        risk_assessment: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create at-risk deal record"""
        
        total_risk_score = risk_assessment['total_risk_score']
        
        # Determine severity based on risk score
        if total_risk_score >= 80:
            severity = 'CRITICAL'
        elif total_risk_score >= 60:
            severity = 'HIGH'
        else:
            severity = 'MEDIUM'
        
        return {
            'opportunity_id': opp.get('Id'),
            'opportunity_name': opp.get('Name'),
            'account_name': opp.get('Account', {}).get('Name', 'Unknown'),
            'owner_name': opp.get('Owner', {}).get('Name', 'Unassigned'),
            'stage_name': opp.get('StageName'),
            'amount': self._safe_float(opp.get('Amount')),
            'probability': self._safe_float(opp.get('Probability')),
            'close_date': opp.get('CloseDate'),
            'last_activity_date': opp.get('ActivityDate'),
            'total_risk_score': total_risk_score,
            'risk_factor_count': risk_assessment['risk_factor_count'],
            'risk_factors': risk_assessment['risk_factors'],
            'severity': severity,
            'risk_level': severity,
            'issue_type': 'DEAL_AT_RISK',
            'priority': 'HIGH' if severity in ['HIGH', 'CRITICAL'] else 'MEDIUM',
            'description': f"Deal at risk with score {total_risk_score}/100 ({risk_assessment['risk_factor_count']} risk factors)",
            'recommended_action': self._get_recommended_action(severity, risk_assessment['risk_factors']),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_recommended_action(self, severity: str, risk_factors: Dict[str, Any]) -> str:
        """Get recommended action based on severity and risk factors"""
        
        actions = []
        
        if severity == 'CRITICAL':
            actions.append("URGENT: Immediate intervention required")
        
        # Specific actions based on risk factors
        if 'customer_disengagement' in risk_factors:
            actions.append("Schedule customer check-in call")
        
        if 'stage_stagnation' in risk_factors:
            actions.append("Review stage progression barriers")
        
        if 'probability_decline' in risk_factors:
            actions.append("Reassess deal qualification")
        
        if 'competitive_pressure' in risk_factors:
            actions.append("Review competitive positioning")
        
        if 'budget_concerns' in risk_factors:
            actions.append("Address budget and procurement issues")
        
        if not actions:
            actions.append("Monitor deal closely and take appropriate action")
        
        return "; ".join(actions)
    
    def _calculate_at_risk_summary_metrics(
        self, 
        deals_at_risk: List[Dict[str, Any]], 
        open_deals: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate summary metrics for deals at risk"""
        
        if not deals_at_risk:
            return {
                'at_risk_percentage': 0,
                'severity_distribution': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0},
                'total_amount_at_risk': 0,
                'average_risk_score': 0,
                'most_common_risk_factors': []
            }
        
        # Calculate percentages
        at_risk_percentage = (len(deals_at_risk) / len(open_deals)) * 100 if open_deals else 0
        
        # Severity distribution
        severity_distribution = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0}
        total_amount_at_risk = 0
        total_risk_score = 0
        all_risk_factors = []
        
        for deal in deals_at_risk:
            severity = deal['severity']
            severity_distribution[severity] += 1
            total_amount_at_risk += deal['amount']
            total_risk_score += deal['total_risk_score']
            
            # Collect risk factor names
            for factor_name in deal['risk_factors'].keys():
                all_risk_factors.append(factor_name.replace('_', ' ').title())
        
        # Most common risk factors
        factor_counts = {}
        for factor in all_risk_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        most_common_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'at_risk_percentage': round(at_risk_percentage, 1),
            'severity_distribution': severity_distribution,
            'total_amount_at_risk': total_amount_at_risk,
            'average_risk_score': round(total_risk_score / len(deals_at_risk), 1),
            'most_common_risk_factors': [{'factor': factor, 'count': count} for factor, count in most_common_factors]
        }
    
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
