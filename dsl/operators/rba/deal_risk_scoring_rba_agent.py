"""
Deal Risk Scoring RBA Agent
Single-purpose, focused RBA agent for deal risk scoring only

This agent ONLY handles deal risk assessment and nothing else.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class DealRiskScoringRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for deal risk scoring
    
    Features:
    - ONLY handles deal risk assessment
    - Configuration-driven risk factors
    - Lightweight and focused
    - Multi-factor risk scoring
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "deal_risk_scoring"
    AGENT_DESCRIPTION = "Assess deal risk levels using multiple risk factors"
    SUPPORTED_ANALYSIS_TYPES = ["deal_risk_scoring", "risk_assessment"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Risk scoring specific defaults
        self.default_config = {
            'high_risk_threshold': 70,
            'medium_risk_threshold': 40,
            'deal_size_weight': 25,
            'stage_weight': 20,
            'close_date_weight': 20,
            'activity_weight': 20,
            'probability_weight': 15
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deal risk scoring analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Deal Risk Scoring RBA: Analyzing {len(opportunities)} opportunities")
            
            # Calculate risk scores for all deals
            risk_assessments = []
            
            for opp in opportunities:

            
                if opp is None:

            
                    continue  # Skip None opportunities
                risk_score = self._calculate_individual_risk_score(opp, config)
                risk_level = self._determine_risk_level(risk_score, config)
                
                if risk_score > 0:  # Include all deals with risk scores
                    risk_assessments.append(
                        self._create_risk_assessment_record(opp, risk_score, risk_level, config)
                    )
            
            # Calculate summary metrics
            summary_metrics = self._calculate_risk_summary_metrics(risk_assessments, config)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'deal_risk_scoring',
                'total_opportunities': len(opportunities),
                'assessed_opportunities': len(risk_assessments),
                'flagged_opportunities': len([r for r in risk_assessments if r['risk_level'] in ['HIGH', 'CRITICAL']]),
                'risk_assessments': risk_assessments,
                'summary_metrics': summary_metrics,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Deal Risk Scoring RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _calculate_individual_risk_score(self, opp: Dict[str, Any], config: Dict[str, Any]) -> int:
        """Calculate risk score for individual opportunity (0-100)"""
        
        risk_score = 0
        current_date = datetime.now()
        
        # Factor 1: Deal Size (configurable weight)
        deal_size_weight = config.get('deal_size_weight', 25)
        amount = self._safe_float(opp.get('Amount'))
        
        if amount > 500000:  # Large deals - lower risk
            risk_score += int(deal_size_weight * 0.2)
        elif amount > 100000:  # Medium deals
            risk_score += int(deal_size_weight * 0.6)
        else:  # Small deals - higher risk
            risk_score += deal_size_weight
        
        # Factor 2: Stage Analysis (configurable weight)
        stage_weight = config.get('stage_weight', 20)
        stage = opp.get('StageName', '').lower()
        
        if 'closed' in stage:
            risk_score += 0  # No risk if closed
        elif any(keyword in stage for keyword in ['proposal', 'negotiation', 'contract']):
            risk_score += int(stage_weight * 0.5)  # Medium risk in late stages
        elif any(keyword in stage for keyword in ['qualification', 'discovery', 'lead']):
            risk_score += stage_weight  # Higher risk in early stages
        else:
            risk_score += int(stage_weight * 0.75)  # Default medium-high risk
        
        # Factor 3: Close Date Proximity (configurable weight)
        close_date_weight = config.get('close_date_weight', 20)
        close_date_str = opp.get('CloseDate', '')
        
        if close_date_str:
            try:
                close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
                days_to_close = (close_date - current_date).days
                
                if days_to_close < 0:  # Overdue
                    risk_score += close_date_weight
                elif days_to_close < 30:  # Close soon
                    risk_score += int(close_date_weight * 0.25)
                elif days_to_close < 90:  # Medium term
                    risk_score += int(close_date_weight * 0.5)
                else:  # Far future
                    risk_score += int(close_date_weight * 0.75)
            except:
                risk_score += close_date_weight  # Invalid date = high risk
        else:
            risk_score += close_date_weight  # No close date = high risk
        
        # Factor 4: Activity Level (configurable weight)
        activity_weight = config.get('activity_weight', 20)
        activity_date_str = opp.get('ActivityDate')
        
        if activity_date_str:
            try:
                activity_date = datetime.strptime(activity_date_str.split('T')[0], '%Y-%m-%d')
                days_since_activity = (current_date - activity_date).days
                
                if days_since_activity <= 7:  # Recent activity
                    risk_score += int(activity_weight * 0.2)
                elif days_since_activity <= 30:  # Some activity
                    risk_score += int(activity_weight * 0.5)
                else:  # Stale activity
                    risk_score += activity_weight
            except:
                risk_score += activity_weight  # Invalid date = high risk
        else:
            risk_score += activity_weight  # No activity = high risk
        
        # Factor 5: Probability Analysis (configurable weight)
        probability_weight = config.get('probability_weight', 15)
        probability = self._safe_float(opp.get('Probability'))
        
        if probability >= 80:  # High probability
            risk_score += int(probability_weight * 0.2)
        elif probability >= 50:  # Medium probability
            risk_score += int(probability_weight * 0.5)
        else:  # Low probability
            risk_score += probability_weight
        
        return min(risk_score, 100)  # Cap at 100
    
    def _determine_risk_level(self, risk_score: int, config: Dict[str, Any]) -> str:
        """Determine risk level based on score"""
        
        high_threshold = config.get('high_risk_threshold', 70)
        medium_threshold = config.get('medium_risk_threshold', 40)
        
        if risk_score >= high_threshold:
            return 'HIGH'
        elif risk_score >= medium_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _create_risk_assessment_record(
        self, 
        opp: Dict[str, Any], 
        risk_score: int, 
        risk_level: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create risk assessment record"""
        
        return {
            'opportunity_id': opp.get('Id'),
            'opportunity_name': opp.get('Name'),
            'account_name': (opp.get('Account') or {}).get('Name', 'Unknown'),
            'owner_name': (opp.get('Owner') or {}).get('Name', 'Unassigned'),
            'stage_name': opp.get('StageName'),
            'amount': self._safe_float(opp.get('Amount')),
            'probability': self._safe_float(opp.get('Probability')),
            'close_date': opp.get('CloseDate'),
            'last_activity_date': opp.get('ActivityDate'),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'severity': risk_level,
            'issue_type': 'DEAL_RISK',
            'priority': 'HIGH' if risk_level == 'HIGH' else 'MEDIUM' if risk_level == 'MEDIUM' else 'LOW',
            'description': f"Deal risk score: {risk_score}/100 ({risk_level} risk)",
            'recommended_action': self._get_recommended_action(risk_level, risk_score),
            'risk_factors': self._identify_risk_factors(opp, risk_score, config),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_recommended_action(self, risk_level: str, risk_score: int) -> str:
        """Get recommended action based on risk level"""
        
        if risk_level == 'HIGH':
            return f"URGENT: High risk deal (score: {risk_score}) - immediate attention required"
        elif risk_level == 'MEDIUM':
            return f"Review medium risk deal (score: {risk_score}) - monitor closely"
        else:
            return f"Low risk deal (score: {risk_score}) - standard follow-up"
    
    def _identify_risk_factors(self, opp: Dict[str, Any], risk_score: int, config: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors for the deal"""
        
        factors = []
        
        # Check deal size
        amount = self._safe_float(opp.get('Amount'))
        if amount < 100000:
            factors.append("Small deal size increases risk")
        
        # Check stage
        stage = opp.get('StageName', '').lower()
        if any(keyword in stage for keyword in ['qualification', 'discovery', 'lead']):
            factors.append("Early stage increases risk")
        
        # Check close date
        close_date_str = opp.get('CloseDate')
        if close_date_str:
            try:
                close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
                days_to_close = (close_date - datetime.now()).days
                if days_to_close < 0:
                    factors.append("Overdue close date")
                elif days_to_close > 180:
                    factors.append("Distant close date")
            except:
                factors.append("Invalid close date")
        else:
            factors.append("Missing close date")
        
        # Check activity
        activity_date_str = opp.get('ActivityDate')
        if activity_date_str:
            try:
                activity_date = datetime.strptime(activity_date_str.split('T')[0], '%Y-%m-%d')
                days_since_activity = (datetime.now() - activity_date).days
                if days_since_activity > 30:
                    factors.append("Stale activity")
            except:
                factors.append("Invalid activity date")
        else:
            factors.append("No recent activity")
        
        # Check probability
        probability = self._safe_float(opp.get('Probability'))
        if probability < 50:
            factors.append("Low probability")
        
        return factors
    
    def _calculate_risk_summary_metrics(self, risk_assessments: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk summary metrics"""
        
        if not risk_assessments:
            return {
                'average_risk_score': 0,
                'risk_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                'total_amount_at_risk': 0,
                'most_common_risk_factors': []
            }
        
        # Risk distribution
        risk_distribution = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        total_risk_score = 0
        total_amount_at_risk = 0
        all_risk_factors = []
        
        for assessment in risk_assessments:
            risk_level = assessment['risk_level']
            risk_distribution[risk_level] += 1
            total_risk_score += assessment['risk_score']
            
            if risk_level in ['HIGH', 'MEDIUM']:
                total_amount_at_risk += assessment['amount']
            
            all_risk_factors.extend(assessment.get('risk_factors', []))
        
        # Most common risk factors
        factor_counts = {}
        for factor in all_risk_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        most_common_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'average_risk_score': round(total_risk_score / len(risk_assessments), 1),
            'risk_distribution': risk_distribution,
            'total_amount_at_risk': total_amount_at_risk,
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
