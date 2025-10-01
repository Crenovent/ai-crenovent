"""
Stale Deals RBA Agent
Single-purpose, focused RBA agent for stale deals detection only

This agent ONLY handles stale deals identification and nothing else.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent
from ...rules.business_rules_engine import BusinessRulesEngine

logger = logging.getLogger(__name__)

class StaleDealsRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for stale deals detection
    
    Features:
    - ONLY handles stale deals detection
    - Configuration-driven via YAML rules
    - Lightweight and focused
    - Stage-specific thresholds
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "stale_deals"
    AGENT_DESCRIPTION = "Identify deals stuck in pipeline stages for too long"
    SUPPORTED_ANALYSIS_TYPES = ["stale_deals", "pipeline_hygiene_stale_deals"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Initialize rules engine
        self.rules_engine = BusinessRulesEngine()
        
        # Stale deals specific defaults
        self.default_config = {
            'stale_threshold_days': 60,
            'critical_threshold_days': 90,
            'minimum_hygiene_score': 70
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stale deals detection analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Stale Deals RBA: Analyzing {len(opportunities)} opportunities")
            
            # Identify stale deals using rules engine
            stale_deals = []
            critical_stale_deals = []
            
            stale_threshold = config.get('stale_threshold_days', 60)
            critical_threshold = config.get('critical_threshold_days', 90)
            
            for opp in opportunities:
                days_in_stage = self._calculate_days_in_stage(opp)
                stage_name = opp.get('StageName', '')
                
                # Skip closed deals
                if 'closed' in stage_name.lower():
                    continue
                
                # Check for stale deals
                if days_in_stage >= critical_threshold:
                    critical_stale_deals.append(self._create_stale_deal_record(opp, days_in_stage, 'CRITICAL'))
                elif days_in_stage >= stale_threshold:
                    stale_deals.append(self._create_stale_deal_record(opp, days_in_stage, 'STALE'))
            
            # Calculate compliance metrics
            total_open_deals = len([opp for opp in opportunities if 'closed' not in opp.get('StageName', '').lower()])
            total_stale = len(stale_deals) + len(critical_stale_deals)
            
            compliance_score = ((total_open_deals - total_stale) / total_open_deals * 100) if total_open_deals > 0 else 100.0
            compliance_status = "COMPLIANT" if compliance_score >= config.get('minimum_hygiene_score', 70) else "NON_COMPLIANT"
            
            all_flagged_deals = stale_deals + critical_stale_deals
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'stale_deals',
                'total_opportunities': len(opportunities),
                'total_open_deals': total_open_deals,
                'flagged_opportunities': len(all_flagged_deals),
                'stale_deals_count': len(stale_deals),
                'critical_stale_deals_count': len(critical_stale_deals),
                'flagged_deals': all_flagged_deals,
                'compliance_score': round(compliance_score, 1),
                'compliance_status': compliance_status,
                'thresholds_used': {
                    'stale_threshold_days': stale_threshold,
                    'critical_threshold_days': critical_threshold
                },
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Stale Deals RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _calculate_days_in_stage(self, opp: Dict[str, Any]) -> int:
        """Calculate days the deal has been in current stage"""
        
        # Try to get stage change date, fallback to last modified date
        stage_date_str = opp.get('StageChangeDate') or opp.get('LastModifiedDate')
        
        if not stage_date_str:
            return 0
        
        try:
            stage_date = datetime.strptime(stage_date_str.split('T')[0], '%Y-%m-%d')
            return (datetime.now() - stage_date).days
        except:
            return 0
    
    def _create_stale_deal_record(self, opp: Dict[str, Any], days_in_stage: int, severity: str) -> Dict[str, Any]:
        """Create stale deal record"""
        
        return {
            'opportunity_id': opp.get('Id'),
            'opportunity_name': opp.get('Name'),
            'account_name': opp.get('Account', {}).get('Name', 'Unknown'),
            'owner_name': opp.get('Owner', {}).get('Name', 'Unassigned'),
            'stage_name': opp.get('StageName'),
            'amount': self._safe_float(opp.get('Amount')),
            'probability': self._safe_float(opp.get('Probability')),
            'close_date': opp.get('CloseDate'),
            'days_in_stage': days_in_stage,
            'severity': severity,
            'risk_level': severity,
            'issue_type': 'STALE_DEAL',
            'priority': 'HIGH' if severity == 'CRITICAL' else 'MEDIUM',
            'description': f"Deal stuck in {opp.get('StageName')} stage for {days_in_stage} days",
            'recommended_action': self._get_recommended_action(severity, opp.get('StageName')),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_recommended_action(self, severity: str, stage_name: str) -> str:
        """Get recommended action based on severity and stage"""
        
        if severity == 'CRITICAL':
            return f"URGENT: Review deal in {stage_name} stage - immediate action required"
        else:
            return f"Review deal progression in {stage_name} stage - update or advance"
    
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
