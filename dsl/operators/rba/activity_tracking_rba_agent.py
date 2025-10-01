"""
Activity Tracking RBA Agent
Single-purpose, focused RBA agent for activity tracking analysis only

This agent ONLY handles missing activity detection and nothing else.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class ActivityTrackingRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for activity tracking
    
    Features:
    - ONLY handles activity tracking analysis
    - Configuration-driven activity thresholds
    - Lightweight and focused
    - Missing activity detection
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "activity_tracking"
    AGENT_DESCRIPTION = "Identify deals with missing recent activities"
    SUPPORTED_ANALYSIS_TYPES = ["activity_tracking", "missing_activities"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Activity tracking specific defaults
        self.default_config = {
            'activity_threshold_days': 14,
            'compliance_threshold': 80,
            'critical_activity_threshold_days': 30
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute activity tracking analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Activity Tracking RBA: Analyzing {len(opportunities)} opportunities")
            
            activity_threshold = config.get('activity_threshold_days', 14)
            critical_threshold = config.get('critical_activity_threshold_days', 30)
            
            # Identify deals with missing activities
            deals_missing_activity = []
            critical_missing_activity = []
            
            for opp in opportunities:
                days_since_activity = self._calculate_days_since_activity(opp)
                stage_name = opp.get('StageName', '')
                
                # Skip closed deals
                if 'closed' in stage_name.lower():
                    continue
                
                # Check for missing activities
                if days_since_activity >= critical_threshold:
                    critical_missing_activity.append(
                        self._create_activity_record(opp, days_since_activity, 'CRITICAL')
                    )
                elif days_since_activity >= activity_threshold:
                    deals_missing_activity.append(
                        self._create_activity_record(opp, days_since_activity, 'MISSING')
                    )
            
            # Calculate compliance metrics
            total_open_deals = len([opp for opp in opportunities if 'closed' not in opp.get('StageName', '').lower()])
            total_missing_activity = len(deals_missing_activity) + len(critical_missing_activity)
            
            activity_score = ((total_open_deals - total_missing_activity) / total_open_deals * 100) if total_open_deals > 0 else 100.0
            compliance_status = "COMPLIANT" if activity_score >= config.get('compliance_threshold', 80) else "NON_COMPLIANT"
            
            all_flagged_deals = deals_missing_activity + critical_missing_activity
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'activity_tracking',
                'total_opportunities': len(opportunities),
                'total_open_deals': total_open_deals,
                'flagged_opportunities': len(all_flagged_deals),
                'missing_activity_count': len(deals_missing_activity),
                'critical_missing_activity_count': len(critical_missing_activity),
                'flagged_deals': all_flagged_deals,
                'activity_score': round(activity_score, 1),
                'compliance_status': compliance_status,
                'thresholds_used': {
                    'activity_threshold_days': activity_threshold,
                    'critical_activity_threshold_days': critical_threshold
                },
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Activity Tracking RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
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
    
    def _create_activity_record(self, opp: Dict[str, Any], days_since_activity: int, severity: str) -> Dict[str, Any]:
        """Create activity tracking record"""
        
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
            'days_since_activity': days_since_activity,
            'severity': severity,
            'risk_level': severity,
            'issue_type': 'MISSING_ACTIVITY',
            'priority': 'HIGH' if severity == 'CRITICAL' else 'MEDIUM',
            'description': f"Deal has no activity for {days_since_activity} days",
            'recommended_action': self._get_recommended_action(severity, days_since_activity),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_recommended_action(self, severity: str, days_since_activity: int) -> str:
        """Get recommended action based on severity"""
        
        if severity == 'CRITICAL':
            return f"URGENT: Log activity immediately - {days_since_activity} days without activity"
        else:
            return f"Schedule follow-up activity - {days_since_activity} days since last activity"
    
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
