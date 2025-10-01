"""
Stage Velocity RBA Agent
Single-purpose, focused RBA agent for stage velocity analysis only

This agent ONLY handles stage progression velocity analysis.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class StageVelocityRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for stage velocity analysis
    
    Features:
    - ONLY handles stage velocity analysis
    - Configuration-driven velocity thresholds
    - Lightweight and focused
    - Stage progression tracking
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "stage_velocity"
    AGENT_DESCRIPTION = "Analyze deal progression velocity through sales stages"
    SUPPORTED_ANALYSIS_TYPES = ["stage_velocity", "progression_analysis"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Stage velocity specific defaults
        self.default_config = {
            'stage_velocity_thresholds': {
                'lead': 14,
                'qualified': 21,
                'discovery': 30,
                'proposal': 21,
                'negotiation': 14,
                'contract': 7
            },
            'slow_velocity_threshold': 1.5,  # 1.5x expected time = slow
            'very_slow_velocity_threshold': 2.0,  # 2x expected time = very slow
            'minimum_deals_for_analysis': 5
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stage velocity analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Stage Velocity RBA: Analyzing {len(opportunities)} opportunities")
            
            # Filter for open deals with stage information
            open_deals = [
                opp for opp in opportunities 
                if 'closed' not in opp.get('StageName', '').lower() and opp.get('StageName')
            ]
            
            if len(open_deals) < config.get('minimum_deals_for_analysis', 5):
                return self._create_insufficient_data_response(len(open_deals), config)
            
            # Analyze stage velocity for each deal
            velocity_analysis = []
            
            for opp in open_deals:
                velocity_assessment = self._assess_stage_velocity(opp, config)
                if velocity_assessment:
                    velocity_analysis.append(velocity_assessment)
            
            # Calculate stage velocity metrics
            stage_metrics = self._calculate_stage_velocity_metrics(velocity_analysis, config)
            
            # Identify slow-moving deals
            slow_deals = [
                deal for deal in velocity_analysis 
                if deal['velocity_status'] in ['SLOW', 'VERY_SLOW']
            ]
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'stage_velocity',
                'total_opportunities': len(opportunities),
                'analyzed_opportunities': len(open_deals),
                'flagged_opportunities': len(slow_deals),
                'velocity_analysis': velocity_analysis,
                'slow_moving_deals': slow_deals,
                'stage_metrics': stage_metrics,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Stage Velocity RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _assess_stage_velocity(self, opp: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess stage velocity for a single opportunity"""
        
        stage_name = opp.get('StageName', '').lower()
        stage_change_date_str = opp.get('StageChangeDate') or opp.get('LastModifiedDate')
        
        if not stage_change_date_str:
            return None
        
        try:
            stage_change_date = datetime.strptime(stage_change_date_str.split('T')[0], '%Y-%m-%d')
            days_in_stage = (datetime.now() - stage_change_date).days
        except:
            return None
        
        # Get expected velocity for this stage
        stage_thresholds = config.get('stage_velocity_thresholds', {})
        expected_days = None
        
        # Find matching stage threshold
        for stage_key, threshold in stage_thresholds.items():
            if stage_key in stage_name:
                expected_days = threshold
                break
        
        if expected_days is None:
            expected_days = 30  # Default threshold
        
        # Calculate velocity metrics
        velocity_ratio = days_in_stage / expected_days
        slow_threshold = config.get('slow_velocity_threshold', 1.5)
        very_slow_threshold = config.get('very_slow_velocity_threshold', 2.0)
        
        # Determine velocity status
        if velocity_ratio >= very_slow_threshold:
            velocity_status = 'VERY_SLOW'
            severity = 'HIGH'
        elif velocity_ratio >= slow_threshold:
            velocity_status = 'SLOW'
            severity = 'MEDIUM'
        else:
            velocity_status = 'NORMAL'
            severity = 'LOW'
        
        return {
            'opportunity_id': opp.get('Id'),
            'opportunity_name': opp.get('Name'),
            'account_name': opp.get('Account', {}).get('Name', 'Unknown'),
            'owner_name': opp.get('Owner', {}).get('Name', 'Unassigned'),
            'stage_name': opp.get('StageName'),
            'amount': self._safe_float(opp.get('Amount')),
            'probability': self._safe_float(opp.get('Probability')),
            'close_date': opp.get('CloseDate'),
            'stage_change_date': stage_change_date_str,
            'days_in_stage': days_in_stage,
            'expected_days_in_stage': expected_days,
            'velocity_ratio': round(velocity_ratio, 2),
            'velocity_status': velocity_status,
            'severity': severity,
            'risk_level': severity,
            'issue_type': 'SLOW_STAGE_VELOCITY',
            'priority': 'HIGH' if severity == 'HIGH' else 'MEDIUM' if severity == 'MEDIUM' else 'LOW',
            'description': f"Deal in {opp.get('StageName')} stage for {days_in_stage} days (expected: {expected_days} days)",
            'recommended_action': self._get_recommended_action(velocity_status, days_in_stage, expected_days),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_recommended_action(self, velocity_status: str, days_in_stage: int, expected_days: int) -> str:
        """Get recommended action based on velocity status"""
        
        if velocity_status == 'VERY_SLOW':
            return f"URGENT: Deal stuck for {days_in_stage} days (expected: {expected_days}) - immediate action required"
        elif velocity_status == 'SLOW':
            return f"Review deal progression - {days_in_stage} days in stage (expected: {expected_days})"
        else:
            return f"Normal progression - {days_in_stage} days in stage (expected: {expected_days})"
    
    def _calculate_stage_velocity_metrics(
        self, 
        velocity_analysis: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate stage velocity metrics"""
        
        if not velocity_analysis:
            return {}
        
        # Group by stage
        stage_groups = defaultdict(list)
        for deal in velocity_analysis:
            stage_name = deal['stage_name']
            stage_groups[stage_name].append(deal)
        
        # Calculate metrics per stage
        stage_metrics = {}
        
        for stage_name, deals in stage_groups.items():
            total_deals = len(deals)
            slow_deals = len([d for d in deals if d['velocity_status'] in ['SLOW', 'VERY_SLOW']])
            very_slow_deals = len([d for d in deals if d['velocity_status'] == 'VERY_SLOW'])
            
            avg_days_in_stage = sum(d['days_in_stage'] for d in deals) / total_deals
            avg_velocity_ratio = sum(d['velocity_ratio'] for d in deals) / total_deals
            
            stage_metrics[stage_name] = {
                'total_deals': total_deals,
                'slow_deals': slow_deals,
                'very_slow_deals': very_slow_deals,
                'slow_percentage': round((slow_deals / total_deals) * 100, 1),
                'average_days_in_stage': round(avg_days_in_stage, 1),
                'average_velocity_ratio': round(avg_velocity_ratio, 2),
                'expected_days': deals[0]['expected_days_in_stage']  # Same for all deals in stage
            }
        
        # Overall metrics
        total_deals = len(velocity_analysis)
        total_slow = len([d for d in velocity_analysis if d['velocity_status'] in ['SLOW', 'VERY_SLOW']])
        total_very_slow = len([d for d in velocity_analysis if d['velocity_status'] == 'VERY_SLOW'])
        
        overall_metrics = {
            'total_deals_analyzed': total_deals,
            'total_slow_deals': total_slow,
            'total_very_slow_deals': total_very_slow,
            'slow_percentage': round((total_slow / total_deals) * 100, 1),
            'very_slow_percentage': round((total_very_slow / total_deals) * 100, 1),
            'stage_breakdown': stage_metrics
        }
        
        return overall_metrics
    
    def _create_insufficient_data_response(self, deal_count: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create response for insufficient data"""
        
        return {
            'success': True,
            'agent_name': self.AGENT_NAME,
            'analysis_type': 'stage_velocity',
            'total_opportunities': deal_count,
            'analyzed_opportunities': deal_count,
            'flagged_opportunities': 0,
            'velocity_analysis': [],
            'slow_moving_deals': [],
            'stage_metrics': {},
            'warning': f"Insufficient data for analysis. Need at least {config.get('minimum_deals_for_analysis', 5)} open deals, found {deal_count}",
            'configuration_used': config,
            'analysis_timestamp': datetime.now().isoformat()
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
