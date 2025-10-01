"""
Ownerless Deals RBA Agent
Single-purpose, focused RBA agent for ownerless deals detection only

This agent ONLY handles deals without assigned owners and nothing else.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class OwnerlessDealsRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for ownerless deals detection
    
    Features:
    - ONLY handles ownerless deals detection
    - Configuration-driven ownership validation
    - Lightweight and focused
    - Inactive owner detection
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "ownerless_deals"
    AGENT_DESCRIPTION = "Identify deals without assigned owners or with inactive owners"
    SUPPORTED_ANALYSIS_TYPES = ["ownerless_deals", "unassigned_deals", "ownerless_deals_detection"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Ownerless deals specific defaults
        self.default_config = {
            'compliance_threshold': 95,
            'check_inactive_owners': True,
            'exclude_closed_deals': True
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ownerless deals detection analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Ownerless Deals RBA: Analyzing {len(opportunities)} opportunities")
            
            check_inactive = config.get('check_inactive_owners', True)
            exclude_closed = config.get('exclude_closed_deals', True)
            
            # Filter opportunities if needed
            filtered_opportunities = opportunities
            if exclude_closed:
                filtered_opportunities = [
                    opp for opp in opportunities 
                    if 'closed' not in opp.get('StageName', '').lower()
                ]
            
            # Identify ownerless deals
            ownerless_deals = []
            inactive_owner_deals = []
            
            for opp in filtered_opportunities:
                owner_id = opp.get('OwnerId')
                owner_info = opp.get('Owner', {})
                
                # Check for missing owner
                if not owner_id or owner_id == '' or owner_id is None:
                    ownerless_deals.append(
                        self._create_ownerless_record(opp, 'NO_OWNER')
                    )
                    continue
                
                # Check for inactive owner (if enabled)
                if check_inactive and owner_info:
                    is_active = owner_info.get('IsActive', True)
                    if not is_active:
                        inactive_owner_deals.append(
                            self._create_ownerless_record(opp, 'INACTIVE_OWNER')
                        )
            
            # Calculate compliance metrics
            total_opportunities = len(filtered_opportunities)
            total_ownership_issues = len(ownerless_deals) + len(inactive_owner_deals)
            clean_deals = total_opportunities - total_ownership_issues
            
            ownership_score = (clean_deals / total_opportunities * 100) if total_opportunities > 0 else 100.0
            compliance_status = "COMPLIANT" if ownership_score >= config.get('compliance_threshold', 95) else "NON_COMPLIANT"
            
            all_flagged_deals = ownerless_deals + inactive_owner_deals
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'ownerless_deals',
                'total_opportunities': len(opportunities),
                'analyzed_opportunities': total_opportunities,
                'flagged_opportunities': len(all_flagged_deals),
                'ownerless_deals_count': len(ownerless_deals),
                'inactive_owner_deals_count': len(inactive_owner_deals),
                'flagged_deals': all_flagged_deals,
                'ownership_score': round(ownership_score, 1),
                'compliance_status': compliance_status,
                'ownership_analysis': self._generate_ownership_analysis(all_flagged_deals),
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Ownerless Deals RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _create_ownerless_record(self, opp: Dict[str, Any], issue_type: str) -> Dict[str, Any]:
        """Create ownerless deal record"""
        
        severity = 'HIGH' if issue_type == 'NO_OWNER' else 'MEDIUM'
        
        return {
            'opportunity_id': opp.get('Id'),
            'opportunity_name': opp.get('Name'),
            'account_name': (opp.get('Account') or {}).get('Name', 'Unknown'),
            'owner_name': (opp.get('Owner') or {}).get('Name', 'Unassigned'),
            'owner_id': opp.get('OwnerId'),
            'stage_name': opp.get('StageName'),
            'amount': self._safe_float(opp.get('Amount')),
            'probability': self._safe_float(opp.get('Probability')),
            'close_date': opp.get('CloseDate'),
            'ownership_issue_type': issue_type,
            'severity': severity,
            'risk_level': severity,
            'issue_type': 'OWNERSHIP_ISSUE',
            'priority': 'HIGH' if issue_type == 'NO_OWNER' else 'MEDIUM',
            'description': self._get_issue_description(issue_type, opp),
            'recommended_action': self._get_recommended_action(issue_type, opp),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_issue_description(self, issue_type: str, opp: Dict[str, Any]) -> str:
        """Get issue description based on type"""
        
        if issue_type == 'NO_OWNER':
            return f"Deal '{opp.get('Name')}' has no assigned owner"
        elif issue_type == 'INACTIVE_OWNER':
            owner_name = (opp.get('Owner') or {}).get('Name', 'Unassigned')
            return f"Deal '{opp.get('Name')}' assigned to inactive owner: {owner_name}"
        else:
            return f"Deal '{opp.get('Name')}' has ownership issue: {issue_type}"
    
    def _get_recommended_action(self, issue_type: str, opp: Dict[str, Any]) -> str:
        """Get recommended action based on issue type"""
        
        if issue_type == 'NO_OWNER':
            return "URGENT: Assign deal owner immediately based on territory rules"
        elif issue_type == 'INACTIVE_OWNER':
            return "Reassign deal to active owner in same territory"
        else:
            return "Review and resolve ownership issue"
    
    def _generate_ownership_analysis(self, flagged_deals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate ownership analysis summary"""
        
        issue_type_counts = {}
        stage_distribution = {}
        amount_impact = 0
        
        for deal in flagged_deals:

        
            if deal is None:

        
                continue  # Skip None deals
            # Count issue types
            issue_type = deal['ownership_issue_type']
            issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
            
            # Stage distribution
            stage = deal['stage_name']
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
            
            # Amount impact
            amount_impact += deal['amount']
        
        return {
            'issue_type_breakdown': issue_type_counts,
            'stage_distribution': stage_distribution,
            'total_amount_at_risk': amount_impact,
            'average_deal_size': amount_impact / len(flagged_deals) if flagged_deals else 0,
            'most_affected_stage': max(stage_distribution.items(), key=lambda x: x[1])[0] if stage_distribution else None
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
