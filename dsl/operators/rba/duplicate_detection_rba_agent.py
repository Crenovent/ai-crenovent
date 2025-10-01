"""
Duplicate Detection RBA Agent
Single-purpose, focused RBA agent for duplicate deals detection only

This agent ONLY handles duplicate deals identification and nothing else.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from difflib import SequenceMatcher

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class DuplicateDetectionRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for duplicate detection
    
    Features:
    - ONLY handles duplicate deals detection
    - Configuration-driven similarity thresholds
    - Lightweight and focused
    - Multiple similarity algorithms
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "duplicate_detection"
    AGENT_DESCRIPTION = "Identify potential duplicate deals"
    SUPPORTED_ANALYSIS_TYPES = ["duplicate_detection", "duplicate_deals"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Duplicate detection specific defaults
        self.default_config = {
            'name_similarity_threshold': 0.85,
            'amount_tolerance_percentage': 5.0,
            'same_account_required': True,
            'compliance_threshold': 95,
            'check_close_date_proximity': True,
            'close_date_proximity_days': 30
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute duplicate detection analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Duplicate Detection RBA: Analyzing {len(opportunities)} opportunities")
            
            # Find potential duplicates
            duplicate_groups = self._find_duplicate_groups(opportunities, config)
            
            # Create duplicate records
            flagged_deals = []
            for group in duplicate_groups:
                flagged_deals.extend(self._create_duplicate_records(group, config))
            
            # Calculate compliance metrics
            total_opportunities = len(opportunities)
            duplicate_count = len(flagged_deals)
            clean_deals = total_opportunities - duplicate_count
            
            duplicate_score = (clean_deals / total_opportunities * 100) if total_opportunities > 0 else 100.0
            compliance_status = "COMPLIANT" if duplicate_score >= config.get('compliance_threshold', 95) else "NON_COMPLIANT"
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'duplicate_detection',
                'total_opportunities': total_opportunities,
                'flagged_opportunities': duplicate_count,
                'duplicate_groups_found': len(duplicate_groups),
                'flagged_deals': flagged_deals,
                'duplicate_score': round(duplicate_score, 1),
                'compliance_status': compliance_status,
                'duplicate_analysis': self._generate_duplicate_analysis(duplicate_groups),
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Duplicate Detection RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _find_duplicate_groups(self, opportunities: List[Dict[str, Any]], config: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Find groups of potential duplicate deals"""
        
        duplicate_groups = []
        processed_ids = set()
        
        name_threshold = config.get('name_similarity_threshold', 0.85)
        amount_tolerance = config.get('amount_tolerance_percentage', 5.0)
        same_account_required = config.get('same_account_required', True)
        check_close_date = config.get('check_close_date_proximity', True)
        close_date_days = config.get('close_date_proximity_days', 30)
        
        for i, opp1 in enumerate(opportunities):
            if opp1.get('Id') in processed_ids:
                continue
                
            potential_group = [opp1]
            
            for j, opp2 in enumerate(opportunities[i+1:], i+1):
                if opp2.get('Id') in processed_ids:
                    continue
                
                if self._are_potential_duplicates(opp1, opp2, name_threshold, amount_tolerance, same_account_required, check_close_date, close_date_days):
                    potential_group.append(opp2)
                    processed_ids.add(opp2.get('Id'))
            
            if len(potential_group) > 1:
                duplicate_groups.append(potential_group)
                for opp in potential_group:
                    processed_ids.add(opp.get('Id'))
        
        return duplicate_groups
    
    def _are_potential_duplicates(
        self, 
        opp1: Dict[str, Any], 
        opp2: Dict[str, Any], 
        name_threshold: float,
        amount_tolerance: float,
        same_account_required: bool,
        check_close_date: bool,
        close_date_days: int
    ) -> bool:
        """Check if two opportunities are potential duplicates"""
        
        # Check account requirement
        if same_account_required:
            account1 = opp1.get('AccountId') or opp1.get('Account', {}).get('Id')
            account2 = opp2.get('AccountId') or opp2.get('Account', {}).get('Id')
            if account1 != account2:
                return False
        
        # Check name similarity
        name1 = (opp1.get('Name') or '').strip().lower()
        name2 = (opp2.get('Name') or '').strip().lower()
        
        if not name1 or not name2:
            return False
        
        name_similarity = SequenceMatcher(None, name1, name2).ratio()
        if name_similarity < name_threshold:
            return False
        
        # Check amount similarity
        amount1 = float(opp1.get('Amount', 0))
        amount2 = float(opp2.get('Amount', 0))
        
        if amount1 > 0 and amount2 > 0:
            amount_diff_percentage = abs(amount1 - amount2) / max(amount1, amount2) * 100
            if amount_diff_percentage > amount_tolerance:
                return False
        
        # Check close date proximity (if enabled)
        if check_close_date:
            close_date1 = opp1.get('CloseDate')
            close_date2 = opp2.get('CloseDate')
            
            if close_date1 and close_date2:
                try:
                    date1 = datetime.strptime(close_date1.split('T')[0], '%Y-%m-%d')
                    date2 = datetime.strptime(close_date2.split('T')[0], '%Y-%m-%d')
                    date_diff_days = abs((date1 - date2).days)
                    
                    if date_diff_days > close_date_days:
                        return False
                except:
                    pass  # Skip date check if parsing fails
        
        return True
    
    def _create_duplicate_records(self, duplicate_group: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create duplicate records for a group"""
        
        records = []
        group_id = f"DUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(duplicate_group)}"
        
        for opp in duplicate_group:
            records.append({
                'opportunity_id': opp.get('Id'),
                'opportunity_name': opp.get('Name'),
                'account_name': opp.get('Account', {}).get('Name', 'Unknown'),
                'owner_name': opp.get('Owner', {}).get('Name', 'Unassigned'),
                'stage_name': opp.get('StageName'),
                'amount': self._safe_float(opp.get('Amount')),
                'probability': self._safe_float(opp.get('Probability')),
                'close_date': opp.get('CloseDate'),
                'duplicate_group_id': group_id,
                'duplicate_group_size': len(duplicate_group),
                'potential_duplicates': [d.get('Id') for d in duplicate_group if d.get('Id') != opp.get('Id')],
                'similarity_factors': self._calculate_similarity_factors(opp, duplicate_group),
                'severity': 'HIGH',
                'risk_level': 'HIGH',
                'issue_type': 'DUPLICATE_DEAL',
                'priority': 'HIGH',
                'description': f"Potential duplicate deal - {len(duplicate_group)} similar deals found",
                'recommended_action': f"Review and merge/close duplicate deals in group {group_id}",
                'analysis_timestamp': datetime.now().isoformat()
            })
        
        return records
    
    def _calculate_similarity_factors(self, opp: Dict[str, Any], group: List[Dict[str, Any]]) -> List[str]:
        """Calculate similarity factors for an opportunity within its group"""
        
        factors = []
        
        for other_opp in group:
            if other_opp.get('Id') == opp.get('Id'):
                continue
            
            # Name similarity
            name1 = (opp.get('Name') or '').strip().lower()
            name2 = (other_opp.get('Name') or '').strip().lower()
            if name1 and name2:
                similarity = SequenceMatcher(None, name1, name2).ratio()
                factors.append(f"Name similarity with {other_opp.get('Id')}: {similarity:.2%}")
            
            # Amount similarity
            amount1 = self._safe_float(opp.get('Amount'))
            amount2 = float(other_opp.get('Amount', 0))
            if amount1 > 0 and amount2 > 0:
                diff_percentage = abs(amount1 - amount2) / max(amount1, amount2) * 100
                factors.append(f"Amount difference with {other_opp.get('Id')}: {diff_percentage:.1f}%")
        
        return factors
    
    def _generate_duplicate_analysis(self, duplicate_groups: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate duplicate analysis summary"""
        
        total_duplicates = sum(len(group) for group in duplicate_groups)
        group_sizes = [len(group) for group in duplicate_groups]
        
        return {
            'total_duplicate_groups': len(duplicate_groups),
            'total_duplicate_deals': total_duplicates,
            'average_group_size': sum(group_sizes) / len(group_sizes) if group_sizes else 0,
            'largest_group_size': max(group_sizes) if group_sizes else 0,
            'group_size_distribution': {
                '2_deals': len([g for g in group_sizes if g == 2]),
                '3_deals': len([g for g in group_sizes if g == 3]),
                '4_plus_deals': len([g for g in group_sizes if g >= 4])
            }
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
