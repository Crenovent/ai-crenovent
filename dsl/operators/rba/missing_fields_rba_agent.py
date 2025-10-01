"""
Missing Fields RBA Agent
Single-purpose, focused RBA agent for missing fields detection only

This agent ONLY handles missing critical fields identification.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any, Set
from datetime import datetime

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class MissingFieldsRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for missing fields detection
    
    Features:
    - ONLY handles missing fields detection
    - Configuration-driven field requirements
    - Lightweight and focused
    - Industry-specific required fields
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "missing_fields"
    AGENT_DESCRIPTION = "Identify deals with missing critical data fields"
    SUPPORTED_ANALYSIS_TYPES = ["missing_fields", "data_quality_audit"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Missing fields specific defaults
        self.default_config = {
            'required_fields': ['CloseDate', 'Amount', 'OwnerId', 'StageName', 'Probability'],
            'compliance_threshold': 95,
            'critical_fields': ['CloseDate', 'Amount', 'OwnerId']
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute missing fields detection analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Missing Fields RBA: Analyzing {len(opportunities)} opportunities")
            
            required_fields = config.get('required_fields', self.default_config['required_fields'])
            critical_fields = config.get('critical_fields', self.default_config['critical_fields'])
            
            # Analyze each opportunity for missing fields
            deals_with_missing_fields = []
            
            for opp in opportunities:

            
                if opp is None:

            
                    continue  # Skip None opportunities
                missing_fields = self._identify_missing_fields(opp, required_fields)
                critical_missing = self._identify_missing_fields(opp, critical_fields)
                
                if missing_fields:
                    deals_with_missing_fields.append(
                        self._create_missing_fields_record(opp, missing_fields, critical_missing)
                    )
            
            # Calculate compliance metrics
            total_opportunities = len(opportunities)
            deals_with_issues = len(deals_with_missing_fields)
            clean_deals = total_opportunities - deals_with_issues
            
            compliance_score = (clean_deals / total_opportunities * 100) if total_opportunities > 0 else 100.0
            compliance_status = "COMPLIANT" if compliance_score >= config.get('compliance_threshold', 95) else "NON_COMPLIANT"
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'missing_fields',
                'total_opportunities': total_opportunities,
                'flagged_opportunities': deals_with_issues,
                'clean_opportunities': clean_deals,
                'flagged_deals': deals_with_missing_fields,
                'compliance_score': round(compliance_score, 1),
                'compliance_status': compliance_status,
                'field_analysis': self._generate_field_analysis(deals_with_missing_fields, required_fields),
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Missing Fields RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _identify_missing_fields(self, opp: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Identify missing fields in opportunity"""
        
        missing_fields = []
        
        for field in required_fields:
            value = opp.get(field)
            
            # Check if field is missing or empty
            if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
                missing_fields.append(field)
        
        return missing_fields
    
    def _create_missing_fields_record(
        self, 
        opp: Dict[str, Any], 
        missing_fields: List[str], 
        critical_missing: List[str]
    ) -> Dict[str, Any]:
        """Create missing fields record"""
        
        severity = 'CRITICAL' if critical_missing else 'MEDIUM'
        
        return {
            'opportunity_id': opp.get('Id'),
            'opportunity_name': opp.get('Name'),
            'account_name': (opp.get('Account') or {}).get('Name', 'Unknown'),
            'owner_name': (opp.get('Owner') or {}).get('Name', 'Unassigned'),
            'stage_name': opp.get('StageName'),
            'missing_fields': missing_fields,
            'critical_missing_fields': critical_missing,
            'missing_fields_count': len(missing_fields),
            'critical_missing_count': len(critical_missing),
            'severity': severity,
            'risk_level': severity,
            'issue_type': 'MISSING_FIELDS',
            'priority': 'HIGH' if critical_missing else 'MEDIUM',
            'description': f"Deal missing {len(missing_fields)} required fields: {', '.join(missing_fields)}",
            'recommended_action': self._get_recommended_action(missing_fields, critical_missing),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_recommended_action(self, missing_fields: List[str], critical_missing: List[str]) -> str:
        """Get recommended action based on missing fields"""
        
        if critical_missing:
            return f"URGENT: Complete critical fields: {', '.join(critical_missing)}"
        else:
            return f"Complete missing fields: {', '.join(missing_fields)}"
    
    def _generate_field_analysis(self, deals_with_issues: List[Dict], required_fields: List[str]) -> Dict[str, Any]:
        """Generate field-level analysis"""
        
        field_missing_counts = {}
        
        # Count missing occurrences per field
        for field in required_fields:
            field_missing_counts[field] = 0
        
        for deal in deals_with_issues:

        
            if deal is None:

        
                continue  # Skip None deals
            for missing_field in deal['missing_fields']:
                if missing_field in field_missing_counts:
                    field_missing_counts[missing_field] += 1
        
        # Sort by most problematic fields
        sorted_fields = sorted(field_missing_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'field_missing_counts': field_missing_counts,
            'most_problematic_fields': [{'field': field, 'missing_count': count} for field, count in sorted_fields[:5]],
            'total_fields_analyzed': len(required_fields)
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
