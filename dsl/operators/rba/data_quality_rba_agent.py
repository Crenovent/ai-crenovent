"""
Data Quality RBA Agent
Single-purpose, focused RBA agent for data quality issues detection only

This agent ONLY handles comprehensive data quality assessment.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any, Set
from datetime import datetime

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class DataQualityRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for data quality assessment
    
    Features:
    - ONLY handles data quality issues detection
    - Configuration-driven quality checks
    - Lightweight and focused
    - Comprehensive field validation
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "data_quality"
    AGENT_DESCRIPTION = "Comprehensive data quality assessment and validation"
    SUPPORTED_ANALYSIS_TYPES = ["data_quality", "data_quality_audit", "field_validation"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Data quality specific defaults
        self.default_config = {
            'required_fields': ['Id', 'Name', 'Amount', 'CloseDate', 'StageName', 'OwnerId', 'Probability'],
            'critical_fields': ['Id', 'Name', 'Amount', 'CloseDate'],
            'optional_fields': ['Description', 'NextStep', 'LeadSource', 'Type'],
            'compliance_threshold': 85,
            'validate_data_formats': True,
            'check_business_logic': True
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data quality assessment"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Data Quality RBA: Analyzing {len(opportunities)} opportunities")
            
            # Perform comprehensive data quality assessment
            quality_issues = []
            
            for opp in opportunities:

            
                if opp is None:

            
                    continue  # Skip None opportunities
                issues = self._assess_opportunity_data_quality(opp, config)
                if issues:
                    quality_issues.extend(issues)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_data_quality_metrics(opportunities, quality_issues, config)
            
            # Generate field-level analysis
            field_analysis = self._generate_field_analysis(quality_issues, config)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'data_quality',
                'total_opportunities': len(opportunities),
                'opportunities_with_issues': len(set(issue['opportunity_id'] for issue in quality_issues)),
                'total_quality_issues': len(quality_issues),
                'quality_issues': quality_issues,
                'quality_metrics': quality_metrics,
                'field_analysis': field_analysis,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Data Quality RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _assess_opportunity_data_quality(self, opp: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess data quality for a single opportunity"""
        
        issues = []
        required_fields = config.get('required_fields', [])
        critical_fields = config.get('critical_fields', [])
        validate_formats = config.get('validate_data_formats', True)
        check_business_logic = config.get('check_business_logic', True)
        
        # Check missing fields
        missing_issues = self._check_missing_fields(opp, required_fields, critical_fields)
        issues.extend(missing_issues)
        
        # Check data format validation (if enabled)
        if validate_formats:
            format_issues = self._check_data_formats(opp)
            issues.extend(format_issues)
        
        # Check business logic validation (if enabled)
        if check_business_logic:
            logic_issues = self._check_business_logic_rules(opp)
            issues.extend(logic_issues)
        
        return issues
    
    def _check_missing_fields(self, opp: Dict[str, Any], required_fields: List[str], critical_fields: List[str]) -> List[Dict[str, Any]]:
        """Check for missing required fields"""
        
        issues = []
        
        for field in required_fields:
            value = opp.get(field)
            
            # Check if field is missing or empty
            if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
                severity = 'CRITICAL' if field in critical_fields else 'HIGH'
                
                issues.append({
                    'opportunity_id': opp.get('Id', 'Unknown'),
                    'opportunity_name': opp.get('Name', 'Unnamed'),
                    'account_name': (opp.get('Account') or {}).get('Name', 'Unknown'),
                    'owner_name': (opp.get('Owner') or {}).get('Name', 'Unassigned'),
                    'issue_type': 'MISSING_FIELD',
                    'field_name': field,
                    'severity': severity,
                    'risk_level': severity,
                    'priority': 'HIGH' if severity == 'CRITICAL' else 'MEDIUM',
                    'description': f"Missing required field: {field}",
                    'recommended_action': f"Complete missing field: {field}",
                    'analysis_timestamp': datetime.now().isoformat()
                })
        
        return issues
    
    def _check_data_formats(self, opp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check data format validation"""
        
        issues = []
        
        # Check Amount format
        amount = opp.get('Amount')
        if amount is not None:
            try:
                float_amount = float(amount)
                if float_amount < 0:
                    issues.append(self._create_format_issue(opp, 'Amount', 'Negative amount value', 'MEDIUM'))
            except (ValueError, TypeError):
                issues.append(self._create_format_issue(opp, 'Amount', 'Invalid amount format', 'HIGH'))
        
        # Check Probability format
        probability = opp.get('Probability')
        if probability is not None:
            try:
                prob_value = float(probability)
                if not (0 <= prob_value <= 100):
                    issues.append(self._create_format_issue(opp, 'Probability', 'Probability outside 0-100 range', 'MEDIUM'))
            except (ValueError, TypeError):
                issues.append(self._create_format_issue(opp, 'Probability', 'Invalid probability format', 'HIGH'))
        
        # Check CloseDate format
        close_date = opp.get('CloseDate')
        if close_date:
            try:
                datetime.strptime(close_date.split('T')[0], '%Y-%m-%d')
            except (ValueError, AttributeError):
                issues.append(self._create_format_issue(opp, 'CloseDate', 'Invalid date format', 'HIGH'))
        
        # Check Email format (if present)
        contact_email = opp.get('ContactEmail') or opp.get('Email')
        if contact_email and '@' not in str(contact_email):
            issues.append(self._create_format_issue(opp, 'Email', 'Invalid email format', 'LOW'))
        
        return issues
    
    def _check_business_logic_rules(self, opp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check business logic validation rules"""
        
        issues = []
        
        # Rule 1: High amount deals should not have very low probability
        amount = self._safe_float(opp.get('Amount'))
        probability = self._safe_float(opp.get('Probability'))
        
        if amount > 100000 and probability < 10:
            issues.append(self._create_logic_issue(
                opp, 'Amount/Probability Mismatch', 
                f'High value deal (${amount:,.0f}) with very low probability ({probability}%)', 
                'MEDIUM'
            ))
        
        # Rule 2: Closed deals should have 0% or 100% probability
        stage_name = opp.get('StageName', '').lower()
        if 'closed' in stage_name:
            if 'won' in stage_name and probability != 100:
                issues.append(self._create_logic_issue(
                    opp, 'Closed Won Probability', 
                    f'Closed Won deal has {probability}% probability (should be 100%)', 
                    'MEDIUM'
                ))
            elif 'lost' in stage_name and probability != 0:
                issues.append(self._create_logic_issue(
                    opp, 'Closed Lost Probability', 
                    f'Closed Lost deal has {probability}% probability (should be 0%)', 
                    'MEDIUM'
                ))
        
        # Rule 3: Future close dates for closed deals
        if 'closed' in stage_name:
            close_date_str = opp.get('CloseDate')
            if close_date_str:
                try:
                    close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
                    if close_date > datetime.now():
                        issues.append(self._create_logic_issue(
                            opp, 'Future Close Date', 
                            f'Closed deal has future close date: {close_date_str}', 
                            'HIGH'
                        ))
                except:
                    pass
        
        # Rule 4: Very old opportunities still open
        create_date_str = opp.get('CreatedDate')
        if create_date_str and 'closed' not in stage_name:
            try:
                create_date = datetime.strptime(create_date_str.split('T')[0], '%Y-%m-%d')
                days_old = (datetime.now() - create_date).days
                if days_old > 365:  # Over a year old
                    issues.append(self._create_logic_issue(
                        opp, 'Very Old Open Deal', 
                        f'Deal open for {days_old} days (over 1 year)', 
                        'LOW'
                    ))
            except:
                pass
        
        return issues
    
    def _create_format_issue(self, opp: Dict[str, Any], field_name: str, description: str, severity: str) -> Dict[str, Any]:
        """Create format validation issue record"""
        
        return {
            'opportunity_id': opp.get('Id', 'Unknown'),
            'opportunity_name': opp.get('Name', 'Unnamed'),
            'account_name': (opp.get('Account') or {}).get('Name', 'Unknown'),
            'owner_name': (opp.get('Owner') or {}).get('Name', 'Unassigned'),
            'issue_type': 'FORMAT_ISSUE',
            'field_name': field_name,
            'severity': severity,
            'risk_level': severity,
            'priority': 'HIGH' if severity in ['HIGH', 'CRITICAL'] else 'MEDIUM' if severity == 'MEDIUM' else 'LOW',
            'description': description,
            'recommended_action': f'Fix format issue in {field_name}',
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _create_logic_issue(self, opp: Dict[str, Any], rule_name: str, description: str, severity: str) -> Dict[str, Any]:
        """Create business logic issue record"""
        
        return {
            'opportunity_id': opp.get('Id', 'Unknown'),
            'opportunity_name': opp.get('Name', 'Unnamed'),
            'account_name': (opp.get('Account') or {}).get('Name', 'Unknown'),
            'owner_name': (opp.get('Owner') or {}).get('Name', 'Unassigned'),
            'issue_type': 'LOGIC_ISSUE',
            'rule_name': rule_name,
            'severity': severity,
            'risk_level': severity,
            'priority': 'HIGH' if severity in ['HIGH', 'CRITICAL'] else 'MEDIUM' if severity == 'MEDIUM' else 'LOW',
            'description': description,
            'recommended_action': f'Review and correct: {rule_name}',
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_data_quality_metrics(
        self, 
        opportunities: List[Dict[str, Any]], 
        quality_issues: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        
        total_opportunities = len(opportunities)
        opportunities_with_issues = len(set(issue['opportunity_id'] for issue in quality_issues))
        clean_opportunities = total_opportunities - opportunities_with_issues
        
        # Calculate overall quality score
        quality_score = (clean_opportunities / total_opportunities * 100) if total_opportunities > 0 else 100
        
        # Issue type breakdown
        issue_type_counts = {}
        severity_counts = {}
        
        for issue in quality_issues:
            issue_type = issue['issue_type']
            severity = issue['severity']
            
            issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        compliance_threshold = config.get('compliance_threshold', 85)
        compliance_status = "COMPLIANT" if quality_score >= compliance_threshold else "NON_COMPLIANT"
        
        return {
            'overall_quality_score': round(quality_score, 1),
            'compliance_status': compliance_status,
            'compliance_threshold': compliance_threshold,
            'total_issues': len(quality_issues),
            'opportunities_with_issues': opportunities_with_issues,
            'clean_opportunities': clean_opportunities,
            'issue_type_breakdown': issue_type_counts,
            'severity_breakdown': severity_counts
        }
    
    def _generate_field_analysis(self, quality_issues: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate field-level analysis"""
        
        field_issue_counts = {}
        most_problematic_fields = []
        
        # Count issues per field
        for issue in quality_issues:
            field_name = issue.get('field_name')
            if field_name:
                field_issue_counts[field_name] = field_issue_counts.get(field_name, 0) + 1
        
        # Sort by most problematic
        if field_issue_counts:
            most_problematic_fields = sorted(
                field_issue_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        
        return {
            'field_issue_counts': field_issue_counts,
            'most_problematic_fields': [
                {'field': field, 'issue_count': count} 
                for field, count in most_problematic_fields
            ],
            'total_fields_with_issues': len(field_issue_counts)
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
