"""
Quarter End Dumping RBA Agent
Single-purpose, focused RBA agent for quarter-end deal dumping detection only

This agent ONLY handles suspicious close date patterns and nothing else.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import calendar

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class QuarterEndDumpingRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for quarter-end dumping detection
    
    Features:
    - ONLY handles quarter-end dumping detection
    - Configuration-driven pattern analysis
    - Lightweight and focused
    - Revenue recognition compliance
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "quarter_end_dumping"
    AGENT_DESCRIPTION = "Identify suspicious close date patterns and quarter-end dumping"
    SUPPORTED_ANALYSIS_TYPES = ["quarter_end_dumping", "close_date_patterns"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Quarter-end dumping specific defaults
        self.default_config = {
            'quarter_end_window_days': 3,
            'dumping_threshold_percentage': 30,
            'last_day_threshold_percentage': 20,
            'minimum_deals_for_analysis': 10,
            'compliance_threshold': 85
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quarter-end dumping detection analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Quarter End Dumping RBA: Analyzing {len(opportunities)} opportunities")
            
            # Filter for closed won deals only
            closed_won_deals = [
                opp for opp in opportunities 
                if 'closed won' in opp.get('StageName', '').lower()
            ]
            
            if len(closed_won_deals) < config.get('minimum_deals_for_analysis', 10):
                return self._create_insufficient_data_response(len(closed_won_deals), config)
            
            # Analyze quarter-end patterns
            quarter_analysis = self._analyze_quarter_patterns(closed_won_deals, config)
            
            # Identify suspicious patterns
            flagged_patterns = self._identify_suspicious_patterns(quarter_analysis, config)
            
            # Create flagged deals list
            flagged_deals = self._create_flagged_deals(closed_won_deals, flagged_patterns, config)
            
            # Calculate compliance metrics
            total_deals = len(closed_won_deals)
            flagged_count = len(flagged_deals)
            
            dumping_score = ((total_deals - flagged_count) / total_deals * 100) if total_deals > 0 else 100.0
            compliance_status = "COMPLIANT" if dumping_score >= config.get('compliance_threshold', 85) else "NON_COMPLIANT"
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'quarter_end_dumping',
                'total_opportunities': len(opportunities),
                'closed_won_deals_analyzed': total_deals,
                'flagged_opportunities': flagged_count,
                'flagged_deals': flagged_deals,
                'quarter_analysis': quarter_analysis,
                'suspicious_patterns': flagged_patterns,
                'dumping_score': round(dumping_score, 1),
                'compliance_status': compliance_status,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Quarter End Dumping RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _analyze_quarter_patterns(self, deals: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quarter-end patterns in deal close dates"""
        
        quarter_data = defaultdict(lambda: {
            'total_deals': 0,
            'total_amount': 0,
            'last_day_deals': 0,
            'last_day_amount': 0,
            'quarter_end_window_deals': 0,
            'quarter_end_window_amount': 0,
            'daily_distribution': defaultdict(int),
            'deals': []
        })
        
        window_days = config.get('quarter_end_window_days', 3)
        
        for deal in deals:
            close_date_str = deal.get('CloseDate')
            if not close_date_str:
                continue
            
            try:
                close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
                amount = float(deal.get('Amount', 0))
                
                # Determine quarter
                quarter = f"{close_date.year}Q{(close_date.month-1)//3 + 1}"
                
                # Get quarter end date
                quarter_end_date = self._get_quarter_end_date(close_date.year, (close_date.month-1)//3 + 1)
                
                # Calculate days from quarter end
                days_from_quarter_end = (quarter_end_date - close_date).days
                
                # Update quarter data
                quarter_info = quarter_data[quarter]
                quarter_info['total_deals'] += 1
                quarter_info['total_amount'] += amount
                quarter_info['deals'].append(deal)
                
                # Daily distribution
                quarter_info['daily_distribution'][close_date.day] += 1
                
                # Check if last day of quarter
                if close_date.date() == quarter_end_date.date():
                    quarter_info['last_day_deals'] += 1
                    quarter_info['last_day_amount'] += amount
                
                # Check if within quarter-end window
                if 0 <= days_from_quarter_end <= window_days:
                    quarter_info['quarter_end_window_deals'] += 1
                    quarter_info['quarter_end_window_amount'] += amount
                    
            except Exception as e:
                logger.warning(f"Failed to parse close date {close_date_str}: {e}")
                continue
        
        return dict(quarter_data)
    
    def _get_quarter_end_date(self, year: int, quarter: int) -> datetime:
        """Get the last day of a quarter"""
        
        if quarter == 1:
            return datetime(year, 3, 31)
        elif quarter == 2:
            return datetime(year, 6, 30)
        elif quarter == 3:
            return datetime(year, 9, 30)
        else:  # quarter == 4
            return datetime(year, 12, 31)
    
    def _identify_suspicious_patterns(self, quarter_analysis: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify suspicious quarter-end patterns"""
        
        suspicious_patterns = []
        
        dumping_threshold = config.get('dumping_threshold_percentage', 30)
        last_day_threshold = config.get('last_day_threshold_percentage', 20)
        
        for quarter, data in quarter_analysis.items():
            total_deals = data['total_deals']
            
            if total_deals < 5:  # Skip quarters with too few deals
                continue
            
            # Check last day clustering
            last_day_percentage = (data['last_day_deals'] / total_deals) * 100
            if last_day_percentage >= last_day_threshold:
                suspicious_patterns.append({
                    'pattern_type': 'LAST_DAY_CLUSTERING',
                    'quarter': quarter,
                    'total_deals': total_deals,
                    'last_day_deals': data['last_day_deals'],
                    'last_day_percentage': round(last_day_percentage, 1),
                    'threshold': last_day_threshold,
                    'severity': 'HIGH' if last_day_percentage >= 40 else 'MEDIUM',
                    'amount_impact': data['last_day_amount'],
                    'description': f"Excessive last-day-of-quarter deals: {last_day_percentage:.1f}% of deals"
                })
            
            # Check quarter-end window clustering
            window_percentage = (data['quarter_end_window_deals'] / total_deals) * 100
            if window_percentage >= dumping_threshold:
                suspicious_patterns.append({
                    'pattern_type': 'QUARTER_END_WINDOW_CLUSTERING',
                    'quarter': quarter,
                    'total_deals': total_deals,
                    'window_deals': data['quarter_end_window_deals'],
                    'window_percentage': round(window_percentage, 1),
                    'threshold': dumping_threshold,
                    'severity': 'HIGH' if window_percentage >= 50 else 'MEDIUM',
                    'amount_impact': data['quarter_end_window_amount'],
                    'description': f"Excessive quarter-end clustering: {window_percentage:.1f}% of deals"
                })
        
        return suspicious_patterns
    
    def _create_flagged_deals(
        self, 
        deals: List[Dict[str, Any]], 
        suspicious_patterns: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create flagged deals based on suspicious patterns"""
        
        flagged_deals = []
        
        for pattern in suspicious_patterns:
            quarter = pattern['quarter']
            pattern_type = pattern['pattern_type']
            
            # Find deals in this quarter that match the pattern
            for deal in deals:
                close_date_str = deal.get('CloseDate')
                if not close_date_str:
                    continue
                
                try:
                    close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
                    deal_quarter = f"{close_date.year}Q{(close_date.month-1)//3 + 1}"
                    
                    if deal_quarter != quarter:
                        continue
                    
                    # Check if deal matches the suspicious pattern
                    if self._deal_matches_pattern(deal, close_date, pattern_type, config):
                        flagged_deals.append({
                            'opportunity_id': deal.get('Id'),
                            'opportunity_name': deal.get('Name'),
                            'account_name': deal.get('Account', {}).get('Name', 'Unknown'),
                            'owner_name': deal.get('Owner', {}).get('Name', 'Unassigned'),
                            'stage_name': deal.get('StageName'),
                            'amount': float(deal.get('Amount', 0)),
                            'close_date': close_date_str,
                            'quarter': quarter,
                            'pattern_type': pattern_type,
                            'pattern_severity': pattern['severity'],
                            'days_from_quarter_end': self._calculate_days_from_quarter_end(close_date),
                            'severity': pattern['severity'],
                            'risk_level': pattern['severity'],
                            'issue_type': 'QUARTER_END_DUMPING',
                            'priority': 'HIGH' if pattern['severity'] == 'HIGH' else 'MEDIUM',
                            'description': f"Deal closed with suspicious quarter-end pattern: {pattern['description']}",
                            'recommended_action': self._get_recommended_action(pattern_type, pattern['severity']),
                            'analysis_timestamp': datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to process deal {deal.get('Id', 'Unknown')}: {e}")
                    continue
        
        return flagged_deals
    
    def _deal_matches_pattern(self, deal: Dict[str, Any], close_date: datetime, pattern_type: str, config: Dict[str, Any]) -> bool:
        """Check if a deal matches a suspicious pattern"""
        
        quarter_end_date = self._get_quarter_end_date(close_date.year, (close_date.month-1)//3 + 1)
        days_from_quarter_end = (quarter_end_date - close_date).days
        
        if pattern_type == 'LAST_DAY_CLUSTERING':
            return close_date.date() == quarter_end_date.date()
        elif pattern_type == 'QUARTER_END_WINDOW_CLUSTERING':
            window_days = config.get('quarter_end_window_days', 3)
            return 0 <= days_from_quarter_end <= window_days
        
        return False
    
    def _calculate_days_from_quarter_end(self, close_date: datetime) -> int:
        """Calculate days from quarter end"""
        
        quarter_end_date = self._get_quarter_end_date(close_date.year, (close_date.month-1)//3 + 1)
        return (quarter_end_date - close_date).days
    
    def _get_recommended_action(self, pattern_type: str, severity: str) -> str:
        """Get recommended action based on pattern and severity"""
        
        if pattern_type == 'LAST_DAY_CLUSTERING':
            if severity == 'HIGH':
                return "URGENT: Review revenue recognition timing and deal authenticity"
            else:
                return "Review deal timing and validate customer commitment"
        elif pattern_type == 'QUARTER_END_WINDOW_CLUSTERING':
            if severity == 'HIGH':
                return "URGENT: Audit quarter-end sales practices and revenue recognition"
            else:
                return "Review sales process compliance and deal timing"
        
        return "Review suspicious close date pattern"
    
    def _create_insufficient_data_response(self, deal_count: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create response for insufficient data"""
        
        return {
            'success': True,
            'agent_name': self.AGENT_NAME,
            'analysis_type': 'quarter_end_dumping',
            'total_opportunities': deal_count,
            'closed_won_deals_analyzed': deal_count,
            'flagged_opportunities': 0,
            'flagged_deals': [],
            'quarter_analysis': {},
            'suspicious_patterns': [],
            'dumping_score': 100.0,
            'compliance_status': 'INSUFFICIENT_DATA',
            'warning': f"Insufficient data for analysis. Need at least {config.get('minimum_deals_for_analysis', 10)} closed won deals, found {deal_count}",
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
