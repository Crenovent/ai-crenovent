"""
Pipeline Summary RBA Agent
Single-purpose, focused RBA agent for pipeline summary analysis only

This agent ONLY handles pipeline summary and overview metrics.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class PipelineSummaryRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for pipeline summary
    
    Features:
    - ONLY handles pipeline summary analysis
    - Configuration-driven metrics calculation
    - Lightweight and focused
    - Comprehensive pipeline overview
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "pipeline_summary"
    AGENT_DESCRIPTION = "Generate comprehensive pipeline summary and overview metrics"
    SUPPORTED_ANALYSIS_TYPES = ["pipeline_summary", "pipeline_overview"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Pipeline summary specific defaults
        self.default_config = {
            'include_closed_deals': True,
            'include_stage_breakdown': True,
            'include_owner_breakdown': True,
            'include_time_analysis': True,
            'probability_weighted': True
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline summary analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Pipeline Summary RBA: Analyzing {len(opportunities)} opportunities")
            
            # Generate comprehensive pipeline summary
            pipeline_summary = self._generate_pipeline_summary(opportunities, config)
            
            # Calculate key metrics
            key_metrics = self._calculate_key_metrics(opportunities, config)
            
            # Generate stage breakdown
            stage_breakdown = self._generate_stage_breakdown(opportunities, config)
            
            # Generate owner breakdown (if enabled)
            owner_breakdown = None
            if config.get('include_owner_breakdown', True):
                owner_breakdown = self._generate_owner_breakdown(opportunities, config)
            
            # Generate time analysis (if enabled)
            time_analysis = None
            if config.get('include_time_analysis', True):
                time_analysis = self._generate_time_analysis(opportunities, config)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'pipeline_summary',
                'total_opportunities': len(opportunities),
                'pipeline_summary': pipeline_summary,
                'key_metrics': key_metrics,
                'stage_breakdown': stage_breakdown,
                'owner_breakdown': owner_breakdown,
                'time_analysis': time_analysis,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline Summary RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _generate_pipeline_summary(self, opportunities: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level pipeline summary"""
        
        total_opportunities = len(opportunities)
        
        # Separate open and closed deals
        open_deals = [opp for opp in opportunities if 'closed' not in opp.get('StageName', '').lower()]
        closed_won_deals = [opp for opp in opportunities if 'closed won' in opp.get('StageName', '').lower()]
        closed_lost_deals = [opp for opp in opportunities if 'closed lost' in opp.get('StageName', '').lower()]
        
        # Calculate amounts
        total_pipeline_value = sum(self._safe_float(opp.get('Amount')) for opp in open_deals)
        total_closed_won_value = sum(self._safe_float(opp.get('Amount')) for opp in closed_won_deals)
        total_closed_lost_value = sum(self._safe_float(opp.get('Amount')) for opp in closed_lost_deals)
        
        # Calculate weighted pipeline (if enabled)
        weighted_pipeline_value = 0
        if config.get('probability_weighted', True):
            for opp in open_deals:
                amount = self._safe_float(opp.get('Amount'))
                probability = self._safe_float(opp.get('Probability')) / 100
                weighted_pipeline_value += amount * probability
        
        # Calculate win rate
        total_closed = len(closed_won_deals) + len(closed_lost_deals)
        win_rate = (len(closed_won_deals) / total_closed * 100) if total_closed > 0 else 0
        
        return {
            'total_opportunities': total_opportunities,
            'open_deals': len(open_deals),
            'closed_won_deals': len(closed_won_deals),
            'closed_lost_deals': len(closed_lost_deals),
            'total_pipeline_value': total_pipeline_value,
            'weighted_pipeline_value': round(weighted_pipeline_value, 2),
            'total_closed_won_value': total_closed_won_value,
            'total_closed_lost_value': total_closed_lost_value,
            'win_rate_percentage': round(win_rate, 1),
            'average_deal_size': round(total_pipeline_value / len(open_deals), 2) if open_deals else 0
        }
    
    def _calculate_key_metrics(self, opportunities: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key pipeline metrics"""
        
        open_deals = [opp for opp in opportunities if 'closed' not in opp.get('StageName', '').lower()]
        
        if not open_deals:
            return {
                'pipeline_health_score': 0,
                'velocity_score': 0,
                'coverage_ratio': 0,
                'conversion_probability': 0
            }
        
        # Calculate pipeline health score (simplified)
        deals_with_activity = len([opp for opp in open_deals if opp.get('ActivityDate')])
        deals_with_close_date = len([opp for opp in open_deals if opp.get('CloseDate')])
        deals_with_amount = len([opp for opp in open_deals if opp.get('Amount')])
        
        health_score = ((deals_with_activity + deals_with_close_date + deals_with_amount) / (len(open_deals) * 3)) * 100
        
        # Calculate velocity score (days since last activity)
        total_days_since_activity = 0
        deals_with_activity_date = 0
        
        for opp in open_deals:
            activity_date_str = opp.get('ActivityDate')
            if activity_date_str:
                try:
                    activity_date = datetime.strptime(activity_date_str.split('T')[0], '%Y-%m-%d')
                    days_since = (datetime.now() - activity_date).days
                    total_days_since_activity += days_since
                    deals_with_activity_date += 1
                except:
                    pass
        
        avg_days_since_activity = total_days_since_activity / deals_with_activity_date if deals_with_activity_date > 0 else 30
        velocity_score = max(0, 100 - (avg_days_since_activity * 2))  # Decreases as days increase
        
        # Calculate coverage ratio (pipeline value vs target)
        # This would typically compare against quotas - using simplified calculation
        total_pipeline = sum(self._safe_float(opp.get('Amount')) for opp in open_deals)
        assumed_target = total_pipeline * 0.7  # Assuming 70% of pipeline as target
        coverage_ratio = (total_pipeline / assumed_target) if assumed_target > 0 else 1
        
        # Calculate conversion probability (weighted average)
        total_weighted_probability = sum(self._safe_float(opp.get('Probability')) for opp in open_deals)
        conversion_probability = total_weighted_probability / len(open_deals) if open_deals else 0
        
        return {
            'pipeline_health_score': round(health_score, 1),
            'velocity_score': round(velocity_score, 1),
            'coverage_ratio': round(coverage_ratio, 2),
            'conversion_probability': round(conversion_probability, 1)
        }
    
    def _generate_stage_breakdown(self, opportunities: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stage breakdown analysis"""
        
        stage_breakdown = defaultdict(lambda: {
            'count': 0,
            'total_value': 0,
            'weighted_value': 0,
            'avg_probability': 0,
            'avg_amount': 0
        })
        
        for opp in opportunities:

        
            if opp is None:

        
                continue  # Skip None opportunities
            stage_name = opp.get('StageName', 'Unknown')
            amount = self._safe_float(opp.get('Amount'))
            probability = self._safe_float(opp.get('Probability'))
            
            stage_data = stage_breakdown[stage_name]
            stage_data['count'] += 1
            stage_data['total_value'] += amount
            stage_data['weighted_value'] += amount * (probability / 100)
        
        # Calculate averages
        for stage_name, data in stage_breakdown.items():
            if data['count'] > 0:
                data['avg_amount'] = round(data['total_value'] / data['count'], 2)
                
                # Calculate average probability for this stage
                stage_opportunities = [opp for opp in opportunities if opp.get('StageName') == stage_name]
                total_probability = sum(self._safe_float(opp.get('Probability')) for opp in stage_opportunities)
                data['avg_probability'] = round(total_probability / len(stage_opportunities), 1)
                
                data['total_value'] = round(data['total_value'], 2)
                data['weighted_value'] = round(data['weighted_value'], 2)
        
        return dict(stage_breakdown)
    
    def _generate_owner_breakdown(self, opportunities: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate owner breakdown analysis"""
        
        owner_breakdown = defaultdict(lambda: {
            'count': 0,
            'total_value': 0,
            'weighted_value': 0,
            'open_deals': 0,
            'closed_won': 0,
            'closed_lost': 0,
            'win_rate': 0
        })
        
        for opp in opportunities:

        
            if opp is None:

        
                continue  # Skip None opportunities
            owner_name = (opp.get('Owner') or {}).get('Name', 'Unassigned')
            amount = self._safe_float(opp.get('Amount'))
            probability = self._safe_float(opp.get('Probability'))
            stage_name = opp.get('StageName', '').lower()
            
            owner_data = owner_breakdown[owner_name]
            owner_data['count'] += 1
            owner_data['total_value'] += amount
            owner_data['weighted_value'] += amount * (probability / 100)
            
            if 'closed won' in stage_name:
                owner_data['closed_won'] += 1
            elif 'closed lost' in stage_name:
                owner_data['closed_lost'] += 1
            else:
                owner_data['open_deals'] += 1
        
        # Calculate win rates
        for owner_name, data in owner_breakdown.items():
            total_closed = data['closed_won'] + data['closed_lost']
            data['win_rate'] = round((data['closed_won'] / total_closed * 100), 1) if total_closed > 0 else 0
            data['total_value'] = round(data['total_value'], 2)
            data['weighted_value'] = round(data['weighted_value'], 2)
        
        return dict(owner_breakdown)
    
    def _generate_time_analysis(self, opportunities: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate time-based analysis"""
        
        current_date = datetime.now()
        
        # Categorize deals by close date
        closing_this_month = []
        closing_next_month = []
        closing_this_quarter = []
        overdue_deals = []
        
        for opp in opportunities:

        
            if opp is None:

        
                continue  # Skip None opportunities
            if 'closed' in opp.get('StageName', '').lower():
                continue
                
            close_date_str = opp.get('CloseDate')
            if not close_date_str:
                continue
                
            try:
                close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
                days_to_close = (close_date - current_date).days
                
                if days_to_close < 0:
                    overdue_deals.append(opp)
                elif days_to_close <= 30:
                    closing_this_month.append(opp)
                elif days_to_close <= 60:
                    closing_next_month.append(opp)
                elif days_to_close <= 90:
                    closing_this_quarter.append(opp)
            except:
                continue
        
        # Calculate values
        def calculate_total_value(deals):
            return sum(self._safe_float(deal.get('Amount')) for deal in deals)
        
        return {
            'closing_this_month': {
                'count': len(closing_this_month),
                'total_value': round(calculate_total_value(closing_this_month), 2)
            },
            'closing_next_month': {
                'count': len(closing_next_month),
                'total_value': round(calculate_total_value(closing_next_month), 2)
            },
            'closing_this_quarter': {
                'count': len(closing_this_quarter),
                'total_value': round(calculate_total_value(closing_this_quarter), 2)
            },
            'overdue_deals': {
                'count': len(overdue_deals),
                'total_value': round(calculate_total_value(overdue_deals), 2)
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
