"""
Forecast Alignment RBA Agent
Single-purpose, focused RBA agent for forecast alignment analysis only

This agent ONLY handles forecast alignment and variance detection.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class ForecastAlignmentRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for forecast alignment
    
    Features:
    - ONLY handles forecast alignment analysis
    - Configuration-driven variance thresholds
    - Lightweight and focused
    - Pipeline vs forecast comparison
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "forecast_alignment"
    AGENT_DESCRIPTION = "Analyze alignment between pipeline and forecast commitments"
    SUPPORTED_ANALYSIS_TYPES = ["forecast_alignment", "forecast_variance"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Forecast alignment specific defaults
        self.default_config = {
            'variance_threshold_percentage': 15,
            'high_variance_threshold_percentage': 25,
            'minimum_forecast_amount': 10000,
            'include_probability_weighting': True,
            'forecast_periods': ['current_quarter', 'current_month', 'next_month']
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute forecast alignment analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Forecast Alignment RBA: Analyzing {len(opportunities)} opportunities")
            
            # Filter for open deals only
            open_deals = [
                opp for opp in opportunities 
                if 'closed' not in opp.get('StageName', '').lower()
            ]
            
            # Analyze forecast alignment for different periods
            alignment_analysis = {}
            forecast_periods = config.get('forecast_periods', ['current_quarter'])
            
            for period in forecast_periods:
                period_analysis = self._analyze_forecast_period(open_deals, period, config)
                alignment_analysis[period] = period_analysis
            
            # Identify variance issues
            variance_issues = self._identify_variance_issues(alignment_analysis, config)
            
            # Calculate overall alignment metrics
            overall_metrics = self._calculate_overall_alignment_metrics(alignment_analysis, config)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'forecast_alignment',
                'total_opportunities': len(opportunities),
                'open_deals_analyzed': len(open_deals),
                'flagged_opportunities': len(variance_issues),
                'alignment_analysis': alignment_analysis,
                'variance_issues': variance_issues,
                'overall_metrics': overall_metrics,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Forecast Alignment RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _analyze_forecast_period(self, opportunities: List[Dict[str, Any]], period: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze forecast alignment for a specific period"""
        
        # Filter opportunities for the period
        period_deals = self._filter_deals_by_period(opportunities, period)
        
        if not period_deals:
            return {
                'period': period,
                'deals_count': 0,
                'pipeline_value': 0,
                'weighted_pipeline_value': 0,
                'forecast_commit': 0,
                'forecast_upside': 0,
                'variance_percentage': 0,
                'alignment_status': 'NO_DATA'
            }
        
        # Calculate pipeline metrics
        pipeline_value = sum(self._safe_float(deal.get('Amount')) for deal in period_deals)
        
        weighted_pipeline_value = 0
        if config.get('include_probability_weighting', True):
            for deal in period_deals:
                amount = self._safe_float(deal.get('Amount'))
                probability = self._safe_float(deal.get('Probability')) / 100
                weighted_pipeline_value += amount * probability
        else:
            weighted_pipeline_value = pipeline_value
        
        # Get forecast commitments (in a real system, this would come from forecast data)
        # For now, we'll simulate forecast data based on pipeline
        forecast_commit = self._simulate_forecast_commit(period_deals, period)
        forecast_upside = self._simulate_forecast_upside(period_deals, period)
        
        # Calculate variance
        variance_amount = weighted_pipeline_value - forecast_commit
        variance_percentage = (variance_amount / forecast_commit * 100) if forecast_commit > 0 else 0
        
        # Determine alignment status
        variance_threshold = config.get('variance_threshold_percentage', 15)
        high_variance_threshold = config.get('high_variance_threshold_percentage', 25)
        
        if abs(variance_percentage) <= variance_threshold:
            alignment_status = 'ALIGNED'
        elif abs(variance_percentage) <= high_variance_threshold:
            alignment_status = 'MODERATE_VARIANCE'
        else:
            alignment_status = 'HIGH_VARIANCE'
        
        return {
            'period': period,
            'deals_count': len(period_deals),
            'pipeline_value': round(pipeline_value, 2),
            'weighted_pipeline_value': round(weighted_pipeline_value, 2),
            'forecast_commit': round(forecast_commit, 2),
            'forecast_upside': round(forecast_upside, 2),
            'variance_amount': round(variance_amount, 2),
            'variance_percentage': round(variance_percentage, 1),
            'alignment_status': alignment_status
        }
    
    def _filter_deals_by_period(self, opportunities: List[Dict[str, Any]], period: str) -> List[Dict[str, Any]]:
        """Filter deals by forecast period"""
        
        current_date = datetime.now()
        period_deals = []
        
        for opp in opportunities:
            close_date_str = opp.get('CloseDate')
            if not close_date_str:
                continue
            
            try:
                close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
                
                if period == 'current_month':
                    if (close_date.year == current_date.year and 
                        close_date.month == current_date.month):
                        period_deals.append(opp)
                
                elif period == 'next_month':
                    next_month = current_date.replace(day=28) + timedelta(days=4)
                    next_month = next_month.replace(day=1)
                    if (close_date.year == next_month.year and 
                        close_date.month == next_month.month):
                        period_deals.append(opp)
                
                elif period == 'current_quarter':
                    current_quarter = (current_date.month - 1) // 3 + 1
                    close_quarter = (close_date.month - 1) // 3 + 1
                    if (close_date.year == current_date.year and 
                        close_quarter == current_quarter):
                        period_deals.append(opp)
                
                elif period == 'next_quarter':
                    # Simplified next quarter calculation
                    if close_date > current_date and close_date <= current_date + timedelta(days=90):
                        period_deals.append(opp)
                        
            except (ValueError, AttributeError):
                continue
        
        return period_deals
    
    def _simulate_forecast_commit(self, deals: List[Dict[str, Any]], period: str) -> float:
        """Simulate forecast commit (in real system, this would come from actual forecast data)"""
        
        # Simulate conservative forecast commit (70-80% of weighted pipeline)
        total_weighted = 0
        for deal in deals:
            amount = self._safe_float(deal.get('Amount'))
            probability = self._safe_float(deal.get('Probability')) / 100
            total_weighted += amount * probability
        
        # Apply conservative factor based on period
        if period == 'current_month':
            commit_factor = 0.85  # Higher confidence for current month
        elif period == 'current_quarter':
            commit_factor = 0.75  # Moderate confidence for quarter
        else:
            commit_factor = 0.70  # Lower confidence for future periods
        
        return total_weighted * commit_factor
    
    def _simulate_forecast_upside(self, deals: List[Dict[str, Any]], period: str) -> float:
        """Simulate forecast upside (in real system, this would come from actual forecast data)"""
        
        # Simulate upside as additional 20-30% of commit
        commit = self._simulate_forecast_commit(deals, period)
        upside_factor = 0.25  # 25% upside
        
        return commit * upside_factor
    
    def _identify_variance_issues(self, alignment_analysis: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific variance issues"""
        
        variance_issues = []
        variance_threshold = config.get('variance_threshold_percentage', 15)
        high_variance_threshold = config.get('high_variance_threshold_percentage', 25)
        
        for period, analysis in alignment_analysis.items():
            if analysis['alignment_status'] in ['MODERATE_VARIANCE', 'HIGH_VARIANCE']:
                
                severity = 'HIGH' if analysis['alignment_status'] == 'HIGH_VARIANCE' else 'MEDIUM'
                
                variance_issues.append({
                    'period': period,
                    'forecast_period': period.replace('_', ' ').title(),
                    'variance_amount': analysis['variance_amount'],
                    'variance_percentage': analysis['variance_percentage'],
                    'pipeline_value': analysis['weighted_pipeline_value'],
                    'forecast_commit': analysis['forecast_commit'],
                    'deals_count': analysis['deals_count'],
                    'severity': severity,
                    'risk_level': severity,
                    'issue_type': 'FORECAST_VARIANCE',
                    'priority': 'HIGH' if severity == 'HIGH' else 'MEDIUM',
                    'description': self._get_variance_description(analysis),
                    'recommended_action': self._get_variance_recommendation(analysis, severity),
                    'analysis_timestamp': datetime.now().isoformat()
                })
        
        return variance_issues
    
    def _get_variance_description(self, analysis: Dict[str, Any]) -> str:
        """Get variance description"""
        
        variance_pct = analysis['variance_percentage']
        period = analysis['period'].replace('_', ' ')
        
        if variance_pct > 0:
            return f"Pipeline exceeds forecast by {abs(variance_pct):.1f}% for {period}"
        else:
            return f"Pipeline falls short of forecast by {abs(variance_pct):.1f}% for {period}"
    
    def _get_variance_recommendation(self, analysis: Dict[str, Any], severity: str) -> str:
        """Get variance recommendation"""
        
        variance_pct = analysis['variance_percentage']
        
        if severity == 'HIGH':
            if variance_pct > 0:
                return "URGENT: Significant pipeline overage - review forecast accuracy and deal qualification"
            else:
                return "URGENT: Significant pipeline shortfall - immediate action needed to close gap"
        else:
            if variance_pct > 0:
                return "Review forecast methodology and deal probability assessments"
            else:
                return "Identify additional opportunities to meet forecast commitments"
    
    def _calculate_overall_alignment_metrics(self, alignment_analysis: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall alignment metrics"""
        
        total_periods = len(alignment_analysis)
        aligned_periods = len([a for a in alignment_analysis.values() if a['alignment_status'] == 'ALIGNED'])
        
        total_pipeline = sum(a['weighted_pipeline_value'] for a in alignment_analysis.values())
        total_forecast = sum(a['forecast_commit'] for a in alignment_analysis.values())
        
        overall_variance_pct = ((total_pipeline - total_forecast) / total_forecast * 100) if total_forecast > 0 else 0
        
        alignment_percentage = (aligned_periods / total_periods * 100) if total_periods > 0 else 0
        
        return {
            'total_periods_analyzed': total_periods,
            'aligned_periods': aligned_periods,
            'alignment_percentage': round(alignment_percentage, 1),
            'overall_pipeline_value': round(total_pipeline, 2),
            'overall_forecast_commit': round(total_forecast, 2),
            'overall_variance_amount': round(total_pipeline - total_forecast, 2),
            'overall_variance_percentage': round(overall_variance_pct, 1),
            'forecast_accuracy_score': max(0, 100 - abs(overall_variance_pct))
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
