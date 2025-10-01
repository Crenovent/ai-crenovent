"""
Coverage Analysis RBA Agent
Single-purpose, focused RBA agent for pipeline coverage analysis only

This agent ONLY handles pipeline coverage ratio and gap analysis.
Lightweight, modular, configuration-driven.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

from .enhanced_base_rba_agent import EnhancedBaseRBAAgent

logger = logging.getLogger(__name__)

class CoverageAnalysisRBAAgent(EnhancedBaseRBAAgent):
    """
    Single-purpose RBA agent for coverage analysis
    
    Features:
    - ONLY handles pipeline coverage analysis
    - Configuration-driven coverage targets
    - Lightweight and focused
    - Quota vs pipeline comparison
    - Clear separation of concerns
    """
    
    # Agent metadata for dynamic discovery
    AGENT_TYPE = "RBA"
    AGENT_NAME = "coverage_analysis"
    AGENT_DESCRIPTION = "Analyze pipeline coverage ratios and identify coverage gaps"
    SUPPORTED_ANALYSIS_TYPES = ["coverage_analysis", "pipeline_coverage"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # Coverage analysis specific defaults
        self.default_config = {
            'target_coverage_ratio': 3.0,  # 3x coverage of quota
            'minimum_coverage_ratio': 2.0,  # Minimum acceptable coverage
            'quota_periods': ['current_quarter', 'current_month'],
            'include_probability_weighting': True,
            'coverage_gap_threshold': 0.5  # 50% gap threshold
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coverage analysis"""
        
        try:
            opportunities = input_data.get('opportunities', [])
            config = {**self.default_config, **input_data.get('config', {})}
            
            logger.info(f"🎯 Coverage Analysis RBA: Analyzing {len(opportunities)} opportunities")
            
            # Filter for open deals only
            open_deals = [
                opp for opp in opportunities 
                if 'closed' not in opp.get('StageName', '').lower()
            ]
            
            # Analyze coverage for different periods
            coverage_analysis = {}
            quota_periods = config.get('quota_periods', ['current_quarter'])
            
            for period in quota_periods:
                period_analysis = self._analyze_coverage_period(open_deals, period, config)
                coverage_analysis[period] = period_analysis
            
            # Identify coverage gaps
            coverage_gaps = self._identify_coverage_gaps(coverage_analysis, config)
            
            # Generate owner-level coverage analysis
            owner_coverage = self._analyze_owner_coverage(open_deals, config)
            
            # Calculate overall coverage metrics
            overall_metrics = self._calculate_overall_coverage_metrics(coverage_analysis, config)
            
            return {
                'success': True,
                'agent_name': self.AGENT_NAME,
                'analysis_type': 'coverage_analysis',
                'total_opportunities': len(opportunities),
                'open_deals_analyzed': len(open_deals),
                'flagged_opportunities': len(coverage_gaps),
                'coverage_analysis': coverage_analysis,
                'coverage_gaps': coverage_gaps,
                'owner_coverage': owner_coverage,
                'overall_metrics': overall_metrics,
                'configuration_used': config,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Coverage Analysis RBA failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.AGENT_NAME,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _analyze_coverage_period(self, opportunities: List[Dict[str, Any]], period: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coverage for a specific period"""
        
        # Filter opportunities for the period
        period_deals = self._filter_deals_by_period(opportunities, period)
        
        if not period_deals:
            return {
                'period': period,
                'deals_count': 0,
                'pipeline_value': 0,
                'weighted_pipeline_value': 0,
                'quota_target': 0,
                'coverage_ratio': 0,
                'coverage_gap': 0,
                'coverage_status': 'NO_DATA'
            }
        
        # Calculate pipeline metrics
        pipeline_value = sum(self._safe_float(deal.get('Amount')) for deal in period_deals)
        
        weighted_pipeline_value = pipeline_value
        if config.get('include_probability_weighting', True):
            weighted_pipeline_value = 0
            for deal in period_deals:
                if deal is None:
                    continue  # Skip None deals
                amount = self._safe_float(deal.get('Amount'))
                probability = self._safe_float(deal.get('Probability')) / 100
                weighted_pipeline_value += amount * probability
        
        # Get quota target (in a real system, this would come from quota/target data)
        quota_target = self._simulate_quota_target(period_deals, period)
        
        # Calculate coverage metrics
        coverage_ratio = (weighted_pipeline_value / quota_target) if quota_target > 0 else 0
        coverage_gap = max(0, quota_target - weighted_pipeline_value)
        
        # Determine coverage status
        target_coverage = config.get('target_coverage_ratio', 3.0)
        minimum_coverage = config.get('minimum_coverage_ratio', 2.0)
        
        if coverage_ratio >= target_coverage:
            coverage_status = 'EXCELLENT'
        elif coverage_ratio >= minimum_coverage:
            coverage_status = 'ADEQUATE'
        elif coverage_ratio >= 1.0:
            coverage_status = 'MINIMAL'
        else:
            coverage_status = 'INSUFFICIENT'
        
        return {
            'period': period,
            'deals_count': len(period_deals),
            'pipeline_value': round(pipeline_value, 2),
            'weighted_pipeline_value': round(weighted_pipeline_value, 2),
            'quota_target': round(quota_target, 2),
            'coverage_ratio': round(coverage_ratio, 2),
            'coverage_gap': round(coverage_gap, 2),
            'coverage_status': coverage_status,
            'coverage_percentage': round(coverage_ratio * 100, 1)
        }
    
    def _filter_deals_by_period(self, opportunities: List[Dict[str, Any]], period: str) -> List[Dict[str, Any]]:
        """Filter deals by period"""
        
        current_date = datetime.now()
        period_deals = []
        
        for opp in opportunities:

        
            if opp is None:

        
                continue  # Skip None opportunities
            close_date_str = opp.get('CloseDate')
            if not close_date_str:
                continue
            
            try:
                close_date = datetime.strptime(close_date_str.split('T')[0], '%Y-%m-%d')
                
                if period == 'current_month':
                    if (close_date.year == current_date.year and 
                        close_date.month == current_date.month):
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
    
    def _simulate_quota_target(self, deals: List[Dict[str, Any]], period: str) -> float:
        """Simulate quota target (in real system, this would come from actual quota data)"""
        
        # Simulate quota based on pipeline size and period
        total_pipeline = sum(self._safe_float(deal.get('Amount')) for deal in deals)
        
        # Apply period-based quota factors
        if period == 'current_month':
            quota_factor = 0.4  # Monthly quota is typically lower
        elif period == 'current_quarter':
            quota_factor = 0.6  # Quarterly quota
        else:
            quota_factor = 0.5  # Default
        
        # Simulate quota as percentage of total pipeline
        return total_pipeline * quota_factor
    
    def _identify_coverage_gaps(self, coverage_analysis: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify coverage gaps that need attention"""
        
        coverage_gaps = []
        minimum_coverage = config.get('minimum_coverage_ratio', 2.0)
        gap_threshold = config.get('coverage_gap_threshold', 0.5)
        
        for period, analysis in coverage_analysis.items():
            if analysis['coverage_status'] in ['INSUFFICIENT', 'MINIMAL']:
                
                severity = 'HIGH' if analysis['coverage_status'] == 'INSUFFICIENT' else 'MEDIUM'
                gap_percentage = (analysis['coverage_gap'] / analysis['quota_target'] * 100) if analysis['quota_target'] > 0 else 0
                
                if gap_percentage >= gap_threshold * 100:  # Convert to percentage
                    coverage_gaps.append({
                        'period': period,
                        'period_name': period.replace('_', ' ').title(),
                        'coverage_ratio': analysis['coverage_ratio'],
                        'coverage_gap': analysis['coverage_gap'],
                        'gap_percentage': round(gap_percentage, 1),
                        'pipeline_value': analysis['weighted_pipeline_value'],
                        'quota_target': analysis['quota_target'],
                        'deals_count': analysis['deals_count'],
                        'severity': severity,
                        'risk_level': severity,
                        'issue_type': 'COVERAGE_GAP',
                        'priority': 'HIGH' if severity == 'HIGH' else 'MEDIUM',
                        'description': self._get_gap_description(analysis),
                        'recommended_action': self._get_gap_recommendation(analysis, severity),
                        'analysis_timestamp': datetime.now().isoformat()
                    })
        
        return coverage_gaps
    
    def _get_gap_description(self, analysis: Dict[str, Any]) -> str:
        """Get coverage gap description"""
        
        period = analysis['period'].replace('_', ' ')
        coverage_ratio = analysis['coverage_ratio']
        gap_amount = analysis['coverage_gap']
        
        return f"Coverage gap of ${gap_amount:,.0f} for {period} (ratio: {coverage_ratio:.1f}x)"
    
    def _get_gap_recommendation(self, analysis: Dict[str, Any], severity: str) -> str:
        """Get coverage gap recommendation"""
        
        if severity == 'HIGH':
            return "URGENT: Significant coverage shortfall - immediate pipeline generation needed"
        else:
            return "Increase pipeline generation activities to improve coverage ratio"
    
    def _analyze_owner_coverage(self, opportunities: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coverage by owner"""
        
        owner_coverage = defaultdict(lambda: {
            'deals_count': 0,
            'pipeline_value': 0,
            'weighted_pipeline_value': 0,
            'estimated_quota': 0,
            'coverage_ratio': 0
        })
        
        # Group by owner
        for opp in opportunities:

            if opp is None:

                continue  # Skip None opportunities
            owner_name = (opp.get('Owner') or {}).get('Name', 'Unassigned')
            amount = self._safe_float(opp.get('Amount'))
            probability = self._safe_float(opp.get('Probability')) / 100
            
            owner_data = owner_coverage[owner_name]
            owner_data['deals_count'] += 1
            owner_data['pipeline_value'] += amount
            
            if config.get('include_probability_weighting', True):
                owner_data['weighted_pipeline_value'] += amount * probability
            else:
                owner_data['weighted_pipeline_value'] += amount
        
        # Calculate coverage ratios
        for owner_name, data in owner_coverage.items():
            # Simulate individual quota (in real system, would come from quota data)
            data['estimated_quota'] = data['pipeline_value'] * 0.6  # 60% of pipeline as quota
            data['coverage_ratio'] = (data['weighted_pipeline_value'] / data['estimated_quota']) if data['estimated_quota'] > 0 else 0
            
            # Round values
            data['pipeline_value'] = round(data['pipeline_value'], 2)
            data['weighted_pipeline_value'] = round(data['weighted_pipeline_value'], 2)
            data['estimated_quota'] = round(data['estimated_quota'], 2)
            data['coverage_ratio'] = round(data['coverage_ratio'], 2)
        
        return dict(owner_coverage)
    
    def _calculate_overall_coverage_metrics(self, coverage_analysis: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall coverage metrics"""
        
        total_pipeline = sum(a['weighted_pipeline_value'] for a in coverage_analysis.values())
        total_quota = sum(a['quota_target'] for a in coverage_analysis.values())
        
        overall_coverage_ratio = (total_pipeline / total_quota) if total_quota > 0 else 0
        overall_coverage_gap = max(0, total_quota - total_pipeline)
        
        # Count periods by status
        status_counts = defaultdict(int)
        for analysis in coverage_analysis.values():
            status_counts[analysis['coverage_status']] += 1
        
        target_coverage = config.get('target_coverage_ratio', 3.0)
        coverage_health_score = min(100, (overall_coverage_ratio / target_coverage) * 100)
        
        return {
            'overall_pipeline_value': round(total_pipeline, 2),
            'overall_quota_target': round(total_quota, 2),
            'overall_coverage_ratio': round(overall_coverage_ratio, 2),
            'overall_coverage_gap': round(overall_coverage_gap, 2),
            'coverage_health_score': round(coverage_health_score, 1),
            'periods_by_status': dict(status_counts),
            'target_coverage_ratio': target_coverage
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
