"""
Data Source Manager for Comprehensive Sandbagging Detection
Handles integration with CRM, historical data, external data, and calculated metrics
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class DataAvailability:
    """Track what data sources are available for parameter calculation"""
    has_crm_data: bool = True
    has_historical_data: bool = False
    has_external_data: bool = False
    has_calculated_metrics: bool = True
    
    # Specific data availability flags
    has_rep_history: bool = False
    has_market_data: bool = False
    has_stakeholder_data: bool = False
    has_activity_history: bool = False
    has_competitive_data: bool = False

class DataSourceManager:
    """
    Unified data source manager for comprehensive sandbagging detection
    Handles all data integration and availability checking
    """
    
    def __init__(self):
        self.data_availability = DataAvailability()
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize and check availability of all data sources"""
        logger.info("🔍 Initializing data source availability...")
        
        # Check CRM data availability (always available from CSV)
        self.data_availability.has_crm_data = True
        
        # Check historical data availability
        self.data_availability.has_historical_data = self._check_historical_data()
        
        # Check external data availability  
        self.data_availability.has_external_data = self._check_external_data()
        
        # Check specific data types
        self.data_availability.has_rep_history = self._check_rep_history()
        self.data_availability.has_activity_history = self._check_activity_history()
        
        logger.info(f"📊 Data availability: {self._get_availability_summary()}")
    
    def _check_historical_data(self) -> bool:
        """Check if historical CRM data is available"""
        try:
            # In a real implementation, this would check database connections
            # For now, simulate availability based on configuration
            return True  # Simulated - would check actual data sources
        except Exception as e:
            logger.warning(f"Historical data unavailable: {e}")
            return False
    
    def _check_external_data(self) -> bool:
        """Check if external market data sources are available"""
        try:
            # In a real implementation, this would check API connections
            # For now, simulate limited availability
            return False  # Simulated - would check external APIs
        except Exception as e:
            logger.warning(f"External data unavailable: {e}")
            return False
    
    def _check_rep_history(self) -> bool:
        """Check if rep historical performance data is available"""
        # Simulated check - in real implementation would query historical tables
        return True
    
    def _check_activity_history(self) -> bool:
        """Check if activity history data is available"""
        # Simulated check - in real implementation would query activity tables
        return True
    
    def _get_availability_summary(self) -> str:
        """Get summary of data availability"""
        available = []
        if self.data_availability.has_crm_data:
            available.append("CRM")
        if self.data_availability.has_historical_data:
            available.append("Historical")
        if self.data_availability.has_external_data:
            available.append("External")
        if self.data_availability.has_calculated_metrics:
            available.append("Calculated")
        
        return f"{len(available)} sources available: {', '.join(available)}"
    
    async def get_rep_historical_data(self, rep_id: str, months_back: int = 12) -> Dict[str, Any]:
        """Get historical performance data for a sales rep"""
        if not self.data_availability.has_rep_history:
            return {}
        
        try:
            # Simulated historical data - in real implementation would query database
            historical_data = {
                'forecast_accuracy': 0.75 + (hash(rep_id) % 25) / 100,  # 0.75-1.00
                'sandbagging_tendency': (hash(rep_id) % 50) / 100,  # 0.00-0.50
                'avg_deal_velocity': 45 + (hash(rep_id) % 30),  # 45-75 days
                'quarter_end_surge_pattern': (hash(rep_id) % 3) / 10 + 0.1,  # 0.1-0.4
                'probability_update_frequency': 2 + (hash(rep_id) % 4),  # 2-6 per month
                'pipeline_coverage_ratio': 3.0 + (hash(rep_id) % 20) / 10,  # 3.0-5.0
                'quota_achievement_rate': 0.8 + (hash(rep_id) % 40) / 100,  # 0.8-1.2
                'last_updated': datetime.now().isoformat()
            }
            
            logger.debug(f"📈 Retrieved historical data for rep {rep_id}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to get rep historical data: {e}")
            return {}
    
    async def get_market_data(self, industry: str, region: str = "global") -> Dict[str, Any]:
        """Get external market intelligence data"""
        if not self.data_availability.has_external_data:
            return {}
        
        try:
            # Simulated market data - in real implementation would call external APIs
            market_data = {
                'industry_growth_rate': 0.15 if industry == "Technology" else 0.08,
                'competitive_intensity': 0.7 + (hash(industry) % 3) / 10,  # 0.7-1.0
                'market_maturity': 0.6 + (hash(industry) % 4) / 10,  # 0.6-1.0
                'economic_indicator': 0.85 + (hash(region) % 15) / 100,  # 0.85-1.00
                'regulatory_complexity': 0.3 + (hash(industry) % 7) / 10,  # 0.3-1.0
                'last_updated': datetime.now().isoformat()
            }
            
            logger.debug(f"🌍 Retrieved market data for {industry} in {region}")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return {}
    
    async def calculate_derived_metrics(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate complex metrics from base deal data"""
        try:
            calculated_metrics = {}
            
            # Deal age calculation
            if 'CreatedDate' in deal_data:
                created_date = pd.to_datetime(deal_data['CreatedDate'])
                deal_age_days = (datetime.now() - created_date).days
                calculated_metrics['deal_age_days'] = deal_age_days
            
            # Activity recency calculation
            if 'ActivityDate' in deal_data:
                activity_date = pd.to_datetime(deal_data['ActivityDate'])
                days_since_activity = (datetime.now() - activity_date).days
                calculated_metrics['days_since_activity'] = days_since_activity
            
            # Stage-probability alignment score
            if 'StageName' in deal_data and 'Probability' in deal_data:
                stage_prob_alignment = self._calculate_stage_probability_alignment(
                    deal_data['StageName'], deal_data['Probability']
                )
                calculated_metrics['stage_probability_alignment'] = stage_prob_alignment
            
            # Deal complexity score
            complexity_score = self._calculate_deal_complexity(deal_data)
            calculated_metrics['deal_complexity_score'] = complexity_score
            
            # Customer engagement score
            engagement_score = self._calculate_customer_engagement(deal_data)
            calculated_metrics['customer_engagement_score'] = engagement_score
            
            # Momentum velocity score
            momentum_score = self._calculate_deal_momentum(deal_data)
            calculated_metrics['deal_momentum_score'] = momentum_score
            
            logger.debug(f"🧮 Calculated {len(calculated_metrics)} derived metrics")
            return calculated_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate derived metrics: {e}")
            return {}
    
    def _calculate_stage_probability_alignment(self, stage: str, probability: float) -> float:
        """Calculate alignment between deal stage and probability"""
        stage_expectations = {
            'Prospecting': 15,
            'Qualification': 25,
            'Needs Analysis': 35,
            'Value Proposition': 45,
            'Id. Decision Makers': 55,
            'Perception Analysis': 65,
            'Proposal/Price Quote': 75,
            'Negotiation/Review': 85,
            'Closed Won': 100,
            'Closed Lost': 0
        }
        
        expected_prob = stage_expectations.get(stage, 50)
        alignment = 1.0 - abs(expected_prob - probability) / 100
        return max(0.0, alignment)
    
    def _calculate_deal_complexity(self, deal_data: Dict[str, Any]) -> float:
        """Calculate deal complexity score"""
        complexity = 0.5  # Base complexity
        
        # Amount-based complexity
        amount = deal_data.get('Amount', 0)
        if amount > 1000000:
            complexity += 0.3
        elif amount > 500000:
            complexity += 0.2
        elif amount > 100000:
            complexity += 0.1
        
        # Stage-based complexity
        stage = deal_data.get('StageName', '')
        if stage in ['Negotiation/Review', 'Proposal/Price Quote']:
            complexity += 0.2
        
        return min(1.0, complexity)
    
    def _calculate_customer_engagement(self, deal_data: Dict[str, Any]) -> float:
        """Calculate customer engagement score"""
        engagement = 0.5  # Base engagement
        
        # Activity recency boost
        if 'ActivityDate' in deal_data:
            activity_date = pd.to_datetime(deal_data['ActivityDate'])
            days_since = (datetime.now() - activity_date).days
            if days_since <= 7:
                engagement += 0.3
            elif days_since <= 14:
                engagement += 0.2
            elif days_since <= 30:
                engagement += 0.1
        
        # Stage progression boost
        stage = deal_data.get('StageName', '')
        if stage in ['Proposal/Price Quote', 'Negotiation/Review']:
            engagement += 0.2
        
        return min(1.0, engagement)
    
    def _calculate_deal_momentum(self, deal_data: Dict[str, Any]) -> float:
        """Calculate deal momentum score"""
        momentum = 0.5  # Base momentum
        
        # Probability vs stage momentum
        stage_prob_alignment = self._calculate_stage_probability_alignment(
            deal_data.get('StageName', ''), deal_data.get('Probability', 0)
        )
        momentum += stage_prob_alignment * 0.3
        
        # Recent activity momentum
        engagement = self._calculate_customer_engagement(deal_data)
        momentum += engagement * 0.2
        
        return min(1.0, momentum)
    
    async def get_contextual_data_for_deal(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get all available contextual data for a specific deal"""
        contextual_data = {}
        
        # Get rep historical data
        if 'OwnerId' in deal_data:
            rep_data = await self.get_rep_historical_data(deal_data['OwnerId'])
            contextual_data['rep_history'] = rep_data
        
        # Get market data
        if 'Industry' in deal_data:
            market_data = await self.get_market_data(deal_data['Industry'])
            contextual_data['market_data'] = market_data
        
        # Get calculated metrics
        calculated_data = await self.calculate_derived_metrics(deal_data)
        contextual_data['calculated_metrics'] = calculated_data
        
        # Add data availability info
        contextual_data['data_availability'] = {
            'has_rep_history': bool(contextual_data.get('rep_history')),
            'has_market_data': bool(contextual_data.get('market_data')),
            'has_calculated_metrics': bool(contextual_data.get('calculated_metrics'))
        }
        
        return contextual_data
    
    def get_available_parameter_sources(self) -> List[str]:
        """Get list of available parameter data sources"""
        sources = []
        if self.data_availability.has_crm_data:
            sources.append('crm')
        if self.data_availability.has_historical_data:
            sources.append('historical')
        if self.data_availability.has_external_data:
            sources.append('external')
        if self.data_availability.has_calculated_metrics:
            sources.append('calculated')
        return sources

# Global instance
data_source_manager = DataSourceManager()
