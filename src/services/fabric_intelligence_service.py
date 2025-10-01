#!/usr/bin/env python3
"""
Azure Fabric Intelligence Service
=================================

Integrates with your existing Azure Fabric Salesforce data warehouse to provide
intelligent insights using the same data patterns as your Pipeline and Forecast modules.

Matches the pattern from crenovent-backend/services/salesforceTokenService.js
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncpg
import os

logger = logging.getLogger(__name__)

class FabricIntelligenceService:
    """
    Intelligence service that uses your existing Azure Fabric Salesforce data
    Matches the pattern from your backend's tryExecuteAzureFabricQuery function
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Azure Fabric SQL configuration (if available)
        self.fabric_sql_server = os.getenv('FABRIC_SQL_SERVER')
        self.fabric_sql_database = os.getenv('FABRIC_SQL_DATABASE')
        self.has_fabric_sql = bool(self.fabric_sql_server and self.fabric_sql_database)
        
        self.logger.info(f"🏭 Fabric Intelligence Service initialized - Fabric SQL: {self.has_fabric_sql}")
    
    async def get_user_opportunities(self, user_id: int, tenant_id: int) -> List[Dict[str, Any]]:
        """
        Get opportunities for a user using your exact Salesforce data pattern
        Matches the pattern from crenovent-backend/controller/pipeline/index.js fetchPipelineOpportunities
        """
        try:
            # Use Azure Fabric SQL connection (matches your backend exactly)
            fabric_service = self.pool_manager.fabric_service
            if not fabric_service:
                self.logger.error("❌ Azure Fabric service not available")
                return []
            
            # Use the exact same query pattern as Node.js backend buildFabricOpportunityQuery
            # IMPORTANT: Node.js backend fetches ALL opportunities (no OwnerId filter) then filters in memory
            opportunity_query = """
            SELECT 
                [OpportunityId] as id,
                [OpportunityName] as name,
                [AccountId] as account_id,
                [AccountName] as account_name,
                [StageName] as stage_name,
                [Amount] as amount,
                [Probability] as probability,
                [CloseDate] as close_date,
                [CreatedDate] as created_date,
                [LastModifiedDate] as last_modified_date,
                [IsClosed] as is_closed,
                [IsWon] as is_won,
                [OwnerId] as owner_id
            FROM dbo.opportunities 
            WHERE [Amount] > 0 
            ORDER BY [CreatedDate] DESC
            OFFSET 0 ROWS FETCH NEXT 500 ROWS ONLY
            """
            
            # Execute using Fabric SQL (matches Node.js backend pattern)
            fabric_result = await fabric_service.execute_query(opportunity_query, [])
            
            all_opportunities = []
            if fabric_result.success and fabric_result.data:
                self.logger.info(f"📊 Retrieved {len(fabric_result.data)} ALL opportunities from Azure Fabric")
                for row in fabric_result.data:
                    all_opportunities.append({
                        'id': row['id'],
                        'name': row['name'],
                        'accountId': row['account_id'],
                        'accountName': row['account_name'],
                        'stageName': row['stage_name'],
                        'amount': float(row['amount']) if row['amount'] else 0,
                        'probability': int(row['probability']) if row['probability'] else 0,
                        'closeDate': str(row['close_date']) if row['close_date'] else None,
                        'createdDate': str(row['created_date']) if row['created_date'] else None,
                        'lastModifiedDate': str(row['last_modified_date']) if row['last_modified_date'] else None,
                        'isClosed': row['is_closed'],
                        'isWon': row['is_won'],
                        'ownerId': row['owner_id'],
                        # Intelligence-specific fields
                        'daysInStage': row.get('days_in_stage', 0),
                        'forecastCategory': row.get('forecast_category'),
                        'leadSource': row.get('lead_source')
                    })
                
                # Filter opportunities for this specific user (matches Node.js backend pattern)
                user_opportunities = [opp for opp in all_opportunities if str(opp['ownerId']) == str(user_id)]
                self.logger.info(f"📊 Filtered {len(user_opportunities)} opportunities for user {user_id} from {len(all_opportunities)} total")
                return user_opportunities
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching user opportunities: {e}")
            return []
    
    async def get_team_opportunities(self, user_ids: List[int], tenant_id: int) -> List[Dict[str, Any]]:
        """
        Get opportunities for a team (hierarchy-aware)
        Matches your backend's team data aggregation pattern
        """
        try:
            # Use Azure Fabric SQL connection (matches your backend exactly)
            fabric_service = self.pool_manager.fabric_service
            if not fabric_service:
                self.logger.error("❌ Azure Fabric service not available")
                return []
            
            # Convert user_ids to strings (matching your backend pattern)
            user_id_strings = [str(uid) for uid in user_ids]
            
            # Use same query as individual user - fetch ALL opportunities then filter
            # This matches the Node.js backend pattern exactly
            opportunity_query = """
            SELECT 
                [OpportunityId] as id,
                [OpportunityName] as name,
                [AccountId] as account_id,
                [AccountName] as account_name,
                [StageName] as stage_name,
                [Amount] as amount,
                [Probability] as probability,
                [CloseDate] as close_date,
                [CreatedDate] as created_date,
                [LastModifiedDate] as last_modified_date,
                [IsClosed] as is_closed,
                [IsWon] as is_won,
                [OwnerId] as owner_id
            FROM dbo.opportunities 
            WHERE [Amount] > 0 
            ORDER BY [CreatedDate] DESC
            OFFSET 0 ROWS FETCH NEXT 500 ROWS ONLY
            """
            
            # Execute using Fabric SQL
            fabric_result = await fabric_service.execute_query(opportunity_query, [])
            
            all_opportunities = []
            if fabric_result.success and fabric_result.data:
                self.logger.info(f"📊 Retrieved {len(fabric_result.data)} ALL opportunities from Azure Fabric")
                for row in fabric_result.data:
                    all_opportunities.append({
                        'id': row['id'],
                        'name': row['name'],
                        'accountId': row['account_id'],
                        'accountName': row['account_name'],
                        'stageName': row['stage_name'],
                        'amount': float(row['amount']) if row['amount'] else 0,
                        'probability': int(row['probability']) if row['probability'] else 0,
                        'closeDate': str(row['close_date']) if row['close_date'] else None,
                        'createdDate': str(row['created_date']) if row['created_date'] else None,
                        'lastModifiedDate': str(row['last_modified_date']) if row['last_modified_date'] else None,
                        'isClosed': row['is_closed'],
                        'isWon': row['is_won'],
                        'ownerId': row['owner_id'],
                        'daysInStage': row.get('days_in_stage', 0),
                        'forecastCategory': row.get('forecast_category'),
                        'leadSource': row.get('lead_source')
                    })
                
                # Filter opportunities for the team users (matches Node.js backend pattern)
                team_opportunities = [opp for opp in all_opportunities if str(opp['ownerId']) in user_id_strings]
                self.logger.info(f"📊 Filtered {len(team_opportunities)} team opportunities for {len(user_ids)} users from {len(all_opportunities)} total")
                return team_opportunities
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching team opportunities: {e}")
            return []
    
    async def calculate_intelligence_metrics(self, opportunities: List[Dict[str, Any]], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate intelligence metrics from opportunities data
        This is where we generate insights for the knowledge graph
        """
        try:
            if not opportunities:
                return self._empty_metrics()
            
            # Basic pipeline metrics
            total_pipeline = sum(opp['amount'] for opp in opportunities)
            total_opportunities = len(opportunities)
            
            # Stage analysis
            stage_distribution = {}
            stage_amounts = {}
            
            for opp in opportunities:
                stage = opp.get('stageName', 'Unknown')
                stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
                stage_amounts[stage] = stage_amounts.get(stage, 0) + opp['amount']
            
            # Risk analysis (opportunities stuck in stage)
            high_risk_opportunities = []
            for opp in opportunities:
                days_in_stage = opp.get('daysInStage', 0)
                if days_in_stage > 30:  # Configurable threshold
                    high_risk_opportunities.append({
                        'id': opp['id'],
                        'name': opp['name'],
                        'stage': opp['stageName'],
                        'daysInStage': days_in_stage,
                        'amount': opp['amount'],
                        'riskLevel': 'high' if days_in_stage > 60 else 'medium'
                    })
            
            # Forecast category analysis
            forecast_categories = {}
            for opp in opportunities:
                category = opp.get('forecastCategory', 'Unknown')
                forecast_categories[category] = forecast_categories.get(category, 0) + opp['amount']
            
            # Intelligence recommendations
            recommendations = self._generate_recommendations(opportunities, high_risk_opportunities)
            
            # Trust score calculation (based on data quality and patterns)
            trust_score = self._calculate_trust_score(opportunities)
            
            metrics = {
                'pipeline_metrics': {
                    'total_pipeline_value': total_pipeline,
                    'total_opportunities': total_opportunities,
                    'average_deal_size': total_pipeline / total_opportunities if total_opportunities > 0 else 0,
                    'stage_distribution': stage_distribution,
                    'stage_amounts': stage_amounts
                },
                'risk_analysis': {
                    'high_risk_count': len(high_risk_opportunities),
                    'high_risk_value': sum(opp['amount'] for opp in high_risk_opportunities),
                    'high_risk_opportunities': high_risk_opportunities[:10]  # Top 10
                },
                'forecast_analysis': {
                    'categories': forecast_categories,
                    'commit_pipeline': forecast_categories.get('Commit', 0),
                    'best_case_pipeline': forecast_categories.get('Best Case', 0)
                },
                'intelligence': {
                    'trust_score': trust_score,
                    'recommendations': recommendations,
                    'data_quality_score': self._calculate_data_quality(opportunities)
                },
                'metadata': {
                    'calculated_at': datetime.utcnow().isoformat(),
                    'user_id': user_context.get('user_id'),
                    'tenant_id': user_context.get('tenant_id'),
                    'data_source': 'azure_fabric'
                }
            }
            
            # Log this calculation to knowledge graph
            await self._log_to_knowledge_graph(metrics, user_context)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating intelligence metrics: {e}")
            return self._empty_metrics()
    
    def _generate_recommendations(self, opportunities: List[Dict], high_risk_opps: List[Dict]) -> List[Dict[str, Any]]:
        """Generate intelligent recommendations based on data patterns"""
        recommendations = []
        
        # Recommendation 1: Focus on high-risk opportunities
        if high_risk_opps:
            recommendations.append({
                'type': 'risk_mitigation',
                'priority': 'high',
                'title': f'Focus on {len(high_risk_opps)} High-Risk Opportunities',
                'description': f'You have {len(high_risk_opps)} opportunities that have been in their current stage for over 30 days.',
                'action': 'Review and update these opportunities to move them forward',
                'impact': 'pipeline_health',
                'opportunities': [opp['id'] for opp in high_risk_opps[:5]]
            })
        
        # Recommendation 2: Stage distribution analysis
        stage_counts = {}
        for opp in opportunities:
            stage = opp.get('stageName', 'Unknown')
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        if stage_counts.get('Prospecting', 0) > len(opportunities) * 0.6:
            recommendations.append({
                'type': 'pipeline_balance',
                'priority': 'medium',
                'title': 'Too Many Opportunities in Prospecting',
                'description': 'Over 60% of your opportunities are still in Prospecting stage.',
                'action': 'Focus on qualifying and moving prospects to the next stage',
                'impact': 'conversion_rate'
            })
        
        return recommendations
    
    def _calculate_trust_score(self, opportunities: List[Dict]) -> float:
        """Calculate trust score based on data quality and patterns"""
        if not opportunities:
            return 0.0
        
        score_factors = []
        
        # Factor 1: Data completeness
        complete_records = sum(1 for opp in opportunities 
                             if all(opp.get(field) for field in ['amount', 'closeDate', 'stageName']))
        completeness_score = complete_records / len(opportunities)
        score_factors.append(completeness_score * 0.3)
        
        # Factor 2: Realistic amounts (not all round numbers)
        non_round_amounts = sum(1 for opp in opportunities 
                               if opp['amount'] % 1000 != 0)
        realism_score = non_round_amounts / len(opportunities) if len(opportunities) > 0 else 0
        score_factors.append(realism_score * 0.2)
        
        # Factor 3: Stage progression (not too many stuck in early stages)
        early_stage_count = sum(1 for opp in opportunities 
                               if opp.get('stageName', '').lower() in ['prospecting', 'qualification'])
        progression_score = 1 - (early_stage_count / len(opportunities))
        score_factors.append(progression_score * 0.3)
        
        # Factor 4: Recent activity (recently modified)
        recent_activity = sum(1 for opp in opportunities 
                             if opp.get('lastModifiedDate') and 
                             datetime.fromisoformat(opp['lastModifiedDate'].replace('Z', '+00:00')) > 
                             datetime.now().replace(tzinfo=None) - timedelta(days=7))
        activity_score = recent_activity / len(opportunities) if len(opportunities) > 0 else 0
        score_factors.append(activity_score * 0.2)
        
        # Calculate final trust score (0-100)
        final_score = sum(score_factors) * 100
        return round(final_score, 2)
    
    def _calculate_data_quality(self, opportunities: List[Dict]) -> float:
        """Calculate data quality score"""
        if not opportunities:
            return 0.0
        
        required_fields = ['name', 'amount', 'stageName', 'closeDate', 'ownerId']
        quality_scores = []
        
        for opp in opportunities:
            field_completeness = sum(1 for field in required_fields if opp.get(field)) / len(required_fields)
            quality_scores.append(field_completeness)
        
        return round(sum(quality_scores) / len(quality_scores) * 100, 2)
    
    async def _log_to_knowledge_graph(self, metrics: Dict[str, Any], user_context: Dict[str, Any]):
        """
        Log intelligence calculation to knowledge graph for future learning
        This implements the Build Plan's knowledge flywheel concept
        """
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                # Insert into dsl_execution_traces for knowledge graph
                trace_query = """
                INSERT INTO dsl_execution_traces (
                    workflow_id, tenant_id, user_id, execution_type,
                    input_data, output_data, execution_status,
                    started_at, completed_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """
                
                workflow_id = f"intelligence_calculation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                
                await conn.execute(
                    trace_query,
                    workflow_id,
                    user_context.get('tenant_id'),
                    user_context.get('user_id'),
                    'intelligence_calculation',
                    {'opportunities_count': metrics['pipeline_metrics']['total_opportunities']},
                    metrics,
                    'completed',
                    datetime.utcnow(),
                    datetime.utcnow(),
                    {
                        'source': 'fabric_intelligence_service',
                        'trust_score': metrics['intelligence']['trust_score'],
                        'recommendations_count': len(metrics['intelligence']['recommendations'])
                    }
                )
                
                self.logger.info(f"📊 Logged intelligence calculation to knowledge graph: {workflow_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Error logging to knowledge graph: {e}")
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'pipeline_metrics': {
                'total_pipeline_value': 0,
                'total_opportunities': 0,
                'average_deal_size': 0,
                'stage_distribution': {},
                'stage_amounts': {}
            },
            'risk_analysis': {
                'high_risk_count': 0,
                'high_risk_value': 0,
                'high_risk_opportunities': []
            },
            'forecast_analysis': {
                'categories': {},
                'commit_pipeline': 0,
                'best_case_pipeline': 0
            },
            'intelligence': {
                'trust_score': 0.0,
                'recommendations': [],
                'data_quality_score': 0.0
            },
            'metadata': {
                'calculated_at': datetime.utcnow().isoformat(),
                'data_source': 'azure_fabric',
                'status': 'no_data'
            }
        }
