#!/usr/bin/env python3
"""
Knowledge Graph Query Engine
===========================

Provides business intelligence queries for the Knowledge Graph
as specified in Vision Doc Ch.8.2

Features:
- Persona-centric queries (CRO, CFO, Compliance, Analyst)
- Business intelligence analytics
- Governance reporting
- Natural language query support
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .ontology import EntityType, RelationshipType
from .kg_store import KnowledgeGraphStore

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Standard query result format"""
    query_type: str
    results: List[Dict[str, Any]]
    total_count: int
    execution_time_ms: int
    metadata: Dict[str, Any]

class KnowledgeGraphQuery:
    """
    Business intelligence query engine for Knowledge Graph
    
    Implements persona-centric queries from Vision Doc Ch.8.2:
    - CRO: Forecast workflows with variance >10% and overrides
    - CFO: Compensation workflows overridden by Finance  
    - Compliance: Workflows governed by SOX with failed SLA
    - Analyst: Override patterns across regions
    """
    
    def __init__(self, kg_store: KnowledgeGraphStore):
        self.kg_store = kg_store
    
    async def cro_forecast_variance_analysis(self, tenant_id: str, 
                                           variance_threshold: float = 0.10,
                                           time_period_days: int = 90) -> QueryResult:
        """
        CRO Query: Show forecast workflows where variance > threshold and overrides applied
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.kg_store.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                query = """
                    WITH forecast_workflows AS (
                        SELECT e.id, e.properties, e.created_at,
                               r.target_id as evidence_id,
                               r2.target_id as override_id
                        FROM kg_entities e
                        LEFT JOIN kg_relationships r ON e.id = r.source_id 
                                                     AND r.relationship_type = 'evidence'
                        LEFT JOIN kg_relationships r2 ON e.id = r2.source_id 
                                                      AND r2.relationship_type = 'overridden_by'
                        WHERE e.entity_type = 'workflow'
                          AND (e.properties->>'module' ILIKE '%forecast%' 
                               OR e.properties->>'name' ILIKE '%forecast%')
                          AND e.created_at > NOW() - INTERVAL '%s days'
                    ),
                    variance_analysis AS (
                        SELECT fw.*,
                               CASE 
                                   WHEN fw.properties->>'variance' IS NOT NULL 
                                   THEN (fw.properties->>'variance')::FLOAT
                                   ELSE 0.0
                               END as variance_value
                        FROM forecast_workflows fw
                    )
                    SELECT * FROM variance_analysis 
                    WHERE ABS(variance_value) > %s
                       OR override_id IS NOT NULL
                    ORDER BY variance_value DESC, created_at DESC
                """ % (time_period_days, variance_threshold)
                
                rows = await conn.fetch(query)
                
                results = []
                for row in rows:
                    results.append({
                        'workflow_id': row['id'],
                        'workflow_name': row['properties'].get('name'),
                        'variance': row.get('variance_value', 0.0),
                        'has_override': row['override_id'] is not None,
                        'override_id': row['override_id'],
                        'evidence_id': row['evidence_id'],
                        'created_at': row['created_at'].isoformat(),
                        'impact_level': 'high' if abs(row.get('variance_value', 0)) > 0.15 else 'medium'
                    })
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                return QueryResult(
                    query_type='cro_forecast_variance_analysis',
                    results=results,
                    total_count=len(results),
                    execution_time_ms=execution_time,
                    metadata={
                        'variance_threshold': variance_threshold,
                        'time_period_days': time_period_days,
                        'high_impact_count': sum(1 for r in results if r['impact_level'] == 'high')
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ CRO variance analysis failed: {e}")
            return QueryResult('cro_forecast_variance_analysis', [], 0, 0, {'error': str(e)})
    
    async def cfo_compensation_overrides(self, tenant_id: str,
                                       time_period_days: int = 90) -> QueryResult:
        """
        CFO Query: List all compensation workflows overridden by Finance in timeframe
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.kg_store.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                query = """
                    SELECT w.id as workflow_id,
                           w.properties as workflow_props,
                           w.created_at,
                           o.id as override_id,
                           o.properties as override_props,
                           o.created_at as override_date
                    FROM kg_entities w
                    JOIN kg_relationships r ON w.id = r.source_id
                    JOIN kg_entities o ON r.target_id = o.id
                    WHERE w.entity_type = 'workflow'
                      AND (w.properties->>'module' ILIKE '%compensation%' 
                           OR w.properties->>'name' ILIKE '%compensation%'
                           OR w.properties->>'name' ILIKE '%payout%')
                      AND r.relationship_type = 'overridden_by'
                      AND o.entity_type = 'override'
                      AND (o.properties->>'overridden_by' ILIKE '%finance%'
                           OR o.properties->>'overridden_by' ILIKE '%cfo%')
                      AND w.created_at > NOW() - INTERVAL '%s days'
                    ORDER BY o.created_at DESC
                """ % time_period_days
                
                rows = await conn.fetch(query)
                
                results = []
                for row in rows:
                    override_props = row['override_props']
                    workflow_props = row['workflow_props']
                    
                    results.append({
                        'workflow_id': row['workflow_id'],
                        'workflow_name': workflow_props.get('name'),
                        'workflow_module': workflow_props.get('module'),
                        'override_id': row['override_id'],
                        'overridden_by': override_props.get('overridden_by'),
                        'override_reason': override_props.get('reason'),
                        'override_date': row['override_date'].isoformat(),
                        'workflow_date': row['created_at'].isoformat(),
                        'risk_level': self._assess_override_risk(override_props)
                    })
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                return QueryResult(
                    query_type='cfo_compensation_overrides',
                    results=results,
                    total_count=len(results),
                    execution_time_ms=execution_time,
                    metadata={
                        'time_period_days': time_period_days,
                        'high_risk_overrides': sum(1 for r in results if r['risk_level'] == 'high')
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ CFO compensation overrides query failed: {e}")
            return QueryResult('cfo_compensation_overrides', [], 0, 0, {'error': str(e)})
    
    async def compliance_sox_violations(self, tenant_id: str,
                                      time_period_days: int = 30) -> QueryResult:
        """
        Compliance Query: Retrieve workflows governed by SOX with failed SLA
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.kg_store.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                query = """
                    SELECT w.id as workflow_id,
                           w.properties as workflow_props,
                           w.created_at,
                           p.id as policy_id,
                           p.properties as policy_props,
                           t.execution_time_ms,
                           t.trust_score
                    FROM kg_entities w
                    JOIN kg_relationships r ON w.id = r.source_id
                    JOIN kg_entities p ON r.target_id = p.id
                    LEFT JOIN kg_execution_traces t ON w.id = t.workflow_id
                    WHERE w.entity_type = 'workflow'
                      AND r.relationship_type = 'governed_by'
                      AND p.entity_type = 'policy_pack'
                      AND p.id ILIKE '%sox%'
                      AND w.created_at > NOW() - INTERVAL '%s days'
                      AND (t.execution_time_ms > 30000 
                           OR t.trust_score < 0.8
                           OR t.trust_score IS NULL)
                    ORDER BY w.created_at DESC
                """ % time_period_days
                
                rows = await conn.fetch(query)
                
                results = []
                for row in rows:
                    workflow_props = row['workflow_props']
                    
                    # Determine violation type
                    violations = []
                    if row.get('execution_time_ms', 0) > 30000:
                        violations.append('SLA_BREACH')
                    if row.get('trust_score', 1.0) < 0.8:
                        violations.append('LOW_TRUST_SCORE')
                    if row.get('trust_score') is None:
                        violations.append('MISSING_TRUST_METRICS')
                    
                    results.append({
                        'workflow_id': row['workflow_id'],
                        'workflow_name': workflow_props.get('name'),
                        'workflow_module': workflow_props.get('module'),
                        'policy_id': row['policy_id'],
                        'violations': violations,
                        'execution_time_ms': row.get('execution_time_ms', 0),
                        'trust_score': row.get('trust_score'),
                        'created_at': row['created_at'].isoformat(),
                        'severity': 'critical' if len(violations) > 1 else 'high'
                    })
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                return QueryResult(
                    query_type='compliance_sox_violations',
                    results=results,
                    total_count=len(results),
                    execution_time_ms=execution_time,
                    metadata={
                        'time_period_days': time_period_days,
                        'critical_violations': sum(1 for r in results if r['severity'] == 'critical'),
                        'violation_types': list(set(v for r in results for v in r['violations']))
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ Compliance SOX violations query failed: {e}")
            return QueryResult('compliance_sox_violations', [], 0, 0, {'error': str(e)})
    
    async def analyst_override_patterns(self, tenant_id: str,
                                      time_period_days: int = 120) -> QueryResult:
        """
        Analyst Query: Analyze override patterns across regions over time
        """
        start_time = datetime.utcnow()
        
        try:
            async with self.kg_store.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                query = """
                    WITH override_analysis AS (
                        SELECT o.properties->>'overridden_by' as override_by,
                               o.properties->>'reason' as override_reason,
                               w.properties->>'module' as workflow_module,
                               e.region,
                               DATE_TRUNC('week', o.created_at) as week_start,
                               COUNT(*) as override_count
                        FROM kg_entities o
                        JOIN kg_relationships r ON o.id = r.target_id
                        JOIN kg_entities w ON r.source_id = w.id
                        LEFT JOIN kg_entities e ON w.id = e.id
                        WHERE o.entity_type = 'override'
                          AND r.relationship_type = 'overridden_by'
                          AND w.entity_type = 'workflow'
                          AND o.created_at > NOW() - INTERVAL '%s days'
                        GROUP BY override_by, override_reason, workflow_module, e.region, week_start
                    )
                    SELECT *,
                           ROW_NUMBER() OVER (PARTITION BY region ORDER BY override_count DESC) as region_rank
                    FROM override_analysis
                    ORDER BY week_start DESC, override_count DESC
                """ % time_period_days
                
                rows = await conn.fetch(query)
                
                # Process results for pattern analysis
                results = []
                regional_stats = {}
                
                for row in rows:
                    region = row['region'] or 'Unknown'
                    
                    if region not in regional_stats:
                        regional_stats[region] = {
                            'total_overrides': 0,
                            'top_reasons': {},
                            'top_modules': {},
                            'trend': []
                        }
                    
                    regional_stats[region]['total_overrides'] += row['override_count']
                    
                    # Track top reasons
                    reason = row['override_reason'] or 'Not specified'
                    regional_stats[region]['top_reasons'][reason] = regional_stats[region]['top_reasons'].get(reason, 0) + row['override_count']
                    
                    # Track top modules
                    module = row['workflow_module'] or 'Unknown'
                    regional_stats[region]['top_modules'][module] = regional_stats[region]['top_modules'].get(module, 0) + row['override_count']
                    
                    results.append({
                        'region': region,
                        'week': row['week_start'].isoformat(),
                        'override_by': row['override_by'],
                        'override_reason': reason,
                        'workflow_module': module,
                        'override_count': row['override_count'],
                        'region_rank': row['region_rank']
                    })
                
                # Generate insights
                insights = []
                for region, stats in regional_stats.items():
                    top_reason = max(stats['top_reasons'].items(), key=lambda x: x[1]) if stats['top_reasons'] else ('None', 0)
                    top_module = max(stats['top_modules'].items(), key=lambda x: x[1]) if stats['top_modules'] else ('None', 0)
                    
                    insights.append({
                        'region': region,
                        'total_overrides': stats['total_overrides'],
                        'top_override_reason': top_reason[0],
                        'top_override_count': top_reason[1],
                        'most_overridden_module': top_module[0],
                        'module_override_count': top_module[1]
                    })
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                return QueryResult(
                    query_type='analyst_override_patterns',
                    results=results,
                    total_count=len(results),
                    execution_time_ms=execution_time,
                    metadata={
                        'time_period_days': time_period_days,
                        'regional_insights': insights,
                        'total_regions': len(regional_stats),
                        'total_overrides': sum(stats['total_overrides'] for stats in regional_stats.values())
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ Analyst override patterns query failed: {e}")
            return QueryResult('analyst_override_patterns', [], 0, 0, {'error': str(e)})
    
    async def workflow_impact_analysis(self, workflow_id: str, tenant_id: str) -> QueryResult:
        """
        Analyze the impact and relationships of a specific workflow
        """
        start_time = datetime.utcnow()
        
        try:
            # Get workflow entity
            workflow = await self.kg_store.get_entity(workflow_id, tenant_id)
            if not workflow:
                return QueryResult('workflow_impact_analysis', [], 0, 0, {'error': 'Workflow not found'})
            
            # Get all relationships
            relationships = await self.kg_store.get_relationships(workflow_id, tenant_id)
            
            # Analyze impact
            impact_data = {
                'workflow_info': workflow.to_dict(),
                'direct_relationships': len(relationships),
                'entities_affected': [],
                'governance_controls': [],
                'execution_history': []
            }
            
            for rel in relationships:
                if rel.relationship_type == RelationshipType.ACTED_ON_BY:
                    # Get business entity details
                    entity = await self.kg_store.get_entity(rel.source_id, tenant_id)
                    if entity:
                        impact_data['entities_affected'].append({
                            'entity_id': entity.id,
                            'entity_type': entity.entity_type.value,
                            'properties': entity.properties
                        })
                
                elif rel.relationship_type == RelationshipType.GOVERNED_BY:
                    # Get policy details
                    policy = await self.kg_store.get_entity(rel.target_id, tenant_id)
                    if policy:
                        impact_data['governance_controls'].append({
                            'policy_id': policy.id,
                            'policy_type': policy.properties.get('type'),
                            'enforcement_level': policy.properties.get('enforcement_level')
                        })
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return QueryResult(
                query_type='workflow_impact_analysis',
                results=[impact_data],
                total_count=1,
                execution_time_ms=execution_time,
                metadata={
                    'workflow_id': workflow_id,
                    'entities_count': len(impact_data['entities_affected']),
                    'governance_count': len(impact_data['governance_controls'])
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Workflow impact analysis failed: {e}")
            return QueryResult('workflow_impact_analysis', [], 0, 0, {'error': str(e)})
    
    def _assess_override_risk(self, override_props: Dict[str, Any]) -> str:
        """Assess risk level of an override"""
        reason = override_props.get('reason', '').lower()
        override_type = override_props.get('override_type', '').lower()
        
        high_risk_keywords = ['emergency', 'urgent', 'bypass', 'exception']
        medium_risk_keywords = ['adjustment', 'correction', 'update']
        
        if any(keyword in reason for keyword in high_risk_keywords) or override_type == 'emergency':
            return 'high'
        elif any(keyword in reason for keyword in medium_risk_keywords):
            return 'medium'
        else:
            return 'low'
