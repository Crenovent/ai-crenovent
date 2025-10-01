"""
RBA Execution Tracer
===================
Captures and stores execution traces for all RBA agent runs in the Knowledge Graph.
Provides rich metadata, performance metrics, and audit trails for ML training and governance.
"""

import json
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import asyncio
from dataclasses import dataclass, asdict

from .kg_store import KnowledgeGraphStore
from ..configuration.universal_parameter_manager import get_universal_parameter_manager

logger = logging.getLogger(__name__)

@dataclass
class RBAExecutionTrace:
    """Structured execution trace for RBA agent runs"""
    
    # Core identification
    execution_id: str
    agent_name: str
    agent_type: str = "RBA"
    tenant_id: str = "1300"
    user_id: str = "1319"
    
    # Execution metadata
    execution_start_time: datetime = None
    execution_end_time: datetime = None
    execution_duration_ms: float = 0.0
    
    # Input/Output data
    input_data_summary: Dict[str, Any] = None
    output_data_summary: Dict[str, Any] = None
    configuration_used: Dict[str, Any] = None
    
    # Results and metrics
    total_opportunities_processed: int = 0
    flagged_opportunities_count: int = 0
    success: bool = True
    error_message: Optional[str] = None
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percentage: float = 0.0
    database_queries_count: int = 0
    api_calls_count: int = 0
    
    # Business impact metrics
    high_risk_deals_identified: int = 0
    total_pipeline_value_analyzed: float = 0.0
    at_risk_pipeline_value: float = 0.0
    compliance_score: float = 0.0
    
    # Governance and audit
    policy_pack_id: Optional[str] = None
    evidence_pack_id: Optional[str] = None
    override_count: int = 0
    approval_required: bool = False
    
    # Knowledge extraction
    entities_created: List[str] = None
    relationships_created: List[str] = None
    insights_generated: List[str] = None
    
    # Context and environment
    workflow_category: str = ""
    execution_source: str = "api"  # api, scheduled, manual
    client_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.execution_start_time is None:
            self.execution_start_time = datetime.now(timezone.utc)
        if self.input_data_summary is None:
            self.input_data_summary = {}
        if self.output_data_summary is None:
            self.output_data_summary = {}
        if self.configuration_used is None:
            self.configuration_used = {}
        if self.entities_created is None:
            self.entities_created = []
        if self.relationships_created is None:
            self.relationships_created = []
        if self.insights_generated is None:
            self.insights_generated = []
        if self.client_info is None:
            self.client_info = {}

class RBAExecutionTracer:
    """
    Captures and stores execution traces for RBA agents in Knowledge Graph.
    
    Features:
    - Automatic trace generation for all RBA executions
    - Rich metadata capture (performance, business impact, governance)
    - Knowledge Graph entity/relationship extraction
    - ML training data preparation
    - Audit trail compliance
    - Real-time trace ingestion
    """
    
    def __init__(self, kg_store: KnowledgeGraphStore):
        self.kg_store = kg_store
        self.parameter_manager = get_universal_parameter_manager()
        
    async def start_trace(self, agent_name: str, input_data: Dict[str, Any], config: Dict[str, Any], context: Dict[str, Any] = None) -> RBAExecutionTrace:
        """
        Start a new execution trace.
        
        Args:
            agent_name: Name of the RBA agent
            input_data: Input data for the execution
            config: Configuration parameters used
            context: Additional context (user_id, tenant_id, etc.)
            
        Returns:
            RBAExecutionTrace object to track the execution
        """
        
        execution_id = f"rba_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Extract context information
        context = context or {}
        tenant_id = context.get('tenant_id', '1300')
        user_id = context.get('user_id', '1319')
        execution_source = context.get('source', 'api')
        
        # Create trace object
        trace = RBAExecutionTrace(
            execution_id=execution_id,
            agent_name=agent_name,
            tenant_id=tenant_id,
            user_id=user_id,
            execution_source=execution_source,
            workflow_category=agent_name,
            configuration_used=config.copy(),
            input_data_summary=self._summarize_input_data(input_data),
            client_info=context.get('client_info', {})
        )
        
        logger.info(f"🚀 Started RBA execution trace: {execution_id}")
        return trace
    
    async def complete_trace(self, trace: RBAExecutionTrace, result: Dict[str, Any], error: Optional[Exception] = None) -> None:
        """
        Complete an execution trace and store it in Knowledge Graph.
        
        Args:
            trace: The execution trace to complete
            result: Execution result data
            error: Exception if execution failed
        """
        
        # Update trace with completion data
        trace.execution_end_time = datetime.now(timezone.utc)
        trace.execution_duration_ms = (trace.execution_end_time - trace.execution_start_time).total_seconds() * 1000
        
        if error:
            trace.success = False
            trace.error_message = str(error)
        else:
            trace.success = result.get('success', True)
            trace.output_data_summary = self._summarize_output_data(result)
            
            # Extract business metrics
            trace.total_opportunities_processed = result.get('total_opportunities', 0)
            trace.flagged_opportunities_count = result.get('flagged_opportunities', 0)
            trace.high_risk_deals_identified = self._count_high_risk_deals(result)
            trace.total_pipeline_value_analyzed = self._calculate_pipeline_value(result)
            trace.at_risk_pipeline_value = self._calculate_at_risk_value(result)
            trace.compliance_score = self._calculate_compliance_score(result)
            
            # Extract insights
            trace.insights_generated = result.get('insights', [])
        
        # Store trace in Knowledge Graph
        await self._store_trace_in_kg(trace)
        
        logger.info(f"✅ Completed RBA execution trace: {trace.execution_id} (duration: {trace.execution_duration_ms:.2f}ms)")
    
    def _summarize_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of input data for trace storage"""
        
        opportunities = input_data.get('opportunities', [])
        config = input_data.get('config', {})
        
        return {
            'opportunities_count': len(opportunities),
            'config_parameters_count': len(config),
            'has_user_input': bool(input_data.get('user_input')),
            'data_sources': input_data.get('data_sources', ['csv']),
            'input_size_bytes': len(str(input_data))
        }
    
    def _summarize_output_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of output data for trace storage"""
        
        return {
            'success': result.get('success', False),
            'analysis_type': result.get('analysis_type', ''),
            'total_opportunities': result.get('total_opportunities', 0),
            'flagged_opportunities': result.get('flagged_opportunities', 0),
            'has_summary_metrics': 'summary_metrics' in result,
            'has_insights': 'insights' in result,
            'has_flagged_deals': 'flagged_deals' in result,
            'output_size_bytes': len(str(result))
        }
    
    def _count_high_risk_deals(self, result: Dict[str, Any]) -> int:
        """Count high-risk deals from result"""
        
        risk_distribution = result.get('risk_distribution', {})
        return risk_distribution.get('high_risk', 0) + risk_distribution.get('critical_risk', 0)
    
    def _calculate_pipeline_value(self, result: Dict[str, Any]) -> float:
        """Calculate total pipeline value analyzed"""
        
        summary_metrics = result.get('summary_metrics', {})
        return summary_metrics.get('total_pipeline_value', 0.0)
    
    def _calculate_at_risk_value(self, result: Dict[str, Any]) -> float:
        """Calculate at-risk pipeline value"""
        
        summary_metrics = result.get('summary_metrics', {})
        return summary_metrics.get('at_risk_pipeline_value', 0.0)
    
    def _calculate_compliance_score(self, result: Dict[str, Any]) -> float:
        """Calculate compliance score from result"""
        
        summary_metrics = result.get('summary_metrics', {})
        total_opps = summary_metrics.get('total_opportunities', 1)
        flagged_opps = result.get('flagged_opportunities', 0)
        
        # Higher compliance = lower flagged percentage
        return max(0.0, 100.0 - (flagged_opps / total_opps * 100)) if total_opps > 0 else 100.0
    
    async def _store_trace_in_kg(self, trace: RBAExecutionTrace) -> None:
        """Store execution trace in Knowledge Graph"""
        
        try:
            # Convert trace to dict for storage
            trace_data = asdict(trace)
            
            # Convert datetime objects to ISO strings
            if trace_data['execution_start_time']:
                trace_data['execution_start_time'] = trace.execution_start_time.isoformat()
            if trace_data['execution_end_time']:
                trace_data['execution_end_time'] = trace.execution_end_time.isoformat()
            
            # Store in kg_execution_traces table
            async with self.kg_store.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS policy
                tenant_id_int = int(trace.tenant_id) if isinstance(trace.tenant_id, str) else trace.tenant_id
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id_int))
                
                await conn.execute("""
                    INSERT INTO kg_execution_traces (
                        trace_id, workflow_id, run_id, tenant_id, 
                        inputs, outputs, governance_metadata,
                        execution_time_ms, trust_score, entities_affected,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                    trace.execution_id,  # Use execution_id as trace_id
                    trace.agent_name,    # workflow_id
                    trace.execution_id,  # run_id (same as trace_id for now)
                    tenant_id_int,
                    json.dumps(trace_data),  # inputs
                    json.dumps({
                        'success': trace.success,
                        'flagged_opportunities': getattr(trace, 'flagged_opportunities', 0),
                        'total_opportunities': getattr(trace, 'total_opportunities', 0),
                        'execution_result': trace_data
                    }),  # outputs
                    json.dumps({
                        'tenant_id': tenant_id_int,
                        'execution_start_time': trace.execution_start_time.isoformat() if trace.execution_start_time else None,
                        'agent_name': trace.agent_name
                    }),  # governance_metadata
                    int(trace.execution_duration_ms) if hasattr(trace, 'execution_duration_ms') else 0,  # execution_time_ms
                    1.0,  # trust_score (default high trust for RBA)
                    trace.entities_created,  # entities_affected
                    trace.execution_start_time
                )
            
            # Create Knowledge Graph entities for this execution
            await self._create_kg_entities_for_trace(trace)
            
        except Exception as e:
            logger.error(f"❌ Failed to store execution trace {trace.execution_id}: {e}")
            # Don't raise - trace storage failure shouldn't break the main execution
    
    def _calculate_impact_score(self, trace: RBAExecutionTrace) -> float:
        """Calculate business impact score for this execution"""
        
        if not trace.success:
            return 0.0
        
        # Score based on multiple factors
        score = 0.0
        
        # Volume impact (0-30 points)
        if trace.total_opportunities_processed > 0:
            volume_score = min(30, trace.total_opportunities_processed / 100 * 30)
            score += volume_score
        
        # Value impact (0-40 points)
        if trace.total_pipeline_value_analyzed > 0:
            value_score = min(40, trace.total_pipeline_value_analyzed / 10000000 * 40)  # $10M = max
            score += value_score
        
        # Risk identification impact (0-20 points)
        if trace.flagged_opportunities_count > 0:
            risk_score = min(20, trace.flagged_opportunities_count / 50 * 20)  # 50 flagged = max
            score += risk_score
        
        # Compliance impact (0-10 points)
        score += trace.compliance_score / 10
        
        return min(100.0, score)
    
    async def _create_kg_entities_for_trace(self, trace: RBAExecutionTrace) -> None:
        """Create Knowledge Graph entities for this execution trace"""
        
        try:
            from dsl.knowledge.ontology import EntityNode
            
            from datetime import datetime
            
            # Create execution entity
            current_time = datetime.now()
            from dsl.knowledge.ontology import EntityType
            
            execution_entity = EntityNode(
                id=f"execution_{trace.execution_id}",
                entity_type=EntityType.EXECUTION_RUN,
                properties={
                    'agent_name': trace.agent_name,
                    'execution_duration_ms': getattr(trace, 'execution_duration_ms', 0),
                    'opportunities_processed': getattr(trace, 'total_opportunities_processed', 0),
                    'flagged_opportunities': getattr(trace, 'flagged_opportunities_count', 0),
                    'success': trace.success,
                    'impact_score': self._calculate_impact_score(trace),
                    'pipeline_value': getattr(trace, 'total_pipeline_value_analyzed', 0),
                    'execution_date': trace.execution_start_time.isoformat() if trace.execution_start_time else None,
                    'search_text': f"{trace.agent_name} execution {trace.execution_id}"  # Move to properties
                },
                tenant_id=str(int(trace.tenant_id) if isinstance(trace.tenant_id, str) else trace.tenant_id),
                created_at=current_time,
                updated_at=current_time
            )
            
            # Store entity
            await self.kg_store.store_entity(execution_entity)
            
            # Create relationships to user and tenant
            from dsl.knowledge.ontology import RelationshipEdge, RelationshipType
            
            user_relationship = RelationshipEdge(
                id=f"rel_user_{trace.execution_id}",
                source_id=f"user_{trace.user_id}",
                target_id=f"execution_{trace.execution_id}",
                relationship_type=RelationshipType.EXECUTED,
                properties={
                    'execution_time': trace.execution_start_time.isoformat() if trace.execution_start_time else None,
                    'agent_name': trace.agent_name
                },
                tenant_id=str(int(trace.tenant_id) if isinstance(trace.tenant_id, str) else trace.tenant_id),
                confidence_score=1.0,
                created_at=current_time
            )
            
            await self.kg_store.store_relationship(user_relationship)
            
        except Exception as e:
            logger.error(f"❌ Failed to create KG entities for trace {trace.execution_id}: {e}")
    
    async def get_execution_history(self, agent_name: Optional[str] = None, tenant_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get execution history from Knowledge Graph.
        
        Args:
            agent_name: Filter by specific agent (optional)
            tenant_id: Filter by tenant (optional)
            limit: Maximum number of traces to return
            
        Returns:
            List of execution trace summaries
        """
        
        try:
            async with self.kg_store.pool_manager.postgres_pool.acquire() as conn:
                query = """
                    SELECT execution_id, workflow_id, trace_data, outcome_type, impact_score, created_at
                    FROM kg_execution_traces 
                    WHERE workflow_type = 'RBA'
                """
                params = []
                param_count = 0
                
                if agent_name:
                    param_count += 1
                    query += f" AND workflow_id = ${param_count}"
                    params.append(agent_name)
                
                if tenant_id:
                    param_count += 1
                    query += f" AND tenant_id = ${param_count}"
                    params.append(tenant_id)
                
                query += f" ORDER BY created_at DESC LIMIT ${param_count + 1}"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                history = []
                for row in rows:
                    trace_data = json.loads(row['trace_data']) if row['trace_data'] else {}
                    history.append({
                        'execution_id': row['execution_id'],
                        'agent_name': row['workflow_id'],
                        'outcome': row['outcome_type'],
                        'impact_score': row['impact_score'],
                        'created_at': row['created_at'].isoformat(),
                        'duration_ms': trace_data.get('execution_duration_ms', 0),
                        'opportunities_processed': trace_data.get('total_opportunities_processed', 0),
                        'flagged_opportunities': trace_data.get('flagged_opportunities_count', 0)
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"❌ Failed to get execution history: {e}")
            return []
    
    async def get_performance_analytics(self, agent_name: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Get performance analytics for RBA executions.
        
        Args:
            agent_name: Filter by specific agent (optional)
            days: Number of days to analyze
            
        Returns:
            Performance analytics summary
        """
        
        try:
            async with self.kg_store.pool_manager.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        workflow_id,
                        COUNT(*) as execution_count,
                        AVG(impact_score) as avg_impact_score,
                        AVG(CAST(trace_data->>'execution_duration_ms' AS FLOAT)) as avg_duration_ms,
                        SUM(CAST(trace_data->>'total_opportunities_processed' AS INT)) as total_opportunities,
                        SUM(CAST(trace_data->>'flagged_opportunities_count' AS INT)) as total_flagged,
                        COUNT(CASE WHEN outcome_type = 'SUCCESS' THEN 1 END) as success_count
                    FROM kg_execution_traces 
                    WHERE workflow_type = 'RBA' 
                    AND created_at >= NOW() - INTERVAL '%s days'
                """ % days
                
                params = []
                if agent_name:
                    query += " AND workflow_id = $1"
                    params.append(agent_name)
                
                query += " GROUP BY workflow_id ORDER BY execution_count DESC"
                
                rows = await conn.fetch(query, *params)
                
                analytics = {
                    'period_days': days,
                    'total_executions': sum(row['execution_count'] for row in rows),
                    'agents_analyzed': len(rows),
                    'agent_performance': []
                }
                
                for row in rows:
                    success_rate = (row['success_count'] / row['execution_count'] * 100) if row['execution_count'] > 0 else 0
                    flagging_rate = (row['total_flagged'] / row['total_opportunities'] * 100) if row['total_opportunities'] > 0 else 0
                    
                    analytics['agent_performance'].append({
                        'agent_name': row['workflow_id'],
                        'execution_count': row['execution_count'],
                        'success_rate': round(success_rate, 2),
                        'avg_impact_score': round(row['avg_impact_score'] or 0, 2),
                        'avg_duration_ms': round(row['avg_duration_ms'] or 0, 2),
                        'total_opportunities_processed': row['total_opportunities'] or 0,
                        'total_flagged_opportunities': row['total_flagged'] or 0,
                        'flagging_rate': round(flagging_rate, 2)
                    })
                
                return analytics
                
        except Exception as e:
            logger.error(f"❌ Failed to get performance analytics: {e}")
            return {'error': str(e)}

# Global tracer instance (will be initialized with KG store)
_rba_tracer: Optional[RBAExecutionTracer] = None

def get_rba_execution_tracer(kg_store: KnowledgeGraphStore) -> RBAExecutionTracer:
    """Get or create the global RBA execution tracer"""
    global _rba_tracer
    
    if _rba_tracer is None:
        _rba_tracer = RBAExecutionTracer(kg_store)
    
    return _rba_tracer
