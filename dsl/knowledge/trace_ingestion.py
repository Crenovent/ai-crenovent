#!/usr/bin/env python3
"""
Trace Ingestion Engine
=====================

Converts execution traces into Knowledge Graph entities and relationships
as specified in Vision Doc Ch.8.2

Features:
- Execution trace parsing and normalization
- Entity extraction from trace data
- Relationship inference
- Governance metadata integration
- Async processing pipeline
"""

import json
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio

from .ontology import (
    EntityNode, RelationshipEdge, EntityType, RelationshipType, 
    RevOpsOntology
)
from .kg_store import KnowledgeGraphStore
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.services.connection_pool_manager import ConnectionPoolManager

logger = logging.getLogger(__name__)

class TraceIngestionEngine:
    """
    Converts execution traces into Knowledge Graph representation
    
    Process:
    1. Parse execution trace JSON
    2. Extract entities (workflows, runs, evidence, etc.)
    3. Infer relationships between entities  
    4. Store in Knowledge Graph with governance metadata
    5. Generate embeddings for semantic search
    """
    
    def __init__(self, kg_store: KnowledgeGraphStore, pool_manager: ConnectionPoolManager):
        self.kg_store = kg_store
        self.pool_manager = pool_manager
        self.ontology = RevOpsOntology()
        
    async def ingest_execution_trace(self, trace_data: Dict[str, Any]) -> bool:
        """
        Ingest an execution trace into the Knowledge Graph
        
        Expected trace format (from Vision Doc Ch.8.2):
        {
            "workflow_id": "wf_forecast_rebalance",
            "run_id": "run_12345", 
            "tenant_id": "t999",
            "policies_applied": ["sox_forecast"],
            "inputs": {...},
            "outputs": {...},
            "trust_score": 0.87,
            "execution_time_ms": 1500,
            "evidence_pack_id": "ep2025",
            "override": {"by": "CRO", "reason": "strategic account"},
            "entities_affected": ["opp_456", "forecast_Q3_NA"],
            "created_at": "2024-01-15T10:30:00Z"
        }
        """
        try:
            logger.info(f"🔄 Ingesting execution trace: {trace_data.get('run_id')}")
            
            # 1. Create workflow execution entities
            entities = await self._extract_entities(trace_data)
            
            # 2. Create relationships between entities
            relationships = await self._infer_relationships(trace_data, entities)
            
            # 3. Store in Knowledge Graph
            success = await self._store_in_kg(entities, relationships, trace_data)
            
            if success:
                logger.info(f"✅ Successfully ingested trace: {trace_data.get('run_id')}")
                # Store raw trace for reference
                await self._store_execution_trace(trace_data)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest trace {trace_data.get('run_id')}: {e}")
            return False
    
    async def _extract_entities(self, trace_data: Dict[str, Any]) -> List[EntityNode]:
        """Extract entities from execution trace"""
        entities = []
        tenant_id = trace_data['tenant_id']
        
        # 1. Workflow entity
        workflow_entity = self.ontology.create_entity_node(
            entity_id=trace_data['workflow_id'],
            entity_type=EntityType.WORKFLOW,
            properties={
                'name': trace_data.get('workflow_name', trace_data['workflow_id']),
                'automation_type': trace_data.get('automation_type', 'RBA'),
                'module': trace_data.get('module', 'unknown'),
                'version': trace_data.get('version', '1.0'),
                'status': 'active'
            },
            tenant_id=tenant_id,
            policy_pack_id=trace_data.get('policies_applied', [None])[0] if trace_data.get('policies_applied') else None
        )
        entities.append(workflow_entity)
        
        # 2. Execution run entity
        run_entity = self.ontology.create_entity_node(
            entity_id=trace_data['run_id'],
            entity_type=EntityType.EXECUTION_RUN,
            properties={
                'workflow_id': trace_data['workflow_id'],
                'status': trace_data.get('status', 'completed'),
                'execution_time_ms': trace_data.get('execution_time_ms', 0),
                'trust_score': trace_data.get('trust_score', 1.0),
                'cost_tokens': trace_data.get('cost_tokens', 0),
                'started_at': trace_data.get('started_at'),
                'ended_at': trace_data.get('ended_at', trace_data.get('created_at')),
                'inputs_hash': self._hash_data(trace_data.get('inputs', {})),
                'outputs_hash': self._hash_data(trace_data.get('outputs', {}))
            },
            tenant_id=tenant_id,
            evidence_pack_id=trace_data.get('evidence_pack_id')
        )
        entities.append(run_entity)
        
        # 3. Policy pack entity (if applied)
        for policy_id in trace_data.get('policies_applied', []):
            policy_entity = self.ontology.create_entity_node(
                entity_id=policy_id,
                entity_type=EntityType.POLICY_PACK,
                properties={
                    'name': policy_id.replace('_', ' ').title(),
                    'type': 'compliance',
                    'status': 'active',
                    'enforcement_level': 'mandatory'
                },
                tenant_id=tenant_id
            )
            entities.append(policy_entity)
        
        # 4. Evidence pack entity (if exists)
        if trace_data.get('evidence_pack_id'):
            evidence_entity = self.ontology.create_entity_node(
                entity_id=trace_data['evidence_pack_id'],
                entity_type=EntityType.EVIDENCE_PACK,
                properties={
                    'workflow_id': trace_data['workflow_id'],
                    'run_id': trace_data['run_id'],
                    'inputs_hash': self._hash_data(trace_data.get('inputs', {})),
                    'outputs_hash': self._hash_data(trace_data.get('outputs', {})),
                    'compliance_status': 'compliant',
                    'retention_period': trace_data.get('retention_period', '7 years')
                },
                tenant_id=tenant_id
            )
            entities.append(evidence_entity)
        
        # 5. Override entity (if exists)
        if trace_data.get('override'):
            override_data = trace_data['override']
            override_entity = self.ontology.create_entity_node(
                entity_id=f"override_{trace_data['run_id']}",
                entity_type=EntityType.OVERRIDE,
                properties={
                    'workflow_id': trace_data['workflow_id'],
                    'run_id': trace_data['run_id'],
                    'overridden_by': override_data.get('by'),
                    'reason': override_data.get('reason'),
                    'override_type': override_data.get('type', 'manual'),
                    'approval_required': override_data.get('approval_required', True)
                },
                tenant_id=tenant_id
            )
            entities.append(override_entity)
        
        # 6. Business entities (from entities_affected)
        for entity_id in trace_data.get('entities_affected', []):
            # Infer entity type from ID pattern
            entity_type = self._infer_entity_type(entity_id)
            
            business_entity = self.ontology.create_entity_node(
                entity_id=entity_id,
                entity_type=entity_type,
                properties={
                    'external_id': entity_id,
                    'last_modified': trace_data.get('created_at'),
                    'modified_by_workflow': trace_data['workflow_id']
                },
                tenant_id=tenant_id
            )
            entities.append(business_entity)
        
        logger.debug(f"📊 Extracted {len(entities)} entities from trace")
        return entities
    
    async def _infer_relationships(self, trace_data: Dict[str, Any], 
                                 entities: List[EntityNode]) -> List[RelationshipEdge]:
        """Infer relationships between extracted entities"""
        relationships = []
        tenant_id = trace_data['tenant_id']
        
        # Find entity IDs by type
        entity_map = {entity.entity_type: entity.id for entity in entities}
        
        # 1. Workflow execution relationship
        if EntityType.WORKFLOW in entity_map and EntityType.EXECUTION_RUN in entity_map:
            rel = self.ontology.create_relationship_edge(
                edge_id=f"rel_{uuid.uuid4().hex[:8]}",
                source_id=entity_map[EntityType.WORKFLOW],
                target_id=entity_map[EntityType.EXECUTION_RUN],
                relationship_type=RelationshipType.EXECUTED,
                properties={'execution_time': trace_data.get('created_at')},
                tenant_id=tenant_id
            )
            relationships.append(rel)
        
        # 2. Policy governance relationships
        workflow_id = entity_map.get(EntityType.WORKFLOW)
        if workflow_id:
            for policy_id in trace_data.get('policies_applied', []):
                rel = self.ontology.create_relationship_edge(
                    edge_id=f"rel_{uuid.uuid4().hex[:8]}",
                    source_id=workflow_id,
                    target_id=policy_id,
                    relationship_type=RelationshipType.GOVERNED_BY,
                    properties={'enforcement_level': 'mandatory'},
                    tenant_id=tenant_id
                )
                relationships.append(rel)
        
        # 3. Evidence relationship
        if EntityType.EVIDENCE_PACK in entity_map and workflow_id:
            rel = self.ontology.create_relationship_edge(
                edge_id=f"rel_{uuid.uuid4().hex[:8]}",
                source_id=workflow_id,
                target_id=entity_map[EntityType.EVIDENCE_PACK],
                relationship_type=RelationshipType.EVIDENCE,
                properties={'evidence_type': 'execution'},
                tenant_id=tenant_id,
                evidence_pack_id=trace_data.get('evidence_pack_id')
            )
            relationships.append(rel)
        
        # 4. Override relationship
        if EntityType.OVERRIDE in entity_map and workflow_id:
            rel = self.ontology.create_relationship_edge(
                edge_id=f"rel_{uuid.uuid4().hex[:8]}",
                source_id=workflow_id,
                target_id=entity_map[EntityType.OVERRIDE],
                relationship_type=RelationshipType.OVERRIDDEN_BY,
                properties={
                    'override_reason': trace_data.get('override', {}).get('reason'),
                    'override_by': trace_data.get('override', {}).get('by')
                },
                tenant_id=tenant_id
            )
            relationships.append(rel)
        
        # 5. Business entity relationships
        for entity_id in trace_data.get('entities_affected', []):
            rel = self.ontology.create_relationship_edge(
                edge_id=f"rel_{uuid.uuid4().hex[:8]}",
                source_id=entity_id,
                target_id=workflow_id,
                relationship_type=RelationshipType.ACTED_ON_BY,
                properties={'action_type': trace_data.get('automation_type', 'RBA')},
                tenant_id=tenant_id
            )
            relationships.append(rel)
        
        logger.debug(f"🔗 Inferred {len(relationships)} relationships")
        return relationships
    
    async def _store_in_kg(self, entities: List[EntityNode], 
                         relationships: List[RelationshipEdge],
                         trace_data: Dict[str, Any]) -> bool:
        """Store entities and relationships in Knowledge Graph"""
        try:
            # Store entities first
            for entity in entities:
                success = await self.kg_store.store_entity(entity)
                if not success:
                    logger.error(f"Failed to store entity {entity.id}")
                    return False
            
            # Store relationships
            for relationship in relationships:
                success = await self.kg_store.store_relationship(relationship)
                if not success:
                    logger.error(f"Failed to store relationship {relationship.id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store in KG: {e}")
            return False
    
    async def _store_execution_trace(self, trace_data: Dict[str, Any]):
        """Store raw execution trace for reference"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", trace_data['tenant_id'])
                
                await conn.execute("""
                    INSERT INTO kg_execution_traces (
                        trace_id, workflow_id, run_id, tenant_id, inputs, outputs,
                        governance_metadata, execution_time_ms, trust_score, cost_tokens,
                        policy_pack_id, evidence_pack_id, override_id, entities_affected,
                        relationships_created, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (trace_id) DO NOTHING
                """,
                    trace_data['run_id'],  # Using run_id as trace_id
                    trace_data['workflow_id'],
                    trace_data['run_id'],
                    trace_data['tenant_id'],
                    json.dumps(trace_data.get('inputs', {})),
                    json.dumps(trace_data.get('outputs', {})),
                    json.dumps({
                        'policies_applied': trace_data.get('policies_applied', []),
                        'override': trace_data.get('override'),
                        'evidence_pack_id': trace_data.get('evidence_pack_id')
                    }),
                    trace_data.get('execution_time_ms', 0),
                    trace_data.get('trust_score', 1.0),
                    trace_data.get('cost_tokens', 0),
                    trace_data.get('policies_applied', [None])[0] if trace_data.get('policies_applied') else None,
                    trace_data.get('evidence_pack_id'),
                    f"override_{trace_data['run_id']}" if trace_data.get('override') else None,
                    trace_data.get('entities_affected', []),
                    [],  # relationships_created - will be populated later
                    datetime.fromisoformat(trace_data.get('created_at', datetime.utcnow().isoformat()).replace('Z', '+00:00'))
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to store execution trace: {e}")
    
    def _infer_entity_type(self, entity_id: str) -> EntityType:
        """Infer entity type from ID pattern"""
        if entity_id.startswith('opp_'):
            return EntityType.OPPORTUNITY
        elif entity_id.startswith('acc_'):
            return EntityType.ACCOUNT
        elif entity_id.startswith('forecast_'):
            return EntityType.FORECAST_CYCLE
        elif entity_id.startswith('quota_'):
            return EntityType.QUOTA
        elif entity_id.startswith('user_'):
            return EntityType.USER
        else:
            return EntityType.CUSTOMER  # Default fallback
    
    def _hash_data(self, data: Dict[str, Any]) -> str:
        """Generate hash for data"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    async def batch_ingest_traces(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch process multiple execution traces"""
        results = {
            'total': len(traces),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        logger.info(f"🔄 Processing {len(traces)} traces in batch")
        
        # Process traces concurrently (with limit)
        semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
        
        async def process_trace(trace):
            async with semaphore:
                try:
                    success = await self.ingest_execution_trace(trace)
                    if success:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to process {trace.get('run_id')}")
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error processing {trace.get('run_id')}: {str(e)}")
        
        await asyncio.gather(*[process_trace(trace) for trace in traces])
        
        logger.info(f"✅ Batch processing complete: {results['successful']}/{results['total']} successful")
        return results
