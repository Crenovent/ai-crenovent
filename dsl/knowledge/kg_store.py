#!/usr/bin/env python3
"""
Knowledge Graph Storage Layer
============================

Implements KG storage using PostgreSQL + pgvector as specified in Vision Doc Ch.8.2
- Entity and relationship storage
- Multi-tenant isolation
- Embedding storage for semantic search
- Query optimization
"""

import json
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncpg
import numpy as np

from .ontology import EntityNode, RelationshipEdge, EntityType, RelationshipType
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.services.connection_pool_manager import ConnectionPoolManager

logger = logging.getLogger(__name__)

class KnowledgeGraphStore:
    """
    PostgreSQL-based Knowledge Graph storage with pgvector integration
    
    Features:
    - Entity and relationship tables
    - Multi-tenant isolation with RLS
    - Embedding storage for semantic search
    - Governance metadata tracking
    - Query optimization
    """
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
        self.initialized = False
    
    async def initialize(self):
        """Initialize KG storage schema"""
        if self.initialized:
            return
            
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await self._create_kg_schema(conn)
                await self._setup_rls_policies(conn)
                await self._create_indexes(conn)
                
            self.initialized = True
            logger.info(" Knowledge Graph storage initialized")
            
        except Exception as e:
            logger.error(f" Failed to initialize KG storage: {e}")
            raise
    
    async def _create_kg_schema(self, conn: asyncpg.Connection):
        """Create Knowledge Graph database schema"""
        
        # Knowledge Graph Entities table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                properties JSONB NOT NULL,
                tenant_id INTEGER NOT NULL,
                region TEXT,
                policy_pack_id TEXT,
                evidence_pack_id TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                -- Embedding for semantic search (using TEXT as fallback if pgvector not available)
                embedding TEXT,
                
                -- Search optimization
                -- Search text for full-text search (Azure compatible)
                search_text TEXT
            );
        """)
        
        # Knowledge Graph Relationships table  
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                relationship_type TEXT NOT NULL,
                properties JSONB NOT NULL DEFAULT '{}',
                tenant_id INTEGER NOT NULL,
                confidence_score FLOAT,
                evidence_pack_id TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                -- Ensure no duplicate relationships
                UNIQUE(source_id, target_id, relationship_type, tenant_id)
            );
        """)
        
        # Knowledge Graph Execution Traces table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_execution_traces (
                trace_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                tenant_id INTEGER NOT NULL,
                
                -- Trace data
                inputs JSONB NOT NULL,
                outputs JSONB NOT NULL,
                governance_metadata JSONB NOT NULL,
                
                -- Performance metrics
                execution_time_ms INTEGER,
                trust_score FLOAT,
                cost_tokens INTEGER,
                
                -- Governance
                policy_pack_id TEXT,
                evidence_pack_id TEXT,
                override_id TEXT,
                
                -- Linked entities
                entities_affected TEXT[] DEFAULT '{}',
                relationships_created TEXT[] DEFAULT '{}',
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Knowledge Graph Query Log for analytics
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kg_query_log (
                query_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
                tenant_id INTEGER NOT NULL,
                user_id TEXT,
                query_type TEXT NOT NULL,
                query_params JSONB NOT NULL,
                results_count INTEGER,
                execution_time_ms INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        logger.info(" Knowledge Graph schema created")
    
    async def _setup_rls_policies(self, conn: asyncpg.Connection):
        """Setup Row Level Security for multi-tenant isolation"""
        
        # Enable RLS on all KG tables
        tables = ['kg_entities', 'kg_relationships', 'kg_execution_traces', 'kg_query_log']
        
        for table in tables:
            await conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
            
            # Create tenant isolation policy (drop first if exists)
            try:
                await conn.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table};")
            except:
                pass  # Policy might not exist
            
            await conn.execute(f"""
                CREATE POLICY {table}_tenant_isolation ON {table}
                USING (tenant_id = current_setting('app.current_tenant_id', true)::INTEGER);
            """)
        
        logger.info("🔒 Knowledge Graph RLS policies created")
    
    async def _create_indexes(self, conn: asyncpg.Connection):
        """Create optimized indexes for KG queries"""
        
        # Entity indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_entities_tenant_type 
            ON kg_entities(tenant_id, entity_type);
        """)
        
        # Use simple text index for Azure compatibility (no pg_trgm)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_entities_search_text 
            ON kg_entities(search_text);
        """)
        
        # Create vector index only if pgvector extension is available
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_entities_embedding 
                ON kg_entities USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            logger.info(" Vector index created successfully")
        except Exception as e:
            logger.warning(f" Vector index creation skipped: {e}")
            # Create a regular index as fallback
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_entities_embedding_fallback 
                ON kg_entities(id);
            """)
        
        # Relationship indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_relationships_source 
            ON kg_relationships(tenant_id, source_id, relationship_type);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_relationships_target 
            ON kg_relationships(tenant_id, target_id, relationship_type);
        """)
        
        # Trace indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_traces_workflow 
            ON kg_execution_traces(tenant_id, workflow_id, created_at DESC);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_traces_entities 
            ON kg_execution_traces USING GIN(entities_affected);
        """)
        
        logger.info(" Knowledge Graph indexes created")
    
    async def store_entity(self, entity: EntityNode, embedding: Optional[np.ndarray] = None) -> bool:
        """Store an entity node in the Knowledge Graph"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", entity.tenant_id)
                
                await conn.execute("""
                    INSERT INTO kg_entities (
                        id, entity_type, properties, tenant_id, region, 
                        policy_pack_id, evidence_pack_id, embedding, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (id) DO UPDATE SET
                        properties = EXCLUDED.properties,
                        updated_at = EXCLUDED.updated_at,
                        embedding = EXCLUDED.embedding
                """, 
                    entity.id,
                    entity.entity_type.value,
                    json.dumps(entity.properties),
                    entity.tenant_id,
                    entity.region,
                    entity.policy_pack_id,
                    entity.evidence_pack_id,
                    embedding.tolist() if embedding is not None else None,
                    entity.created_at,
                    entity.updated_at
                )
                
            logger.debug(f" Stored entity: {entity.id} ({entity.entity_type.value})")
            return True
            
        except Exception as e:
            logger.error(f" Failed to store entity {entity.id}: {e}")
            return False
    
    async def store_relationship(self, relationship: RelationshipEdge) -> bool:
        """Store a relationship edge in the Knowledge Graph"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", relationship.tenant_id)
                
                await conn.execute("""
                    INSERT INTO kg_relationships (
                        id, source_id, target_id, relationship_type, properties,
                        tenant_id, confidence_score, evidence_pack_id, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (source_id, target_id, relationship_type, tenant_id) 
                    DO UPDATE SET
                        properties = EXCLUDED.properties,
                        confidence_score = EXCLUDED.confidence_score,
                        evidence_pack_id = EXCLUDED.evidence_pack_id
                """,
                    relationship.id,
                    relationship.source_id,
                    relationship.target_id,
                    relationship.relationship_type.value,
                    json.dumps(relationship.properties),
                    relationship.tenant_id,
                    relationship.confidence_score,
                    relationship.evidence_pack_id,
                    relationship.created_at
                )
                
            logger.debug(f" Stored relationship: {relationship.source_id} -[{relationship.relationship_type.value}]-> {relationship.target_id}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to store relationship {relationship.id}: {e}")
            return False
    
    async def get_entity(self, entity_id: str, tenant_id: str) -> Optional[EntityNode]:
        """Retrieve an entity by ID"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                row = await conn.fetchrow("""
                    SELECT id, entity_type, properties, tenant_id, region,
                           policy_pack_id, evidence_pack_id, created_at, updated_at
                    FROM kg_entities WHERE id = $1
                """, entity_id)
                
                if not row:
                    return None
                
                return EntityNode(
                    id=row['id'],
                    entity_type=EntityType(row['entity_type']),
                    properties=row['properties'],
                    tenant_id=row['tenant_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    region=row['region'],
                    policy_pack_id=row['policy_pack_id'],
                    evidence_pack_id=row['evidence_pack_id']
                )
                
        except Exception as e:
            logger.error(f" Failed to get entity {entity_id}: {e}")
            return None
    
    async def find_entities(self, tenant_id: str, entity_type: Optional[EntityType] = None,
                          search_text: Optional[str] = None, limit: int = 100) -> List[EntityNode]:
        """Find entities with optional filtering"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                query = """
                    SELECT id, entity_type, properties, tenant_id, region,
                           policy_pack_id, evidence_pack_id, created_at, updated_at
                    FROM kg_entities WHERE 1=1
                """
                params = []
                
                if entity_type:
                    query += f" AND entity_type = ${len(params) + 1}"
                    params.append(entity_type.value)
                
                if search_text:
                    query += f" AND search_text ILIKE ${len(params) + 1}"
                    params.append(f"%{search_text}%")
                
                query += f" ORDER BY updated_at DESC LIMIT ${len(params) + 1}"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                entities = []
                for row in rows:
                    entities.append(EntityNode(
                        id=row['id'],
                        entity_type=EntityType(row['entity_type']),
                        properties=row['properties'],
                        tenant_id=row['tenant_id'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        region=row['region'],
                        policy_pack_id=row['policy_pack_id'],
                        evidence_pack_id=row['evidence_pack_id']
                    ))
                
                return entities
                
        except Exception as e:
            logger.error(f" Failed to find entities: {e}")
            return []
    
    async def get_relationships(self, entity_id: str, tenant_id: str, 
                              relationship_type: Optional[RelationshipType] = None,
                              direction: str = 'both') -> List[RelationshipEdge]:
        """Get relationships for an entity"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                conditions = []
                params = [entity_id]
                
                if direction in ['out', 'both']:
                    conditions.append(f"source_id = ${len(params)}")
                
                if direction in ['in', 'both']:
                    if conditions:
                        conditions.append(f"OR target_id = ${len(params)}")
                    else:
                        conditions.append(f"target_id = ${len(params)}")
                
                if relationship_type:
                    params.append(relationship_type.value)
                    conditions.append(f"AND relationship_type = ${len(params)}")
                
                query = f"""
                    SELECT id, source_id, target_id, relationship_type, properties,
                           tenant_id, confidence_score, evidence_pack_id, created_at
                    FROM kg_relationships 
                    WHERE ({' '.join(conditions)})
                    ORDER BY created_at DESC
                """
                
                rows = await conn.fetch(query, *params)
                
                relationships = []
                for row in rows:
                    relationships.append(RelationshipEdge(
                        id=row['id'],
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        relationship_type=RelationshipType(row['relationship_type']),
                        properties=row['properties'],
                        tenant_id=row['tenant_id'],
                        created_at=row['created_at'],
                        confidence_score=row['confidence_score'],
                        evidence_pack_id=row['evidence_pack_id']
                    ))
                
                return relationships
                
        except Exception as e:
            logger.error(f" Failed to get relationships for {entity_id}: {e}")
            return []
    
    async def semantic_search(self, query_embedding: np.ndarray, tenant_id: str,
                            entity_type: Optional[EntityType] = None,
                            limit: int = 10) -> List[Tuple[EntityNode, float]]:
        """Perform semantic search using embeddings"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                query = """
                    SELECT id, entity_type, properties, tenant_id, region,
                           policy_pack_id, evidence_pack_id, created_at, updated_at,
                           (embedding <=> $1) as distance
                    FROM kg_entities 
                    WHERE embedding IS NOT NULL
                """
                params = [query_embedding.tolist()]
                
                if entity_type:
                    query += f" AND entity_type = ${len(params) + 1}"
                    params.append(entity_type.value)
                
                query += f" ORDER BY distance LIMIT ${len(params) + 1}"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                results = []
                for row in rows:
                    entity = EntityNode(
                        id=row['id'],
                        entity_type=EntityType(row['entity_type']),
                        properties=row['properties'],
                        tenant_id=row['tenant_id'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        region=row['region'],
                        policy_pack_id=row['policy_pack_id'],
                        evidence_pack_id=row['evidence_pack_id']
                    )
                    similarity = 1.0 - row['distance']  # Convert distance to similarity
                    results.append((entity, similarity))
                
                return results
                
        except Exception as e:
            logger.error(f" Failed semantic search: {e}")
            return []
    
    async def get_knowledge_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get Knowledge Graph statistics for a tenant"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute("SELECT set_config('app.current_tenant', $1, false)", tenant_id)
                
                # Entity counts by type
                entity_counts = await conn.fetch("""
                    SELECT entity_type, COUNT(*) as count
                    FROM kg_entities
                    GROUP BY entity_type
                    ORDER BY count DESC
                """)
                
                # Relationship counts by type
                relationship_counts = await conn.fetch("""
                    SELECT relationship_type, COUNT(*) as count
                    FROM kg_relationships
                    GROUP BY relationship_type
                    ORDER BY count DESC
                """)
                
                # Recent activity
                recent_entities = await conn.fetchval("""
                    SELECT COUNT(*) FROM kg_entities 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                
                recent_traces = await conn.fetchval("""
                    SELECT COUNT(*) FROM kg_execution_traces 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    'entity_counts': {row['entity_type']: row['count'] for row in entity_counts},
                    'relationship_counts': {row['relationship_type']: row['count'] for row in relationship_counts},
                    'total_entities': sum(row['count'] for row in entity_counts),
                    'total_relationships': sum(row['count'] for row in relationship_counts),
                    'recent_activity': {
                        'new_entities_24h': recent_entities,
                        'new_traces_24h': recent_traces
                    }
                }
                
        except Exception as e:
            logger.error(f" Failed to get KG stats: {e}")
            return {}
