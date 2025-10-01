#!/usr/bin/env python3
"""
Embeddings and Policy-Aware RAG Implementation
==============================================

Implements the Knowledge Graph embeddings pipeline and policy-aware RAG system
as specified in the Vision Doc Chapter 8.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncpg
from sentence_transformers import SentenceTransformer
import openai
from config.database import DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    entity_id: str
    entity_type: str
    embedding: List[float]
    metadata: Dict[str, Any]
    tenant_id: str

@dataclass
class RAGResult:
    """Result of RAG retrieval"""
    entity_id: str
    entity_type: str
    content: str
    similarity_score: float
    policy_compliant: bool
    metadata: Dict[str, Any]

class KnowledgeGraphEmbeddings:
    """
    Generates and manages embeddings for Knowledge Graph entities
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.db_config = DatabaseConfig()
        
    async def generate_entity_embeddings(self, tenant_id: str) -> List[EmbeddingResult]:
        """Generate embeddings for all entities in tenant's KG"""
        embeddings = []
        
        async with asyncpg.create_pool(self.db_config.postgres_url) as pool:
            async with pool.acquire() as conn:
                # Fetch all entities for tenant
                rows = await conn.fetch("""
                    SELECT id, entity_type, properties, tenant_id, region, policy_pack_id
                    FROM kg_entities 
                    WHERE tenant_id = $1
                """, tenant_id)
                
                for row in rows:
                    # Create searchable text from entity properties
                    searchable_text = self._create_searchable_text(
                        row['entity_type'], 
                        json.loads(row['properties'])
                    )
                    
                    # Generate embedding
                    embedding = self.model.encode(searchable_text).tolist()
                    
                    result = EmbeddingResult(
                        entity_id=row['id'],
                        entity_type=row['entity_type'],
                        embedding=embedding,
                        metadata={
                            'region': row['region'],
                            'policy_pack_id': row['policy_pack_id'],
                            'searchable_text': searchable_text
                        },
                        tenant_id=row['tenant_id']
                    )
                    embeddings.append(result)
                    
        return embeddings
    
    def _create_searchable_text(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Create searchable text representation of entity"""
        
        if entity_type == 'User':
            return f"User: {properties.get('name', '')} Role: {properties.get('role', '')} Email: {properties.get('email', '')} Permissions: {', '.join(properties.get('permissions', []))}"
            
        elif entity_type == 'Account':
            return f"Account: {properties.get('name', '')} Industry: {properties.get('industry', '')} Tier: {properties.get('tier', '')} ARR: ${properties.get('arr', 0):,} Health Score: {properties.get('health_score', 0)}"
            
        elif entity_type == 'Opportunity':
            return f"Opportunity: {properties.get('name', '')} Stage: {properties.get('stage', '')} Amount: ${properties.get('amount', 0):,} Probability: {properties.get('probability', 0)*100}% Account: {properties.get('account_name', '')}"
            
        elif entity_type == 'Forecast':
            return f"Forecast: {properties.get('period', '')} Region: {properties.get('region', '')} Category: {properties.get('category', '')} Amount: ${properties.get('amount', 0):,} Confidence: {properties.get('confidence', 0)*100}%"
            
        elif entity_type == 'Quota':
            return f"Quota: {properties.get('period', '')} Owner: {properties.get('owner_name', '')} Amount: ${properties.get('quota_amount', 0):,} Attainment: {properties.get('attainment', 0)*100}% Territory: {properties.get('territory', '')}"
            
        elif entity_type == 'CompensationPlan':
            return f"Compensation Plan: {properties.get('plan_name', '')} Base Salary: ${properties.get('base_salary', 0):,} Commission Rate: {properties.get('commission_rate', 0)*100}% Territory: {properties.get('territory', '')}"
            
        else:
            # Generic fallback
            return f"{entity_type}: {json.dumps(properties, separators=(',', ':'))}"
    
    async def store_embeddings(self, embeddings: List[EmbeddingResult]) -> bool:
        """Store embeddings in database with pgvector"""
        try:
            async with asyncpg.create_pool(self.db_config.postgres_url) as pool:
                async with pool.acquire() as conn:
                    # Create embeddings table if not exists
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS kg_entity_embeddings (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            entity_id TEXT NOT NULL,
                            entity_type TEXT NOT NULL,
                            tenant_id UUID NOT NULL,
                            embedding vector(384),
                            metadata JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_kg_embeddings_tenant 
                        ON kg_entity_embeddings(tenant_id);
                        
                        CREATE INDEX IF NOT EXISTS idx_kg_embeddings_type 
                        ON kg_entity_embeddings(entity_type);
                        
                        CREATE INDEX IF NOT EXISTS idx_kg_embeddings_vector 
                        ON kg_entity_embeddings USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    # Insert embeddings
                    for emb in embeddings:
                        await conn.execute("""
                            INSERT INTO kg_entity_embeddings 
                            (entity_id, entity_type, tenant_id, embedding, metadata)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (entity_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                updated_at = NOW()
                        """, emb.entity_id, emb.entity_type, emb.tenant_id, 
                             emb.embedding, json.dumps(emb.metadata))
                    
            logger.info(f"✅ Stored {len(embeddings)} embeddings successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store embeddings: {e}")
            return False

class PolicyAwareRAG:
    """
    Policy-aware RAG system with tenant isolation and compliance filtering
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.db_config = DatabaseConfig()
    
    async def search(
        self, 
        query: str, 
        tenant_id: str, 
        user_context: Dict[str, Any],
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[RAGResult]:
        """
        Perform policy-aware semantic search
        """
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        async with asyncpg.create_pool(self.db_config.postgres_url) as pool:
            async with pool.acquire() as conn:
                # Get user's policy constraints
                policy_constraints = await self._get_policy_constraints(
                    conn, tenant_id, user_context
                )
                
                # Perform vector similarity search with policy filtering
                results = await conn.fetch("""
                    SELECT 
                        e.entity_id,
                        e.entity_type,
                        e.metadata->>'searchable_text' as content,
                        1 - (e.embedding <=> $1::vector) as similarity_score,
                        e.metadata,
                        kg.properties,
                        kg.region,
                        kg.policy_pack_id
                    FROM kg_entity_embeddings e
                    JOIN kg_entities kg ON e.entity_id = kg.id
                    WHERE e.tenant_id = $2
                      AND 1 - (e.embedding <=> $1::vector) > $3
                      AND ($4 = 'global' OR kg.region = $4 OR kg.region IS NULL)
                    ORDER BY e.embedding <=> $1::vector
                    LIMIT $5
                """, query_embedding, tenant_id, similarity_threshold, 
                     user_context.get('region', 'global'), limit)
                
                rag_results = []
                for row in results:
                    # Check policy compliance
                    is_compliant = await self._check_policy_compliance(
                        conn, row, user_context, policy_constraints
                    )
                    
                    result = RAGResult(
                        entity_id=row['entity_id'],
                        entity_type=row['entity_type'],
                        content=row['content'] or '',
                        similarity_score=float(row['similarity_score']),
                        policy_compliant=is_compliant,
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    )
                    rag_results.append(result)
                
                return rag_results
    
    async def _get_policy_constraints(
        self, 
        conn: asyncpg.Connection, 
        tenant_id: str, 
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get applicable policy constraints for user"""
        
        # Get policy packs for tenant
        policy_packs = await conn.fetch("""
            SELECT policy_pack_name, policy_rules, enforcement_level
            FROM ro_policy_packs 
            WHERE tenant_id = $1 AND enforcement_level != 'DISABLED'
        """, int(tenant_id) if tenant_id.isdigit() else tenant_id)
        
        constraints = {
            'allowed_regions': user_context.get('allowed_regions', ['global']),
            'permission_level': user_context.get('permissions', ['read_only']),
            'data_classification': user_context.get('data_classification', 'internal'),
            'policy_packs': [dict(p) for p in policy_packs]
        }
        
        return constraints
    
    async def _check_policy_compliance(
        self,
        conn: asyncpg.Connection,
        entity_row: Dict[str, Any],
        user_context: Dict[str, Any],
        policy_constraints: Dict[str, Any]
    ) -> bool:
        """Check if entity access complies with policies"""
        
        # Basic region check
        entity_region = entity_row.get('region', 'global')
        user_regions = user_context.get('allowed_regions', ['global'])
        
        if entity_region not in user_regions and 'global' not in user_regions:
            return False
        
        # Permission level check
        user_permissions = user_context.get('permissions', [])
        
        # Financial data requires financial permissions
        if entity_row['entity_type'] in ['CompensationPlan', 'Quota'] and 'financial' not in user_permissions:
            return False
        
        # Account data requires appropriate permissions
        if entity_row['entity_type'] == 'Account' and not any(p in user_permissions for p in ['all_data', 'sales_data']):
            return False
        
        return True

class EmbeddingsManager:
    """
    Main manager for embeddings and RAG operations
    """
    
    def __init__(self):
        self.kg_embeddings = KnowledgeGraphEmbeddings()
        self.rag_system = PolicyAwareRAG()
    
    async def initialize_tenant_embeddings(self, tenant_id: str) -> bool:
        """Initialize embeddings for a tenant"""
        try:
            logger.info(f"🔄 Generating embeddings for tenant {tenant_id}")
            
            # Generate embeddings for all entities
            embeddings = await self.kg_embeddings.generate_entity_embeddings(tenant_id)
            
            if not embeddings:
                logger.warning(f"⚠️ No entities found for tenant {tenant_id}")
                return False
            
            # Store embeddings
            success = await self.kg_embeddings.store_embeddings(embeddings)
            
            if success:
                logger.info(f"✅ Successfully initialized {len(embeddings)} embeddings for tenant {tenant_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings for tenant {tenant_id}: {e}")
            return False
    
    async def search_knowledge(
        self,
        query: str,
        tenant_id: str,
        user_context: Dict[str, Any],
        limit: int = 10
    ) -> List[RAGResult]:
        """Search knowledge graph with policy awareness"""
        return await self.rag_system.search(query, tenant_id, user_context, limit)

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

async def main():
    """Main function for testing embeddings and RAG"""
    
    embeddings_manager = EmbeddingsManager()
    
    # Initialize embeddings for tenant 1300
    tenant_id = "1300"
    
    print("🚀 Initializing Knowledge Graph Embeddings...")
    success = await embeddings_manager.initialize_tenant_embeddings(tenant_id)
    
    if not success:
        print("❌ Failed to initialize embeddings")
        return
    
    # Test RAG search
    print("\n🔍 Testing Policy-Aware RAG Search...")
    
    test_queries = [
        "Show me Microsoft account information",
        "What are the Q4 forecast numbers for Americas?", 
        "Find opportunities in negotiation stage",
        "Show me Ronald's quota attainment",
        "What accounts are at risk?"
    ]
    
    user_context = {
        'user_id': 1316,
        'role': 'CRO',
        'permissions': ['all_data', 'financial'],
        'region': 'americas',
        'allowed_regions': ['americas', 'global']
    }
    
    for query in test_queries:
        print(f"\n📊 Query: '{query}'")
        results = await embeddings_manager.search_knowledge(
            query, tenant_id, user_context, limit=3
        )
        
        for i, result in enumerate(results, 1):
            compliance = "✅" if result.policy_compliant else "❌"
            print(f"  {i}. {compliance} {result.entity_type}: {result.content[:100]}...")
            print(f"     Similarity: {result.similarity_score:.3f}")
    
    print("\n🎉 Embeddings and RAG system testing complete!")

if __name__ == "__main__":
    asyncio.run(main())


