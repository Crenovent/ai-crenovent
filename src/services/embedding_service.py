"""
Real Embedding Service for AI agents with OpenAI integration
Handles vector embeddings for knowledge management and retrieval using pgvector
"""

import asyncio
import json
import logging
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncpg
from openai import AsyncAzureOpenAI

@dataclass
class EmbeddingResult:
    success: bool
    embedding_id: Optional[str] = None
    error_message: Optional[str] = None
    similarity_score: Optional[float] = None

@dataclass
class SearchResult:
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    source_type: str  # 'plan', 'conversation', 'account'

class EmbeddingService:
    """
    Optimized embedding service using shared connection pool
    Handles strategic planning data embeddings with pgvector only
    """
    
    def __init__(self):
        from .connection_pool_manager import pool_manager
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536
    
    async def initialize(self):
        """Initialize using shared connection pool manager"""
        try:
            if not self.initialized:
                # Ensure pool manager is initialized
                if not self.pool_manager._initialized:
                    await self.pool_manager.initialize()
                
                self.initialized = True
                self.logger.info("✅ Embedding service initialized with shared connection pool")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize embedding service: {str(e)}")
            raise
    
    async def _test_connections(self):
        """Test OpenAI and PostgreSQL connections"""
        try:
            # Test OpenAI connection
            test_embedding = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=["test connection"]
            )
            
            # Test PostgreSQL connection
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            self.logger.info("✅ Connection tests passed")
            
        except Exception as e:
            self.logger.error(f"❌ Connection test failed: {str(e)}")
            raise
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using shared OpenAI client"""
        try:
            return await self.pool_manager.create_openai_embedding(text, self.embedding_model)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create embedding: {str(e)}")
            raise
    
    def _create_content_hash(self, content: str) -> str:
        """Create SHA-256 hash for content deduplication"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def store_plan_embedding(self, 
                                   plan_id: int,
                                   content_type: str,
                                   content_text: str,
                                   metadata: Dict[str, Any],
                                   tenant_id: int,
                                   user_id: int) -> EmbeddingResult:
        """Store plan content as embedding in pgvector"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Create content hash for deduplication
            content_hash = self._create_content_hash(content_text)
            
            # Check if embedding already exists
            existing = await self.pool_manager.execute_postgres_fetchrow(
                "SELECT id FROM plan_embeddings WHERE content_hash = $1",
                content_hash
            )
            
            if existing:
                self.logger.info(f"Embedding already exists for plan {plan_id}, content type {content_type}")
                return EmbeddingResult(
                    success=True,
                    embedding_id=str(existing['id'])
                )
            
            # Create new embedding
            embedding = await self.create_embedding(content_text)
            
            # Store in database
            embedding_id = await self.pool_manager.execute_postgres_fetchval("""
                INSERT INTO plan_embeddings 
                (plan_id, content_type, embedding, content_text, content_hash, metadata, tenant_id, created_by_user_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """, plan_id, content_type, embedding, content_text, content_hash, json.dumps(metadata), tenant_id, user_id)
            
            self.logger.info(f"✅ Stored embedding {embedding_id} for plan {plan_id}")
            
            return EmbeddingResult(
                success=True,
                embedding_id=str(embedding_id)
            )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store plan embedding: {str(e)}")
            return EmbeddingResult(
                success=False,
                error_message=str(e)
            )
    
    async def store_conversation_embedding(self,
                                         user_id: int,
                                         session_id: str,
                                         message_text: str,
                                         response_text: str,
                                         intent_category: str,
                                         confidence_score: float,
                                         metadata: Dict[str, Any],
                                         tenant_id: int) -> EmbeddingResult:
        """Store conversation as embedding for learning"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Combine message and response for embedding
            combined_text = f"User: {message_text}\nAssistant: {response_text}"
            embedding = await self.create_embedding(combined_text)
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                embedding_id = await conn.fetchval("""
                    INSERT INTO conversation_embeddings 
                    (user_id, session_id, message_text, response_text, embedding, intent_category, confidence_score, metadata, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """, user_id, session_id, message_text, response_text, embedding, intent_category, confidence_score, json.dumps(metadata), tenant_id)
                
                self.logger.info(f"✅ Stored conversation embedding {embedding_id}")
                
                return EmbeddingResult(
                    success=True,
                    embedding_id=str(embedding_id)
                )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store conversation embedding: {str(e)}")
            return EmbeddingResult(
                success=False,
                error_message=str(e)
            )
    
    async def search_similar_plans(self,
                                  query_text: str,
                                  tenant_id: int,
                                  similarity_threshold: float = 0.7,
                                  max_results: int = 10,
                                  content_types: List[str] = None) -> List[SearchResult]:
        """Search for similar plans using vector similarity"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Create embedding for query
            query_embedding = await self.create_embedding(query_text)
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Use the similarity function we created in the schema
                if content_types:
                    content_filter = "AND pe.content_type = ANY($4)"
                    rows = await conn.fetch(f"""
                        SELECT 
                            pe.plan_id,
                            pe.content_type,
                            pe.content_text,
                            (1 - (pe.embedding <=> $1))::DECIMAL AS similarity_score,
                            pe.metadata
                        FROM plan_embeddings pe
                        WHERE 
                            pe.tenant_id = $2
                            AND (1 - (pe.embedding <=> $1)) >= $3
                            {content_filter}
                        ORDER BY pe.embedding <=> $1
                        LIMIT {max_results}
                    """, query_embedding, tenant_id, similarity_threshold, content_types)
                else:
                    rows = await conn.fetch("""
                        SELECT 
                            pe.plan_id,
                            pe.content_type,
                            pe.content_text,
                            (1 - (pe.embedding <=> $1))::DECIMAL AS similarity_score,
                            pe.metadata
                        FROM plan_embeddings pe
                        WHERE 
                            pe.tenant_id = $2
                            AND (1 - (pe.embedding <=> $1)) >= $3
                        ORDER BY pe.embedding <=> $1
                        LIMIT $4
                    """, query_embedding, tenant_id, similarity_threshold, max_results)
                
                results = []
                for row in rows:
                    results.append(SearchResult(
                        content=row['content_text'],
                        similarity_score=float(row['similarity_score']),
                        metadata={
                            'plan_id': row['plan_id'],
                            'content_type': row['content_type'],
                            **json.loads(row['metadata'] or '{}')
                        },
                        source_type='plan'
                    ))
                
                self.logger.info(f"✅ Found {len(results)} similar plans for query")
                return results
                
        except Exception as e:
            self.logger.error(f"❌ Failed to search similar plans: {str(e)}")
            return []
    
    async def search_similar_conversations(self,
                                         query_text: str,
                                         user_id: int = None,
                                         similarity_threshold: float = 0.6,
                                         max_results: int = 5) -> List[SearchResult]:
        """Search for similar past conversations"""
        try:
            if not self.initialized:
                await self.initialize()
            
            query_embedding = await self.create_embedding(query_text)
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                if user_id:
                    rows = await conn.fetch("""
                        SELECT 
                            ce.message_text,
                            ce.response_text,
                            ce.intent_category,
                            (1 - (ce.embedding <=> $1))::DECIMAL AS similarity_score,
                            ce.metadata,
                            ce.created_at
                        FROM conversation_embeddings ce
                        WHERE 
                            ce.user_id = $2
                            AND (1 - (ce.embedding <=> $1)) >= $3
                        ORDER BY ce.embedding <=> $1
                        LIMIT $4
                    """, query_embedding, user_id, similarity_threshold, max_results)
                else:
                    rows = await conn.fetch("""
                        SELECT 
                            ce.message_text,
                            ce.response_text,
                            ce.intent_category,
                            (1 - (ce.embedding <=> $1))::DECIMAL AS similarity_score,
                            ce.metadata,
                            ce.created_at
                        FROM conversation_embeddings ce
                        WHERE (1 - (ce.embedding <=> $1)) >= $2
                        ORDER BY ce.embedding <=> $1
                        LIMIT $3
                    """, query_embedding, similarity_threshold, max_results)
                
                results = []
                for row in rows:
                    results.append(SearchResult(
                        content=f"Previous Q: {row['message_text']}\nPrevious A: {row['response_text']}",
                        similarity_score=float(row['similarity_score']),
                        metadata={
                            'intent_category': row['intent_category'],
                            'created_at': row['created_at'].isoformat(),
                            **json.loads(row['metadata'] or '{}')
                        },
                        source_type='conversation'
                    ))
                
                return results
                
        except Exception as e:
            self.logger.error(f"❌ Failed to search similar conversations: {str(e)}")
            return []
    
    async def get_plan_context(self, plan_id: int, tenant_id: int) -> Dict[str, Any]:
        """Get all embeddings and context for a specific plan"""
        try:
            if not self.initialized:
                await self.initialize()
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT content_type, content_text, metadata, created_at
                    FROM plan_embeddings
                    WHERE plan_id = $1 AND tenant_id = $2
                    ORDER BY content_type
                """, plan_id, tenant_id)
                
                context = {}
                for row in rows:
                    context[row['content_type']] = {
                        'content': row['content_text'],
                        'metadata': json.loads(row['metadata'] or '{}'),
                        'created_at': row['created_at'].isoformat()
                    }
                
                return context
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get plan context: {str(e)}")
            return {}
    
    async def create_workflow_embeddings(self, 
                                       workflow_name: str, 
                                       data: Dict[str, Any],
                                       user_id: str,
                                       tenant_id: str,
                                       confidence_score: float) -> EmbeddingResult:
        """Legacy method for compatibility - redirects to conversation embeddings"""
        try:
            workflow_text = f"Workflow: {workflow_name}\nData: {json.dumps(data, indent=2)}"
            
            return await self.store_conversation_embedding(
                user_id=int(user_id),
                session_id=f"workflow_{workflow_name}",
                message_text=f"Execute workflow: {workflow_name}",
                response_text=workflow_text,
                intent_category="workflow",
                confidence_score=confidence_score,
                metadata={"workflow_name": workflow_name, "data": data},
                tenant_id=int(tenant_id)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error creating workflow embeddings: {str(e)}")
            return EmbeddingResult(
                success=False,
                error_message=str(e)
            )
    
    async def create_conversation_embedding(self,
                                          user_message: str,
                                          ai_response: str,
                                          context: Dict[str, Any],
                                          user_id: str,
                                          tenant_id: str) -> EmbeddingResult:
        """Legacy method for compatibility"""
        try:
            session_id = context.get('session_id', f"session_{datetime.now().timestamp()}")
            intent_category = context.get('intent_category', 'general')
            confidence_score = context.get('confidence_score', 0.8)
            
            return await self.store_conversation_embedding(
                user_id=int(user_id),
                session_id=session_id,
                message_text=user_message,
                response_text=ai_response,
                intent_category=intent_category,
                confidence_score=confidence_score,
                metadata=context,
                tenant_id=int(tenant_id)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error creating conversation embedding: {str(e)}")
            return EmbeddingResult(
                success=False,
                error_message=str(e)
            )
    
    async def close(self):
        """Clean up connections"""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            if self.openai_client:
                await self.openai_client.close()
            self.logger.info("✅ Embedding service connections closed")
        except Exception as e:
            self.logger.error(f"❌ Error closing embedding service: {str(e)}")