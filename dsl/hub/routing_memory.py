"""
Routing Memory System - Enterprise Caching Layer
================================================

Implements persistent routing cache with semantic similarity lookup
to reduce LLM calls by 70-90% while maintaining accuracy.

Based on Vision Doc Chapter 18 - Memory-Based Routing Architecture
"""

import asyncio
import json
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class CachedRoutingDecision:
    """Cached routing decision with metadata"""
    user_input: str
    input_hash: str
    workflow_category: str
    parameters: Dict[str, Any]
    confidence: float
    embedding: Optional[np.ndarray] = None
    created_at: datetime = None
    last_used: datetime = None
    usage_count: int = 0
    success_rate: float = 1.0
    tenant_id: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_used is None:
            self.last_used = datetime.now()

@dataclass
class RoutingStats:
    """Routing performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    pattern_matches: int = 0
    llm_fallbacks: int = 0
    avg_response_time_ms: float = 0.0
    cost_savings: float = 0.0

class RoutingMemorySystem:
    """
    Enterprise-grade routing memory with semantic caching
    
    Features:
    - Semantic similarity matching using local embeddings
    - Tenant-scoped caching for multi-tenancy
    - Performance tracking and cost optimization
    - Automatic cache eviction and cleanup
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Local embedding model (no API costs)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # In-memory cache (TODO: Replace with Redis for production)
        self.cache: Dict[str, CachedRoutingDecision] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Configuration
        self.similarity_threshold = 0.75  # Lowered threshold for better hit rates (was 0.85)
        self.max_cache_size = 10000       # Maximum cached decisions
        self.cache_ttl_hours = 24         # Cache TTL in hours
        
        # Performance tracking
        self.stats = RoutingStats()
        
        self.logger.info("🧠 Routing Memory System initialized with local embeddings")
    
    async def initialize(self):
        """Initialize the memory system"""
        try:
            # Load existing cache from database if available
            await self._load_cache_from_db()
            self.logger.info(f"✅ Routing Memory initialized with {len(self.cache)} cached decisions")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load cache from DB: {e}")
    
    async def lookup_routing_decision(
        self, 
        user_input: str, 
        tenant_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Look up cached routing decision using semantic similarity
        
        Returns:
            Cached routing decision if found, None otherwise
        """
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Generate input hash for exact matches
            input_hash = self._generate_input_hash(user_input, tenant_id)
            
            # Try exact hash match first (fastest)
            if input_hash in self.cache:
                cached_decision = self.cache[input_hash]
                if not self._is_cache_expired(cached_decision):
                    await self._update_cache_usage(cached_decision)
                    self.stats.cache_hits += 1
                    
                    self.logger.info(f"🎯 EXACT cache hit for: '{user_input[:50]}...' -> {cached_decision.workflow_category}")
                    return {
                        'workflow_category': cached_decision.workflow_category,
                        'parameters': cached_decision.parameters,
                        'confidence': cached_decision.confidence,
                        'cache_hit': True,
                        'cache_type': 'exact_match'
                    }
            
            # Try semantic similarity matching
            query_embedding = self.embedding_model.encode(user_input)
            best_match = await self._find_semantic_match(query_embedding, tenant_id)
            
            if best_match:
                cached_decision, similarity = best_match
                await self._update_cache_usage(cached_decision)
                self.stats.cache_hits += 1
                
                self.logger.info(f"🎯 Semantic cache hit (similarity: {similarity:.3f}) for: {user_input[:50]}...")
                return {
                    'workflow_category': cached_decision.workflow_category,
                    'parameters': cached_decision.parameters,
                    'confidence': cached_decision.confidence * similarity,  # Adjust confidence by similarity
                    'cache_hit': True,
                    'cache_type': 'semantic',
                    'similarity': similarity
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error in cache lookup: {e}")
            return None
        finally:
            response_time = (time.time() - start_time) * 1000
            self.stats.avg_response_time_ms = (
                (self.stats.avg_response_time_ms * (self.stats.total_requests - 1) + response_time) 
                / self.stats.total_requests
            )
    
    async def store_routing_decision(
        self,
        user_input: str,
        workflow_category: str,
        parameters: Dict[str, Any],
        confidence: float,
        tenant_id: str = None
    ):
        """Store successful routing decision in cache"""
        try:
            input_hash = self._generate_input_hash(user_input, tenant_id)
            
            # Generate embedding for semantic matching
            embedding = self.embedding_model.encode(user_input)
            
            cached_decision = CachedRoutingDecision(
                user_input=user_input,
                input_hash=input_hash,
                workflow_category=workflow_category,
                parameters=parameters,
                confidence=confidence,
                embedding=embedding,
                tenant_id=tenant_id
            )
            
            # Store in cache
            self.cache[input_hash] = cached_decision
            self.embeddings_cache[input_hash] = embedding
            
            # Cleanup if cache is too large
            await self._cleanup_cache()
            
            # Persist to database (async)
            asyncio.create_task(self._persist_to_db(cached_decision))
            
            self.logger.info(f"💾 Cached routing decision: {workflow_category} for '{user_input[:50]}...'")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing routing decision: {e}")
    
    async def _find_semantic_match(
        self, 
        query_embedding: np.ndarray, 
        tenant_id: str = None
    ) -> Optional[Tuple[CachedRoutingDecision, float]]:
        """Find best semantic match in cache"""
        
        best_match = None
        best_similarity = 0.0
        
        for input_hash, cached_decision in self.cache.items():
            # Skip expired entries
            if self._is_cache_expired(cached_decision):
                continue
            
            # Skip different tenants (unless None)
            if tenant_id and cached_decision.tenant_id and cached_decision.tenant_id != tenant_id:
                continue
            
            # Calculate similarity
            if input_hash in self.embeddings_cache:
                cached_embedding = self.embeddings_cache[input_hash]
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    cached_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = (cached_decision, similarity)
        
        return best_match
    
    def _generate_input_hash(self, user_input: str, tenant_id: str = None) -> str:
        """Generate hash for user input + tenant"""
        combined = f"{user_input.lower().strip()}|{tenant_id or 'default'}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_cache_expired(self, cached_decision: CachedRoutingDecision) -> bool:
        """Check if cache entry is expired"""
        age = datetime.now() - cached_decision.created_at
        return age > timedelta(hours=self.cache_ttl_hours)
    
    async def _update_cache_usage(self, cached_decision: CachedRoutingDecision):
        """Update cache usage statistics"""
        cached_decision.last_used = datetime.now()
        cached_decision.usage_count += 1
    
    async def _cleanup_cache(self):
        """Remove expired and least-used entries"""
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Remove expired entries first
        expired_keys = [
            key for key, decision in self.cache.items()
            if self._is_cache_expired(decision)
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.embeddings_cache:
                del self.embeddings_cache[key]
        
        # If still too large, remove least-used entries
        if len(self.cache) > self.max_cache_size:
            # Sort by usage_count and last_used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1].usage_count, x[1].last_used)
            )
            
            # Remove bottom 20%
            to_remove = len(sorted_entries) - self.max_cache_size
            for key, _ in sorted_entries[:to_remove]:
                del self.cache[key]
                if key in self.embeddings_cache:
                    del self.embeddings_cache[key]
        
        self.logger.info(f"🧹 Cache cleanup completed. Size: {len(self.cache)}")
    
    async def _load_cache_from_db(self):
        """Load cache from database (if available)"""
        # TODO: Implement database persistence
        pass
    
    async def _persist_to_db(self, cached_decision: CachedRoutingDecision):
        """Persist cache entry to database"""
        # TODO: Implement database persistence
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        cache_hit_rate = (
            (self.stats.cache_hits / self.stats.total_requests * 100)
            if self.stats.total_requests > 0 else 0
        )
        
        # Check for cache collapse (hit rate crash) - Task 6.4.25
        self._check_cache_collapse_sync(cache_hit_rate)
        
        # Estimate cost savings (assuming $0.0003 per LLM call)
        llm_calls_avoided = self.stats.cache_hits
        estimated_savings = llm_calls_avoided * 0.0003
        
        return {
            'total_requests': self.stats.total_requests,
            'cache_hits': self.stats.cache_hits,
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'avg_response_time_ms': round(self.stats.avg_response_time_ms, 2),
            'cached_decisions': len(self.cache),
            'estimated_cost_savings_usd': round(estimated_savings, 4),
            'llm_calls_avoided': llm_calls_avoided
        }
    
    async def clear_cache(self, tenant_id: str = None):
        """Clear cache (optionally for specific tenant)"""
        if tenant_id:
            # Clear only for specific tenant
            keys_to_remove = [
                key for key, decision in self.cache.items()
                if decision.tenant_id == tenant_id
            ]
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.embeddings_cache:
                    del self.embeddings_cache[key]
            self.logger.info(f"🧹 Cleared cache for tenant: {tenant_id}")
        else:
            # Clear all cache
            self.cache.clear()
            self.embeddings_cache.clear()
            self.logger.info("🧹 Cleared entire routing cache")
    
    def _check_cache_collapse_sync(self, current_hit_rate: float):
        """Check for cache collapse and trigger fallback - Task 6.4.25 (sync version)"""
        try:
            # Define cache collapse threshold (hit rate drops below 30%)
            CACHE_COLLAPSE_THRESHOLD = 30.0
            MIN_REQUESTS_FOR_CHECK = 10  # Only check if we have enough data
            
            if (self.stats.total_requests >= MIN_REQUESTS_FOR_CHECK and 
                current_hit_rate < CACHE_COLLAPSE_THRESHOLD):
                
                logger.warning(f"Cache collapse detected: hit rate {current_hit_rate}% < {CACHE_COLLAPSE_THRESHOLD}%")
                
                # Trigger cache collapse fallback
                self._trigger_cache_collapse_fallback_sync(current_hit_rate)
                
        except Exception as e:
            logger.error(f"Failed to check cache collapse: {e}")
    
    def _trigger_cache_collapse_fallback_sync(self, hit_rate: float):
        """Trigger fallback when cache collapse occurs (sync version)"""
        try:
            fallback_data = {
                "request_id": f"cache_collapse_{int(time.time())}",
                "tenant_id": "system",
                "workflow_id": "routing_cache",
                "current_system": "rbia",
                "error_type": "cache_collapse",
                "error_message": f"Cache hit rate crashed to {hit_rate}%",
                "cache_hit_rate": hit_rate,
                "total_requests": self.stats.total_requests
            }
            
            logger.warning(f"Triggering cache collapse fallback: {json.dumps(fallback_data)}")
            
        except Exception as fallback_error:
            logger.error(f"Failed to trigger cache collapse fallback: {fallback_error}")
    
    async def _check_cache_collapse(self, current_hit_rate: float):
        """Check for cache collapse and trigger fallback - Task 6.4.25"""
        try:
            # Define cache collapse threshold (hit rate drops below 30%)
            CACHE_COLLAPSE_THRESHOLD = 30.0
            MIN_REQUESTS_FOR_CHECK = 10  # Only check if we have enough data
            
            if (self.stats.total_requests >= MIN_REQUESTS_FOR_CHECK and 
                current_hit_rate < CACHE_COLLAPSE_THRESHOLD):
                
                logger.warning(f"Cache collapse detected: hit rate {current_hit_rate}% < {CACHE_COLLAPSE_THRESHOLD}%")
                
                # Trigger cache collapse fallback
                await self._trigger_cache_collapse_fallback(current_hit_rate)
                
        except Exception as e:
            logger.error(f"Failed to check cache collapse: {e}")
    
    async def _trigger_cache_collapse_fallback(self, hit_rate: float):
        """Trigger fallback when cache collapse occurs"""
        try:
            fallback_data = {
                "request_id": f"cache_collapse_{int(time.time())}",
                "tenant_id": "system",
                "workflow_id": "routing_cache",
                "current_system": "rbia",
                "error_type": "cache_collapse",
                "error_message": f"Cache hit rate crashed to {hit_rate}%",
                "cache_hit_rate": hit_rate,
                "total_requests": self.stats.total_requests
            }
            
            logger.warning(f"Triggering cache collapse fallback: {json.dumps(fallback_data)}")
            
        except Exception as fallback_error:
            logger.error(f"Failed to trigger cache collapse fallback: {fallback_error}")
    
    async def warm_cache(self, tenant_id: str = "1300"):
        """Warm cache with common workflow patterns for faster responses"""
        
        # Common query patterns mapped to workflows
        cache_warming_data = [
            # Data Quality Workflows
            ("Check duplicate deals across Salesforce/CRM and flag for review", "duplicate_detection", {"actions": ["check", "flag"], "entities": ["duplicate deals"]}, 0.95),
            ("Detect ownerless or unassigned deals", "ownerless_deals_detection", {"entities": ["ownerless deals"], "actions": ["detect"]}, 0.93),
            ("Highlight deals missing activities/logs (no calls/emails logged in last 14 days)", "activity_tracking_audit", {"time_period": "14 days", "entities": ["deals"], "actions": ["highlight"]}, 0.91),
            ("Audit pipeline data quality – identify deals missing close dates, amounts, or owners", "missing_fields_audit", {"entities": ["deals"], "actions": ["audit", "identify"]}, 0.89),
            ("Run pipeline hygiene check – deals stuck >60 days in stage", "pipeline_hygiene_stale_deals", {"time_period": "60 days", "entities": ["deals"], "actions": ["check"]}, 0.94),
            
            # Risk Analysis Workflows
            ("Apply risk scoring to deals and categorize into high/medium/low", "risk_scoring_analysis", {"threshold": "high/medium/low", "entities": ["deals"], "actions": ["categorize"]}, 0.92),
            
            # Velocity Analysis Workflows  
            ("Review pipeline velocity by stage (days per stage)", "stage_velocity_analysis", {"entities": ["pipeline", "stage"], "actions": ["review"]}, 0.88),
            ("Analyze stage conversion rates by team/rep", "conversion_rate_analysis", {"entities": ["stage", "team", "rep"], "actions": ["analyze"]}, 0.87),
            
            # Performance Analysis Workflows
            ("Enforce pipeline coverage ratio (e.g., 3x quota rule)", "pipeline_coverage_analysis", {"threshold": "3x quota", "entities": ["pipeline"], "actions": ["enforce"]}, 0.90),
            ("Compare rep forecast vs system forecast vs AI prediction", "forecast_comparison", {"entities": ["forecast"], "actions": ["compare"]}, 0.86),
            
            # Coaching Insights Workflows
            ("Coach reps on stalled deals (pipeline stuck >30 days)", "stalled_deal_coaching", {"time_period": "30 days", "entities": ["reps", "deals"], "actions": ["coach"]}, 0.85),
            ("Get my next best actions for deals in negotiation stage", "next_best_actions", {"filters": {"stage": "negotiation"}, "entities": ["deals"], "actions": ["recommend"]}, 0.84),
            
            # Common variations
            ("deals stuck in stage for more than 60 days", "pipeline_hygiene_stale_deals", {"time_period": "60 days", "entities": ["deals"]}, 0.93),
            ("Check my deals at risk (slipping, no activity, customer disengagement)", "activity_tracking_audit", {"entities": ["my deals"], "actions": ["check"]}, 0.90),
            ("duplicate deals", "duplicate_detection", {"entities": ["duplicate deals"]}, 0.88),
            ("missing activities", "activity_tracking_audit", {"entities": ["activities"]}, 0.87),
        ]
        
        self.logger.info(f"🔥 Warming cache with {len(cache_warming_data)} common patterns...")
        
        warmed_count = 0
        for user_input, workflow_category, parameters, confidence in cache_warming_data:
            try:
                await self.store_routing_decision(
                    user_input=user_input,
                    workflow_category=workflow_category,
                    parameters=parameters,
                    confidence=confidence,
                    tenant_id=tenant_id
                )
                warmed_count += 1
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to warm cache for '{user_input[:30]}...': {e}")
        
        self.logger.info(f"🔥 Cache warming complete: {warmed_count}/{len(cache_warming_data)} patterns cached")
        return warmed_count

    async def _evict_cache_entry(self, user_input: str, tenant_id: str):
        """Evict a problematic cache entry"""
        try:
            input_hash = self._generate_input_hash(user_input, tenant_id)
            if input_hash in self.cache:
                del self.cache[input_hash]
                self.logger.info(f"🗑️ Evicted cache entry for: {user_input[:30]}...")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to evict cache entry: {e}")
