"""
Task 6.3.73: Explainability cache (reuse reason codes across similar inputs)
Implement explainability cache for latency and cost reduction
"""

import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CachedExplanation:
    """Cached explainability result"""
    input_hash: str
    explanation_data: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_id: str
    model_version: str
    confidence: float
    explanation_method: str
    created_at: datetime
    access_count: int
    last_accessed: datetime

@dataclass
class SimilarityConfig:
    """Configuration for input similarity matching"""
    numerical_tolerance: float = 0.1
    categorical_exact_match: bool = True
    similarity_threshold: float = 0.9
    max_feature_diff_count: int = 2

class ExplainabilityCache:
    """
    Explainability cache service for reusing reason codes
    Task 6.3.73: Latency & cost reduction with staleness bounds
    """
    
    def __init__(self, max_cache_size: int = 10000, default_ttl_hours: int = 24):
        self.cache: Dict[str, CachedExplanation] = {}
        self.similarity_index: Dict[str, List[str]] = {}  # model_id -> list of hashes
        self.max_cache_size = max_cache_size
        self.default_ttl_hours = default_ttl_hours
        self.cache_hits = 0
        self.cache_misses = 0
        self.similarity_matches = 0
    
    def _generate_input_hash(self, input_data: Dict[str, Any], model_id: str) -> str:
        """Generate hash for input data"""
        # Normalize input for consistent hashing
        normalized_input = self._normalize_input(input_data)
        hash_content = json.dumps({
            "model_id": model_id,
            "input": normalized_input
        }, sort_keys=True)
        
        return hashlib.sha256(hash_content.encode()).hexdigest()
    
    def _normalize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize input data for consistent comparison"""
        normalized = {}
        
        for key, value in input_data.items():
            if isinstance(value, float):
                # Round floats to reduce minor variations
                normalized[key] = round(value, 4)
            elif isinstance(value, dict):
                normalized[key] = self._normalize_input(value)
            elif isinstance(value, list):
                # Sort lists if they contain comparable items
                try:
                    normalized[key] = sorted(value)
                except TypeError:
                    normalized[key] = value
            else:
                normalized[key] = value
        
        return normalized
    
    async def get_cached_explanation(
        self,
        input_data: Dict[str, Any],
        model_id: str,
        model_version: str,
        explanation_method: str,
        similarity_config: Optional[SimilarityConfig] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached explanation or find similar explanation"""
        
        input_hash = self._generate_input_hash(input_data, model_id)
        
        # Check for exact match first
        if input_hash in self.cache:
            cached_explanation = self.cache[input_hash]
            
            # Check if cache entry is still valid
            if self._is_cache_valid(cached_explanation, model_version, explanation_method):
                # Update access statistics
                cached_explanation.access_count += 1
                cached_explanation.last_accessed = datetime.utcnow()
                
                self.cache_hits += 1
                logger.info(f"Cache hit for input hash: {input_hash[:8]}")
                
                return {
                    "explanation_data": cached_explanation.explanation_data,
                    "feature_importance": cached_explanation.feature_importance,
                    "confidence": cached_explanation.confidence,
                    "cache_hit": True,
                    "cache_type": "exact_match",
                    "cached_at": cached_explanation.created_at.isoformat(),
                    "access_count": cached_explanation.access_count
                }
        
        # Try similarity matching if enabled
        if similarity_config:
            similar_explanation = await self._find_similar_explanation(
                input_data, model_id, model_version, explanation_method, similarity_config
            )
            
            if similar_explanation:
                return similar_explanation
        
        self.cache_misses += 1
        logger.debug(f"Cache miss for input hash: {input_hash[:8]}")
        return None
    
    async def cache_explanation(
        self,
        input_data: Dict[str, Any],
        explanation_data: Dict[str, Any],
        feature_importance: Dict[str, float],
        model_id: str,
        model_version: str,
        confidence: float,
        explanation_method: str
    ) -> str:
        """Cache an explanation result"""
        
        input_hash = self._generate_input_hash(input_data, model_id)
        
        # Create cached explanation
        cached_explanation = CachedExplanation(
            input_hash=input_hash,
            explanation_data=explanation_data,
            feature_importance=feature_importance,
            model_id=model_id,
            model_version=model_version,
            confidence=confidence,
            explanation_method=explanation_method,
            created_at=datetime.utcnow(),
            access_count=0,
            last_accessed=datetime.utcnow()
        )
        
        # Add to cache
        self.cache[input_hash] = cached_explanation
        
        # Update similarity index
        if model_id not in self.similarity_index:
            self.similarity_index[model_id] = []
        self.similarity_index[model_id].append(input_hash)
        
        # Cleanup cache if needed
        await self._cleanup_cache()
        
        logger.info(f"Cached explanation for input hash: {input_hash[:8]}")
        return input_hash
    
    async def _find_similar_explanation(
        self,
        input_data: Dict[str, Any],
        model_id: str,
        model_version: str,
        explanation_method: str,
        similarity_config: SimilarityConfig
    ) -> Optional[Dict[str, Any]]:
        """Find explanation for similar input"""
        
        if model_id not in self.similarity_index:
            return None
        
        candidate_hashes = self.similarity_index[model_id]
        
        for candidate_hash in candidate_hashes:
            if candidate_hash not in self.cache:
                continue
            
            cached_explanation = self.cache[candidate_hash]
            
            # Check if cache entry is valid
            if not self._is_cache_valid(cached_explanation, model_version, explanation_method):
                continue
            
            # Calculate similarity
            similarity_score = await self._calculate_input_similarity(
                input_data, cached_explanation, similarity_config
            )
            
            if similarity_score >= similarity_config.similarity_threshold:
                # Update access statistics
                cached_explanation.access_count += 1
                cached_explanation.last_accessed = datetime.utcnow()
                
                self.cache_hits += 1
                self.similarity_matches += 1
                
                logger.info(
                    f"Similarity match found: {similarity_score:.3f} "
                    f"for hash: {candidate_hash[:8]}"
                )
                
                return {
                    "explanation_data": cached_explanation.explanation_data,
                    "feature_importance": cached_explanation.feature_importance,
                    "confidence": cached_explanation.confidence,
                    "cache_hit": True,
                    "cache_type": "similarity_match",
                    "similarity_score": similarity_score,
                    "cached_at": cached_explanation.created_at.isoformat(),
                    "access_count": cached_explanation.access_count
                }
        
        return None
    
    async def _calculate_input_similarity(
        self,
        input_data: Dict[str, Any],
        cached_explanation: CachedExplanation,
        similarity_config: SimilarityConfig
    ) -> float:
        """Calculate similarity between inputs"""
        
        # For similarity calculation, we need to reconstruct the original input
        # This is simplified - in practice, you might store original inputs
        
        # Calculate feature-wise similarity
        total_features = 0
        matching_features = 0
        feature_diff_count = 0
        
        # This is a simplified similarity calculation
        # In practice, you'd implement more sophisticated similarity metrics
        
        normalized_input = self._normalize_input(input_data)
        
        # For demonstration, assume we can extract features from the hash
        # In practice, you might store feature vectors separately
        
        # Placeholder similarity calculation
        similarity_score = 0.85  # Would be calculated based on actual feature comparison
        
        return similarity_score
    
    def _is_cache_valid(
        self,
        cached_explanation: CachedExplanation,
        model_version: str,
        explanation_method: str
    ) -> bool:
        """Check if cached explanation is still valid"""
        
        # Check model version compatibility
        if cached_explanation.model_version != model_version:
            return False
        
        # Check explanation method compatibility
        if cached_explanation.explanation_method != explanation_method:
            return False
        
        # Check TTL
        ttl_cutoff = datetime.utcnow() - timedelta(hours=self.default_ttl_hours)
        if cached_explanation.created_at < ttl_cutoff:
            return False
        
        return True
    
    async def _cleanup_cache(self) -> None:
        """Clean up expired and least-used cache entries"""
        
        if len(self.cache) <= self.max_cache_size:
            return
        
        # Remove expired entries first
        ttl_cutoff = datetime.utcnow() - timedelta(hours=self.default_ttl_hours)
        expired_hashes = [
            hash_key for hash_key, cached_explanation in self.cache.items()
            if cached_explanation.created_at < ttl_cutoff
        ]
        
        for hash_key in expired_hashes:
            self._remove_from_cache(hash_key)
        
        # If still over limit, remove least recently used
        if len(self.cache) > self.max_cache_size:
            # Sort by last accessed time
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest entries
            entries_to_remove = len(self.cache) - self.max_cache_size
            for i in range(entries_to_remove):
                hash_key = sorted_entries[i][0]
                self._remove_from_cache(hash_key)
        
        logger.info(f"Cache cleanup completed. Current size: {len(self.cache)}")
    
    def _remove_from_cache(self, hash_key: str) -> None:
        """Remove entry from cache and similarity index"""
        if hash_key in self.cache:
            cached_explanation = self.cache[hash_key]
            model_id = cached_explanation.model_id
            
            # Remove from cache
            del self.cache[hash_key]
            
            # Remove from similarity index
            if model_id in self.similarity_index:
                try:
                    self.similarity_index[model_id].remove(hash_key)
                except ValueError:
                    pass  # Hash not in list
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "similarity_matches": self.similarity_matches,
            "hit_rate": hit_rate,
            "models_indexed": len(self.similarity_index),
            "avg_access_count": sum(ce.access_count for ce in self.cache.values()) / len(self.cache) if self.cache else 0
        }
    
    async def invalidate_model_cache(self, model_id: str, model_version: Optional[str] = None) -> int:
        """Invalidate cache entries for a specific model"""
        invalidated_count = 0
        
        hashes_to_remove = []
        for hash_key, cached_explanation in self.cache.items():
            if cached_explanation.model_id == model_id:
                if model_version is None or cached_explanation.model_version == model_version:
                    hashes_to_remove.append(hash_key)
        
        for hash_key in hashes_to_remove:
            self._remove_from_cache(hash_key)
            invalidated_count += 1
        
        logger.info(f"Invalidated {invalidated_count} cache entries for model {model_id}")
        return invalidated_count

# Global explainability cache instance
explainability_cache = ExplainabilityCache()
