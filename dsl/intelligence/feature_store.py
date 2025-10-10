"""
Centralized Feature Store - Task 6.1.14
========================================
Feature Store with offline (Fabric/Postgres) and online (Redis) storage

Features:
- Offline feature storage for training/batch
- Online feature serving for real-time inference
- Feature versioning and lineage
- Multi-tenant isolation
- Feature consistency across environments
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Feature data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"


class FeatureStore(str, Enum):
    """Feature storage backends"""
    OFFLINE = "offline"  # Fabric Silver/Gold, Postgres
    ONLINE = "online"    # Redis cache
    BOTH = "both"        # Synced to both


@dataclass
class FeatureMetadata:
    """Metadata for a feature"""
    feature_id: str
    feature_name: str
    feature_type: FeatureType
    description: str
    
    # Versioning
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Ownership
    owner: str = "system"
    tenant_id: str = "1300"
    
    # Lineage
    source_dataset: Optional[str] = None
    transformation_logic: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    # Schema
    data_type: str = "float"
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Quality
    null_allowed: bool = False
    importance_score: float = 0.5
    
    # Tags
    tags: List[str] = field(default_factory=list)
    
    # Storage
    storage_backend: FeatureStore = FeatureStore.BOTH
    ttl_seconds: int = 3600  # 1 hour default for online cache


@dataclass
class FeatureValue:
    """A feature value with metadata"""
    feature_id: str
    entity_id: str  # e.g., customer_id, opportunity_id
    value: Any
    
    # Timestamps
    event_timestamp: datetime
    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    tenant_id: str = "1300"
    version: str = "1.0.0"
    source: str = "unknown"


@dataclass
class FeatureVector:
    """A collection of features for an entity"""
    entity_id: str
    entity_type: str  # customer, opportunity, account, etc.
    features: Dict[str, Any]
    
    # Metadata
    tenant_id: str = "1300"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    
    # Quality indicators
    completeness: float = 1.0  # % of features present
    freshness_seconds: int = 0  # Age of oldest feature


class CentralizedFeatureStore:
    """
    Centralized Feature Store for RBIA
    
    Provides:
    - Offline storage for training/batch (Fabric, Postgres)
    - Online storage for real-time serving (Redis)
    - Feature versioning and lineage
    - Multi-tenant isolation
    - Feature quality monitoring
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Feature registry (in-memory, would be in DB)
        self.feature_registry: Dict[str, FeatureMetadata] = {}
        
        # Offline storage (simulated)
        self.offline_storage: Dict[str, List[FeatureValue]] = {}
        
        # Online cache (simulated Redis)
        self.online_cache: Dict[str, Tuple[Any, datetime]] = {}  # key -> (value, expiry)
        
        self.logger.info("🏪 Feature Store initialized")
    
    async def register_feature(self, metadata: FeatureMetadata) -> bool:
        """
        Register a new feature in the feature store
        
        Args:
            metadata: Feature metadata
            
        Returns:
            bool: Success status
        """
        try:
            feature_key = f"{metadata.tenant_id}:{metadata.feature_id}:{metadata.version}"
            
            # Validate feature doesn't exist
            if feature_key in self.feature_registry:
                self.logger.warning(f"Feature already registered: {feature_key}")
                return False
            
            # Store metadata
            self.feature_registry[feature_key] = metadata
            
            self.logger.info(f"✅ Registered feature: {metadata.feature_name} ({feature_key})")
            
            # If using DB, persist to feature_metadata table
            if self.pool_manager and self.pool_manager.postgres_pool:
                await self._persist_feature_metadata(metadata)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register feature: {e}")
            return False
    
    async def write_feature(
        self,
        feature_id: str,
        entity_id: str,
        value: Any,
        tenant_id: str = "1300",
        version: str = "1.0.0",
        store: FeatureStore = FeatureStore.BOTH
    ) -> bool:
        """
        Write a feature value to the feature store
        
        Args:
            feature_id: Feature identifier
            entity_id: Entity identifier (customer_id, etc.)
            value: Feature value
            tenant_id: Tenant ID
            version: Feature version
            store: Storage backend(s) to write to
            
        Returns:
            bool: Success status
        """
        try:
            feature_key = f"{tenant_id}:{feature_id}:{version}"
            
            # Validate feature is registered
            if feature_key not in self.feature_registry:
                self.logger.warning(f"Feature not registered: {feature_key}")
                return False
            
            metadata = self.feature_registry[feature_key]
            
            # Create feature value
            feature_value = FeatureValue(
                feature_id=feature_id,
                entity_id=entity_id,
                value=value,
                event_timestamp=datetime.utcnow(),
                tenant_id=tenant_id,
                version=version,
                source="feature_store_write"
            )
            
            # Write to offline storage
            if store in [FeatureStore.OFFLINE, FeatureStore.BOTH]:
                await self._write_offline(feature_value)
            
            # Write to online cache
            if store in [FeatureStore.ONLINE, FeatureStore.BOTH]:
                await self._write_online(feature_value, metadata.ttl_seconds)
            
            self.logger.debug(
                f"✅ Wrote feature: {feature_id} for entity {entity_id} "
                f"to {store.value} storage"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write feature: {e}")
            return False
    
    async def read_feature(
        self,
        feature_id: str,
        entity_id: str,
        tenant_id: str = "1300",
        version: str = "1.0.0",
        prefer_online: bool = True
    ) -> Optional[Any]:
        """
        Read a feature value from the feature store
        
        Args:
            feature_id: Feature identifier
            entity_id: Entity identifier
            tenant_id: Tenant ID
            version: Feature version
            prefer_online: Try online cache first
            
        Returns:
            Feature value or None if not found
        """
        try:
            # Try online cache first if preferred
            if prefer_online:
                online_value = await self._read_online(feature_id, entity_id, tenant_id, version)
                if online_value is not None:
                    return online_value
            
            # Fall back to offline storage
            offline_value = await self._read_offline(feature_id, entity_id, tenant_id, version)
            
            # If found offline, populate online cache
            if offline_value is not None and prefer_online:
                feature_key = f"{tenant_id}:{feature_id}:{version}"
                if feature_key in self.feature_registry:
                    metadata = self.feature_registry[feature_key]
                    feature_value = FeatureValue(
                        feature_id=feature_id,
                        entity_id=entity_id,
                        value=offline_value,
                        event_timestamp=datetime.utcnow(),
                        tenant_id=tenant_id,
                        version=version
                    )
                    await self._write_online(feature_value, metadata.ttl_seconds)
            
            return offline_value
            
        except Exception as e:
            self.logger.error(f"Failed to read feature: {e}")
            return None
    
    async def read_feature_vector(
        self,
        entity_id: str,
        feature_ids: List[str],
        entity_type: str = "unknown",
        tenant_id: str = "1300",
        version: str = "1.0.0"
    ) -> FeatureVector:
        """
        Read multiple features for an entity (feature vector)
        
        Args:
            entity_id: Entity identifier
            feature_ids: List of feature IDs to retrieve
            entity_type: Type of entity
            tenant_id: Tenant ID
            version: Feature version
            
        Returns:
            FeatureVector with all requested features
        """
        features = {}
        oldest_timestamp = datetime.utcnow()
        
        for feature_id in feature_ids:
            value = await self.read_feature(feature_id, entity_id, tenant_id, version)
            if value is not None:
                features[feature_id] = value
        
        # Calculate completeness
        completeness = len(features) / len(feature_ids) if feature_ids else 0.0
        
        # Calculate freshness (simplified - would track actual timestamps)
        freshness_seconds = 0  # Would calculate from actual feature timestamps
        
        return FeatureVector(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            tenant_id=tenant_id,
            version=version,
            completeness=completeness,
            freshness_seconds=freshness_seconds
        )
    
    async def batch_write_features(
        self,
        features: List[Tuple[str, str, Any]],  # (feature_id, entity_id, value)
        tenant_id: str = "1300",
        version: str = "1.0.0",
        store: FeatureStore = FeatureStore.OFFLINE
    ) -> Dict[str, Any]:
        """
        Batch write features (for offline/training scenarios)
        
        Args:
            features: List of (feature_id, entity_id, value) tuples
            tenant_id: Tenant ID
            version: Feature version
            store: Storage backend
            
        Returns:
            Dict with write statistics
        """
        stats = {
            'total': len(features),
            'successful': 0,
            'failed': 0
        }
        
        for feature_id, entity_id, value in features:
            success = await self.write_feature(
                feature_id, entity_id, value,
                tenant_id=tenant_id,
                version=version,
                store=store
            )
            if success:
                stats['successful'] += 1
            else:
                stats['failed'] += 1
        
        self.logger.info(
            f"📦 Batch write completed: {stats['successful']}/{stats['total']} successful"
        )
        
        return stats
    
    async def get_feature_lineage(
        self,
        feature_id: str,
        tenant_id: str = "1300",
        version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Get lineage information for a feature
        
        Args:
            feature_id: Feature identifier
            tenant_id: Tenant ID
            version: Feature version
            
        Returns:
            Dict with lineage information
        """
        feature_key = f"{tenant_id}:{feature_id}:{version}"
        
        if feature_key not in self.feature_registry:
            return {'error': 'Feature not found'}
        
        metadata = self.feature_registry[feature_key]
        
        return {
            'feature_id': feature_id,
            'feature_name': metadata.feature_name,
            'version': metadata.version,
            'source_dataset': metadata.source_dataset,
            'transformation_logic': metadata.transformation_logic,
            'dependencies': metadata.dependencies,
            'created_at': metadata.created_at.isoformat(),
            'owner': metadata.owner
        }
    
    async def _write_offline(self, feature_value: FeatureValue):
        """Write feature to offline storage (Fabric/Postgres)"""
        key = f"{feature_value.tenant_id}:{feature_value.feature_id}:{feature_value.entity_id}"
        
        if key not in self.offline_storage:
            self.offline_storage[key] = []
        
        self.offline_storage[key].append(feature_value)
        
        # In production, write to Postgres or Fabric
        if self.pool_manager and self.pool_manager.postgres_pool:
            await self._persist_feature_value_offline(feature_value)
    
    async def _write_online(self, feature_value: FeatureValue, ttl_seconds: int):
        """Write feature to online cache (Redis)"""
        cache_key = f"{feature_value.tenant_id}:{feature_value.feature_id}:{feature_value.entity_id}:online"
        expiry = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        self.online_cache[cache_key] = (feature_value.value, expiry)
        
        # In production, write to Redis
        # await redis.setex(cache_key, ttl_seconds, json.dumps(feature_value.value))
    
    async def _read_online(
        self,
        feature_id: str,
        entity_id: str,
        tenant_id: str,
        version: str
    ) -> Optional[Any]:
        """Read feature from online cache"""
        cache_key = f"{tenant_id}:{feature_id}:{entity_id}:online"
        
        if cache_key in self.online_cache:
            value, expiry = self.online_cache[cache_key]
            
            # Check if expired
            if datetime.utcnow() < expiry:
                return value
            else:
                # Remove expired entry
                del self.online_cache[cache_key]
        
        return None
    
    async def _read_offline(
        self,
        feature_id: str,
        entity_id: str,
        tenant_id: str,
        version: str
    ) -> Optional[Any]:
        """Read feature from offline storage"""
        key = f"{tenant_id}:{feature_id}:{entity_id}"
        
        if key in self.offline_storage and self.offline_storage[key]:
            # Return most recent value
            latest = self.offline_storage[key][-1]
            return latest.value
        
        return None
    
    async def _persist_feature_metadata(self, metadata: FeatureMetadata):
        """Persist feature metadata to database"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO feature_metadata (
                        feature_id, feature_name, feature_type, description,
                        version, owner, tenant_id, source_dataset,
                        transformation_logic, data_type, storage_backend,
                        ttl_seconds, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (tenant_id, feature_id, version) DO UPDATE
                    SET updated_at = NOW()
                """,
                    metadata.feature_id, metadata.feature_name,
                    metadata.feature_type.value, metadata.description,
                    metadata.version, metadata.owner, metadata.tenant_id,
                    metadata.source_dataset, metadata.transformation_logic,
                    metadata.data_type, metadata.storage_backend.value,
                    metadata.ttl_seconds, metadata.created_at
                )
        except Exception as e:
            self.logger.error(f"Failed to persist feature metadata: {e}")
    
    async def _persist_feature_value_offline(self, feature_value: FeatureValue):
        """Persist feature value to offline storage"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO feature_values_offline (
                        feature_id, entity_id, value, event_timestamp,
                        ingestion_timestamp, tenant_id, version, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    feature_value.feature_id, feature_value.entity_id,
                    json.dumps(feature_value.value), feature_value.event_timestamp,
                    feature_value.ingestion_timestamp, feature_value.tenant_id,
                    feature_value.version, feature_value.source
                )
        except Exception as e:
            self.logger.error(f"Failed to persist feature value: {e}")


# Singleton instance
_feature_store_instance: Optional[CentralizedFeatureStore] = None


def get_feature_store(pool_manager=None) -> CentralizedFeatureStore:
    """Get or create feature store singleton"""
    global _feature_store_instance
    
    if _feature_store_instance is None:
        _feature_store_instance = CentralizedFeatureStore(pool_manager)
    
    return _feature_store_instance

