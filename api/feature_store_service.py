"""
Feature Store Service API - Task 6.1.14
========================================
REST API for centralized feature management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl.intelligence.feature_store import (
    CentralizedFeatureStore,
    FeatureMetadata,
    FeatureType,
    FeatureStore as FeatureStoreEnum,
    get_feature_store
)

app = FastAPI(
    title="RBIA Feature Store Service",
    description="Centralized feature management with offline/online storage",
    version="1.0.0"
)


class RegisterFeatureRequest(BaseModel):
    """Request to register a new feature"""
    feature_id: str
    feature_name: str
    feature_type: str = Field(..., description="numeric, categorical, boolean, text, embedding, timestamp")
    description: str
    version: str = "1.0.0"
    owner: str = "system"
    tenant_id: str = "1300"
    source_dataset: Optional[str] = None
    transformation_logic: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    data_type: str = "float"
    storage_backend: str = Field(default="both", description="offline, online, both")
    ttl_seconds: int = Field(default=3600, description="TTL for online cache")
    tags: List[str] = Field(default_factory=list)


class WriteFeatureRequest(BaseModel):
    """Request to write a feature value"""
    feature_id: str
    entity_id: str
    value: Any
    tenant_id: str = "1300"
    version: str = "1.0.0"
    store: str = Field(default="both", description="offline, online, both")


class ReadFeatureRequest(BaseModel):
    """Request to read a feature value"""
    feature_id: str
    entity_id: str
    tenant_id: str = "1300"
    version: str = "1.0.0"
    prefer_online: bool = True


class ReadFeatureVectorRequest(BaseModel):
    """Request to read feature vector"""
    entity_id: str
    feature_ids: List[str]
    entity_type: str = "unknown"
    tenant_id: str = "1300"
    version: str = "1.0.0"


class BatchWriteRequest(BaseModel):
    """Request for batch feature write"""
    features: List[Dict[str, Any]]  # List of {feature_id, entity_id, value}
    tenant_id: str = "1300"
    version: str = "1.0.0"
    store: str = "offline"


# Initialize feature store
feature_store = get_feature_store()


@app.post("/features/register")
async def register_feature(request: RegisterFeatureRequest):
    """Register a new feature in the feature store"""
    try:
        # Validate feature type
        try:
            feature_type_enum = FeatureType(request.feature_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature_type: {request.feature_type}"
            )
        
        # Validate storage backend
        try:
            storage_enum = FeatureStoreEnum(request.storage_backend)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid storage_backend: {request.storage_backend}"
            )
        
        # Create metadata
        metadata = FeatureMetadata(
            feature_id=request.feature_id,
            feature_name=request.feature_name,
            feature_type=feature_type_enum,
            description=request.description,
            version=request.version,
            owner=request.owner,
            tenant_id=request.tenant_id,
            source_dataset=request.source_dataset,
            transformation_logic=request.transformation_logic,
            dependencies=request.dependencies,
            data_type=request.data_type,
            storage_backend=storage_enum,
            ttl_seconds=request.ttl_seconds,
            tags=request.tags
        )
        
        # Register feature
        success = await feature_store.register_feature(metadata)
        
        if success:
            return {
                'status': 'success',
                'feature_id': request.feature_id,
                'version': request.version,
                'message': 'Feature registered successfully'
            }
        else:
            raise HTTPException(
                status_code=409,
                detail='Feature already exists'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/write")
async def write_feature(request: WriteFeatureRequest):
    """Write a feature value to the feature store"""
    try:
        # Validate storage backend
        try:
            store_enum = FeatureStoreEnum(request.store)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid store: {request.store}"
            )
        
        # Write feature
        success = await feature_store.write_feature(
            feature_id=request.feature_id,
            entity_id=request.entity_id,
            value=request.value,
            tenant_id=request.tenant_id,
            version=request.version,
            store=store_enum
        )
        
        if success:
            return {
                'status': 'success',
                'feature_id': request.feature_id,
                'entity_id': request.entity_id,
                'store': request.store
            }
        else:
            raise HTTPException(
                status_code=400,
                detail='Feature not registered or write failed'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/read")
async def read_feature(request: ReadFeatureRequest):
    """Read a feature value from the feature store"""
    try:
        value = await feature_store.read_feature(
            feature_id=request.feature_id,
            entity_id=request.entity_id,
            tenant_id=request.tenant_id,
            version=request.version,
            prefer_online=request.prefer_online
        )
        
        if value is not None:
            return {
                'status': 'success',
                'feature_id': request.feature_id,
                'entity_id': request.entity_id,
                'value': value,
                'source': 'online' if request.prefer_online else 'offline'
            }
        else:
            raise HTTPException(
                status_code=404,
                detail='Feature value not found'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/read-vector")
async def read_feature_vector(request: ReadFeatureVectorRequest):
    """Read multiple features for an entity (feature vector)"""
    try:
        vector = await feature_store.read_feature_vector(
            entity_id=request.entity_id,
            feature_ids=request.feature_ids,
            entity_type=request.entity_type,
            tenant_id=request.tenant_id,
            version=request.version
        )
        
        return {
            'status': 'success',
            'entity_id': vector.entity_id,
            'entity_type': vector.entity_type,
            'features': vector.features,
            'completeness': vector.completeness,
            'freshness_seconds': vector.freshness_seconds,
            'timestamp': vector.timestamp.isoformat()
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/batch-write")
async def batch_write_features(request: BatchWriteRequest, background_tasks: BackgroundTasks):
    """Batch write features (for offline/training scenarios)"""
    try:
        # Validate storage backend
        try:
            store_enum = FeatureStoreEnum(request.store)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid store: {request.store}"
            )
        
        # Convert to list of tuples
        features = [
            (f['feature_id'], f['entity_id'], f['value'])
            for f in request.features
        ]
        
        # Batch write
        stats = await feature_store.batch_write_features(
            features=features,
            tenant_id=request.tenant_id,
            version=request.version,
            store=store_enum
        )
        
        return {
            'status': 'success',
            'statistics': stats,
            'message': f"Wrote {stats['successful']}/{stats['total']} features successfully"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/lineage/{feature_id}")
async def get_feature_lineage(
    feature_id: str,
    tenant_id: str = Query("1300"),
    version: str = Query("1.0.0")
):
    """Get lineage information for a feature"""
    try:
        lineage = await feature_store.get_feature_lineage(
            feature_id=feature_id,
            tenant_id=tenant_id,
            version=version
        )
        
        if 'error' in lineage:
            raise HTTPException(status_code=404, detail=lineage['error'])
        
        return {
            'status': 'success',
            'lineage': lineage
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/list/{tenant_id}")
async def list_features(tenant_id: str, limit: int = Query(100, le=1000)):
    """List all features for a tenant"""
    try:
        # Get features from registry (filtered by tenant)
        tenant_features = [
            {
                'feature_id': metadata.feature_id,
                'feature_name': metadata.feature_name,
                'feature_type': metadata.feature_type.value,
                'version': metadata.version,
                'owner': metadata.owner,
                'storage_backend': metadata.storage_backend.value,
                'created_at': metadata.created_at.isoformat()
            }
            for key, metadata in feature_store.feature_registry.items()
            if key.startswith(f"{tenant_id}:")
        ][:limit]
        
        return {
            'status': 'success',
            'tenant_id': tenant_id,
            'feature_count': len(tenant_features),
            'features': tenant_features
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'RBIA Feature Store Service',
        'version': '1.0.0',
        'features_registered': len(feature_store.feature_registry),
        'offline_storage_size': len(feature_store.offline_storage),
        'online_cache_size': len(feature_store.online_cache)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8016)

