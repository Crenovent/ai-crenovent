"""
Schema Registry API
Task 9.2.3: Build Schema Registry service API endpoints

Provides REST API for schema management, validation, and governance
"""

from fastapi import APIRouter, HTTPException, Depends, Body, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from dsl.registry.comprehensive_schema_registry import (
    ComprehensiveSchemaRegistry, SchemaType, SchemaCompatibility, 
    SchemaStatus, IndustryOverlay, SchemaMetadata
)
from src.services.connection_pool_manager import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API requests/responses

class RegisterSchemaRequest(BaseModel):
    schema_name: str = Field(..., description="Name of the schema")
    schema_content: Dict[str, Any] = Field(..., description="Schema definition")
    schema_type: SchemaType = Field(..., description="Type of schema")
    tenant_id: int = Field(..., description="Tenant identifier")
    industry_overlay: IndustryOverlay = Field(..., description="Industry overlay")
    description: str = Field("", description="Schema description")
    tags: List[str] = Field(default_factory=list, description="Schema tags")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")
    created_by: str = Field(..., description="User who created the schema")

class EvolveSchemaRequest(BaseModel):
    schema_id: str = Field(..., description="Schema ID to evolve")
    new_schema_content: Dict[str, Any] = Field(..., description="New schema definition")
    new_version: str = Field(..., description="New version number")
    updated_by: str = Field(..., description="User updating the schema")
    compatibility_level: Optional[SchemaCompatibility] = Field(None, description="Compatibility level")

class ValidateSchemaRequest(BaseModel):
    schema_id: str = Field(..., description="Schema ID for validation")
    data: Dict[str, Any] = Field(..., description="Data to validate")

class SchemaResponse(BaseModel):
    schema_id: str
    schema_name: str
    version: str
    schema_type: SchemaType
    tenant_id: int
    industry_overlay: IndustryOverlay
    status: SchemaStatus
    created_at: datetime
    updated_at: datetime

class ValidationResponse(BaseModel):
    valid: bool
    errors: List[str]
    schema_id: str
    validation_timestamp: datetime

class CompatibilityResponse(BaseModel):
    compatible: bool
    issues: List[str]
    compatibility_level: str

class SchemaDependencyResponse(BaseModel):
    schema_id: str
    dependencies: List[str]
    dependents: List[str]

# Global schema registry instance
schema_registry = None

async def get_schema_registry(pool_manager=Depends(get_pool_manager)) -> ComprehensiveSchemaRegistry:
    """Get or create schema registry instance"""
    global schema_registry
    if schema_registry is None:
        schema_registry = ComprehensiveSchemaRegistry(pool_manager)
        await schema_registry.initialize()
    return schema_registry

@router.post("/register", response_model=Dict[str, str])
async def register_schema(
    request: RegisterSchemaRequest,
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Register a new schema in the registry
    Task 9.2.3: Schema registration with governance
    """
    try:
        schema_id = await registry.register_schema(
            schema_name=request.schema_name,
            schema_content=request.schema_content,
            schema_type=request.schema_type,
            tenant_id=request.tenant_id,
            industry_overlay=request.industry_overlay,
            created_by=request.created_by,
            description=request.description,
            tags=request.tags,
            compliance_frameworks=request.compliance_frameworks
        )
        
        return {
            "schema_id": schema_id,
            "message": f"Schema '{request.schema_name}' registered successfully",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Schema registration failed: {e}")
        raise HTTPException(status_code=400, detail=f"Schema registration failed: {str(e)}")

@router.post("/evolve", response_model=Dict[str, Any])
async def evolve_schema(
    request: EvolveSchemaRequest,
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Evolve an existing schema to a new version
    Task 9.2.3: Schema evolution with compatibility validation
    """
    try:
        success = await registry.evolve_schema(
            schema_id=request.schema_id,
            new_schema_content=request.new_schema_content,
            new_version=request.new_version,
            updated_by=request.updated_by,
            compatibility_level=request.compatibility_level
        )
        
        if success:
            return {
                "schema_id": request.schema_id,
                "new_version": request.new_version,
                "message": "Schema evolved successfully",
                "status": "success"
            }
        else:
            return {
                "schema_id": request.schema_id,
                "new_version": request.new_version,
                "message": "Schema evolution requires approval due to breaking changes",
                "status": "pending_approval"
            }
        
    except Exception as e:
        logger.error(f"Schema evolution failed: {e}")
        raise HTTPException(status_code=400, detail=f"Schema evolution failed: {str(e)}")

@router.get("/schema/{schema_id}", response_model=Dict[str, Any])
async def get_schema(
    schema_id: str,
    version: Optional[str] = Query(None, description="Specific version to retrieve"),
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Get schema by ID and optional version
    Task 9.2.3: Schema retrieval
    """
    try:
        schema_version = await registry.get_schema(schema_id, version)
        
        if not schema_version:
            raise HTTPException(status_code=404, detail=f"Schema not found: {schema_id}")
        
        return {
            "schema_id": schema_version.schema_id,
            "version": schema_version.version,
            "schema_content": schema_version.schema_content,
            "metadata": schema_version.metadata.to_dict(),
            "compatibility_report": schema_version.compatibility_report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get schema {schema_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve schema: {str(e)}")

@router.get("/schemas", response_model=List[Dict[str, Any]])
async def list_schemas(
    tenant_id: Optional[int] = Query(None, description="Filter by tenant ID"),
    industry_overlay: Optional[IndustryOverlay] = Query(None, description="Filter by industry"),
    schema_type: Optional[SchemaType] = Query(None, description="Filter by schema type"),
    status: Optional[SchemaStatus] = Query(None, description="Filter by status"),
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    List schemas with optional filters
    Task 9.2.3: Schema discovery and listing
    """
    try:
        schemas = await registry.list_schemas(
            tenant_id=tenant_id,
            industry_overlay=industry_overlay,
            schema_type=schema_type,
            status=status
        )
        
        return [schema.to_dict() for schema in schemas]
        
    except Exception as e:
        logger.error(f"Failed to list schemas: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list schemas: {str(e)}")

@router.post("/validate", response_model=ValidationResponse)
async def validate_data(
    request: ValidateSchemaRequest,
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Validate data against a registered schema
    Task 9.2.3: Schema validation
    """
    try:
        validation_result = await registry.validate_schema_usage(
            schema_id=request.schema_id,
            data=request.data
        )
        
        return ValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result["errors"],
            schema_id=request.schema_id,
            validation_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Schema validation failed: {str(e)}")

@router.post("/deprecate/{schema_id}", response_model=Dict[str, str])
async def deprecate_schema(
    schema_id: str,
    deprecated_by: str = Body(..., description="User deprecating the schema"),
    reason: str = Body(..., description="Reason for deprecation"),
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Deprecate a schema version
    Task 9.2.3: Schema lifecycle management
    """
    try:
        success = await registry.deprecate_schema(
            schema_id=schema_id,
            deprecated_by=deprecated_by,
            reason=reason
        )
        
        if success:
            return {
                "schema_id": schema_id,
                "message": "Schema deprecated successfully",
                "status": "deprecated"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to deprecate schema")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schema deprecation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Schema deprecation failed: {str(e)}")

@router.get("/dependencies/{schema_id}", response_model=SchemaDependencyResponse)
async def get_schema_dependencies(
    schema_id: str,
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Get schema dependencies and dependents
    Task 9.2.3: Schema dependency management
    """
    try:
        dependencies = await registry.get_schema_dependencies(schema_id)
        
        return SchemaDependencyResponse(
            schema_id=schema_id,
            dependencies=dependencies["dependencies"],
            dependents=dependencies["dependents"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get schema dependencies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dependencies: {str(e)}")

@router.get("/templates/{industry_overlay}", response_model=Dict[str, Any])
async def get_industry_templates(
    industry_overlay: IndustryOverlay,
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Get industry-specific schema templates
    Task 9.2.3: Industry template management
    """
    try:
        if industry_overlay in registry.industry_templates:
            templates = registry.industry_templates[industry_overlay]
            return {
                "industry": industry_overlay.value,
                "templates": templates,
                "count": len(templates)
            }
        else:
            return {
                "industry": industry_overlay.value,
                "templates": {},
                "count": 0
            }
        
    except Exception as e:
        logger.error(f"Failed to get industry templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.post("/compatibility-check", response_model=CompatibilityResponse)
async def check_compatibility(
    old_schema: Dict[str, Any] = Body(..., description="Old schema definition"),
    new_schema: Dict[str, Any] = Body(..., description="New schema definition"),
    compatibility_level: SchemaCompatibility = Body(..., description="Compatibility level to check"),
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Check compatibility between two schema versions
    Task 9.2.3: Schema compatibility validation
    """
    try:
        compatibility_result = await registry._validate_compatibility(
            old_schema, new_schema, compatibility_level
        )
        
        return CompatibilityResponse(
            compatible=compatibility_result["compatible"],
            issues=compatibility_result["issues"],
            compatibility_level=compatibility_result["compatibility_level"]
        )
        
    except Exception as e:
        logger.error(f"Compatibility check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compatibility check failed: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Health check for schema registry service
    """
    try:
        # Basic health checks
        schema_count = len(registry.schema_cache)
        
        return {
            "status": "healthy",
            "service": "schema_registry",
            "cached_schemas": schema_count,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "schema_registry",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/stats", response_model=Dict[str, Any])
async def get_registry_stats(
    registry: ComprehensiveSchemaRegistry = Depends(get_schema_registry)
):
    """
    Get schema registry statistics
    Task 9.2.3: Registry monitoring and observability
    """
    try:
        # Get all schemas for statistics
        all_schemas = await registry.list_schemas()
        
        # Calculate statistics
        stats = {
            "total_schemas": len(all_schemas),
            "schemas_by_type": {},
            "schemas_by_industry": {},
            "schemas_by_status": {},
            "schemas_by_tenant": {},
            "average_quality_score": 0.0
        }
        
        total_quality = 0.0
        
        for schema in all_schemas:
            # Count by type
            schema_type = schema.schema_type.value
            stats["schemas_by_type"][schema_type] = stats["schemas_by_type"].get(schema_type, 0) + 1
            
            # Count by industry
            industry = schema.industry_overlay.value
            stats["schemas_by_industry"][industry] = stats["schemas_by_industry"].get(industry, 0) + 1
            
            # Count by status
            status = schema.status.value
            stats["schemas_by_status"][status] = stats["schemas_by_status"].get(status, 0) + 1
            
            # Count by tenant
            tenant_id = str(schema.tenant_id)
            stats["schemas_by_tenant"][tenant_id] = stats["schemas_by_tenant"].get(tenant_id, 0) + 1
            
            # Sum quality scores
            total_quality += schema.quality_score
        
        # Calculate average quality score
        if len(all_schemas) > 0:
            stats["average_quality_score"] = total_quality / len(all_schemas)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# Export router
__all__ = ["router"]
