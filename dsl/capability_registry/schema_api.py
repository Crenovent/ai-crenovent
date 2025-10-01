"""
CHAPTER 14.1 CAPABILITY REGISTRY SCHEMA - TASK 14.1.10
Build Registry Schema APIs (CRUD schema objects - FastAPI + JWT)

This module implements the REST API endpoints for managing capability schemas
according to Build Plan Epic 1 specifications.
"""

from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
import asyncpg
import json
import os
from enum import Enum

# Import existing authentication and database utilities
# from ...auth.jwt_handler import get_current_user
# from ...database.connection import get_database_connection
# Note: These imports are temporarily commented out to avoid import errors
# The dynamic capabilities system will use its own authentication system

# =============================================================================
# PYDANTIC MODELS FOR API SCHEMAS
# =============================================================================

class SchemaType(str, Enum):
    RBA_TEMPLATE = "RBA_TEMPLATE"
    RBIA_MODEL = "RBIA_MODEL"
    AALA_AGENT = "AALA_AGENT"

class Industry(str, Enum):
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    ECOMMERCE = "E-commerce"
    IT_SERVICES = "IT-Services"
    GLOBAL = "Global"

class SchemaStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class LifecycleState(str, Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    SANDBOX_CERT = "SANDBOX_CERT"
    BETA = "BETA"
    CERTIFIED = "CERTIFIED"
    ENTERPRISE = "ENTERPRISE"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"

class SLATier(str, Enum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"

class RelationType(str, Enum):
    DEPENDS_ON = "DEPENDS_ON"
    EXTENDS = "EXTENDS"
    REPLACES = "REPLACES"
    SUPERSEDES = "SUPERSEDES"
    IMPLEMENTS = "IMPLEMENTS"
    USES = "USES"
    TRIGGERS = "TRIGGERS"
    FALLBACK_FOR = "FALLBACK_FOR"

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SchemaCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    schema_type: SchemaType
    description: Optional[str] = None
    industry: Industry
    category: Optional[str] = Field(None, max_length=100)
    owner_team: Optional[str] = Field(None, max_length=100)
    policy_pack_id: Optional[str] = Field(None, max_length=255)
    compliance_frameworks: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    status: SchemaStatus = SchemaStatus.DRAFT

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty or whitespace only')
        return v.strip()

class SchemaVersionCreateRequest(BaseModel):
    version: str = Field(..., regex=r'^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$')
    schema_definition: Dict[str, Any]
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    config_schema: Optional[Dict[str, Any]] = None
    changelog: Optional[str] = None
    migration_notes: Optional[str] = None
    breaking_changes: bool = False
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    estimated_cost_per_execution: float = Field(default=0.0, ge=0.0)
    lifecycle_state: LifecycleState = LifecycleState.DRAFT
    is_current_version: bool = False

class SchemaBindingCreateRequest(BaseModel):
    schema_id: UUID
    bound_to_tenant_id: int
    sla_tier: SLATier
    industry_overlay: Optional[str] = None
    industry_specific_config: Dict[str, Any] = Field(default_factory=dict)
    compliance_packs: List[str] = Field(default_factory=list)
    policy_overrides: Dict[str, Any] = Field(default_factory=dict)
    authorized_roles: List[str] = Field(default_factory=list)
    authorized_regions: List[str] = Field(default_factory=list)
    usage_limits: Dict[str, Any] = Field(default_factory=dict)
    binding_reason: Optional[str] = Field(None, max_length=255)
    binding_metadata: Dict[str, Any] = Field(default_factory=dict)

class SchemaRelationCreateRequest(BaseModel):
    source_schema_id: UUID
    target_schema_id: UUID
    relation_type: RelationType
    relationship_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    relation_metadata: Dict[str, Any] = Field(default_factory=dict)
    relationship_config: Dict[str, Any] = Field(default_factory=dict)
    execution_order: Optional[int] = None
    is_required_dependency: bool = True
    is_blocking_dependency: bool = False
    source_version_constraint: Optional[str] = None
    target_version_constraint: Optional[str] = None
    reasoning: Optional[str] = None

    @validator('source_schema_id', 'target_schema_id')
    def validate_different_schemas(cls, v, values):
        if 'source_schema_id' in values and v == values['source_schema_id']:
            raise ValueError('Source and target schema cannot be the same')
        return v

# Response Models
class SchemaResponse(BaseModel):
    schema_id: UUID
    name: str
    schema_type: SchemaType
    description: Optional[str]
    industry: Industry
    category: Optional[str]
    tenant_id: int
    owner_team: Optional[str]
    created_by_user_id: int
    created_at: datetime
    updated_at: datetime
    status: SchemaStatus
    policy_pack_id: Optional[str]
    compliance_frameworks: List[str]
    tags: List[str]

class SchemaVersionResponse(BaseModel):
    version_id: UUID
    schema_id: UUID
    version: str
    lifecycle_state: LifecycleState
    is_current_version: bool
    schema_definition: Dict[str, Any]
    input_schema: Optional[Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]]
    config_schema: Optional[Dict[str, Any]]
    changelog: Optional[str]
    breaking_changes: bool
    avg_execution_time_ms: int
    success_rate: float
    trust_score: float
    resource_requirements: Dict[str, Any]
    estimated_cost_per_execution: float
    tenant_id: int
    created_by_user_id: int
    created_at: datetime
    updated_at: datetime

class SchemaBindingResponse(BaseModel):
    binding_id: UUID
    schema_id: UUID
    tenant_id: int
    bound_to_tenant_id: int
    sla_tier: SLATier
    industry_overlay: Optional[str]
    compliance_packs: List[str]
    authorized_roles: List[str]
    authorized_regions: List[str]
    is_active: bool
    created_at: datetime

class SchemaRelationResponse(BaseModel):
    relation_id: UUID
    source_schema_id: UUID
    target_schema_id: UUID
    relation_type: RelationType
    relationship_strength: float
    confidence_score: float
    relation_metadata: Dict[str, Any]
    is_active: bool
    tenant_id: int
    created_at: datetime

# =============================================================================
# FASTAPI ROUTER SETUP
# =============================================================================

router = APIRouter(prefix="/api/v1/capability-registry/schemas", tags=["Capability Registry Schemas"])
security = HTTPBearer()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def get_tenant_id_from_user(current_user: dict) -> int:
    """Extract tenant_id from current user context"""
    tenant_id = current_user.get('tenant_id')
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must have a valid tenant_id"
        )
    return int(tenant_id)

async def set_tenant_context(conn: asyncpg.Connection, tenant_id: int):
    """Set the tenant context for RLS policies"""
    await conn.execute("SET app.current_tenant_id = $1", str(tenant_id))

# =============================================================================
# SCHEMA CRUD ENDPOINTS
# =============================================================================

@router.post("/", response_model=SchemaResponse, status_code=status.HTTP_201_CREATED)
async def create_schema(
    schema_request: SchemaCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new capability schema
    
    **Task 14.1.10**: Build Registry Schema APIs (CRUD schema objects)
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    user_id = current_user['user_id']
    
    conn = await get_database_connection()
    try:
        await set_tenant_context(conn, tenant_id)
        
        # Insert new schema
        query = """
        INSERT INTO capability_schema_meta (
            name, schema_type, description, industry, category,
            tenant_id, owner_team, created_by_user_id, status,
            policy_pack_id, compliance_frameworks, tags
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING schema_id, name, schema_type, description, industry, category,
                  tenant_id, owner_team, created_by_user_id, created_at, updated_at,
                  status, policy_pack_id, compliance_frameworks, tags
        """
        
        row = await conn.fetchrow(
            query,
            schema_request.name,
            schema_request.schema_type.value,
            schema_request.description,
            schema_request.industry.value,
            schema_request.category,
            tenant_id,
            schema_request.owner_team,
            user_id,
            schema_request.status.value,
            schema_request.policy_pack_id,
            schema_request.compliance_frameworks,
            schema_request.tags
        )
        
        return SchemaResponse(**dict(row))
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create schema: {str(e)}"
        )
    finally:
        await conn.close()

@router.get("/{schema_id}", response_model=SchemaResponse)
async def get_schema(
    schema_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific capability schema by ID"""
    tenant_id = await get_tenant_id_from_user(current_user)
    
    conn = await get_database_connection()
    try:
        await set_tenant_context(conn, tenant_id)
        
        query = """
        SELECT schema_id, name, schema_type, description, industry, category,
               tenant_id, owner_team, created_by_user_id, created_at, updated_at,
               status, policy_pack_id, compliance_frameworks, tags
        FROM capability_schema_meta
        WHERE schema_id = $1
        """
        
        row = await conn.fetchrow(query, schema_id)
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema {schema_id} not found"
            )
        
        return SchemaResponse(**dict(row))
        
    finally:
        await conn.close()

@router.get("/", response_model=List[SchemaResponse])
async def list_schemas(
    schema_type: Optional[SchemaType] = None,
    industry: Optional[Industry] = None,
    status: Optional[SchemaStatus] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """List capability schemas with optional filtering"""
    tenant_id = await get_tenant_id_from_user(current_user)
    
    conn = await get_database_connection()
    try:
        await set_tenant_context(conn, tenant_id)
        
        # Build dynamic query with filters
        where_conditions = ["tenant_id = $1"]
        params = [tenant_id]
        param_count = 1
        
        if schema_type:
            param_count += 1
            where_conditions.append(f"schema_type = ${param_count}")
            params.append(schema_type.value)
        
        if industry:
            param_count += 1
            where_conditions.append(f"industry = ${param_count}")
            params.append(industry.value)
            
        if status:
            param_count += 1
            where_conditions.append(f"status = ${param_count}")
            params.append(status.value)
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        SELECT schema_id, name, schema_type, description, industry, category,
               tenant_id, owner_team, created_by_user_id, created_at, updated_at,
               status, policy_pack_id, compliance_frameworks, tags
        FROM capability_schema_meta
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        
        params.extend([limit, offset])
        rows = await conn.fetch(query, *params)
        
        return [SchemaResponse(**dict(row)) for row in rows]
        
    finally:
        await conn.close()

@router.put("/{schema_id}", response_model=SchemaResponse)
async def update_schema(
    schema_id: UUID,
    schema_request: SchemaCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update an existing capability schema"""
    tenant_id = await get_tenant_id_from_user(current_user)
    user_id = current_user['user_id']
    
    conn = await get_database_connection()
    try:
        await set_tenant_context(conn, tenant_id)
        
        # Check if schema exists
        exists_query = "SELECT 1 FROM capability_schema_meta WHERE schema_id = $1"
        exists = await conn.fetchval(exists_query, schema_id)
        
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema {schema_id} not found"
            )
        
        # Update schema
        query = """
        UPDATE capability_schema_meta SET
            name = $2, schema_type = $3, description = $4, industry = $5,
            category = $6, owner_team = $7, status = $8, policy_pack_id = $9,
            compliance_frameworks = $10, tags = $11, updated_at = NOW()
        WHERE schema_id = $1
        RETURNING schema_id, name, schema_type, description, industry, category,
                  tenant_id, owner_team, created_by_user_id, created_at, updated_at,
                  status, policy_pack_id, compliance_frameworks, tags
        """
        
        row = await conn.fetchrow(
            query,
            schema_id,
            schema_request.name,
            schema_request.schema_type.value,
            schema_request.description,
            schema_request.industry.value,
            schema_request.category,
            schema_request.owner_team,
            schema_request.status.value,
            schema_request.policy_pack_id,
            schema_request.compliance_frameworks,
            schema_request.tags
        )
        
        return SchemaResponse(**dict(row))
        
    finally:
        await conn.close()

@router.delete("/{schema_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schema(
    schema_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """Delete a capability schema (soft delete by setting status to retired)"""
    tenant_id = await get_tenant_id_from_user(current_user)
    
    conn = await get_database_connection()
    try:
        await set_tenant_context(conn, tenant_id)
        
        # Soft delete by updating status
        query = """
        UPDATE capability_schema_meta 
        SET status = 'retired', updated_at = NOW()
        WHERE schema_id = $1
        """
        
        result = await conn.execute(query, schema_id)
        
        if result == "UPDATE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema {schema_id} not found"
            )
        
    finally:
        await conn.close()

# =============================================================================
# SCHEMA VERSION ENDPOINTS
# =============================================================================

@router.post("/{schema_id}/versions", response_model=SchemaVersionResponse, status_code=status.HTTP_201_CREATED)
async def create_schema_version(
    schema_id: UUID,
    version_request: SchemaVersionCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new version of a capability schema"""
    tenant_id = await get_tenant_id_from_user(current_user)
    user_id = current_user['user_id']
    
    conn = await get_database_connection()
    try:
        await set_tenant_context(conn, tenant_id)
        
        # Verify schema exists
        schema_exists = await conn.fetchval(
            "SELECT 1 FROM capability_schema_meta WHERE schema_id = $1", schema_id
        )
        
        if not schema_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema {schema_id} not found"
            )
        
        # Insert new version
        query = """
        INSERT INTO capability_schema_version (
            schema_id, version, lifecycle_state, schema_definition,
            input_schema, output_schema, config_schema, changelog,
            migration_notes, breaking_changes, resource_requirements,
            estimated_cost_per_execution, tenant_id, created_by_user_id,
            is_current_version
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        RETURNING version_id, schema_id, version, lifecycle_state, is_current_version,
                  schema_definition, input_schema, output_schema, config_schema,
                  changelog, breaking_changes, avg_execution_time_ms, success_rate,
                  trust_score, resource_requirements, estimated_cost_per_execution,
                  tenant_id, created_by_user_id, created_at, updated_at
        """
        
        row = await conn.fetchrow(
            query,
            schema_id,
            version_request.version,
            version_request.lifecycle_state.value,
            json.dumps(version_request.schema_definition),
            json.dumps(version_request.input_schema) if version_request.input_schema else None,
            json.dumps(version_request.output_schema) if version_request.output_schema else None,
            json.dumps(version_request.config_schema) if version_request.config_schema else None,
            version_request.changelog,
            version_request.migration_notes,
            version_request.breaking_changes,
            json.dumps(version_request.resource_requirements),
            version_request.estimated_cost_per_execution,
            tenant_id,
            user_id,
            version_request.is_current_version
        )
        
        return SchemaVersionResponse(**dict(row))
        
    finally:
        await conn.close()

@router.get("/{schema_id}/versions", response_model=List[SchemaVersionResponse])
async def list_schema_versions(
    schema_id: UUID,
    current_user: dict = Depends(get_current_user)
):
    """List all versions of a capability schema"""
    tenant_id = await get_tenant_id_from_user(current_user)
    
    conn = await get_database_connection()
    try:
        await set_tenant_context(conn, tenant_id)
        
        query = """
        SELECT version_id, schema_id, version, lifecycle_state, is_current_version,
               schema_definition, input_schema, output_schema, config_schema,
               changelog, breaking_changes, avg_execution_time_ms, success_rate,
               trust_score, resource_requirements, estimated_cost_per_execution,
               tenant_id, created_by_user_id, created_at, updated_at
        FROM capability_schema_version
        WHERE schema_id = $1
        ORDER BY created_at DESC
        """
        
        rows = await conn.fetch(query, schema_id)
        return [SchemaVersionResponse(**dict(row)) for row in rows]
        
    finally:
        await conn.close()

# =============================================================================
# SEARCH AND DISCOVERY ENDPOINTS
# =============================================================================

@router.get("/search", response_model=List[SchemaResponse])
async def search_schemas(
    q: Optional[str] = Query(None, description="Search query for full-text search"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    compliance: Optional[List[str]] = Query(None, description="Filter by compliance frameworks"),
    limit: int = Query(default=20, le=100),
    current_user: dict = Depends(get_current_user)
):
    """
    Search capability schemas using full-text search and filters
    
    Supports:
    - Full-text search across name, description, tags
    - Tag filtering
    - Compliance framework filtering
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    
    conn = await get_database_connection()
    try:
        await set_tenant_context(conn, tenant_id)
        
        where_conditions = ["tenant_id = $1", "status != 'retired'"]
        params = [tenant_id]
        param_count = 1
        
        # Full-text search
        if q:
            param_count += 1
            where_conditions.append(f"search_vector @@ plainto_tsquery('english', ${param_count})")
            params.append(q)
        
        # Tag filtering
        if tags:
            param_count += 1
            where_conditions.append(f"tags && ${param_count}")
            params.append(tags)
        
        # Compliance filtering
        if compliance:
            param_count += 1
            where_conditions.append(f"compliance_frameworks && ${param_count}")
            params.append(compliance)
        
        where_clause = " AND ".join(where_conditions)
        
        # Add ranking for full-text search
        order_clause = "ORDER BY created_at DESC"
        if q:
            order_clause = f"ORDER BY ts_rank(search_vector, plainto_tsquery('english', ${params.index(q) + 1})) DESC, created_at DESC"
        
        query = f"""
        SELECT schema_id, name, schema_type, description, industry, category,
               tenant_id, owner_team, created_by_user_id, created_at, updated_at,
               status, policy_pack_id, compliance_frameworks, tags
        FROM capability_schema_meta
        WHERE {where_clause}
        {order_clause}
        LIMIT ${param_count + 1}
        """
        
        params.append(limit)
        rows = await conn.fetch(query, *params)
        
        return [SchemaResponse(**dict(row)) for row in rows]
        
    finally:
        await conn.close()

# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint for capability registry schema API"""
    return {
        "status": "healthy",
        "service": "capability-registry-schema-api",
        "version": "1.0.0",
        "build_plan_task": "14.1.10",
        "timestamp": datetime.utcnow().isoformat()
    }
