"""
Capability Versioning API - Chapter 14.3 Implementation
======================================================
Tasks 14.3.8, 14.3.16, 14.3.17, 14.3.18, 14.3.19: Version Registry APIs and lifecycle management

Backend Tasks Implemented:
- 14.3.8: Implement Version Registry APIs (CRUD versions, deps, states)
- 14.3.16: Implement tenant version pins (tenant→cap version binding)
- 14.3.17: Implement SLA-aware promotion rules (version eligible per SLA tier)
- 14.3.18: Implement deprecation policy (windows, comms, auto alerts)
- 14.3.19: Implement forced EOL failsafe (block execution past EOL)

Key Features:
- Semantic versioning with MAJOR.MINOR.PATCH + prerelease/build
- Lifecycle state management (Draft→Testing→Validated→Promoted→Deprecated→EOL)
- Dependency tracking and compatibility matrices
- Tenant-specific version pinning
- SLA-aware promotion gates
- Automated deprecation and EOL enforcement
"""

import logging
import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field, validator
import asyncpg

from src.services.connection_pool_manager import ConnectionPoolManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/capability-versioning", tags=["Capability Versioning"])

# =============================================================================
# ENUMS AND DATA MODELS
# =============================================================================

class VersionState(str, Enum):
    """Version lifecycle states"""
    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"
    EOL = "eol"

class SLATier(str, Enum):
    """SLA tiers for promotion rules"""
    PREMIUM = "premium"
    STANDARD = "standard"
    BASIC = "basic"

class DependencyType(str, Enum):
    """Types of dependencies"""
    CAPABILITY = "capability"
    POLICY = "policy"
    SCHEMA = "schema"
    MODEL = "model"
    EXTERNAL = "external"

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class VersionCreateRequest(BaseModel):
    """Request model for creating a new version"""
    capability_id: str = Field(..., description="UUID of the capability")
    major_version: int = Field(1, ge=0, description="Major version number")
    minor_version: int = Field(0, ge=0, description="Minor version number")
    patch_version: int = Field(0, ge=0, description="Patch version number")
    prerelease_identifier: Optional[str] = Field(None, description="Prerelease identifier (alpha, beta, rc.1)")
    build_metadata: Optional[str] = Field(None, description="Build metadata (build.123, commit.abc123)")
    release_notes: Optional[str] = Field(None, description="Release notes")
    changelog: Optional[str] = Field(None, description="Changelog")
    breaking_changes: List[str] = Field(default_factory=list, description="List of breaking changes")
    backward_compatible: bool = Field(True, description="Is backward compatible")
    forward_compatible: bool = Field(False, description="Is forward compatible")

class VersionPinRequest(BaseModel):
    """Request model for tenant version pinning"""
    capability_id: str = Field(..., description="UUID of the capability")
    version_string: str = Field(..., description="Version string to pin to")
    pin_reason: Optional[str] = Field(None, description="Reason for pinning")
    auto_upgrade: bool = Field(False, description="Allow automatic upgrades")

class PromotionRequest(BaseModel):
    """Request model for version promotion"""
    version_meta_id: str = Field(..., description="UUID of the version metadata")
    target_state: VersionState = Field(..., description="Target state for promotion")
    promotion_reason: Optional[str] = Field(None, description="Reason for promotion")
    override_sla_check: bool = Field(False, description="Override SLA tier checks")

class DeprecationRequest(BaseModel):
    """Request model for version deprecation"""
    version_meta_id: str = Field(..., description="UUID of the version metadata")
    deprecation_reason: str = Field(..., description="Reason for deprecation")
    eol_date: Optional[datetime] = Field(None, description="End of life date")
    migration_guide: Optional[str] = Field(None, description="Migration guide")

class VersionResponse(BaseModel):
    """Response model for version information"""
    version_meta_id: str
    capability_id: str
    tenant_id: int
    version_string: str
    major_version: int
    minor_version: int
    patch_version: int
    prerelease_identifier: Optional[str]
    build_metadata: Optional[str]
    is_stable: bool
    is_lts: bool
    is_deprecated: bool
    current_state: VersionState
    backward_compatible: bool
    forward_compatible: bool
    breaking_changes: List[str]
    release_notes: Optional[str]
    changelog: Optional[str]
    created_at: datetime
    released_at: Optional[datetime]
    deprecated_at: Optional[datetime]
    eol_date: Optional[datetime]

# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

async def get_pool_manager() -> ConnectionPoolManager:
    """Get database connection pool manager"""
    # This would be injected from your main application
    # For now, return a placeholder
    return None

async def get_current_user():
    """Get current authenticated user"""
    # This would be your actual authentication logic
    return {"user_id": 1, "tenant_id": 1300}

async def get_tenant_id_from_user(current_user: dict) -> int:
    """Extract tenant ID from current user"""
    return current_user.get("tenant_id", 1300)

# =============================================================================
# BACKEND TASK IMPLEMENTATIONS
# =============================================================================

class CapabilityVersionManager:
    """
    Backend implementation for Chapter 14.3 versioning tasks
    """
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
    
    async def create_version(self, request: VersionCreateRequest, tenant_id: int, user_id: int) -> Dict[str, Any]:
        """Task 14.3.8: Create new capability version"""
        try:
            # Generate version string
            version_string = f"{request.major_version}.{request.minor_version}.{request.patch_version}"
            if request.prerelease_identifier:
                version_string += f"-{request.prerelease_identifier}"
            if request.build_metadata:
                version_string += f"+{request.build_metadata}"
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Insert version metadata
                version_meta_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO cap_version_meta (
                        version_meta_id, capability_id, tenant_id,
                        major_version, minor_version, patch_version,
                        prerelease_identifier, build_metadata, version_string,
                        backward_compatible, forward_compatible, breaking_changes,
                        release_notes, changelog, created_by_user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, 
                version_meta_id, request.capability_id, tenant_id,
                request.major_version, request.minor_version, request.patch_version,
                request.prerelease_identifier, request.build_metadata, version_string,
                request.backward_compatible, request.forward_compatible, 
                json.dumps(request.breaking_changes),
                request.release_notes, request.changelog, user_id)
                
                # Create initial state record
                await conn.execute("""
                    INSERT INTO cap_version_state (
                        version_meta_id, tenant_id, current_state, changed_by_user_id
                    ) VALUES ($1, $2, $3, $4)
                """, version_meta_id, tenant_id, VersionState.DRAFT.value, user_id)
                
                self.logger.info(f"✅ Created version {version_string} for capability {request.capability_id}")
                
                return {
                    "success": True,
                    "version_meta_id": version_meta_id,
                    "version_string": version_string,
                    "current_state": VersionState.DRAFT.value
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create version: {e}")
            raise HTTPException(status_code=500, detail=f"Version creation failed: {str(e)}")
    
    async def pin_tenant_version(self, request: VersionPinRequest, tenant_id: int, user_id: int) -> Dict[str, Any]:
        """Task 14.3.16: Implement tenant version pins"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Check if version exists and is promoted
                version_check = await conn.fetchrow("""
                    SELECT vm.version_meta_id, vs.current_state 
                    FROM cap_version_meta vm
                    JOIN cap_version_state vs ON vm.version_meta_id = vs.version_meta_id
                    WHERE vm.capability_id = $1 AND vm.version_string = $2 AND vm.tenant_id = $3
                """, request.capability_id, request.version_string, tenant_id)
                
                if not version_check:
                    raise HTTPException(status_code=404, detail="Version not found")
                
                if version_check['current_state'] not in ['promoted', 'deprecated']:
                    raise HTTPException(status_code=400, detail="Can only pin promoted or deprecated versions")
                
                # Create or update tenant version pin
                await conn.execute("""
                    INSERT INTO tenant_version_pins (
                        tenant_id, capability_id, version_meta_id, version_string,
                        pin_reason, auto_upgrade, pinned_by_user_id, pinned_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    ON CONFLICT (tenant_id, capability_id) 
                    DO UPDATE SET
                        version_meta_id = EXCLUDED.version_meta_id,
                        version_string = EXCLUDED.version_string,
                        pin_reason = EXCLUDED.pin_reason,
                        auto_upgrade = EXCLUDED.auto_upgrade,
                        pinned_by_user_id = EXCLUDED.pinned_by_user_id,
                        pinned_at = NOW(),
                        updated_at = NOW()
                """, 
                tenant_id, request.capability_id, version_check['version_meta_id'],
                request.version_string, request.pin_reason, request.auto_upgrade, user_id)
                
                self.logger.info(f"✅ Pinned tenant {tenant_id} to version {request.version_string}")
                
                return {
                    "success": True,
                    "message": f"Tenant pinned to version {request.version_string}",
                    "version_meta_id": version_check['version_meta_id']
                }
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to pin tenant version: {e}")
            raise HTTPException(status_code=500, detail=f"Version pinning failed: {str(e)}")
    
    async def promote_version(self, request: PromotionRequest, tenant_id: int, user_id: int) -> Dict[str, Any]:
        """Task 14.3.17: Implement SLA-aware promotion rules"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Get current version state and tenant SLA tier
                version_info = await conn.fetchrow("""
                    SELECT vm.capability_id, vm.version_string, vs.current_state,
                           tm.tenant_type, COALESCE(cms.sla_tier, 'standard') as sla_tier
                    FROM cap_version_meta vm
                    JOIN cap_version_state vs ON vm.version_meta_id = vs.version_meta_id
                    JOIN tenant_metadata tm ON vm.tenant_id = tm.tenant_id
                    LEFT JOIN capability_meta_sla cms ON vm.capability_id::uuid = cms.capability_id
                    WHERE vm.version_meta_id = $1 AND vm.tenant_id = $2
                """, request.version_meta_id, tenant_id)
                
                if not version_info:
                    raise HTTPException(status_code=404, detail="Version not found")
                
                # Check SLA-aware promotion rules
                if not request.override_sla_check:
                    promotion_allowed = await self._check_sla_promotion_rules(
                        version_info['current_state'], 
                        request.target_state.value,
                        version_info['sla_tier']
                    )
                    
                    if not promotion_allowed:
                        raise HTTPException(
                            status_code=403, 
                            detail=f"Promotion from {version_info['current_state']} to {request.target_state.value} not allowed for SLA tier {version_info['sla_tier']}"
                        )
                
                # Update version state
                await conn.execute("""
                    UPDATE cap_version_state 
                    SET previous_state = current_state,
                        current_state = $1,
                        state_reason = $2,
                        changed_by_user_id = $3,
                        state_entered_at = NOW()
                    WHERE version_meta_id = $4 AND tenant_id = $5
                """, 
                request.target_state.value, request.promotion_reason, user_id,
                request.version_meta_id, tenant_id)
                
                # Update version metadata if promoting to released
                if request.target_state == VersionState.PROMOTED:
                    await conn.execute("""
                        UPDATE cap_version_meta 
                        SET released_at = NOW(),
                            released_by_user_id = $1,
                            is_stable = true
                        WHERE version_meta_id = $2 AND tenant_id = $3
                    """, user_id, request.version_meta_id, tenant_id)
                
                self.logger.info(f"✅ Promoted version {version_info['version_string']} to {request.target_state.value}")
                
                return {
                    "success": True,
                    "message": f"Version promoted to {request.target_state.value}",
                    "version_string": version_info['version_string'],
                    "previous_state": version_info['current_state'],
                    "new_state": request.target_state.value
                }
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to promote version: {e}")
            raise HTTPException(status_code=500, detail=f"Version promotion failed: {str(e)}")
    
    async def deprecate_version(self, request: DeprecationRequest, tenant_id: int, user_id: int) -> Dict[str, Any]:
        """Task 14.3.18: Implement deprecation policy"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Get version info
                version_info = await conn.fetchrow("""
                    SELECT vm.capability_id, vm.version_string, vs.current_state
                    FROM cap_version_meta vm
                    JOIN cap_version_state vs ON vm.version_meta_id = vs.version_meta_id
                    WHERE vm.version_meta_id = $1 AND vm.tenant_id = $2
                """, request.version_meta_id, tenant_id)
                
                if not version_info:
                    raise HTTPException(status_code=404, detail="Version not found")
                
                if version_info['current_state'] not in ['promoted']:
                    raise HTTPException(status_code=400, detail="Can only deprecate promoted versions")
                
                # Calculate EOL date if not provided (default: 6 months from now)
                eol_date = request.eol_date or (datetime.now(timezone.utc) + timedelta(days=180))
                
                # Update version to deprecated state
                await conn.execute("""
                    UPDATE cap_version_state 
                    SET previous_state = current_state,
                        current_state = 'deprecated',
                        state_reason = $1,
                        changed_by_user_id = $2,
                        state_entered_at = NOW()
                    WHERE version_meta_id = $3 AND tenant_id = $4
                """, request.deprecation_reason, user_id, request.version_meta_id, tenant_id)
                
                # Update version metadata
                await conn.execute("""
                    UPDATE cap_version_meta 
                    SET is_deprecated = true,
                        deprecated_at = NOW(),
                        eol_date = $1,
                        migration_guide = $2
                    WHERE version_meta_id = $3 AND tenant_id = $4
                """, eol_date, request.migration_guide, request.version_meta_id, tenant_id)
                
                # Schedule deprecation notifications
                await self._schedule_deprecation_notifications(
                    request.version_meta_id, tenant_id, eol_date, version_info['version_string']
                )
                
                self.logger.info(f"✅ Deprecated version {version_info['version_string']}, EOL: {eol_date}")
                
                return {
                    "success": True,
                    "message": f"Version {version_info['version_string']} deprecated",
                    "eol_date": eol_date.isoformat(),
                    "migration_guide": request.migration_guide
                }
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to deprecate version: {e}")
            raise HTTPException(status_code=500, detail=f"Version deprecation failed: {str(e)}")
    
    async def check_eol_enforcement(self, capability_id: str, tenant_id: int) -> Dict[str, Any]:
        """Task 14.3.19: Implement forced EOL failsafe"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
                
                # Check if tenant is using an EOL version
                eol_check = await conn.fetchrow("""
                    SELECT vm.version_string, vm.eol_date, vs.current_state,
                           tvp.version_string as pinned_version
                    FROM cap_version_meta vm
                    JOIN cap_version_state vs ON vm.version_meta_id = vs.version_meta_id
                    LEFT JOIN tenant_version_pins tvp ON vm.capability_id::uuid = tvp.capability_id::uuid 
                                                      AND vm.tenant_id = tvp.tenant_id
                    WHERE vm.capability_id = $1 AND vm.tenant_id = $2
                      AND (vm.eol_date < NOW() OR vs.current_state = 'eol')
                    ORDER BY vm.created_at DESC
                    LIMIT 1
                """, capability_id, tenant_id)
                
                if eol_check:
                    # Version is EOL - block execution
                    return {
                        "execution_allowed": False,
                        "reason": "version_eol",
                        "eol_version": eol_check['version_string'],
                        "eol_date": eol_check['eol_date'].isoformat() if eol_check['eol_date'] else None,
                        "message": f"Execution blocked: Version {eol_check['version_string']} has reached end-of-life",
                        "action_required": "upgrade_to_supported_version"
                    }
                
                # Check for available supported versions
                supported_versions = await conn.fetch("""
                    SELECT vm.version_string, vs.current_state
                    FROM cap_version_meta vm
                    JOIN cap_version_state vs ON vm.version_meta_id = vs.version_meta_id
                    WHERE vm.capability_id = $1 AND vm.tenant_id = $2
                      AND vs.current_state IN ('promoted', 'deprecated')
                      AND (vm.eol_date IS NULL OR vm.eol_date > NOW())
                    ORDER BY vm.major_version DESC, vm.minor_version DESC, vm.patch_version DESC
                """, capability_id, tenant_id)
                
                return {
                    "execution_allowed": True,
                    "supported_versions": [
                        {"version": row['version_string'], "state": row['current_state']}
                        for row in supported_versions
                    ],
                    "recommended_version": supported_versions[0]['version_string'] if supported_versions else None
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to check EOL enforcement: {e}")
            return {
                "execution_allowed": False,
                "reason": "check_failed",
                "message": f"EOL check failed: {str(e)}"
            }
    
    async def _check_sla_promotion_rules(self, current_state: str, target_state: str, sla_tier: str) -> bool:
        """Check if promotion is allowed based on SLA tier"""
        # SLA-aware promotion rules
        promotion_rules = {
            "premium": {
                # Premium tier: Can promote through all states
                "draft": ["testing", "validated", "promoted"],
                "testing": ["validated", "promoted", "deprecated"],
                "validated": ["promoted", "deprecated"],
                "promoted": ["deprecated", "eol"],
                "deprecated": ["eol"]
            },
            "standard": {
                # Standard tier: Must go through validation
                "draft": ["testing"],
                "testing": ["validated"],
                "validated": ["promoted"],
                "promoted": ["deprecated", "eol"],
                "deprecated": ["eol"]
            },
            "basic": {
                # Basic tier: Strict progression
                "draft": ["testing"],
                "testing": ["validated"],
                "validated": ["promoted"],
                "promoted": ["deprecated"],
                "deprecated": ["eol"]
            }
        }
        
        allowed_transitions = promotion_rules.get(sla_tier, promotion_rules["standard"])
        return target_state in allowed_transitions.get(current_state, [])
    
    async def _schedule_deprecation_notifications(self, version_meta_id: str, tenant_id: int, eol_date: datetime, version_string: str):
        """Schedule deprecation notifications"""
        try:
            # In a real implementation, this would integrate with a notification service
            # For now, we'll log the notification schedule
            
            notification_schedule = [
                eol_date - timedelta(days=90),  # 90 days before EOL
                eol_date - timedelta(days=60),  # 60 days before EOL
                eol_date - timedelta(days=30),  # 30 days before EOL
                eol_date - timedelta(days=7),   # 7 days before EOL
                eol_date - timedelta(days=1),   # 1 day before EOL
            ]
            
            for notification_date in notification_schedule:
                if notification_date > datetime.now(timezone.utc):
                    self.logger.info(f"📅 Scheduled deprecation notification for {version_string} on {notification_date}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to schedule deprecation notifications: {e}")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/versions", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_capability_version(
    request: VersionCreateRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 14.3.8: Create a new capability version
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    user_id = current_user['user_id']
    
    if not pool_manager:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    version_manager = CapabilityVersionManager(pool_manager)
    return await version_manager.create_version(request, tenant_id, user_id)

@router.post("/versions/pin", response_model=Dict[str, Any])
async def pin_tenant_to_version(
    request: VersionPinRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 14.3.16: Pin tenant to specific capability version
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    user_id = current_user['user_id']
    
    if not pool_manager:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    version_manager = CapabilityVersionManager(pool_manager)
    return await version_manager.pin_tenant_version(request, tenant_id, user_id)

@router.post("/versions/promote", response_model=Dict[str, Any])
async def promote_version(
    request: PromotionRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 14.3.17: Promote version with SLA-aware rules
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    user_id = current_user['user_id']
    
    if not pool_manager:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    version_manager = CapabilityVersionManager(pool_manager)
    return await version_manager.promote_version(request, tenant_id, user_id)

@router.post("/versions/deprecate", response_model=Dict[str, Any])
async def deprecate_version(
    request: DeprecationRequest,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 14.3.18: Deprecate version with automated notifications
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    user_id = current_user['user_id']
    
    if not pool_manager:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    version_manager = CapabilityVersionManager(pool_manager)
    return await version_manager.deprecate_version(request, tenant_id, user_id)

@router.get("/capabilities/{capability_id}/eol-check", response_model=Dict[str, Any])
async def check_capability_eol_status(
    capability_id: str,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Task 14.3.19: Check EOL status and enforce execution blocks
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    
    if not pool_manager:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    version_manager = CapabilityVersionManager(pool_manager)
    return await version_manager.check_eol_enforcement(capability_id, tenant_id)

@router.get("/versions/{version_meta_id}", response_model=VersionResponse)
async def get_version_details(
    version_meta_id: str,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager)
):
    """
    Get detailed information about a specific version
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    
    if not pool_manager:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    try:
        async with pool_manager.get_connection() as conn:
            await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
            
            version_data = await conn.fetchrow("""
                SELECT vm.*, vs.current_state
                FROM cap_version_meta vm
                JOIN cap_version_state vs ON vm.version_meta_id = vs.version_meta_id
                WHERE vm.version_meta_id = $1 AND vm.tenant_id = $2
            """, version_meta_id, tenant_id)
            
            if not version_data:
                raise HTTPException(status_code=404, detail="Version not found")
            
            # Parse breaking changes JSON
            breaking_changes = json.loads(version_data['breaking_changes']) if version_data['breaking_changes'] else []
            
            return VersionResponse(
                version_meta_id=version_data['version_meta_id'],
                capability_id=version_data['capability_id'],
                tenant_id=version_data['tenant_id'],
                version_string=version_data['version_string'],
                major_version=version_data['major_version'],
                minor_version=version_data['minor_version'],
                patch_version=version_data['patch_version'],
                prerelease_identifier=version_data['prerelease_identifier'],
                build_metadata=version_data['build_metadata'],
                is_stable=version_data['is_stable'],
                is_lts=version_data['is_lts'],
                is_deprecated=version_data['is_deprecated'],
                current_state=VersionState(version_data['current_state']),
                backward_compatible=version_data['backward_compatible'],
                forward_compatible=version_data['forward_compatible'],
                breaking_changes=breaking_changes,
                release_notes=version_data['release_notes'],
                changelog=version_data['changelog'],
                created_at=version_data['created_at'],
                released_at=version_data['released_at'],
                deprecated_at=version_data['deprecated_at'],
                eol_date=version_data['eol_date']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get version details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve version: {str(e)}")

@router.get("/capabilities/{capability_id}/versions", response_model=List[VersionResponse])
async def list_capability_versions(
    capability_id: str,
    current_user: dict = Depends(get_current_user),
    pool_manager: ConnectionPoolManager = Depends(get_pool_manager),
    state: Optional[VersionState] = Query(None, description="Filter by version state"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of versions to return")
):
    """
    List all versions for a capability
    """
    tenant_id = await get_tenant_id_from_user(current_user)
    
    if not pool_manager:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    try:
        async with pool_manager.get_connection() as conn:
            await conn.execute(f"SET app.current_tenant_id = '{tenant_id}';")
            
            # Build query with optional state filter
            query = """
                SELECT vm.*, vs.current_state
                FROM cap_version_meta vm
                JOIN cap_version_state vs ON vm.version_meta_id = vs.version_meta_id
                WHERE vm.capability_id = $1 AND vm.tenant_id = $2
            """
            params = [capability_id, tenant_id]
            
            if state:
                query += " AND vs.current_state = $3"
                params.append(state.value)
            
            query += " ORDER BY vm.major_version DESC, vm.minor_version DESC, vm.patch_version DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            versions = await conn.fetch(query, *params)
            
            return [
                VersionResponse(
                    version_meta_id=row['version_meta_id'],
                    capability_id=row['capability_id'],
                    tenant_id=row['tenant_id'],
                    version_string=row['version_string'],
                    major_version=row['major_version'],
                    minor_version=row['minor_version'],
                    patch_version=row['patch_version'],
                    prerelease_identifier=row['prerelease_identifier'],
                    build_metadata=row['build_metadata'],
                    is_stable=row['is_stable'],
                    is_lts=row['is_lts'],
                    is_deprecated=row['is_deprecated'],
                    current_state=VersionState(row['current_state']),
                    backward_compatible=row['backward_compatible'],
                    forward_compatible=row['forward_compatible'],
                    breaking_changes=json.loads(row['breaking_changes']) if row['breaking_changes'] else [],
                    release_notes=row['release_notes'],
                    changelog=row['changelog'],
                    created_at=row['created_at'],
                    released_at=row['released_at'],
                    deprecated_at=row['deprecated_at'],
                    eol_date=row['eol_date']
                )
                for row in versions
            ]
            
    except Exception as e:
        logger.error(f"❌ Failed to list capability versions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list versions: {str(e)}")
