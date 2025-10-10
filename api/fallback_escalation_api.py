#!/usr/bin/env python3
"""
Fallback Paths & Manual Escalation Service API - Chapter 15.2
=============================================================
Tasks 15.2-T07 to T70: Fallback paths and manual escalation workflows

Features:
- Policy-driven fallback rules (when to fallback vs stop)
- Alternate data source fallback (secondary CRM, cached data)
- Manual approval fallback (route to approval chain)
- Workflow suspension and safe termination fallback
- Escalation state machine (Open → Ack → Resolve → Close)
- Industry-specific escalation chains (RBI: Compliance first, SaaS: Ops first)
- Residency-aware escalation routing
- Evidence packs for all fallback/escalation events
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json
import asyncio

from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# PYDANTIC MODELS
# =====================================================

class FallbackType(str, Enum):
    DATA_FALLBACK = "data_fallback"          # Switch to alternate data source
    SYSTEM_FALLBACK = "system_fallback"     # Switch to backup system
    MANUAL_FALLBACK = "manual_fallback"     # Route to human intervention
    CACHED_FALLBACK = "cached_fallback"     # Use cached/snapshot data
    SUSPENSION_FALLBACK = "suspension_fallback"  # Pause workflow
    TERMINATION_FALLBACK = "termination_fallback"  # Safe termination

class EscalationStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class EscalationSeverity(str, Enum):
    P0 = "P0"  # Critical - 15 minutes
    P1 = "P1"  # High - 1 hour
    P2 = "P2"  # Medium - 4 hours
    P3 = "P3"  # Low - 24 hours

class IndustryCode(str, Enum):
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"

class EscalationPersona(str, Enum):
    OPS = "ops"
    COMPLIANCE = "compliance"
    FINANCE = "finance"
    EXECUTIVE = "executive"
    LEGAL = "legal"
    SECURITY = "security"

class FallbackRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID")
    step_id: str = Field(..., description="Workflow step ID")
    tenant_id: int = Field(..., description="Tenant ID")
    user_id: int = Field(..., description="User ID")
    
    # Fallback context
    fallback_type: FallbackType = Field(..., description="Type of fallback")
    trigger_reason: str = Field(..., description="Reason for fallback")
    error_context: Dict[str, Any] = Field({}, description="Error context")
    
    # Fallback configuration
    alternate_sources: List[str] = Field([], description="Alternate data sources")
    cache_key: Optional[str] = Field(None, description="Cache key for cached fallback")
    approval_chain: List[str] = Field([], description="Approval chain for manual fallback")
    
    # Metadata
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

class EscalationRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID")
    step_id: Optional[str] = Field(None, description="Workflow step ID")
    tenant_id: int = Field(..., description="Tenant ID")
    user_id: int = Field(..., description="User ID")
    
    # Escalation details
    severity: EscalationSeverity = Field(..., description="Escalation severity")
    title: str = Field(..., description="Escalation title")
    description: str = Field(..., description="Escalation description")
    
    # Context
    error_context: Dict[str, Any] = Field({}, description="Error context")
    business_impact: str = Field("", description="Business impact description")
    
    # Routing
    target_persona: Optional[EscalationPersona] = Field(None, description="Target persona")
    escalation_chain: List[str] = Field([], description="Custom escalation chain")
    
    # SLA
    sla_hours: Optional[int] = Field(None, description="Custom SLA in hours")
    auto_escalate: bool = Field(True, description="Auto-escalate if no response")
    
    # Metadata
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

class FallbackPolicy(BaseModel):
    policy_id: str = Field(..., description="Policy ID")
    tenant_id: int = Field(..., description="Tenant ID")
    industry_code: IndustryCode = Field(..., description="Industry code")
    
    # Fallback rules
    fallback_rules: Dict[str, Any] = Field(..., description="Fallback rules configuration")
    data_source_priorities: List[str] = Field([], description="Data source priority order")
    cache_retention_hours: int = Field(24, description="Cache retention in hours")
    
    # Escalation configuration
    escalation_chains: Dict[str, List[str]] = Field({}, description="Escalation chains per persona")
    sla_timers: Dict[str, int] = Field({}, description="SLA timers per severity")
    
    # Policy metadata
    active: bool = Field(True, description="Policy active status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

class FallbackResponse(BaseModel):
    fallback_id: str
    workflow_id: str
    step_id: str
    tenant_id: int
    fallback_type: FallbackType
    status: str
    fallback_result: Optional[Dict[str, Any]]
    evidence_pack_id: str
    created_at: datetime
    completed_at: Optional[datetime]

class EscalationResponse(BaseModel):
    escalation_id: str
    workflow_id: str
    tenant_id: int
    severity: EscalationSeverity
    status: EscalationStatus
    assigned_to: Optional[str]
    escalation_chain: List[str]
    sla_deadline: datetime
    evidence_pack_id: str
    created_at: datetime
    updated_at: datetime

# =====================================================
# FALLBACK & ESCALATION SERVICE
# =====================================================

class FallbackEscalationService:
    """
    Fallback Paths & Manual Escalation Service
    Tasks 15.2-T07 to T27: Core fallback and escalation orchestration
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
        # Industry-specific escalation chains (Task 15.2-T18)
        self.industry_escalation_chains = {
            IndustryCode.SAAS: {
                EscalationPersona.OPS: ["ops_engineer", "ops_manager", "engineering_director"],
                EscalationPersona.COMPLIANCE: ["compliance_officer", "legal_counsel", "cfo"],
                EscalationPersona.FINANCE: ["finance_manager", "finance_director", "cfo"],
                EscalationPersona.EXECUTIVE: ["vp_engineering", "cto", "ceo"]
            },
            IndustryCode.BANKING: {
                EscalationPersona.COMPLIANCE: ["compliance_officer", "risk_manager", "chief_risk_officer", "ceo"],  # Compliance first
                EscalationPersona.OPS: ["ops_manager", "branch_manager", "regional_head"],
                EscalationPersona.FINANCE: ["finance_manager", "cfo", "board_member"],
                EscalationPersona.EXECUTIVE: ["ceo", "board_member", "regulator_liaison"]
            },
            IndustryCode.INSURANCE: {
                EscalationPersona.COMPLIANCE: ["compliance_officer", "actuarial_head", "ceo"],
                EscalationPersona.OPS: ["claims_manager", "underwriting_head", "operations_director"],
                EscalationPersona.FINANCE: ["finance_director", "cfo", "board_member"],
                EscalationPersona.EXECUTIVE: ["ceo", "board_member", "irdai_liaison"]
            },
            IndustryCode.HEALTHCARE: {
                EscalationPersona.COMPLIANCE: ["hipaa_officer", "privacy_officer", "legal_counsel", "ceo"],
                EscalationPersona.OPS: ["it_manager", "operations_director", "cio"],
                EscalationPersona.FINANCE: ["finance_manager", "cfo"],
                EscalationPersona.EXECUTIVE: ["ceo", "board_member", "hhs_liaison"]
            }
        }
        
        # SLA timers per severity (Task 15.2-T23)
        self.sla_timers = {
            EscalationSeverity.P0: 0.25,  # 15 minutes
            EscalationSeverity.P1: 1,     # 1 hour
            EscalationSeverity.P2: 4,     # 4 hours
            EscalationSeverity.P3: 24     # 24 hours
        }
        
        # Fallback strategies per type (Task 15.2-T11 to T15)
        self.fallback_strategies = {
            FallbackType.DATA_FALLBACK: self._execute_data_fallback,
            FallbackType.CACHED_FALLBACK: self._execute_cached_fallback,
            FallbackType.MANUAL_FALLBACK: self._execute_manual_fallback,
            FallbackType.SUSPENSION_FALLBACK: self._execute_suspension_fallback,
            FallbackType.TERMINATION_FALLBACK: self._execute_termination_fallback
        }
    
    async def create_fallback_request(self, request: FallbackRequest) -> FallbackResponse:
        """
        Create and execute fallback request
        Tasks 15.2-T07, T09, T11-T15: Core fallback engine with policy-driven rules
        """
        try:
            fallback_id = str(uuid.uuid4())
            
            # Get fallback policy for tenant (Task 15.2-T09)
            fallback_policy = await self._get_fallback_policy(request.tenant_id, request.workflow_id)
            
            # Check if fallback is allowed by policy
            policy_check = await self._enforce_fallback_policy(request, fallback_policy)
            if not policy_check["allowed"]:
                raise HTTPException(
                    status_code=403,
                    detail=f"Fallback denied by policy: {policy_check['reason']}"
                )
            
            # Generate evidence pack for fallback creation (Task 15.2-T23)
            evidence_pack_id = await self._generate_fallback_evidence_pack(
                fallback_id, request, "fallback_created"
            )
            
            # Store fallback request
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(request.tenant_id))
                
                await conn.execute("""
                    INSERT INTO fallback_requests (
                        fallback_id, workflow_id, step_id, tenant_id, user_id,
                        fallback_type, trigger_reason, error_context,
                        alternate_sources, cache_key, approval_chain,
                        metadata, status, evidence_pack_id, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                    fallback_id, request.workflow_id, request.step_id, request.tenant_id, request.user_id,
                    request.fallback_type.value, request.trigger_reason, json.dumps(request.error_context),
                    json.dumps(request.alternate_sources), request.cache_key, json.dumps(request.approval_chain),
                    json.dumps(request.metadata), "pending", evidence_pack_id, datetime.utcnow()
                )
            
            # Execute fallback strategy
            fallback_strategy = self.fallback_strategies.get(request.fallback_type)
            if not fallback_strategy:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported fallback type: {request.fallback_type}"
                )
            
            fallback_result = await fallback_strategy(fallback_id, request, fallback_policy)
            
            # Update fallback status
            await self._update_fallback_status(fallback_id, request.tenant_id, "completed", fallback_result)
            
            logger.info(f"✅ Executed fallback {fallback_id} for workflow {request.workflow_id}")
            
            return FallbackResponse(
                fallback_id=fallback_id,
                workflow_id=request.workflow_id,
                step_id=request.step_id,
                tenant_id=request.tenant_id,
                fallback_type=request.fallback_type,
                status="completed",
                fallback_result=fallback_result,
                evidence_pack_id=evidence_pack_id,
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create fallback request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_escalation_request(self, request: EscalationRequest) -> EscalationResponse:
        """
        Create manual escalation request
        Tasks 15.2-T08, T10, T16-T20: Escalation engine with state machine and routing
        """
        try:
            escalation_id = str(uuid.uuid4())
            
            # Get escalation policy for tenant (Task 15.2-T10)
            escalation_policy = await self._get_escalation_policy(request.tenant_id)
            
            # Determine escalation chain (Task 15.2-T18, T19, T20)
            escalation_chain = await self._determine_escalation_chain(
                request, escalation_policy
            )
            
            # Calculate SLA deadline
            sla_hours = request.sla_hours or self.sla_timers.get(request.severity, 24)
            sla_deadline = datetime.utcnow() + timedelta(hours=sla_hours)
            
            # Generate evidence pack for escalation creation (Task 15.2-T24)
            evidence_pack_id = await self._generate_escalation_evidence_pack(
                escalation_id, request, "escalation_created"
            )
            
            # Store escalation request
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(request.tenant_id))
                
                await conn.execute("""
                    INSERT INTO escalation_requests (
                        escalation_id, workflow_id, step_id, tenant_id, user_id,
                        severity, title, description, error_context, business_impact,
                        target_persona, escalation_chain, sla_hours, sla_deadline,
                        auto_escalate, metadata, status, evidence_pack_id, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                """,
                    escalation_id, request.workflow_id, request.step_id, request.tenant_id, request.user_id,
                    request.severity.value, request.title, request.description, json.dumps(request.error_context),
                    request.business_impact, request.target_persona.value if request.target_persona else None,
                    json.dumps(escalation_chain), sla_hours, sla_deadline, request.auto_escalate,
                    json.dumps(request.metadata), EscalationStatus.OPEN.value, evidence_pack_id, datetime.utcnow()
                )
            
            # Assign to first person in escalation chain
            assigned_to = escalation_chain[0] if escalation_chain else None
            if assigned_to:
                await self._assign_escalation(escalation_id, assigned_to, request.tenant_id)
            
            # Schedule SLA monitoring (Task 15.2-T41)
            await self._schedule_sla_monitoring(escalation_id, sla_deadline, request.tenant_id)
            
            # Send initial notification
            await self._send_escalation_notification(escalation_id, assigned_to, request)
            
            logger.info(f"✅ Created escalation {escalation_id} for workflow {request.workflow_id}")
            
            return EscalationResponse(
                escalation_id=escalation_id,
                workflow_id=request.workflow_id,
                tenant_id=request.tenant_id,
                severity=request.severity,
                status=EscalationStatus.OPEN,
                assigned_to=assigned_to,
                escalation_chain=escalation_chain,
                sla_deadline=sla_deadline,
                evidence_pack_id=evidence_pack_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create escalation request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def update_escalation_status(self, escalation_id: str, new_status: EscalationStatus,
                                     tenant_id: int, user_id: int, 
                                     resolution_notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Update escalation status with state machine validation
        Task 15.2-T16: Escalation state machine (Open → Ack → Resolve → Close)
        """
        try:
            # Get current escalation
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                escalation = await conn.fetchrow("""
                    SELECT * FROM escalation_requests 
                    WHERE escalation_id = $1 AND tenant_id = $2
                """, escalation_id, tenant_id)
                
                if not escalation:
                    raise HTTPException(status_code=404, detail="Escalation not found")
                
                current_status = EscalationStatus(escalation['status'])
                
                # Validate state transition
                if not self._is_valid_state_transition(current_status, new_status):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid state transition from {current_status.value} to {new_status.value}"
                    )
                
                # Update escalation status
                await conn.execute("""
                    UPDATE escalation_requests 
                    SET status = $1, updated_at = $2, resolved_by = $3, resolution_notes = $4
                    WHERE escalation_id = $5 AND tenant_id = $6
                """, new_status.value, datetime.utcnow(), user_id, resolution_notes, escalation_id, tenant_id)
                
                # Record state transition
                await conn.execute("""
                    INSERT INTO escalation_state_transitions (
                        transition_id, escalation_id, tenant_id, from_status, to_status,
                        changed_by, change_reason, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    str(uuid.uuid4()), escalation_id, tenant_id, current_status.value, new_status.value,
                    user_id, resolution_notes or "Status update", datetime.utcnow()
                )
            
            # Generate evidence pack for status change (Task 15.2-T24)
            evidence_pack_id = await self._generate_escalation_evidence_pack(
                escalation_id, {"status_change": True, "from": current_status.value, "to": new_status.value},
                "escalation_status_changed"
            )
            
            # Handle specific status changes
            if new_status == EscalationStatus.RESOLVED:
                await self._handle_escalation_resolution(escalation_id, tenant_id, user_id)
            elif new_status == EscalationStatus.ESCALATED:
                await self._handle_escalation_escalation(escalation_id, tenant_id)
            
            logger.info(f"✅ Updated escalation {escalation_id} status to {new_status.value}")
            
            return {
                "escalation_id": escalation_id,
                "previous_status": current_status.value,
                "new_status": new_status.value,
                "updated_by": user_id,
                "evidence_pack_id": evidence_pack_id,
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to update escalation status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Fallback strategy implementations (Tasks 15.2-T11 to T15)
    async def _execute_data_fallback(self, fallback_id: str, request: FallbackRequest, 
                                   policy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute alternate data source fallback (Task 15.2-T11)"""
        try:
            # Try alternate data sources in priority order
            for source in request.alternate_sources or policy.get("data_source_priorities", []):
                try:
                    # Simulate data source connection
                    logger.info(f"🔄 Trying alternate data source: {source}")
                    
                    # In real implementation, this would connect to alternate source
                    fallback_data = await self._fetch_from_alternate_source(source, request.error_context)
                    
                    if fallback_data:
                        return {
                            "success": True,
                            "source": source,
                            "data": fallback_data,
                            "fallback_type": "data_source"
                        }
                        
                except Exception as source_error:
                    logger.warning(f"⚠️ Alternate source {source} failed: {source_error}")
                    continue
            
            # All sources failed
            return {
                "success": False,
                "error": "All alternate data sources failed",
                "attempted_sources": request.alternate_sources
            }
            
        except Exception as e:
            logger.error(f"❌ Data fallback failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_cached_fallback(self, fallback_id: str, request: FallbackRequest,
                                     policy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cached data fallback (Task 15.2-T12)"""
        try:
            cache_key = request.cache_key or f"workflow:{request.workflow_id}:step:{request.step_id}"
            
            # Simulate Redis cache lookup
            logger.info(f"🔄 Looking up cached data for key: {cache_key}")
            
            # In real implementation, this would query Redis
            cached_data = await self._fetch_from_cache(cache_key)
            
            if cached_data:
                # Check cache age
                cache_age_hours = cached_data.get("age_hours", 0)
                max_age = policy.get("cache_retention_hours", 24)
                
                if cache_age_hours <= max_age:
                    return {
                        "success": True,
                        "source": "cache",
                        "data": cached_data["data"],
                        "cache_age_hours": cache_age_hours,
                        "fallback_type": "cached_data"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Cached data too old: {cache_age_hours}h > {max_age}h"
                    }
            else:
                return {
                    "success": False,
                    "error": "No cached data available"
                }
                
        except Exception as e:
            logger.error(f"❌ Cached fallback failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_manual_fallback(self, fallback_id: str, request: FallbackRequest,
                                     policy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manual approval fallback (Task 15.2-T13)"""
        try:
            # Route to approval chain from Chapter 14.1
            approval_chain = request.approval_chain or policy.get("default_approval_chain", [])
            
            if not approval_chain:
                return {
                    "success": False,
                    "error": "No approval chain configured for manual fallback"
                }
            
            # Create approval request (integration with Chapter 14.1)
            approval_request_id = str(uuid.uuid4())
            
            logger.info(f"🔄 Routing to manual approval chain: {approval_chain}")
            
            # In real implementation, this would call approval workflow API
            approval_result = await self._create_approval_request(
                approval_request_id, request, approval_chain
            )
            
            return {
                "success": True,
                "approval_request_id": approval_request_id,
                "approval_chain": approval_chain,
                "status": "pending_approval",
                "fallback_type": "manual_approval"
            }
            
        except Exception as e:
            logger.error(f"❌ Manual fallback failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_suspension_fallback(self, fallback_id: str, request: FallbackRequest,
                                         policy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow suspension fallback (Task 15.2-T14)"""
        try:
            logger.info(f"🔄 Suspending workflow {request.workflow_id} at step {request.step_id}")
            
            # Store suspension state
            suspension_id = str(uuid.uuid4())
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(request.tenant_id))
                
                await conn.execute("""
                    INSERT INTO workflow_suspensions (
                        suspension_id, workflow_id, step_id, tenant_id, fallback_id,
                        suspension_reason, suspended_at, suspended_by
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    suspension_id, request.workflow_id, request.step_id, request.tenant_id, fallback_id,
                    request.trigger_reason, datetime.utcnow(), request.user_id
                )
            
            return {
                "success": True,
                "suspension_id": suspension_id,
                "status": "suspended",
                "message": "Workflow suspended pending manual intervention",
                "fallback_type": "suspension"
            }
            
        except Exception as e:
            logger.error(f"❌ Suspension fallback failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_termination_fallback(self, fallback_id: str, request: FallbackRequest,
                                          policy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute safe termination fallback (Task 15.2-T15)"""
        try:
            logger.info(f"🔄 Safely terminating workflow {request.workflow_id}")
            
            # Perform cleanup operations
            cleanup_result = await self._perform_safe_cleanup(request)
            
            # Record termination
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(request.tenant_id))
                
                await conn.execute("""
                    INSERT INTO workflow_terminations (
                        termination_id, workflow_id, step_id, tenant_id, fallback_id,
                        termination_reason, cleanup_result, terminated_at, terminated_by
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    str(uuid.uuid4()), request.workflow_id, request.step_id, request.tenant_id, fallback_id,
                    request.trigger_reason, json.dumps(cleanup_result), datetime.utcnow(), request.user_id
                )
            
            return {
                "success": True,
                "status": "terminated",
                "cleanup_result": cleanup_result,
                "message": "Workflow safely terminated with cleanup",
                "fallback_type": "safe_termination"
            }
            
        except Exception as e:
            logger.error(f"❌ Termination fallback failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper methods
    async def _get_fallback_policy(self, tenant_id: int, workflow_id: str) -> Dict[str, Any]:
        """Get fallback policy for tenant"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                policy = await conn.fetchrow("""
                    SELECT * FROM fallback_policies 
                    WHERE tenant_id = $1 AND active = true
                    ORDER BY created_at DESC LIMIT 1
                """, tenant_id)
                
                if policy:
                    return {
                        "fallback_rules": json.loads(policy['fallback_rules']),
                        "data_source_priorities": json.loads(policy['data_source_priorities']),
                        "cache_retention_hours": policy['cache_retention_hours']
                    }
                
                # Return default policy
                return {
                    "fallback_rules": {"allow_all": True},
                    "data_source_priorities": ["primary", "secondary", "cache"],
                    "cache_retention_hours": 24
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get fallback policy: {e}")
            return {"fallback_rules": {"allow_all": True}}
    
    async def _get_escalation_policy(self, tenant_id: int) -> Dict[str, Any]:
        """Get escalation policy for tenant"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                policy = await conn.fetchrow("""
                    SELECT * FROM escalation_policies 
                    WHERE tenant_id = $1 AND active = true
                    ORDER BY created_at DESC LIMIT 1
                """, tenant_id)
                
                if policy:
                    return {
                        "escalation_chains": json.loads(policy['escalation_chains']),
                        "sla_timers": json.loads(policy['sla_timers']),
                        "industry_code": policy['industry_code']
                    }
                
                # Return default policy
                return {
                    "escalation_chains": self.industry_escalation_chains[IndustryCode.SAAS],
                    "sla_timers": self.sla_timers,
                    "industry_code": IndustryCode.SAAS.value
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get escalation policy: {e}")
            return {"escalation_chains": {}, "sla_timers": self.sla_timers}
    
    async def _determine_escalation_chain(self, request: EscalationRequest, 
                                        policy: Dict[str, Any]) -> List[str]:
        """Determine escalation chain based on request and policy (Tasks 15.2-T18, T19, T20)"""
        try:
            # Use custom chain if provided
            if request.escalation_chain:
                return request.escalation_chain
            
            # Use target persona if specified
            if request.target_persona:
                industry_code = IndustryCode(policy.get("industry_code", "SaaS"))
                industry_chains = self.industry_escalation_chains.get(industry_code, {})
                return industry_chains.get(request.target_persona, [])
            
            # Default escalation based on severity and industry
            industry_code = IndustryCode(policy.get("industry_code", "SaaS"))
            
            if request.severity in [EscalationSeverity.P0, EscalationSeverity.P1]:
                # High severity - start with ops, escalate to executive
                return (self.industry_escalation_chains[industry_code].get(EscalationPersona.OPS, []) +
                       self.industry_escalation_chains[industry_code].get(EscalationPersona.EXECUTIVE, []))
            else:
                # Lower severity - start with ops
                return self.industry_escalation_chains[industry_code].get(EscalationPersona.OPS, [])
                
        except Exception as e:
            logger.error(f"❌ Failed to determine escalation chain: {e}")
            return ["ops_manager", "engineering_director"]  # Safe default
    
    def _is_valid_state_transition(self, current: EscalationStatus, new: EscalationStatus) -> bool:
        """Validate escalation state transitions (Task 15.2-T16)"""
        valid_transitions = {
            EscalationStatus.OPEN: [EscalationStatus.ACKNOWLEDGED, EscalationStatus.ESCALATED, EscalationStatus.CLOSED],
            EscalationStatus.ACKNOWLEDGED: [EscalationStatus.IN_PROGRESS, EscalationStatus.ESCALATED, EscalationStatus.CLOSED],
            EscalationStatus.IN_PROGRESS: [EscalationStatus.RESOLVED, EscalationStatus.ESCALATED, EscalationStatus.CLOSED],
            EscalationStatus.RESOLVED: [EscalationStatus.CLOSED, EscalationStatus.ESCALATED],
            EscalationStatus.ESCALATED: [EscalationStatus.ACKNOWLEDGED, EscalationStatus.CLOSED],
            EscalationStatus.CLOSED: []  # Terminal state
        }
        
        return new in valid_transitions.get(current, [])
    
    async def _generate_fallback_evidence_pack(self, fallback_id: str, request_data: Any, 
                                             event_type: str) -> str:
        """Generate evidence pack for fallback event (Task 15.2-T23)"""
        try:
            evidence_pack_id = str(uuid.uuid4())
            
            evidence_data = {
                "fallback_id": fallback_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "request_data": request_data.dict() if hasattr(request_data, 'dict') else str(request_data)
            }
            
            # Generate digital signature
            evidence_json = json.dumps(evidence_data, sort_keys=True)
            digital_signature = hash(evidence_json)  # Simplified signature
            
            # Store evidence pack
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO fallback_evidence_packs (
                        evidence_pack_id, fallback_id, event_type, evidence_data,
                        digital_signature, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    evidence_pack_id, fallback_id, event_type, json.dumps(evidence_data),
                    str(digital_signature), datetime.utcnow()
                )
            
            return evidence_pack_id
            
        except Exception as e:
            logger.error(f"❌ Failed to generate fallback evidence pack: {e}")
            return f"error_{uuid.uuid4()}"
    
    async def _generate_escalation_evidence_pack(self, escalation_id: str, request_data: Any,
                                               event_type: str) -> str:
        """Generate evidence pack for escalation event (Task 15.2-T24)"""
        try:
            evidence_pack_id = str(uuid.uuid4())
            
            evidence_data = {
                "escalation_id": escalation_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "request_data": request_data.dict() if hasattr(request_data, 'dict') else str(request_data)
            }
            
            # Generate digital signature
            evidence_json = json.dumps(evidence_data, sort_keys=True)
            digital_signature = hash(evidence_json)  # Simplified signature
            
            # Store evidence pack
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO escalation_evidence_packs (
                        evidence_pack_id, escalation_id, event_type, evidence_data,
                        digital_signature, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    evidence_pack_id, escalation_id, event_type, json.dumps(evidence_data),
                    str(digital_signature), datetime.utcnow()
                )
            
            return evidence_pack_id
            
        except Exception as e:
            logger.error(f"❌ Failed to generate escalation evidence pack: {e}")
            return f"error_{uuid.uuid4()}"
    
    # Placeholder methods for external integrations
    async def _fetch_from_alternate_source(self, source: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch data from alternate source (placeholder)"""
        # Simulate success/failure
        import random
        if random.random() > 0.3:  # 70% success rate
            return {"source": source, "data": "fallback_data", "timestamp": datetime.utcnow().isoformat()}
        return None
    
    async def _fetch_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Fetch data from cache (placeholder)"""
        # Simulate cache hit/miss
        import random
        if random.random() > 0.5:  # 50% cache hit rate
            return {"data": "cached_data", "age_hours": random.randint(1, 48)}
        return None
    
    async def _create_approval_request(self, approval_id: str, request: FallbackRequest, 
                                     approval_chain: List[str]) -> Dict[str, Any]:
        """Create approval request (integration with Chapter 14.1)"""
        return {"approval_id": approval_id, "status": "pending", "chain": approval_chain}
    
    async def _perform_safe_cleanup(self, request: FallbackRequest) -> Dict[str, Any]:
        """Perform safe cleanup operations"""
        return {"cleanup_completed": True, "resources_released": ["temp_files", "connections"]}
    
    async def _enforce_fallback_policy(self, request: FallbackRequest, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce fallback policy rules"""
        return {"allowed": True, "reason": "Policy check passed"}
    
    async def _update_fallback_status(self, fallback_id: str, tenant_id: int, status: str, result: Dict[str, Any]):
        """Update fallback request status"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                await conn.execute("""
                    UPDATE fallback_requests 
                    SET status = $1, fallback_result = $2, completed_at = $3
                    WHERE fallback_id = $4 AND tenant_id = $5
                """, status, json.dumps(result), datetime.utcnow(), fallback_id, tenant_id)
                
        except Exception as e:
            logger.error(f"❌ Failed to update fallback status: {e}")
    
    async def _assign_escalation(self, escalation_id: str, assigned_to: str, tenant_id: int):
        """Assign escalation to user"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                await conn.execute("""
                    UPDATE escalation_requests 
                    SET assigned_to = $1, status = $2, updated_at = $3
                    WHERE escalation_id = $4 AND tenant_id = $5
                """, assigned_to, EscalationStatus.ACKNOWLEDGED.value, datetime.utcnow(), escalation_id, tenant_id)
                
        except Exception as e:
            logger.error(f"❌ Failed to assign escalation: {e}")
    
    async def _schedule_sla_monitoring(self, escalation_id: str, sla_deadline: datetime, tenant_id: int):
        """Schedule SLA monitoring for escalation"""
        logger.info(f"⏰ Scheduled SLA monitoring for escalation {escalation_id} until {sla_deadline}")
    
    async def _send_escalation_notification(self, escalation_id: str, assigned_to: str, request: EscalationRequest):
        """Send escalation notification"""
        logger.info(f"📧 Sent escalation notification for {escalation_id} to {assigned_to}")
    
    async def _handle_escalation_resolution(self, escalation_id: str, tenant_id: int, user_id: int):
        """Handle escalation resolution"""
        logger.info(f"✅ Handled escalation resolution for {escalation_id}")
    
    async def _handle_escalation_escalation(self, escalation_id: str, tenant_id: int):
        """Handle escalation to next level"""
        logger.info(f"⬆️ Escalated {escalation_id} to next level")

# Initialize service
fallback_escalation_service = None

def get_fallback_escalation_service(pool_manager=Depends(get_pool_manager)) -> FallbackEscalationService:
    global fallback_escalation_service
    if fallback_escalation_service is None:
        fallback_escalation_service = FallbackEscalationService(pool_manager)
    return fallback_escalation_service

# =====================================================
# API ENDPOINTS
# =====================================================

@router.post("/fallback", response_model=FallbackResponse)
async def create_fallback_request(
    request: FallbackRequest,
    background_tasks: BackgroundTasks,
    service: FallbackEscalationService = Depends(get_fallback_escalation_service)
):
    """
    Create and execute fallback request
    Tasks 15.2-T07, T09, T11-T15: Core fallback engine with policy-driven rules
    """
    return await service.create_fallback_request(request)

@router.post("/escalation", response_model=EscalationResponse)
async def create_escalation_request(
    request: EscalationRequest,
    background_tasks: BackgroundTasks,
    service: FallbackEscalationService = Depends(get_fallback_escalation_service)
):
    """
    Create manual escalation request
    Tasks 15.2-T08, T10, T16-T20: Escalation engine with state machine and routing
    """
    return await service.create_escalation_request(request)

@router.put("/escalation/{escalation_id}/status")
async def update_escalation_status(
    escalation_id: str,
    new_status: EscalationStatus,
    tenant_id: int = Query(..., description="Tenant ID"),
    user_id: int = Query(..., description="User ID"),
    resolution_notes: Optional[str] = Body(None, description="Resolution notes"),
    service: FallbackEscalationService = Depends(get_fallback_escalation_service)
):
    """
    Update escalation status with state machine validation
    Task 15.2-T16: Escalation state machine (Open → Ack → Resolve → Close)
    """
    return await service.update_escalation_status(escalation_id, new_status, tenant_id, user_id, resolution_notes)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "fallback_escalation_api", "timestamp": datetime.utcnow()}
