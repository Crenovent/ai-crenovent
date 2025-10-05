#!/usr/bin/env python3
"""
Approval State Machine with Quorum Rules - Chapter 14.1
=======================================================
Tasks 14.1-T12, 14.1-T13: Approval quorum rules and time-bound approvals

Features:
- Multi-level approval chains with quorum requirements
- Time-bound approvals with auto-escalation
- Dynamic approver selection based on workload and availability
- SLA enforcement with configurable timeouts
- Integration with policy enforcement and override ledger
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

from pydantic import BaseModel, Field
from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)

class ApprovalState(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class ApprovalLevel(str, Enum):
    L1_MANAGER = "l1_manager"
    L2_SENIOR_MANAGER = "l2_senior_manager"
    L3_DIRECTOR = "l3_director"
    L4_VP = "l4_vp"
    L5_C_SUITE = "l5_c_suite"

class QuorumType(str, Enum):
    SIMPLE_MAJORITY = "simple_majority"  # >50%
    SUPER_MAJORITY = "super_majority"    # >=66%
    UNANIMOUS = "unanimous"              # 100%
    MINIMUM_COUNT = "minimum_count"      # Fixed number
    WEIGHTED = "weighted"                # Weighted voting

class ApprovalChainConfig(BaseModel):
    """Configuration for approval chain"""
    chain_id: str
    name: str
    description: str
    levels: List[Dict[str, Any]]  # Level configurations
    quorum_rules: Dict[str, Any]
    sla_hours: int = 48
    auto_escalate: bool = True
    escalation_levels: List[str] = Field(default_factory=list)
    industry_specific: bool = False
    compliance_frameworks: List[str] = Field(default_factory=list)

class ApprovalRequest(BaseModel):
    """Approval request with quorum requirements"""
    request_id: str
    workflow_id: str
    tenant_id: int
    requested_by_user_id: int
    approval_type: str
    business_justification: str
    risk_level: str = "medium"
    financial_amount: Optional[float] = None
    chain_config_id: str
    context_data: Dict[str, Any] = Field(default_factory=dict)
    required_evidence: List[str] = Field(default_factory=list)
    compliance_frameworks: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

class ApprovalDecision(BaseModel):
    """Individual approval decision"""
    decision_id: str
    request_id: str
    approver_user_id: int
    approver_role: str
    approval_level: ApprovalLevel
    decision: str  # 'approve', 'reject', 'delegate'
    comments: str = ""
    conditions: List[str] = Field(default_factory=list)
    evidence_provided: List[str] = Field(default_factory=list)
    decision_timestamp: datetime = Field(default_factory=datetime.utcnow)
    digital_signature: Optional[str] = None

class QuorumResult(BaseModel):
    """Result of quorum calculation"""
    quorum_met: bool
    total_required: int
    total_received: int
    approval_percentage: float
    rejection_percentage: float
    pending_approvers: List[Dict[str, Any]]
    next_level_required: bool = False
    escalation_needed: bool = False
    reason: str = ""

class ApprovalStateMachine:
    """
    Advanced Approval State Machine with Quorum Rules
    Tasks 14.1-T12, 14.1-T13: Quorum rules and time-bound approvals
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
        # Default approval chain configurations
        self.default_chains = {
            "standard_workflow": ApprovalChainConfig(
                chain_id="standard_workflow",
                name="Standard Workflow Approval",
                description="Standard approval chain for regular workflows",
                levels=[
                    {
                        "level": ApprovalLevel.L1_MANAGER.value,
                        "roles": ["sales_manager", "revops_manager"],
                        "quorum_type": QuorumType.MINIMUM_COUNT.value,
                        "quorum_value": 1,
                        "sla_hours": 24
                    },
                    {
                        "level": ApprovalLevel.L2_SENIOR_MANAGER.value,
                        "roles": ["compliance_officer", "finance_manager"],
                        "quorum_type": QuorumType.SIMPLE_MAJORITY.value,
                        "quorum_value": 0.5,
                        "sla_hours": 48
                    }
                ],
                quorum_rules={
                    "overall_type": QuorumType.SIMPLE_MAJORITY.value,
                    "level_progression": "sequential",  # or "parallel"
                    "allow_delegation": True,
                    "require_all_levels": False
                },
                sla_hours=72,
                auto_escalate=True,
                escalation_levels=["l3_director", "l4_vp"]
            ),
            
            "high_value_financial": ApprovalChainConfig(
                chain_id="high_value_financial",
                name="High Value Financial Approval",
                description="Approval chain for high-value financial transactions",
                levels=[
                    {
                        "level": ApprovalLevel.L2_SENIOR_MANAGER.value,
                        "roles": ["finance_manager", "accounting_manager"],
                        "quorum_type": QuorumType.MINIMUM_COUNT.value,
                        "quorum_value": 2,
                        "sla_hours": 12
                    },
                    {
                        "level": ApprovalLevel.L3_DIRECTOR.value,
                        "roles": ["finance_director", "compliance_director"],
                        "quorum_type": QuorumType.UNANIMOUS.value,
                        "quorum_value": 1.0,
                        "sla_hours": 24
                    },
                    {
                        "level": ApprovalLevel.L4_VP.value,
                        "roles": ["cfo", "ceo"],
                        "quorum_type": QuorumType.MINIMUM_COUNT.value,
                        "quorum_value": 1,
                        "sla_hours": 48
                    }
                ],
                quorum_rules={
                    "overall_type": QuorumType.UNANIMOUS.value,
                    "level_progression": "sequential",
                    "allow_delegation": False,
                    "require_all_levels": True
                },
                sla_hours=84,  # 3.5 days
                auto_escalate=True,
                escalation_levels=["l5_c_suite"],
                compliance_frameworks=["SOX", "RBI"]
            ),
            
            "regulatory_compliance": ApprovalChainConfig(
                chain_id="regulatory_compliance",
                name="Regulatory Compliance Approval",
                description="Approval chain for regulatory compliance matters",
                levels=[
                    {
                        "level": ApprovalLevel.L2_SENIOR_MANAGER.value,
                        "roles": ["compliance_officer", "risk_officer"],
                        "quorum_type": QuorumType.SUPER_MAJORITY.value,
                        "quorum_value": 0.67,
                        "sla_hours": 8
                    },
                    {
                        "level": ApprovalLevel.L3_DIRECTOR.value,
                        "roles": ["compliance_director", "legal_director"],
                        "quorum_type": QuorumType.UNANIMOUS.value,
                        "quorum_value": 1.0,
                        "sla_hours": 16
                    },
                    {
                        "level": ApprovalLevel.L5_C_SUITE.value,
                        "roles": ["ceo", "chief_compliance_officer"],
                        "quorum_type": QuorumType.MINIMUM_COUNT.value,
                        "quorum_value": 1,
                        "sla_hours": 24
                    }
                ],
                quorum_rules={
                    "overall_type": QuorumType.UNANIMOUS.value,
                    "level_progression": "sequential",
                    "allow_delegation": False,
                    "require_all_levels": True
                },
                sla_hours=48,
                auto_escalate=True,
                escalation_levels=[],
                compliance_frameworks=["GDPR", "HIPAA", "RBI", "IRDAI"]
            )
        }
        
        # SLA configurations by risk level
        self.sla_configs = {
            "low": {"hours": 72, "escalation_threshold": 0.8},      # 3 days
            "medium": {"hours": 48, "escalation_threshold": 0.75},  # 2 days
            "high": {"hours": 24, "escalation_threshold": 0.67},    # 1 day
            "critical": {"hours": 8, "escalation_threshold": 0.5}   # 8 hours
        }
    
    async def create_approval_request(self, request: ApprovalRequest) -> Dict[str, Any]:
        """
        Create a new approval request with quorum requirements
        Task 14.1-T12: Approval quorum rules
        """
        try:
            logger.info(f"🔄 Creating approval request {request.request_id}")
            
            # Get approval chain configuration
            chain_config = await self._get_chain_config(request.chain_config_id)
            if not chain_config:
                raise ValueError(f"Invalid chain config ID: {request.chain_config_id}")
            
            # Calculate SLA and expiry
            sla_config = self.sla_configs.get(request.risk_level, self.sla_configs["medium"])
            expires_at = request.created_at + timedelta(hours=sla_config["hours"])
            
            # Initialize approval state
            approval_state = {
                "request_id": request.request_id,
                "current_state": ApprovalState.PENDING.value,
                "current_level": 0,
                "chain_config": chain_config.dict(),
                "decisions": [],
                "quorum_status": {},
                "sla_breaches": [],
                "escalations": [],
                "created_at": request.created_at,
                "expires_at": expires_at,
                "last_updated": request.created_at
            }
            
            # Assign initial approvers
            initial_approvers = await self._assign_level_approvers(
                request.tenant_id, chain_config.levels[0], request.context_data
            )
            
            # Store in database
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO approval_requests (
                        request_id, workflow_id, tenant_id, requested_by_user_id,
                        approval_type, business_justification, risk_level,
                        financial_amount, chain_config_id, context_data,
                        required_evidence, compliance_frameworks,
                        approval_state, assigned_approvers, created_at, expires_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """, 
                request.request_id, request.workflow_id, request.tenant_id, 
                request.requested_by_user_id, request.approval_type,
                request.business_justification, request.risk_level,
                request.financial_amount, request.chain_config_id,
                json.dumps(request.context_data), request.required_evidence,
                request.compliance_frameworks, json.dumps(approval_state),
                json.dumps(initial_approvers), request.created_at, expires_at)
            
            # Schedule SLA monitoring
            await self._schedule_sla_monitoring(request.request_id, expires_at)
            
            logger.info(f"✅ Approval request created: {request.request_id}")
            return {
                "request_id": request.request_id,
                "state": ApprovalState.PENDING.value,
                "assigned_approvers": initial_approvers,
                "expires_at": expires_at,
                "sla_hours": sla_config["hours"]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create approval request: {e}")
            raise
    
    async def process_approval_decision(self, decision: ApprovalDecision) -> Dict[str, Any]:
        """
        Process an individual approval decision and update quorum status
        Task 14.1-T12: Approval quorum rules
        """
        try:
            logger.info(f"🔄 Processing approval decision {decision.decision_id}")
            
            # Get current approval request state
            approval_state = await self._get_approval_state(decision.request_id)
            if not approval_state:
                raise ValueError(f"Approval request not found: {decision.request_id}")
            
            # Validate approver authorization
            is_authorized = await self._validate_approver_authorization(
                decision.request_id, decision.approver_user_id, decision.approver_role
            )
            if not is_authorized:
                raise ValueError("Approver not authorized for this request")
            
            # Add decision to state
            approval_state["decisions"].append(decision.dict())
            approval_state["last_updated"] = datetime.utcnow()
            
            # Calculate quorum for current level
            chain_config = ApprovalChainConfig(**approval_state["chain_config"])
            current_level = approval_state["current_level"]
            
            quorum_result = await self._calculate_level_quorum(
                approval_state["decisions"], chain_config.levels[current_level]
            )
            
            approval_state["quorum_status"][str(current_level)] = quorum_result.dict()
            
            # Determine next state based on quorum result
            next_state = await self._determine_next_state(approval_state, quorum_result, chain_config)
            
            # Update state in database
            await self._update_approval_state(decision.request_id, approval_state, next_state)
            
            # Handle state transitions
            if next_state == ApprovalState.APPROVED:
                await self._handle_approval_completion(decision.request_id, approval_state)
            elif next_state == ApprovalState.REJECTED:
                await self._handle_approval_rejection(decision.request_id, approval_state)
            elif next_state == ApprovalState.ESCALATED:
                await self._handle_approval_escalation(decision.request_id, approval_state, chain_config)
            
            logger.info(f"✅ Approval decision processed: {next_state.value}")
            return {
                "decision_id": decision.decision_id,
                "request_id": decision.request_id,
                "new_state": next_state.value,
                "quorum_result": quorum_result.dict(),
                "next_level_required": quorum_result.next_level_required
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process approval decision: {e}")
            raise
    
    async def check_sla_compliance(self, request_id: str) -> Dict[str, Any]:
        """
        Check SLA compliance and trigger auto-escalation if needed
        Task 14.1-T13: Time-bound approvals with auto-escalation
        """
        try:
            approval_state = await self._get_approval_state(request_id)
            if not approval_state:
                return {"error": "Request not found"}
            
            current_time = datetime.utcnow()
            expires_at = approval_state["expires_at"]
            
            # Calculate time remaining
            time_remaining = expires_at - current_time
            total_duration = expires_at - approval_state["created_at"]
            elapsed_percentage = 1.0 - (time_remaining.total_seconds() / total_duration.total_seconds())
            
            # Check if SLA is breached
            sla_breached = current_time > expires_at
            
            # Check if escalation threshold is reached
            chain_config = ApprovalChainConfig(**approval_state["chain_config"])
            risk_level = await self._get_request_risk_level(request_id)
            escalation_threshold = self.sla_configs.get(risk_level, {}).get("escalation_threshold", 0.75)
            
            escalation_needed = elapsed_percentage >= escalation_threshold and not sla_breached
            
            result = {
                "request_id": request_id,
                "sla_breached": sla_breached,
                "escalation_needed": escalation_needed,
                "elapsed_percentage": elapsed_percentage,
                "time_remaining_hours": time_remaining.total_seconds() / 3600,
                "escalation_threshold": escalation_threshold
            }
            
            # Trigger auto-escalation if needed
            if escalation_needed and chain_config.auto_escalate:
                escalation_result = await self._trigger_auto_escalation(request_id, approval_state)
                result["escalation_triggered"] = True
                result["escalation_result"] = escalation_result
            
            # Mark as expired if SLA breached
            if sla_breached and approval_state["current_state"] not in [ApprovalState.APPROVED.value, ApprovalState.REJECTED.value]:
                await self._mark_request_expired(request_id, approval_state)
                result["marked_expired"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"❌ SLA compliance check failed: {e}")
            return {"error": str(e)}
    
    async def _get_chain_config(self, chain_config_id: str) -> Optional[ApprovalChainConfig]:
        """Get approval chain configuration"""
        # First check database for custom configurations
        try:
            async with self.pool_manager.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT config_data FROM approval_chain_configs 
                    WHERE chain_id = $1 AND active = true
                """, chain_config_id)
                
                if row:
                    return ApprovalChainConfig(**json.loads(row['config_data']))
        except Exception as e:
            logger.warning(f"⚠️ Failed to load custom chain config: {e}")
        
        # Fall back to default configurations
        return self.default_chains.get(chain_config_id)
    
    async def _assign_level_approvers(self, tenant_id: int, level_config: Dict[str, Any], 
                                    context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Assign approvers for a specific level with dynamic selection
        Task 14.1-T14: Dynamic approver selection
        """
        try:
            required_roles = level_config.get("roles", [])
            quorum_type = level_config.get("quorum_type")
            quorum_value = level_config.get("quorum_value", 1)
            
            assigned_approvers = []
            
            for role in required_roles:
                # Get available users for this role
                available_users = await self._get_available_approvers(tenant_id, role, context_data)
                
                if quorum_type == QuorumType.MINIMUM_COUNT.value:
                    # Assign specific number of approvers
                    selected_users = available_users[:int(quorum_value)]
                elif quorum_type in [QuorumType.SIMPLE_MAJORITY.value, QuorumType.SUPER_MAJORITY.value, QuorumType.UNANIMOUS.value]:
                    # Assign all available users for percentage-based quorum
                    selected_users = available_users
                else:
                    # Default: assign first available user
                    selected_users = available_users[:1]
                
                for user in selected_users:
                    assigned_approvers.append({
                        "user_id": user["user_id"],
                        "role": role,
                        "level": level_config.get("level"),
                        "assigned_at": datetime.utcnow().isoformat(),
                        "sla_hours": level_config.get("sla_hours", 24),
                        "workload_score": user.get("workload_score", 0)
                    })
            
            return assigned_approvers
            
        except Exception as e:
            logger.error(f"❌ Failed to assign level approvers: {e}")
            return []
    
    async def _get_available_approvers(self, tenant_id: int, role: str, 
                                     context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get available approvers for a role with workload balancing"""
        try:
            async with self.pool_manager.get_connection() as conn:
                # Get users with the required role, ordered by current workload
                rows = await conn.fetch("""
                    SELECT u.user_id, u.name, ur.role, 
                           COALESCE(w.pending_approvals, 0) as workload_score,
                           ur.region, ur.segment
                    FROM users u
                    JOIN users_role ur ON u.user_id = ur.user_id
                    LEFT JOIN (
                        SELECT approver_user_id, COUNT(*) as pending_approvals
                        FROM approval_requests ar
                        JOIN jsonb_array_elements(ar.assigned_approvers) approver ON true
                        WHERE (approver->>'user_id')::int = ur.user_id
                        AND ar.approval_state->>'current_state' = 'pending'
                        GROUP BY approver_user_id
                    ) w ON w.approver_user_id = u.user_id
                    WHERE u.tenant_id = $1 
                    AND ur.role = $2
                    AND u.status = 'active'
                    ORDER BY workload_score ASC, u.user_id ASC
                """, tenant_id, role)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get available approvers: {e}")
            return []
    
    async def _calculate_level_quorum(self, decisions: List[Dict], level_config: Dict[str, Any]) -> QuorumResult:
        """Calculate if quorum is met for a specific level"""
        try:
            level = level_config.get("level")
            quorum_type = level_config.get("quorum_type")
            quorum_value = level_config.get("quorum_value", 1)
            
            # Filter decisions for this level
            level_decisions = [d for d in decisions if d.get("approval_level") == level]
            
            # Count approvals and rejections
            approvals = [d for d in level_decisions if d.get("decision") == "approve"]
            rejections = [d for d in level_decisions if d.get("decision") == "reject"]
            
            total_decisions = len(level_decisions)
            approval_count = len(approvals)
            rejection_count = len(rejections)
            
            # Get total required approvers for this level
            total_required = await self._get_level_approver_count(level_config)
            
            # Calculate percentages
            approval_percentage = approval_count / total_required if total_required > 0 else 0
            rejection_percentage = rejection_count / total_required if total_required > 0 else 0
            
            # Determine if quorum is met
            quorum_met = False
            reason = ""
            
            if quorum_type == QuorumType.MINIMUM_COUNT.value:
                quorum_met = approval_count >= int(quorum_value)
                reason = f"Need {int(quorum_value)} approvals, got {approval_count}"
            elif quorum_type == QuorumType.SIMPLE_MAJORITY.value:
                quorum_met = approval_percentage > 0.5
                reason = f"Need >50% approval, got {approval_percentage:.1%}"
            elif quorum_type == QuorumType.SUPER_MAJORITY.value:
                quorum_met = approval_percentage >= 0.67
                reason = f"Need ≥67% approval, got {approval_percentage:.1%}"
            elif quorum_type == QuorumType.UNANIMOUS.value:
                quorum_met = approval_count == total_required and rejection_count == 0
                reason = f"Need 100% approval, got {approval_percentage:.1%}"
            
            # Check if rejection threshold is met (auto-reject)
            rejection_threshold_met = rejection_percentage > 0.5
            
            return QuorumResult(
                quorum_met=quorum_met and not rejection_threshold_met,
                total_required=total_required,
                total_received=total_decisions,
                approval_percentage=approval_percentage,
                rejection_percentage=rejection_percentage,
                pending_approvers=[],  # Will be populated by caller
                next_level_required=quorum_met and not rejection_threshold_met,
                escalation_needed=False,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"❌ Quorum calculation failed: {e}")
            return QuorumResult(
                quorum_met=False,
                total_required=0,
                total_received=0,
                approval_percentage=0.0,
                rejection_percentage=0.0,
                pending_approvers=[],
                reason=f"Calculation error: {str(e)}"
            )
    
    async def _get_level_approver_count(self, level_config: Dict[str, Any]) -> int:
        """Get total number of approvers required for a level"""
        roles = level_config.get("roles", [])
        quorum_type = level_config.get("quorum_type")
        quorum_value = level_config.get("quorum_value", 1)
        
        if quorum_type == QuorumType.MINIMUM_COUNT.value:
            return int(quorum_value)
        else:
            # For percentage-based quorum, return total available approvers
            return len(roles) * 2  # Assume 2 approvers per role on average
    
    async def _determine_next_state(self, approval_state: Dict, quorum_result: QuorumResult, 
                                  chain_config: ApprovalChainConfig) -> ApprovalState:
        """Determine the next state based on quorum result"""
        current_level = approval_state["current_level"]
        total_levels = len(chain_config.levels)
        
        # Check if current level is rejected
        if quorum_result.rejection_percentage > 0.5:
            return ApprovalState.REJECTED
        
        # Check if current level quorum is met
        if quorum_result.quorum_met:
            # Check if this is the final level
            if current_level >= total_levels - 1:
                return ApprovalState.APPROVED
            else:
                # Move to next level
                approval_state["current_level"] = current_level + 1
                return ApprovalState.IN_REVIEW
        
        # Check if escalation is needed due to SLA
        if quorum_result.escalation_needed:
            return ApprovalState.ESCALATED
        
        # Otherwise, stay pending
        return ApprovalState.PENDING
    
    async def _get_approval_state(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get current approval state from database"""
        try:
            async with self.pool_manager.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT approval_state, expires_at FROM approval_requests 
                    WHERE request_id = $1
                """, request_id)
                
                if row:
                    state = json.loads(row['approval_state'])
                    state["expires_at"] = row['expires_at']
                    return state
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get approval state: {e}")
            return None
    
    async def _update_approval_state(self, request_id: str, approval_state: Dict, new_state: ApprovalState):
        """Update approval state in database"""
        try:
            approval_state["current_state"] = new_state.value
            approval_state["last_updated"] = datetime.utcnow()
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    UPDATE approval_requests 
                    SET approval_state = $1, last_updated = $2
                    WHERE request_id = $3
                """, json.dumps(approval_state, default=str), datetime.utcnow(), request_id)
                
        except Exception as e:
            logger.error(f"❌ Failed to update approval state: {e}")
            raise
    
    async def _validate_approver_authorization(self, request_id: str, approver_user_id: int, 
                                             approver_role: str) -> bool:
        """Validate that the approver is authorized for this request"""
        try:
            async with self.pool_manager.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT assigned_approvers FROM approval_requests 
                    WHERE request_id = $1
                """, request_id)
                
                if not row:
                    return False
                
                assigned_approvers = json.loads(row['assigned_approvers'])
                
                # Check if user is in assigned approvers list
                for approver in assigned_approvers:
                    if (approver.get("user_id") == approver_user_id and 
                        approver.get("role") == approver_role):
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"❌ Approver authorization check failed: {e}")
            return False
    
    async def _handle_approval_completion(self, request_id: str, approval_state: Dict):
        """Handle completion of approval process"""
        logger.info(f"✅ Approval request {request_id} completed successfully")
        
        # Generate evidence pack
        await self._generate_approval_evidence(request_id, approval_state, "approved")
        
        # Update trust scoring
        await self._update_trust_score_for_approval(request_id, "approved")
        
        # Notify stakeholders
        await self._notify_approval_completion(request_id, "approved")
    
    async def _handle_approval_rejection(self, request_id: str, approval_state: Dict):
        """Handle rejection of approval process"""
        logger.info(f"❌ Approval request {request_id} rejected")
        
        # Generate evidence pack
        await self._generate_approval_evidence(request_id, approval_state, "rejected")
        
        # Update risk register
        await self._update_risk_register_for_rejection(request_id, approval_state)
        
        # Notify stakeholders
        await self._notify_approval_completion(request_id, "rejected")
    
    async def _handle_approval_escalation(self, request_id: str, approval_state: Dict, 
                                        chain_config: ApprovalChainConfig):
        """Handle escalation of approval process"""
        logger.info(f"⬆️ Approval request {request_id} escalated")
        
        # Add escalation record
        escalation_record = {
            "escalated_at": datetime.utcnow().isoformat(),
            "reason": "SLA breach or quorum not met",
            "escalated_to": chain_config.escalation_levels
        }
        
        approval_state["escalations"].append(escalation_record)
        
        # Assign escalation approvers
        if chain_config.escalation_levels:
            escalation_approvers = await self._assign_escalation_approvers(
                request_id, chain_config.escalation_levels[0]
            )
            approval_state["escalation_approvers"] = escalation_approvers
        
        # Update state
        await self._update_approval_state(request_id, approval_state, ApprovalState.ESCALATED)
    
    async def _trigger_auto_escalation(self, request_id: str, approval_state: Dict) -> Dict[str, Any]:
        """
        Trigger automatic escalation due to SLA breach
        Task 14.1-T13: Time-bound approvals with auto-escalation
        """
        try:
            chain_config = ApprovalChainConfig(**approval_state["chain_config"])
            
            if not chain_config.auto_escalate or not chain_config.escalation_levels:
                return {"escalated": False, "reason": "Auto-escalation not configured"}
            
            # Escalate to next level
            next_escalation_level = chain_config.escalation_levels[0]
            
            # Assign escalation approvers
            escalation_approvers = await self._assign_escalation_approvers(request_id, next_escalation_level)
            
            # Update approval state
            approval_state["current_state"] = ApprovalState.ESCALATED.value
            approval_state["escalation_approvers"] = escalation_approvers
            approval_state["escalations"].append({
                "escalated_at": datetime.utcnow().isoformat(),
                "reason": "Auto-escalation due to SLA breach",
                "escalated_to": next_escalation_level
            })
            
            await self._update_approval_state(request_id, approval_state, ApprovalState.ESCALATED)
            
            logger.info(f"⬆️ Auto-escalated approval request {request_id} to {next_escalation_level}")
            return {
                "escalated": True,
                "escalated_to": next_escalation_level,
                "escalation_approvers": escalation_approvers
            }
            
        except Exception as e:
            logger.error(f"❌ Auto-escalation failed: {e}")
            return {"escalated": False, "error": str(e)}
    
    async def _assign_escalation_approvers(self, request_id: str, escalation_level: str) -> List[Dict[str, Any]]:
        """Assign approvers for escalation level"""
        # This would typically assign C-suite or senior executives
        escalation_roles = {
            "l3_director": ["finance_director", "compliance_director"],
            "l4_vp": ["vp_sales", "vp_finance"],
            "l5_c_suite": ["ceo", "cfo", "coo"]
        }
        
        roles = escalation_roles.get(escalation_level, ["ceo"])
        
        # Get tenant_id from request
        async with self.pool_manager.get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT tenant_id FROM approval_requests WHERE request_id = $1
            """, request_id)
            
            if not row:
                return []
            
            tenant_id = row['tenant_id']
        
        # Assign escalation approvers
        escalation_approvers = []
        for role in roles:
            available_users = await self._get_available_approvers(tenant_id, role, {})
            if available_users:
                escalation_approvers.append({
                    "user_id": available_users[0]["user_id"],
                    "role": role,
                    "level": escalation_level,
                    "assigned_at": datetime.utcnow().isoformat(),
                    "escalation": True
                })
        
        return escalation_approvers
    
    async def _schedule_sla_monitoring(self, request_id: str, expires_at: datetime):
        """Schedule SLA monitoring for the request"""
        # This would typically integrate with a job scheduler like Celery or APScheduler
        logger.info(f"📅 Scheduled SLA monitoring for {request_id} until {expires_at}")
    
    async def _get_request_risk_level(self, request_id: str) -> str:
        """Get risk level for a request"""
        try:
            async with self.pool_manager.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT risk_level FROM approval_requests WHERE request_id = $1
                """, request_id)
                
                return row['risk_level'] if row else "medium"
                
        except Exception as e:
            logger.error(f"❌ Failed to get request risk level: {e}")
            return "medium"
    
    async def _mark_request_expired(self, request_id: str, approval_state: Dict):
        """Mark request as expired due to SLA breach"""
        approval_state["current_state"] = ApprovalState.EXPIRED.value
        approval_state["expired_at"] = datetime.utcnow().isoformat()
        
        await self._update_approval_state(request_id, approval_state, ApprovalState.EXPIRED)
        
        # Generate evidence pack for expiry
        await self._generate_approval_evidence(request_id, approval_state, "expired")
        
        logger.warning(f"⏰ Approval request {request_id} marked as expired")
    
    # Placeholder methods for integration points
    async def _generate_approval_evidence(self, request_id: str, approval_state: Dict, outcome: str):
        """Generate evidence pack for approval outcome"""
        logger.info(f"📋 Generating evidence pack for {request_id}: {outcome}")
    
    async def _update_trust_score_for_approval(self, request_id: str, outcome: str):
        """Update trust scoring based on approval outcome"""
        logger.info(f"📊 Updating trust score for {request_id}: {outcome}")
    
    async def _update_risk_register_for_rejection(self, request_id: str, approval_state: Dict):
        """Update risk register for rejected approval"""
        logger.info(f"⚠️ Updating risk register for rejected approval: {request_id}")
    
    async def _notify_approval_completion(self, request_id: str, outcome: str):
        """Notify stakeholders of approval completion"""
        logger.info(f"📧 Notifying stakeholders of approval {outcome}: {request_id}")

# Service factory
class ApprovalStateMachineService:
    """Service factory for approval state machine"""
    
    def __init__(self, pool_manager):
        self.state_machine = ApprovalStateMachine(pool_manager)
    
    async def create_approval_request(self, request: ApprovalRequest) -> Dict[str, Any]:
        """Create new approval request"""
        return await self.state_machine.create_approval_request(request)
    
    async def process_decision(self, decision: ApprovalDecision) -> Dict[str, Any]:
        """Process approval decision"""
        return await self.state_machine.process_approval_decision(decision)
    
    async def check_sla_compliance(self, request_id: str) -> Dict[str, Any]:
        """Check SLA compliance"""
        return await self.state_machine.check_sla_compliance(request_id)
    
    async def get_pending_approvals(self, user_id: int, tenant_id: int) -> List[Dict[str, Any]]:
        """Get pending approvals for a user"""
        try:
            async with self.state_machine.pool_manager.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT ar.request_id, ar.workflow_id, ar.approval_type,
                           ar.business_justification, ar.risk_level, ar.financial_amount,
                           ar.created_at, ar.expires_at, ar.approval_state
                    FROM approval_requests ar
                    WHERE ar.tenant_id = $1
                    AND ar.approval_state->>'current_state' IN ('pending', 'in_review')
                    AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements(ar.assigned_approvers) approver
                        WHERE (approver->>'user_id')::int = $2
                    )
                    ORDER BY ar.created_at ASC
                """, tenant_id, user_id)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get pending approvals: {e}")
            return []
