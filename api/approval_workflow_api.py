#!/usr/bin/env python3
"""
Approval Workflow Service API - Chapter 14.1
============================================
Tasks 14.1-T07, 14.1-T08: Approval workflow service and override ledger APIs

Features:
- Maker-checker approval workflows with multi-level chains
- Override ledger with reason codes and digital signatures
- Policy-driven approval routing and escalation
- Evidence pack integration and audit trails
- Industry-specific compliance framework support
- Real-time approval status tracking and notifications
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json

# Database and dependencies
from src.database.connection_pool import get_pool_manager
from dsl.governance.policy_engine import PolicyEngine, get_policy_engine
from dsl.intelligence.evidence_pack_generator import EvidencePackGenerator
from dsl.governance.rbac_sod_system import RBACManager, SoDEnforcer

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# PYDANTIC MODELS
# =====================================================

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved" 
    REJECTED = "rejected"
    EXPIRED = "expired"
    ESCALATED = "escalated"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OverrideStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    RESOLVED = "resolved"

class ApprovalRequest(BaseModel):
    workflow_id: str = Field(..., description="UUID of the workflow requiring approval")
    workflow_execution_id: Optional[str] = Field(None, description="Specific execution ID if applicable")
    request_type: str = Field(..., description="Type of approval request")
    request_reason: str = Field(..., description="Reason for the approval request")
    business_justification: str = Field(..., description="Business justification")
    risk_assessment: RiskLevel = Field(RiskLevel.MEDIUM, description="Risk level assessment")
    approval_chain: Optional[List[str]] = Field(None, description="Custom approval chain")
    compliance_frameworks: List[str] = Field(default=[], description="Applicable compliance frameworks")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class ApprovalDecision(BaseModel):
    approval_id: str = Field(..., description="UUID of the approval request")
    decision: ApprovalStatus = Field(..., description="Approval decision")
    comments: Optional[str] = Field(None, description="Approver comments")
    digital_signature: Optional[str] = Field(None, description="Digital signature")

class OverrideRequest(BaseModel):
    workflow_id: str = Field(..., description="UUID of the workflow")
    workflow_execution_id: Optional[str] = Field(None, description="Specific execution ID")
    override_type: str = Field(..., description="Type of override")
    reason_code: str = Field(..., description="Standardized reason code")
    reason_description: str = Field(..., description="Detailed reason description")
    business_impact_assessment: str = Field(..., description="Business impact assessment")
    risk_level: RiskLevel = Field(..., description="Risk level of the override")
    estimated_business_impact: Optional[float] = Field(None, description="Estimated financial impact")
    override_end_time: Optional[datetime] = Field(None, description="When override should expire")
    remediation_plan: Optional[str] = Field(None, description="Plan to remediate the override")
    bypassed_policies: List[str] = Field(default=[], description="Policy IDs being bypassed")
    compliance_frameworks_affected: List[str] = Field(default=[], description="Affected compliance frameworks")

class ApprovalResponse(BaseModel):
    approval_id: str
    status: ApprovalStatus
    workflow_id: str
    request_type: str
    created_at: datetime
    expires_at: Optional[datetime]
    current_approver: Optional[str]
    approvals_received: int
    approvals_required: int
    evidence_pack_id: Optional[str]

class OverrideResponse(BaseModel):
    override_id: str
    status: OverrideStatus
    workflow_id: str
    override_type: str
    reason_code: str
    risk_level: RiskLevel
    created_at: datetime
    override_end_time: Optional[datetime]
    approved_by: Optional[str]
    evidence_pack_id: Optional[str]

# =====================================================
# APPROVAL WORKFLOW SERVICE
# =====================================================

class ApprovalWorkflowService:
    """
    Enhanced approval workflow service with policy enforcement and quorum rules
    Tasks 14.1-T07, 14.1-T09, 14.1-T10, 14.1-T11, 14.1-T12, 14.1-T13: 
    - Approval service API and state machine
    - Policy pack enforcement in approvals
    - Residency enforcement in approvals  
    - Approval quorum rules
    - Time-bound approvals with auto-escalation
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.policy_engine = get_policy_engine(pool_manager)
        self.evidence_generator = EvidencePackGenerator(pool_manager)
        self.rbac_manager = RBACManager(pool_manager)
        self.sod_enforcer = SoDEnforcer(pool_manager)
        
        # Default approval chains by request type
        self.default_approval_chains = {
            "workflow_approval": ["sales_manager", "compliance_officer"],
            "policy_exception": ["compliance_manager", "legal_director", "ceo"],
            "override_request": ["sales_director", "cfo"],
            "emergency_override": ["on_call_manager", "security_director"],
            "financial_workflow": ["finance_manager", "cfo", "compliance_officer"]  # SOX compliance
        }
        
        # SLA timers by risk level (in hours)
        self.approval_sla_hours = {
            RiskLevel.LOW: 72,      # 3 days
            RiskLevel.MEDIUM: 48,   # 2 days  
            RiskLevel.HIGH: 24,     # 1 day
            RiskLevel.CRITICAL: 4   # 4 hours
        }
        
        # Quorum rules by approval type (Task 14.1-T12)
        self.quorum_rules = {
            "workflow_approval": {"type": "simple_majority", "minimum_count": 1},
            "policy_exception": {"type": "super_majority", "minimum_count": 2},
            "override_request": {"type": "unanimous", "minimum_count": 2},
            "emergency_override": {"type": "minimum_count", "minimum_count": 1},
            "financial_workflow": {"type": "super_majority", "minimum_count": 3}
        }
        
        # Escalation configuration for auto-escalation (Task 14.1-T13)
        self.escalation_config = {
            "sla_breach_escalation": True,
            "escalation_chains": {
                "workflow_approval": ["sales_director", "vp_sales", "cro"],
                "policy_exception": ["legal_director", "chief_legal_officer", "ceo"],
                "override_request": ["cfo", "ceo"],
                "emergency_override": ["security_director", "ciso", "ceo"]
            },
            "escalation_intervals_hours": [4, 8, 24]  # Escalate after 4h, 8h, 24h
        }
        
        # Dynamic approver selection weights (Task 14.1-T14)
        self.selection_weights = {
            "availability": 0.30,
            "workload": 0.25,
            "expertise": 0.20,
            "response_time": 0.15,
            "approval_rate": 0.10
        }
        
        # Bulk operation limits (Task 14.1-T38)
        self.bulk_limits = {
            "max_batch_size": 100,
            "max_concurrent_batches": 5,
            "batch_timeout_minutes": 30
        }
        
        # AI recommendation thresholds (Task 14.1-T40)
        self.ai_thresholds = {
            "auto_approve_threshold": 0.95,
            "recommend_approve_threshold": 0.80,
            "flag_for_review_threshold": 0.60,
            "recommend_reject_threshold": 0.40
        }
        
        # Auto-escalation configuration (Task 14.1-T13)
        self.escalation_config = {
            "enabled": True,
            "escalation_threshold": 0.75,  # 75% of SLA time elapsed
            "escalation_chains": {
                "workflow_approval": ["senior_manager", "director"],
                "policy_exception": ["ceo", "board_member"],
                "override_request": ["cfo", "ceo"],
                "financial_workflow": ["audit_committee", "board_chair"]
            }
        }
    
    async def create_approval_request(self, request: ApprovalRequest, tenant_id: int, 
                                    requested_by_user_id: int) -> ApprovalResponse:
        """
        Create approval request with enhanced policy enforcement (Tasks 14.1-T10, 14.1-T11, 14.1-T12)
        """
        
        try:
            # Enhanced policy enforcement (Tasks 14.1-T10, 14.1-T11)
            workflow_data = {
                'workflow_id': request.workflow_id,
                'created_by_user_id': requested_by_user_id,
                'financial_amount': request.metadata.get('financial_amount', 0),
                'processes_personal_data': request.metadata.get('processes_personal_data', False),
                'compliance_frameworks': request.compliance_frameworks
            }
            
            # Determine approval chain with quorum rules (Task 14.1-T12)
            approval_chain = request.approval_chain or self.default_approval_chains.get(
                request.request_type, ["compliance_officer"]
            )
            
            # Apply quorum rules
            quorum_rule = self.quorum_rules.get(request.request_type, {"type": "simple_majority", "min_approvers": 1})
            if len(approval_chain) < quorum_rule["min_approvers"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Approval chain must have at least {quorum_rule['min_approvers']} approvers for {request.request_type}"
                )
            
            # Enhanced policy enforcement
            policy_result = await self.policy_engine.enforce_approval_policy(
                tenant_id=tenant_id,
                user_id=requested_by_user_id,
                workflow_data=workflow_data,
                approval_type=request.request_type,
                residency_region=request.metadata.get('residency_region', 'GLOBAL'),
                approval_chain=approval_chain
            )
            
            # Check for critical policy violations
            if not policy_result['allowed']:
                critical_violations = [v for v in policy_result['violations'] if v['severity'] == 'critical']
                if critical_violations and not policy_result['override_options']:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Policy violations prevent approval: {', '.join([v['description'] for v in critical_violations])}"
                    )
            
            # Validate SoD constraints
            if not policy_result['sod_compliant']:
                raise HTTPException(
                    status_code=403,
                    detail="Segregation of Duties violation: User cannot approve their own workflow"
                )
            
            # Calculate expiration based on risk level (Task 14.1-T13)
            expires_at = datetime.utcnow() + timedelta(
                hours=self.approval_sla_hours[request.risk_assessment]
            )
            
            # Generate approval ID
            approval_id = str(uuid.uuid4())
            
            # Create evidence pack for the approval request
            evidence_pack_id = await self.evidence_generator.generate_evidence_pack(
                pack_type="approval_request",
                workflow_id=request.workflow_id,
                tenant_id=tenant_id,
                evidence_data={
                    "approval_id": approval_id,
                    "request_type": request.request_type,
                    "requested_by": requested_by_user_id,
                    "risk_assessment": request.risk_assessment.value,
                    "business_justification": request.business_justification,
                    "approval_chain": approval_chain,
                    "compliance_frameworks": request.compliance_frameworks
                }
            )
            
            # Store in database
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO approval_ledger (
                        approval_id, workflow_id, workflow_execution_id, tenant_id,
                        requested_by_user_id, request_type, request_reason, 
                        business_justification, risk_assessment, approval_chain,
                        expires_at, evidence_pack_id, compliance_frameworks,
                        policy_pack_refs, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, 
                    approval_id, request.workflow_id, request.workflow_execution_id,
                    tenant_id, requested_by_user_id, request.request_type,
                    request.request_reason, request.business_justification,
                    request.risk_assessment.value, json.dumps(approval_chain),
                    expires_at, evidence_pack_id, request.compliance_frameworks,
                    json.dumps({}), json.dumps(request.metadata)
                )
            
            logger.info(f"✅ Created approval request {approval_id} for workflow {request.workflow_id}")
            
            return ApprovalResponse(
                approval_id=approval_id,
                status=ApprovalStatus.PENDING,
                workflow_id=request.workflow_id,
                request_type=request.request_type,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                current_approver=approval_chain[0] if approval_chain else None,
                approvals_received=0,
                approvals_required=len(approval_chain),
                evidence_pack_id=evidence_pack_id
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create approval request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_approval_decision(self, decision: ApprovalDecision, 
                                      approver_user_id: int, tenant_id: int) -> ApprovalResponse:
        """Process an approval decision with digital signature"""
        
        try:
            # Fetch approval request
            async with self.pool_manager.get_connection() as conn:
                approval_row = await conn.fetchrow("""
                    SELECT * FROM approval_ledger 
                    WHERE approval_id = $1 AND tenant_id = $2
                """, decision.approval_id, tenant_id)
                
                if not approval_row:
                    raise HTTPException(status_code=404, detail="Approval request not found")
                
                if approval_row['status'] != 'pending':
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Approval request is already {approval_row['status']}"
                    )
                
                # Check if expired
                if approval_row['expires_at'] and datetime.utcnow() > approval_row['expires_at']:
                    await conn.execute("""
                        UPDATE approval_ledger SET status = 'expired' 
                        WHERE approval_id = $1
                    """, decision.approval_id)
                    raise HTTPException(status_code=400, detail="Approval request has expired")
                
                # Validate approver is in chain
                approval_chain = json.loads(approval_row['approval_chain'])
                current_approvals = json.loads(approval_row['approvals'] or '[]')
                
                # SoD check - approver cannot be the requester
                if approver_user_id == approval_row['requested_by_user_id']:
                    raise HTTPException(
                        status_code=403,
                        detail="SoD violation: Approver cannot be the same as requester"
                    )
                
                # Add approval record
                approval_record = {
                    "approver_user_id": approver_user_id,
                    "decision": decision.decision.value,
                    "comments": decision.comments,
                    "approved_at": datetime.utcnow().isoformat(),
                    "digital_signature": decision.digital_signature
                }
                current_approvals.append(approval_record)
                
                # Determine final status
                final_status = ApprovalStatus.PENDING
                if decision.decision == ApprovalStatus.REJECTED:
                    final_status = ApprovalStatus.REJECTED
                elif len(current_approvals) >= len(approval_chain):
                    # Check if all decisions are approved
                    if all(a["decision"] == "approved" for a in current_approvals):
                        final_status = ApprovalStatus.APPROVED
                    else:
                        final_status = ApprovalStatus.REJECTED
                
                # Update approval record
                completed_at = datetime.utcnow() if final_status != ApprovalStatus.PENDING else None
                
                await conn.execute("""
                    UPDATE approval_ledger 
                    SET approvals = $1, status = $2, completed_at = $3
                    WHERE approval_id = $4
                """, json.dumps(current_approvals), final_status.value, completed_at, decision.approval_id)
                
                # Generate evidence pack for the decision
                await self.evidence_generator.generate_evidence_pack(
                    pack_type="approval_decision",
                    workflow_id=approval_row['workflow_id'],
                    tenant_id=tenant_id,
                    evidence_data={
                        "approval_id": decision.approval_id,
                        "decision": decision.decision.value,
                        "approver_user_id": approver_user_id,
                        "comments": decision.comments,
                        "final_status": final_status.value,
                        "approvals_received": len(current_approvals),
                        "digital_signature": decision.digital_signature
                    }
                )
            
            logger.info(f"✅ Processed approval decision {decision.approval_id}: {decision.decision.value}")
            
            return ApprovalResponse(
                approval_id=decision.approval_id,
                status=final_status,
                workflow_id=approval_row['workflow_id'],
                request_type=approval_row['request_type'],
                created_at=approval_row['created_at'],
                expires_at=approval_row['expires_at'],
                current_approver=None if final_status != ApprovalStatus.PENDING else approval_chain[len(current_approvals)],
                approvals_received=len(current_approvals),
                approvals_required=len(approval_chain),
                evidence_pack_id=approval_row['evidence_pack_id']
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to process approval decision: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def check_sla_compliance_and_escalate(self, approval_id: str, tenant_id: int) -> Dict[str, Any]:
        """
        Check SLA compliance and trigger auto-escalation if needed (Task 14.1-T13)
        """
        try:
            async with self.pool_manager.get_connection() as conn:
                # Get approval request details
                approval_row = await conn.fetchrow("""
                    SELECT approval_id, workflow_id, request_type, risk_assessment,
                           created_at, expires_at, status, approval_chain, approvals
                    FROM approval_ledger 
                    WHERE approval_id = $1 AND tenant_id = $2
                """, approval_id, tenant_id)
                
                if not approval_row or approval_row['status'] != 'pending':
                    return {"escalation_needed": False, "reason": "Request not found or not pending"}
                
                current_time = datetime.utcnow()
                created_at = approval_row['created_at']
                expires_at = approval_row['expires_at']
                
                # Calculate time elapsed
                total_duration = expires_at - created_at
                elapsed_time = current_time - created_at
                elapsed_percentage = elapsed_time.total_seconds() / total_duration.total_seconds()
                
                # Check if escalation threshold is reached
                escalation_threshold = self.escalation_config["escalation_threshold"]
                escalation_needed = elapsed_percentage >= escalation_threshold
                
                result = {
                    "approval_id": approval_id,
                    "elapsed_percentage": round(elapsed_percentage * 100, 2),
                    "escalation_threshold": round(escalation_threshold * 100, 2),
                    "escalation_needed": escalation_needed,
                    "sla_breached": current_time > expires_at,
                    "time_remaining_hours": max(0, (expires_at - current_time).total_seconds() / 3600)
                }
                
                # Trigger auto-escalation if needed
                if escalation_needed and self.escalation_config["enabled"]:
                    escalation_chain = self.escalation_config["escalation_chains"].get(
                        approval_row['request_type'], ["senior_manager"]
                    )
                    
                    # Add escalation approvers to the chain
                    current_chain = json.loads(approval_row['approval_chain'])
                    escalated_chain = current_chain + escalation_chain
                    
                    # Update approval request with escalated chain
                    await conn.execute("""
                        UPDATE approval_ledger 
                        SET approval_chain = $1, status = 'escalated'
                        WHERE approval_id = $2
                    """, json.dumps(escalated_chain), approval_id)
                    
                    # Generate evidence pack for escalation
                    await self.evidence_generator.generate_evidence_pack(
                        pack_type="approval_escalation",
                        workflow_id=approval_row['workflow_id'],
                        tenant_id=tenant_id,
                        evidence_data={
                            "approval_id": approval_id,
                            "escalation_reason": "SLA threshold exceeded",
                            "elapsed_percentage": elapsed_percentage,
                            "original_chain": current_chain,
                            "escalated_chain": escalated_chain,
                            "escalation_timestamp": current_time.isoformat()
                        }
                    )
                    
                    result.update({
                        "escalation_triggered": True,
                        "escalated_to": escalation_chain,
                        "new_chain": escalated_chain
                    })
                    
                    logger.info(f"⬆️ Auto-escalated approval {approval_id} due to SLA threshold")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ SLA compliance check failed: {e}")
            return {"error": str(e), "escalation_needed": False}
    
    async def calculate_quorum_status(self, approval_id: str, tenant_id: int) -> Dict[str, Any]:
        """
        Calculate quorum status for approval request (Task 14.1-T12)
        """
        try:
            async with self.pool_manager.get_connection() as conn:
                approval_row = await conn.fetchrow("""
                    SELECT request_type, approval_chain, approvals, status
                    FROM approval_ledger 
                    WHERE approval_id = $1 AND tenant_id = $2
                """, approval_id, tenant_id)
                
                if not approval_row:
                    return {"error": "Approval request not found"}
                
                request_type = approval_row['request_type']
                approval_chain = json.loads(approval_row['approval_chain'])
                approvals = json.loads(approval_row['approvals'] or '[]')
                
                # Get quorum rules for this request type
                quorum_rule = self.quorum_rules.get(request_type, {"type": "simple_majority", "min_approvers": 1})
                
                total_approvers = len(approval_chain)
                received_approvals = len([a for a in approvals if a.get('decision') == 'approved'])
                received_rejections = len([a for a in approvals if a.get('decision') == 'rejected'])
                
                # Calculate quorum based on type
                quorum_met = False
                if quorum_rule["type"] == "simple_majority":
                    quorum_met = received_approvals > (total_approvers / 2)
                elif quorum_rule["type"] == "super_majority":
                    quorum_met = received_approvals >= (total_approvers * 0.67)
                elif quorum_rule["type"] == "unanimous":
                    quorum_met = received_approvals == total_approvers and received_rejections == 0
                elif quorum_rule["type"] == "minimum_count":
                    quorum_met = received_approvals >= quorum_rule["min_approvers"]
                
                # Check if rejection threshold is met
                rejection_threshold_met = received_rejections > (total_approvers / 2)
                
                return {
                    "approval_id": approval_id,
                    "quorum_type": quorum_rule["type"],
                    "total_approvers": total_approvers,
                    "received_approvals": received_approvals,
                    "received_rejections": received_rejections,
                    "quorum_met": quorum_met and not rejection_threshold_met,
                    "rejection_threshold_met": rejection_threshold_met,
                    "approval_percentage": round((received_approvals / total_approvers) * 100, 2) if total_approvers > 0 else 0,
                    "status": approval_row['status']
                }
                
        except Exception as e:
            logger.error(f"❌ Quorum calculation failed: {e}")
            return {"error": str(e)}
    
    async def select_dynamic_approvers(self, approval_request: Dict[str, Any], 
                                     required_count: int = 1) -> List[int]:
        """
        Dynamic approver selection based on workload and availability (Task 14.1-T14)
        """
        try:
            tenant_id = approval_request["tenant_id"]
            request_type = approval_request["request_type"]
            
            # Get eligible approvers based on role and permissions
            eligible_approvers = await self._get_eligible_approvers(tenant_id, request_type)
            
            if len(eligible_approvers) < required_count:
                logger.warning(f"⚠️ Insufficient eligible approvers: {len(eligible_approvers)} < {required_count}")
                return eligible_approvers[:required_count] if eligible_approvers else []
            
            # Calculate workload metrics for each approver
            approver_scores = []
            for approver_id in eligible_approvers:
                workload_score = await self._calculate_approver_workload_score(approver_id, tenant_id)
                approver_scores.append((approver_id, workload_score))
            
            # Sort by score (highest first) and select top N
            approver_scores.sort(key=lambda x: x[1], reverse=True)
            selected_approvers = [approver_id for approver_id, _ in approver_scores[:required_count]]
            
            logger.info(f"✅ Selected {len(selected_approvers)} dynamic approvers for {request_type}")
            return selected_approvers
            
        except Exception as e:
            logger.error(f"❌ Dynamic approver selection failed: {e}")
            return eligible_approvers[:required_count] if eligible_approvers else []
    
    async def process_bulk_approvals(self, approval_requests: List[Dict[str, Any]], 
                                   approver_id: int, tenant_id: int) -> Dict[str, Any]:
        """
        Process bulk approval operations (Task 14.1-T38)
        """
        try:
            if len(approval_requests) > self.bulk_limits["max_batch_size"]:
                raise ValueError(f"Batch size {len(approval_requests)} exceeds limit {self.bulk_limits['max_batch_size']}")
            
            batch_id = str(uuid.uuid4())
            results = {
                "batch_id": batch_id,
                "total_requests": len(approval_requests),
                "processed": 0,
                "approved": 0,
                "rejected": 0,
                "errors": 0,
                "results": []
            }
            
            # Process in chunks for better performance
            chunk_size = 10
            for i in range(0, len(approval_requests), chunk_size):
                chunk = approval_requests[i:i + chunk_size]
                
                for request in chunk:
                    try:
                        approval_id = request.get("approval_id")
                        decision = request.get("decision", "approved")
                        
                        # Process individual approval
                        if decision == "approved":
                            # Simulate approval processing
                            results["approved"] += 1
                            status = "approved"
                        else:
                            results["rejected"] += 1
                            status = "rejected"
                        
                        results["results"].append({
                            "approval_id": approval_id,
                            "status": status,
                            "processed_at": datetime.utcnow().isoformat(),
                            "approver_id": approver_id
                        })
                        results["processed"] += 1
                        
                    except Exception as e:
                        results["results"].append({
                            "approval_id": request.get("approval_id"),
                            "status": "error",
                            "error": str(e),
                            "processed_at": datetime.utcnow().isoformat()
                        })
                        results["errors"] += 1
            
            logger.info(f"✅ Processed bulk approvals: {results['approved']} approved, {results['rejected']} rejected, {results['errors']} errors")
            return results
            
        except Exception as e:
            logger.error(f"❌ Bulk approval processing failed: {e}")
            raise
    
    async def get_ai_approval_recommendation(self, approval_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-powered approval recommendation (Task 14.1-T40)
        """
        try:
            tenant_id = approval_request["tenant_id"]
            workflow_id = approval_request.get("workflow_id")
            
            # Calculate trust score for the workflow (simplified)
            trust_score = 0.85  # Would integrate with actual trust scoring engine
            
            # Analyze historical patterns (simplified)
            historical_approval_rate = 0.75
            
            # Risk assessment
            risk_factors = []
            if approval_request.get("financial_amount", 0) > 100000:
                risk_factors.append("High financial impact")
            if approval_request.get("cross_border_data"):
                risk_factors.append("Cross-border data transfer")
            
            # Generate recommendation based on trust score and risk
            if trust_score >= self.ai_thresholds["auto_approve_threshold"] and not risk_factors:
                recommendation = "auto_approve"
                confidence = 0.95
            elif trust_score >= self.ai_thresholds["recommend_approve_threshold"]:
                recommendation = "recommend_approve"
                confidence = 0.80
            elif trust_score >= self.ai_thresholds["flag_for_review_threshold"]:
                recommendation = "manual_review"
                confidence = 0.60
            else:
                recommendation = "recommend_reject"
                confidence = 0.70
            
            # Adjust confidence based on risk factors
            confidence = max(0.1, confidence - (len(risk_factors) * 0.1))
            
            return {
                "approval_id": approval_request.get("approval_id"),
                "recommendation": recommendation,
                "confidence": confidence,
                "reasoning": [
                    f"Trust score: {trust_score:.2f}",
                    f"Historical approval rate: {historical_approval_rate:.1%}",
                    f"Risk factors: {len(risk_factors)}"
                ] + risk_factors,
                "trust_score": trust_score,
                "risk_factors": risk_factors,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ AI approval recommendation failed: {e}")
            return {
                "recommendation": "manual_review",
                "confidence": 0.0,
                "reasoning": ["AI recommendation system error"],
                "error": str(e)
            }
    
    # Helper methods for advanced features
    async def _get_eligible_approvers(self, tenant_id: int, request_type: str) -> List[int]:
        """Get eligible approvers based on role and permissions"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                rows = await conn.fetch("""
                    SELECT DISTINCT u.user_id
                    FROM users u
                    JOIN users_role ur ON u.user_id = ur.user_id
                    WHERE u.tenant_id = $1 
                    AND u.active = true
                    AND (
                        ur.crux_approve = true OR
                        ur.role_name IN ('compliance_officer', 'finance_manager', 'cfo', 'ceo')
                    )
                """, tenant_id)
                
                return [row['user_id'] for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get eligible approvers: {e}")
            return []
    
    async def _calculate_approver_workload_score(self, approver_id: int, tenant_id: int) -> float:
        """Calculate workload score for approver selection"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Get pending approvals count
                pending_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM approval_ledger
                    WHERE tenant_id = $1 AND status = 'pending'
                    AND $2 = ANY(string_to_array(approval_chain::text, ',')::int[])
                """, tenant_id, approver_id) or 0
                
                # Get average response time (simplified)
                avg_response_hours = 24.0  # Default
                
                # Get approval rate (simplified)
                approval_rate = 0.8  # Default
                
                # Calculate composite score (higher is better)
                availability_score = 1.0  # Assume available
                workload_score = max(0.0, 1.0 - (pending_count / 10.0))  # Normalize workload
                response_score = max(0.0, 1.0 - (avg_response_hours / 48.0))  # Normalize response time
                
                composite_score = (
                    (availability_score * self.selection_weights["availability"]) +
                    (workload_score * self.selection_weights["workload"]) +
                    (0.8 * self.selection_weights["expertise"]) +  # Default expertise
                    (response_score * self.selection_weights["response_time"]) +
                    (approval_rate * self.selection_weights["approval_rate"])
                )
                
                return composite_score
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate approver workload score: {e}")
            return 0.5  # Default score

# =====================================================
# OVERRIDE LEDGER SERVICE  
# =====================================================

class OverrideLedgerService:
    """
    Override ledger service for compliance bypass tracking
    Task 14.1-T08: Override ledger service API
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.evidence_generator = EvidencePackGenerator(pool_manager)
        self.approval_service = ApprovalWorkflowService(pool_manager)
    
    async def create_override_request(self, request: OverrideRequest, tenant_id: int,
                                    requested_by_user_id: int) -> OverrideResponse:
        """Create a new override request with approval workflow"""
        
        try:
            override_id = str(uuid.uuid4())
            
            # High-risk overrides require approval
            requires_approval = request.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            approval_id = None
            
            if requires_approval:
                # Create approval request for the override
                approval_request = ApprovalRequest(
                    workflow_id=request.workflow_id,
                    workflow_execution_id=request.workflow_execution_id,
                    request_type="override_request",
                    request_reason=f"Override request: {request.reason_description}",
                    business_justification=request.business_impact_assessment,
                    risk_assessment=request.risk_level,
                    compliance_frameworks=request.compliance_frameworks_affected
                )
                
                approval_response = await self.approval_service.create_approval_request(
                    approval_request, tenant_id, requested_by_user_id
                )
                approval_id = approval_response.approval_id
            
            # Generate evidence pack
            evidence_pack_id = await self.evidence_generator.generate_evidence_pack(
                pack_type="override_request",
                workflow_id=request.workflow_id,
                tenant_id=tenant_id,
                evidence_data={
                    "override_id": override_id,
                    "override_type": request.override_type,
                    "reason_code": request.reason_code,
                    "reason_description": request.reason_description,
                    "risk_level": request.risk_level.value,
                    "bypassed_policies": request.bypassed_policies,
                    "compliance_frameworks_affected": request.compliance_frameworks_affected,
                    "business_impact_assessment": request.business_impact_assessment,
                    "requires_approval": requires_approval,
                    "approval_id": approval_id
                }
            )
            
            # Store in database
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO override_ledger (
                        override_id, workflow_id, workflow_execution_id, tenant_id,
                        override_type, reason_code, reason_description, 
                        business_impact_assessment, requested_by_user_id,
                        approval_id, bypassed_policies, compliance_frameworks_affected,
                        risk_level, estimated_business_impact, override_end_time,
                        evidence_pack_id, remediation_plan, status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                """,
                    override_id, request.workflow_id, request.workflow_execution_id,
                    tenant_id, request.override_type, request.reason_code,
                    request.reason_description, request.business_impact_assessment,
                    requested_by_user_id, approval_id, json.dumps(request.bypassed_policies),
                    request.compliance_frameworks_affected, request.risk_level.value,
                    request.estimated_business_impact, request.override_end_time,
                    evidence_pack_id, request.remediation_plan,
                    "pending_approval" if requires_approval else "active"
                )
            
            logger.info(f"✅ Created override request {override_id} for workflow {request.workflow_id}")
            
            return OverrideResponse(
                override_id=override_id,
                status=OverrideStatus.ACTIVE if not requires_approval else "pending_approval",
                workflow_id=request.workflow_id,
                override_type=request.override_type,
                reason_code=request.reason_code,
                risk_level=request.risk_level,
                created_at=datetime.utcnow(),
                override_end_time=request.override_end_time,
                approved_by=None,
                evidence_pack_id=evidence_pack_id
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create override request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# API ENDPOINTS
# =====================================================

# Initialize services
approval_service = None
override_service = None

def get_approval_service(pool_manager=Depends(get_pool_manager)) -> ApprovalWorkflowService:
    global approval_service
    if approval_service is None:
        approval_service = ApprovalWorkflowService(pool_manager)
    return approval_service

def get_override_service(pool_manager=Depends(get_pool_manager)) -> OverrideLedgerService:
    global override_service
    if override_service is None:
        override_service = OverrideLedgerService(pool_manager)
    return override_service

@router.post("/approval-requests", response_model=ApprovalResponse)
async def create_approval_request(
    request: ApprovalRequest,
    tenant_id: int = Query(..., description="Tenant ID"),
    requested_by_user_id: int = Query(..., description="User ID of requester"),
    service: ApprovalWorkflowService = Depends(get_approval_service)
):
    """
    Create a new approval request
    Task 14.1-T07: Approval workflow service API
    """
    return await service.create_approval_request(request, tenant_id, requested_by_user_id)

@router.post("/approval-decisions", response_model=ApprovalResponse)
async def process_approval_decision(
    decision: ApprovalDecision,
    tenant_id: int = Query(..., description="Tenant ID"),
    approver_user_id: int = Query(..., description="User ID of approver"),
    service: ApprovalWorkflowService = Depends(get_approval_service)
):
    """
    Process an approval decision
    Task 14.1-T07: Approval workflow service API
    """
    return await service.process_approval_decision(decision, approver_user_id, tenant_id)

@router.get("/approval-requests/{approval_id}", response_model=ApprovalResponse)
async def get_approval_request(
    approval_id: str,
    tenant_id: int = Query(..., description="Tenant ID"),
    pool_manager=Depends(get_pool_manager)
):
    """Get approval request details"""
    
    async with pool_manager.get_connection() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM approval_ledger 
            WHERE approval_id = $1 AND tenant_id = $2
        """, approval_id, tenant_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Approval request not found")
        
        approval_chain = json.loads(row['approval_chain'])
        approvals = json.loads(row['approvals'] or '[]')
        
        return ApprovalResponse(
            approval_id=row['approval_id'],
            status=ApprovalStatus(row['status']),
            workflow_id=row['workflow_id'],
            request_type=row['request_type'],
            created_at=row['created_at'],
            expires_at=row['expires_at'],
            current_approver=approval_chain[len(approvals)] if len(approvals) < len(approval_chain) and row['status'] == 'pending' else None,
            approvals_received=len(approvals),
            approvals_required=len(approval_chain),
            evidence_pack_id=row['evidence_pack_id']
        )

@router.post("/override-requests", response_model=OverrideResponse)
async def create_override_request(
    request: OverrideRequest,
    tenant_id: int = Query(..., description="Tenant ID"),
    requested_by_user_id: int = Query(..., description="User ID of requester"),
    service: OverrideLedgerService = Depends(get_override_service)
):
    """
    Create a new override request
    Task 14.1-T08: Override ledger service API
    """
    return await service.create_override_request(request, tenant_id, requested_by_user_id)

@router.get("/pending-approvals")
async def get_pending_approvals(
    tenant_id: int = Query(..., description="Tenant ID"),
    approver_role: Optional[str] = Query(None, description="Filter by approver role"),
    pool_manager=Depends(get_pool_manager)
):
    """Get pending approval requests for a tenant"""
    
    async with pool_manager.get_connection() as conn:
        query = """
            SELECT approval_id, workflow_id, request_type, request_reason,
                   risk_assessment, created_at, expires_at, approval_chain
            FROM approval_ledger 
            WHERE tenant_id = $1 AND status = 'pending'
            ORDER BY created_at DESC
        """
        
        rows = await conn.fetch(query, tenant_id)
        
        pending_approvals = []
        for row in rows:
            approval_chain = json.loads(row['approval_chain'])
            if not approver_role or approver_role in approval_chain:
                pending_approvals.append({
                    "approval_id": row['approval_id'],
                    "workflow_id": row['workflow_id'],
                    "request_type": row['request_type'],
                    "request_reason": row['request_reason'],
                    "risk_assessment": row['risk_assessment'],
                    "created_at": row['created_at'],
                    "expires_at": row['expires_at'],
                    "approval_chain": approval_chain
                })
        
        return {"pending_approvals": pending_approvals, "count": len(pending_approvals)}

@router.get("/override-history")
async def get_override_history(
    tenant_id: int = Query(..., description="Tenant ID"),
    workflow_id: Optional[str] = Query(None, description="Filter by workflow ID"),
    limit: int = Query(50, description="Limit results"),
    pool_manager=Depends(get_pool_manager)
):
    """Get override history for a tenant"""
    
    async with pool_manager.get_connection() as conn:
        if workflow_id:
            query = """
                SELECT override_id, workflow_id, override_type, reason_code,
                       risk_level, created_at, status, estimated_business_impact
                FROM override_ledger 
                WHERE tenant_id = $1 AND workflow_id = $2
                ORDER BY created_at DESC LIMIT $3
            """
            rows = await conn.fetch(query, tenant_id, workflow_id, limit)
        else:
            query = """
                SELECT override_id, workflow_id, override_type, reason_code,
                       risk_level, created_at, status, estimated_business_impact
                FROM override_ledger 
                WHERE tenant_id = $1
                ORDER BY created_at DESC LIMIT $2
            """
            rows = await conn.fetch(query, tenant_id, limit)
        
        overrides = []
        for row in rows:
            overrides.append({
                "override_id": row['override_id'],
                "workflow_id": row['workflow_id'],
                "override_type": row['override_type'],
                "reason_code": row['reason_code'],
                "risk_level": row['risk_level'],
                "created_at": row['created_at'],
                "status": row['status'],
                "estimated_business_impact": row['estimated_business_impact']
            })
        
        return {"overrides": overrides, "count": len(overrides)}

@router.post("/sla-check/{approval_id}")
async def check_sla_compliance(
    approval_id: str,
    tenant_id: int = Query(..., description="Tenant ID"),
    service: ApprovalWorkflowService = Depends(get_approval_service)
):
    """
    Check SLA compliance and trigger auto-escalation if needed
    Task 14.1-T13: Time-bound approvals with auto-escalation
    """
    return await service.check_sla_compliance_and_escalate(approval_id, tenant_id)

@router.get("/quorum-status/{approval_id}")
async def get_quorum_status(
    approval_id: str,
    tenant_id: int = Query(..., description="Tenant ID"),
    service: ApprovalWorkflowService = Depends(get_approval_service)
):
    """
    Get quorum status for approval request
    Task 14.1-T12: Approval quorum rules
    """
    return await service.calculate_quorum_status(approval_id, tenant_id)

@router.post("/policy-enforcement-check")
async def check_policy_enforcement(
    tenant_id: int = Query(..., description="Tenant ID"),
    user_id: int = Query(..., description="User ID"),
    workflow_data: Dict[str, Any] = Body(..., description="Workflow data"),
    approval_type: str = Body(..., description="Approval type"),
    residency_region: str = Body("GLOBAL", description="Residency region"),
    approval_chain: Optional[List[str]] = Body(None, description="Proposed approval chain"),
    pool_manager=Depends(get_pool_manager)
):
    """
    Check policy enforcement for approval workflow
    Tasks 14.1-T10, 14.1-T11: Policy pack enforcement and residency enforcement
    """
    try:
        policy_engine = get_policy_engine(pool_manager)
        result = await policy_engine.enforce_approval_policy(
            tenant_id=tenant_id,
            user_id=user_id,
            workflow_data=workflow_data,
            approval_type=approval_type,
            residency_region=residency_region,
            approval_chain=approval_chain
        )
        return result
    except Exception as e:
        logger.error(f"❌ Policy enforcement check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dynamic-approver-selection")
async def select_dynamic_approvers(
    approval_request: Dict[str, Any] = Body(...),
    required_count: int = Query(1, description="Number of approvers required"),
    service: ApprovalWorkflowService = Depends(get_approval_service)
):
    """
    Select dynamic approvers based on workload and availability
    Task 14.1-T14: Dynamic approver selection
    """
    try:
        selected_approvers = await service.select_dynamic_approvers(approval_request, required_count)
        return {
            "selected_approvers": selected_approvers,
            "count": len(selected_approvers),
            "selection_criteria": "workload_and_availability"
        }
    except Exception as e:
        logger.error(f"❌ Dynamic approver selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-approvals")
async def process_bulk_approvals(
    approval_requests: List[Dict[str, Any]] = Body(...),
    approver_id: int = Query(..., description="ID of the approver"),
    tenant_id: int = Query(..., description="Tenant ID"),
    service: ApprovalWorkflowService = Depends(get_approval_service)
):
    """
    Process bulk approval operations
    Task 14.1-T38: Bulk approval operations
    """
    try:
        results = await service.process_bulk_approvals(approval_requests, approver_id, tenant_id)
        return results
    except Exception as e:
        logger.error(f"❌ Bulk approval processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-approval-recommendation")
async def get_ai_approval_recommendation(
    approval_request: Dict[str, Any] = Body(...),
    service: ApprovalWorkflowService = Depends(get_approval_service)
):
    """
    Get AI-powered approval recommendation
    Task 14.1-T40: AI approval assistance
    """
    try:
        recommendation = await service.get_ai_approval_recommendation(approval_request)
        return recommendation
    except Exception as e:
        logger.error(f"❌ AI approval recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "approval_workflow_api", "timestamp": datetime.utcnow()}
