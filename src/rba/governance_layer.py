"""
RBA Governance and Compliance Layer
Complete governance framework for Rule-Based Automation with SOX, GDPR, and audit compliance
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import json
import hashlib
import logging
import uuid
from dataclasses import dataclass
import asyncio

from ..saas.schemas import (
    SaaSWorkflowTrace, SaaSAuditTrail, SaaSEvidencePack, 
    SaaSRuleViolation, SaaSOverrideRecord, SaaSSchemaFactory
)

logger = logging.getLogger(__name__)

class GovernanceLevel(Enum):
    BASIC = "basic"
    ENHANCED = "enhanced"
    STRICT = "strict"
    REGULATORY = "regulatory"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceFramework(Enum):
    SOX = "sox"
    GDPR = "gdpr"
    DSAR = "dsar"
    CCPA = "ccpa"
    HIPAA = "hipaa"

@dataclass
class GovernanceConfig:
    """Configuration for governance controls"""
    governance_level: GovernanceLevel
    compliance_frameworks: List[ComplianceFramework]
    audit_retention_days: int
    immutable_logs: bool
    maker_checker_required: bool
    approval_timeout_hours: int
    risk_threshold: RiskLevel
    evidence_collection_required: bool

class ApprovalRequest(BaseModel):
    """Model for approval requests"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_execution_id: str
    requested_by: str
    request_type: str  # "override", "parameter_change", "policy_exception"
    request_reason: str
    business_justification: str
    risk_assessment: RiskLevel
    approval_chain: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approvals: List[Dict[str, Any]] = []
    
    class Config:
        use_enum_values = True

class TrustScore(BaseModel):
    """Trust scoring model"""
    entity_id: str  # workflow_id, user_id, etc.
    entity_type: str  # "workflow", "user", "tenant"
    score: float  # 0.0 to 1.0
    factors: Dict[str, float] = {}
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    history: List[Dict[str, Any]] = []

class GovernanceEvent(BaseModel):
    """Governance event model"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    workflow_execution_id: Optional[str] = None
    tenant_id: str
    user_id: Optional[str] = None
    event_data: Dict[str, Any] = {}
    risk_level: RiskLevel = RiskLevel.LOW
    compliance_impact: List[ComplianceFramework] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True

# Governance Components
# Note: Trust scoring is now handled by the enhanced TrustScoringEngine in dsl/intelligence/trust_scoring_engine.py
# This provides more comprehensive, tenant-aware, and industry-specific trust scoring with dynamic configuration

class ApprovalWorkflowManager:
    """Manages approval workflows for overrides and exceptions"""
    
    def __init__(self):
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.approval_history: List[ApprovalRequest] = []
        self.approval_chains = self._initialize_approval_chains()
    
    def _initialize_approval_chains(self) -> Dict[str, List[str]]:
        """Initialize default approval chains"""
        return {
            "pipeline_override": ["sales_manager", "sales_director", "cro"],
            "forecast_override": ["sales_manager", "finance_director", "cfo"],
            "policy_exception": ["compliance_manager", "legal_director", "ceo"],
            "parameter_change": ["workflow_admin", "compliance_manager"],
            "emergency_override": ["on_call_manager", "security_director"]
        }
    
    async def create_approval_request(
        self,
        workflow_execution_id: str,
        requested_by: str,
        request_type: str,
        request_reason: str,
        business_justification: str,
        risk_assessment: RiskLevel,
        custom_approval_chain: Optional[List[str]] = None
    ) -> ApprovalRequest:
        """Create a new approval request"""
        
        # Determine approval chain
        approval_chain = custom_approval_chain or self.approval_chains.get(
            request_type, ["manager", "director"]
        )
        
        # Create request
        request = ApprovalRequest(
            workflow_execution_id=workflow_execution_id,
            requested_by=requested_by,
            request_type=request_type,
            request_reason=request_reason,
            business_justification=business_justification,
            risk_assessment=risk_assessment,
            approval_chain=approval_chain,
            expires_at=datetime.utcnow() + timedelta(hours=24)  # Default 24h expiry
        )
        
        self.pending_requests[request.request_id] = request
        
        # Send notifications to approvers
        await self._notify_approvers(request)
        
        logger.info(f"Created approval request: {request.request_id} for {request_type}")
        
        return request
    
    async def process_approval(
        self,
        request_id: str,
        approver: str,
        decision: ApprovalStatus,
        comments: str = ""
    ) -> Dict[str, Any]:
        """Process an approval decision"""
        
        request = self.pending_requests.get(request_id)
        if not request:
            return {"success": False, "error": "Request not found"}
        
        if request.status != ApprovalStatus.PENDING:
            return {"success": False, "error": f"Request already {request.status.value}"}
        
        if datetime.utcnow() > request.expires_at:
            request.status = ApprovalStatus.EXPIRED
            return {"success": False, "error": "Request has expired"}
        
        # Verify approver is in chain
        if approver not in request.approval_chain:
            return {"success": False, "error": "Approver not authorized for this request"}
        
        # Add approval
        approval_record = {
            "approver": approver,
            "decision": decision.value,
            "comments": comments,
            "approved_at": datetime.utcnow().isoformat()
        }
        request.approvals.append(approval_record)
        
        # Check if all approvals received
        if decision == ApprovalStatus.REJECTED:
            request.status = ApprovalStatus.REJECTED
        elif len(request.approvals) >= len(request.approval_chain):
            # All approvers have responded
            if all(a["decision"] == "approved" for a in request.approvals):
                request.status = ApprovalStatus.APPROVED
            else:
                request.status = ApprovalStatus.REJECTED
        
        # Move to history if completed
        if request.status in [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED]:
            self.approval_history.append(request)
            self.pending_requests.pop(request_id)
        
        logger.info(f"Processed approval: {request_id} - {decision.value} by {approver}")
        
        return {
            "success": True,
            "request_status": request.status.value,
            "approvals_received": len(request.approvals),
            "approvals_required": len(request.approval_chain)
        }
    
    async def _notify_approvers(self, request: ApprovalRequest):
        """Send notifications to approvers"""
        # Mock notification - in real implementation, send actual notifications
        logger.info(f"Notifying approvers for request {request.request_id}: {request.approval_chain}")
    
    async def get_pending_approvals(self, approver: str) -> List[ApprovalRequest]:
        """Get pending approvals for a specific approver"""
        return [
            request for request in self.pending_requests.values()
            if approver in request.approval_chain and 
            request.status == ApprovalStatus.PENDING and
            datetime.utcnow() <= request.expires_at
        ]

class ComplianceMonitor:
    """Monitors and ensures compliance across all workflows"""
    
    def __init__(self):
        self.compliance_rules = self._initialize_compliance_rules()
        self.violation_handlers = self._initialize_violation_handlers()
        self.schema_factory = SaaSSchemaFactory()
    
    def _initialize_compliance_rules(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Initialize compliance rules for each framework"""
        return {
            ComplianceFramework.SOX: {
                "audit_trail_required": True,
                "immutable_logs": True,
                "retention_days": 2555,  # 7 years
                "maker_checker_required": True,
                "financial_controls": True,
                "segregation_of_duties": True
            },
            ComplianceFramework.GDPR: {
                "consent_required": True,
                "data_minimization": True,
                "right_to_erasure": True,
                "data_portability": True,
                "privacy_by_design": True,
                "breach_notification_hours": 72
            },
            ComplianceFramework.DSAR: {
                "data_subject_rights": True,
                "automated_decision_transparency": True,
                "lawful_basis_documentation": True,
                "data_protection_impact_assessment": True
            }
        }
    
    def _initialize_violation_handlers(self) -> Dict[ComplianceFramework, Callable]:
        """Initialize violation handlers for each framework"""
        return {
            ComplianceFramework.SOX: self._handle_sox_violation,
            ComplianceFramework.GDPR: self._handle_gdpr_violation,
            ComplianceFramework.DSAR: self._handle_dsar_violation
        }
    
    async def monitor_workflow_compliance(
        self,
        workflow_execution_id: str,
        workflow_definition: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Monitor workflow for compliance violations"""
        
        compliance_frameworks = workflow_definition.get("compliance_frameworks", [])
        violations = []
        compliance_status = {}
        
        for framework_str in compliance_frameworks:
            try:
                framework = ComplianceFramework(framework_str.lower())
                framework_violations = await self._check_framework_compliance(
                    framework, workflow_definition, execution_context
                )
                
                violations.extend(framework_violations)
                compliance_status[framework.value] = "compliant" if not framework_violations else "violation"
                
            except ValueError:
                logger.warning(f"Unknown compliance framework: {framework_str}")
        
        return {
            "workflow_execution_id": workflow_execution_id,
            "compliance_status": compliance_status,
            "violations": violations,
            "overall_compliant": len(violations) == 0
        }
    
    async def _check_framework_compliance(
        self,
        framework: ComplianceFramework,
        workflow_definition: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> List[SaaSRuleViolation]:
        """Check compliance for a specific framework"""
        
        rules = self.compliance_rules.get(framework, {})
        violations = []
        
        for rule_name, rule_requirement in rules.items():
            violation = await self._check_compliance_rule(
                framework, rule_name, rule_requirement, workflow_definition, execution_context
            )
            if violation:
                violations.append(violation)
        
        return violations
    
    async def _check_compliance_rule(
        self,
        framework: ComplianceFramework,
        rule_name: str,
        rule_requirement: Any,
        workflow_definition: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Optional[SaaSRuleViolation]:
        """Check a specific compliance rule"""
        
        if rule_name == "audit_trail_required" and rule_requirement:
            if not execution_context.get("audit_trail_enabled", False):
                return SaaSRuleViolation(
                    rule_id=f"{framework.value}_{rule_name}",
                    rule_name=f"{framework.value.upper()} Audit Trail Required",
                    rule_description=f"Audit trail is required for {framework.value.upper()} compliance",
                    violation_type="compliance",
                    severity="high",
                    expected_value=True,
                    actual_value=False,
                    business_impact_description="Audit trail missing - compliance violation",
                    risk_category="compliance"
                )
        
        elif rule_name == "consent_required" and rule_requirement:
            if execution_context.get("contains_personal_data", False) and not execution_context.get("consent_verified", False):
                return SaaSRuleViolation(
                    rule_id=f"{framework.value}_{rule_name}",
                    rule_name=f"{framework.value.upper()} Consent Required",
                    rule_description="User consent required for personal data processing",
                    violation_type="privacy",
                    severity="critical",
                    expected_value=True,
                    actual_value=False,
                    business_impact_description="Processing personal data without consent",
                    risk_category="privacy"
                )
        
        elif rule_name == "maker_checker_required" and rule_requirement:
            if execution_context.get("high_risk_operation", False) and not execution_context.get("maker_checker_applied", False):
                return SaaSRuleViolation(
                    rule_id=f"{framework.value}_{rule_name}",
                    rule_name=f"{framework.value.upper()} Maker-Checker Required",
                    rule_description="Maker-checker control required for high-risk operations",
                    violation_type="governance",
                    severity="high",
                    expected_value=True,
                    actual_value=False,
                    business_impact_description="High-risk operation without dual control",
                    risk_category="governance"
                )
        
        return None
    
    async def _handle_sox_violation(self, violation: SaaSRuleViolation, context: Dict[str, Any]):
        """Handle SOX compliance violation"""
        logger.critical(f"SOX Violation: {violation.rule_name}")
        # In real implementation, trigger SOX-specific remediation
    
    async def _handle_gdpr_violation(self, violation: SaaSRuleViolation, context: Dict[str, Any]):
        """Handle GDPR compliance violation"""
        logger.critical(f"GDPR Violation: {violation.rule_name}")
        # In real implementation, trigger GDPR-specific remediation
    
    async def _handle_dsar_violation(self, violation: SaaSRuleViolation, context: Dict[str, Any]):
        """Handle DSAR compliance violation"""
        logger.critical(f"DSAR Violation: {violation.rule_name}")
        # In real implementation, trigger DSAR-specific remediation

class EvidenceCollector:
    """Collects and manages evidence for audit and compliance"""
    
    def __init__(self):
        self.schema_factory = SaaSSchemaFactory()
        self.evidence_store: Dict[str, SaaSEvidencePack] = {}
    
    async def collect_workflow_evidence(
        self,
        workflow_execution_id: str,
        tenant_id: str,
        workflow_type: str,
        execution_data: Dict[str, Any],
        compliance_frameworks: List[ComplianceFramework]
    ) -> SaaSEvidencePack:
        """Collect comprehensive evidence for a workflow execution"""
        
        evidence_pack = self.schema_factory.create_evidence_pack(
            tenant_id=tenant_id,
            workflow_type=workflow_type,
            workflow_execution_id=workflow_execution_id,
            pack_name=f"Evidence Pack - {workflow_execution_id}"
        )
        
        # Collect framework-specific evidence
        for framework in compliance_frameworks:
            if framework == ComplianceFramework.SOX:
                sox_evidence = await self._collect_sox_evidence(execution_data)
                evidence_pack.sox_evidence.extend(sox_evidence)
            elif framework == ComplianceFramework.GDPR:
                gdpr_evidence = await self._collect_gdpr_evidence(execution_data)
                evidence_pack.gdpr_evidence.extend(gdpr_evidence)
        
        # Update evidence pack metadata
        evidence_pack.evidence_count = (
            len(evidence_pack.sox_evidence) + 
            len(evidence_pack.gdpr_evidence) + 
            len(evidence_pack.general_evidence)
        )
        evidence_pack.collection_end = datetime.utcnow()
        evidence_pack.is_complete = True
        
        # Store evidence pack
        self.evidence_store[evidence_pack.evidence_pack_id] = evidence_pack
        
        logger.info(f"Collected evidence pack: {evidence_pack.evidence_pack_id} with {evidence_pack.evidence_count} items")
        
        return evidence_pack
    
    async def _collect_sox_evidence(self, execution_data: Dict[str, Any]) -> List[Any]:
        """Collect SOX-specific evidence"""
        # Mock SOX evidence collection
        return []
    
    async def _collect_gdpr_evidence(self, execution_data: Dict[str, Any]) -> List[Any]:
        """Collect GDPR-specific evidence"""
        # Mock GDPR evidence collection
        return []

class GovernanceOrchestrator:
    """Main orchestrator for all governance activities"""
    
    def __init__(self, config: GovernanceConfig):
        self.config = config
        self.trust_scorer = TrustScorer()
        self.approval_manager = ApprovalWorkflowManager()
        self.compliance_monitor = ComplianceMonitor()
        self.evidence_collector = EvidenceCollector()
        self.governance_events: List[GovernanceEvent] = []
    
    async def apply_governance_controls(
        self,
        workflow_execution_id: str,
        workflow_definition: Dict[str, Any],
        execution_context: Dict[str, Any],
        tenant_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Apply comprehensive governance controls to workflow execution"""
        
        governance_result = {
            "can_proceed": True,
            "governance_applied": [],
            "approvals_required": [],
            "compliance_status": {},
            "trust_scores": {},
            "evidence_collected": None,
            "governance_events": []
        }
        
        try:
            # 1. Trust Scoring
            if user_id:
                user_trust = await self.trust_scorer.get_trust_score(user_id)
                if not user_trust:
                    # Calculate initial trust score
                    user_trust = await self.trust_scorer.calculate_trust_score(
                        user_id, "user", {"approval_accuracy": 0.8, "tenure": 0.7}
                    )
                governance_result["trust_scores"]["user"] = user_trust.score
            
            # 2. Compliance Monitoring
            compliance_result = await self.compliance_monitor.monitor_workflow_compliance(
                workflow_execution_id, workflow_definition, execution_context
            )
            governance_result["compliance_status"] = compliance_result["compliance_status"]
            
            # If compliance violations, may need approval
            if not compliance_result["overall_compliant"]:
                if self.config.governance_level in [GovernanceLevel.STRICT, GovernanceLevel.REGULATORY]:
                    governance_result["can_proceed"] = False
                    
                    # Create approval request for compliance violation
                    approval_request = await self.approval_manager.create_approval_request(
                        workflow_execution_id=workflow_execution_id,
                        requested_by=user_id or "system",
                        request_type="compliance_exception",
                        request_reason="Compliance violations detected",
                        business_justification="Business critical workflow with compliance exceptions",
                        risk_assessment=RiskLevel.HIGH
                    )
                    governance_result["approvals_required"].append(approval_request.request_id)
            
            # 3. Evidence Collection
            if self.config.evidence_collection_required:
                evidence_pack = await self.evidence_collector.collect_workflow_evidence(
                    workflow_execution_id=workflow_execution_id,
                    tenant_id=tenant_id,
                    workflow_type=workflow_definition.get("workflow_type", "unknown"),
                    execution_data=execution_context,
                    compliance_frameworks=self.config.compliance_frameworks
                )
                governance_result["evidence_collected"] = evidence_pack.evidence_pack_id
            
            # 4. Record Governance Event
            governance_event = GovernanceEvent(
                event_type="governance_controls_applied",
                workflow_execution_id=workflow_execution_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_data={
                    "governance_level": self.config.governance_level.value,
                    "compliance_frameworks": [cf.value for cf in self.config.compliance_frameworks],
                    "controls_applied": governance_result["governance_applied"]
                },
                risk_level=RiskLevel.MEDIUM if not compliance_result["overall_compliant"] else RiskLevel.LOW,
                compliance_impact=self.config.compliance_frameworks
            )
            
            self.governance_events.append(governance_event)
            governance_result["governance_events"].append(governance_event.event_id)
            
            logger.info(f"Applied governance controls for {workflow_execution_id}: {governance_result['can_proceed']}")
            
        except Exception as e:
            logger.error(f"Error applying governance controls: {e}")
            governance_result["can_proceed"] = False
            governance_result["error"] = str(e)
        
        return governance_result
    
    async def get_governance_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get governance dashboard data"""
        
        # Calculate metrics
        total_events = len([e for e in self.governance_events if e.tenant_id == tenant_id])
        high_risk_events = len([e for e in self.governance_events if e.tenant_id == tenant_id and e.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
        pending_approvals = len(self.approval_manager.pending_requests)
        
        return {
            "tenant_id": tenant_id,
            "governance_level": self.config.governance_level.value,
            "compliance_frameworks": [cf.value for cf in self.config.compliance_frameworks],
            "metrics": {
                "total_governance_events": total_events,
                "high_risk_events": high_risk_events,
                "pending_approvals": pending_approvals,
                "compliance_score": 0.95,  # Mock calculation
                "governance_maturity": "enhanced"
            },
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "risk_level": event.risk_level.value,
                    "created_at": event.created_at.isoformat()
                }
                for event in sorted(self.governance_events, key=lambda x: x.created_at, reverse=True)[:10]
                if event.tenant_id == tenant_id
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_governance_layer():
        """Test governance layer functionality"""
        
        # Initialize governance config
        config = GovernanceConfig(
            governance_level=GovernanceLevel.ENHANCED,
            compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.GDPR],
            audit_retention_days=2555,
            immutable_logs=True,
            maker_checker_required=True,
            approval_timeout_hours=24,
            risk_threshold=RiskLevel.MEDIUM,
            evidence_collection_required=True
        )
        
        # Create governance orchestrator
        governance = GovernanceOrchestrator(config)
        
        # Test governance controls
        workflow_definition = {
            "workflow_id": "saas_pipeline_hygiene_v1.0",
            "compliance_frameworks": ["sox", "gdpr"],
            "workflow_type": "pipeline_hygiene"
        }
        
        execution_context = {
            "execution_id": "exec_123",
            "audit_trail_enabled": True,
            "contains_personal_data": True,
            "consent_verified": False,  # This will cause GDPR violation
            "high_risk_operation": True
        }
        
        result = await governance.apply_governance_controls(
            workflow_execution_id="exec_123",
            workflow_definition=workflow_definition,
            execution_context=execution_context,
            tenant_id="tenant_123",
            user_id="user_456"
        )
        
        print("Governance Result:")
        print(f"  Can Proceed: {result['can_proceed']}")
        print(f"  Compliance Status: {result['compliance_status']}")
        print(f"  Approvals Required: {len(result['approvals_required'])}")
        print(f"  Evidence Collected: {result['evidence_collected'] is not None}")
        
        # Test dashboard
        dashboard = await governance.get_governance_dashboard("tenant_123")
        print(f"\nDashboard Metrics:")
        print(f"  Governance Events: {dashboard['metrics']['total_governance_events']}")
        print(f"  High Risk Events: {dashboard['metrics']['high_risk_events']}")
        print(f"  Compliance Score: {dashboard['metrics']['compliance_score']}")
    
    # Run test
    asyncio.run(test_governance_layer())
