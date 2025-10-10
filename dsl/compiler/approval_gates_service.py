"""
Approval Gates - Task 6.2.61
=============================

Governance approval gates with CAB sign-off references
- Embeds approval requirements in IR
- Validates in CI
- Tracks approval status and references
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ApprovalType(Enum):
    CAB_APPROVAL = "cab_approval"           # Change Advisory Board
    SECURITY_APPROVAL = "security_approval"
    COMPLIANCE_APPROVAL = "compliance_approval"
    BUSINESS_APPROVAL = "business_approval"
    TECHNICAL_APPROVAL = "technical_approval"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ApprovalGate:
    gate_id: str
    approval_type: ApprovalType
    required_approvers: List[str]
    approval_criteria: List[str]
    timeout_hours: int
    blocking: bool  # Whether this blocks execution

@dataclass
class ApprovalRecord:
    approval_id: str
    gate_id: str
    approver_id: str
    status: ApprovalStatus
    approved_at: Optional[datetime]
    comments: str
    approval_reference: str  # Reference to approval system

class ApprovalGatesService:
    """Service for managing approval gates in plans"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.approval_gates_config = self._load_approval_gates_config()
    
    def _load_approval_gates_config(self) -> Dict[str, ApprovalGate]:
        """Load approval gates configuration"""
        gates = [
            ApprovalGate(
                gate_id="ml_model_deployment",
                approval_type=ApprovalType.CAB_APPROVAL,
                required_approvers=["ml_architect", "security_lead"],
                approval_criteria=[
                    "Model performance meets SLA requirements",
                    "Security review completed",
                    "Bias testing passed"
                ],
                timeout_hours=72,
                blocking=True
            ),
            ApprovalGate(
                gate_id="high_risk_workflow",
                approval_type=ApprovalType.BUSINESS_APPROVAL,
                required_approvers=["business_owner", "compliance_officer"],
                approval_criteria=[
                    "Business impact assessed",
                    "Compliance requirements verified",
                    "Risk mitigation plan approved"
                ],
                timeout_hours=48,
                blocking=True
            ),
            ApprovalGate(
                gate_id="external_integration",
                approval_type=ApprovalType.SECURITY_APPROVAL,
                required_approvers=["security_architect"],
                approval_criteria=[
                    "Security assessment completed",
                    "Data flow reviewed",
                    "Access controls validated"
                ],
                timeout_hours=24,
                blocking=True
            )
        ]
        
        return {gate.gate_id: gate for gate in gates}
    
    def analyze_approval_requirements(self, ir_data: Dict[str, Any]) -> List[ApprovalGate]:
        """Analyze IR and determine required approval gates"""
        required_gates = []
        nodes = ir_data.get('nodes', [])
        
        has_ml_nodes = any('ml_' in node.get('type', '') for node in nodes)
        has_external_calls = any('external' in node.get('type', '') for node in nodes)
        has_high_risk = any(
            node.get('policies', {}).get('risk_level') == 'high' 
            for node in nodes
        )
        
        # Determine required gates based on plan characteristics
        if has_ml_nodes:
            required_gates.append(self.approval_gates_config["ml_model_deployment"])
        
        if has_external_calls:
            required_gates.append(self.approval_gates_config["external_integration"])
        
        if has_high_risk:
            required_gates.append(self.approval_gates_config["high_risk_workflow"])
        
        return required_gates
    
    def embed_approval_gates(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Embed approval gate requirements into IR"""
        required_gates = self.analyze_approval_requirements(ir_data)
        
        # Add approval gates to plan metadata
        if 'metadata' not in ir_data:
            ir_data['metadata'] = {}
        
        ir_data['metadata']['approval_gates'] = [
            {
                'gate_id': gate.gate_id,
                'approval_type': gate.approval_type.value,
                'required_approvers': gate.required_approvers,
                'approval_criteria': gate.approval_criteria,
                'timeout_hours': gate.timeout_hours,
                'blocking': gate.blocking,
                'status': ApprovalStatus.PENDING.value
            }
            for gate in required_gates
        ]
        
        # Add approval requirements to nodes that trigger gates
        nodes = ir_data.get('nodes', [])
        for node in nodes:
            node_gates = []
            
            # ML nodes require ML approval
            if 'ml_' in node.get('type', ''):
                node_gates.append('ml_model_deployment')
            
            # External nodes require security approval
            if 'external' in node.get('type', ''):
                node_gates.append('external_integration')
            
            # High-risk nodes require business approval
            if node.get('policies', {}).get('risk_level') == 'high':
                node_gates.append('high_risk_workflow')
            
            if node_gates:
                node['approval_gates_required'] = node_gates
        
        return ir_data
    
    def validate_approvals(self, ir_data: Dict[str, Any]) -> List[str]:
        """Validate approval status for CI integration"""
        issues = []
        approval_gates = ir_data.get('metadata', {}).get('approval_gates', [])
        
        for gate in approval_gates:
            gate_id = gate.get('gate_id')
            status = gate.get('status', 'pending')
            blocking = gate.get('blocking', True)
            
            if blocking and status != 'approved':
                issues.append(f"Blocking approval gate '{gate_id}' not approved (status: {status})")
            
            # Check for expired approvals
            if status == 'expired':
                issues.append(f"Approval gate '{gate_id}' has expired - re-approval required")
        
        return issues
    
    def create_approval_record(self, gate_id: str, approver_id: str, 
                             status: ApprovalStatus, comments: str = "",
                             approval_reference: str = "") -> ApprovalRecord:
        """Create approval record"""
        return ApprovalRecord(
            approval_id=f"approval_{gate_id}_{approver_id}_{int(datetime.now().timestamp())}",
            gate_id=gate_id,
            approver_id=approver_id,
            status=status,
            approved_at=datetime.now(timezone.utc) if status == ApprovalStatus.APPROVED else None,
            comments=comments,
            approval_reference=approval_reference
        )
    
    def update_approval_status(self, ir_data: Dict[str, Any], gate_id: str, 
                             approval_record: ApprovalRecord) -> Dict[str, Any]:
        """Update approval status in IR"""
        approval_gates = ir_data.get('metadata', {}).get('approval_gates', [])
        
        for gate in approval_gates:
            if gate.get('gate_id') == gate_id:
                gate['status'] = approval_record.status.value
                gate['last_updated'] = datetime.now(timezone.utc).isoformat()
                gate['approver_id'] = approval_record.approver_id
                gate['approval_reference'] = approval_record.approval_reference
                gate['comments'] = approval_record.comments
                break
        
        return ir_data
    
    def get_approval_summary(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get approval summary for the plan"""
        approval_gates = ir_data.get('metadata', {}).get('approval_gates', [])
        
        total_gates = len(approval_gates)
        approved_gates = len([g for g in approval_gates if g.get('status') == 'approved'])
        pending_gates = len([g for g in approval_gates if g.get('status') == 'pending'])
        blocking_pending = len([
            g for g in approval_gates 
            if g.get('status') == 'pending' and g.get('blocking', True)
        ])
        
        return {
            'total_gates': total_gates,
            'approved_gates': approved_gates,
            'pending_gates': pending_gates,
            'blocking_pending': blocking_pending,
            'approval_complete': blocking_pending == 0,
            'approval_percentage': (approved_gates / total_gates * 100) if total_gates > 0 else 100
        }

