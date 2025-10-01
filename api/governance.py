"""
Governance API
FastAPI endpoints for audit trails, compliance, and governance features
"""

from fastapi import APIRouter
from typing import Dict, List, Any
from datetime import datetime

# Create governance router
router = APIRouter(prefix="/api/builder", tags=["Governance"])


@router.get("/governance/audit-trail")
async def get_audit_trail(tenant_id: str, limit: int = 100):
    """Get governance audit trail"""
    # Mock audit trail data
    audit_entries = [
        {
            "id": "audit_001",
            "timestamp": datetime.now().isoformat(),
            "event_type": "WORKFLOW_EXECUTION",
            "user_id": "1319",
            "tenant_id": tenant_id,
            "workflow_name": "Pipeline Hygiene Check",
            "action": "EXECUTED",
            "compliance_status": "COMPLIANT",
            "evidence_pack_id": "EP_001"
        },
        {
            "id": "audit_002",
            "timestamp": (datetime.now()).isoformat(),
            "event_type": "CONFIGURATION_CHANGE",
            "user_id": "1319",
            "tenant_id": tenant_id,
            "workflow_name": "Forecast Approval",
            "action": "MODIFIED",
            "compliance_status": "COMPLIANT",
            "changes": ["variance_threshold: 10% -> 15%"]
        }
    ]
    
    return {
        "audit_entries": audit_entries[:limit],
        "total_count": len(audit_entries),
        "compliance_summary": {
            "total_events": len(audit_entries),
            "compliant_events": len(audit_entries),
            "non_compliant_events": 0,
            "compliance_percentage": 100.0
        }
    }
