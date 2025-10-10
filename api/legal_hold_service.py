"""
Legal Hold & eDiscovery API - Task 6.1.33
=========================================
Immutable evidence export for litigation and legal holds
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import hashlib
import json

app = FastAPI(title="Legal Hold & eDiscovery Service")


class HoldStatus(str, Enum):
    ACTIVE = "active"
    RELEASED = "released"


class LegalHold(BaseModel):
    """Legal hold on evidence"""
    hold_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    case_id: str
    case_name: str
    
    # Scope
    workflow_ids: List[str] = Field(default_factory=list)
    date_range_start: datetime
    date_range_end: datetime
    
    # Status
    status: HoldStatus = HoldStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    
    # Chain of custody
    custody_chain: List[Dict[str, Any]] = Field(default_factory=list)


class eDiscoveryExport(BaseModel):
    """eDiscovery export package"""
    export_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hold_id: str
    tenant_id: str
    
    # Export contents
    evidence_records: List[Dict[str, Any]]
    execution_traces: List[Dict[str, Any]]
    override_records: List[Dict[str, Any]]
    
    # Integrity
    export_hash: str
    record_count: int
    
    # Metadata
    exported_at: datetime = Field(default_factory=datetime.utcnow)
    exported_by: str
    format: str = "json"


# Storage
legal_holds: Dict[str, LegalHold] = {}
ediscovery_exports: Dict[str, eDiscoveryExport] = {}


@app.post("/legal-holds", response_model=LegalHold)
async def create_legal_hold(hold: LegalHold):
    """Create a legal hold on evidence"""
    # Add initial custody entry
    hold.custody_chain.append({
        'action': 'hold_created',
        'timestamp': datetime.utcnow().isoformat(),
        'user': hold.created_by
    })
    
    legal_holds[hold.hold_id] = hold
    return hold


@app.post("/legal-holds/{hold_id}/export", response_model=eDiscoveryExport)
async def export_for_ediscovery(hold_id: str, exported_by: str):
    """Export evidence for eDiscovery"""
    if hold_id not in legal_holds:
        raise HTTPException(status_code=404, detail="Legal hold not found")
    
    hold = legal_holds[hold_id]
    
    # Collect evidence (simplified)
    evidence_records = []  # Would query evidence DB
    execution_traces = []  # Would query execution traces
    override_records = []  # Would query override ledger
    
    # Calculate integrity hash
    export_content = json.dumps({
        'evidence': evidence_records,
        'traces': execution_traces,
        'overrides': override_records
    }, sort_keys=True)
    export_hash = hashlib.sha256(export_content.encode()).hexdigest()
    
    # Create export
    export = eDiscoveryExport(
            hold_id=hold_id,
            tenant_id=hold.tenant_id,
        evidence_records=evidence_records,
        execution_traces=execution_traces,
        override_records=override_records,
        export_hash=export_hash,
        record_count=len(evidence_records) + len(execution_traces) + len(override_records),
        exported_by=exported_by
    )
    
    ediscovery_exports[export.export_id] = export
    
    # Update custody chain
    hold.custody_chain.append({
        'action': 'evidence_exported',
        'export_id': export.export_id,
        'timestamp': datetime.utcnow().isoformat(),
        'user': exported_by,
        'hash': export_hash
    })
    
    return export


@app.get("/legal-holds/{tenant_id}")
async def list_legal_holds(tenant_id: str):
    """List legal holds for a tenant"""
    tenant_holds = [h for h in legal_holds.values() if h.tenant_id == tenant_id]
    return {'holds': tenant_holds}
