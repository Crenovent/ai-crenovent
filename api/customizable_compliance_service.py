"""
Task 3.4.37: Implement customizable compliance overlays
- Policy engine extension for tenant-defined policies
- Dynamic policy validation and enforcement
- Policy template system with inheritance
- Database schema for custom policy storage
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import json

app = FastAPI(title="RBIA Customizable Compliance Overlays")

class PolicyType(str, Enum):
    CUSTOM = "custom"
    TEMPLATE = "template"
    INHERITED = "inherited"

class ComplianceFramework(str, Enum):
    SOX = "SOX"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    RBI = "RBI"
    CUSTOM = "CUSTOM"

class CustomComplianceOverlay(BaseModel):
    overlay_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    overlay_name: str
    framework_base: ComplianceFramework
    
    # Policy definitions
    custom_policies: Dict[str, Any] = Field(default_factory=dict)
    policy_rules: List[Dict[str, Any]] = Field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Template inheritance
    parent_template_id: Optional[str] = None
    inherited_policies: Dict[str, Any] = Field(default_factory=dict)
    
    # Enforcement settings
    enforcement_level: str = "strict"  # strict, warn, log
    auto_remediation: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str

# Storage
overlays_store: Dict[str, CustomComplianceOverlay] = {}

@app.post("/compliance/overlays", response_model=CustomComplianceOverlay)
async def create_custom_overlay(overlay: CustomComplianceOverlay):
    """Create custom compliance overlay"""
    overlays_store[overlay.overlay_id] = overlay
    return overlay

@app.get("/compliance/overlays/{tenant_id}", response_model=List[CustomComplianceOverlay])
async def get_tenant_overlays(tenant_id: str):
    """Get all overlays for tenant"""
    return [o for o in overlays_store.values() if o.tenant_id == tenant_id]

@app.post("/compliance/validate")
async def validate_against_overlay(
    tenant_id: str,
    overlay_id: str,
    workflow_data: Dict[str, Any]
):
    """Validate workflow against custom overlay"""
    if overlay_id not in overlays_store:
        raise HTTPException(status_code=404, detail="Overlay not found")
    
    overlay = overlays_store[overlay_id]
    violations = []
    
    # Simple validation logic
    for rule in overlay.validation_rules:
        if not _evaluate_rule(rule, workflow_data):
            violations.append(rule.get("violation_message", "Policy violation"))
    
    return {
        "overlay_id": overlay_id,
        "compliant": len(violations) == 0,
        "violations": violations,
        "enforcement_level": overlay.enforcement_level
    }

def _evaluate_rule(rule: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Simple rule evaluation"""
    field = rule.get("field")
    operator = rule.get("operator", "equals")
    expected = rule.get("value")
    
    if field not in data:
        return False
    
    actual = data[field]
    
    if operator == "equals":
        return actual == expected
    elif operator == "greater_than":
        return actual > expected
    elif operator == "contains":
        return expected in str(actual)
    
    return True

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Customizable Compliance Overlays", "task": "3.4.37"}
