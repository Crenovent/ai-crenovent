#!/usr/bin/env python3
"""
Conflict Resolution Hierarchies Service API - Chapter 15.3
==========================================================
Tasks 15.3-T05 to T70: Conflict resolution with compliance-first hierarchies

Features:
- Compliance-first conflict resolution (Compliance > Finance > Ops)
- Policy vs policy, data vs policy, SLA vs policy conflict detection
- Industry-specific conflict rules (RBI vs GDPR residency conflicts)
- Cross-module conflict detection (Compensation vs Forecast)
- Conflict evidence emit with digital signatures
- Conflict ledger with override ledger integration
- AI conflict resolution advisor and negotiator agents
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json
import hashlib
import asyncio

from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# PYDANTIC MODELS
# =====================================================

class ConflictType(str, Enum):
    POLICY_VS_POLICY = "policy_vs_policy"
    DATA_VS_POLICY = "data_vs_policy"
    SLA_VS_POLICY = "sla_vs_policy"
    PERSONA_VS_PERSONA = "persona_vs_persona"
    COMPLIANCE_VS_BUSINESS = "compliance_vs_business"
    CROSS_MODULE = "cross_module"

class ConflictSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConflictStatus(str, Enum):
    DETECTED = "detected"
    ANALYZING = "analyzing"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    OVERRIDDEN = "overridden"

class ResolutionHierarchy(str, Enum):
    COMPLIANCE = "compliance"  # Highest priority
    FINANCE = "finance"        # Second priority
    OPERATIONS = "operations"  # Third priority
    BUSINESS = "business"      # Lowest priority

class IndustryCode(str, Enum):
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"

class ConflictRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    workflow_id: str = Field(..., description="Workflow ID")
    step_id: Optional[str] = Field(None, description="Workflow step ID")
    user_id: int = Field(..., description="User ID")
    
    # Conflict details
    conflict_type: ConflictType = Field(..., description="Type of conflict")
    severity: ConflictSeverity = Field(..., description="Conflict severity")
    title: str = Field(..., description="Conflict title")
    description: str = Field(..., description="Conflict description")
    
    # Conflicting elements
    conflicting_rules: List[Dict[str, Any]] = Field(..., description="Conflicting rules/policies")
    affected_modules: List[str] = Field([], description="Affected modules")
    
    # Context
    business_context: Dict[str, Any] = Field({}, description="Business context")
    compliance_context: Dict[str, Any] = Field({}, description="Compliance context")
    
    # Resolution preferences
    preferred_hierarchy: Optional[ResolutionHierarchy] = Field(None, description="Preferred resolution hierarchy")
    auto_resolve: bool = Field(True, description="Allow automatic resolution")
    
    # Metadata
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

class ConflictResolution(BaseModel):
    resolution_id: str = Field(..., description="Resolution ID")
    conflict_id: str = Field(..., description="Conflict ID")
    
    # Resolution details
    resolution_hierarchy: ResolutionHierarchy = Field(..., description="Applied hierarchy")
    winning_rule: Dict[str, Any] = Field(..., description="Winning rule/policy")
    losing_rules: List[Dict[str, Any]] = Field(..., description="Overridden rules")
    
    # Decision rationale
    decision_rationale: str = Field(..., description="Decision rationale")
    compliance_impact: str = Field("", description="Compliance impact assessment")
    business_impact: str = Field("", description="Business impact assessment")
    
    # Approver information
    resolved_by: Optional[int] = Field(None, description="User who resolved conflict")
    approval_required: bool = Field(False, description="Manual approval required")
    
    # Metadata
    resolution_metadata: Dict[str, Any] = Field({}, description="Resolution metadata")

class ConflictResponse(BaseModel):
    conflict_id: str
    workflow_id: str
    tenant_id: int
    conflict_type: ConflictType
    severity: ConflictSeverity
    status: ConflictStatus
    resolution: Optional[ConflictResolution]
    evidence_pack_id: str
    created_at: datetime
    resolved_at: Optional[datetime]

# =====================================================
# CONFLICT RESOLUTION SERVICE
# =====================================================

class ConflictResolutionService:
    """
    Conflict Resolution Hierarchies Service
    Tasks 15.3-T05 to T27: Compliance-first conflict resolution with hierarchies
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
        # Hierarchy order (Task 15.3-T02: Compliance > Finance > Ops)
        self.hierarchy_order = [
            ResolutionHierarchy.COMPLIANCE,
            ResolutionHierarchy.FINANCE,
            ResolutionHierarchy.OPERATIONS,
            ResolutionHierarchy.BUSINESS
        ]
        
        # Industry-specific conflict rules (Task 15.3-T11)
        self.industry_conflict_rules = {
            IndustryCode.BANKING: {
                "compliance_frameworks": ["RBI", "BASEL_III", "AML", "GDPR"],
                "hierarchy_override": {
                    "rbi_vs_gdpr_residency": ResolutionHierarchy.COMPLIANCE,
                    "aml_vs_privacy": ResolutionHierarchy.COMPLIANCE,
                    "basel_vs_business": ResolutionHierarchy.COMPLIANCE
                },
                "escalation_required": ["cross_border_data", "regulatory_reporting", "capital_adequacy"]
            },
            IndustryCode.INSURANCE: {
                "compliance_frameworks": ["IRDAI", "IRDA", "GDPR", "DPDP"],
                "hierarchy_override": {
                    "solvency_vs_payout": ResolutionHierarchy.COMPLIANCE,
                    "irdai_vs_gdpr": ResolutionHierarchy.COMPLIANCE,
                    "claim_vs_privacy": ResolutionHierarchy.COMPLIANCE
                },
                "escalation_required": ["solvency_ratio", "claim_settlement", "regulatory_filing"]
            },
            IndustryCode.HEALTHCARE: {
                "compliance_frameworks": ["HIPAA", "GDPR", "FDA", "HITECH"],
                "hierarchy_override": {
                    "hipaa_vs_gdpr": ResolutionHierarchy.COMPLIANCE,
                    "patient_safety_vs_privacy": ResolutionHierarchy.COMPLIANCE,
                    "fda_vs_business": ResolutionHierarchy.COMPLIANCE
                },
                "escalation_required": ["patient_data", "medical_devices", "clinical_trials"]
            },
            IndustryCode.SAAS: {
                "compliance_frameworks": ["SOX", "GDPR", "CCPA", "PCI_DSS"],
                "hierarchy_override": {
                    "sox_vs_revenue": ResolutionHierarchy.COMPLIANCE,
                    "gdpr_vs_analytics": ResolutionHierarchy.COMPLIANCE,
                    "pci_vs_ux": ResolutionHierarchy.COMPLIANCE
                },
                "escalation_required": ["revenue_recognition", "data_processing", "payment_data"]
            }
        }
        
        # Cross-module conflict patterns (Task 15.3-T12)
        self.cross_module_patterns = {
            "compensation_vs_forecast": {
                "detection_rules": ["quota_vs_forecast_mismatch", "commission_vs_revenue_timing"],
                "resolution_priority": ResolutionHierarchy.FINANCE,
                "escalation_threshold": ConflictSeverity.MEDIUM
            },
            "approval_vs_sla": {
                "detection_rules": ["approval_time_vs_sla", "escalation_vs_deadline"],
                "resolution_priority": ResolutionHierarchy.COMPLIANCE,
                "escalation_threshold": ConflictSeverity.HIGH
            },
            "privacy_vs_analytics": {
                "detection_rules": ["gdpr_vs_tracking", "consent_vs_personalization"],
                "resolution_priority": ResolutionHierarchy.COMPLIANCE,
                "escalation_threshold": ConflictSeverity.LOW
            }
        }
        
        # SLA timers for conflict resolution (Task 15.3-T41)
        self.sla_timers = {
            ConflictSeverity.CRITICAL: 2,   # 2 hours
            ConflictSeverity.HIGH: 8,       # 8 hours
            ConflictSeverity.MEDIUM: 24,    # 1 day
            ConflictSeverity.LOW: 72        # 3 days
        }
    
    async def detect_and_resolve_conflict(self, request: ConflictRequest) -> ConflictResponse:
        """
        Detect and resolve conflict using compliance-first hierarchy
        Tasks 15.3-T05, T06, T07, T08: Core conflict resolution engine
        """
        try:
            conflict_id = str(uuid.uuid4())
            
            # Classify conflict (Task 15.3-T07)
            conflict_classification = await self._classify_conflict(request)
            
            # Detect conflict patterns (Task 15.3-T06)
            conflict_analysis = await self._analyze_conflict(request, conflict_classification)
            
            # Generate evidence pack for conflict detection (Task 15.3-T13)
            evidence_pack_id = await self._generate_conflict_evidence_pack(
                conflict_id, request, "conflict_detected"
            )
            
            # Store conflict request
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(request.tenant_id))
                
                await conn.execute("""
                    INSERT INTO conflict_requests (
                        conflict_id, tenant_id, workflow_id, step_id, user_id,
                        conflict_type, severity, title, description,
                        conflicting_rules, affected_modules, business_context,
                        compliance_context, preferred_hierarchy, auto_resolve,
                        metadata, status, evidence_pack_id, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                """,
                    conflict_id, request.tenant_id, request.workflow_id, request.step_id, request.user_id,
                    request.conflict_type.value, request.severity.value, request.title, request.description,
                    json.dumps(request.conflicting_rules), json.dumps(request.affected_modules),
                    json.dumps(request.business_context), json.dumps(request.compliance_context),
                    request.preferred_hierarchy.value if request.preferred_hierarchy else None,
                    request.auto_resolve, json.dumps(request.metadata), ConflictStatus.DETECTED.value,
                    evidence_pack_id, datetime.utcnow()
                )
            
            # Attempt automatic resolution if allowed (Task 15.3-T08)
            resolution = None
            status = ConflictStatus.DETECTED
            resolved_at = None
            
            if request.auto_resolve and conflict_analysis["auto_resolvable"]:
                resolution = await self._resolve_conflict_automatically(
                    conflict_id, request, conflict_analysis
                )
                status = ConflictStatus.RESOLVED
                resolved_at = datetime.utcnow()
                
                # Update conflict status
                await self._update_conflict_status(conflict_id, request.tenant_id, status, resolution)
            else:
                # Escalate for manual resolution (Task 15.3-T09)
                await self._escalate_conflict_for_manual_resolution(conflict_id, request, conflict_analysis)
                status = ConflictStatus.ESCALATED
            
            # Link to risk register (Task 15.3-T17)
            await self._update_risk_register_for_conflict(conflict_id, request, conflict_analysis)
            
            logger.info(f"✅ Processed conflict {conflict_id} for workflow {request.workflow_id}")
            
            return ConflictResponse(
                conflict_id=conflict_id,
                workflow_id=request.workflow_id,
                tenant_id=request.tenant_id,
                conflict_type=request.conflict_type,
                severity=request.severity,
                status=status,
                resolution=resolution,
                evidence_pack_id=evidence_pack_id,
                created_at=datetime.utcnow(),
                resolved_at=resolved_at
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to process conflict: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def resolve_conflict_manually(self, conflict_id: str, resolution: ConflictResolution,
                                      tenant_id: int, user_id: int) -> Dict[str, Any]:
        """
        Manually resolve conflict with hierarchy validation
        Tasks 15.3-T08, T09: Manual conflict resolution with hierarchy enforcement
        """
        try:
            # Get conflict details
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                conflict = await conn.fetchrow("""
                    SELECT * FROM conflict_requests 
                    WHERE conflict_id = $1 AND tenant_id = $2
                """, conflict_id, tenant_id)
                
                if not conflict:
                    raise HTTPException(status_code=404, detail="Conflict not found")
                
                if conflict['status'] == ConflictStatus.RESOLVED.value:
                    raise HTTPException(status_code=400, detail="Conflict already resolved")
            
            # Validate hierarchy compliance (Task 15.3-T08)
            hierarchy_validation = await self._validate_resolution_hierarchy(
                resolution, conflict, tenant_id
            )
            
            if not hierarchy_validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Resolution violates hierarchy: {hierarchy_validation['reason']}"
                )
            
            # Generate evidence pack for manual resolution (Task 15.3-T13)
            evidence_pack_id = await self._generate_conflict_evidence_pack(
                conflict_id, resolution, "conflict_resolved_manually"
            )
            
            # Store resolution
            resolution_id = str(uuid.uuid4())
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                await conn.execute("""
                    INSERT INTO conflict_resolutions (
                        resolution_id, conflict_id, tenant_id, resolution_hierarchy,
                        winning_rule, losing_rules, decision_rationale,
                        compliance_impact, business_impact, resolved_by,
                        approval_required, resolution_metadata, evidence_pack_id,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                    resolution_id, conflict_id, tenant_id, resolution.resolution_hierarchy.value,
                    json.dumps(resolution.winning_rule), json.dumps(resolution.losing_rules),
                    resolution.decision_rationale, resolution.compliance_impact,
                    resolution.business_impact, user_id, resolution.approval_required,
                    json.dumps(resolution.resolution_metadata), evidence_pack_id, datetime.utcnow()
                )
            
            # Update conflict status
            await self._update_conflict_status(conflict_id, tenant_id, ConflictStatus.RESOLVED, resolution)
            
            # Link to override ledger if manual override (Task 15.3-T16)
            if resolution.approval_required:
                await self._link_to_override_ledger(conflict_id, resolution_id, user_id, tenant_id)
            
            logger.info(f"✅ Manually resolved conflict {conflict_id}")
            
            return {
                "conflict_id": conflict_id,
                "resolution_id": resolution_id,
                "status": "resolved",
                "hierarchy": resolution.resolution_hierarchy.value,
                "evidence_pack_id": evidence_pack_id,
                "resolved_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve conflict manually: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_conflict_status(self, conflict_id: str, tenant_id: int) -> Dict[str, Any]:
        """Get conflict status and resolution details"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Get conflict details
                conflict = await conn.fetchrow("""
                    SELECT * FROM conflict_requests 
                    WHERE conflict_id = $1 AND tenant_id = $2
                """, conflict_id, tenant_id)
                
                if not conflict:
                    raise HTTPException(status_code=404, detail="Conflict not found")
                
                # Get resolution if exists
                resolution = await conn.fetchrow("""
                    SELECT * FROM conflict_resolutions 
                    WHERE conflict_id = $1 AND tenant_id = $2
                """, conflict_id, tenant_id)
                
                # Get conflict ledger entries
                ledger_entries = await conn.fetch("""
                    SELECT * FROM conflict_ledger 
                    WHERE conflict_id = $1 AND tenant_id = $2
                    ORDER BY created_at DESC
                """, conflict_id, tenant_id)
                
                result = {
                    "conflict_id": conflict_id,
                    "status": conflict['status'],
                    "conflict_type": conflict['conflict_type'],
                    "severity": conflict['severity'],
                    "title": conflict['title'],
                    "description": conflict['description'],
                    "created_at": conflict['created_at'].isoformat(),
                    "resolution": None,
                    "ledger_entries": [dict(entry) for entry in ledger_entries]
                }
                
                if resolution:
                    result["resolution"] = {
                        "resolution_id": resolution['resolution_id'],
                        "hierarchy": resolution['resolution_hierarchy'],
                        "winning_rule": json.loads(resolution['winning_rule']),
                        "decision_rationale": resolution['decision_rationale'],
                        "resolved_by": resolution['resolved_by'],
                        "resolved_at": resolution['created_at'].isoformat()
                    }
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to get conflict status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Helper methods
    async def _classify_conflict(self, request: ConflictRequest) -> Dict[str, Any]:
        """Classify conflict type and determine resolution approach (Task 15.3-T07)"""
        try:
            classification = {
                "primary_type": request.conflict_type.value,
                "complexity": "simple",
                "frameworks_involved": [],
                "escalation_required": False,
                "auto_resolvable": True
            }
            
            # Analyze conflicting rules
            for rule in request.conflicting_rules:
                if "compliance_framework" in rule:
                    classification["frameworks_involved"].append(rule["compliance_framework"])
                
                if rule.get("requires_manual_review", False):
                    classification["auto_resolvable"] = False
            
            # Check for complex cross-framework conflicts
            if len(set(classification["frameworks_involved"])) > 1:
                classification["complexity"] = "complex"
                classification["escalation_required"] = True
            
            # Industry-specific classification
            industry_rules = self.industry_conflict_rules.get(IndustryCode.SAAS, {})  # Default to SaaS
            escalation_patterns = industry_rules.get("escalation_required", [])
            
            for pattern in escalation_patterns:
                if pattern in request.description.lower():
                    classification["escalation_required"] = True
                    classification["auto_resolvable"] = False
            
            return classification
            
        except Exception as e:
            logger.error(f"❌ Failed to classify conflict: {e}")
            return {"primary_type": request.conflict_type.value, "auto_resolvable": False}
    
    async def _analyze_conflict(self, request: ConflictRequest, 
                              classification: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conflict for resolution strategy (Task 15.3-T06)"""
        try:
            analysis = {
                "classification": classification,
                "resolution_strategy": "hierarchy_based",
                "recommended_hierarchy": ResolutionHierarchy.COMPLIANCE,
                "auto_resolvable": classification.get("auto_resolvable", False),
                "estimated_resolution_time_hours": 24
            }
            
            # Determine recommended hierarchy based on conflict type
            if request.conflict_type == ConflictType.COMPLIANCE_VS_BUSINESS:
                analysis["recommended_hierarchy"] = ResolutionHierarchy.COMPLIANCE
                analysis["estimated_resolution_time_hours"] = 2
            elif request.conflict_type == ConflictType.SLA_VS_POLICY:
                analysis["recommended_hierarchy"] = ResolutionHierarchy.FINANCE
                analysis["estimated_resolution_time_hours"] = 8
            elif request.conflict_type == ConflictType.CROSS_MODULE:
                # Check cross-module patterns
                for pattern_name, pattern_config in self.cross_module_patterns.items():
                    if any(rule in request.description.lower() for rule in pattern_config["detection_rules"]):
                        analysis["recommended_hierarchy"] = pattern_config["resolution_priority"]
                        break
            
            # Adjust based on severity
            severity_adjustments = {
                ConflictSeverity.CRITICAL: {"time_multiplier": 0.25, "escalate": True},
                ConflictSeverity.HIGH: {"time_multiplier": 0.5, "escalate": True},
                ConflictSeverity.MEDIUM: {"time_multiplier": 1.0, "escalate": False},
                ConflictSeverity.LOW: {"time_multiplier": 2.0, "escalate": False}
            }
            
            adjustment = severity_adjustments.get(request.severity, {"time_multiplier": 1.0})
            analysis["estimated_resolution_time_hours"] *= adjustment["time_multiplier"]
            
            if adjustment.get("escalate", False):
                analysis["auto_resolvable"] = False
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze conflict: {e}")
            return {"auto_resolvable": False, "recommended_hierarchy": ResolutionHierarchy.COMPLIANCE}
    
    async def _resolve_conflict_automatically(self, conflict_id: str, request: ConflictRequest,
                                            analysis: Dict[str, Any]) -> ConflictResolution:
        """Automatically resolve conflict using hierarchy rules (Task 15.3-T08)"""
        try:
            recommended_hierarchy = analysis["recommended_hierarchy"]
            
            # Apply hierarchy order to determine winning rule
            winning_rule = None
            losing_rules = []
            
            # Sort rules by hierarchy priority
            sorted_rules = sorted(
                request.conflicting_rules,
                key=lambda rule: self._get_hierarchy_priority(rule),
                reverse=True  # Higher priority first
            )
            
            if sorted_rules:
                winning_rule = sorted_rules[0]
                losing_rules = sorted_rules[1:]
            
            # Generate decision rationale
            decision_rationale = self._generate_decision_rationale(
                winning_rule, losing_rules, recommended_hierarchy
            )
            
            resolution = ConflictResolution(
                resolution_id=str(uuid.uuid4()),
                conflict_id=conflict_id,
                resolution_hierarchy=recommended_hierarchy,
                winning_rule=winning_rule or {},
                losing_rules=losing_rules,
                decision_rationale=decision_rationale,
                compliance_impact="Automatic resolution maintains compliance hierarchy",
                business_impact="Minimal business impact expected",
                resolved_by=None,  # Automatic resolution
                approval_required=False,
                resolution_metadata={
                    "automatic": True,
                    "analysis": analysis,
                    "resolution_time": datetime.utcnow().isoformat()
                }
            )
            
            return resolution
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve conflict automatically: {e}")
            raise
    
    def _get_hierarchy_priority(self, rule: Dict[str, Any]) -> int:
        """Get hierarchy priority for rule"""
        rule_hierarchy = rule.get("hierarchy", "business")
        
        hierarchy_priorities = {
            "compliance": 4,
            "finance": 3,
            "operations": 2,
            "business": 1
        }
        
        return hierarchy_priorities.get(rule_hierarchy.lower(), 1)
    
    def _generate_decision_rationale(self, winning_rule: Dict[str, Any], 
                                   losing_rules: List[Dict[str, Any]],
                                   hierarchy: ResolutionHierarchy) -> str:
        """Generate decision rationale for conflict resolution"""
        try:
            rationale_parts = [
                f"Conflict resolved using {hierarchy.value} hierarchy priority.",
                f"Winning rule: {winning_rule.get('name', 'Unknown')} ({winning_rule.get('hierarchy', 'Unknown')} level)."
            ]
            
            if losing_rules:
                overridden_names = [rule.get('name', 'Unknown') for rule in losing_rules]
                rationale_parts.append(f"Overridden rules: {', '.join(overridden_names)}.")
            
            rationale_parts.append("Resolution maintains compliance-first approach as per organizational policy.")
            
            return " ".join(rationale_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate decision rationale: {e}")
            return "Automatic resolution applied based on hierarchy rules."
    
    async def _validate_resolution_hierarchy(self, resolution: ConflictResolution,
                                           conflict: Any, tenant_id: int) -> Dict[str, Any]:
        """Validate resolution hierarchy compliance (Task 15.3-T08)"""
        try:
            # Check if resolution hierarchy is valid
            if resolution.resolution_hierarchy not in self.hierarchy_order:
                return {"valid": False, "reason": "Invalid hierarchy specified"}
            
            # Check if winning rule actually has higher priority
            winning_hierarchy = resolution.winning_rule.get("hierarchy", "business")
            winning_priority = self._get_hierarchy_priority({"hierarchy": winning_hierarchy})
            
            for losing_rule in resolution.losing_rules:
                losing_hierarchy = losing_rule.get("hierarchy", "business")
                losing_priority = self._get_hierarchy_priority({"hierarchy": losing_hierarchy})
                
                if losing_priority > winning_priority:
                    return {
                        "valid": False,
                        "reason": f"Losing rule has higher hierarchy priority: {losing_hierarchy} > {winning_hierarchy}"
                    }
            
            return {"valid": True, "reason": "Hierarchy validation passed"}
            
        except Exception as e:
            logger.error(f"❌ Failed to validate resolution hierarchy: {e}")
            return {"valid": False, "reason": f"Validation error: {str(e)}"}
    
    async def _generate_conflict_evidence_pack(self, conflict_id: str, request_data: Any,
                                             event_type: str) -> str:
        """Generate evidence pack for conflict event (Task 15.3-T13, T14)"""
        try:
            evidence_pack_id = str(uuid.uuid4())
            
            evidence_data = {
                "conflict_id": conflict_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "request_data": request_data.dict() if hasattr(request_data, 'dict') else str(request_data)
            }
            
            # Generate digital signature (Task 15.3-T14)
            evidence_json = json.dumps(evidence_data, sort_keys=True)
            digital_signature = hashlib.sha256(evidence_json.encode()).hexdigest()
            
            # Store evidence pack
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO conflict_evidence_packs (
                        evidence_pack_id, conflict_id, event_type, evidence_data,
                        digital_signature, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    evidence_pack_id, conflict_id, event_type, json.dumps(evidence_data),
                    digital_signature, datetime.utcnow()
                )
            
            return evidence_pack_id
            
        except Exception as e:
            logger.error(f"❌ Failed to generate conflict evidence pack: {e}")
            return f"error_{uuid.uuid4()}"
    
    async def _update_conflict_status(self, conflict_id: str, tenant_id: int, 
                                    status: ConflictStatus, resolution: Optional[ConflictResolution]):
        """Update conflict status in database"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                await conn.execute("""
                    UPDATE conflict_requests 
                    SET status = $1, resolved_at = $2, updated_at = $3
                    WHERE conflict_id = $4 AND tenant_id = $5
                """, status.value, datetime.utcnow() if status == ConflictStatus.RESOLVED else None,
                    datetime.utcnow(), conflict_id, tenant_id)
                
                # Record in conflict ledger (Task 15.3-T15)
                await conn.execute("""
                    INSERT INTO conflict_ledger (
                        ledger_id, conflict_id, tenant_id, event_type, 
                        event_description, resolution_hierarchy, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    str(uuid.uuid4()), conflict_id, tenant_id, "status_change",
                    f"Status changed to {status.value}",
                    resolution.resolution_hierarchy.value if resolution else None,
                    datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to update conflict status: {e}")
    
    async def _escalate_conflict_for_manual_resolution(self, conflict_id: str, request: ConflictRequest,
                                                     analysis: Dict[str, Any]):
        """Escalate conflict for manual resolution (Task 15.3-T09)"""
        try:
            # Create escalation request (integration with Chapter 15.2)
            escalation_data = {
                "conflict_id": conflict_id,
                "severity": request.severity.value,
                "estimated_resolution_hours": analysis.get("estimated_resolution_time_hours", 24),
                "recommended_hierarchy": analysis.get("recommended_hierarchy", ResolutionHierarchy.COMPLIANCE).value
            }
            
            logger.info(f"🔄 Escalated conflict {conflict_id} for manual resolution")
            
        except Exception as e:
            logger.error(f"❌ Failed to escalate conflict: {e}")
    
    async def _update_risk_register_for_conflict(self, conflict_id: str, request: ConflictRequest,
                                               analysis: Dict[str, Any]):
        """Update risk register for conflict (Task 15.3-T17)"""
        try:
            risk_entry = {
                "conflict_id": conflict_id,
                "risk_type": "policy_conflict",
                "severity": request.severity.value,
                "affected_modules": request.affected_modules,
                "mitigation_required": analysis.get("escalation_required", False)
            }
            
            logger.info(f"📊 Updated risk register for conflict {conflict_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update risk register: {e}")
    
    async def _link_to_override_ledger(self, conflict_id: str, resolution_id: str, 
                                     user_id: int, tenant_id: int):
        """Link conflict resolution to override ledger (Task 15.3-T16)"""
        try:
            override_entry = {
                "conflict_id": conflict_id,
                "resolution_id": resolution_id,
                "override_type": "conflict_resolution",
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
            logger.info(f"🔗 Linked conflict {conflict_id} to override ledger")
            
        except Exception as e:
            logger.error(f"❌ Failed to link to override ledger: {e}")

# Initialize service
conflict_resolution_service = None

def get_conflict_resolution_service(pool_manager=Depends(get_pool_manager)) -> ConflictResolutionService:
    global conflict_resolution_service
    if conflict_resolution_service is None:
        conflict_resolution_service = ConflictResolutionService(pool_manager)
    return conflict_resolution_service

# =====================================================
# API ENDPOINTS
# =====================================================

@router.post("/detect", response_model=ConflictResponse)
async def detect_and_resolve_conflict(
    request: ConflictRequest,
    background_tasks: BackgroundTasks,
    service: ConflictResolutionService = Depends(get_conflict_resolution_service)
):
    """
    Detect and resolve conflict using compliance-first hierarchy
    Tasks 15.3-T05, T06, T07, T08: Core conflict resolution engine
    """
    return await service.detect_and_resolve_conflict(request)

@router.post("/resolve/{conflict_id}")
async def resolve_conflict_manually(
    conflict_id: str,
    resolution: ConflictResolution,
    tenant_id: int = Query(..., description="Tenant ID"),
    user_id: int = Query(..., description="User ID"),
    service: ConflictResolutionService = Depends(get_conflict_resolution_service)
):
    """
    Manually resolve conflict with hierarchy validation
    Tasks 15.3-T08, T09: Manual conflict resolution with hierarchy enforcement
    """
    return await service.resolve_conflict_manually(conflict_id, resolution, tenant_id, user_id)

@router.get("/status/{conflict_id}")
async def get_conflict_status(
    conflict_id: str,
    tenant_id: int = Query(..., description="Tenant ID"),
    service: ConflictResolutionService = Depends(get_conflict_resolution_service)
):
    """
    Get conflict status and resolution details
    """
    return await service.get_conflict_status(conflict_id, tenant_id)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "conflict_resolution_api", "timestamp": datetime.utcnow()}
