"""
Task 9.2.35: Build Audit Trail Service
Comprehensive audit trail with immutable logging and compliance reporting

Features:
- Immutable audit event logging with cryptographic anchoring
- Chain integrity verification with hash linking
- Compliance-aware retention policies
- Cross-plane audit event correlation
- Real-time monitoring and alerting
- Tamper detection and evidence preservation
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
import hashlib

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    GOVERNANCE_ACTION = "governance_action"
    POLICY_VIOLATION = "policy_violation"
    OVERRIDE_LEDGER_ENTRY = "override_ledger_entry"
    EVIDENCE_PACK_GENERATION = "evidence_pack_generation"
    REGULATOR_NOTIFICATION = "regulator_notification"
    DATA_ACCESS_EVENT = "data_access_event"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    COMPLIANCE_BREACH = "compliance_breach"
    CROSS_PLANE_OPERATION = "cross_plane_operation"
    AUTHENTICATION_EVENT = "authentication_event"
    AUTHORIZATION_EVENT = "authorization_event"
    WORKFLOW_EXECUTION = "workflow_execution"
    SYSTEM_CONFIGURATION = "system_configuration"

class RetentionCategory(Enum):
    """Audit event retention categories"""
    DEFAULT = "default"
    COMPLIANCE = "compliance"
    HIGH_RISK = "high_risk"
    REGULATORY = "regulatory"

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "SOX"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    RBI = "RBI"
    DPDP = "DPDP"
    CCPA = "CCPA"
    PCI_DSS = "PCI_DSS"

@dataclass
class AuditEvent:
    """Immutable audit event with chain integrity"""
    audit_event_id: str
    tenant_id: int
    user_id: Optional[str]
    plane: Optional[str]
    event_type: str
    event_data: Dict[str, Any]
    timestamp: str
    source_service: str
    compliance_relevant: bool
    retention_category: str
    previous_event_hash: Optional[str]
    event_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_event_id": self.audit_event_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "plane": self.plane,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "timestamp": self.timestamp,
            "source_service": self.source_service,
            "compliance_relevant": self.compliance_relevant,
            "retention_category": self.retention_category,
            "previous_event_hash": self.previous_event_hash,
            "event_hash": self.event_hash
        }

class AuditTrailService:
    """
    Task 9.2.35: Build Audit Trail Service
    Comprehensive audit trail with immutable logging and compliance reporting
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Service configuration - DYNAMIC, no hardcoding
        self.config = {
            "service_name": "audit_trail_service",
            "version": "1.0.0",
            "retention_policies": {
                "default_retention_days": int(os.getenv("AUDIT_RETENTION_DAYS", "2555")),  # 7 years
                "compliance_retention_days": int(os.getenv("COMPLIANCE_AUDIT_RETENTION", "3650")),  # 10 years
                "high_risk_retention_days": int(os.getenv("HIGH_RISK_AUDIT_RETENTION", "5475")),  # 15 years
                "regulatory_retention_days": int(os.getenv("REGULATORY_AUDIT_RETENTION", "7300"))  # 20 years
            },
            "audit_categories": [
                event_type.value for event_type in AuditEventType
            ],
            "compliance_frameworks": [
                framework.value for framework in ComplianceFramework
            ],
            "features": {
                "immutable_logging": True,
                "cryptographic_anchoring": True,
                "compliance_reporting": True,
                "real_time_monitoring": True,
                "cross_plane_auditing": True,
                "tamper_detection": True,
                "chain_integrity_verification": True
            },
            "monitoring": {
                "alert_on_chain_break": os.getenv("AUDIT_ALERT_CHAIN_BREAK", "true").lower() == "true",
                "alert_on_tampering": os.getenv("AUDIT_ALERT_TAMPERING", "true").lower() == "true",
                "real_time_verification": os.getenv("AUDIT_REAL_TIME_VERIFICATION", "true").lower() == "true"
            }
        }
        
        # In-memory storage for demo (would be database in production)
        self.audit_events: Dict[str, AuditEvent] = {}
        self.event_chains: Dict[int, List[str]] = {}  # tenant_id -> event_id list
        self.compliance_index: Dict[str, List[str]] = {}  # framework -> event_id list
        
        # Metrics and monitoring
        self.metrics = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_tenant": {},
            "compliance_events": 0,
            "chain_integrity_checks": 0,
            "integrity_violations": 0
        }
        
        logger.info("🔍 Audit Trail Service initialized with comprehensive compliance support")
    
    async def log_audit_event(self, event_type: AuditEventType, event_data: Dict[str, Any], 
                             tenant_id: int, user_id: Optional[str] = None,
                             plane: Optional[str] = None,
                             source_service: str = "unknown") -> str:
        """
        Log audit events with immutable storage and chain integrity
        """
        try:
            audit_event_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            # Get previous event hash for chain integrity
            previous_hash = await self._get_last_event_hash(tenant_id)
            
            # Determine compliance relevance and retention
            compliance_relevant = self._is_compliance_relevant(event_type)
            retention_category = self._determine_retention_category(event_type)
            
            # Create audit event (without hash initially)
            event_dict = {
                "audit_event_id": audit_event_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "plane": plane,
                "event_type": event_type.value,
                "event_data": event_data,
                "timestamp": timestamp.isoformat(),
                "source_service": source_service,
                "compliance_relevant": compliance_relevant,
                "retention_category": retention_category.value,
                "previous_event_hash": previous_hash
            }
            
            # Calculate event hash for integrity
            event_hash = self._calculate_event_hash(event_dict)
            event_dict["event_hash"] = event_hash
            
            # Create immutable audit event
            audit_event = AuditEvent(**event_dict)
            
            # Store in audit trail
            await self._store_audit_event(audit_event)
            
            # Update event chain
            if tenant_id not in self.event_chains:
                self.event_chains[tenant_id] = []
            self.event_chains[tenant_id].append(audit_event_id)
            
            # Update compliance index if relevant
            if compliance_relevant:
                await self._update_compliance_index(audit_event)
            
            # Update metrics
            self._update_metrics(audit_event)
            
            logger.info(f"📝 Audit event logged: {event_type.value} for tenant {tenant_id}")
            return audit_event_id
            
        except Exception as e:
            logger.error(f"❌ Failed to log audit event: {e}")
            return ""
    
    async def get_audit_trail(self, tenant_id: int, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             event_types: Optional[List[AuditEventType]] = None,
                             compliance_only: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail for tenant with filtering
        """
        try:
            # Get all events for tenant
            tenant_event_ids = self.event_chains.get(tenant_id, [])
            tenant_events = [
                self.audit_events[event_id] for event_id in tenant_event_ids
                if event_id in self.audit_events
            ]
            
            # Apply filters
            if start_date:
                tenant_events = [
                    event for event in tenant_events
                    if datetime.fromisoformat(event.timestamp) >= start_date
                ]
            
            if end_date:
                tenant_events = [
                    event for event in tenant_events
                    if datetime.fromisoformat(event.timestamp) <= end_date
                ]
            
            if event_types:
                event_type_values = [et.value for et in event_types]
                tenant_events = [
                    event for event in tenant_events
                    if event.event_type in event_type_values
                ]
            
            if compliance_only:
                tenant_events = [
                    event for event in tenant_events
                    if event.compliance_relevant
                ]
            
            # Sort by timestamp
            tenant_events.sort(key=lambda x: x.timestamp)
            
            return [event.to_dict() for event in tenant_events]
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve audit trail: {e}")
            return []
    
    async def verify_audit_chain_integrity(self, tenant_id: int) -> Dict[str, Any]:
        """
        Verify the integrity of the audit event chain for a tenant
        """
        try:
            if tenant_id not in self.event_chains:
                return {"valid": True, "message": "No events found for tenant"}
            
            event_ids = self.event_chains[tenant_id]
            integrity_issues = []
            verified_events = 0
            
            for i, event_id in enumerate(event_ids):
                event = self.audit_events.get(event_id)
                if not event:
                    integrity_issues.append(f"Missing event: {event_id}")
                    continue
                
                # Verify event hash
                calculated_hash = self._calculate_event_hash(event.to_dict())
                if calculated_hash != event.event_hash:
                    integrity_issues.append(f"Hash mismatch for event: {event_id}")
                
                # Verify chain linkage (except for first event)
                if i > 0:
                    previous_event_id = event_ids[i-1]
                    previous_event = self.audit_events.get(previous_event_id)
                    if previous_event and event.previous_event_hash != previous_event.event_hash:
                        integrity_issues.append(f"Chain break between {previous_event_id} and {event_id}")
                
                verified_events += 1
            
            # Update metrics
            self.metrics["chain_integrity_checks"] += 1
            if integrity_issues:
                self.metrics["integrity_violations"] += len(integrity_issues)
            
            verification_result = {
                "valid": len(integrity_issues) == 0,
                "total_events": len(event_ids),
                "verified_events": verified_events,
                "integrity_issues": integrity_issues,
                "verification_timestamp": datetime.utcnow().isoformat(),
                "tenant_id": tenant_id
            }
            
            # Alert on integrity violations if configured
            if integrity_issues and self.config["monitoring"]["alert_on_chain_break"]:
                await self._alert_integrity_violation(tenant_id, verification_result)
            
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Failed to verify audit chain integrity: {e}")
            return {"valid": False, "error": str(e)}
    
    async def generate_compliance_report(self, tenant_id: int, 
                                       framework: ComplianceFramework,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance report for specific framework and time period
        """
        try:
            # Get compliance-relevant events
            compliance_events = await self.get_audit_trail(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
                compliance_only=True
            )
            
            # Filter by framework-specific events
            framework_events = [
                event for event in compliance_events
                if self._is_framework_relevant(event, framework)
            ]
            
            # Generate report
            report = {
                "report_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "compliance_framework": framework.value,
                "reporting_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_events": len(framework_events),
                    "policy_violations": len([e for e in framework_events if e["event_type"] == "policy_violation"]),
                    "compliance_breaches": len([e for e in framework_events if e["event_type"] == "compliance_breach"]),
                    "regulator_notifications": len([e for e in framework_events if e["event_type"] == "regulator_notification"]),
                    "evidence_packs_generated": len([e for e in framework_events if e["event_type"] == "evidence_pack_generation"])
                },
                "events": framework_events,
                "chain_integrity": await self.verify_audit_chain_integrity(tenant_id),
                "generated_at": datetime.utcnow().isoformat(),
                "generated_by": "audit_trail_service"
            }
            
            logger.info(f"📊 Compliance report generated for {framework.value}: {len(framework_events)} events")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate compliance report: {e}")
            return {}
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        return {
            "service_config": self.config,
            "metrics": self.metrics,
            "active_tenants": len(self.event_chains),
            "total_audit_events": len(self.audit_events),
            "compliance_index_size": len(self.compliance_index),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _is_compliance_relevant(self, event_type: AuditEventType) -> bool:
        """Determine if event type is compliance relevant"""
        compliance_events = [
            AuditEventType.POLICY_VIOLATION,
            AuditEventType.COMPLIANCE_BREACH,
            AuditEventType.REGULATOR_NOTIFICATION,
            AuditEventType.EVIDENCE_PACK_GENERATION,
            AuditEventType.OVERRIDE_LEDGER_ENTRY,
            AuditEventType.DATA_ACCESS_EVENT,
            AuditEventType.PRIVILEGE_ESCALATION,
            AuditEventType.GOVERNANCE_ACTION
        ]
        return event_type in compliance_events
    
    def _determine_retention_category(self, event_type: AuditEventType) -> RetentionCategory:
        """Determine retention category based on event type"""
        high_risk_events = [
            AuditEventType.POLICY_VIOLATION,
            AuditEventType.COMPLIANCE_BREACH,
            AuditEventType.PRIVILEGE_ESCALATION,
            AuditEventType.REGULATOR_NOTIFICATION
        ]
        
        regulatory_events = [
            AuditEventType.REGULATOR_NOTIFICATION,
            AuditEventType.COMPLIANCE_BREACH
        ]
        
        compliance_events = [
            AuditEventType.EVIDENCE_PACK_GENERATION,
            AuditEventType.GOVERNANCE_ACTION,
            AuditEventType.DATA_ACCESS_EVENT,
            AuditEventType.OVERRIDE_LEDGER_ENTRY
        ]
        
        if event_type in regulatory_events:
            return RetentionCategory.REGULATORY
        elif event_type in high_risk_events:
            return RetentionCategory.HIGH_RISK
        elif event_type in compliance_events:
            return RetentionCategory.COMPLIANCE
        else:
            return RetentionCategory.DEFAULT
    
    def _is_framework_relevant(self, event: Dict[str, Any], framework: ComplianceFramework) -> bool:
        """Check if event is relevant to specific compliance framework"""
        # Framework-specific event filtering logic
        framework_mappings = {
            ComplianceFramework.SOX: ["governance_action", "evidence_pack_generation", "override_ledger_entry"],
            ComplianceFramework.GDPR: ["data_access_event", "regulator_notification", "compliance_breach"],
            ComplianceFramework.HIPAA: ["data_access_event", "privilege_escalation", "compliance_breach"],
            ComplianceFramework.RBI: ["policy_violation", "regulator_notification", "governance_action"],
            ComplianceFramework.DPDP: ["data_access_event", "regulator_notification", "compliance_breach"]
        }
        
        relevant_types = framework_mappings.get(framework, [])
        return event["event_type"] in relevant_types
    
    def _calculate_event_hash(self, event_dict: Dict[str, Any]) -> str:
        """Calculate cryptographic hash for event integrity"""
        # Create copy without the hash field
        event_copy = event_dict.copy()
        event_copy.pop("event_hash", None)
        
        # Create deterministic JSON string
        event_json = json.dumps(event_copy, sort_keys=True, separators=(',', ':'))
        
        # Calculate SHA-256 hash
        return hashlib.sha256(event_json.encode('utf-8')).hexdigest()
    
    async def _get_last_event_hash(self, tenant_id: int) -> Optional[str]:
        """Get hash of last audit event for chain integrity"""
        if tenant_id not in self.event_chains or not self.event_chains[tenant_id]:
            return None
        
        last_event_id = self.event_chains[tenant_id][-1]
        last_event = self.audit_events.get(last_event_id)
        
        return last_event.event_hash if last_event else None
    
    async def _store_audit_event(self, audit_event: AuditEvent) -> bool:
        """Store audit event in immutable storage"""
        try:
            # Store in memory (would be database with WORM characteristics in production)
            self.audit_events[audit_event.audit_event_id] = audit_event
            
            logger.info(f"🔒 Stored audit event: {audit_event.audit_event_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store audit event: {e}")
            return False
    
    async def _update_compliance_index(self, audit_event: AuditEvent):
        """Update compliance framework index"""
        for framework in ComplianceFramework:
            if self._is_framework_relevant(audit_event.to_dict(), framework):
                if framework.value not in self.compliance_index:
                    self.compliance_index[framework.value] = []
                self.compliance_index[framework.value].append(audit_event.audit_event_id)
    
    def _update_metrics(self, audit_event: AuditEvent):
        """Update service metrics"""
        self.metrics["total_events"] += 1
        
        # Update by type
        event_type = audit_event.event_type
        if event_type not in self.metrics["events_by_type"]:
            self.metrics["events_by_type"][event_type] = 0
        self.metrics["events_by_type"][event_type] += 1
        
        # Update by tenant
        tenant_id = str(audit_event.tenant_id)
        if tenant_id not in self.metrics["events_by_tenant"]:
            self.metrics["events_by_tenant"][tenant_id] = 0
        self.metrics["events_by_tenant"][tenant_id] += 1
        
        # Update compliance events
        if audit_event.compliance_relevant:
            self.metrics["compliance_events"] += 1
    
    async def _alert_integrity_violation(self, tenant_id: int, verification_result: Dict[str, Any]):
        """Alert on audit chain integrity violations"""
        alert = {
            "alert_type": "audit_chain_integrity_violation",
            "tenant_id": tenant_id,
            "severity": "critical",
            "verification_result": verification_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # In production, this would send alerts to monitoring systems
        logger.critical(f"🚨 AUDIT CHAIN INTEGRITY VIOLATION: {alert}")

# Global audit trail service instance
audit_trail_service = AuditTrailService()
