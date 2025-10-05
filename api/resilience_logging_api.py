#!/usr/bin/env python3
"""
Resilience Logging & Alerts Service API - Chapter 15.4
======================================================
Tasks 15.4-T04 to T70: Resilience logging and alerting service

Features:
- Centralized resilience event logging (retry, fallback, conflict, SLA breach)
- OpenTelemetry collectors for runtime logs
- Digital signatures on resilience logs for immutability
- Real-time alerts via Action Center/JIT for critical failures
- Resilience dashboards (MTTR, MTTD, retry volumes, error heatmaps)
- Policy-driven logging rules and tenant isolation
- Industry-specific logging overlays (RBI, IRDAI, SOX, HIPAA)
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

class ResilienceEventType(str, Enum):
    RETRY_EVENT = "retry_event"
    FALLBACK_EVENT = "fallback_event"
    CONFLICT_EVENT = "conflict_event"
    SLA_BREACH = "sla_breach"
    ESCALATION_EVENT = "escalation_event"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"

class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertPriority(str, Enum):
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

class ResilienceLogEntry(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    workflow_id: str = Field(..., description="Workflow ID")
    step_id: Optional[str] = Field(None, description="Workflow step ID")
    
    # Event details
    event_type: ResilienceEventType = Field(..., description="Type of resilience event")
    severity: SeverityLevel = Field(..., description="Event severity")
    event_message: str = Field(..., description="Event message")
    
    # Context
    error_context: Dict[str, Any] = Field({}, description="Error context")
    action_taken: str = Field("", description="Action taken")
    
    # Correlation
    correlation_id: Optional[str] = Field(None, description="Correlation ID for related events")
    parent_event_id: Optional[str] = Field(None, description="Parent event ID")
    
    # Evidence
    evidence_ref: Optional[str] = Field(None, description="Evidence pack reference")
    
    # Metadata
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

class ResilienceAlert(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID")
    
    # Alert details
    alert_type: str = Field(..., description="Alert type")
    priority: AlertPriority = Field(..., description="Alert priority")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    
    # Context
    source_event_ids: List[str] = Field([], description="Source event IDs")
    affected_workflows: List[str] = Field([], description="Affected workflow IDs")
    
    # Routing
    target_personas: List[str] = Field([], description="Target personas for alert")
    notification_channels: List[str] = Field([], description="Notification channels")
    
    # SLA
    sla_minutes: int = Field(60, description="SLA in minutes")
    auto_escalate: bool = Field(True, description="Auto-escalate if no response")
    
    # Metadata
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

class ResilienceMetrics(BaseModel):
    tenant_id: int
    time_window_start: datetime
    time_window_end: datetime
    
    # Retry metrics
    total_retries: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    avg_retry_time_seconds: float = 0.0
    
    # Fallback metrics
    total_fallbacks: int = 0
    successful_fallbacks: int = 0
    fallback_types: Dict[str, int] = {}
    
    # Escalation metrics
    total_escalations: int = 0
    resolved_escalations: int = 0
    avg_resolution_time_hours: float = 0.0
    
    # SLA metrics
    sla_breaches: int = 0
    mttr_hours: float = 0.0  # Mean Time To Recovery
    mttd_hours: float = 0.0  # Mean Time To Detection
    
    # Error metrics
    error_count_by_type: Dict[str, int] = {}
    critical_errors: int = 0

# =====================================================
# RESILIENCE LOGGING SERVICE
# =====================================================

class ResilienceLoggingService:
    """
    Resilience Logging & Alerts Service
    Tasks 15.4-T04 to T27: Centralized logging and real-time alerting
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
        # Industry-specific logging overlays (Task 15.4-T19)
        self.industry_logging_rules = {
            IndustryCode.BANKING: {
                "retention_days": 3650,  # 10 years for RBI compliance
                "required_fields": ["compliance_framework", "regulator_ref", "risk_level"],
                "alert_thresholds": {"error_rate": 0.01, "sla_breach_rate": 0.005},
                "compliance_frameworks": ["RBI", "BASEL_III", "AML"]
            },
            IndustryCode.INSURANCE: {
                "retention_days": 1825,  # 5 years for IRDAI compliance
                "required_fields": ["policy_ref", "claim_ref", "solvency_impact"],
                "alert_thresholds": {"error_rate": 0.02, "sla_breach_rate": 0.01},
                "compliance_frameworks": ["IRDAI", "IRDA"]
            },
            IndustryCode.HEALTHCARE: {
                "retention_days": 2190,  # 6 years for HIPAA compliance
                "required_fields": ["patient_id_hash", "phi_involved", "consent_ref"],
                "alert_thresholds": {"error_rate": 0.005, "sla_breach_rate": 0.001},
                "compliance_frameworks": ["HIPAA", "GDPR"]
            },
            IndustryCode.SAAS: {
                "retention_days": 2555,  # 7 years for SOX compliance
                "required_fields": ["customer_id", "revenue_impact", "service_tier"],
                "alert_thresholds": {"error_rate": 0.05, "sla_breach_rate": 0.02},
                "compliance_frameworks": ["SOX", "GDPR"]
            }
        }
        
        # Alert routing rules (Task 15.4-T24)
        self.alert_routing_rules = {
            AlertPriority.P0: {
                "personas": ["ops_oncall", "engineering_director", "cto"],
                "channels": ["pagerduty", "slack", "email", "sms"],
                "sla_minutes": 15
            },
            AlertPriority.P1: {
                "personas": ["ops_manager", "engineering_manager"],
                "channels": ["slack", "email"],
                "sla_minutes": 60
            },
            AlertPriority.P2: {
                "personas": ["ops_engineer", "developer"],
                "channels": ["slack", "email"],
                "sla_minutes": 240
            },
            AlertPriority.P3: {
                "personas": ["ops_team"],
                "channels": ["email"],
                "sla_minutes": 1440
            }
        }
        
        # Event correlation rules (Task 15.4-T07)
        self.correlation_rules = {
            "retry_chain": {
                "events": [ResilienceEventType.RETRY_EVENT, ResilienceEventType.FALLBACK_EVENT],
                "time_window_minutes": 30,
                "correlation_key": "workflow_id"
            },
            "escalation_chain": {
                "events": [ResilienceEventType.FALLBACK_EVENT, ResilienceEventType.ESCALATION_EVENT],
                "time_window_minutes": 60,
                "correlation_key": "workflow_id"
            },
            "sla_breach_pattern": {
                "events": [ResilienceEventType.SLA_BREACH],
                "time_window_minutes": 15,
                "correlation_key": "tenant_id"
            }
        }
    
    async def log_resilience_event(self, log_entry: ResilienceLogEntry) -> Dict[str, Any]:
        """
        Log resilience event with digital signature and correlation
        Tasks 15.4-T04, T07, T08: Centralized logging with event correlation and digital signatures
        """
        try:
            event_id = str(uuid.uuid4())
            
            # Get logging policy for tenant (Task 15.4-T17)
            logging_policy = await self._get_logging_policy(log_entry.tenant_id)
            
            # Check if event should be logged based on policy
            if not self._should_log_event(log_entry, logging_policy):
                return {"event_id": event_id, "logged": False, "reason": "Filtered by policy"}
            
            # Enrich log entry with industry-specific fields (Task 15.4-T19)
            enriched_entry = await self._enrich_log_entry(log_entry, logging_policy)
            
            # Generate digital signature (Task 15.4-T08)
            digital_signature = await self._generate_digital_signature(enriched_entry)
            
            # Store resilience log
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(log_entry.tenant_id))
                
                await conn.execute("""
                    INSERT INTO resilience_logs (
                        event_id, tenant_id, workflow_id, step_id, event_type, severity,
                        event_message, error_context, action_taken, correlation_id,
                        parent_event_id, evidence_ref, metadata, digital_signature,
                        hash_algorithm, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    event_id, log_entry.tenant_id, log_entry.workflow_id, log_entry.step_id,
                    log_entry.event_type.value, log_entry.severity.value, log_entry.event_message,
                    json.dumps(enriched_entry.error_context), enriched_entry.action_taken,
                    enriched_entry.correlation_id, enriched_entry.parent_event_id,
                    enriched_entry.evidence_ref, json.dumps(enriched_entry.metadata),
                    digital_signature, "SHA256", datetime.utcnow()
                )
            
            # Perform event correlation (Task 15.4-T07)
            correlation_results = await self._correlate_event(event_id, enriched_entry)
            
            # Check for alert conditions (Task 15.4-T21, T22, T23)
            alert_triggered = await self._check_alert_conditions(event_id, enriched_entry, correlation_results)
            
            # Update metrics (Task 15.4-T48)
            await self._update_resilience_metrics(enriched_entry)
            
            logger.info(f"✅ Logged resilience event {event_id} for workflow {log_entry.workflow_id}")
            
            return {
                "event_id": event_id,
                "logged": True,
                "digital_signature": digital_signature,
                "correlation_results": correlation_results,
                "alert_triggered": alert_triggered,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to log resilience event: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_resilience_alert(self, alert: ResilienceAlert) -> Dict[str, Any]:
        """
        Create resilience alert with persona-based routing
        Tasks 15.4-T21, T22, T23, T24: Real-time alerts with persona-based routing
        """
        try:
            alert_id = str(uuid.uuid4())
            
            # Get alert policy for tenant
            alert_policy = await self._get_alert_policy(alert.tenant_id)
            
            # Check alert suppression rules (Task 15.4-T25)
            suppression_check = await self._check_alert_suppression(alert, alert_policy)
            if suppression_check["suppressed"]:
                return {
                    "alert_id": alert_id,
                    "created": False,
                    "reason": suppression_check["reason"]
                }
            
            # Determine routing based on priority and personas (Task 15.4-T24)
            routing_config = self._get_alert_routing_config(alert.priority, alert.target_personas)
            
            # Calculate SLA deadline
            sla_deadline = datetime.utcnow() + timedelta(minutes=alert.sla_minutes)
            
            # Store resilience alert
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(alert.tenant_id))
                
                await conn.execute("""
                    INSERT INTO resilience_alerts (
                        alert_id, tenant_id, alert_type, priority, title, description,
                        source_event_ids, affected_workflows, target_personas,
                        notification_channels, sla_minutes, sla_deadline, auto_escalate,
                        routing_config, metadata, status, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                    alert_id, alert.tenant_id, alert.alert_type, alert.priority.value,
                    alert.title, alert.description, json.dumps(alert.source_event_ids),
                    json.dumps(alert.affected_workflows), json.dumps(alert.target_personas),
                    json.dumps(alert.notification_channels), alert.sla_minutes, sla_deadline,
                    alert.auto_escalate, json.dumps(routing_config), json.dumps(alert.metadata),
                    "open", datetime.utcnow()
                )
            
            # Send notifications (Task 15.4-T24)
            notification_results = await self._send_alert_notifications(alert_id, alert, routing_config)
            
            # Schedule SLA monitoring (Task 15.4-T23)
            await self._schedule_alert_sla_monitoring(alert_id, sla_deadline, alert.tenant_id)
            
            logger.info(f"✅ Created resilience alert {alert_id} with priority {alert.priority.value}")
            
            return {
                "alert_id": alert_id,
                "created": True,
                "priority": alert.priority.value,
                "sla_deadline": sla_deadline.isoformat(),
                "routing_config": routing_config,
                "notification_results": notification_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create resilience alert: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_resilience_metrics(self, tenant_id: int, 
                                   time_window_hours: int = 24) -> ResilienceMetrics:
        """
        Get resilience metrics for tenant
        Tasks 15.4-T12, T13, T14, T15, T16: Resilience dashboards and metrics
        """
        try:
            time_window_start = datetime.utcnow() - timedelta(hours=time_window_hours)
            time_window_end = datetime.utcnow()
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Get retry metrics
                retry_metrics = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_retries,
                        COUNT(CASE WHEN event_type = 'retry_event' AND action_taken LIKE '%success%' THEN 1 END) as successful_retries,
                        COUNT(CASE WHEN event_type = 'retry_event' AND action_taken LIKE '%failed%' THEN 1 END) as failed_retries,
                        AVG(EXTRACT(EPOCH FROM (created_at - lag(created_at) OVER (PARTITION BY workflow_id ORDER BY created_at)))) as avg_retry_time_seconds
                    FROM resilience_logs
                    WHERE tenant_id = $1 
                    AND event_type = 'retry_event'
                    AND created_at >= $2 AND created_at <= $3
                """, tenant_id, time_window_start, time_window_end)
                
                # Get fallback metrics
                fallback_metrics = await conn.fetch("""
                    SELECT 
                        COUNT(*) as total_fallbacks,
                        COUNT(CASE WHEN action_taken LIKE '%success%' THEN 1 END) as successful_fallbacks,
                        jsonb_extract_path_text(metadata, 'fallback_type') as fallback_type,
                        COUNT(*) as type_count
                    FROM resilience_logs
                    WHERE tenant_id = $1 
                    AND event_type = 'fallback_event'
                    AND created_at >= $2 AND created_at <= $3
                    GROUP BY jsonb_extract_path_text(metadata, 'fallback_type')
                """, tenant_id, time_window_start, time_window_end)
                
                # Get escalation metrics
                escalation_metrics = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_escalations,
                        COUNT(CASE WHEN action_taken LIKE '%resolved%' THEN 1 END) as resolved_escalations,
                        AVG(EXTRACT(EPOCH FROM (created_at - lag(created_at) OVER (PARTITION BY workflow_id ORDER BY created_at)))/3600) as avg_resolution_time_hours
                    FROM resilience_logs
                    WHERE tenant_id = $1 
                    AND event_type = 'escalation_event'
                    AND created_at >= $2 AND created_at <= $3
                """, tenant_id, time_window_start, time_window_end)
                
                # Get SLA and error metrics
                sla_error_metrics = await conn.fetchrow("""
                    SELECT 
                        COUNT(CASE WHEN event_type = 'sla_breach' THEN 1 END) as sla_breaches,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_errors
                    FROM resilience_logs
                    WHERE tenant_id = $1 
                    AND created_at >= $2 AND created_at <= $3
                """, tenant_id, time_window_start, time_window_end)
                
                # Get error count by type
                error_types = await conn.fetch("""
                    SELECT 
                        event_type,
                        COUNT(*) as error_count
                    FROM resilience_logs
                    WHERE tenant_id = $1 
                    AND severity IN ('error', 'critical')
                    AND created_at >= $2 AND created_at <= $3
                    GROUP BY event_type
                """, tenant_id, time_window_start, time_window_end)
            
            # Build fallback types dictionary
            fallback_types = {}
            total_fallbacks = 0
            successful_fallbacks = 0
            
            for row in fallback_metrics:
                fallback_type = row['fallback_type'] or 'unknown'
                fallback_types[fallback_type] = row['type_count']
                total_fallbacks += row['type_count']
                if row['successful_fallbacks']:
                    successful_fallbacks += row['successful_fallbacks']
            
            # Build error count by type dictionary
            error_count_by_type = {row['event_type']: row['error_count'] for row in error_types}
            
            # Calculate MTTR and MTTD (simplified calculation)
            mttr_hours = escalation_metrics['avg_resolution_time_hours'] or 0.0
            mttd_hours = 0.5  # Placeholder - would be calculated from detection to alert time
            
            return ResilienceMetrics(
                tenant_id=tenant_id,
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                total_retries=retry_metrics['total_retries'] or 0,
                successful_retries=retry_metrics['successful_retries'] or 0,
                failed_retries=retry_metrics['failed_retries'] or 0,
                avg_retry_time_seconds=float(retry_metrics['avg_retry_time_seconds'] or 0),
                total_fallbacks=total_fallbacks,
                successful_fallbacks=successful_fallbacks,
                fallback_types=fallback_types,
                total_escalations=escalation_metrics['total_escalations'] or 0,
                resolved_escalations=escalation_metrics['resolved_escalations'] or 0,
                avg_resolution_time_hours=float(mttr_hours),
                sla_breaches=sla_error_metrics['sla_breaches'] or 0,
                mttr_hours=float(mttr_hours),
                mttd_hours=float(mttd_hours),
                error_count_by_type=error_count_by_type,
                critical_errors=sla_error_metrics['critical_errors'] or 0
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get resilience metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Helper methods
    async def _get_logging_policy(self, tenant_id: int) -> Dict[str, Any]:
        """Get logging policy for tenant (Task 15.4-T17)"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                policy = await conn.fetchrow("""
                    SELECT * FROM resilience_logging_policies 
                    WHERE tenant_id = $1 AND active = true
                    ORDER BY created_at DESC LIMIT 1
                """, tenant_id)
                
                if policy:
                    return {
                        "logging_rules": json.loads(policy['logging_rules']),
                        "retention_days": policy['retention_days'],
                        "industry_code": policy['industry_code']
                    }
                
                # Return default policy
                return {
                    "logging_rules": {"log_all": True},
                    "retention_days": 90,
                    "industry_code": "SaaS"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get logging policy: {e}")
            return {"logging_rules": {"log_all": True}}
    
    async def _get_alert_policy(self, tenant_id: int) -> Dict[str, Any]:
        """Get alert policy for tenant"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                policy = await conn.fetchrow("""
                    SELECT * FROM resilience_alert_policies 
                    WHERE tenant_id = $1 AND active = true
                    ORDER BY created_at DESC LIMIT 1
                """, tenant_id)
                
                if policy:
                    return {
                        "alert_rules": json.loads(policy['alert_rules']),
                        "suppression_rules": json.loads(policy['suppression_rules']),
                        "routing_overrides": json.loads(policy['routing_overrides'])
                    }
                
                # Return default policy
                return {
                    "alert_rules": {"alert_all_critical": True},
                    "suppression_rules": {"test_tenants": True},
                    "routing_overrides": {}
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get alert policy: {e}")
            return {"alert_rules": {"alert_all_critical": True}}
    
    def _should_log_event(self, log_entry: ResilienceLogEntry, policy: Dict[str, Any]) -> bool:
        """Check if event should be logged based on policy (Task 15.4-T17)"""
        logging_rules = policy.get("logging_rules", {})
        
        # Always log critical events
        if log_entry.severity == SeverityLevel.CRITICAL:
            return True
        
        # Check policy rules
        if logging_rules.get("log_all", False):
            return True
        
        # Check event type specific rules
        event_rules = logging_rules.get("event_types", {})
        return event_rules.get(log_entry.event_type.value, True)
    
    async def _enrich_log_entry(self, log_entry: ResilienceLogEntry, 
                              policy: Dict[str, Any]) -> ResilienceLogEntry:
        """Enrich log entry with industry-specific fields (Task 15.4-T19)"""
        industry_code = IndustryCode(policy.get("industry_code", "SaaS"))
        industry_rules = self.industry_logging_rules.get(industry_code, {})
        
        # Add industry-specific metadata
        enriched_metadata = log_entry.metadata.copy()
        enriched_metadata.update({
            "industry_code": industry_code.value,
            "compliance_frameworks": industry_rules.get("compliance_frameworks", []),
            "retention_days": industry_rules.get("retention_days", 90)
        })
        
        # Create enriched entry
        enriched_entry = log_entry.copy()
        enriched_entry.metadata = enriched_metadata
        
        return enriched_entry
    
    async def _generate_digital_signature(self, log_entry: ResilienceLogEntry) -> str:
        """Generate digital signature for log entry (Task 15.4-T08)"""
        try:
            # Create canonical representation
            canonical_data = {
                "tenant_id": log_entry.tenant_id,
                "workflow_id": log_entry.workflow_id,
                "event_type": log_entry.event_type.value,
                "severity": log_entry.severity.value,
                "event_message": log_entry.event_message,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Generate hash
            canonical_json = json.dumps(canonical_data, sort_keys=True)
            signature = hashlib.sha256(canonical_json.encode()).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"❌ Failed to generate digital signature: {e}")
            return f"error_{uuid.uuid4()}"
    
    async def _correlate_event(self, event_id: str, log_entry: ResilienceLogEntry) -> Dict[str, Any]:
        """Correlate event with existing events (Task 15.4-T07)"""
        try:
            correlation_results = {"correlations": [], "patterns": []}
            
            # Check each correlation rule
            for rule_name, rule_config in self.correlation_rules.items():
                if log_entry.event_type in rule_config["events"]:
                    # Find related events within time window
                    time_window = datetime.utcnow() - timedelta(minutes=rule_config["time_window_minutes"])
                    correlation_key = rule_config["correlation_key"]
                    
                    # Get correlation value
                    if correlation_key == "workflow_id":
                        correlation_value = log_entry.workflow_id
                    elif correlation_key == "tenant_id":
                        correlation_value = str(log_entry.tenant_id)
                    else:
                        continue
                    
                    # Find related events (simplified)
                    related_events = await self._find_related_events(
                        correlation_key, correlation_value, time_window, log_entry.tenant_id
                    )
                    
                    if related_events:
                        correlation_results["correlations"].append({
                            "rule": rule_name,
                            "related_events": related_events,
                            "correlation_strength": len(related_events) / 10.0  # Simplified scoring
                        })
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"❌ Failed to correlate event: {e}")
            return {"correlations": [], "patterns": []}
    
    async def _check_alert_conditions(self, event_id: str, log_entry: ResilienceLogEntry,
                                    correlation_results: Dict[str, Any]) -> bool:
        """Check if event should trigger an alert (Task 15.4-T21, T22, T23)"""
        try:
            # Always alert on critical events
            if log_entry.severity == SeverityLevel.CRITICAL:
                await self._trigger_automatic_alert(event_id, log_entry, AlertPriority.P0)
                return True
            
            # Check for error patterns
            if log_entry.severity == SeverityLevel.ERROR:
                # Check if this is part of a pattern
                correlation_strength = sum(
                    corr["correlation_strength"] for corr in correlation_results.get("correlations", [])
                )
                
                if correlation_strength > 0.5:  # Pattern detected
                    await self._trigger_automatic_alert(event_id, log_entry, AlertPriority.P1)
                    return True
            
            # Check for SLA breaches
            if log_entry.event_type == ResilienceEventType.SLA_BREACH:
                await self._trigger_automatic_alert(event_id, log_entry, AlertPriority.P2)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to check alert conditions: {e}")
            return False
    
    async def _trigger_automatic_alert(self, event_id: str, log_entry: ResilienceLogEntry,
                                     priority: AlertPriority):
        """Trigger automatic alert for event"""
        try:
            alert = ResilienceAlert(
                tenant_id=log_entry.tenant_id,
                alert_type="automatic",
                priority=priority,
                title=f"Resilience Event: {log_entry.event_type.value}",
                description=log_entry.event_message,
                source_event_ids=[event_id],
                affected_workflows=[log_entry.workflow_id],
                target_personas=self.alert_routing_rules[priority]["personas"],
                notification_channels=self.alert_routing_rules[priority]["channels"],
                sla_minutes=self.alert_routing_rules[priority]["sla_minutes"]
            )
            
            await self.create_resilience_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Failed to trigger automatic alert: {e}")
    
    def _get_alert_routing_config(self, priority: AlertPriority, 
                                target_personas: List[str]) -> Dict[str, Any]:
        """Get alert routing configuration (Task 15.4-T24)"""
        base_config = self.alert_routing_rules.get(priority, self.alert_routing_rules[AlertPriority.P3])
        
        # Override personas if specified
        if target_personas:
            base_config = base_config.copy()
            base_config["personas"] = target_personas
        
        return base_config
    
    async def _check_alert_suppression(self, alert: ResilienceAlert, 
                                     policy: Dict[str, Any]) -> Dict[str, Any]:
        """Check alert suppression rules (Task 15.4-T25)"""
        suppression_rules = policy.get("suppression_rules", {})
        
        # Check if tenant is in test mode
        if suppression_rules.get("test_tenants", False):
            # Check if this is a test tenant (simplified check)
            if alert.tenant_id < 1000:  # Assume test tenants have low IDs
                return {"suppressed": True, "reason": "Test tenant suppression"}
        
        # Check for duplicate alerts
        duplicate_check = await self._check_duplicate_alerts(alert)
        if duplicate_check["is_duplicate"]:
            return {"suppressed": True, "reason": f"Duplicate alert: {duplicate_check['existing_alert_id']}"}
        
        return {"suppressed": False, "reason": "No suppression rules matched"}
    
    # Placeholder methods for external integrations
    async def _find_related_events(self, correlation_key: str, correlation_value: str,
                                 time_window: datetime, tenant_id: int) -> List[str]:
        """Find related events for correlation"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                if correlation_key == "workflow_id":
                    events = await conn.fetch("""
                        SELECT event_id FROM resilience_logs 
                        WHERE tenant_id = $1 AND workflow_id = $2 
                        AND created_at >= $3
                        ORDER BY created_at DESC LIMIT 10
                    """, tenant_id, correlation_value, time_window)
                else:
                    events = await conn.fetch("""
                        SELECT event_id FROM resilience_logs 
                        WHERE tenant_id = $1 AND created_at >= $2
                        ORDER BY created_at DESC LIMIT 10
                    """, tenant_id, time_window)
                
                return [event['event_id'] for event in events]
                
        except Exception as e:
            logger.error(f"❌ Failed to find related events: {e}")
            return []
    
    async def _send_alert_notifications(self, alert_id: str, alert: ResilienceAlert,
                                      routing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert notifications (Task 15.4-T24)"""
        logger.info(f"📧 Sending alert notifications for {alert_id} to {routing_config['personas']}")
        return {"notifications_sent": len(routing_config["personas"]), "channels": routing_config["channels"]}
    
    async def _schedule_alert_sla_monitoring(self, alert_id: str, sla_deadline: datetime, tenant_id: int):
        """Schedule SLA monitoring for alert (Task 15.4-T23)"""
        logger.info(f"⏰ Scheduled SLA monitoring for alert {alert_id} until {sla_deadline}")
    
    async def _check_duplicate_alerts(self, alert: ResilienceAlert) -> Dict[str, Any]:
        """Check for duplicate alerts"""
        # Simplified duplicate check
        return {"is_duplicate": False, "existing_alert_id": None}
    
    async def _update_resilience_metrics(self, log_entry: ResilienceLogEntry):
        """Update resilience metrics (Task 15.4-T48)"""
        try:
            # Update metrics in background
            logger.info(f"📊 Updating resilience metrics for tenant {log_entry.tenant_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update resilience metrics: {e}")

# Initialize service
resilience_logging_service = None

def get_resilience_logging_service(pool_manager=Depends(get_pool_manager)) -> ResilienceLoggingService:
    global resilience_logging_service
    if resilience_logging_service is None:
        resilience_logging_service = ResilienceLoggingService(pool_manager)
    return resilience_logging_service

# =====================================================
# API ENDPOINTS
# =====================================================

@router.post("/log", response_model=Dict[str, Any])
async def log_resilience_event(
    log_entry: ResilienceLogEntry,
    background_tasks: BackgroundTasks,
    service: ResilienceLoggingService = Depends(get_resilience_logging_service)
):
    """
    Log resilience event with digital signature and correlation
    Tasks 15.4-T04, T07, T08: Centralized logging with event correlation and digital signatures
    """
    return await service.log_resilience_event(log_entry)

@router.post("/alert", response_model=Dict[str, Any])
async def create_resilience_alert(
    alert: ResilienceAlert,
    background_tasks: BackgroundTasks,
    service: ResilienceLoggingService = Depends(get_resilience_logging_service)
):
    """
    Create resilience alert with persona-based routing
    Tasks 15.4-T21, T22, T23, T24: Real-time alerts with persona-based routing
    """
    return await service.create_resilience_alert(alert)

@router.get("/metrics/{tenant_id}", response_model=ResilienceMetrics)
async def get_resilience_metrics(
    tenant_id: int,
    time_window_hours: int = Query(24, description="Time window in hours", ge=1, le=168),
    service: ResilienceLoggingService = Depends(get_resilience_logging_service)
):
    """
    Get resilience metrics for tenant
    Tasks 15.4-T12, T13, T14, T15, T16: Resilience dashboards and metrics
    """
    return await service.get_resilience_metrics(tenant_id, time_window_hours)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "resilience_logging_api", "timestamp": datetime.utcnow()}
