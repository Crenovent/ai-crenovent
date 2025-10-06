"""
Task 9.2.21: Drift Detection Service - Schema/policy/config drift
Identifies and alerts on schema, policy, and configuration drift across the platform.

Features:
- Schema drift detection with automatic remediation
- Policy configuration drift monitoring
- Configuration baseline comparison
- Auto-remediation task creation
- Multi-tenant drift isolation
- DYNAMIC thresholds and rules (no hardcoding)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid
import difflib

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of drift detection"""
    SCHEMA = "schema"
    POLICY = "policy"
    CONFIG = "config"
    CAPABILITY = "capability"
    GOVERNANCE = "governance"

class DriftSeverity(Enum):
    """Drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DriftStatus(Enum):
    """Drift remediation status"""
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    REMEDIATION_CREATED = "remediation_created"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    IGNORED = "ignored"

@dataclass
class DriftBaseline:
    """Baseline configuration for drift detection"""
    baseline_id: str
    drift_type: DriftType
    tenant_id: int
    component_name: str
    baseline_content: Dict[str, Any]
    baseline_hash: str
    created_at: datetime
    updated_at: datetime
    
    # Metadata
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_hash(self) -> str:
        """Calculate hash of baseline content"""
        content_str = json.dumps(self.baseline_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

@dataclass
class DriftDetection:
    """Detected drift instance"""
    drift_id: str
    baseline_id: str
    drift_type: DriftType
    tenant_id: int
    component_name: str
    
    # Drift details
    current_content: Dict[str, Any]
    current_hash: str
    baseline_hash: str
    drift_diff: List[str]
    
    # Classification
    severity: DriftSeverity
    status: DriftStatus
    
    # Timestamps
    detected_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Remediation
    remediation_task_id: Optional[str] = None
    auto_remediation_enabled: bool = True
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "baseline_id": self.baseline_id,
            "drift_type": self.drift_type.value,
            "tenant_id": self.tenant_id,
            "component_name": self.component_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "detected_at": self.detected_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "remediation_task_id": self.remediation_task_id,
            "drift_diff_count": len(self.drift_diff),
            "metadata": self.metadata
        }

@dataclass
class RemediationTask:
    """Auto-generated remediation task"""
    task_id: str
    drift_id: str
    tenant_id: int
    task_type: str
    priority: str
    description: str
    
    # Task details
    target_component: str
    remediation_actions: List[Dict[str, Any]]
    estimated_duration_minutes: int
    
    # Status
    status: str = "created"
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "drift_id": self.drift_id,
            "tenant_id": self.tenant_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "description": self.description,
            "target_component": self.target_component,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "estimated_duration_minutes": self.estimated_duration_minutes
        }

class DriftDetectionService:
    """
    Task 9.2.21: Drift Detection Service
    Comprehensive drift detection and auto-remediation
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage (would be database in production)
        self.baselines: Dict[str, DriftBaseline] = {}
        self.detections: Dict[str, DriftDetection] = {}
        self.remediation_tasks: Dict[str, RemediationTask] = {}
        
        # Dynamic configuration - NO hardcoding
        self.drift_config = {
            "detection_interval_seconds": int(os.getenv("DRIFT_DETECTION_INTERVAL", "300")),  # 5 minutes
            "auto_remediation_enabled": os.getenv("DRIFT_AUTO_REMEDIATION", "true").lower() == "true",
            "severity_thresholds": {
                "schema_changes": int(os.getenv("DRIFT_SCHEMA_THRESHOLD", "5")),
                "policy_changes": int(os.getenv("DRIFT_POLICY_THRESHOLD", "3")),
                "config_changes": int(os.getenv("DRIFT_CONFIG_THRESHOLD", "10"))
            },
            "notification_enabled": os.getenv("DRIFT_NOTIFICATIONS", "true").lower() == "true",
            "retention_days": int(os.getenv("DRIFT_RETENTION_DAYS", "90"))
        }
        
        # Drift detection rules - dynamically configurable
        self.detection_rules = {
            DriftType.SCHEMA: {
                "enabled": True,
                "check_interval_seconds": 300,
                "severity_mapping": {
                    "field_added": DriftSeverity.LOW,
                    "field_removed": DriftSeverity.HIGH,
                    "field_type_changed": DriftSeverity.CRITICAL,
                    "constraint_changed": DriftSeverity.MEDIUM
                }
            },
            DriftType.POLICY: {
                "enabled": True,
                "check_interval_seconds": 600,
                "severity_mapping": {
                    "rule_added": DriftSeverity.LOW,
                    "rule_removed": DriftSeverity.HIGH,
                    "rule_modified": DriftSeverity.MEDIUM,
                    "enforcement_changed": DriftSeverity.CRITICAL
                }
            },
            DriftType.CONFIG: {
                "enabled": True,
                "check_interval_seconds": 900,
                "severity_mapping": {
                    "setting_added": DriftSeverity.LOW,
                    "setting_removed": DriftSeverity.MEDIUM,
                    "setting_modified": DriftSeverity.MEDIUM,
                    "critical_setting_changed": DriftSeverity.CRITICAL
                }
            }
        }
        
        # Metrics and monitoring
        self.metrics = {
            "baselines_created": 0,
            "drift_detections": 0,
            "remediation_tasks_created": 0,
            "auto_remediations_successful": 0,
            "detection_runs": 0,
            "detection_errors": 0
        }
        
        # Component registry for dynamic discovery
        self.monitored_components: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🔍 Drift Detection Service initialized")
    
    async def register_baseline(self, baseline: DriftBaseline) -> bool:
        """Register a new baseline for drift detection"""
        try:
            # Calculate baseline hash
            baseline.baseline_hash = baseline.calculate_hash()
            
            # Store baseline
            self.baselines[baseline.baseline_id] = baseline
            
            # Register component for monitoring
            component_key = f"{baseline.tenant_id}_{baseline.component_name}"
            self.monitored_components[component_key] = {
                "baseline_id": baseline.baseline_id,
                "drift_type": baseline.drift_type,
                "last_check": None,
                "check_interval": self.detection_rules[baseline.drift_type]["check_interval_seconds"]
            }
            
            self.metrics["baselines_created"] += 1
            logger.info(f"✅ Registered baseline {baseline.baseline_id} for {baseline.component_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register baseline {baseline.baseline_id}: {e}")
            return False
    
    async def detect_drift(self, tenant_id: int, component_name: str, 
                          current_content: Dict[str, Any]) -> Optional[DriftDetection]:
        """Detect drift for a specific component"""
        try:
            # Find baseline for component
            component_key = f"{tenant_id}_{component_name}"
            component_info = self.monitored_components.get(component_key)
            
            if not component_info:
                logger.warning(f"No baseline found for component {component_name} in tenant {tenant_id}")
                return None
            
            baseline = self.baselines.get(component_info["baseline_id"])
            if not baseline:
                logger.error(f"Baseline {component_info['baseline_id']} not found")
                return None
            
            # Calculate current content hash
            current_content_str = json.dumps(current_content, sort_keys=True)
            current_hash = hashlib.sha256(current_content_str.encode()).hexdigest()
            
            # Check if drift exists
            if current_hash == baseline.baseline_hash:
                # No drift detected
                return None
            
            # Calculate drift diff
            drift_diff = self._calculate_diff(baseline.baseline_content, current_content)
            
            # Determine severity
            severity = self._calculate_drift_severity(baseline.drift_type, drift_diff)
            
            # Create drift detection
            drift = DriftDetection(
                drift_id=str(uuid.uuid4()),
                baseline_id=baseline.baseline_id,
                drift_type=baseline.drift_type,
                tenant_id=tenant_id,
                component_name=component_name,
                current_content=current_content,
                current_hash=current_hash,
                baseline_hash=baseline.baseline_hash,
                drift_diff=drift_diff,
                severity=severity,
                status=DriftStatus.DETECTED,
                detected_at=datetime.utcnow()
            )
            
            # Store detection
            self.detections[drift.drift_id] = drift
            self.metrics["drift_detections"] += 1
            
            # Auto-create remediation task if enabled
            if self.drift_config["auto_remediation_enabled"] and drift.auto_remediation_enabled:
                remediation_task = await self._create_remediation_task(drift)
                if remediation_task:
                    drift.remediation_task_id = remediation_task.task_id
                    drift.status = DriftStatus.REMEDIATION_CREATED
            
            logger.warning(f"🚨 Drift detected: {drift.drift_id} for {component_name} (severity: {severity.value})")
            
            return drift
            
        except Exception as e:
            logger.error(f"❌ Drift detection failed for {component_name}: {e}")
            self.metrics["detection_errors"] += 1
            return None
    
    def _calculate_diff(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
        """Calculate detailed diff between baseline and current configuration"""
        try:
            baseline_str = json.dumps(baseline, indent=2, sort_keys=True).splitlines()
            current_str = json.dumps(current, indent=2, sort_keys=True).splitlines()
            
            diff = list(difflib.unified_diff(
                baseline_str,
                current_str,
                fromfile='baseline',
                tofile='current',
                lineterm=''
            ))
            
            return diff
            
        except Exception as e:
            logger.error(f"Failed to calculate diff: {e}")
            return [f"Error calculating diff: {str(e)}"]
    
    def _calculate_drift_severity(self, drift_type: DriftType, drift_diff: List[str]) -> DriftSeverity:
        """Calculate drift severity based on type and changes - DYNAMIC rules"""
        try:
            # Count different types of changes
            changes = {
                "additions": len([line for line in drift_diff if line.startswith('+')]),
                "deletions": len([line for line in drift_diff if line.startswith('-')]),
                "total_changes": len([line for line in drift_diff if line.startswith(('+', '-'))])
            }
            
            # Get severity thresholds from configuration
            thresholds = self.drift_config["severity_thresholds"]
            
            if drift_type == DriftType.SCHEMA:
                threshold = thresholds.get("schema_changes", 5)
                if changes["deletions"] > 0:  # Field removals are always high severity
                    return DriftSeverity.HIGH
                elif changes["total_changes"] > threshold * 2:
                    return DriftSeverity.CRITICAL
                elif changes["total_changes"] > threshold:
                    return DriftSeverity.MEDIUM
                else:
                    return DriftSeverity.LOW
            
            elif drift_type == DriftType.POLICY:
                threshold = thresholds.get("policy_changes", 3)
                if changes["deletions"] > threshold:  # Policy removals are critical
                    return DriftSeverity.CRITICAL
                elif changes["total_changes"] > threshold * 2:
                    return DriftSeverity.HIGH
                elif changes["total_changes"] > threshold:
                    return DriftSeverity.MEDIUM
                else:
                    return DriftSeverity.LOW
            
            else:  # CONFIG or other types
                threshold = thresholds.get("config_changes", 10)
                if changes["total_changes"] > threshold * 3:
                    return DriftSeverity.CRITICAL
                elif changes["total_changes"] > threshold * 2:
                    return DriftSeverity.HIGH
                elif changes["total_changes"] > threshold:
                    return DriftSeverity.MEDIUM
                else:
                    return DriftSeverity.LOW
            
        except Exception as e:
            logger.error(f"Failed to calculate drift severity: {e}")
            return DriftSeverity.MEDIUM  # Default to medium severity
    
    async def _create_remediation_task(self, drift: DriftDetection) -> Optional[RemediationTask]:
        """Create auto-remediation task for detected drift"""
        try:
            # Determine task priority based on severity
            priority_mapping = {
                DriftSeverity.CRITICAL: "critical",
                DriftSeverity.HIGH: "high",
                DriftSeverity.MEDIUM: "medium",
                DriftSeverity.LOW: "low"
            }
            
            # Generate remediation actions based on drift type
            remediation_actions = self._generate_remediation_actions(drift)
            
            # Estimate duration based on drift complexity
            estimated_duration = self._estimate_remediation_duration(drift)
            
            task = RemediationTask(
                task_id=str(uuid.uuid4()),
                drift_id=drift.drift_id,
                tenant_id=drift.tenant_id,
                task_type=f"{drift.drift_type.value}_drift_remediation",
                priority=priority_mapping[drift.severity],
                description=f"Auto-remediation for {drift.drift_type.value} drift in {drift.component_name}",
                target_component=drift.component_name,
                remediation_actions=remediation_actions,
                estimated_duration_minutes=estimated_duration,
                due_date=datetime.utcnow() + timedelta(hours=24 if drift.severity == DriftSeverity.CRITICAL else 72)
            )
            
            # Store task
            self.remediation_tasks[task.task_id] = task
            self.metrics["remediation_tasks_created"] += 1
            
            logger.info(f"📋 Created remediation task {task.task_id} for drift {drift.drift_id}")
            
            return task
            
        except Exception as e:
            logger.error(f"❌ Failed to create remediation task for drift {drift.drift_id}: {e}")
            return None
    
    def _generate_remediation_actions(self, drift: DriftDetection) -> List[Dict[str, Any]]:
        """Generate specific remediation actions based on drift type and changes"""
        actions = []
        
        try:
            if drift.drift_type == DriftType.SCHEMA:
                actions.extend([
                    {
                        "action": "validate_schema_changes",
                        "description": "Validate schema changes against compatibility rules",
                        "automated": True
                    },
                    {
                        "action": "update_schema_registry",
                        "description": "Update schema registry with new version",
                        "automated": True
                    },
                    {
                        "action": "notify_stakeholders",
                        "description": "Notify affected teams of schema changes",
                        "automated": True
                    }
                ])
            
            elif drift.drift_type == DriftType.POLICY:
                actions.extend([
                    {
                        "action": "review_policy_changes",
                        "description": "Review policy changes for compliance impact",
                        "automated": False
                    },
                    {
                        "action": "update_policy_registry",
                        "description": "Update policy registry with approved changes",
                        "automated": True
                    },
                    {
                        "action": "audit_policy_enforcement",
                        "description": "Audit policy enforcement after changes",
                        "automated": True
                    }
                ])
            
            elif drift.drift_type == DriftType.CONFIG:
                actions.extend([
                    {
                        "action": "validate_config_changes",
                        "description": "Validate configuration changes",
                        "automated": True
                    },
                    {
                        "action": "apply_config_updates",
                        "description": "Apply validated configuration updates",
                        "automated": True
                    },
                    {
                        "action": "restart_affected_services",
                        "description": "Restart services affected by config changes",
                        "automated": False  # Requires approval
                    }
                ])
            
            # Add common actions
            actions.append({
                "action": "update_baseline",
                "description": "Update baseline after successful remediation",
                "automated": True
            })
            
        except Exception as e:
            logger.error(f"Failed to generate remediation actions: {e}")
            actions = [{
                "action": "manual_review",
                "description": "Manual review required due to error in action generation",
                "automated": False
            }]
        
        return actions
    
    def _estimate_remediation_duration(self, drift: DriftDetection) -> int:
        """Estimate remediation duration in minutes based on drift complexity"""
        base_duration = 30  # Base 30 minutes
        
        # Add time based on number of changes
        change_count = len(drift.drift_diff)
        duration = base_duration + (change_count * 5)  # 5 minutes per change
        
        # Adjust based on severity
        severity_multipliers = {
            DriftSeverity.LOW: 1.0,
            DriftSeverity.MEDIUM: 1.5,
            DriftSeverity.HIGH: 2.0,
            DriftSeverity.CRITICAL: 3.0
        }
        
        duration = int(duration * severity_multipliers[drift.severity])
        
        # Cap at reasonable maximum
        return min(duration, 480)  # Max 8 hours
    
    async def start_drift_monitoring(self):
        """Start background drift monitoring for all registered components"""
        async def drift_monitor():
            while True:
                try:
                    await asyncio.sleep(self.drift_config["detection_interval_seconds"])
                    
                    # Check all monitored components
                    for component_key, component_info in self.monitored_components.items():
                        try:
                            # Check if it's time to scan this component
                            last_check = component_info.get("last_check")
                            check_interval = component_info.get("check_interval", 300)
                            
                            if last_check and (datetime.utcnow() - last_check).total_seconds() < check_interval:
                                continue
                            
                            # Extract tenant_id and component_name
                            tenant_id, component_name = component_key.split("_", 1)
                            tenant_id = int(tenant_id)
                            
                            # Get current configuration (placeholder - would integrate with actual services)
                            current_config = await self._get_current_configuration(tenant_id, component_name)
                            
                            if current_config:
                                # Detect drift
                                drift = await self.detect_drift(tenant_id, component_name, current_config)
                                
                                if drift:
                                    logger.warning(f"🚨 Drift detected in monitoring: {drift.drift_id}")
                            
                            # Update last check time
                            component_info["last_check"] = datetime.utcnow()
                            self.metrics["detection_runs"] += 1
                            
                        except Exception as e:
                            logger.error(f"Error checking component {component_key}: {e}")
                            self.metrics["detection_errors"] += 1
                    
                except Exception as e:
                    logger.error(f"Drift monitoring error: {e}")
        
        # Start monitoring task
        asyncio.create_task(drift_monitor())
        logger.info("🔍 Started drift monitoring for all components")
    
    async def _get_current_configuration(self, tenant_id: int, component_name: str) -> Optional[Dict[str, Any]]:
        """Get current configuration for a component (placeholder)"""
        # This would integrate with actual services to get current configuration
        # For now, return a placeholder configuration
        return {
            "component": component_name,
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "setting1": "value1",
                "setting2": "value2"
            }
        }
    
    def get_drift_status(self) -> Dict[str, Any]:
        """Get comprehensive drift detection status"""
        # Calculate statistics
        detections_by_severity = {}
        detections_by_type = {}
        detections_by_status = {}
        
        for detection in self.detections.values():
            # By severity
            severity = detection.severity.value
            detections_by_severity[severity] = detections_by_severity.get(severity, 0) + 1
            
            # By type
            drift_type = detection.drift_type.value
            detections_by_type[drift_type] = detections_by_type.get(drift_type, 0) + 1
            
            # By status
            status = detection.status.value
            detections_by_status[status] = detections_by_status.get(status, 0) + 1
        
        return {
            "service_status": "active",
            "configuration": self.drift_config.copy(),
            "metrics": self.metrics.copy(),
            "statistics": {
                "total_baselines": len(self.baselines),
                "total_detections": len(self.detections),
                "total_remediation_tasks": len(self.remediation_tasks),
                "monitored_components": len(self.monitored_components),
                "detections_by_severity": detections_by_severity,
                "detections_by_type": detections_by_type,
                "detections_by_status": detections_by_status
            },
            "recent_detections": [
                detection.to_dict() 
                for detection in sorted(
                    self.detections.values(), 
                    key=lambda d: d.detected_at, 
                    reverse=True
                )[:10]
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

# Global drift detection service instance
drift_detection_service = DriftDetectionService()
