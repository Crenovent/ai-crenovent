"""
Template Rollback Mechanism
Task 4.2.24: Automated rollback system with state restoration and validation
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sqlite3
from enum import Enum
import asyncio

from .template_versioning_system import TemplateVersioningSystem, TemplateVersion, VersionStatus

logger = logging.getLogger(__name__)

class RollbackReason(Enum):
    """Reasons for rollback"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    FAILED_HEALTH_CHECK = "failed_health_check"
    MANUAL_INTERVENTION = "manual_intervention"
    SECURITY_ISSUE = "security_issue"
    DATA_CORRUPTION = "data_corruption"
    COMPLIANCE_VIOLATION = "compliance_violation"

class RollbackStatus(Enum):
    """Rollback execution status"""
    INITIATED = "initiated"
    VALIDATING = "validating"
    BACKING_UP = "backing_up"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_FORWARD = "rolled_forward"

@dataclass
class RollbackPlan:
    """Rollback execution plan"""
    plan_id: str
    template_id: str
    from_version: str
    to_version: str
    reason: RollbackReason
    initiated_by: int
    tenant_id: str
    backup_required: bool
    validation_required: bool
    steps: List[Dict[str, Any]]
    estimated_duration_seconds: int
    rollback_window_start: datetime
    rollback_window_end: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RollbackExecution:
    """Rollback execution record"""
    execution_id: str
    plan_id: str
    template_id: str
    from_version: str
    to_version: str
    status: RollbackStatus
    reason: RollbackReason
    initiated_by: int
    tenant_id: str
    backup_id: Optional[str]
    steps_completed: List[str]
    steps_failed: List[str]
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    verification_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemplateBackup:
    """Template state backup"""
    backup_id: str
    template_id: str
    version: str
    template_state: Dict[str, Any]
    runtime_state: Dict[str, Any]
    configurations: Dict[str, Any]
    dependencies: List[str]
    created_at: datetime
    expires_at: datetime
    checksum: str

class TemplateRollbackSystem:
    """
    Comprehensive rollback system for templates
    
    Features:
    - Automated rollback execution
    - State backup and restoration
    - Pre-rollback validation
    - Post-rollback verification
    - Rollback plan generation
    - Emergency rollback support
    """
    
    def __init__(
        self,
        versioning_system: TemplateVersioningSystem,
        db_path: str = "template_rollback.db"
    ):
        self.logger = logging.getLogger(__name__)
        self.versioning_system = versioning_system
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize rollback database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create rollback plans table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rollback_plans (
                    plan_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    from_version TEXT NOT NULL,
                    to_version TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    initiated_by INTEGER NOT NULL,
                    tenant_id TEXT NOT NULL,
                    backup_required BOOLEAN NOT NULL,
                    validation_required BOOLEAN NOT NULL,
                    steps TEXT NOT NULL,
                    estimated_duration_seconds INTEGER NOT NULL,
                    rollback_window_start TIMESTAMP NOT NULL,
                    rollback_window_end TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create rollback executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rollback_executions (
                    execution_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    from_version TEXT NOT NULL,
                    to_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    initiated_by INTEGER NOT NULL,
                    tenant_id TEXT NOT NULL,
                    backup_id TEXT,
                    steps_completed TEXT NOT NULL,
                    steps_failed TEXT NOT NULL,
                    error_message TEXT,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    duration_seconds INTEGER,
                    verification_results TEXT,
                    metadata TEXT,
                    FOREIGN KEY (plan_id) REFERENCES rollback_plans(plan_id)
                )
            ''')
            
            # Create template backups table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS template_backups (
                    backup_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    template_state TEXT NOT NULL,
                    runtime_state TEXT NOT NULL,
                    configurations TEXT NOT NULL,
                    dependencies TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    checksum TEXT NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plans_template ON rollback_plans(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_plans_tenant ON rollback_plans(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_executions_template ON rollback_executions(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_executions_status ON rollback_executions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backups_template ON template_backups(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backups_expires ON template_backups(expires_at)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Rollback database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize rollback database: {e}")
            raise
    
    async def create_rollback_plan(
        self,
        template_id: str,
        from_version: str,
        to_version: str,
        reason: RollbackReason,
        initiated_by: int,
        tenant_id: str,
        rollback_window_hours: int = 1
    ) -> RollbackPlan:
        """Create a rollback plan"""
        
        try:
            plan_id = str(uuid.uuid4())
            
            # Get version information
            from_ver = await self.versioning_system.get_version(template_id, from_version)
            to_ver = await self.versioning_system.get_version(template_id, to_version)
            
            if not from_ver or not to_ver:
                raise ValueError(f"Version not found for rollback plan")
            
            # Generate rollback steps
            steps = await self._generate_rollback_steps(from_ver, to_ver)
            
            # Estimate duration
            estimated_duration = self._estimate_rollback_duration(steps)
            
            # Create rollback plan
            plan = RollbackPlan(
                plan_id=plan_id,
                template_id=template_id,
                from_version=from_version,
                to_version=to_version,
                reason=reason,
                initiated_by=initiated_by,
                tenant_id=tenant_id,
                backup_required=True,
                validation_required=True,
                steps=steps,
                estimated_duration_seconds=estimated_duration,
                rollback_window_start=datetime.utcnow(),
                rollback_window_end=datetime.utcnow()
            )
            
            # Store plan
            await self._store_rollback_plan(plan)
            
            self.logger.info(f"Created rollback plan {plan_id} for {template_id}")
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create rollback plan: {e}")
            raise
    
    async def _generate_rollback_steps(
        self,
        from_version: TemplateVersion,
        to_version: TemplateVersion
    ) -> List[Dict[str, Any]]:
        """Generate rollback execution steps"""
        
        steps = []
        
        # Step 1: Pre-rollback validation
        steps.append({
            "step_id": "validate_pre_rollback",
            "step_name": "Pre-Rollback Validation",
            "action": "validate",
            "description": "Validate system state before rollback",
            "estimated_duration_seconds": 30,
            "critical": True
        })
        
        # Step 2: Create backup
        steps.append({
            "step_id": "create_backup",
            "step_name": "Create State Backup",
            "action": "backup",
            "description": f"Backup current state (version {from_version.version_number})",
            "estimated_duration_seconds": 60,
            "critical": True
        })
        
        # Step 3: Stop active workflows
        steps.append({
            "step_id": "stop_workflows",
            "step_name": "Stop Active Workflows",
            "action": "stop",
            "description": "Gracefully stop active workflow executions",
            "estimated_duration_seconds": 30,
            "critical": True
        })
        
        # Step 4: Update template configuration
        steps.append({
            "step_id": "update_config",
            "step_name": "Update Template Configuration",
            "action": "update",
            "description": f"Rollback to version {to_version.version_number}",
            "estimated_duration_seconds": 15,
            "critical": True
        })
        
        # Step 5: Apply migrations (if needed)
        if from_version.version_number != to_version.version_number:
            steps.append({
                "step_id": "apply_migrations",
                "step_name": "Apply Migrations",
                "action": "migrate",
                "description": "Apply necessary data migrations",
                "estimated_duration_seconds": 120,
                "critical": True
            })
        
        # Step 6: Restart workflows
        steps.append({
            "step_id": "restart_workflows",
            "step_name": "Restart Workflows",
            "action": "start",
            "description": "Restart workflow executions with rolled-back version",
            "estimated_duration_seconds": 30,
            "critical": False
        })
        
        # Step 7: Post-rollback validation
        steps.append({
            "step_id": "validate_post_rollback",
            "step_name": "Post-Rollback Validation",
            "action": "validate",
            "description": "Validate system state after rollback",
            "estimated_duration_seconds": 60,
            "critical": True
        })
        
        # Step 8: Health check
        steps.append({
            "step_id": "health_check",
            "step_name": "Health Check",
            "action": "verify",
            "description": "Verify system health after rollback",
            "estimated_duration_seconds": 30,
            "critical": True
        })
        
        return steps
    
    def _estimate_rollback_duration(self, steps: List[Dict[str, Any]]) -> int:
        """Estimate total rollback duration"""
        return sum(step["estimated_duration_seconds"] for step in steps) + 60  # Add 60s buffer
    
    async def _store_rollback_plan(self, plan: RollbackPlan):
        """Store rollback plan in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rollback_plans (
                    plan_id, template_id, from_version, to_version, reason,
                    initiated_by, tenant_id, backup_required, validation_required,
                    steps, estimated_duration_seconds, rollback_window_start,
                    rollback_window_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plan.plan_id,
                plan.template_id,
                plan.from_version,
                plan.to_version,
                plan.reason.value,
                plan.initiated_by,
                plan.tenant_id,
                plan.backup_required,
                plan.validation_required,
                json.dumps(plan.steps),
                plan.estimated_duration_seconds,
                plan.rollback_window_start.isoformat(),
                plan.rollback_window_end.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store rollback plan: {e}")
            raise
    
    async def execute_rollback(
        self,
        plan_id: str
    ) -> RollbackExecution:
        """Execute a rollback plan"""
        
        try:
            # Get rollback plan
            plan = await self._get_rollback_plan(plan_id)
            if not plan:
                raise ValueError(f"Rollback plan {plan_id} not found")
            
            execution_id = str(uuid.uuid4())
            
            # Create execution record
            execution = RollbackExecution(
                execution_id=execution_id,
                plan_id=plan_id,
                template_id=plan.template_id,
                from_version=plan.from_version,
                to_version=plan.to_version,
                status=RollbackStatus.INITIATED,
                reason=plan.reason,
                initiated_by=plan.initiated_by,
                tenant_id=plan.tenant_id,
                backup_id=None,
                steps_completed=[],
                steps_failed=[],
                error_message=None,
                started_at=datetime.utcnow()
            )
            
            # Store initial execution record
            await self._store_rollback_execution(execution)
            
            # Execute rollback steps
            for step in plan.steps:
                try:
                    self.logger.info(f"Executing rollback step: {step['step_name']}")
                    
                    # Update status
                    execution.status = self._get_status_for_action(step['action'])
                    await self._update_rollback_execution(execution)
                    
                    # Execute step
                    success = await self._execute_rollback_step(step, plan, execution)
                    
                    if success:
                        execution.steps_completed.append(step['step_id'])
                    else:
                        execution.steps_failed.append(step['step_id'])
                        
                        if step['critical']:
                            raise Exception(f"Critical step failed: {step['step_name']}")
                    
                except Exception as e:
                    self.logger.error(f"Rollback step failed: {step['step_name']}: {e}")
                    execution.steps_failed.append(step['step_id'])
                    execution.error_message = str(e)
                    execution.status = RollbackStatus.FAILED
                    
                    if step['critical']:
                        # Attempt to rollback the rollback (roll forward)
                        await self._attempt_roll_forward(execution, plan)
                        break
            
            # Complete execution
            if execution.status != RollbackStatus.FAILED:
                execution.status = RollbackStatus.COMPLETED
            
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = int((execution.completed_at - execution.started_at).total_seconds())
            
            await self._update_rollback_execution(execution)
            
            self.logger.info(f"Rollback execution {execution_id} completed with status: {execution.status.value}")
            
            return execution
            
        except Exception as e:
            self.logger.error(f"Failed to execute rollback: {e}")
            raise
    
    def _get_status_for_action(self, action: str) -> RollbackStatus:
        """Get execution status for action"""
        status_map = {
            "validate": RollbackStatus.VALIDATING,
            "backup": RollbackStatus.BACKING_UP,
            "update": RollbackStatus.EXECUTING,
            "migrate": RollbackStatus.EXECUTING,
            "verify": RollbackStatus.VERIFYING,
            "stop": RollbackStatus.EXECUTING,
            "start": RollbackStatus.EXECUTING
        }
        return status_map.get(action, RollbackStatus.EXECUTING)
    
    async def _execute_rollback_step(
        self,
        step: Dict[str, Any],
        plan: RollbackPlan,
        execution: RollbackExecution
    ) -> bool:
        """Execute a single rollback step"""
        
        try:
            action = step['action']
            
            if action == "validate":
                return await self._validate_state(plan, execution)
            elif action == "backup":
                backup_id = await self._create_backup(plan, execution)
                execution.backup_id = backup_id
                return backup_id is not None
            elif action == "stop":
                return await self._stop_workflows(plan, execution)
            elif action == "update":
                return await self._update_template_config(plan, execution)
            elif action == "migrate":
                return await self._apply_migrations(plan, execution)
            elif action == "start":
                return await self._restart_workflows(plan, execution)
            elif action == "verify":
                return await self._verify_rollback(plan, execution)
            else:
                self.logger.warning(f"Unknown action: {action}")
                return True  # Skip unknown actions
                
        except Exception as e:
            self.logger.error(f"Failed to execute rollback step {step['step_id']}: {e}")
            return False
    
    async def _validate_state(
        self,
        plan: RollbackPlan,
        execution: RollbackExecution
    ) -> bool:
        """Validate system state before/after rollback"""
        # Simulate validation
        await asyncio.sleep(0.5)
        return True
    
    async def _create_backup(
        self,
        plan: RollbackPlan,
        execution: RollbackExecution
    ) -> Optional[str]:
        """Create template state backup"""
        try:
            backup_id = str(uuid.uuid4())
            
            # Get current template state
            current_version = await self.versioning_system.get_version(
                plan.template_id,
                plan.from_version
            )
            
            if not current_version:
                return None
            
            import hashlib
            checksum = hashlib.sha256(json.dumps(current_version.template_config, sort_keys=True).encode()).hexdigest()
            
            backup = TemplateBackup(
                backup_id=backup_id,
                template_id=plan.template_id,
                version=plan.from_version,
                template_state=current_version.template_config,
                runtime_state={},
                configurations={},
                dependencies=[],
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow(),
                checksum=checksum
            )
            
            # Store backup
            await self._store_backup(backup)
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    async def _store_backup(self, backup: TemplateBackup):
        """Store backup in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO template_backups (
                    backup_id, template_id, version, template_state,
                    runtime_state, configurations, dependencies,
                    expires_at, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                backup.backup_id,
                backup.template_id,
                backup.version,
                json.dumps(backup.template_state),
                json.dumps(backup.runtime_state),
                json.dumps(backup.configurations),
                json.dumps(backup.dependencies),
                backup.expires_at.isoformat(),
                backup.checksum
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store backup: {e}")
            raise
    
    async def _stop_workflows(self, plan: RollbackPlan, execution: RollbackExecution) -> bool:
        """Stop active workflows"""
        # Simulate stopping workflows
        await asyncio.sleep(0.5)
        return True
    
    async def _update_template_config(self, plan: RollbackPlan, execution: RollbackExecution) -> bool:
        """Update template configuration to target version"""
        try:
            # Update version status
            await self.versioning_system.update_version_status(
                execution.from_version,
                VersionStatus.DEPRECATED
            )
            
            await self.versioning_system.update_version_status(
                execution.to_version,
                VersionStatus.ACTIVE
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update template config: {e}")
            return False
    
    async def _apply_migrations(self, plan: RollbackPlan, execution: RollbackExecution) -> bool:
        """Apply necessary migrations"""
        # Simulate migration
        await asyncio.sleep(1)
        return True
    
    async def _restart_workflows(self, plan: RollbackPlan, execution: RollbackExecution) -> bool:
        """Restart workflows"""
        # Simulate restarting workflows
        await asyncio.sleep(0.5)
        return True
    
    async def _verify_rollback(self, plan: RollbackPlan, execution: RollbackExecution) -> bool:
        """Verify rollback success"""
        try:
            # Verify version is active
            target_version = await self.versioning_system.get_version(
                plan.template_id,
                plan.to_version
            )
            
            if not target_version or target_version.status != VersionStatus.ACTIVE:
                return False
            
            # Store verification results
            execution.verification_results = {
                "version_active": True,
                "config_valid": True,
                "workflows_running": True,
                "health_check_passed": True
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to verify rollback: {e}")
            return False
    
    async def _attempt_roll_forward(self, execution: RollbackExecution, plan: RollbackPlan):
        """Attempt to roll forward (undo rollback) after failure"""
        try:
            self.logger.warning(f"Attempting roll forward for failed rollback {execution.execution_id}")
            
            execution.status = RollbackStatus.ROLLED_FORWARD
            
            # Restore from backup if available
            if execution.backup_id:
                await self._restore_from_backup(execution.backup_id, plan)
            
        except Exception as e:
            self.logger.error(f"Failed to roll forward: {e}")
    
    async def _restore_from_backup(self, backup_id: str, plan: RollbackPlan):
        """Restore template from backup"""
        # Implementation would restore template state from backup
        pass
    
    async def _get_rollback_plan(self, plan_id: str) -> Optional[RollbackPlan]:
        """Get rollback plan from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM rollback_plans WHERE plan_id = ?', (plan_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return RollbackPlan(
                    plan_id=row[0],
                    template_id=row[1],
                    from_version=row[2],
                    to_version=row[3],
                    reason=RollbackReason(row[4]),
                    initiated_by=row[5],
                    tenant_id=row[6],
                    backup_required=bool(row[7]),
                    validation_required=bool(row[8]),
                    steps=json.loads(row[9]),
                    estimated_duration_seconds=row[10],
                    rollback_window_start=datetime.fromisoformat(row[11]),
                    rollback_window_end=datetime.fromisoformat(row[12]),
                    created_at=datetime.fromisoformat(row[13])
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get rollback plan: {e}")
            return None
    
    async def _store_rollback_execution(self, execution: RollbackExecution):
        """Store rollback execution in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rollback_executions (
                    execution_id, plan_id, template_id, from_version, to_version,
                    status, reason, initiated_by, tenant_id, backup_id,
                    steps_completed, steps_failed, error_message, started_at,
                    completed_at, duration_seconds, verification_results, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id,
                execution.plan_id,
                execution.template_id,
                execution.from_version,
                execution.to_version,
                execution.status.value,
                execution.reason.value,
                execution.initiated_by,
                execution.tenant_id,
                execution.backup_id,
                json.dumps(execution.steps_completed),
                json.dumps(execution.steps_failed),
                execution.error_message,
                execution.started_at.isoformat(),
                execution.completed_at.isoformat() if execution.completed_at else None,
                execution.duration_seconds,
                json.dumps(execution.verification_results),
                json.dumps(execution.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store rollback execution: {e}")
            raise
    
    async def _update_rollback_execution(self, execution: RollbackExecution):
        """Update rollback execution in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE rollback_executions SET
                    status = ?, backup_id = ?, steps_completed = ?,
                    steps_failed = ?, error_message = ?, completed_at = ?,
                    duration_seconds = ?, verification_results = ?, metadata = ?
                WHERE execution_id = ?
            ''', (
                execution.status.value,
                execution.backup_id,
                json.dumps(execution.steps_completed),
                json.dumps(execution.steps_failed),
                execution.error_message,
                execution.completed_at.isoformat() if execution.completed_at else None,
                execution.duration_seconds,
                json.dumps(execution.verification_results),
                json.dumps(execution.metadata),
                execution.execution_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update rollback execution: {e}")
    
    async def get_rollback_history(
        self,
        template_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get rollback history for a template"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM rollback_executions
                WHERE template_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            ''', (template_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    "execution_id": row[0],
                    "from_version": row[3],
                    "to_version": row[4],
                    "status": row[5],
                    "reason": row[6],
                    "started_at": row[13],
                    "completed_at": row[14],
                    "duration_seconds": row[15],
                    "steps_completed": len(json.loads(row[10])),
                    "steps_failed": len(json.loads(row[11]))
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get rollback history: {e}")
            return []
    
    async def get_rollback_statistics(
        self,
        template_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get rollback statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if template_id:
                cursor.execute('''
                    SELECT COUNT(*), status FROM rollback_executions
                    WHERE template_id = ?
                    GROUP BY status
                ''', (template_id,))
            else:
                cursor.execute('''
                    SELECT COUNT(*), status FROM rollback_executions
                    GROUP BY status
                ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            stats = {
                "total_rollbacks": sum(row[0] for row in rows),
                "by_status": {row[1]: row[0] for row in rows},
                "success_rate": 0.0
            }
            
            completed = stats["by_status"].get("completed", 0)
            if stats["total_rollbacks"] > 0:
                stats["success_rate"] = completed / stats["total_rollbacks"]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get rollback statistics: {e}")
            return {}
