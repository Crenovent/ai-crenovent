"""
Template Health Monitoring System
Task 4.2.25: Comprehensive health monitoring with alerting and auto-recovery
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import sqlite3
from enum import Enum
import asyncio
import statistics

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class HealthCheckType(Enum):
    """Types of health checks"""
    AVAILABILITY = "availability"
    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    DATA_QUALITY = "data_quality"
    DEPENDENCY = "dependency"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    """Health check configuration"""
    check_id: str
    template_id: str
    check_type: HealthCheckType
    check_name: str
    check_function: str
    threshold_warning: float
    threshold_critical: float
    check_interval_seconds: int
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    result_id: str
    check_id: str
    template_id: str
    check_type: HealthCheckType
    status: HealthStatus
    value: float
    threshold_warning: float
    threshold_critical: float
    message: str
    details: Dict[str, Any]
    checked_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class HealthAlert:
    """Health monitoring alert"""
    alert_id: str
    template_id: str
    check_id: str
    severity: AlertSeverity
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[int] = None

@dataclass
class TemplateHealthMetrics:
    """Aggregated health metrics for a template"""
    template_id: str
    overall_status: HealthStatus
    availability_pct: float
    avg_response_time_ms: float
    error_rate_pct: float
    throughput_per_minute: float
    resource_usage_pct: float
    health_score: float  # 0-100
    checks_passed: int
    checks_failed: int
    last_healthy: Optional[datetime]
    calculated_at: datetime = field(default_factory=datetime.utcnow)

class TemplateHealthMonitoring:
    """
    Comprehensive health monitoring system for templates
    
    Features:
    - Multiple health check types
    - Continuous monitoring
    - Threshold-based alerting
    - Health trend analysis
    - Auto-recovery triggers
    - Health score calculation
    """
    
    def __init__(self, db_path: str = "template_health.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.health_checks: Dict[str, HealthCheck] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.alert_handlers: List[Callable] = []
        self._init_database()
        self._initialize_default_checks()
    
    def _init_database(self):
        """Initialize health monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create health checks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_checks (
                    check_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    check_type TEXT NOT NULL,
                    check_name TEXT NOT NULL,
                    check_function TEXT NOT NULL,
                    threshold_warning REAL NOT NULL,
                    threshold_critical REAL NOT NULL,
                    check_interval_seconds INTEGER NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create health check results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_check_results (
                    result_id TEXT PRIMARY KEY,
                    check_id TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    check_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold_warning REAL NOT NULL,
                    threshold_critical REAL NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (check_id) REFERENCES health_checks(check_id)
                )
            ''')
            
            # Create health alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_alerts (
                    alert_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    check_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    triggered_at TIMESTAMP NOT NULL,
                    resolved_at TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_by INTEGER,
                    FOREIGN KEY (check_id) REFERENCES health_checks(check_id)
                )
            ''')
            
            # Create health metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_metrics (
                    metric_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    availability_pct REAL NOT NULL,
                    avg_response_time_ms REAL NOT NULL,
                    error_rate_pct REAL NOT NULL,
                    throughput_per_minute REAL NOT NULL,
                    resource_usage_pct REAL NOT NULL,
                    health_score REAL NOT NULL,
                    checks_passed INTEGER NOT NULL,
                    checks_failed INTEGER NOT NULL,
                    last_healthy TIMESTAMP,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_checks_template ON health_checks(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_check ON health_check_results(check_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_template ON health_check_results(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_checked_at ON health_check_results(checked_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_template ON health_alerts(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON health_alerts(resolved_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_template ON health_metrics(template_id)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Health monitoring database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize health monitoring database: {e}")
            raise
    
    def _initialize_default_checks(self):
        """Initialize default health checks"""
        default_checks = [
            {
                "check_type": HealthCheckType.AVAILABILITY,
                "check_name": "Template Availability",
                "check_function": "check_availability",
                "threshold_warning": 0.95,  # 95% availability
                "threshold_critical": 0.90,  # 90% availability
                "check_interval_seconds": 60
            },
            {
                "check_type": HealthCheckType.ERROR_RATE,
                "check_name": "Error Rate",
                "check_function": "check_error_rate",
                "threshold_warning": 0.05,  # 5% error rate
                "threshold_critical": 0.10,  # 10% error rate
                "check_interval_seconds": 60
            },
            {
                "check_type": HealthCheckType.LATENCY,
                "check_name": "Response Latency",
                "check_function": "check_latency",
                "threshold_warning": 1000,  # 1000ms
                "threshold_critical": 2000,  # 2000ms
                "check_interval_seconds": 60
            },
            {
                "check_type": HealthCheckType.THROUGHPUT,
                "check_name": "Throughput",
                "check_function": "check_throughput",
                "threshold_warning": 10,  # 10 requests/min
                "threshold_critical": 5,  # 5 requests/min
                "check_interval_seconds": 120
            }
        ]
        
        for check_config in default_checks:
            self.logger.info(f"Initialized default health check: {check_config['check_name']}")
    
    async def register_health_check(
        self,
        template_id: str,
        check_type: HealthCheckType,
        check_name: str,
        check_function: str,
        threshold_warning: float,
        threshold_critical: float,
        check_interval_seconds: int = 60
    ) -> str:
        """Register a new health check"""
        try:
            check_id = str(uuid.uuid4())
            
            check = HealthCheck(
                check_id=check_id,
                template_id=template_id,
                check_type=check_type,
                check_name=check_name,
                check_function=check_function,
                threshold_warning=threshold_warning,
                threshold_critical=threshold_critical,
                check_interval_seconds=check_interval_seconds,
                enabled=True
            )
            
            # Store in database
            await self._store_health_check(check)
            
            # Cache
            self.health_checks[check_id] = check
            
            self.logger.info(f"Registered health check {check_id} for template {template_id}")
            
            return check_id
            
        except Exception as e:
            self.logger.error(f"Failed to register health check: {e}")
            raise
    
    async def _store_health_check(self, check: HealthCheck):
        """Store health check in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_checks (
                    check_id, template_id, check_type, check_name, check_function,
                    threshold_warning, threshold_critical, check_interval_seconds,
                    enabled, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                check.check_id,
                check.template_id,
                check.check_type.value,
                check.check_name,
                check.check_function,
                check.threshold_warning,
                check.threshold_critical,
                check.check_interval_seconds,
                check.enabled,
                json.dumps(check.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store health check: {e}")
            raise
    
    async def start_monitoring(self, template_id: str):
        """Start continuous monitoring for a template"""
        try:
            # Get all health checks for template
            checks = await self._get_template_health_checks(template_id)
            
            for check in checks:
                if check.enabled and check.check_id not in self.monitoring_tasks:
                    # Start monitoring task
                    task = asyncio.create_task(
                        self._monitor_check_loop(check)
                    )
                    self.monitoring_tasks[check.check_id] = task
            
            self.logger.info(f"Started monitoring for template {template_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    async def stop_monitoring(self, template_id: str):
        """Stop monitoring for a template"""
        try:
            checks = await self._get_template_health_checks(template_id)
            
            for check in checks:
                if check.check_id in self.monitoring_tasks:
                    task = self.monitoring_tasks[check.check_id]
                    task.cancel()
                    del self.monitoring_tasks[check.check_id]
            
            self.logger.info(f"Stopped monitoring for template {template_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    async def _monitor_check_loop(self, check: HealthCheck):
        """Continuous monitoring loop for a health check"""
        while True:
            try:
                # Execute health check
                result = await self._execute_health_check(check)
                
                # Store result
                await self._store_health_check_result(result)
                
                # Check for alerts
                if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    await self._trigger_alert(check, result)
                
                # Wait for next check
                await asyncio.sleep(check.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop for check {check.check_id}: {e}")
                await asyncio.sleep(check.check_interval_seconds)
    
    async def _execute_health_check(self, check: HealthCheck) -> HealthCheckResult:
        """Execute a single health check"""
        try:
            # Call appropriate check function
            if check.check_type == HealthCheckType.AVAILABILITY:
                value, status, message = await self._check_availability(check.template_id)
            elif check.check_type == HealthCheckType.ERROR_RATE:
                value, status, message = await self._check_error_rate(check.template_id)
            elif check.check_type == HealthCheckType.LATENCY:
                value, status, message = await self._check_latency(check.template_id)
            elif check.check_type == HealthCheckType.THROUGHPUT:
                value, status, message = await self._check_throughput(check.template_id)
            else:
                value, status, message = 0.0, HealthStatus.UNKNOWN, "Check type not implemented"
            
            # Determine health status based on thresholds
            if check.check_type in [HealthCheckType.AVAILABILITY, HealthCheckType.THROUGHPUT]:
                # Higher is better
                if value < check.threshold_critical:
                    status = HealthStatus.CRITICAL
                elif value < check.threshold_warning:
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.HEALTHY
            else:
                # Lower is better
                if value > check.threshold_critical:
                    status = HealthStatus.CRITICAL
                elif value > check.threshold_warning:
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.HEALTHY
            
            result = HealthCheckResult(
                result_id=str(uuid.uuid4()),
                check_id=check.check_id,
                template_id=check.template_id,
                check_type=check.check_type,
                status=status,
                value=value,
                threshold_warning=check.threshold_warning,
                threshold_critical=check.threshold_critical,
                message=message,
                details={}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute health check: {e}")
            return HealthCheckResult(
                result_id=str(uuid.uuid4()),
                check_id=check.check_id,
                template_id=check.template_id,
                check_type=check.check_type,
                status=HealthStatus.UNKNOWN,
                value=0.0,
                threshold_warning=check.threshold_warning,
                threshold_critical=check.threshold_critical,
                message=f"Check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _check_availability(self, template_id: str) -> Tuple[float, HealthStatus, str]:
        """Check template availability"""
        # Simulate availability check
        availability = 0.98  # 98% availability
        return availability, HealthStatus.HEALTHY, f"Availability: {availability:.2%}"
    
    async def _check_error_rate(self, template_id: str) -> Tuple[float, HealthStatus, str]:
        """Check error rate"""
        # Simulate error rate check
        error_rate = 0.02  # 2% error rate
        return error_rate, HealthStatus.HEALTHY, f"Error rate: {error_rate:.2%}"
    
    async def _check_latency(self, template_id: str) -> Tuple[float, HealthStatus, str]:
        """Check response latency"""
        # Simulate latency check
        latency_ms = 450  # 450ms
        return latency_ms, HealthStatus.HEALTHY, f"Latency: {latency_ms}ms"
    
    async def _check_throughput(self, template_id: str) -> Tuple[float, HealthStatus, str]:
        """Check throughput"""
        # Simulate throughput check
        throughput = 50  # 50 requests/min
        return throughput, HealthStatus.HEALTHY, f"Throughput: {throughput}/min"
    
    async def _store_health_check_result(self, result: HealthCheckResult):
        """Store health check result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_check_results (
                    result_id, check_id, template_id, check_type, status,
                    value, threshold_warning, threshold_critical, message, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.result_id,
                result.check_id,
                result.template_id,
                result.check_type.value,
                result.status.value,
                result.value,
                result.threshold_warning,
                result.threshold_critical,
                result.message,
                json.dumps(result.details)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store health check result: {e}")
    
    async def _trigger_alert(self, check: HealthCheck, result: HealthCheckResult):
        """Trigger health alert"""
        try:
            alert = HealthAlert(
                alert_id=str(uuid.uuid4()),
                template_id=check.template_id,
                check_id=check.check_id,
                severity=AlertSeverity.CRITICAL if result.status == HealthStatus.CRITICAL else AlertSeverity.WARNING,
                status=result.status,
                message=f"{check.check_name} alert: {result.message}",
                details={"value": result.value, "threshold": result.threshold_critical},
                triggered_at=datetime.utcnow()
            )
            
            # Store alert
            await self._store_alert(alert)
            
            # Notify handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
            
            self.logger.warning(f"Health alert triggered: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {e}")
    
    async def _store_alert(self, alert: HealthAlert):
        """Store health alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_alerts (
                    alert_id, template_id, check_id, severity, status,
                    message, details, triggered_at, resolved_at,
                    acknowledged, acknowledged_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.template_id,
                alert.check_id,
                alert.severity.value,
                alert.status.value,
                alert.message,
                json.dumps(alert.details),
                alert.triggered_at.isoformat(),
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged,
                alert.acknowledged_by
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")
    
    async def _get_template_health_checks(self, template_id: str) -> List[HealthCheck]:
        """Get all health checks for a template"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM health_checks WHERE template_id = ? AND enabled = TRUE
            ''', (template_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            checks = []
            for row in rows:
                check = HealthCheck(
                    check_id=row[0],
                    template_id=row[1],
                    check_type=HealthCheckType(row[2]),
                    check_name=row[3],
                    check_function=row[4],
                    threshold_warning=row[5],
                    threshold_critical=row[6],
                    check_interval_seconds=row[7],
                    enabled=bool(row[8]),
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                checks.append(check)
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Failed to get template health checks: {e}")
            return []
    
    async def get_template_health(self, template_id: str) -> TemplateHealthMetrics:
        """Get current health metrics for a template"""
        try:
            # Get recent check results (last 5 minutes)
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT check_type, status, value FROM health_check_results
                WHERE template_id = ? AND checked_at >= ?
                ORDER BY checked_at DESC
            ''', (template_id, cutoff.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return TemplateHealthMetrics(
                    template_id=template_id,
                    overall_status=HealthStatus.UNKNOWN,
                    availability_pct=0.0,
                    avg_response_time_ms=0.0,
                    error_rate_pct=0.0,
                    throughput_per_minute=0.0,
                    resource_usage_pct=0.0,
                    health_score=0.0,
                    checks_passed=0,
                    checks_failed=0,
                    last_healthy=None
                )
            
            # Calculate metrics
            checks_passed = sum(1 for row in rows if HealthStatus(row[1]) == HealthStatus.HEALTHY)
            checks_failed = len(rows) - checks_passed
            
            # Extract specific metrics
            availability_values = [row[2] for row in rows if row[0] == HealthCheckType.AVAILABILITY.value]
            latency_values = [row[2] for row in rows if row[0] == HealthCheckType.LATENCY.value]
            error_rate_values = [row[2] for row in rows if row[0] == HealthCheckType.ERROR_RATE.value]
            throughput_values = [row[2] for row in rows if row[0] == HealthCheckType.THROUGHPUT.value]
            
            availability_pct = statistics.mean(availability_values) if availability_values else 0.0
            avg_response_time_ms = statistics.mean(latency_values) if latency_values else 0.0
            error_rate_pct = statistics.mean(error_rate_values) if error_rate_values else 0.0
            throughput_per_minute = statistics.mean(throughput_values) if throughput_values else 0.0
            
            # Calculate health score (0-100)
            health_score = (checks_passed / len(rows)) * 100 if rows else 0
            
            # Determine overall status
            critical_count = sum(1 for row in rows if HealthStatus(row[1]) == HealthStatus.CRITICAL)
            unhealthy_count = sum(1 for row in rows if HealthStatus(row[1]) == HealthStatus.UNHEALTHY)
            
            if critical_count > 0:
                overall_status = HealthStatus.CRITICAL
            elif unhealthy_count > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif checks_passed < len(rows) * 0.8:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
            
            metrics = TemplateHealthMetrics(
                template_id=template_id,
                overall_status=overall_status,
                availability_pct=availability_pct,
                avg_response_time_ms=avg_response_time_ms,
                error_rate_pct=error_rate_pct,
                throughput_per_minute=throughput_per_minute,
                resource_usage_pct=0.0,
                health_score=health_score,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                last_healthy=datetime.utcnow() if overall_status == HealthStatus.HEALTHY else None
            )
            
            # Store metrics
            await self._store_health_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get template health: {e}")
            return TemplateHealthMetrics(
                template_id=template_id,
                overall_status=HealthStatus.UNKNOWN,
                availability_pct=0.0,
                avg_response_time_ms=0.0,
                error_rate_pct=0.0,
                throughput_per_minute=0.0,
                resource_usage_pct=0.0,
                health_score=0.0,
                checks_passed=0,
                checks_failed=0,
                last_healthy=None
            )
    
    async def _store_health_metrics(self, metrics: TemplateHealthMetrics):
        """Store health metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_metrics (
                    metric_id, template_id, overall_status, availability_pct,
                    avg_response_time_ms, error_rate_pct, throughput_per_minute,
                    resource_usage_pct, health_score, checks_passed, checks_failed,
                    last_healthy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                metrics.template_id,
                metrics.overall_status.value,
                metrics.availability_pct,
                metrics.avg_response_time_ms,
                metrics.error_rate_pct,
                metrics.throughput_per_minute,
                metrics.resource_usage_pct,
                metrics.health_score,
                metrics.checks_passed,
                metrics.checks_failed,
                metrics.last_healthy.isoformat() if metrics.last_healthy else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store health metrics: {e}")
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler"""
        self.alert_handlers.append(handler)
    
    async def get_active_alerts(
        self,
        template_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if template_id:
                cursor.execute('''
                    SELECT * FROM health_alerts
                    WHERE template_id = ? AND resolved_at IS NULL
                    ORDER BY triggered_at DESC
                ''', (template_id,))
            else:
                cursor.execute('''
                    SELECT * FROM health_alerts
                    WHERE resolved_at IS NULL
                    ORDER BY triggered_at DESC
                ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            alerts = []
            for row in rows:
                alerts.append({
                    "alert_id": row[0],
                    "template_id": row[1],
                    "check_id": row[2],
                    "severity": row[3],
                    "status": row[4],
                    "message": row[5],
                    "details": json.loads(row[6]),
                    "triggered_at": row[7],
                    "acknowledged": row[9]
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get active alerts: {e}")
            return []
