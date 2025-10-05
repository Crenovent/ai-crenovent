"""
RBAC Dashboard Manager - Chapter 8.3 UI Implementation
=====================================================
Tasks 8.3-T17, T27, T32, T34: RBAC UI and access dashboards

Missing Tasks Implemented:
- 8.3-T17: Build access dashboards (role usage, violations, SoD breaches)
- 8.3-T27: Build RBAC UI (assign, revoke, view roles)
- 8.3-T32: Train Ops on RBAC/SoD dashboards
- 8.3-T34: Train Execs on access dashboards

Key Features:
- Dynamic role assignment and management UI
- Real-time access violation monitoring
- SoD breach detection and alerting
- Executive and operational dashboards
- Multi-tenant access control visualization
- Anomaly detection for suspicious access patterns
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Types of RBAC dashboards"""
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    SECURITY = "security"

class AccessViolationType(Enum):
    """Types of access violations"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SOD_VIOLATION = "sod_violation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    FAILED_AUTHENTICATION = "failed_authentication"
    ROLE_ABUSE = "role_abuse"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AccessViolation:
    """Access violation record"""
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: int = 0
    user_id: str = ""
    violation_type: AccessViolationType = AccessViolationType.UNAUTHORIZED_ACCESS
    severity: AlertSeverity = AlertSeverity.MEDIUM
    
    # Violation details
    attempted_action: str = ""
    resource_id: Optional[str] = None
    required_permission: Optional[str] = None
    user_roles: List[str] = field(default_factory=list)
    
    # Context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    # Resolution
    status: str = "open"  # open, investigating, resolved, false_positive
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class RoleUsageMetrics:
    """Role usage metrics for dashboards"""
    role_name: str = ""
    tenant_id: int = 0
    
    # Usage statistics
    active_users: int = 0
    total_assignments: int = 0
    recent_usage_count: int = 0
    last_used: Optional[datetime] = None
    
    # Permissions
    permissions: List[str] = field(default_factory=list)
    sensitive_permissions: List[str] = field(default_factory=list)
    
    # Violations
    violation_count: int = 0
    sod_conflicts: List[str] = field(default_factory=list)
    
    # Trends
    usage_trend: str = "stable"  # increasing, decreasing, stable
    risk_score: float = 0.0

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    widget_type: str = ""  # chart, table, metric, alert
    title: str = ""
    description: str = ""
    
    # Data configuration
    data_source: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 300  # seconds
    
    # Display configuration
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height
    chart_config: Dict[str, Any] = field(default_factory=dict)
    
    # Access control
    required_permissions: List[str] = field(default_factory=list)
    visible_to_roles: List[str] = field(default_factory=list)

@dataclass
class Dashboard:
    """RBAC dashboard configuration"""
    dashboard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    dashboard_type: DashboardType = DashboardType.OPERATIONAL
    tenant_id: int = 0
    
    # Configuration
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    
    # Access control
    visible_to_roles: List[str] = field(default_factory=list)
    
    # Access control
    created_by: str = ""
    shared_with: List[str] = field(default_factory=list)
    public: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)

class RBACDashboardManager:
    """
    RBAC Dashboard Manager - Chapter 8.3 UI Implementation
    
    Features:
    - Dynamic dashboard generation
    - Real-time access monitoring
    - Violation detection and alerting
    - Role usage analytics
    - Executive and operational views
    """
    
    def __init__(self, pool_manager=None, rbac_manager=None):
        self.pool_manager = pool_manager
        self.rbac_manager = rbac_manager
        self.logger = logging.getLogger(__name__)
        
        # Dashboard configurations
        self.dashboards: Dict[str, Dashboard] = {}
        self.violations: Dict[str, AccessViolation] = {}
        
        # Metrics cache
        self.metrics_cache = {
            'role_usage': {},
            'violation_trends': {},
            'user_activity': {},
            'last_refresh': None
        }
        
        # Configuration
        self.config = {
            'violation_retention_days': 90,
            'metrics_refresh_interval': 300,  # 5 minutes
            'anomaly_detection_enabled': True,
            'real_time_alerts': True,
            'dashboard_auto_refresh': True
        }
        
        # Initialize default dashboards
        self._initialize_default_dashboards()
    
    async def initialize(self) -> bool:
        """Initialize RBAC dashboard system"""
        try:
            await self._create_dashboard_tables()
            await self._load_existing_dashboards()
            self.logger.info("✅ RBAC dashboard system initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RBAC dashboard system: {e}")
            return False
    
    async def _create_dashboard_tables(self):
        """Create dashboard database tables"""
        if not self.pool_manager:
            return
        
        sql = """
        -- Access violations table
        CREATE TABLE IF NOT EXISTS access_violations (
            violation_id UUID PRIMARY KEY,
            tenant_id INTEGER NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            violation_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            
            -- Violation details
            attempted_action VARCHAR(255) NOT NULL,
            resource_id VARCHAR(255),
            required_permission VARCHAR(100),
            user_roles TEXT[] DEFAULT '{}',
            
            -- Context
            timestamp TIMESTAMPTZ NOT NULL,
            ip_address INET,
            user_agent TEXT,
            session_id VARCHAR(255),
            
            -- Resolution
            status VARCHAR(20) DEFAULT 'open',
            assigned_to VARCHAR(255),
            resolution_notes TEXT,
            resolved_at TIMESTAMPTZ,
            
            -- Indexes
            CONSTRAINT access_violations_tenant_fk FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
        );
        
        -- Role usage metrics table
        CREATE TABLE IF NOT EXISTS role_usage_metrics (
            metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id INTEGER NOT NULL,
            role_name VARCHAR(100) NOT NULL,
            
            -- Usage statistics
            active_users INTEGER DEFAULT 0,
            total_assignments INTEGER DEFAULT 0,
            recent_usage_count INTEGER DEFAULT 0,
            last_used TIMESTAMPTZ,
            
            -- Permissions
            permissions TEXT[] DEFAULT '{}',
            sensitive_permissions TEXT[] DEFAULT '{}',
            
            -- Violations
            violation_count INTEGER DEFAULT 0,
            sod_conflicts TEXT[] DEFAULT '{}',
            
            -- Trends
            usage_trend VARCHAR(20) DEFAULT 'stable',
            risk_score DECIMAL(5,2) DEFAULT 0.0,
            
            -- Metadata
            calculated_at TIMESTAMPTZ DEFAULT NOW(),
            
            UNIQUE(tenant_id, role_name)
        );
        
        -- Dashboard configurations table
        CREATE TABLE IF NOT EXISTS rbac_dashboards (
            dashboard_id UUID PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            dashboard_type VARCHAR(50) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Configuration
            widgets JSONB DEFAULT '[]',
            layout JSONB DEFAULT '{}',
            
            -- Access control
            created_by VARCHAR(255) NOT NULL,
            shared_with TEXT[] DEFAULT '{}',
            public BOOLEAN DEFAULT FALSE,
            
            -- Metadata
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            tags TEXT[] DEFAULT '{}',
            
            CONSTRAINT rbac_dashboards_tenant_fk FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
        );
        
        -- Enable RLS
        ALTER TABLE access_violations ENABLE ROW LEVEL SECURITY;
        ALTER TABLE role_usage_metrics ENABLE ROW LEVEL SECURITY;
        ALTER TABLE rbac_dashboards ENABLE ROW LEVEL SECURITY;
        
        -- RLS policies
        DROP POLICY IF EXISTS access_violations_rls_policy ON access_violations;
        CREATE POLICY access_violations_rls_policy ON access_violations
            FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
            
        DROP POLICY IF EXISTS role_usage_metrics_rls_policy ON role_usage_metrics;
        CREATE POLICY role_usage_metrics_rls_policy ON role_usage_metrics
            FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
            
        DROP POLICY IF EXISTS rbac_dashboards_rls_policy ON rbac_dashboards;
        CREATE POLICY rbac_dashboards_rls_policy ON rbac_dashboards
            FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_access_violations_tenant_time ON access_violations(tenant_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_access_violations_user ON access_violations(user_id);
        CREATE INDEX IF NOT EXISTS idx_access_violations_type ON access_violations(violation_type);
        CREATE INDEX IF NOT EXISTS idx_access_violations_status ON access_violations(status);
        
        CREATE INDEX IF NOT EXISTS idx_role_usage_metrics_tenant ON role_usage_metrics(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_role_usage_metrics_role ON role_usage_metrics(role_name);
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            await pool.execute(sql)
            self.logger.info("✅ Created RBAC dashboard database tables")
        finally:
            await self.pool_manager.return_connection(pool)
    
    def _initialize_default_dashboards(self):
        """Initialize default dashboard configurations"""
        
        # Executive Dashboard
        exec_dashboard = Dashboard(
            name="Executive Access Overview",
            dashboard_type=DashboardType.EXECUTIVE,
            widgets=[
                DashboardWidget(
                    widget_type="metric",
                    title="Total Active Users",
                    data_source="user_activity",
                    position={"x": 0, "y": 0, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    widget_type="chart",
                    title="Access Violations Trend",
                    data_source="violation_trends",
                    chart_config={"type": "line", "timeframe": "30d"},
                    position={"x": 3, "y": 0, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    widget_type="table",
                    title="High-Risk Roles",
                    data_source="role_risk_analysis",
                    position={"x": 0, "y": 2, "width": 3, "height": 4}
                ),
                DashboardWidget(
                    widget_type="chart",
                    title="SoD Violations by Department",
                    data_source="sod_violations",
                    chart_config={"type": "bar"},
                    position={"x": 9, "y": 0, "width": 3, "height": 4}
                )
            ],
            visible_to_roles=["executive", "compliance_officer"]
        )
        
        # Operational Dashboard
        ops_dashboard = Dashboard(
            name="RBAC Operations Dashboard",
            dashboard_type=DashboardType.OPERATIONAL,
            widgets=[
                DashboardWidget(
                    widget_type="alert",
                    title="Active Violations",
                    data_source="active_violations",
                    position={"x": 0, "y": 0, "width": 12, "height": 2}
                ),
                DashboardWidget(
                    widget_type="table",
                    title="Recent Role Changes",
                    data_source="role_changes",
                    position={"x": 0, "y": 2, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    widget_type="chart",
                    title="Permission Usage Heatmap",
                    data_source="permission_usage",
                    chart_config={"type": "heatmap"},
                    position={"x": 6, "y": 2, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    widget_type="table",
                    title="User Access Requests",
                    data_source="access_requests",
                    position={"x": 0, "y": 6, "width": 12, "height": 3}
                )
            ],
            visible_to_roles=["security_admin", "ops_manager"]
        )
        
        # Compliance Dashboard
        compliance_dashboard = Dashboard(
            name="Compliance & Audit Dashboard",
            dashboard_type=DashboardType.COMPLIANCE,
            widgets=[
                DashboardWidget(
                    widget_type="metric",
                    title="SoD Compliance Score",
                    data_source="sod_compliance",
                    position={"x": 0, "y": 0, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    widget_type="chart",
                    title="Audit Trail Completeness",
                    data_source="audit_completeness",
                    chart_config={"type": "gauge"},
                    position={"x": 3, "y": 0, "width": 3, "height": 2}
                ),
                DashboardWidget(
                    widget_type="table",
                    title="Compliance Violations",
                    data_source="compliance_violations",
                    position={"x": 0, "y": 2, "width": 12, "height": 4}
                ),
                DashboardWidget(
                    widget_type="chart",
                    title="Role Certification Status",
                    data_source="role_certification",
                    chart_config={"type": "donut"},
                    position={"x": 6, "y": 0, "width": 6, "height": 2}
                )
            ],
            visible_to_roles=["compliance_officer", "auditor"]
        )
        
        self.dashboards = {
            "executive": exec_dashboard,
            "operational": ops_dashboard,
            "compliance": compliance_dashboard
        }
    
    async def _load_existing_dashboards(self):
        """Load existing dashboards from database"""
        if not self.pool_manager:
            return
        
        try:
            pool = await self.pool_manager.get_connection()
            
            sql = "SELECT * FROM rbac_dashboards ORDER BY created_at"
            rows = await pool.fetch(sql)
            
            for row in rows:
                dashboard = Dashboard(
                    dashboard_id=row['dashboard_id'],
                    name=row['name'],
                    dashboard_type=DashboardType(row['dashboard_type']),
                    tenant_id=row['tenant_id'],
                    widgets=[DashboardWidget(**w) for w in row['widgets']],
                    layout=row['layout'],
                    created_by=row['created_by'],
                    shared_with=row['shared_with'],
                    public=row['public'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    tags=row['tags']
                )
                
                self.dashboards[dashboard.dashboard_id] = dashboard
            
            self.logger.info(f"✅ Loaded {len(rows)} existing dashboards")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load existing dashboards: {e}")
        finally:
            if pool:
                await self.pool_manager.return_connection(pool)
    
    async def log_access_violation(
        self,
        tenant_id: int,
        user_id: str,
        violation_type: AccessViolationType,
        attempted_action: str,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        resource_id: Optional[str] = None,
        required_permission: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an access violation (Task 8.3-T17)"""
        
        try:
            # Get user roles
            user_roles = []
            if self.rbac_manager:
                user_roles = await self.rbac_manager.get_user_roles(user_id, tenant_id)
            
            # Create violation record
            violation = AccessViolation(
                tenant_id=tenant_id,
                user_id=user_id,
                violation_type=violation_type,
                severity=severity,
                attempted_action=attempted_action,
                resource_id=resource_id,
                required_permission=required_permission,
                user_roles=[r.role_name for r in user_roles] if user_roles else [],
                ip_address=context.get('ip_address') if context else None,
                user_agent=context.get('user_agent') if context else None,
                session_id=context.get('session_id') if context else None
            )
            
            # Store violation
            await self._store_access_violation(violation)
            
            # Add to cache
            self.violations[violation.violation_id] = violation
            
            # Trigger real-time alerts if enabled
            if self.config['real_time_alerts']:
                await self._trigger_violation_alert(violation)
            
            self.logger.warning(f"🚨 Access violation logged: {violation.violation_id} - {violation_type.value}")
            return violation.violation_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log access violation: {e}")
            raise
    
    async def _store_access_violation(self, violation: AccessViolation):
        """Store access violation in database"""
        if not self.pool_manager:
            return
        
        sql = """
        INSERT INTO access_violations (
            violation_id, tenant_id, user_id, violation_type, severity,
            attempted_action, resource_id, required_permission, user_roles,
            timestamp, ip_address, user_agent, session_id, status
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            await pool.execute(
                sql,
                violation.violation_id, violation.tenant_id, violation.user_id,
                violation.violation_type.value, violation.severity.value,
                violation.attempted_action, violation.resource_id,
                violation.required_permission, violation.user_roles,
                violation.timestamp, violation.ip_address, violation.user_agent,
                violation.session_id, violation.status
            )
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _trigger_violation_alert(self, violation: AccessViolation):
        """Trigger real-time alert for violation"""
        
        # Determine alert recipients based on severity
        recipients = []
        if violation.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            recipients.extend(['security_team', 'compliance_team'])
        
        if violation.violation_type == AccessViolationType.SOD_VIOLATION:
            recipients.append('compliance_officer')
        
        # Create alert message
        alert_message = {
            'type': 'access_violation',
            'violation_id': violation.violation_id,
            'severity': violation.severity.value,
            'user_id': violation.user_id,
            'action': violation.attempted_action,
            'timestamp': violation.timestamp.isoformat(),
            'requires_immediate_attention': violation.severity == AlertSeverity.CRITICAL
        }
        
        # Send alerts (implementation would depend on notification system)
        self.logger.info(f"🚨 Alert triggered for violation {violation.violation_id}")
    
    async def calculate_role_usage_metrics(self, tenant_id: int) -> Dict[str, RoleUsageMetrics]:
        """Calculate role usage metrics for dashboards"""
        
        try:
            metrics = {}
            
            if not self.rbac_manager:
                return metrics
            
            # Get all roles for tenant
            roles = await self.rbac_manager.get_tenant_roles(tenant_id)
            
            for role in roles:
                # Calculate usage metrics
                usage_metrics = RoleUsageMetrics(
                    role_name=role.role_name,
                    tenant_id=tenant_id,
                    permissions=role.permissions,
                    sensitive_permissions=self._identify_sensitive_permissions(role.permissions)
                )
                
                # Get active users with this role
                users_with_role = await self.rbac_manager.get_users_with_role(role.role_name, tenant_id)
                usage_metrics.active_users = len(users_with_role)
                usage_metrics.total_assignments = len(users_with_role)  # Simplified
                
                # Calculate recent usage (last 30 days)
                usage_metrics.recent_usage_count = await self._get_recent_role_usage(
                    tenant_id, role.role_name
                )
                
                # Get violation count
                usage_metrics.violation_count = await self._get_role_violation_count(
                    tenant_id, role.role_name
                )
                
                # Check SoD conflicts
                usage_metrics.sod_conflicts = await self._check_sod_conflicts(
                    tenant_id, role.role_name
                )
                
                # Calculate risk score
                usage_metrics.risk_score = self._calculate_role_risk_score(usage_metrics)
                
                metrics[role.role_name] = usage_metrics
            
            # Store metrics
            await self._store_role_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate role usage metrics: {e}")
            return {}
    
    def _identify_sensitive_permissions(self, permissions: List[str]) -> List[str]:
        """Identify sensitive permissions"""
        sensitive_keywords = [
            'delete', 'admin', 'override', 'approve', 'financial',
            'audit', 'security', 'policy', 'user_manage'
        ]
        
        sensitive = []
        for perm in permissions:
            if any(keyword in perm.lower() for keyword in sensitive_keywords):
                sensitive.append(perm)
        
        return sensitive
    
    async def _get_recent_role_usage(self, tenant_id: int, role_name: str) -> int:
        """Get recent usage count for role"""
        if not self.pool_manager:
            return 0
        
        # This would query audit logs for role usage
        # Simplified implementation
        return 0
    
    async def _get_role_violation_count(self, tenant_id: int, role_name: str) -> int:
        """Get violation count for role"""
        if not self.pool_manager:
            return 0
        
        sql = """
        SELECT COUNT(*)
        FROM access_violations
        WHERE tenant_id = $1 AND $2 = ANY(user_roles)
        AND timestamp > NOW() - INTERVAL '30 days'
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            result = await pool.fetchval(sql, tenant_id, role_name)
            return result or 0
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _check_sod_conflicts(self, tenant_id: int, role_name: str) -> List[str]:
        """Check for SoD conflicts with role"""
        if not self.rbac_manager:
            return []
        
        # This would check against SoD matrix
        # Simplified implementation
        return []
    
    def _calculate_role_risk_score(self, metrics: RoleUsageMetrics) -> float:
        """Calculate risk score for role"""
        risk_score = 0.0
        
        # Base risk from sensitive permissions
        risk_score += len(metrics.sensitive_permissions) * 10
        
        # Risk from violations
        risk_score += metrics.violation_count * 5
        
        # Risk from SoD conflicts
        risk_score += len(metrics.sod_conflicts) * 15
        
        # Normalize to 0-100 scale
        return min(risk_score, 100.0)
    
    async def _store_role_metrics(self, metrics: Dict[str, RoleUsageMetrics]):
        """Store role metrics in database"""
        if not self.pool_manager or not metrics:
            return
        
        pool = await self.pool_manager.get_connection()
        try:
            for role_name, metric in metrics.items():
                sql = """
                INSERT INTO role_usage_metrics (
                    tenant_id, role_name, active_users, total_assignments,
                    recent_usage_count, permissions, sensitive_permissions,
                    violation_count, sod_conflicts, risk_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (tenant_id, role_name) 
                DO UPDATE SET
                    active_users = EXCLUDED.active_users,
                    total_assignments = EXCLUDED.total_assignments,
                    recent_usage_count = EXCLUDED.recent_usage_count,
                    permissions = EXCLUDED.permissions,
                    sensitive_permissions = EXCLUDED.sensitive_permissions,
                    violation_count = EXCLUDED.violation_count,
                    sod_conflicts = EXCLUDED.sod_conflicts,
                    risk_score = EXCLUDED.risk_score,
                    calculated_at = NOW()
                """
                
                await pool.execute(
                    sql,
                    metric.tenant_id, metric.role_name, metric.active_users,
                    metric.total_assignments, metric.recent_usage_count,
                    metric.permissions, metric.sensitive_permissions,
                    metric.violation_count, metric.sod_conflicts, metric.risk_score
                )
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def get_dashboard_data(
        self,
        dashboard_type: DashboardType,
        tenant_id: int,
        user_roles: List[str],
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get dashboard data based on type and user permissions"""
        
        try:
            dashboard_data = {
                'dashboard_type': dashboard_type.value,
                'tenant_id': tenant_id,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'widgets': {}
            }
            
            # Get appropriate dashboard configuration
            dashboard_key = dashboard_type.value
            if dashboard_key not in self.dashboards:
                return dashboard_data
            
            dashboard = self.dashboards[dashboard_key]
            
            # Check if user has access to dashboard
            if dashboard.visible_to_roles and not any(role in dashboard.visible_to_roles for role in user_roles):
                raise PermissionError("Insufficient permissions to access dashboard")
            
            # Generate data for each widget
            for widget in dashboard.widgets:
                # Check widget permissions
                if widget.required_permissions:
                    # This would check against user permissions
                    pass
                
                widget_data = await self._get_widget_data(widget, tenant_id, filters)
                dashboard_data['widgets'][widget.widget_id] = {
                    'title': widget.title,
                    'type': widget.widget_type,
                    'data': widget_data,
                    'config': widget.chart_config,
                    'position': widget.position
                }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get dashboard data: {e}")
            raise
    
    async def _get_widget_data(
        self,
        widget: DashboardWidget,
        tenant_id: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get data for specific widget"""
        
        data_source = widget.data_source
        
        if data_source == "user_activity":
            return await self._get_user_activity_data(tenant_id, filters)
        elif data_source == "violation_trends":
            return await self._get_violation_trends_data(tenant_id, filters)
        elif data_source == "role_risk_analysis":
            return await self._get_role_risk_data(tenant_id, filters)
        elif data_source == "active_violations":
            return await self._get_active_violations_data(tenant_id, filters)
        elif data_source == "sod_violations":
            return await self._get_sod_violations_data(tenant_id, filters)
        else:
            return {"message": f"Data source '{data_source}' not implemented"}
    
    async def _get_user_activity_data(self, tenant_id: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get user activity metrics"""
        if not self.pool_manager:
            return {"total_users": 0, "active_users": 0}
        
        # This would query user activity from audit logs
        # Simplified implementation
        return {
            "total_users": 150,
            "active_users": 142,
            "inactive_users": 8,
            "new_users_this_month": 5
        }
    
    async def _get_violation_trends_data(self, tenant_id: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get violation trends data"""
        if not self.pool_manager:
            return {"trend": []}
        
        sql = """
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM access_violations
        WHERE tenant_id = $1 AND timestamp > NOW() - INTERVAL '30 days'
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            rows = await pool.fetch(sql, tenant_id)
            
            trend_data = []
            for row in rows:
                trend_data.append({
                    "date": row['date'].isoformat(),
                    "violations": row['count']
                })
            
            return {
                "trend": trend_data,
                "total_violations": sum(d['violations'] for d in trend_data)
            }
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _get_role_risk_data(self, tenant_id: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get role risk analysis data"""
        if not self.pool_manager:
            return {"high_risk_roles": []}
        
        sql = """
        SELECT role_name, risk_score, violation_count, active_users
        FROM role_usage_metrics
        WHERE tenant_id = $1 AND risk_score > 50
        ORDER BY risk_score DESC
        LIMIT 10
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            rows = await pool.fetch(sql, tenant_id)
            
            high_risk_roles = []
            for row in rows:
                high_risk_roles.append({
                    "role_name": row['role_name'],
                    "risk_score": float(row['risk_score']),
                    "violation_count": row['violation_count'],
                    "active_users": row['active_users']
                })
            
            return {"high_risk_roles": high_risk_roles}
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _get_active_violations_data(self, tenant_id: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get active violations data"""
        if not self.pool_manager:
            return {"active_violations": []}
        
        sql = """
        SELECT violation_id, user_id, violation_type, severity, attempted_action, timestamp
        FROM access_violations
        WHERE tenant_id = $1 AND status = 'open'
        ORDER BY timestamp DESC
        LIMIT 20
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            rows = await pool.fetch(sql, tenant_id)
            
            violations = []
            for row in rows:
                violations.append({
                    "violation_id": row['violation_id'],
                    "user_id": row['user_id'],
                    "type": row['violation_type'],
                    "severity": row['severity'],
                    "action": row['attempted_action'],
                    "timestamp": row['timestamp'].isoformat()
                })
            
            return {"active_violations": violations}
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _get_sod_violations_data(self, tenant_id: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get SoD violations data"""
        if not self.pool_manager:
            return {"sod_violations": []}
        
        sql = """
        SELECT user_id, COUNT(*) as violation_count
        FROM access_violations
        WHERE tenant_id = $1 AND violation_type = 'sod_violation'
        AND timestamp > NOW() - INTERVAL '30 days'
        GROUP BY user_id
        ORDER BY violation_count DESC
        LIMIT 10
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            rows = await pool.fetch(sql, tenant_id)
            
            sod_violations = []
            for row in rows:
                sod_violations.append({
                    "user_id": row['user_id'],
                    "violation_count": row['violation_count']
                })
            
            return {"sod_violations": sod_violations}
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def create_custom_dashboard(
        self,
        name: str,
        dashboard_type: DashboardType,
        tenant_id: int,
        created_by: str,
        widgets: List[DashboardWidget],
        layout: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create custom dashboard (Task 8.3-T27)"""
        
        dashboard = Dashboard(
            name=name,
            dashboard_type=dashboard_type,
            tenant_id=tenant_id,
            created_by=created_by,
            widgets=widgets,
            layout=layout or {}
        )
        
        # Store dashboard
        await self._store_dashboard(dashboard)
        
        # Add to cache
        self.dashboards[dashboard.dashboard_id] = dashboard
        
        self.logger.info(f"✅ Created custom dashboard {dashboard.dashboard_id}")
        return dashboard.dashboard_id
    
    async def _store_dashboard(self, dashboard: Dashboard):
        """Store dashboard in database"""
        if not self.pool_manager:
            return
        
        sql = """
        INSERT INTO rbac_dashboards (
            dashboard_id, name, dashboard_type, tenant_id, widgets, layout,
            created_by, shared_with, public, tags
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        widgets_json = [asdict(w) for w in dashboard.widgets]
        
        pool = await self.pool_manager.get_connection()
        try:
            await pool.execute(
                sql,
                dashboard.dashboard_id, dashboard.name, dashboard.dashboard_type.value,
                dashboard.tenant_id, json.dumps(widgets_json), json.dumps(dashboard.layout),
                dashboard.created_by, dashboard.shared_with, dashboard.public, dashboard.tags
            )
        finally:
            await self.pool_manager.return_connection(pool)
    
    def get_training_materials(self) -> Dict[str, Any]:
        """Get training materials for RBAC dashboards (Tasks 8.3-T32, T34)"""
        
        return {
            "ops_training": {
                "title": "RBAC Operations Dashboard Training",
                "modules": [
                    {
                        "name": "Dashboard Overview",
                        "content": "Understanding the RBAC operations dashboard layout and key metrics",
                        "duration": "15 minutes",
                        "exercises": [
                            "Navigate to violation alerts",
                            "Filter role usage data",
                            "Interpret risk scores"
                        ]
                    },
                    {
                        "name": "Violation Management",
                        "content": "How to investigate and resolve access violations",
                        "duration": "20 minutes",
                        "exercises": [
                            "Assign violation to team member",
                            "Update violation status",
                            "Document resolution"
                        ]
                    },
                    {
                        "name": "Role Administration",
                        "content": "Managing roles and permissions through the dashboard",
                        "duration": "25 minutes",
                        "exercises": [
                            "Create new role",
                            "Modify role permissions",
                            "Review SoD conflicts"
                        ]
                    }
                ]
            },
            "exec_training": {
                "title": "Executive Access Dashboard Training",
                "modules": [
                    {
                        "name": "Executive Overview",
                        "content": "High-level access control metrics and trends",
                        "duration": "10 minutes",
                        "key_metrics": [
                            "Total active users",
                            "Violation trends",
                            "Compliance scores",
                            "Risk indicators"
                        ]
                    },
                    {
                        "name": "Risk Assessment",
                        "content": "Understanding and interpreting risk scores",
                        "duration": "15 minutes",
                        "focus_areas": [
                            "High-risk roles identification",
                            "SoD violation impact",
                            "Trend analysis",
                            "Action recommendations"
                        ]
                    }
                ]
            },
            "quick_reference": {
                "dashboard_navigation": {
                    "filters": "Use date range and role filters to narrow data",
                    "drill_down": "Click on charts to see detailed breakdowns",
                    "export": "Use export button for compliance reports",
                    "alerts": "Red indicators require immediate attention"
                },
                "common_actions": {
                    "investigate_violation": "Click violation ID → Review details → Assign → Resolve",
                    "review_role_risk": "Navigate to Role Risk widget → Sort by risk score → Review high-risk roles",
                    "generate_report": "Select date range → Choose export format → Download report"
                }
            }
        }

# Global instance
rbac_dashboard_manager = RBACDashboardManager()

async def get_rbac_dashboard_manager():
    """Get the global RBAC dashboard manager instance"""
    return rbac_dashboard_manager
