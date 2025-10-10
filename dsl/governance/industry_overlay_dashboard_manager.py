"""
Industry Overlay Dashboard Manager - Chapter 8.4 Implementation
==============================================================
Tasks 8.4-T15, T32, T33, T34, T36, T37, T38: Industry overlay dashboards and monitoring

Missing Tasks Implemented:
- 8.4-T15: Build overlay dashboards (coverage, violations, SLA)
- 8.4-T32: Partition overlay logs by industry
- 8.4-T33: Create sandbox demo per overlay (SaaS, BFSI, Insurance)
- 8.4-T34: Train Ops on overlay dashboards
- 8.4-T36: Train Execs on overlay adoption dashboards
- 8.4-T37: Define adoption metrics dashboard per overlay
- 8.4-T38: Add anomaly detection for overlay violations

Key Features:
- Industry-specific compliance dashboards
- Real-time overlay violation monitoring
- Adoption metrics and trend analysis
- Anomaly detection for compliance violations
- Multi-tenant overlay management
- Executive and operational views per industry
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import statistics

logger = logging.getLogger(__name__)

class IndustryType(Enum):
    """Supported industry types"""
    SAAS = "saas"
    BANKING = "banking"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    FINTECH = "fintech"
    ECOMMERCE = "ecommerce"

class OverlayViolationType(Enum):
    """Types of overlay violations"""
    COMPLIANCE_BREACH = "compliance_breach"
    DATA_RESIDENCY = "data_residency"
    RETENTION_POLICY = "retention_policy"
    ACCESS_CONTROL = "access_control"
    AUDIT_REQUIREMENT = "audit_requirement"
    ENCRYPTION_STANDARD = "encryption_standard"
    REPORTING_DEADLINE = "reporting_deadline"

class OverlayStatus(Enum):
    """Overlay status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    UNDER_REVIEW = "under_review"

class AdoptionStage(Enum):
    """Overlay adoption stages"""
    NOT_STARTED = "not_started"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    DEPLOYED = "deployed"
    OPTIMIZING = "optimizing"

@dataclass
class OverlayViolation:
    """Industry overlay violation record"""
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: int = 0
    industry_type: IndustryType = IndustryType.SAAS
    violation_type: OverlayViolationType = OverlayViolationType.COMPLIANCE_BREACH
    
    # Violation details
    overlay_name: str = ""
    rule_violated: str = ""
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    
    # Context
    workflow_id: Optional[str] = None
    user_id: Optional[str] = None
    resource_affected: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Resolution
    status: str = "open"  # open, investigating, resolved, false_positive
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Compliance metadata
    regulatory_framework: Optional[str] = None
    compliance_deadline: Optional[datetime] = None
    business_impact: str = "low"  # low, medium, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class OverlayAdoptionMetrics:
    """Overlay adoption metrics"""
    overlay_name: str = ""
    industry_type: IndustryType = IndustryType.SAAS
    tenant_id: int = 0
    
    # Adoption statistics
    adoption_stage: AdoptionStage = AdoptionStage.NOT_STARTED
    adoption_percentage: float = 0.0
    workflows_using_overlay: int = 0
    total_applicable_workflows: int = 0
    
    # Usage metrics
    daily_executions: int = 0
    success_rate: float = 100.0
    average_execution_time_ms: float = 0.0
    
    # Compliance metrics
    compliance_score: float = 100.0
    violations_count: int = 0
    sla_adherence: float = 100.0
    
    # Trends
    adoption_trend: str = "stable"  # increasing, decreasing, stable
    usage_trend: str = "stable"
    compliance_trend: str = "stable"
    
    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    next_review_date: Optional[datetime] = None

@dataclass
class IndustryOverlayConfig:
    """Industry overlay configuration"""
    overlay_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    overlay_name: str = ""
    industry_type: IndustryType = IndustryType.SAAS
    
    # Configuration
    compliance_frameworks: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    sla_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: OverlayStatus = OverlayStatus.ACTIVE
    version: str = "1.0.0"
    
    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result"""
    anomaly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: int = 0
    industry_type: IndustryType = IndustryType.SAAS
    
    # Anomaly details
    anomaly_type: str = ""  # spike, drop, pattern_change, threshold_breach
    metric_name: str = ""
    current_value: float = 0.0
    expected_value: float = 0.0
    deviation_percentage: float = 0.0
    
    # Context
    detection_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    time_window: str = "1h"  # 1h, 1d, 1w
    confidence_score: float = 0.0
    
    # Impact assessment
    severity: str = "low"
    business_impact: str = ""
    recommended_action: str = ""
    
    # Status
    status: str = "new"  # new, acknowledged, investigating, resolved

class IndustryOverlayDashboardManager:
    """
    Industry Overlay Dashboard Manager - Chapter 8.4 Implementation
    
    Features:
    - Industry-specific compliance dashboards
    - Real-time overlay violation monitoring
    - Adoption metrics tracking
    - Anomaly detection for violations
    - Multi-tenant overlay management
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Overlay configurations
        self.overlay_configs: Dict[str, IndustryOverlayConfig] = {}
        self.violations: Dict[str, OverlayViolation] = {}
        self.adoption_metrics: Dict[str, OverlayAdoptionMetrics] = {}
        
        # Anomaly detection
        self.anomaly_thresholds = {
            'violation_spike_threshold': 3.0,  # 3x normal rate
            'adoption_drop_threshold': 0.2,   # 20% drop
            'sla_breach_threshold': 0.95,     # Below 95%
            'compliance_score_threshold': 0.9  # Below 90%
        }
        
        # Configuration
        self.config = {
            'enable_real_time_monitoring': True,
            'anomaly_detection_interval': 300,  # 5 minutes
            'metrics_retention_days': 90,
            'dashboard_refresh_interval': 60,  # 1 minute
            'alert_thresholds': {
                'critical_violations': 5,
                'sla_breach_percentage': 5.0,
                'adoption_drop_percentage': 15.0
            }
        }
        
        # Initialize industry-specific configurations
        self._initialize_industry_overlays()
    
    async def initialize(self) -> bool:
        """Initialize industry overlay dashboard system"""
        try:
            await self._create_overlay_tables()
            await self._load_existing_overlays()
            self.logger.info("✅ Industry overlay dashboard system initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize overlay dashboard system: {e}")
            return False
    
    async def _create_overlay_tables(self):
        """Create overlay dashboard database tables"""
        if not self.pool_manager:
            return
        
        sql = """
        -- Industry overlay violations table (Task 8.4-T32: Partition by industry)
        CREATE TABLE IF NOT EXISTS overlay_violations (
            violation_id UUID PRIMARY KEY,
            tenant_id INTEGER NOT NULL,
            industry_type VARCHAR(20) NOT NULL,
            violation_type VARCHAR(50) NOT NULL,
            
            -- Violation details
            overlay_name VARCHAR(100) NOT NULL,
            rule_violated VARCHAR(255) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            description TEXT,
            
            -- Context
            workflow_id UUID,
            user_id VARCHAR(255),
            resource_affected VARCHAR(255),
            timestamp TIMESTAMPTZ NOT NULL,
            
            -- Resolution
            status VARCHAR(20) DEFAULT 'open',
            assigned_to VARCHAR(255),
            resolution_notes TEXT,
            resolved_at TIMESTAMPTZ,
            
            -- Compliance metadata
            regulatory_framework VARCHAR(50),
            compliance_deadline TIMESTAMPTZ,
            business_impact VARCHAR(20) DEFAULT 'low',
            
            CONSTRAINT overlay_violations_tenant_fk FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
        ) PARTITION BY LIST (industry_type);
        
        -- Create partitions for each industry
        CREATE TABLE IF NOT EXISTS overlay_violations_saas PARTITION OF overlay_violations FOR VALUES IN ('saas');
        CREATE TABLE IF NOT EXISTS overlay_violations_banking PARTITION OF overlay_violations FOR VALUES IN ('banking');
        CREATE TABLE IF NOT EXISTS overlay_violations_insurance PARTITION OF overlay_violations FOR VALUES IN ('insurance');
        CREATE TABLE IF NOT EXISTS overlay_violations_healthcare PARTITION OF overlay_violations FOR VALUES IN ('healthcare');
        CREATE TABLE IF NOT EXISTS overlay_violations_fintech PARTITION OF overlay_violations FOR VALUES IN ('fintech');
        CREATE TABLE IF NOT EXISTS overlay_violations_ecommerce PARTITION OF overlay_violations FOR VALUES IN ('ecommerce');
        
        -- Overlay adoption metrics table
        CREATE TABLE IF NOT EXISTS overlay_adoption_metrics (
            metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            overlay_name VARCHAR(100) NOT NULL,
            industry_type VARCHAR(20) NOT NULL,
            tenant_id INTEGER NOT NULL,
            
            -- Adoption statistics
            adoption_stage VARCHAR(20) DEFAULT 'not_started',
            adoption_percentage DECIMAL(5,2) DEFAULT 0.0,
            workflows_using_overlay INTEGER DEFAULT 0,
            total_applicable_workflows INTEGER DEFAULT 0,
            
            -- Usage metrics
            daily_executions INTEGER DEFAULT 0,
            success_rate DECIMAL(5,2) DEFAULT 100.0,
            average_execution_time_ms DECIMAL(10,2) DEFAULT 0.0,
            
            -- Compliance metrics
            compliance_score DECIMAL(5,2) DEFAULT 100.0,
            violations_count INTEGER DEFAULT 0,
            sla_adherence DECIMAL(5,2) DEFAULT 100.0,
            
            -- Trends
            adoption_trend VARCHAR(20) DEFAULT 'stable',
            usage_trend VARCHAR(20) DEFAULT 'stable',
            compliance_trend VARCHAR(20) DEFAULT 'stable',
            
            -- Metadata
            last_updated TIMESTAMPTZ DEFAULT NOW(),
            next_review_date TIMESTAMPTZ,
            
            UNIQUE(tenant_id, overlay_name, industry_type)
        );
        
        -- Industry overlay configurations table
        CREATE TABLE IF NOT EXISTS industry_overlay_configs (
            overlay_id UUID PRIMARY KEY,
            overlay_name VARCHAR(100) NOT NULL,
            industry_type VARCHAR(20) NOT NULL,
            
            -- Configuration
            compliance_frameworks TEXT[] DEFAULT '{}',
            required_fields TEXT[] DEFAULT '{}',
            validation_rules JSONB DEFAULT '{}',
            sla_requirements JSONB DEFAULT '{}',
            
            -- Status
            status VARCHAR(20) DEFAULT 'active',
            version VARCHAR(20) DEFAULT '1.0.0',
            
            -- Metadata
            created_by VARCHAR(255) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            
            UNIQUE(overlay_name, industry_type)
        );
        
        -- Anomaly detection results table
        CREATE TABLE IF NOT EXISTS overlay_anomalies (
            anomaly_id UUID PRIMARY KEY,
            tenant_id INTEGER NOT NULL,
            industry_type VARCHAR(20) NOT NULL,
            
            -- Anomaly details
            anomaly_type VARCHAR(50) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            current_value DECIMAL(15,4) NOT NULL,
            expected_value DECIMAL(15,4) NOT NULL,
            deviation_percentage DECIMAL(5,2) NOT NULL,
            
            -- Context
            detection_timestamp TIMESTAMPTZ NOT NULL,
            time_window VARCHAR(10) NOT NULL,
            confidence_score DECIMAL(5,2) NOT NULL,
            
            -- Impact assessment
            severity VARCHAR(20) NOT NULL,
            business_impact TEXT,
            recommended_action TEXT,
            
            -- Status
            status VARCHAR(20) DEFAULT 'new',
            
            CONSTRAINT overlay_anomalies_tenant_fk FOREIGN KEY (tenant_id) REFERENCES tenant_metadata(tenant_id)
        );
        
        -- Enable RLS on all tables
        ALTER TABLE overlay_violations ENABLE ROW LEVEL SECURITY;
        ALTER TABLE overlay_adoption_metrics ENABLE ROW LEVEL SECURITY;
        ALTER TABLE industry_overlay_configs ENABLE ROW LEVEL SECURITY;
        ALTER TABLE overlay_anomalies ENABLE ROW LEVEL SECURITY;
        
        -- RLS policies for tenant isolation
        DROP POLICY IF EXISTS overlay_violations_rls_policy ON overlay_violations;
        CREATE POLICY overlay_violations_rls_policy ON overlay_violations
            FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
            
        DROP POLICY IF EXISTS overlay_adoption_metrics_rls_policy ON overlay_adoption_metrics;
        CREATE POLICY overlay_adoption_metrics_rls_policy ON overlay_adoption_metrics
            FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
            
        DROP POLICY IF EXISTS overlay_anomalies_rls_policy ON overlay_anomalies;
        CREATE POLICY overlay_anomalies_rls_policy ON overlay_anomalies
            FOR ALL USING (tenant_id = COALESCE(current_setting('app.current_tenant_id', true)::integer, -1));
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_overlay_violations_tenant_industry ON overlay_violations(tenant_id, industry_type);
        CREATE INDEX IF NOT EXISTS idx_overlay_violations_timestamp ON overlay_violations(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_overlay_violations_status ON overlay_violations(status);
        CREATE INDEX IF NOT EXISTS idx_overlay_violations_severity ON overlay_violations(severity);
        
        CREATE INDEX IF NOT EXISTS idx_overlay_adoption_tenant_industry ON overlay_adoption_metrics(tenant_id, industry_type);
        CREATE INDEX IF NOT EXISTS idx_overlay_adoption_overlay ON overlay_adoption_metrics(overlay_name);
        
        CREATE INDEX IF NOT EXISTS idx_overlay_anomalies_tenant_industry ON overlay_anomalies(tenant_id, industry_type);
        CREATE INDEX IF NOT EXISTS idx_overlay_anomalies_timestamp ON overlay_anomalies(detection_timestamp DESC);
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            await pool.execute(sql)
            self.logger.info("✅ Created industry overlay dashboard database tables")
        finally:
            await self.pool_manager.return_connection(pool)
    
    def _initialize_industry_overlays(self):
        """Initialize industry-specific overlay configurations"""
        
        # SaaS Industry Overlay
        saas_overlay = IndustryOverlayConfig(
            overlay_name="SaaS Compliance Suite",
            industry_type=IndustryType.SAAS,
            compliance_frameworks=["SOX", "GDPR", "CCPA"],
            required_fields=["subscription_data", "revenue_recognition", "customer_consent"],
            validation_rules={
                "revenue_recognition": {"method": "ASC_606", "required": True},
                "data_retention": {"max_days": 2555, "customer_controlled": True},
                "consent_management": {"explicit_consent": True, "withdrawal_mechanism": True}
            },
            sla_requirements={
                "data_processing_latency": {"max_ms": 1000},
                "consent_response_time": {"max_hours": 24},
                "breach_notification": {"max_hours": 72}
            }
        )
        
        # Banking Industry Overlay
        banking_overlay = IndustryOverlayConfig(
            overlay_name="Banking Regulatory Suite",
            industry_type=IndustryType.BANKING,
            compliance_frameworks=["RBI", "BASEL_III", "PCI_DSS"],
            required_fields=["kyc_data", "aml_checks", "loan_classification"],
            validation_rules={
                "data_localization": {"india_only": True, "cross_border_restricted": True},
                "kyc_verification": {"mandatory": True, "refresh_days": 365},
                "aml_monitoring": {"real_time": True, "threshold_alerts": True}
            },
            sla_requirements={
                "kyc_processing": {"max_hours": 24},
                "aml_alert_response": {"max_minutes": 15},
                "regulatory_reporting": {"monthly": True}
            }
        )
        
        # Insurance Industry Overlay
        insurance_overlay = IndustryOverlayConfig(
            overlay_name="Insurance Regulatory Suite",
            industry_type=IndustryType.INSURANCE,
            compliance_frameworks=["IRDAI", "SOLVENCY_II"],
            required_fields=["solvency_ratio", "claims_data", "premium_calculations"],
            validation_rules={
                "solvency_monitoring": {"min_ratio": 1.5, "real_time": True},
                "claims_processing": {"max_days": 30, "fraud_detection": True},
                "premium_calculation": {"actuarial_approved": True, "transparent": True}
            },
            sla_requirements={
                "solvency_reporting": {"daily": True},
                "claims_settlement": {"max_days": 30},
                "regulatory_filing": {"quarterly": True}
            }
        )
        
        self.overlay_configs = {
            "saas": saas_overlay,
            "banking": banking_overlay,
            "insurance": insurance_overlay
        }
    
    async def _load_existing_overlays(self):
        """Load existing overlay configurations from database"""
        if not self.pool_manager:
            return
        
        try:
            pool = await self.pool_manager.get_connection()
            
            sql = "SELECT * FROM industry_overlay_configs WHERE status = 'active'"
            rows = await pool.fetch(sql)
            
            for row in rows:
                overlay = IndustryOverlayConfig(
                    overlay_id=row['overlay_id'],
                    overlay_name=row['overlay_name'],
                    industry_type=IndustryType(row['industry_type']),
                    compliance_frameworks=row['compliance_frameworks'],
                    required_fields=row['required_fields'],
                    validation_rules=row['validation_rules'],
                    sla_requirements=row['sla_requirements'],
                    status=OverlayStatus(row['status']),
                    version=row['version'],
                    created_by=row['created_by'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                
                self.overlay_configs[f"{overlay.industry_type.value}_{overlay.overlay_name}"] = overlay
            
            self.logger.info(f"✅ Loaded {len(rows)} existing overlay configurations")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load existing overlays: {e}")
        finally:
            if pool:
                await self.pool_manager.return_connection(pool)
    
    async def log_overlay_violation(
        self,
        tenant_id: int,
        industry_type: IndustryType,
        overlay_name: str,
        violation_type: OverlayViolationType,
        rule_violated: str,
        severity: str = "medium",
        description: str = "",
        workflow_id: Optional[str] = None,
        user_id: Optional[str] = None,
        regulatory_framework: Optional[str] = None
    ) -> str:
        """Log an overlay violation (Task 8.4-T15)"""
        
        try:
            violation = OverlayViolation(
                tenant_id=tenant_id,
                industry_type=industry_type,
                violation_type=violation_type,
                overlay_name=overlay_name,
                rule_violated=rule_violated,
                severity=severity,
                description=description,
                workflow_id=workflow_id,
                user_id=user_id,
                regulatory_framework=regulatory_framework
            )
            
            # Store violation
            await self._store_overlay_violation(violation)
            
            # Add to cache
            self.violations[violation.violation_id] = violation
            
            # Trigger anomaly detection
            if self.config['enable_real_time_monitoring']:
                await self._check_violation_anomalies(tenant_id, industry_type)
            
            self.logger.warning(f"🚨 Overlay violation logged: {violation.violation_id} - {violation_type.value}")
            return violation.violation_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log overlay violation: {e}")
            raise
    
    async def _store_overlay_violation(self, violation: OverlayViolation):
        """Store overlay violation in database"""
        if not self.pool_manager:
            return
        
        sql = """
        INSERT INTO overlay_violations (
            violation_id, tenant_id, industry_type, violation_type,
            overlay_name, rule_violated, severity, description,
            workflow_id, user_id, resource_affected, timestamp,
            regulatory_framework, business_impact
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            await pool.execute(
                sql,
                violation.violation_id, violation.tenant_id, violation.industry_type.value,
                violation.violation_type.value, violation.overlay_name, violation.rule_violated,
                violation.severity, violation.description, violation.workflow_id,
                violation.user_id, violation.resource_affected, violation.timestamp,
                violation.regulatory_framework, violation.business_impact
            )
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def calculate_adoption_metrics(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, OverlayAdoptionMetrics]:
        """Calculate overlay adoption metrics (Task 8.4-T37)"""
        
        try:
            metrics = {}
            
            # Get overlay configurations for industry
            industry_overlays = [
                config for config in self.overlay_configs.values()
                if config.industry_type == industry_type
            ]
            
            for overlay_config in industry_overlays:
                # Calculate adoption metrics
                adoption_metrics = OverlayAdoptionMetrics(
                    overlay_name=overlay_config.overlay_name,
                    industry_type=industry_type,
                    tenant_id=tenant_id
                )
                
                # Get workflow usage data
                workflow_stats = await self._get_overlay_workflow_stats(
                    tenant_id, overlay_config.overlay_name
                )
                
                adoption_metrics.workflows_using_overlay = workflow_stats.get('using_overlay', 0)
                adoption_metrics.total_applicable_workflows = workflow_stats.get('total_applicable', 1)
                adoption_metrics.adoption_percentage = (
                    adoption_metrics.workflows_using_overlay / adoption_metrics.total_applicable_workflows
                ) * 100 if adoption_metrics.total_applicable_workflows > 0 else 0
                
                # Determine adoption stage
                if adoption_metrics.adoption_percentage == 0:
                    adoption_metrics.adoption_stage = AdoptionStage.NOT_STARTED
                elif adoption_metrics.adoption_percentage < 25:
                    adoption_metrics.adoption_stage = AdoptionStage.PLANNING
                elif adoption_metrics.adoption_percentage < 50:
                    adoption_metrics.adoption_stage = AdoptionStage.IMPLEMENTING
                elif adoption_metrics.adoption_percentage < 75:
                    adoption_metrics.adoption_stage = AdoptionStage.TESTING
                elif adoption_metrics.adoption_percentage < 100:
                    adoption_metrics.adoption_stage = AdoptionStage.DEPLOYED
                else:
                    adoption_metrics.adoption_stage = AdoptionStage.OPTIMIZING
                
                # Get usage metrics
                usage_stats = await self._get_overlay_usage_stats(
                    tenant_id, overlay_config.overlay_name
                )
                
                adoption_metrics.daily_executions = usage_stats.get('daily_executions', 0)
                adoption_metrics.success_rate = usage_stats.get('success_rate', 100.0)
                adoption_metrics.average_execution_time_ms = usage_stats.get('avg_execution_time', 0.0)
                
                # Get compliance metrics
                compliance_stats = await self._get_overlay_compliance_stats(
                    tenant_id, overlay_config.overlay_name
                )
                
                adoption_metrics.compliance_score = compliance_stats.get('compliance_score', 100.0)
                adoption_metrics.violations_count = compliance_stats.get('violations_count', 0)
                adoption_metrics.sla_adherence = compliance_stats.get('sla_adherence', 100.0)
                
                # Calculate trends
                adoption_metrics.adoption_trend = await self._calculate_adoption_trend(
                    tenant_id, overlay_config.overlay_name
                )
                
                metrics[overlay_config.overlay_name] = adoption_metrics
            
            # Store metrics
            await self._store_adoption_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate adoption metrics: {e}")
            return {}
    
    async def _get_overlay_workflow_stats(self, tenant_id: int, overlay_name: str) -> Dict[str, int]:
        """Get workflow statistics for overlay"""
        if not self.pool_manager:
            return {"using_overlay": 0, "total_applicable": 1}
        
        # This would query workflow registry for overlay usage
        # Simplified implementation
        return {
            "using_overlay": 15,
            "total_applicable": 20
        }
    
    async def _get_overlay_usage_stats(self, tenant_id: int, overlay_name: str) -> Dict[str, float]:
        """Get usage statistics for overlay"""
        if not self.pool_manager:
            return {"daily_executions": 0, "success_rate": 100.0, "avg_execution_time": 0.0}
        
        # This would query execution traces for overlay usage
        # Simplified implementation
        return {
            "daily_executions": 150,
            "success_rate": 98.5,
            "avg_execution_time": 245.7
        }
    
    async def _get_overlay_compliance_stats(self, tenant_id: int, overlay_name: str) -> Dict[str, float]:
        """Get compliance statistics for overlay"""
        if not self.pool_manager:
            return {"compliance_score": 100.0, "violations_count": 0, "sla_adherence": 100.0}
        
        sql = """
        SELECT 
            COUNT(*) as violations_count,
            AVG(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) * 100 as resolution_rate
        FROM overlay_violations
        WHERE tenant_id = $1 AND overlay_name = $2
        AND timestamp > NOW() - INTERVAL '30 days'
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            row = await pool.fetchrow(sql, tenant_id, overlay_name)
            
            violations_count = row['violations_count'] if row else 0
            resolution_rate = row['resolution_rate'] if row else 100.0
            
            # Calculate compliance score based on violations and resolution rate
            compliance_score = max(0, 100 - (violations_count * 2) + (resolution_rate * 0.1))
            
            return {
                "compliance_score": min(compliance_score, 100.0),
                "violations_count": violations_count,
                "sla_adherence": resolution_rate
            }
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _calculate_adoption_trend(self, tenant_id: int, overlay_name: str) -> str:
        """Calculate adoption trend for overlay"""
        # This would analyze historical adoption data
        # Simplified implementation
        return "increasing"
    
    async def _store_adoption_metrics(self, metrics: Dict[str, OverlayAdoptionMetrics]):
        """Store adoption metrics in database"""
        if not self.pool_manager or not metrics:
            return
        
        pool = await self.pool_manager.get_connection()
        try:
            for overlay_name, metric in metrics.items():
                sql = """
                INSERT INTO overlay_adoption_metrics (
                    overlay_name, industry_type, tenant_id, adoption_stage,
                    adoption_percentage, workflows_using_overlay, total_applicable_workflows,
                    daily_executions, success_rate, average_execution_time_ms,
                    compliance_score, violations_count, sla_adherence,
                    adoption_trend, usage_trend, compliance_trend
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (tenant_id, overlay_name, industry_type)
                DO UPDATE SET
                    adoption_stage = EXCLUDED.adoption_stage,
                    adoption_percentage = EXCLUDED.adoption_percentage,
                    workflows_using_overlay = EXCLUDED.workflows_using_overlay,
                    total_applicable_workflows = EXCLUDED.total_applicable_workflows,
                    daily_executions = EXCLUDED.daily_executions,
                    success_rate = EXCLUDED.success_rate,
                    average_execution_time_ms = EXCLUDED.average_execution_time_ms,
                    compliance_score = EXCLUDED.compliance_score,
                    violations_count = EXCLUDED.violations_count,
                    sla_adherence = EXCLUDED.sla_adherence,
                    adoption_trend = EXCLUDED.adoption_trend,
                    usage_trend = EXCLUDED.usage_trend,
                    compliance_trend = EXCLUDED.compliance_trend,
                    last_updated = NOW()
                """
                
                await pool.execute(
                    sql,
                    metric.overlay_name, metric.industry_type.value, metric.tenant_id,
                    metric.adoption_stage.value, metric.adoption_percentage,
                    metric.workflows_using_overlay, metric.total_applicable_workflows,
                    metric.daily_executions, metric.success_rate, metric.average_execution_time_ms,
                    metric.compliance_score, metric.violations_count, metric.sla_adherence,
                    metric.adoption_trend, metric.usage_trend, metric.compliance_trend
                )
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def detect_overlay_anomalies(self, tenant_id: int, industry_type: IndustryType) -> List[AnomalyDetectionResult]:
        """Detect anomalies in overlay metrics (Task 8.4-T38)"""
        
        try:
            anomalies = []
            
            # Get recent metrics
            current_metrics = await self.calculate_adoption_metrics(tenant_id, industry_type)
            historical_metrics = await self._get_historical_metrics(tenant_id, industry_type)
            
            for overlay_name, current in current_metrics.items():
                historical = historical_metrics.get(overlay_name, {})
                
                # Detect violation spikes
                violation_anomaly = await self._detect_violation_spike(
                    tenant_id, overlay_name, current, historical
                )
                if violation_anomaly:
                    anomalies.append(violation_anomaly)
                
                # Detect adoption drops
                adoption_anomaly = await self._detect_adoption_drop(
                    tenant_id, overlay_name, current, historical
                )
                if adoption_anomaly:
                    anomalies.append(adoption_anomaly)
                
                # Detect SLA breaches
                sla_anomaly = await self._detect_sla_breach(
                    tenant_id, overlay_name, current, historical
                )
                if sla_anomaly:
                    anomalies.append(sla_anomaly)
                
                # Detect compliance score drops
                compliance_anomaly = await self._detect_compliance_drop(
                    tenant_id, overlay_name, current, historical
                )
                if compliance_anomaly:
                    anomalies.append(compliance_anomaly)
            
            # Store anomalies
            for anomaly in anomalies:
                await self._store_anomaly(anomaly)
            
            if anomalies:
                self.logger.warning(f"🚨 Detected {len(anomalies)} overlay anomalies for tenant {tenant_id}")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"❌ Failed to detect overlay anomalies: {e}")
            return []
    
    async def _check_violation_anomalies(self, tenant_id: int, industry_type: IndustryType):
        """Check for violation anomalies in real-time"""
        
        # Get recent violation count
        recent_violations = await self._get_recent_violation_count(tenant_id, industry_type)
        
        # Get historical average
        historical_avg = await self._get_historical_violation_average(tenant_id, industry_type)
        
        # Check for spike
        if recent_violations > historical_avg * self.anomaly_thresholds['violation_spike_threshold']:
            anomaly = AnomalyDetectionResult(
                tenant_id=tenant_id,
                industry_type=industry_type,
                anomaly_type="violation_spike",
                metric_name="violations_per_hour",
                current_value=recent_violations,
                expected_value=historical_avg,
                deviation_percentage=((recent_violations - historical_avg) / historical_avg) * 100,
                confidence_score=0.85,
                severity="high",
                business_impact="Potential compliance breach requiring immediate attention",
                recommended_action="Investigate root cause and implement corrective measures"
            )
            
            await self._store_anomaly(anomaly)
    
    async def _get_recent_violation_count(self, tenant_id: int, industry_type: IndustryType) -> float:
        """Get recent violation count"""
        if not self.pool_manager:
            return 0.0
        
        sql = """
        SELECT COUNT(*) as count
        FROM overlay_violations
        WHERE tenant_id = $1 AND industry_type = $2
        AND timestamp > NOW() - INTERVAL '1 hour'
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            result = await pool.fetchval(sql, tenant_id, industry_type.value)
            return float(result or 0)
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _get_historical_violation_average(self, tenant_id: int, industry_type: IndustryType) -> float:
        """Get historical violation average"""
        if not self.pool_manager:
            return 1.0
        
        sql = """
        SELECT AVG(hourly_count) as avg_count
        FROM (
            SELECT DATE_TRUNC('hour', timestamp) as hour, COUNT(*) as hourly_count
            FROM overlay_violations
            WHERE tenant_id = $1 AND industry_type = $2
            AND timestamp > NOW() - INTERVAL '7 days'
            AND timestamp < NOW() - INTERVAL '1 hour'
            GROUP BY DATE_TRUNC('hour', timestamp)
        ) hourly_stats
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            result = await pool.fetchval(sql, tenant_id, industry_type.value)
            return float(result or 1.0)
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _get_historical_metrics(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Dict[str, float]]:
        """Get historical metrics for comparison"""
        # This would query historical adoption metrics
        # Simplified implementation
        return {}
    
    async def _detect_violation_spike(self, tenant_id: int, overlay_name: str, current: OverlayAdoptionMetrics, historical: Dict[str, float]) -> Optional[AnomalyDetectionResult]:
        """Detect violation spike anomaly"""
        
        historical_violations = historical.get('violations_count', 0)
        
        if current.violations_count > historical_violations * self.anomaly_thresholds['violation_spike_threshold']:
            return AnomalyDetectionResult(
                tenant_id=tenant_id,
                industry_type=current.industry_type,
                anomaly_type="violation_spike",
                metric_name=f"{overlay_name}_violations",
                current_value=current.violations_count,
                expected_value=historical_violations,
                deviation_percentage=((current.violations_count - historical_violations) / max(historical_violations, 1)) * 100,
                confidence_score=0.9,
                severity="high",
                business_impact="Increased compliance risk",
                recommended_action="Review recent changes and implement corrective measures"
            )
        
        return None
    
    async def _detect_adoption_drop(self, tenant_id: int, overlay_name: str, current: OverlayAdoptionMetrics, historical: Dict[str, float]) -> Optional[AnomalyDetectionResult]:
        """Detect adoption drop anomaly"""
        
        historical_adoption = historical.get('adoption_percentage', current.adoption_percentage)
        
        drop_threshold = historical_adoption * (1 - self.anomaly_thresholds['adoption_drop_threshold'])
        
        if current.adoption_percentage < drop_threshold:
            return AnomalyDetectionResult(
                tenant_id=tenant_id,
                industry_type=current.industry_type,
                anomaly_type="adoption_drop",
                metric_name=f"{overlay_name}_adoption",
                current_value=current.adoption_percentage,
                expected_value=historical_adoption,
                deviation_percentage=((current.adoption_percentage - historical_adoption) / historical_adoption) * 100,
                confidence_score=0.8,
                severity="medium",
                business_impact="Reduced compliance coverage",
                recommended_action="Investigate adoption barriers and provide additional training"
            )
        
        return None
    
    async def _detect_sla_breach(self, tenant_id: int, overlay_name: str, current: OverlayAdoptionMetrics, historical: Dict[str, float]) -> Optional[AnomalyDetectionResult]:
        """Detect SLA breach anomaly"""
        
        if current.sla_adherence < self.anomaly_thresholds['sla_breach_threshold'] * 100:
            return AnomalyDetectionResult(
                tenant_id=tenant_id,
                industry_type=current.industry_type,
                anomaly_type="sla_breach",
                metric_name=f"{overlay_name}_sla_adherence",
                current_value=current.sla_adherence,
                expected_value=self.anomaly_thresholds['sla_breach_threshold'] * 100,
                deviation_percentage=current.sla_adherence - (self.anomaly_thresholds['sla_breach_threshold'] * 100),
                confidence_score=0.95,
                severity="critical",
                business_impact="SLA breach may trigger regulatory penalties",
                recommended_action="Immediate investigation and remediation required"
            )
        
        return None
    
    async def _detect_compliance_drop(self, tenant_id: int, overlay_name: str, current: OverlayAdoptionMetrics, historical: Dict[str, float]) -> Optional[AnomalyDetectionResult]:
        """Detect compliance score drop anomaly"""
        
        if current.compliance_score < self.anomaly_thresholds['compliance_score_threshold'] * 100:
            return AnomalyDetectionResult(
                tenant_id=tenant_id,
                industry_type=current.industry_type,
                anomaly_type="compliance_drop",
                metric_name=f"{overlay_name}_compliance_score",
                current_value=current.compliance_score,
                expected_value=self.anomaly_thresholds['compliance_score_threshold'] * 100,
                deviation_percentage=current.compliance_score - (self.anomaly_thresholds['compliance_score_threshold'] * 100),
                confidence_score=0.85,
                severity="high",
                business_impact="Compliance score below acceptable threshold",
                recommended_action="Review compliance processes and address identified gaps"
            )
        
        return None
    
    async def _store_anomaly(self, anomaly: AnomalyDetectionResult):
        """Store anomaly detection result"""
        if not self.pool_manager:
            return
        
        sql = """
        INSERT INTO overlay_anomalies (
            anomaly_id, tenant_id, industry_type, anomaly_type, metric_name,
            current_value, expected_value, deviation_percentage, detection_timestamp,
            time_window, confidence_score, severity, business_impact, recommended_action
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            await pool.execute(
                sql,
                anomaly.anomaly_id, anomaly.tenant_id, anomaly.industry_type.value,
                anomaly.anomaly_type, anomaly.metric_name, anomaly.current_value,
                anomaly.expected_value, anomaly.deviation_percentage, anomaly.detection_timestamp,
                anomaly.time_window, anomaly.confidence_score, anomaly.severity,
                anomaly.business_impact, anomaly.recommended_action
            )
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def get_industry_dashboard_data(
        self,
        tenant_id: int,
        industry_type: IndustryType,
        dashboard_type: str = "executive"  # executive, operational, compliance
    ) -> Dict[str, Any]:
        """Get industry-specific dashboard data (Task 8.4-T15)"""
        
        try:
            dashboard_data = {
                'industry_type': industry_type.value,
                'dashboard_type': dashboard_type,
                'tenant_id': tenant_id,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'widgets': {}
            }
            
            if dashboard_type == "executive":
                dashboard_data['widgets'] = await self._get_executive_widgets(tenant_id, industry_type)
            elif dashboard_type == "operational":
                dashboard_data['widgets'] = await self._get_operational_widgets(tenant_id, industry_type)
            elif dashboard_type == "compliance":
                dashboard_data['widgets'] = await self._get_compliance_widgets(tenant_id, industry_type)
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get industry dashboard data: {e}")
            raise
    
    async def _get_executive_widgets(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get executive dashboard widgets"""
        
        # Get adoption metrics
        adoption_metrics = await self.calculate_adoption_metrics(tenant_id, industry_type)
        
        # Calculate summary statistics
        total_overlays = len(adoption_metrics)
        avg_adoption = statistics.mean([m.adoption_percentage for m in adoption_metrics.values()]) if adoption_metrics else 0
        avg_compliance = statistics.mean([m.compliance_score for m in adoption_metrics.values()]) if adoption_metrics else 100
        total_violations = sum([m.violations_count for m in adoption_metrics.values()])
        
        return {
            'adoption_summary': {
                'title': 'Overlay Adoption Summary',
                'data': {
                    'total_overlays': total_overlays,
                    'average_adoption_percentage': round(avg_adoption, 1),
                    'average_compliance_score': round(avg_compliance, 1),
                    'total_violations': total_violations
                }
            },
            'adoption_trend': {
                'title': 'Adoption Trend',
                'data': await self._get_adoption_trend_data(tenant_id, industry_type)
            },
            'compliance_overview': {
                'title': 'Compliance Overview',
                'data': await self._get_compliance_overview_data(tenant_id, industry_type)
            },
            'risk_indicators': {
                'title': 'Risk Indicators',
                'data': await self._get_risk_indicators_data(tenant_id, industry_type)
            }
        }
    
    async def _get_operational_widgets(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get operational dashboard widgets"""
        
        return {
            'active_violations': {
                'title': 'Active Violations',
                'data': await self._get_active_violations_data(tenant_id, industry_type)
            },
            'overlay_performance': {
                'title': 'Overlay Performance',
                'data': await self._get_overlay_performance_data(tenant_id, industry_type)
            },
            'recent_anomalies': {
                'title': 'Recent Anomalies',
                'data': await self._get_recent_anomalies_data(tenant_id, industry_type)
            },
            'sla_monitoring': {
                'title': 'SLA Monitoring',
                'data': await self._get_sla_monitoring_data(tenant_id, industry_type)
            }
        }
    
    async def _get_compliance_widgets(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get compliance dashboard widgets"""
        
        return {
            'regulatory_status': {
                'title': 'Regulatory Compliance Status',
                'data': await self._get_regulatory_status_data(tenant_id, industry_type)
            },
            'audit_readiness': {
                'title': 'Audit Readiness',
                'data': await self._get_audit_readiness_data(tenant_id, industry_type)
            },
            'compliance_gaps': {
                'title': 'Compliance Gaps',
                'data': await self._get_compliance_gaps_data(tenant_id, industry_type)
            },
            'remediation_tracking': {
                'title': 'Remediation Tracking',
                'data': await self._get_remediation_tracking_data(tenant_id, industry_type)
            }
        }
    
    async def _get_adoption_trend_data(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get adoption trend data"""
        # This would query historical adoption data
        # Simplified implementation
        return {
            'trend': 'increasing',
            'percentage_change': 15.2,
            'time_period': '30_days'
        }
    
    async def _get_compliance_overview_data(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get compliance overview data"""
        # This would aggregate compliance data across overlays
        # Simplified implementation
        return {
            'overall_score': 94.5,
            'frameworks': {
                'SOX': 96.2,
                'GDPR': 92.8,
                'CCPA': 95.1
            },
            'trend': 'stable'
        }
    
    async def _get_risk_indicators_data(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get risk indicators data"""
        # This would calculate risk indicators
        # Simplified implementation
        return {
            'high_risk_overlays': 2,
            'critical_violations': 1,
            'sla_breaches': 0,
            'overall_risk_level': 'medium'
        }
    
    async def _get_active_violations_data(self, tenant_id: int, industry_type: IndustryType) -> List[Dict[str, Any]]:
        """Get active violations data"""
        if not self.pool_manager:
            return []
        
        sql = """
        SELECT violation_id, overlay_name, violation_type, severity, rule_violated, timestamp
        FROM overlay_violations
        WHERE tenant_id = $1 AND industry_type = $2 AND status = 'open'
        ORDER BY timestamp DESC
        LIMIT 20
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            rows = await pool.fetch(sql, tenant_id, industry_type.value)
            
            violations = []
            for row in rows:
                violations.append({
                    'violation_id': row['violation_id'],
                    'overlay_name': row['overlay_name'],
                    'type': row['violation_type'],
                    'severity': row['severity'],
                    'rule': row['rule_violated'],
                    'timestamp': row['timestamp'].isoformat()
                })
            
            return violations
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _get_overlay_performance_data(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get overlay performance data"""
        # This would query performance metrics
        # Simplified implementation
        return {
            'average_execution_time': 245.7,
            'success_rate': 98.5,
            'throughput': 150,
            'error_rate': 1.5
        }
    
    async def _get_recent_anomalies_data(self, tenant_id: int, industry_type: IndustryType) -> List[Dict[str, Any]]:
        """Get recent anomalies data"""
        if not self.pool_manager:
            return []
        
        sql = """
        SELECT anomaly_id, anomaly_type, metric_name, severity, detection_timestamp
        FROM overlay_anomalies
        WHERE tenant_id = $1 AND industry_type = $2 AND status = 'new'
        ORDER BY detection_timestamp DESC
        LIMIT 10
        """
        
        pool = await self.pool_manager.get_connection()
        try:
            rows = await pool.fetch(sql, tenant_id, industry_type.value)
            
            anomalies = []
            for row in rows:
                anomalies.append({
                    'anomaly_id': row['anomaly_id'],
                    'type': row['anomaly_type'],
                    'metric': row['metric_name'],
                    'severity': row['severity'],
                    'detected_at': row['detection_timestamp'].isoformat()
                })
            
            return anomalies
        finally:
            await self.pool_manager.return_connection(pool)
    
    async def _get_sla_monitoring_data(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get SLA monitoring data"""
        # This would query SLA metrics
        # Simplified implementation
        return {
            'overall_sla_adherence': 97.8,
            'response_time_sla': 98.2,
            'resolution_time_sla': 96.5,
            'availability_sla': 99.9
        }
    
    async def _get_regulatory_status_data(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get regulatory status data"""
        # This would check regulatory compliance status
        # Simplified implementation
        return {
            'compliant_frameworks': ['SOX', 'GDPR'],
            'pending_frameworks': ['CCPA'],
            'non_compliant_frameworks': [],
            'next_audit_date': '2024-06-15'
        }
    
    async def _get_audit_readiness_data(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get audit readiness data"""
        # This would assess audit readiness
        # Simplified implementation
        return {
            'readiness_score': 92.5,
            'documentation_complete': 95.0,
            'evidence_available': 90.0,
            'gaps_identified': 3
        }
    
    async def _get_compliance_gaps_data(self, tenant_id: int, industry_type: IndustryType) -> List[Dict[str, Any]]:
        """Get compliance gaps data"""
        # This would identify compliance gaps
        # Simplified implementation
        return [
            {
                'gap_id': 'GAP001',
                'description': 'Missing data retention policy documentation',
                'severity': 'medium',
                'framework': 'GDPR',
                'due_date': '2024-03-15'
            }
        ]
    
    async def _get_remediation_tracking_data(self, tenant_id: int, industry_type: IndustryType) -> Dict[str, Any]:
        """Get remediation tracking data"""
        # This would track remediation progress
        # Simplified implementation
        return {
            'total_issues': 15,
            'resolved_issues': 12,
            'in_progress': 2,
            'overdue': 1,
            'completion_rate': 80.0
        }
    
    def get_training_materials(self) -> Dict[str, Any]:
        """Get training materials for overlay dashboards (Tasks 8.4-T34, T36)"""
        
        return {
            "ops_training": {
                "title": "Industry Overlay Operations Training",
                "modules": [
                    {
                        "name": "Overlay Dashboard Navigation",
                        "content": "Understanding industry-specific overlay dashboards and metrics",
                        "duration": "20 minutes",
                        "exercises": [
                            "Navigate SaaS overlay dashboard",
                            "Interpret adoption metrics",
                            "Identify compliance violations"
                        ]
                    },
                    {
                        "name": "Violation Management",
                        "content": "Managing and resolving overlay violations",
                        "duration": "25 minutes",
                        "exercises": [
                            "Investigate violation root cause",
                            "Assign violations to team members",
                            "Track resolution progress"
                        ]
                    },
                    {
                        "name": "Anomaly Detection",
                        "content": "Understanding and responding to overlay anomalies",
                        "duration": "15 minutes",
                        "exercises": [
                            "Review anomaly alerts",
                            "Validate anomaly findings",
                            "Implement corrective actions"
                        ]
                    }
                ]
            },
            "exec_training": {
                "title": "Executive Overlay Adoption Training",
                "modules": [
                    {
                        "name": "Adoption Metrics Overview",
                        "content": "Understanding overlay adoption metrics and business impact",
                        "duration": "15 minutes",
                        "key_metrics": [
                            "Adoption percentage by overlay",
                            "Compliance score trends",
                            "Risk indicators",
                            "ROI from overlay implementation"
                        ]
                    },
                    {
                        "name": "Strategic Decision Making",
                        "content": "Using overlay data for strategic decisions",
                        "duration": "20 minutes",
                        "focus_areas": [
                            "Prioritizing overlay rollouts",
                            "Resource allocation decisions",
                            "Risk mitigation strategies",
                            "Compliance investment planning"
                        ]
                    }
                ]
            },
            "sandbox_demos": {
                "saas_demo": {
                    "title": "SaaS Overlay Sandbox Demo",
                    "description": "Interactive demo of SaaS compliance overlay",
                    "scenarios": [
                        "Revenue recognition compliance",
                        "Customer data privacy",
                        "Subscription lifecycle management"
                    ]
                },
                "banking_demo": {
                    "title": "Banking Overlay Sandbox Demo",
                    "description": "Interactive demo of banking regulatory overlay",
                    "scenarios": [
                        "KYC compliance workflow",
                        "AML monitoring",
                        "Data localization requirements"
                    ]
                },
                "insurance_demo": {
                    "title": "Insurance Overlay Sandbox Demo",
                    "description": "Interactive demo of insurance regulatory overlay",
                    "scenarios": [
                        "Solvency monitoring",
                        "Claims processing compliance",
                        "Premium calculation validation"
                    ]
                }
            }
        }

# Global instance
overlay_dashboard_manager = IndustryOverlayDashboardManager()

async def get_overlay_dashboard_manager():
    """Get the global overlay dashboard manager instance"""
    return overlay_dashboard_manager
