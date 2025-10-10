"""
Task 9.2.18: Build Data Quality Service
Comprehensive data quality monitoring and validation service

Features:
- Real-time data quality monitoring
- Configurable quality rules and thresholds
- Multi-dimensional quality scoring
- Automated quality alerts and remediation
- Compliance-aware quality standards
- Cross-plane data quality correlation
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import statistics

logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"      # Missing/null values
    ACCURACY = "accuracy"              # Correctness of data
    CONSISTENCY = "consistency"        # Data consistency across sources
    VALIDITY = "validity"              # Format and constraint compliance
    UNIQUENESS = "uniqueness"          # Duplicate detection
    TIMELINESS = "timeliness"          # Data freshness and currency
    INTEGRITY = "integrity"            # Referential integrity
    CONFORMITY = "conformity"          # Schema and standard compliance

class QualitySeverity(Enum):
    """Quality issue severity levels"""
    CRITICAL = "critical"    # Blocks processing
    HIGH = "high"           # Significant impact
    MEDIUM = "medium"       # Moderate impact
    LOW = "low"             # Minor impact
    INFO = "info"           # Informational

class DataSource(Enum):
    """Data sources for quality monitoring"""
    SALESFORCE = "salesforce"
    POSTGRES = "postgres"
    AZURE_BLOB = "azure_blob"
    API_ENDPOINTS = "api_endpoints"
    EXECUTION_TRACES = "execution_traces"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    EVIDENCE_PACKS = "evidence_packs"
    TELEMETRY = "telemetry"

@dataclass
class QualityRule:
    """Data quality rule definition"""
    rule_id: str
    rule_name: str
    dimension: QualityDimension
    data_source: DataSource
    
    # Rule configuration
    rule_type: str  # threshold, pattern, custom
    rule_expression: str  # SQL, regex, or custom logic
    threshold_value: Optional[float] = None
    expected_pattern: Optional[str] = None
    
    # Severity and impact
    severity: QualitySeverity = QualitySeverity.MEDIUM
    blocking: bool = False  # Blocks processing if failed
    
    # Compliance relevance
    compliance_relevant: bool = False
    compliance_frameworks: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True
    tenant_specific: bool = False

@dataclass
class QualityCheck:
    """Data quality check execution result"""
    check_id: str
    tenant_id: int
    rule_id: str
    data_source: DataSource
    dimension: QualityDimension
    
    # Execution details
    executed_at: datetime
    execution_duration_ms: float
    
    # Results
    passed: bool
    score: float  # 0.0 to 1.0
    records_checked: int
    records_failed: int
    
    # Issue details
    severity: QualitySeverity
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    remediation_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    blocking_issue: bool = False
    compliance_impact: bool = False

@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    report_id: str
    tenant_id: int
    reporting_period: Dict[str, str]
    
    # Overall quality metrics
    overall_score: float
    dimension_scores: Dict[str, float]
    source_scores: Dict[str, float]
    
    # Issue summary
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_issues: int
    blocking_issues: int
    
    # Trends
    score_trend: str  # improving, declining, stable
    issue_trend: str
    
    # Compliance status
    compliance_score: float
    compliance_issues: List[Dict[str, Any]]
    
    # Generated metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)

class DataQualityService:
    """
    Task 9.2.18: Build Data Quality Service
    Comprehensive data quality monitoring and validation
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Service configuration - DYNAMIC, no hardcoding
        self.config = {
            "service_name": "data_quality_service",
            "version": "1.0.0",
            "monitoring_enabled": os.getenv("DQ_MONITORING_ENABLED", "true").lower() == "true",
            "real_time_checks": os.getenv("DQ_REAL_TIME_CHECKS", "true").lower() == "true",
            "batch_check_interval_hours": int(os.getenv("DQ_BATCH_INTERVAL", "6")),
            "alert_threshold": float(os.getenv("DQ_ALERT_THRESHOLD", "0.8")),
            "blocking_threshold": float(os.getenv("DQ_BLOCKING_THRESHOLD", "0.5")),
            "compliance_threshold": float(os.getenv("DQ_COMPLIANCE_THRESHOLD", "0.95")),
            "max_issues_per_check": int(os.getenv("DQ_MAX_ISSUES", "100"))
        }
        
        # Quality rules and checks
        self.quality_rules: Dict[str, QualityRule] = {}
        self.tenant_rules: Dict[int, Dict[str, QualityRule]] = {}
        self.quality_checks: Dict[str, QualityCheck] = {}
        self.quality_reports: Dict[str, QualityReport] = {}
        
        # Metrics and monitoring
        self.metrics = {
            "total_rules": 0,
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "critical_issues": 0,
            "blocking_issues": 0,
            "average_score": 0.0,
            "compliance_violations": 0
        }
        
        # Initialize default quality rules
        self._initialize_default_rules()
        
        logger.info("🔍 Data Quality Service initialized with comprehensive monitoring")
    
    def _initialize_default_rules(self):
        """Initialize default data quality rules"""
        
        default_rules = [
            # Completeness rules
            QualityRule(
                rule_id="completeness_required_fields",
                rule_name="Required Fields Completeness",
                dimension=QualityDimension.COMPLETENESS,
                data_source=DataSource.SALESFORCE,
                rule_type="threshold",
                rule_expression="SELECT (COUNT(*) - COUNT(required_field)) / COUNT(*) FROM table",
                threshold_value=0.05,  # Max 5% missing values
                severity=QualitySeverity.HIGH,
                blocking=True,
                compliance_relevant=True,
                compliance_frameworks=["SOX", "GDPR"]
            ),
            
            # Accuracy rules
            QualityRule(
                rule_id="accuracy_email_format",
                rule_name="Email Format Accuracy",
                dimension=QualityDimension.ACCURACY,
                data_source=DataSource.POSTGRES,
                rule_type="pattern",
                rule_expression="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
                severity=QualitySeverity.MEDIUM,
                compliance_relevant=True,
                compliance_frameworks=["GDPR", "DPDP"]
            ),
            
            # Consistency rules
            QualityRule(
                rule_id="consistency_cross_source",
                rule_name="Cross-Source Data Consistency",
                dimension=QualityDimension.CONSISTENCY,
                data_source=DataSource.API_ENDPOINTS,
                rule_type="custom",
                rule_expression="compare_sources(source1, source2, key_field)",
                threshold_value=0.95,  # 95% consistency required
                severity=QualitySeverity.HIGH,
                compliance_relevant=True
            ),
            
            # Validity rules
            QualityRule(
                rule_id="validity_date_range",
                rule_name="Date Range Validity",
                dimension=QualityDimension.VALIDITY,
                data_source=DataSource.EXECUTION_TRACES,
                rule_type="threshold",
                rule_expression="SELECT COUNT(*) FROM table WHERE date_field > NOW() OR date_field < '1900-01-01'",
                threshold_value=0.0,  # No invalid dates allowed
                severity=QualitySeverity.CRITICAL,
                blocking=True
            ),
            
            # Uniqueness rules
            QualityRule(
                rule_id="uniqueness_primary_keys",
                rule_name="Primary Key Uniqueness",
                dimension=QualityDimension.UNIQUENESS,
                data_source=DataSource.POSTGRES,
                rule_type="threshold",
                rule_expression="SELECT COUNT(*) - COUNT(DISTINCT primary_key) FROM table",
                threshold_value=0.0,  # No duplicates allowed
                severity=QualitySeverity.CRITICAL,
                blocking=True,
                compliance_relevant=True
            ),
            
            # Timeliness rules
            QualityRule(
                rule_id="timeliness_data_freshness",
                rule_name="Data Freshness Timeliness",
                dimension=QualityDimension.TIMELINESS,
                data_source=DataSource.TELEMETRY,
                rule_type="threshold",
                rule_expression="SELECT COUNT(*) FROM table WHERE updated_at < NOW() - INTERVAL '24 hours'",
                threshold_value=0.1,  # Max 10% stale data
                severity=QualitySeverity.MEDIUM
            ),
            
            # Integrity rules
            QualityRule(
                rule_id="integrity_referential",
                rule_name="Referential Integrity",
                dimension=QualityDimension.INTEGRITY,
                data_source=DataSource.POSTGRES,
                rule_type="custom",
                rule_expression="check_foreign_key_constraints()",
                threshold_value=1.0,  # 100% integrity required
                severity=QualitySeverity.CRITICAL,
                blocking=True,
                compliance_relevant=True
            ),
            
            # Conformity rules
            QualityRule(
                rule_id="conformity_schema_compliance",
                rule_name="Schema Conformity",
                dimension=QualityDimension.CONFORMITY,
                data_source=DataSource.KNOWLEDGE_GRAPH,
                rule_type="custom",
                rule_expression="validate_schema_compliance()",
                threshold_value=0.98,  # 98% conformity required
                severity=QualitySeverity.HIGH,
                compliance_relevant=True
            )
        ]
        
        # Register default rules
        for rule in default_rules:
            self.quality_rules[rule.rule_id] = rule
            self.metrics["total_rules"] += 1
        
        logger.info(f"✅ Initialized {len(default_rules)} default quality rules")
    
    async def create_quality_rule(self, rule: QualityRule, 
                                tenant_id: Optional[int] = None) -> bool:
        """Create a new data quality rule"""
        try:
            # Validate rule
            if not self._validate_quality_rule(rule):
                logger.error(f"❌ Invalid quality rule: {rule.rule_id}")
                return False
            
            # Store rule
            if tenant_id:
                # Tenant-specific rule
                if tenant_id not in self.tenant_rules:
                    self.tenant_rules[tenant_id] = {}
                self.tenant_rules[tenant_id][rule.rule_id] = rule
                rule.tenant_specific = True
            else:
                # Global rule
                self.quality_rules[rule.rule_id] = rule
            
            self.metrics["total_rules"] += 1
            
            logger.info(f"✅ Created quality rule: {rule.rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create quality rule: {e}")
            return False
    
    async def execute_quality_check(self, tenant_id: int, rule_id: str,
                                  data_sample: Optional[Dict[str, Any]] = None) -> Optional[QualityCheck]:
        """Execute a data quality check"""
        try:
            # Get rule
            rule = await self._get_quality_rule(tenant_id, rule_id)
            if not rule or not rule.enabled:
                logger.warning(f"⚠️ Quality rule not found or disabled: {rule_id}")
                return None
            
            start_time = datetime.utcnow()
            
            # Execute quality check
            check_result = await self._execute_rule_check(rule, tenant_id, data_sample)
            
            end_time = datetime.utcnow()
            execution_duration = (end_time - start_time).total_seconds() * 1000
            
            # Create quality check record
            check = QualityCheck(
                check_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                rule_id=rule_id,
                data_source=rule.data_source,
                dimension=rule.dimension,
                executed_at=start_time,
                execution_duration_ms=execution_duration,
                passed=check_result["passed"],
                score=check_result["score"],
                records_checked=check_result["records_checked"],
                records_failed=check_result["records_failed"],
                severity=rule.severity,
                issues_found=check_result["issues"],
                remediation_suggestions=check_result["remediation"],
                blocking_issue=rule.blocking and not check_result["passed"],
                compliance_impact=rule.compliance_relevant and not check_result["passed"]
            )
            
            # Store check result
            self.quality_checks[check.check_id] = check
            
            # Update metrics
            self._update_check_metrics(check)
            
            # Handle alerts if needed
            if not check.passed:
                await self._handle_quality_alert(check, rule)
            
            logger.info(f"{'✅' if check.passed else '❌'} Quality check: {rule.rule_name} (Score: {check.score:.2f})")
            return check
            
        except Exception as e:
            logger.error(f"❌ Failed to execute quality check {rule_id}: {e}")
            return None
    
    async def run_quality_assessment(self, tenant_id: int,
                                   data_sources: Optional[List[DataSource]] = None,
                                   dimensions: Optional[List[QualityDimension]] = None) -> Optional[QualityReport]:
        """Run comprehensive quality assessment"""
        try:
            # Get applicable rules
            rules = await self._get_applicable_rules(tenant_id, data_sources, dimensions)
            
            if not rules:
                logger.warning(f"⚠️ No applicable quality rules found for tenant {tenant_id}")
                return None
            
            # Execute all quality checks
            checks = []
            for rule in rules:
                check = await self.execute_quality_check(tenant_id, rule.rule_id)
                if check:
                    checks.append(check)
            
            # Generate quality report
            report = await self._generate_quality_report(tenant_id, checks)
            
            # Store report
            self.quality_reports[report.report_id] = report
            
            logger.info(f"📊 Quality assessment completed: {len(checks)} checks, score: {report.overall_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to run quality assessment: {e}")
            return None
    
    async def get_quality_dashboard(self, tenant_id: int) -> Dict[str, Any]:
        """Get quality dashboard data for tenant"""
        try:
            # Get recent checks
            recent_checks = [
                check for check in self.quality_checks.values()
                if (check.tenant_id == tenant_id and 
                    check.executed_at > datetime.utcnow() - timedelta(days=7))
            ]
            
            # Calculate dashboard metrics
            dashboard = {
                "tenant_id": tenant_id,
                "overview": {
                    "total_checks": len(recent_checks),
                    "passed_checks": len([c for c in recent_checks if c.passed]),
                    "failed_checks": len([c for c in recent_checks if not c.passed]),
                    "average_score": statistics.mean([c.score for c in recent_checks]) if recent_checks else 0.0,
                    "critical_issues": len([c for c in recent_checks if c.severity == QualitySeverity.CRITICAL]),
                    "blocking_issues": len([c for c in recent_checks if c.blocking_issue])
                },
                "dimension_scores": self._calculate_dimension_scores(recent_checks),
                "source_scores": self._calculate_source_scores(recent_checks),
                "recent_issues": [
                    {
                        "check_id": c.check_id,
                        "rule_name": await self._get_rule_name(c.rule_id),
                        "dimension": c.dimension.value,
                        "severity": c.severity.value,
                        "score": c.score,
                        "executed_at": c.executed_at.isoformat(),
                        "blocking": c.blocking_issue
                    }
                    for c in recent_checks if not c.passed
                ][:10],  # Last 10 issues
                "compliance_status": await self._get_compliance_status(tenant_id, recent_checks),
                "trends": await self._calculate_quality_trends(tenant_id),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"❌ Failed to get quality dashboard: {e}")
            return {}
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        return {
            "service_config": self.config,
            "metrics": self.metrics,
            "active_rules": len(self.quality_rules),
            "tenant_specific_rules": sum(len(rules) for rules in self.tenant_rules.values()),
            "recent_checks": len([
                c for c in self.quality_checks.values()
                if c.executed_at > datetime.utcnow() - timedelta(hours=24)
            ]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _validate_quality_rule(self, rule: QualityRule) -> bool:
        """Validate quality rule configuration"""
        if not rule.rule_id or not rule.rule_name:
            return False
        
        if rule.rule_type == "threshold" and rule.threshold_value is None:
            return False
        
        if rule.rule_type == "pattern" and not rule.expected_pattern:
            return False
        
        return True
    
    async def _get_quality_rule(self, tenant_id: int, rule_id: str) -> Optional[QualityRule]:
        """Get quality rule by ID (tenant-specific or global)"""
        # Check tenant-specific rules first
        if tenant_id in self.tenant_rules:
            tenant_rule = self.tenant_rules[tenant_id].get(rule_id)
            if tenant_rule:
                return tenant_rule
        
        # Check global rules
        return self.quality_rules.get(rule_id)
    
    async def _get_applicable_rules(self, tenant_id: int,
                                  data_sources: Optional[List[DataSource]] = None,
                                  dimensions: Optional[List[QualityDimension]] = None) -> List[QualityRule]:
        """Get applicable quality rules for assessment"""
        all_rules = []
        
        # Add global rules
        all_rules.extend(self.quality_rules.values())
        
        # Add tenant-specific rules
        if tenant_id in self.tenant_rules:
            all_rules.extend(self.tenant_rules[tenant_id].values())
        
        # Filter by data sources
        if data_sources:
            all_rules = [rule for rule in all_rules if rule.data_source in data_sources]
        
        # Filter by dimensions
        if dimensions:
            all_rules = [rule for rule in all_rules if rule.dimension in dimensions]
        
        # Only enabled rules
        return [rule for rule in all_rules if rule.enabled]
    
    async def _execute_rule_check(self, rule: QualityRule, tenant_id: int,
                                data_sample: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the actual quality rule check"""
        try:
            # Simulate quality check execution
            # In production, this would execute actual SQL queries, API calls, etc.
            
            import random
            
            # Simulate different outcomes based on rule type and severity
            base_score = 0.9 if rule.severity in [QualitySeverity.LOW, QualitySeverity.INFO] else 0.7
            score = base_score + random.uniform(-0.2, 0.1)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            passed = score >= (rule.threshold_value or 0.8)
            
            records_checked = random.randint(100, 10000)
            records_failed = int(records_checked * (1 - score)) if not passed else 0
            
            issues = []
            remediation = []
            
            if not passed:
                # Generate sample issues based on dimension
                if rule.dimension == QualityDimension.COMPLETENESS:
                    issues.append({
                        "issue_type": "missing_values",
                        "field": "required_field",
                        "count": records_failed,
                        "percentage": (1 - score) * 100
                    })
                    remediation.append("Implement data validation at source")
                    remediation.append("Add default value handling")
                
                elif rule.dimension == QualityDimension.ACCURACY:
                    issues.append({
                        "issue_type": "format_violation",
                        "field": "email_field",
                        "invalid_count": records_failed,
                        "examples": ["invalid@", "notanemail", "missing@domain"]
                    })
                    remediation.append("Add format validation")
                    remediation.append("Implement data cleansing rules")
                
                elif rule.dimension == QualityDimension.UNIQUENESS:
                    issues.append({
                        "issue_type": "duplicate_records",
                        "field": "primary_key",
                        "duplicate_count": records_failed
                    })
                    remediation.append("Implement deduplication logic")
                    remediation.append("Add unique constraints")
            
            return {
                "passed": passed,
                "score": score,
                "records_checked": records_checked,
                "records_failed": records_failed,
                "issues": issues,
                "remediation": remediation
            }
            
        except Exception as e:
            logger.error(f"❌ Quality rule execution failed: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "records_checked": 0,
                "records_failed": 0,
                "issues": [{"issue_type": "execution_error", "error": str(e)}],
                "remediation": ["Fix rule configuration", "Check data source connectivity"]
            }
    
    async def _generate_quality_report(self, tenant_id: int, 
                                     checks: List[QualityCheck]) -> QualityReport:
        """Generate comprehensive quality report"""
        if not checks:
            # Empty report
            return QualityReport(
                report_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                reporting_period={
                    "start": datetime.utcnow().isoformat(),
                    "end": datetime.utcnow().isoformat()
                },
                overall_score=0.0,
                dimension_scores={},
                source_scores={},
                total_checks=0,
                passed_checks=0,
                failed_checks=0,
                critical_issues=0,
                blocking_issues=0,
                score_trend="stable",
                issue_trend="stable",
                compliance_score=0.0,
                compliance_issues=[]
            )
        
        # Calculate overall metrics
        overall_score = statistics.mean([c.score for c in checks])
        passed_checks = len([c for c in checks if c.passed])
        failed_checks = len(checks) - passed_checks
        critical_issues = len([c for c in checks if c.severity == QualitySeverity.CRITICAL])
        blocking_issues = len([c for c in checks if c.blocking_issue])
        
        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(checks)
        source_scores = self._calculate_source_scores(checks)
        
        # Calculate compliance metrics
        compliance_checks = [c for c in checks if c.compliance_impact]
        compliance_score = statistics.mean([c.score for c in compliance_checks]) if compliance_checks else 1.0
        compliance_issues = [
            {
                "check_id": c.check_id,
                "dimension": c.dimension.value,
                "severity": c.severity.value,
                "score": c.score,
                "frameworks": await self._get_rule_frameworks(c.rule_id)
            }
            for c in compliance_checks if not c.passed
        ]
        
        return QualityReport(
            report_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            reporting_period={
                "start": min(c.executed_at for c in checks).isoformat(),
                "end": max(c.executed_at for c in checks).isoformat()
            },
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            source_scores=source_scores,
            total_checks=len(checks),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            critical_issues=critical_issues,
            blocking_issues=blocking_issues,
            score_trend="stable",  # Would calculate from historical data
            issue_trend="stable",
            compliance_score=compliance_score,
            compliance_issues=compliance_issues
        )
    
    def _calculate_dimension_scores(self, checks: List[QualityCheck]) -> Dict[str, float]:
        """Calculate quality scores by dimension"""
        dimension_scores = {}
        
        for dimension in QualityDimension:
            dimension_checks = [c for c in checks if c.dimension == dimension]
            if dimension_checks:
                dimension_scores[dimension.value] = statistics.mean([c.score for c in dimension_checks])
        
        return dimension_scores
    
    def _calculate_source_scores(self, checks: List[QualityCheck]) -> Dict[str, float]:
        """Calculate quality scores by data source"""
        source_scores = {}
        
        for source in DataSource:
            source_checks = [c for c in checks if c.data_source == source]
            if source_checks:
                source_scores[source.value] = statistics.mean([c.score for c in source_checks])
        
        return source_scores
    
    def _update_check_metrics(self, check: QualityCheck):
        """Update service metrics based on check result"""
        self.metrics["total_checks"] += 1
        
        if check.passed:
            self.metrics["passed_checks"] += 1
        else:
            self.metrics["failed_checks"] += 1
        
        if check.severity == QualitySeverity.CRITICAL:
            self.metrics["critical_issues"] += 1
        
        if check.blocking_issue:
            self.metrics["blocking_issues"] += 1
        
        if check.compliance_impact:
            self.metrics["compliance_violations"] += 1
        
        # Update average score
        total_checks = self.metrics["total_checks"]
        current_avg = self.metrics["average_score"]
        self.metrics["average_score"] = ((current_avg * (total_checks - 1)) + check.score) / total_checks
    
    async def _handle_quality_alert(self, check: QualityCheck, rule: QualityRule):
        """Handle quality alert for failed check"""
        alert = {
            "alert_type": "data_quality_issue",
            "check_id": check.check_id,
            "rule_name": rule.rule_name,
            "dimension": check.dimension.value,
            "severity": check.severity.value,
            "score": check.score,
            "blocking": check.blocking_issue,
            "compliance_impact": check.compliance_impact,
            "tenant_id": check.tenant_id,
            "timestamp": check.executed_at.isoformat()
        }
        
        # In production, this would send alerts to monitoring systems
        logger.warning(f"⚠️ DATA QUALITY ALERT: {alert}")
    
    async def _get_rule_name(self, rule_id: str) -> str:
        """Get rule name by ID"""
        # Check global rules
        if rule_id in self.quality_rules:
            return self.quality_rules[rule_id].rule_name
        
        # Check tenant rules
        for tenant_rules in self.tenant_rules.values():
            if rule_id in tenant_rules:
                return tenant_rules[rule_id].rule_name
        
        return "Unknown Rule"
    
    async def _get_rule_frameworks(self, rule_id: str) -> List[str]:
        """Get compliance frameworks for rule"""
        # Check global rules
        if rule_id in self.quality_rules:
            return self.quality_rules[rule_id].compliance_frameworks
        
        # Check tenant rules
        for tenant_rules in self.tenant_rules.values():
            if rule_id in tenant_rules:
                return tenant_rules[rule_id].compliance_frameworks
        
        return []
    
    async def _get_compliance_status(self, tenant_id: int, 
                                   checks: List[QualityCheck]) -> Dict[str, Any]:
        """Get compliance status from quality checks"""
        compliance_checks = [c for c in checks if c.compliance_impact]
        
        if not compliance_checks:
            return {"status": "no_compliance_checks", "score": 1.0}
        
        compliance_score = statistics.mean([c.score for c in compliance_checks])
        failed_compliance = [c for c in compliance_checks if not c.passed]
        
        return {
            "status": "compliant" if compliance_score >= self.config["compliance_threshold"] else "non_compliant",
            "score": compliance_score,
            "total_compliance_checks": len(compliance_checks),
            "failed_compliance_checks": len(failed_compliance),
            "critical_compliance_issues": len([c for c in failed_compliance if c.severity == QualitySeverity.CRITICAL])
        }
    
    async def _calculate_quality_trends(self, tenant_id: int) -> Dict[str, str]:
        """Calculate quality trends (would use historical data in production)"""
        return {
            "score_trend": "stable",
            "issue_trend": "stable",
            "compliance_trend": "stable"
        }

# Global data quality service instance
data_quality_service = DataQualityService()
