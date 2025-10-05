# SLO/SLA Management System
# Tasks 6.6-T01, T02, T03, T23, T25: SLOs, SLAs, performance budgets, alarms, mapping

import asyncio
import json
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class SLOType(Enum):
    """Types of Service Level Objectives"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    EVIDENCE_GENERATION = "evidence_generation"
    COMPLIANCE_SCORE = "compliance_score"

class SLACategory(Enum):
    """Categories of Service Level Agreements"""
    OPERATIONAL = "operational"
    BUSINESS = "business"
    COMPLIANCE = "compliance"
    SECURITY = "security"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SLODefinition:
    """Service Level Objective definition"""
    slo_id: str
    name: str
    slo_type: SLOType
    target_value: float
    measurement_unit: str
    measurement_window_minutes: int
    comparison_operator: str  # <=, >=, ==
    industry_code: str
    tenant_tier: str
    description: str
    created_at: str
    updated_at: str

@dataclass
class SLACommitment:
    """Service Level Agreement commitment"""
    sla_id: str
    name: str
    category: SLACategory
    tenant_id: Optional[int]
    industry_code: str
    tenant_tier: str
    slo_mappings: List[str]  # List of SLO IDs
    commitment_value: float
    measurement_unit: str
    penalty_clause: str
    contract_reference: str
    effective_from: str
    effective_until: str
    created_at: str

@dataclass
class PerformanceBudget:
    """Performance budget for workflows"""
    budget_id: str
    workflow_id: str
    tenant_id: int
    max_execution_time_ms: int
    max_memory_mb: int
    max_cpu_cores: float
    max_storage_mb: int
    max_network_mb: int
    complexity_score_limit: float
    evidence_generation_time_ms: int
    created_at: str
    updated_at: str

@dataclass
class SLOViolation:
    """SLO violation record"""
    violation_id: str
    slo_id: str
    tenant_id: int
    violation_time: str
    actual_value: float
    target_value: float
    severity: AlertSeverity
    duration_minutes: int
    root_cause: Optional[str]
    resolution_time: Optional[str]
    resolved_by: Optional[str]

@dataclass
class SLABreach:
    """SLA breach record"""
    breach_id: str
    sla_id: str
    tenant_id: int
    breach_time: str
    related_violations: List[str]  # SLO violation IDs
    impact_assessment: str
    customer_notified: bool
    penalty_applied: bool
    resolution_time: Optional[str]

class SLOSLAManager:
    """
    Comprehensive SLO/SLA Management System
    Tasks 6.6-T01, T02, T03, T23, T25: SLOs, SLAs, budgets, alarms, mapping
    """
    
    def __init__(self, tenant_context_manager=None, metrics_collector=None, alert_manager=None):
        self.tenant_context_manager = tenant_context_manager
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        # Storage for SLO/SLA definitions
        self.slo_definitions: Dict[str, SLODefinition] = {}
        self.sla_commitments: Dict[str, SLACommitment] = {}
        self.performance_budgets: Dict[str, PerformanceBudget] = {}
        
        # Violation and breach tracking
        self.slo_violations: Dict[str, List[SLOViolation]] = defaultdict(list)
        self.sla_breaches: Dict[str, List[SLABreach]] = defaultdict(list)
        
        # Real-time monitoring data
        self.current_measurements: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.measurement_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize baseline SLOs
        self._initialize_baseline_slos()
        
        # Initialize industry-specific SLAs
        self._initialize_industry_slas()
    
    def _initialize_baseline_slos(self) -> None:
        """Initialize baseline SLO definitions"""
        
        # Baseline SLOs for different tenant tiers and industries
        baseline_slos = [
            # SaaS Industry SLOs
            {
                'slo_id': 'saas_latency_p95',
                'name': 'SaaS Workflow P95 Latency',
                'slo_type': SLOType.LATENCY,
                'target_value': 2000.0,
                'measurement_unit': 'milliseconds',
                'measurement_window_minutes': 60,
                'comparison_operator': '<=',
                'industry_code': 'SaaS',
                'tenant_tier': 'professional',
                'description': 'P95 latency for SaaS workflow execution should be under 2 seconds'
            },
            {
                'slo_id': 'saas_throughput',
                'name': 'SaaS Workflow Throughput',
                'slo_type': SLOType.THROUGHPUT,
                'target_value': 1000.0,
                'measurement_unit': 'workflows_per_minute',
                'measurement_window_minutes': 60,
                'comparison_operator': '>=',
                'industry_code': 'SaaS',
                'tenant_tier': 'professional',
                'description': 'SaaS workflows should process at least 1000 workflows per minute'
            },
            {
                'slo_id': 'saas_error_rate',
                'name': 'SaaS Workflow Error Rate',
                'slo_type': SLOType.ERROR_RATE,
                'target_value': 2.0,
                'measurement_unit': 'percent',
                'measurement_window_minutes': 60,
                'comparison_operator': '<=',
                'industry_code': 'SaaS',
                'tenant_tier': 'professional',
                'description': 'SaaS workflow error rate should be under 2%'
            },
            {
                'slo_id': 'saas_availability',
                'name': 'SaaS Service Availability',
                'slo_type': SLOType.AVAILABILITY,
                'target_value': 99.5,
                'measurement_unit': 'percent',
                'measurement_window_minutes': 1440,  # 24 hours
                'comparison_operator': '>=',
                'industry_code': 'SaaS',
                'tenant_tier': 'professional',
                'description': 'SaaS service availability should be at least 99.5%'
            },
            
            # Banking Industry SLOs (stricter requirements)
            {
                'slo_id': 'banking_latency_p95',
                'name': 'Banking Workflow P95 Latency',
                'slo_type': SLOType.LATENCY,
                'target_value': 1000.0,
                'measurement_unit': 'milliseconds',
                'measurement_window_minutes': 60,
                'comparison_operator': '<=',
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'description': 'P95 latency for Banking workflows should be under 1 second'
            },
            {
                'slo_id': 'banking_error_rate',
                'name': 'Banking Workflow Error Rate',
                'slo_type': SLOType.ERROR_RATE,
                'target_value': 0.5,
                'measurement_unit': 'percent',
                'measurement_window_minutes': 60,
                'comparison_operator': '<=',
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'description': 'Banking workflow error rate should be under 0.5%'
            },
            {
                'slo_id': 'banking_availability',
                'name': 'Banking Service Availability',
                'slo_type': SLOType.AVAILABILITY,
                'target_value': 99.9,
                'measurement_unit': 'percent',
                'measurement_window_minutes': 1440,
                'comparison_operator': '>=',
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'description': 'Banking service availability should be at least 99.9%'
            },
            {
                'slo_id': 'banking_evidence_generation',
                'name': 'Banking Evidence Generation Time',
                'slo_type': SLOType.EVIDENCE_GENERATION,
                'target_value': 500.0,
                'measurement_unit': 'milliseconds',
                'measurement_window_minutes': 60,
                'comparison_operator': '<=',
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'description': 'Banking evidence pack generation should complete within 500ms'
            },
            {
                'slo_id': 'banking_compliance_score',
                'name': 'Banking Compliance Score',
                'slo_type': SLOType.COMPLIANCE_SCORE,
                'target_value': 95.0,
                'measurement_unit': 'percent',
                'measurement_window_minutes': 1440,
                'comparison_operator': '>=',
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'description': 'Banking compliance score should be at least 95%'
            },
            
            # Insurance Industry SLOs
            {
                'slo_id': 'insurance_latency_p95',
                'name': 'Insurance Workflow P95 Latency',
                'slo_type': SLOType.LATENCY,
                'target_value': 1500.0,
                'measurement_unit': 'milliseconds',
                'measurement_window_minutes': 60,
                'comparison_operator': '<=',
                'industry_code': 'Insurance',
                'tenant_tier': 'enterprise',
                'description': 'P95 latency for Insurance workflows should be under 1.5 seconds'
            },
            {
                'slo_id': 'insurance_availability',
                'name': 'Insurance Service Availability',
                'slo_type': SLOType.AVAILABILITY,
                'target_value': 99.8,
                'measurement_unit': 'percent',
                'measurement_window_minutes': 1440,
                'comparison_operator': '>=',
                'industry_code': 'Insurance',
                'tenant_tier': 'enterprise',
                'description': 'Insurance service availability should be at least 99.8%'
            }
        ]
        
        # Create SLO definitions
        for slo_data in baseline_slos:
            slo = SLODefinition(
                slo_id=slo_data['slo_id'],
                name=slo_data['name'],
                slo_type=slo_data['slo_type'],
                target_value=slo_data['target_value'],
                measurement_unit=slo_data['measurement_unit'],
                measurement_window_minutes=slo_data['measurement_window_minutes'],
                comparison_operator=slo_data['comparison_operator'],
                industry_code=slo_data['industry_code'],
                tenant_tier=slo_data['tenant_tier'],
                description=slo_data['description'],
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat()
            )
            self.slo_definitions[slo.slo_id] = slo
    
    def _initialize_industry_slas(self) -> None:
        """Initialize industry-specific SLA commitments"""
        
        # Industry-specific SLA templates
        industry_slas = [
            # SaaS SLA
            {
                'sla_id': 'saas_operational_sla',
                'name': 'SaaS Operational Service Level Agreement',
                'category': SLACategory.OPERATIONAL,
                'industry_code': 'SaaS',
                'tenant_tier': 'professional',
                'slo_mappings': ['saas_latency_p95', 'saas_throughput', 'saas_error_rate', 'saas_availability'],
                'commitment_value': 99.5,
                'measurement_unit': 'percent_uptime',
                'penalty_clause': '5% monthly fee credit for each 0.1% below 99.5% uptime',
                'contract_reference': 'SaaS-SLA-v1.0'
            },
            
            # Banking SLA (stricter)
            {
                'sla_id': 'banking_operational_sla',
                'name': 'Banking Operational Service Level Agreement',
                'category': SLACategory.OPERATIONAL,
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'slo_mappings': ['banking_latency_p95', 'banking_error_rate', 'banking_availability'],
                'commitment_value': 99.9,
                'measurement_unit': 'percent_uptime',
                'penalty_clause': '10% monthly fee credit for each 0.1% below 99.9% uptime',
                'contract_reference': 'Banking-SLA-v1.0'
            },
            {
                'sla_id': 'banking_compliance_sla',
                'name': 'Banking Compliance Service Level Agreement',
                'category': SLACategory.COMPLIANCE,
                'industry_code': 'Banking',
                'tenant_tier': 'enterprise',
                'slo_mappings': ['banking_evidence_generation', 'banking_compliance_score'],
                'commitment_value': 95.0,
                'measurement_unit': 'percent_compliance',
                'penalty_clause': 'Regulatory escalation and remediation plan required below 95%',
                'contract_reference': 'Banking-Compliance-SLA-v1.0'
            },
            
            # Insurance SLA
            {
                'sla_id': 'insurance_operational_sla',
                'name': 'Insurance Operational Service Level Agreement',
                'category': SLACategory.OPERATIONAL,
                'industry_code': 'Insurance',
                'tenant_tier': 'enterprise',
                'slo_mappings': ['insurance_latency_p95', 'insurance_availability'],
                'commitment_value': 99.8,
                'measurement_unit': 'percent_uptime',
                'penalty_clause': '7.5% monthly fee credit for each 0.1% below 99.8% uptime',
                'contract_reference': 'Insurance-SLA-v1.0'
            }
        ]
        
        # Create SLA commitments
        for sla_data in industry_slas:
            sla = SLACommitment(
                sla_id=sla_data['sla_id'],
                name=sla_data['name'],
                category=sla_data['category'],
                tenant_id=None,  # Template, not tenant-specific
                industry_code=sla_data['industry_code'],
                tenant_tier=sla_data['tenant_tier'],
                slo_mappings=sla_data['slo_mappings'],
                commitment_value=sla_data['commitment_value'],
                measurement_unit=sla_data['measurement_unit'],
                penalty_clause=sla_data['penalty_clause'],
                contract_reference=sla_data['contract_reference'],
                effective_from=datetime.now(timezone.utc).isoformat(),
                effective_until=(datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
                created_at=datetime.now(timezone.utc).isoformat()
            )
            self.sla_commitments[sla.sla_id] = sla
    
    async def create_performance_budget(
        self,
        workflow_id: str,
        tenant_id: int,
        max_execution_time_ms: int = 5000,
        max_memory_mb: int = 512,
        max_cpu_cores: float = 1.0,
        max_storage_mb: int = 100,
        max_network_mb: int = 50,
        complexity_score_limit: float = 0.8,
        evidence_generation_time_ms: int = 1000
    ) -> PerformanceBudget:
        """Create performance budget for workflow"""
        
        budget_id = f"budget_{workflow_id}_{tenant_id}_{int(time.time())}"
        
        budget = PerformanceBudget(
            budget_id=budget_id,
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            max_execution_time_ms=max_execution_time_ms,
            max_memory_mb=max_memory_mb,
            max_cpu_cores=max_cpu_cores,
            max_storage_mb=max_storage_mb,
            max_network_mb=max_network_mb,
            complexity_score_limit=complexity_score_limit,
            evidence_generation_time_ms=evidence_generation_time_ms,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        
        self.performance_budgets[budget_id] = budget
        
        logger.info(f"Created performance budget: {budget_id} for workflow {workflow_id}")
        return budget
    
    async def check_performance_budget_compliance(
        self,
        workflow_id: str,
        tenant_id: int,
        actual_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check if workflow execution complies with performance budget"""
        
        # Find budget for workflow
        budget = None
        for b in self.performance_budgets.values():
            if b.workflow_id == workflow_id and b.tenant_id == tenant_id:
                budget = b
                break
        
        if not budget:
            return {
                'compliant': True,
                'reason': 'No performance budget defined',
                'violations': []
            }
        
        violations = []
        
        # Check each budget constraint
        if actual_metrics.get('execution_time_ms', 0) > budget.max_execution_time_ms:
            violations.append({
                'constraint': 'execution_time',
                'budget_limit': budget.max_execution_time_ms,
                'actual_value': actual_metrics.get('execution_time_ms'),
                'violation_percent': ((actual_metrics.get('execution_time_ms', 0) - budget.max_execution_time_ms) / budget.max_execution_time_ms) * 100
            })
        
        if actual_metrics.get('memory_mb', 0) > budget.max_memory_mb:
            violations.append({
                'constraint': 'memory_usage',
                'budget_limit': budget.max_memory_mb,
                'actual_value': actual_metrics.get('memory_mb'),
                'violation_percent': ((actual_metrics.get('memory_mb', 0) - budget.max_memory_mb) / budget.max_memory_mb) * 100
            })
        
        if actual_metrics.get('cpu_cores', 0) > budget.max_cpu_cores:
            violations.append({
                'constraint': 'cpu_usage',
                'budget_limit': budget.max_cpu_cores,
                'actual_value': actual_metrics.get('cpu_cores'),
                'violation_percent': ((actual_metrics.get('cpu_cores', 0) - budget.max_cpu_cores) / budget.max_cpu_cores) * 100
            })
        
        if actual_metrics.get('complexity_score', 0) > budget.complexity_score_limit:
            violations.append({
                'constraint': 'complexity_score',
                'budget_limit': budget.complexity_score_limit,
                'actual_value': actual_metrics.get('complexity_score'),
                'violation_percent': ((actual_metrics.get('complexity_score', 0) - budget.complexity_score_limit) / budget.complexity_score_limit) * 100
            })
        
        return {
            'compliant': len(violations) == 0,
            'budget_id': budget.budget_id,
            'violations': violations,
            'compliance_score': max(0, 100 - sum(v['violation_percent'] for v in violations))
        }
    
    async def record_slo_measurement(
        self,
        slo_id: str,
        tenant_id: int,
        measurement_value: float,
        timestamp: Optional[str] = None
    ) -> bool:
        """Record SLO measurement and check for violations"""
        
        if slo_id not in self.slo_definitions:
            logger.warning(f"Unknown SLO ID: {slo_id}")
            return False
        
        slo = self.slo_definitions[slo_id]
        measurement_time = timestamp or datetime.now(timezone.utc).isoformat()
        
        # Store measurement
        measurement_key = f"{slo_id}_{tenant_id}"
        self.current_measurements[measurement_key][slo.slo_type.value] = measurement_value
        self.measurement_history[measurement_key].append({
            'timestamp': measurement_time,
            'value': measurement_value
        })
        
        # Check for SLO violation
        is_violation = self._check_slo_violation(slo, measurement_value)
        
        if is_violation:
            await self._handle_slo_violation(slo, tenant_id, measurement_value, measurement_time)
        
        return True
    
    def _check_slo_violation(self, slo: SLODefinition, measurement_value: float) -> bool:
        """Check if measurement violates SLO"""
        
        if slo.comparison_operator == "<=":
            return measurement_value > slo.target_value
        elif slo.comparison_operator == ">=":
            return measurement_value < slo.target_value
        elif slo.comparison_operator == "==":
            return abs(measurement_value - slo.target_value) > (slo.target_value * 0.05)  # 5% tolerance
        
        return False
    
    async def _handle_slo_violation(
        self,
        slo: SLODefinition,
        tenant_id: int,
        actual_value: float,
        violation_time: str
    ) -> None:
        """Handle SLO violation"""
        
        # Determine severity
        violation_percent = abs((actual_value - slo.target_value) / slo.target_value) * 100
        
        if violation_percent > 50:
            severity = AlertSeverity.EMERGENCY
        elif violation_percent > 25:
            severity = AlertSeverity.CRITICAL
        elif violation_percent > 10:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        # Create violation record
        violation = SLOViolation(
            violation_id=str(uuid.uuid4()),
            slo_id=slo.slo_id,
            tenant_id=tenant_id,
            violation_time=violation_time,
            actual_value=actual_value,
            target_value=slo.target_value,
            severity=severity,
            duration_minutes=0,  # Will be updated if violation persists
            root_cause=None,
            resolution_time=None,
            resolved_by=None
        )
        
        self.slo_violations[slo.slo_id].append(violation)
        
        # Send alert
        if self.alert_manager:
            await self.alert_manager.send_slo_violation_alert(violation, slo)
        
        # Check for SLA breach
        await self._check_sla_breach(slo, tenant_id, violation)
        
        logger.warning(f"SLO violation detected: {slo.slo_id} for tenant {tenant_id} - {actual_value} vs {slo.target_value}")
    
    async def _check_sla_breach(
        self,
        slo: SLODefinition,
        tenant_id: int,
        violation: SLOViolation
    ) -> None:
        """Check if SLO violation constitutes SLA breach"""
        
        # Find applicable SLAs
        applicable_slas = []
        for sla in self.sla_commitments.values():
            if (slo.slo_id in sla.slo_mappings and 
                sla.industry_code == slo.industry_code and
                sla.tenant_tier == slo.tenant_tier):
                applicable_slas.append(sla)
        
        for sla in applicable_slas:
            # Check if this violation breaches SLA
            recent_violations = self._get_recent_violations(sla, tenant_id, hours=24)
            
            if len(recent_violations) >= 3:  # Multiple violations in 24h = breach
                breach = SLABreach(
                    breach_id=str(uuid.uuid4()),
                    sla_id=sla.sla_id,
                    tenant_id=tenant_id,
                    breach_time=violation.violation_time,
                    related_violations=[v.violation_id for v in recent_violations],
                    impact_assessment=f"Multiple SLO violations affecting {sla.name}",
                    customer_notified=False,
                    penalty_applied=False,
                    resolution_time=None
                )
                
                self.sla_breaches[sla.sla_id].append(breach)
                
                # Send critical alert for SLA breach
                if self.alert_manager:
                    await self.alert_manager.send_sla_breach_alert(breach, sla)
                
                logger.critical(f"SLA breach detected: {sla.sla_id} for tenant {tenant_id}")
    
    def _get_recent_violations(
        self,
        sla: SLACommitment,
        tenant_id: int,
        hours: int = 24
    ) -> List[SLOViolation]:
        """Get recent violations for SLA"""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_violations = []
        
        for slo_id in sla.slo_mappings:
            violations = self.slo_violations.get(slo_id, [])
            for violation in violations:
                if (violation.tenant_id == tenant_id and
                    datetime.fromisoformat(violation.violation_time.replace('Z', '+00:00')) > cutoff_time):
                    recent_violations.append(violation)
        
        return recent_violations
    
    async def get_slo_status_dashboard(self, tenant_id: int) -> Dict[str, Any]:
        """Get SLO status dashboard for tenant"""
        
        # Get tenant context for industry/tier
        tenant_context = None
        if self.tenant_context_manager:
            tenant_context = await self.tenant_context_manager.get_tenant_context(tenant_id)
        
        industry_code = tenant_context.industry_code.value if tenant_context else 'SaaS'
        tenant_tier = tenant_context.tenant_tier.value if tenant_context else 'professional'
        
        # Get applicable SLOs
        applicable_slos = []
        for slo in self.slo_definitions.values():
            if slo.industry_code == industry_code and slo.tenant_tier == tenant_tier:
                applicable_slos.append(slo)
        
        # Get current status for each SLO
        slo_statuses = []
        for slo in applicable_slos:
            measurement_key = f"{slo.slo_id}_{tenant_id}"
            current_value = self.current_measurements[measurement_key].get(slo.slo_type.value, 0.0)
            
            # Check compliance
            is_compliant = not self._check_slo_violation(slo, current_value)
            
            # Get recent violations
            recent_violations = [
                v for v in self.slo_violations.get(slo.slo_id, [])
                if v.tenant_id == tenant_id and
                datetime.fromisoformat(v.violation_time.replace('Z', '+00:00')) > 
                datetime.now(timezone.utc) - timedelta(hours=24)
            ]
            
            slo_statuses.append({
                'slo_id': slo.slo_id,
                'name': slo.name,
                'type': slo.slo_type.value,
                'target_value': slo.target_value,
                'current_value': current_value,
                'measurement_unit': slo.measurement_unit,
                'is_compliant': is_compliant,
                'compliance_percent': self._calculate_compliance_percent(slo, current_value),
                'violations_24h': len(recent_violations),
                'last_violation': recent_violations[-1].violation_time if recent_violations else None
            })
        
        # Get applicable SLAs
        applicable_slas = []
        for sla in self.sla_commitments.values():
            if sla.industry_code == industry_code and sla.tenant_tier == tenant_tier:
                applicable_slas.append({
                    'sla_id': sla.sla_id,
                    'name': sla.name,
                    'category': sla.category.value,
                    'commitment_value': sla.commitment_value,
                    'measurement_unit': sla.measurement_unit,
                    'penalty_clause': sla.penalty_clause,
                    'breaches_30d': len([
                        b for b in self.sla_breaches.get(sla.sla_id, [])
                        if b.tenant_id == tenant_id and
                        datetime.fromisoformat(b.breach_time.replace('Z', '+00:00')) >
                        datetime.now(timezone.utc) - timedelta(days=30)
                    ])
                })
        
        return {
            'tenant_id': tenant_id,
            'industry_code': industry_code,
            'tenant_tier': tenant_tier,
            'slo_status': slo_statuses,
            'sla_commitments': applicable_slas,
            'overall_compliance': sum(s['compliance_percent'] for s in slo_statuses) / len(slo_statuses) if slo_statuses else 100.0,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_compliance_percent(self, slo: SLODefinition, current_value: float) -> float:
        """Calculate compliance percentage for SLO"""
        
        if slo.comparison_operator == "<=":
            if current_value <= slo.target_value:
                return 100.0
            else:
                return max(0.0, 100.0 - ((current_value - slo.target_value) / slo.target_value * 100))
        elif slo.comparison_operator == ">=":
            if current_value >= slo.target_value:
                return 100.0
            else:
                return max(0.0, (current_value / slo.target_value) * 100)
        
        return 100.0 if abs(current_value - slo.target_value) <= (slo.target_value * 0.05) else 0.0
    
    async def get_slo_sla_mapping_matrix(self) -> Dict[str, Any]:
        """Get SLO to SLA mapping matrix for governance clarity"""
        
        mapping_matrix = {
            'slo_definitions': [asdict(slo) for slo in self.slo_definitions.values()],
            'sla_commitments': [asdict(sla) for sla in self.sla_commitments.values()],
            'mappings': [],
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Create mapping relationships
        for sla in self.sla_commitments.values():
            for slo_id in sla.slo_mappings:
                if slo_id in self.slo_definitions:
                    slo = self.slo_definitions[slo_id]
                    mapping_matrix['mappings'].append({
                        'sla_id': sla.sla_id,
                        'sla_name': sla.name,
                        'slo_id': slo.slo_id,
                        'slo_name': slo.name,
                        'industry_code': sla.industry_code,
                        'tenant_tier': sla.tenant_tier,
                        'target_value': slo.target_value,
                        'measurement_unit': slo.measurement_unit,
                        'penalty_clause': sla.penalty_clause
                    })
        
        return mapping_matrix

# Global SLO/SLA manager
slo_sla_manager = SLOSLAManager()
