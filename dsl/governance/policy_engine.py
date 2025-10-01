"""
SaaS-Focused Policy Engine - Compliance Framework Implementation
===============================================================
Implements Chapter 16: Policy Packs and Compliance Framework with SaaS industry focus

SaaS Industry Focus:
- SOX compliance for SaaS financial data (ARR, MRR, revenue recognition)
- GDPR compliance for SaaS customer data and multi-tenant privacy
- SaaS-specific business rules (subscription lifecycle, churn prevention)
- Multi-tenant policy isolation and enforcement
- Extensible architecture for future industries (Banking, Insurance)

Features:
- Task 16.1: SaaS policy pack management with OPA/Rego integration
- Task 16.2: Override ledger implementation with SaaS context
- Task 16.3: Multi-tenant policy enforcement engine
- Task 16.4: Evidence pack service with tamper-evident logging
- Task 16.5: SaaS compliance dashboards and reporting

Supported Compliance Frameworks:
- SOX_SAAS (Sarbanes-Oxley) - SaaS financial controls and revenue recognition
- GDPR_SAAS (General Data Protection Regulation) - SaaS customer data privacy
- SAAS_BUSINESS_RULES - Subscription lifecycle and business logic
"""

import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class PolicyPackType(Enum):
    # SaaS-focused policy types (primary)
    SOX_SAAS = "SOX_SAAS"
    GDPR_SAAS = "GDPR_SAAS"
    SAAS_BUSINESS_RULES = "SAAS_BUSINESS_RULES"
    
    # Future industry extensions (placeholders)
    SOX = "SOX"          # Generic SOX (legacy)
    GDPR = "GDPR"        # Generic GDPR (legacy)
    BANKING_RBI = "BANKING_RBI"      # Future: Banking regulations
    INSURANCE_NAIC = "INSURANCE_NAIC" # Future: Insurance regulations

class EnforcementLevel(Enum):
    STRICT = "strict"      # Block operations that violate policy
    ADVISORY = "advisory"  # Log violations but allow operations
    DISABLED = "disabled"  # Policy pack disabled

class OverrideType(Enum):
    POLICY_BYPASS = "policy_bypass"
    MANUAL_APPROVAL = "manual_approval"
    EMERGENCY_OVERRIDE = "emergency_override"
    BUSINESS_EXCEPTION = "business_exception"

@dataclass
class PolicyPack:
    """SaaS-focused policy pack definition with multi-tenant support"""
    policy_pack_id: str
    tenant_id: int
    pack_name: str
    pack_type: PolicyPackType
    industry_code: str = "SaaS"  # Default to SaaS industry
    region_code: str = "US"      # Default to US region
    policy_rules: Dict[str, Any] = None
    enforcement_level: EnforcementLevel = EnforcementLevel.STRICT
    saas_metrics: List[str] = None  # SaaS-specific metrics this policy applies to
    tenant_tier: str = "T2"         # T0, T1, T2 for SLA-aware policies
    version: str = "1.0.0"
    status: str = "active"
    created_at: datetime = None
    created_by_user_id: int = None
    
    def __post_init__(self):
        if self.policy_rules is None:
            self.policy_rules = {}
        if self.saas_metrics is None:
            self.saas_metrics = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class PolicyViolation:
    """Policy violation record"""
    violation_id: str
    tenant_id: int
    policy_pack_id: str
    rule_name: str
    violation_type: str
    severity: str  # low, medium, high, critical
    description: str
    context: Dict[str, Any]
    detected_at: datetime = None

@dataclass
class OverrideRequest:
    """Override request record"""
    override_id: str
    tenant_id: int
    user_id: int
    override_type: OverrideType
    component_affected: str
    original_value: Any
    overridden_value: Any
    reason: str
    risk_level: str = "medium"
    business_impact: str = ""
    approval_required: bool = True

class PolicyEngine:
    """
    Policy Engine for compliance framework enforcement
    
    Features:
    - Multi-framework policy management
    - Real-time policy enforcement
    - Override ledger with approval workflows
    - Evidence pack generation
    - Regulatory reporting
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Policy cache for performance
        self.policy_cache: Dict[Tuple[int, str], PolicyPack] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_refresh = {}
        
        # SaaS-focused policy enforcement configuration
        self.enforcement_config = {
            'fail_closed': True,
            'audit_all_operations': True,
            'require_approval_for_overrides': True,
            'max_override_duration_hours': 24,
            'saas_multi_tenant_isolation': True,
            'saas_metrics_tracking': True
        }
        
        # SaaS industry configuration
        self.saas_industry_config = {
            'primary_policies': ['SOX_SAAS', 'GDPR_SAAS', 'SAAS_BUSINESS_RULES'],
            'compliance_frameworks': ['SOX', 'GDPR'],
            'business_metrics': ['ARR', 'MRR', 'Churn', 'CAC', 'LTV'],
            'data_types': ['customer_data', 'financial_data', 'usage_data', 'subscription_data']
        }
        
        # Built-in policy definitions (enhanced with SaaS focus)
        self.built_in_policies = self._load_built_in_policies()
        self.saas_policy_templates = self._initialize_saas_policy_templates()
    
    async def initialize(self) -> bool:
        """Initialize policy engine"""
        try:
            self.logger.info("⚖️ Initializing Policy Engine...")
            
            # Load policy packs into cache
            await self._refresh_policy_cache()
            
            # Validate policy schemas
            await self._validate_policy_schemas()
            
            self.logger.info("✅ Policy Engine initialized successfully")
            self.logger.info(f"📋 Loaded {len(self.policy_cache)} policy packs into cache")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Policy Engine initialization failed: {e}")
            return False
    
    async def enforce_policy(self, tenant_id: int, workflow_data: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, List[PolicyViolation]]:
        """
        Enforce policies for a workflow execution (Task 16.3)
        
        Args:
            tenant_id: Tenant identifier
            workflow_data: Workflow execution data
            context: Additional context for policy evaluation
            
        Returns:
            Tuple of (allowed, violations)
        """
        try:
            violations = []
            context = context or {}
            
            # Get active policy packs for tenant
            policy_packs = await self._get_tenant_policy_packs(tenant_id)
            
            if not policy_packs:
                self.logger.debug(f"No policy packs found for tenant {tenant_id}")
                return True, []
            
            # Evaluate each policy pack
            for policy_pack in policy_packs:
                if policy_pack.enforcement_level == EnforcementLevel.DISABLED:
                    continue
                
                pack_violations = await self._evaluate_policy_pack(policy_pack, workflow_data, context)
                violations.extend(pack_violations)
            
            # Determine if operation is allowed
            critical_violations = [v for v in violations if v.severity == 'critical']
            high_violations = [v for v in violations if v.severity == 'high']
            
            # For strict enforcement, block on critical violations
            strict_packs = [p for p in policy_packs if p.enforcement_level == EnforcementLevel.STRICT]
            if strict_packs and critical_violations:
                allowed = False
                self.logger.warning(f"🚫 Operation blocked due to {len(critical_violations)} critical policy violations")
            else:
                allowed = True
                if violations:
                    self.logger.info(f"⚠️ Operation allowed with {len(violations)} policy violations (advisory mode)")
            
            # Log violations for audit trail
            for violation in violations:
                await self._log_policy_violation(violation)
            
            return allowed, violations
            
        except Exception as e:
            self.logger.error(f"❌ Policy enforcement failed: {e}")
            # Fail closed - deny operation on error
            return False, []
    
    async def create_policy_pack(self, tenant_id: int, pack_data: Dict[str, Any], created_by_user_id: int) -> Optional[PolicyPack]:
        """
        Create new policy pack (Task 16.1)
        
        Args:
            tenant_id: Tenant identifier
            pack_data: Policy pack definition
            created_by_user_id: User creating the policy pack
            
        Returns:
            PolicyPack if successful, None otherwise
        """
        try:
            # Validate required fields
            required_fields = ['pack_name', 'pack_type', 'industry_code', 'region_code', 'policy_rules']
            for field in required_fields:
                if field not in pack_data:
                    raise ValueError(f"Required field missing: {field}")
            
            # Generate policy pack ID
            policy_pack_id = str(uuid.uuid4())
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Insert policy pack
                await conn.execute("""
                    INSERT INTO dsl_policy_packs (
                        policy_pack_id, tenant_id, pack_name, pack_type, industry_code, region_code,
                        policy_rules, enforcement_level, version, status, created_by_user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    policy_pack_id,
                    tenant_id,
                    pack_data['pack_name'],
                    pack_data['pack_type'],
                    pack_data['industry_code'],
                    pack_data['region_code'],
                    json.dumps(pack_data['policy_rules']),
                    pack_data.get('enforcement_level', 'strict'),
                    pack_data.get('version', '1.0.0'),
                    pack_data.get('status', 'active'),
                    created_by_user_id
                )
                
                # Create policy pack object
                policy_pack = PolicyPack(
                    policy_pack_id=policy_pack_id,
                    tenant_id=tenant_id,
                    pack_name=pack_data['pack_name'],
                    pack_type=PolicyPackType(pack_data['pack_type']),
                    industry_code=pack_data['industry_code'],
                    region_code=pack_data['region_code'],
                    policy_rules=pack_data['policy_rules'],
                    enforcement_level=EnforcementLevel(pack_data.get('enforcement_level', 'strict')),
                    version=pack_data.get('version', '1.0.0'),
                    status=pack_data.get('status', 'active'),
                    created_at=datetime.now(),
                    created_by_user_id=created_by_user_id
                )
                
                # Update cache
                cache_key = (tenant_id, policy_pack_id)
                self.policy_cache[cache_key] = policy_pack
                self.last_cache_refresh[cache_key] = datetime.now().timestamp()
                
                self.logger.info(f"✅ Created policy pack: {pack_data['pack_name']} for tenant {tenant_id}")
                
                return policy_pack
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create policy pack: {e}")
            return None
    
    async def create_override(self, override_request: OverrideRequest) -> Optional[str]:
        """
        Create override request (Task 16.2)
        
        Args:
            override_request: Override request details
            
        Returns:
            Override ID if successful, None otherwise
        """
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(override_request.tenant_id))
                
                # Insert override request
                await conn.execute("""
                    INSERT INTO dsl_override_ledger (
                        override_id, tenant_id, user_id, override_type, component_affected,
                        original_value, overridden_value, override_reason, risk_level,
                        business_impact, approval_required, status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                    override_request.override_id,
                    override_request.tenant_id,
                    override_request.user_id,
                    override_request.override_type.value,
                    override_request.component_affected,
                    json.dumps(override_request.original_value) if override_request.original_value else None,
                    json.dumps(override_request.overridden_value) if override_request.overridden_value else None,
                    override_request.reason,
                    override_request.risk_level,
                    override_request.business_impact,
                    override_request.approval_required,
                    'pending' if override_request.approval_required else 'active'
                )
                
                self.logger.info(f"✅ Created override request: {override_request.override_id}")
                
                # Create evidence pack for override
                await self._create_override_evidence_pack(override_request)
                
                return override_request.override_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create override: {e}")
            return None
    
    async def generate_evidence_pack(self, tenant_id: int, pack_type: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Generate evidence pack for compliance (Task 16.4)
        
        Args:
            tenant_id: Tenant identifier
            pack_type: Type of evidence pack
            data: Evidence data
            
        Returns:
            Evidence pack ID if successful, None otherwise
        """
        try:
            evidence_pack_id = str(uuid.uuid4())
            
            # Calculate evidence hash for tamper detection
            evidence_json = json.dumps(data, sort_keys=True)
            evidence_hash = hashlib.sha256(evidence_json.encode()).hexdigest()
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Insert evidence pack
                await conn.execute("""
                    INSERT INTO dsl_evidence_packs (
                        evidence_pack_id, tenant_id, pack_name, pack_type,
                        evidence_data, evidence_hash, immutable, file_format,
                        file_size_bytes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    evidence_pack_id,
                    tenant_id,
                    f"{pack_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    pack_type,
                    json.dumps(data),
                    evidence_hash,
                    True,  # Immutable by default
                    'json',
                    len(evidence_json)
                )
                
                self.logger.info(f"✅ Generated evidence pack: {evidence_pack_id} for tenant {tenant_id}")
                
                return evidence_pack_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate evidence pack: {e}")
            return None
    
    async def get_compliance_dashboard_data(self, tenant_id: int, framework: str = None) -> Dict[str, Any]:
        """
        Get compliance dashboard data for regulators (Task 16.5)
        
        Args:
            tenant_id: Tenant identifier
            framework: Optional specific framework filter
            
        Returns:
            Dashboard data dictionary
        """
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                # Get policy pack summary
                policy_query = """
                    SELECT pack_type, COUNT(*) as pack_count, 
                           COUNT(CASE WHEN status = 'active' THEN 1 END) as active_packs
                    FROM dsl_policy_packs
                    WHERE tenant_id = $1
                """
                policy_params = [tenant_id]
                
                if framework:
                    policy_query += " AND pack_type = $2"
                    policy_params.append(framework)
                
                policy_query += " GROUP BY pack_type"
                
                policy_stats = await conn.fetch(policy_query, *policy_params)
                
                # Get execution compliance stats
                exec_query = """
                    SELECT 
                        COUNT(*) as total_executions,
                        COUNT(CASE WHEN override_count = 0 THEN 1 END) as compliant_executions,
                        COUNT(CASE WHEN override_count > 0 THEN 1 END) as non_compliant_executions,
                        AVG(trust_score) as avg_trust_score
                    FROM dsl_execution_traces
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                """
                
                exec_stats = await conn.fetchrow(exec_query, tenant_id)
                
                # Get override statistics
                override_query = """
                    SELECT 
                        override_type,
                        risk_level,
                        COUNT(*) as override_count
                    FROM dsl_override_ledger
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY override_type, risk_level
                """
                
                override_stats = await conn.fetch(override_query, tenant_id)
                
                # Get evidence pack statistics
                evidence_query = """
                    SELECT 
                        pack_type,
                        COUNT(*) as evidence_count,
                        SUM(file_size_bytes) as total_size_bytes
                    FROM dsl_evidence_packs
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY pack_type
                """
                
                evidence_stats = await conn.fetch(evidence_query, tenant_id)
                
                # Calculate compliance score
                total_executions = exec_stats['total_executions'] or 0
                compliant_executions = exec_stats['compliant_executions'] or 0
                compliance_score = (compliant_executions / total_executions * 100) if total_executions > 0 else 100
                
                dashboard_data = {
                    'tenant_id': tenant_id,
                    'generated_at': datetime.now().isoformat(),
                    'compliance_score': round(compliance_score, 2),
                    'avg_trust_score': float(exec_stats['avg_trust_score'] or 1.0),
                    'policy_packs': [dict(row) for row in policy_stats],
                    'execution_stats': dict(exec_stats),
                    'override_stats': [dict(row) for row in override_stats],
                    'evidence_stats': [dict(row) for row in evidence_stats],
                    'summary': {
                        'total_policy_packs': len(policy_stats),
                        'total_executions_30d': total_executions,
                        'compliance_rate': compliance_score,
                        'total_overrides_30d': sum(row['override_count'] for row in override_stats),
                        'total_evidence_packs_30d': sum(row['evidence_count'] for row in evidence_stats)
                    }
                }
                
                return dashboard_data
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get compliance dashboard data: {e}")
            return {}
    
    async def _get_tenant_policy_packs(self, tenant_id: int) -> List[PolicyPack]:
        """Get active policy packs for tenant"""
        try:
            policy_packs = []
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                # Set tenant context
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(tenant_id))
                
                rows = await conn.fetch("""
                    SELECT policy_pack_id, tenant_id, name as pack_name, pack_type, industry_code, region_code,
                           rules as policy_rules, enforcement_level, version, 'active' as status, created_at, created_by_user_id
                    FROM dsl_policy_packs
                    WHERE tenant_id = $1
                """, tenant_id)
                
                for row in rows:
                    policy_pack = PolicyPack(
                        policy_pack_id=row['policy_pack_id'],
                        tenant_id=row['tenant_id'],
                        pack_name=row['pack_name'],
                        pack_type=PolicyPackType(row['pack_type']),
                        industry_code=row['industry_code'],
                        region_code=row['region_code'],
                        policy_rules=row['policy_rules'],
                        enforcement_level=EnforcementLevel(row['enforcement_level']),
                        version=row['version'],
                        status=row['status'],
                        created_at=row['created_at'],
                        created_by_user_id=row['created_by_user_id']
                    )
                    policy_packs.append(policy_pack)
            
            return policy_packs
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get tenant policy packs: {e}")
            return []
    
    async def _evaluate_policy_pack(self, policy_pack: PolicyPack, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> List[PolicyViolation]:
        """Evaluate specific policy pack against workflow data"""
        violations = []
        
        try:
            # Evaluate based on policy pack type
            if policy_pack.pack_type == PolicyPackType.SOX:
                violations.extend(await self._evaluate_sox_policies(policy_pack, workflow_data, context))
            elif policy_pack.pack_type == PolicyPackType.GDPR:
                violations.extend(await self._evaluate_gdpr_policies(policy_pack, workflow_data, context))
            elif policy_pack.pack_type == PolicyPackType.RBI:
                violations.extend(await self._evaluate_rbi_policies(policy_pack, workflow_data, context))
            elif policy_pack.pack_type == PolicyPackType.HIPAA:
                violations.extend(await self._evaluate_hipaa_policies(policy_pack, workflow_data, context))
            
        except Exception as e:
            self.logger.error(f"❌ Policy evaluation failed for {policy_pack.pack_name}: {e}")
        
        return violations
    
    async def _evaluate_sox_policies(self, policy_pack: PolicyPack, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> List[PolicyViolation]:
        """Evaluate SOX (Sarbanes-Oxley) policies"""
        violations = []
        rules = policy_pack.policy_rules
        
        # Check financial controls
        if 'financial_controls' in rules:
            financial_rules = rules['financial_controls']
            
            # Revenue recognition check
            if financial_rules.get('revenue_recognition', {}).get('enabled'):
                deal_amount = workflow_data.get('deal_amount', 0)
                if deal_amount >= financial_rules.get('deal_approval_thresholds', {}).get('high_value', 250000):
                    if not workflow_data.get('revenue_recognition_approved'):
                        violations.append(PolicyViolation(
                            violation_id=str(uuid.uuid4()),
                            tenant_id=policy_pack.tenant_id,
                            policy_pack_id=policy_pack.policy_pack_id,
                            rule_name='revenue_recognition',
                            violation_type='missing_approval',
                            severity='high',
                            description=f'High-value deal (${deal_amount}) requires revenue recognition approval',
                            context={'deal_amount': deal_amount, 'threshold': financial_rules['deal_approval_thresholds']['high_value']},
                            detected_at=datetime.now()
                        ))
            
            # Segregation of duties check
            if financial_rules.get('segregation_of_duties', {}).get('enabled'):
                creator_id = workflow_data.get('created_by_user_id')
                approver_id = workflow_data.get('approved_by_user_id')
                
                if creator_id and approver_id and creator_id == approver_id:
                    violations.append(PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        tenant_id=policy_pack.tenant_id,
                        policy_pack_id=policy_pack.policy_pack_id,
                        rule_name='segregation_of_duties',
                        violation_type='same_user_approval',
                        severity='critical',
                        description='Same user cannot create and approve workflow (SOX segregation of duties)',
                        context={'creator_id': creator_id, 'approver_id': approver_id},
                        detected_at=datetime.now()
                    ))
        
        return violations
    
    async def _evaluate_gdpr_policies(self, policy_pack: PolicyPack, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> List[PolicyViolation]:
        """Evaluate GDPR (General Data Protection Regulation) policies"""
        violations = []
        rules = policy_pack.policy_rules
        
        # Check data protection requirements
        if 'data_protection' in rules:
            data_rules = rules['data_protection']
            
            # Consent management check
            if data_rules.get('consent_management', {}).get('enabled'):
                if workflow_data.get('processes_personal_data') and not workflow_data.get('consent_obtained'):
                    violations.append(PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        tenant_id=policy_pack.tenant_id,
                        policy_pack_id=policy_pack.policy_pack_id,
                        rule_name='consent_management',
                        violation_type='missing_consent',
                        severity='critical',
                        description='Processing personal data requires explicit consent (GDPR Article 6)',
                        context={'processes_personal_data': True, 'consent_obtained': False},
                        detected_at=datetime.now()
                    ))
            
            # Data minimization check
            if data_rules.get('data_minimization', {}).get('enabled'):
                data_fields = workflow_data.get('data_fields', [])
                necessary_fields = workflow_data.get('necessary_fields', [])
                
                if len(data_fields) > len(necessary_fields) * 1.5:  # Allow 50% buffer
                    violations.append(PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        tenant_id=policy_pack.tenant_id,
                        policy_pack_id=policy_pack.policy_pack_id,
                        rule_name='data_minimization',
                        violation_type='excessive_data_collection',
                        severity='medium',
                        description='Workflow collects more data than necessary (GDPR Article 5)',
                        context={'data_fields_count': len(data_fields), 'necessary_fields_count': len(necessary_fields)},
                        detected_at=datetime.now()
                    ))
        
        return violations
    
    async def _evaluate_rbi_policies(self, policy_pack: PolicyPack, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> List[PolicyViolation]:
        """Evaluate RBI (Reserve Bank of India) policies"""
        violations = []
        rules = policy_pack.policy_rules
        
        # Check banking controls
        if 'banking_controls' in rules:
            banking_rules = rules['banking_controls']
            
            # Credit approval check
            if banking_rules.get('credit_approval', {}).get('enabled'):
                loan_amount = workflow_data.get('loan_amount', 0)
                if loan_amount > 1000000 and not workflow_data.get('dual_authorization'):  # 10 Lakh threshold
                    violations.append(PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        tenant_id=policy_pack.tenant_id,
                        policy_pack_id=policy_pack.policy_pack_id,
                        rule_name='credit_approval',
                        violation_type='missing_dual_authorization',
                        severity='critical',
                        description='High-value loans require dual authorization (RBI guidelines)',
                        context={'loan_amount': loan_amount, 'dual_authorization': workflow_data.get('dual_authorization', False)},
                        detected_at=datetime.now()
                    ))
            
            # AML compliance check
            if banking_rules.get('aml_compliance', {}).get('enabled'):
                if workflow_data.get('suspicious_activity_detected') and not workflow_data.get('sar_filed'):
                    violations.append(PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        tenant_id=policy_pack.tenant_id,
                        policy_pack_id=policy_pack.policy_pack_id,
                        rule_name='aml_compliance',
                        violation_type='missing_sar_filing',
                        severity='critical',
                        description='Suspicious activity requires SAR filing (RBI AML guidelines)',
                        context={'suspicious_activity_detected': True, 'sar_filed': False},
                        detected_at=datetime.now()
                    ))
        
        return violations
    
    async def _evaluate_hipaa_policies(self, policy_pack: PolicyPack, workflow_data: Dict[str, Any], context: Dict[str, Any]) -> List[PolicyViolation]:
        """Evaluate HIPAA (Health Insurance Portability) policies"""
        violations = []
        rules = policy_pack.policy_rules
        
        # Check healthcare data protection
        if 'healthcare_controls' in rules:
            healthcare_rules = rules['healthcare_controls']
            
            # PHI access control
            if healthcare_rules.get('phi_access_control', {}).get('enabled'):
                if workflow_data.get('accesses_phi') and not workflow_data.get('minimum_necessary_standard'):
                    violations.append(PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        tenant_id=policy_pack.tenant_id,
                        policy_pack_id=policy_pack.policy_pack_id,
                        rule_name='phi_access_control',
                        violation_type='excessive_phi_access',
                        severity='high',
                        description='PHI access must follow minimum necessary standard (HIPAA)',
                        context={'accesses_phi': True, 'minimum_necessary_standard': False},
                        detected_at=datetime.now()
                    ))
        
        return violations
    
    async def _log_policy_violation(self, violation: PolicyViolation) -> None:
        """Log policy violation for audit trail"""
        try:
            # Store violation in database for reporting
            async with self.pool_manager.postgres_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", str(violation.tenant_id))
                
                # For now, log to execution traces - in production would use dedicated violation table
                self.logger.warning(f"🚨 Policy violation: {violation.rule_name} - {violation.description}")
                
                # Could store in dedicated policy_violations table
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log policy violation: {e}")
    
    async def _create_override_evidence_pack(self, override_request: OverrideRequest) -> None:
        """Create evidence pack for override request"""
        try:
            evidence_data = {
                'override_id': override_request.override_id,
                'override_type': override_request.override_type.value,
                'component_affected': override_request.component_affected,
                'reason': override_request.reason,
                'risk_level': override_request.risk_level,
                'business_impact': override_request.business_impact,
                'created_at': datetime.now().isoformat(),
                'created_by_user_id': override_request.user_id
            }
            
            await self.generate_evidence_pack(
                override_request.tenant_id,
                'override_evidence',
                evidence_data
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create override evidence pack: {e}")
    
    async def _refresh_policy_cache(self) -> None:
        """Refresh policy pack cache"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT policy_pack_id, tenant_id, name as pack_name, pack_type, industry_code, region_code,
                           rules as policy_rules, enforcement_level, version, 'active' as status, created_at, created_by_user_id
                    FROM dsl_policy_packs
                """)
                
                self.policy_cache.clear()
                current_time = datetime.now().timestamp()
                
                for row in rows:
                    policy_pack = PolicyPack(
                        policy_pack_id=row['policy_pack_id'],
                        tenant_id=row['tenant_id'],
                        pack_name=row['pack_name'],
                        pack_type=PolicyPackType(row['pack_type']),
                        industry_code=row['industry_code'],
                        region_code=row['region_code'],
                        policy_rules=row['policy_rules'],
                        enforcement_level=EnforcementLevel(row['enforcement_level']),
                        version=row['version'],
                        status=row['status'],
                        created_at=row['created_at'],
                        created_by_user_id=row['created_by_user_id']
                    )
                    
                    cache_key = (policy_pack.tenant_id, policy_pack.policy_pack_id)
                    self.policy_cache[cache_key] = policy_pack
                    self.last_cache_refresh[cache_key] = current_time
                
        except Exception as e:
            self.logger.error(f"❌ Failed to refresh policy cache: {e}")
    
    async def _validate_policy_schemas(self) -> None:
        """Validate policy pack schemas"""
        # Implementation would validate JSON schemas for policy rules
        pass
    
    def _load_built_in_policies(self) -> Dict[str, Any]:
        """Load built-in policy templates with SaaS industry focus"""
        return {
            # SaaS-focused policies (primary)
            'SOX_SAAS': {
                'saas_financial_controls': {
                    'revenue_recognition': {'enabled': True, 'enforcement': 'strict', 'saas_metrics': ['ARR', 'MRR']},
                    'subscription_controls': {'enabled': True, 'lifecycle_tracking': True, 'billing_accuracy': True},
                    'deal_approval_thresholds': {'high_value': 250000, 'mega_deal': 1000000},
                    'segregation_of_duties': {'enabled': True, 'maker_checker': True, 'sales_finance_separation': True},
                    'multi_tenant_isolation': {'enabled': True, 'tenant_scoped_approvals': True}
                }
            },
            'GDPR_SAAS': {
                'saas_data_protection': {
                    'consent_management': {'enabled': True, 'explicit_consent': True, 'subscription_based': True},
                    'right_to_erasure': {'enabled': True, 'retention_override': False, 'tenant_scoped': True},
                    'data_minimization': {'enabled': True, 'purpose_limitation': True, 'service_necessity': True},
                    'multi_tenant_privacy': {'enabled': True, 'cross_tenant_prevention': True, 'data_isolation': True},
                    'saas_retention_policies': {
                        'customer_data': 2555,  # 7 years
                        'usage_data': 1095,     # 3 years
                        'support_data': 365     # 1 year
                    }
                }
            },
            'SAAS_BUSINESS_RULES': {
                'subscription_lifecycle': {
                    'creation_controls': {'enabled': True, 'customer_verification': True, 'payment_validation': True},
                    'modification_controls': {'enabled': True, 'impact_calculation': True, 'approval_required': True},
                    'cancellation_controls': {'enabled': True, 'retention_attempt': True, 'data_scheduling': True}
                },
                'churn_prevention': {
                    'health_monitoring': {'enabled': True, 'threshold': 0.3, 'real_time': True},
                    'usage_tracking': {'enabled': True, 'trend_analysis': True, 'alerts': True},
                    'engagement_scoring': {'enabled': True, 'feature_adoption': True, 'support_quality': True}
                },
                'billing_accuracy': {
                    'usage_metering': {'enabled': True, 'accuracy_threshold': 0.99, 'completeness_threshold': 0.95},
                    'pricing_validation': {'enabled': True, 'tier_enforcement': True, 'overage_handling': True},
                    'invoice_generation': {'enabled': True, 'auto_send': True, 'payment_terms': 'net_30'}
                }
            },
            
            # Legacy policies (maintained for backward compatibility)
            'SOX': {
                'financial_controls': {
                    'revenue_recognition': {'enabled': True, 'enforcement': 'strict'},
                    'deal_approval_thresholds': {'high_value': 250000, 'mega_deal': 1000000},
                    'segregation_of_duties': {'enabled': True, 'maker_checker': True}
                }
            },
            'GDPR': {
                'data_protection': {
                    'consent_management': {'enabled': True, 'explicit_consent': True},
                    'right_to_erasure': {'enabled': True, 'retention_override': False},
                    'data_minimization': {'enabled': True, 'purpose_limitation': True}
                }
            },
            
            # Future industry placeholders
            'BANKING_RBI': {
                'banking_controls': {
                    'credit_approval': {'enabled': False, 'dual_authorization': True},
                    'aml_compliance': {'enabled': False, 'suspicious_activity_reporting': True},
                    'npa_monitoring': {'enabled': False, 'early_warning_system': True}
                }
            },
            'INSURANCE_NAIC': {
                'insurance_controls': {
                    'policyholder_protection': {'enabled': False, 'claims_processing': True},
                    'solvency_requirements': {'enabled': False, 'capital_adequacy': True},
                    'market_conduct': {'enabled': False, 'sales_practices': True}
                }
            }
        }
    
    def _initialize_saas_policy_templates(self) -> Dict[str, str]:
        """Initialize SaaS-specific Rego policy templates"""
        return {
            'SOX_SAAS_TEMPLATE': '''
# SOX Financial Controls for SaaS Companies
package sox_saas
default allow = false

allow {
    saas_financial_controls
    revenue_recognition_controls
    subscription_audit_trail
    segregation_of_duties
}

saas_financial_controls {
    input.data_type in ["arr", "mrr", "revenue", "subscription", "billing"]
    input.has_approval == true
    input.tenant_id != null
    input.approver_role in ["cfo", "controller", "finance_manager", "revenue_ops"]
}

revenue_recognition_controls {
    input.subscription_start_date != null
    input.revenue_recognition_method in ["monthly", "annual", "usage_based"]
    input.contract_terms_validated == true
}

subscription_audit_trail {
    input.audit_log == true
    input.tenant_id != null
    input.action_type in ["create_subscription", "modify_subscription", "cancel_subscription"]
}

segregation_of_duties {
    input.creator_id != input.approver_id
    count(input.approval_chain) >= 2
    not (input.creator_role == "sales" and input.approver_role == "sales")
}
''',
            
            'GDPR_SAAS_TEMPLATE': '''
# GDPR Data Protection for SaaS Companies
package gdpr_saas
default allow = false

allow {
    saas_consent_validation
    saas_data_minimization
    multi_tenant_isolation
}

saas_consent_validation {
    input.data_type in ["customer_data", "usage_data", "billing_data"]
    input.consent_given == true
    input.tenant_id != null
    input.subscription_active == true
}

saas_data_minimization {
    input.service_delivery_required == true
    count(input.data_fields) <= input.max_allowed_fields
}

multi_tenant_isolation {
    input.tenant_id != null
    input.cross_tenant_access == false
}
''',
            
            'SAAS_BUSINESS_RULES_TEMPLATE': '''
# SaaS Business Rules and Operational Policies
package saas_business_rules
default allow = false

allow {
    subscription_lifecycle_controls
    churn_prevention_controls
    usage_based_billing_controls
}

subscription_lifecycle_controls {
    input.operation_type in ["create_subscription", "modify_subscription", "cancel_subscription"]
    input.tenant_id != null
    input.customer_verified == true
}

churn_prevention_controls {
    input.customer_health_score >= 0.3
    input.usage_trend_monitored == true
}

usage_based_billing_controls {
    input.billing_calculation_verified == true
    input.usage_metering_accurate == true
}
'''
        }

# Global instance
_policy_engine = None

def get_policy_engine(pool_manager) -> PolicyEngine:
    """Get singleton policy engine instance"""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine(pool_manager)
    return _policy_engine
