# Data Residency Enforcement Service
# Tasks 18.3.3-18.3.13: Residency enforcement, region binding, policy engine integration

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
import os

logger = logging.getLogger(__name__)

class Region(Enum):
    """Supported regions"""
    US = "US"
    EU = "EU"
    UK = "UK"
    IN = "IN"
    CA = "CA"
    APAC = "APAC"
    GLOBAL = "GLOBAL"

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    CRITICAL_PERSONAL = "critical_personal"
    FINANCIAL = "financial"
    GOVERNMENT = "government"
    DEFENSE = "defense"

class EnforcementAction(Enum):
    """Residency enforcement actions"""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    LOG_ONLY = "log_only"

class ViolationSeverity(Enum):
    """Violation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ResidencyPolicy:
    """Data residency policy"""
    tenant_id: int
    primary_region: Region
    allowed_regions: Set[Region]
    restricted_regions: Set[Region]
    data_classification_rules: Dict[DataClassification, Dict[str, Any]]
    cross_border_restrictions: Dict[str, Any]
    industry_overlay: str
    compliance_frameworks: List[str]
    failover_allowed: bool = False
    backup_region_restricted: bool = True
    compute_region_restricted: bool = True
    model_inference_region_restricted: bool = True

@dataclass
class ResidencyRule:
    """Residency enforcement rule"""
    rule_id: str
    rule_name: str
    rule_type: str
    source_region: Region
    target_region: Region
    data_classifications: List[DataClassification]
    enforcement_action: EnforcementAction
    violation_severity: ViolationSeverity
    industry_overlay: str
    compliance_frameworks: List[str]
    exception_conditions: Dict[str, Any]

@dataclass
class ResidencyRequest:
    """Data residency request"""
    request_id: str
    tenant_id: int
    user_id: int
    source_region: Region
    target_region: Region
    data_classification: DataClassification
    operation_type: str
    resource_type: str
    resource_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class ResidencyResult:
    """Residency enforcement result"""
    request_id: str
    decision: EnforcementAction
    reason: str
    applicable_rules: List[str]
    violations: List[str]
    evidence_id: str
    compliance_frameworks: List[str]
    override_required: bool = False

class ResidencyPolicyManager:
    """
    Residency Policy Management
    Tasks 18.3.4-18.3.6: Define industry-specific residency overlays
    """
    
    def __init__(self):
        self.tenant_policies: Dict[int, ResidencyPolicy] = {}
        self.global_rules: Dict[str, ResidencyRule] = {}
        self._initialize_industry_overlays()
        self._initialize_global_rules()
    
    def _initialize_industry_overlays(self):
        """Initialize industry-specific residency overlays"""
        
        # Task 18.3.4: SaaS residency overlays
        self.saas_overlay = {
            "default_regions": {Region.US, Region.EU},
            "allowed_transfers": {
                (Region.US, Region.CA): True,
                (Region.EU, Region.UK): True,
                (Region.US, Region.EU): "with_adequacy_decision"
            },
            "restricted_data": [DataClassification.PERSONAL, DataClassification.SENSITIVE],
            "compliance_frameworks": ["GDPR", "CCPA", "SOX"]
        }
        
        # Task 18.3.5: Banking residency overlays (RBI/DPDP strict)
        self.banking_overlay = {
            "default_regions": {Region.IN},
            "allowed_transfers": {
                # RBI requires strict data localization
            },
            "restricted_data": [
                DataClassification.CRITICAL_PERSONAL,
                DataClassification.FINANCIAL,
                DataClassification.SENSITIVE
            ],
            "compliance_frameworks": ["RBI", "DPDP", "AML", "KYC"]
        }
        
        # Task 18.3.6: Insurance residency overlays (HIPAA/NAIC US region)
        self.insurance_overlay = {
            "default_regions": {Region.US},
            "allowed_transfers": {
                (Region.US, Region.CA): "with_privacy_shield"
            },
            "restricted_data": [
                DataClassification.PERSONAL,
                DataClassification.SENSITIVE,
                DataClassification.CONFIDENTIAL
            ],
            "compliance_frameworks": ["HIPAA", "NAIC", "SOX"]
        }
    
    def _initialize_global_rules(self):
        """Initialize global residency rules"""
        
        # GDPR EU data protection rules
        self.global_rules["gdpr_eu_protection"] = ResidencyRule(
            rule_id="gdpr_eu_protection",
            rule_name="GDPR EU Data Protection",
            rule_type="data_movement",
            source_region=Region.EU,
            target_region=Region.US,
            data_classifications=[DataClassification.PERSONAL, DataClassification.SENSITIVE],
            enforcement_action=EnforcementAction.REQUIRE_APPROVAL,
            violation_severity=ViolationSeverity.HIGH,
            industry_overlay="global",
            compliance_frameworks=["GDPR"],
            exception_conditions={"adequacy_decision": True, "user_consent": True}
        )
        
        # RBI India data localization
        self.global_rules["rbi_data_localization"] = ResidencyRule(
            rule_id="rbi_data_localization",
            rule_name="RBI Data Localization",
            rule_type="data_movement",
            source_region=Region.IN,
            target_region=Region.US,
            data_classifications=[DataClassification.CRITICAL_PERSONAL, DataClassification.FINANCIAL],
            enforcement_action=EnforcementAction.DENY,
            violation_severity=ViolationSeverity.CRITICAL,
            industry_overlay="banking",
            compliance_frameworks=["RBI", "DPDP"],
            exception_conditions={}
        )
        
        # HIPAA PHI protection
        self.global_rules["hipaa_phi_protection"] = ResidencyRule(
            rule_id="hipaa_phi_protection",
            rule_name="HIPAA PHI Protection",
            rule_type="data_movement",
            source_region=Region.US,
            target_region=Region.EU,
            data_classifications=[DataClassification.PERSONAL, DataClassification.SENSITIVE],
            enforcement_action=EnforcementAction.REQUIRE_APPROVAL,
            violation_severity=ViolationSeverity.HIGH,
            industry_overlay="insurance",
            compliance_frameworks=["HIPAA", "NAIC"],
            exception_conditions={"business_associate_agreement": True}
        )
    
    async def get_tenant_policy(self, tenant_id: int) -> Optional[ResidencyPolicy]:
        """Get residency policy for tenant"""
        if tenant_id in self.tenant_policies:
            return self.tenant_policies[tenant_id]
        
        # In production, query from database
        return await self._load_tenant_policy_from_db(tenant_id)
    
    async def _load_tenant_policy_from_db(self, tenant_id: int) -> Optional[ResidencyPolicy]:
        """Load tenant policy from database"""
        # Mock implementation - in production, query residency_policy_tenant table
        
        # Default policy for demo
        return ResidencyPolicy(
            tenant_id=tenant_id,
            primary_region=Region.US,
            allowed_regions={Region.US, Region.CA},
            restricted_regions={Region.IN},
            data_classification_rules={
                DataClassification.PERSONAL: {"cross_border_allowed": False},
                DataClassification.SENSITIVE: {"cross_border_allowed": False},
                DataClassification.INTERNAL: {"cross_border_allowed": True}
            },
            cross_border_restrictions={
                "require_encryption": True,
                "require_approval": True,
                "log_all_transfers": True
            },
            industry_overlay="saas",
            compliance_frameworks=["GDPR", "CCPA"],
            failover_allowed=False,
            backup_region_restricted=True,
            compute_region_restricted=True,
            model_inference_region_restricted=True
        )
    
    async def create_tenant_policy(
        self,
        tenant_id: int,
        industry_overlay: str,
        primary_region: Region,
        compliance_frameworks: List[str]
    ) -> ResidencyPolicy:
        """Create residency policy for tenant"""
        
        # Get industry overlay configuration
        overlay_config = getattr(self, f"{industry_overlay}_overlay", self.saas_overlay)
        
        policy = ResidencyPolicy(
            tenant_id=tenant_id,
            primary_region=primary_region,
            allowed_regions=overlay_config["default_regions"],
            restricted_regions=set(),
            data_classification_rules={
                dc: {"cross_border_allowed": dc not in overlay_config["restricted_data"]}
                for dc in DataClassification
            },
            cross_border_restrictions={
                "require_encryption": True,
                "require_approval": True,
                "log_all_transfers": True
            },
            industry_overlay=industry_overlay,
            compliance_frameworks=compliance_frameworks,
            failover_allowed=industry_overlay != "banking",  # Banking is strict
            backup_region_restricted=True,
            compute_region_restricted=industry_overlay in ["banking", "insurance"],
            model_inference_region_restricted=industry_overlay in ["banking", "insurance"]
        )
        
        self.tenant_policies[tenant_id] = policy
        
        # In production, save to database
        await self._save_tenant_policy_to_db(policy)
        
        return policy
    
    async def _save_tenant_policy_to_db(self, policy: ResidencyPolicy):
        """Save tenant policy to database"""
        # In production, insert into residency_policy_tenant table
        logger.info(f"💾 Saved residency policy for tenant {policy.tenant_id}")

class ResidencyEnforcementEngine:
    """
    Residency Enforcement Engine
    Tasks 18.3.7-18.3.13: ABAC attributes, region binding, policy enforcement
    """
    
    def __init__(self):
        self.policy_manager = ResidencyPolicyManager()
        self.violation_log: List[Dict[str, Any]] = []
    
    async def enforce_residency_policy(
        self,
        request: ResidencyRequest
    ) -> ResidencyResult:
        """
        Enforce data residency policy for request
        Task 18.3.13: Configure policy engine for residency
        """
        try:
            # Get tenant policy
            tenant_policy = await self.policy_manager.get_tenant_policy(request.tenant_id)
            
            if not tenant_policy:
                return ResidencyResult(
                    request_id=request.request_id,
                    decision=EnforcementAction.DENY,
                    reason="No residency policy found for tenant",
                    applicable_rules=[],
                    violations=["missing_tenant_policy"],
                    evidence_id=str(uuid.uuid4()),
                    compliance_frameworks=[],
                    override_required=True
                )
            
            # Check policy violations
            violations = []
            applicable_rules = []
            
            # Check 1: Target region allowed
            if request.target_region not in tenant_policy.allowed_regions:
                violations.append(f"Target region {request.target_region.value} not in allowed regions")
            
            # Check 2: Source region not restricted
            if request.target_region in tenant_policy.restricted_regions:
                violations.append(f"Target region {request.target_region.value} is restricted")
            
            # Check 3: Data classification rules
            data_rules = tenant_policy.data_classification_rules.get(request.data_classification, {})
            if not data_rules.get("cross_border_allowed", True) and request.source_region != request.target_region:
                violations.append(f"Cross-border transfer not allowed for {request.data_classification.value} data")
            
            # Check 4: Apply global rules
            global_violations, global_rules = await self._check_global_rules(request, tenant_policy)
            violations.extend(global_violations)
            applicable_rules.extend(global_rules)
            
            # Check 5: Industry-specific restrictions
            industry_violations = await self._check_industry_restrictions(request, tenant_policy)
            violations.extend(industry_violations)
            
            # Determine enforcement decision
            if violations:
                # Check if approval can override
                if tenant_policy.cross_border_restrictions.get("require_approval", False):
                    decision = EnforcementAction.REQUIRE_APPROVAL
                    override_required = True
                else:
                    decision = EnforcementAction.DENY
                    override_required = False
            else:
                decision = EnforcementAction.ALLOW
                override_required = False
            
            # Log violation if any
            if violations:
                await self._log_residency_violation(request, violations, tenant_policy)
            
            result = ResidencyResult(
                request_id=request.request_id,
                decision=decision,
                reason="; ".join(violations) if violations else "Residency policy allows operation",
                applicable_rules=applicable_rules,
                violations=violations,
                evidence_id=str(uuid.uuid4()),
                compliance_frameworks=tenant_policy.compliance_frameworks,
                override_required=override_required
            )
            
            # Log evidence
            await self._log_residency_evidence(request, result, tenant_policy)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Residency enforcement error: {e}")
            return ResidencyResult(
                request_id=request.request_id,
                decision=EnforcementAction.DENY,
                reason=f"Residency enforcement error: {str(e)}",
                applicable_rules=[],
                violations=[f"system_error: {str(e)}"],
                evidence_id=str(uuid.uuid4()),
                compliance_frameworks=[],
                override_required=True
            )
    
    async def _check_global_rules(
        self,
        request: ResidencyRequest,
        tenant_policy: ResidencyPolicy
    ) -> Tuple[List[str], List[str]]:
        """Check global residency rules"""
        violations = []
        applicable_rules = []
        
        for rule_id, rule in self.policy_manager.global_rules.items():
            # Check if rule applies
            if (rule.source_region == request.source_region and
                rule.target_region == request.target_region and
                request.data_classification in rule.data_classifications and
                (rule.industry_overlay == tenant_policy.industry_overlay or rule.industry_overlay == "global")):
                
                applicable_rules.append(rule_id)
                
                if rule.enforcement_action == EnforcementAction.DENY:
                    violations.append(f"Global rule {rule.rule_name} denies operation")
                elif rule.enforcement_action == EnforcementAction.REQUIRE_APPROVAL:
                    violations.append(f"Global rule {rule.rule_name} requires approval")
        
        return violations, applicable_rules
    
    async def _check_industry_restrictions(
        self,
        request: ResidencyRequest,
        tenant_policy: ResidencyPolicy
    ) -> List[str]:
        """Check industry-specific restrictions"""
        violations = []
        
        if tenant_policy.industry_overlay == "banking":
            # RBI strict data localization
            if (request.source_region == Region.IN and 
                request.target_region != Region.IN and
                request.data_classification in [DataClassification.CRITICAL_PERSONAL, DataClassification.FINANCIAL]):
                violations.append("RBI data localization: Critical personal/financial data cannot leave India")
        
        elif tenant_policy.industry_overlay == "insurance":
            # HIPAA PHI restrictions
            if (request.data_classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE] and
                request.source_region != request.target_region):
                violations.append("HIPAA PHI protection: Cross-border transfer requires business associate agreement")
        
        return violations
    
    async def _log_residency_violation(
        self,
        request: ResidencyRequest,
        violations: List[str],
        tenant_policy: ResidencyPolicy
    ):
        """Log residency policy violation"""
        violation_record = {
            "violation_id": str(uuid.uuid4()),
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "source_region": request.source_region.value,
            "target_region": request.target_region.value,
            "data_classification": request.data_classification.value,
            "violations": violations,
            "industry_overlay": tenant_policy.industry_overlay,
            "compliance_frameworks": tenant_policy.compliance_frameworks,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.violation_log.append(violation_record)
        
        logger.warning(f"🚨 Residency violation: {json.dumps(violation_record, indent=2)}")
    
    async def _log_residency_evidence(
        self,
        request: ResidencyRequest,
        result: ResidencyResult,
        tenant_policy: ResidencyPolicy
    ):
        """Log residency enforcement evidence"""
        evidence = {
            "evidence_id": result.evidence_id,
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "decision": result.decision.value,
            "reason": result.reason,
            "source_region": request.source_region.value,
            "target_region": request.target_region.value,
            "data_classification": request.data_classification.value,
            "operation_type": request.operation_type,
            "applicable_rules": result.applicable_rules,
            "violations": result.violations,
            "compliance_frameworks": result.compliance_frameworks,
            "industry_overlay": tenant_policy.industry_overlay,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # In production, store in evidence service
        logger.info(f"📋 Residency evidence: {json.dumps(evidence, indent=2)}")

class InfrastructureRegionManager:
    """
    Infrastructure Region Management
    Tasks 18.3.8-18.3.12: Region binding for storage, DB, compute, KeyVault
    """
    
    def __init__(self):
        self.region_mappings = self._initialize_region_mappings()
    
    def _initialize_region_mappings(self):
        """Initialize Azure region mappings"""
        return {
            Region.US: {
                "azure_regions": ["eastus", "westus2", "centralus"],
                "storage_accounts": ["revaiprouseast", "revaiprouswest"],
                "keyvault_regions": ["eastus", "westus2"],
                "aks_regions": ["eastus", "westus2"],
                "postgres_regions": ["eastus", "westus2"]
            },
            Region.EU: {
                "azure_regions": ["westeurope", "northeurope"],
                "storage_accounts": ["revaiproeuwest", "revaiproeuNorth"],
                "keyvault_regions": ["westeurope", "northeurope"],
                "aks_regions": ["westeurope", "northeurope"],
                "postgres_regions": ["westeurope", "northeurope"]
            },
            Region.IN: {
                "azure_regions": ["centralindia", "southindia"],
                "storage_accounts": ["revaiproincentral", "revaiproinsouth"],
                "keyvault_regions": ["centralindia", "southindia"],
                "aks_regions": ["centralindia", "southindia"],
                "postgres_regions": ["centralindia", "southindia"]
            }
        }
    
    async def bind_tenant_to_region(
        self,
        tenant_id: int,
        region: Region,
        services: List[str] = None
    ) -> Dict[str, Any]:
        """
        Bind tenant infrastructure to specific region
        Tasks 18.3.8-18.3.12: Infrastructure region binding
        """
        if services is None:
            services = ["storage", "database", "compute", "keyvault"]
        
        region_config = self.region_mappings.get(region, {})
        
        binding_result = {
            "tenant_id": tenant_id,
            "region": region.value,
            "services_bound": {},
            "binding_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Task 18.3.9: Enforce storage residency
        if "storage" in services:
            storage_config = await self._bind_storage_residency(tenant_id, region_config)
            binding_result["services_bound"]["storage"] = storage_config
        
        # Task 18.3.10: Enforce DB residency
        if "database" in services:
            db_config = await self._bind_db_residency(tenant_id, region_config)
            binding_result["services_bound"]["database"] = db_config
        
        # Task 18.3.11: Enforce KeyVault residency
        if "keyvault" in services:
            kv_config = await self._bind_keyvault_residency(tenant_id, region_config)
            binding_result["services_bound"]["keyvault"] = kv_config
        
        # Task 18.3.12: Enforce compute residency
        if "compute" in services:
            compute_config = await self._bind_compute_residency(tenant_id, region_config)
            binding_result["services_bound"]["compute"] = compute_config
        
        logger.info(f"🌍 Infrastructure region binding: {json.dumps(binding_result, indent=2)}")
        
        return binding_result
    
    async def _bind_storage_residency(self, tenant_id: int, region_config: Dict[str, Any]) -> Dict[str, Any]:
        """Bind storage to region"""
        return {
            "service": "azure_storage",
            "storage_accounts": region_config.get("storage_accounts", []),
            "azure_regions": region_config.get("azure_regions", []),
            "tenant_container": f"tenant-{tenant_id}",
            "encryption_enabled": True,
            "cross_region_replication": False
        }
    
    async def _bind_db_residency(self, tenant_id: int, region_config: Dict[str, Any]) -> Dict[str, Any]:
        """Bind database to region"""
        return {
            "service": "postgresql",
            "regions": region_config.get("postgres_regions", []),
            "tenant_schema": f"tenant_{tenant_id}",
            "rls_enabled": True,
            "backup_region_restricted": True
        }
    
    async def _bind_keyvault_residency(self, tenant_id: int, region_config: Dict[str, Any]) -> Dict[str, Any]:
        """Bind KeyVault to region"""
        return {
            "service": "azure_keyvault",
            "regions": region_config.get("keyvault_regions", []),
            "tenant_vault": f"revai-tenant-{tenant_id}",
            "key_rotation_enabled": True,
            "cross_region_backup": False
        }
    
    async def _bind_compute_residency(self, tenant_id: int, region_config: Dict[str, Any]) -> Dict[str, Any]:
        """Bind compute to region"""
        return {
            "service": "azure_aks",
            "regions": region_config.get("aks_regions", []),
            "tenant_namespace": f"tenant-{tenant_id}",
            "node_pools": region_config.get("aks_regions", []),
            "network_policies_enabled": True
        }

class ResidencyComplianceManager:
    """
    Residency Compliance Management
    Coordinates all residency enforcement components
    """
    
    def __init__(self):
        self.enforcement_engine = ResidencyEnforcementEngine()
        self.infra_manager = InfrastructureRegionManager()
    
    async def setup_tenant_residency(
        self,
        tenant_id: int,
        industry_overlay: str,
        primary_region: Region,
        compliance_frameworks: List[str]
    ) -> Dict[str, Any]:
        """Setup complete residency compliance for tenant"""
        
        # Create residency policy
        policy = await self.enforcement_engine.policy_manager.create_tenant_policy(
            tenant_id, industry_overlay, primary_region, compliance_frameworks
        )
        
        # Bind infrastructure to region
        infra_binding = await self.infra_manager.bind_tenant_to_region(
            tenant_id, primary_region
        )
        
        setup_result = {
            "tenant_id": tenant_id,
            "industry_overlay": industry_overlay,
            "primary_region": primary_region.value,
            "compliance_frameworks": compliance_frameworks,
            "policy_created": True,
            "infrastructure_bound": True,
            "setup_timestamp": datetime.now(timezone.utc).isoformat(),
            "policy_details": asdict(policy),
            "infrastructure_details": infra_binding
        }
        
        logger.info(f"🏗️ Tenant residency setup complete: {json.dumps(setup_result, indent=2)}")
        
        return setup_result
    
    async def validate_cross_border_request(
        self,
        tenant_id: int,
        user_id: int,
        source_region: str,
        target_region: str,
        data_classification: str,
        operation_type: str
    ) -> Dict[str, Any]:
        """Validate cross-border data request"""
        
        request = ResidencyRequest(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            source_region=Region(source_region),
            target_region=Region(target_region),
            data_classification=DataClassification(data_classification),
            operation_type=operation_type,
            resource_type="data_transfer"
        )
        
        result = await self.enforcement_engine.enforce_residency_policy(request)
        
        return {
            "request_id": request.request_id,
            "decision": result.decision.value,
            "reason": result.reason,
            "violations": result.violations,
            "compliance_frameworks": result.compliance_frameworks,
            "override_required": result.override_required,
            "evidence_id": result.evidence_id
        }
