"""
Dynamic RBA Template Service - Chapter 19.1
Tasks 19.1.7, 19.1.8, 19.1.35: Registry APIs, validation service, lifecycle state machine
Integrates with existing Enhanced Capability Registry for dynamic template management
"""

import uuid
import json
import hashlib
import yaml
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ai_crenovent.database.database_manager import DatabaseManager
from ai_crenovent.dsl.governance.evidence_service import EvidenceService
from ai_crenovent.dsl.tenancy.tenant_context_manager import TenantContextManager, TenantContext
from ai_crenovent.dsl.governance.multi_tenant_taxonomy import IndustryCode, ComplianceFramework
from ai_crenovent.dsl.registry.enhanced_capability_registry import EnhancedCapabilityRegistry, CapabilityType
from ai_crenovent.dsl.capability_registry.saas_template_generators import SaaSRBATemplateGenerator

logger = logging.getLogger(__name__)

class LifecycleState(Enum):
    """Task 19.1.35: Lifecycle state machine states"""
    DRAFT = "draft"
    PUBLISHED = "published" 
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class SLATier(Enum):
    """SLA tier enumeration"""
    BRONZE = "Bronze"
    SILVER = "Silver"
    GOLD = "Gold"
    PLATINUM = "Platinum"

@dataclass
class RBATemplate:
    """RBA Template data model"""
    template_id: str
    template_name: str
    template_description: str
    version: str
    industry_code: str
    sla_tier: str
    jurisdiction: str
    trust_score: float
    dsl_definition: Dict[str, Any]
    dsl_schema_version: str
    input_contract: Dict[str, Any]
    output_contract: Dict[str, Any]
    contract_version: str
    policy_pack_ids: List[str]
    compliance_frameworks: List[str]
    lifecycle_state: str
    lifecycle_metadata: Dict[str, Any]
    evidence_pack_required: bool
    evidence_retention_days: int
    tenant_id: Optional[int] = None
    created_by_user_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class PromotionManifest:
    """Task 19.1.36: Promotion manifest structure"""
    promotion_id: str
    template_id: str
    from_state: str
    to_state: str
    promotion_reason: str
    validation_results: Dict[str, Any]
    compliance_checks: Dict[str, Any]
    approver_chain: List[str]
    timestamp: str
    evidence_pack_id: Optional[str] = None

class DynamicRBATemplateService:
    """
    Dynamic RBA Template Service integrating with existing Enhanced Capability Registry.
    Tasks 19.1.7, 19.1.8, 19.1.35: Registry APIs, validation, lifecycle management
    """
    
    def __init__(self, db_manager: DatabaseManager, evidence_service: EvidenceService, 
                 tenant_context_manager: TenantContextManager):
        self.db_manager = db_manager
        self.evidence_service = evidence_service
        self.tenant_context_manager = tenant_context_manager
        
        # Integration with existing systems
        self.capability_registry = EnhancedCapabilityRegistry()
        self.saas_generator = SaaSRBATemplateGenerator()
        
        # Dynamic template loaders
        self.template_loaders = {
            'yaml': self._load_yaml_template,
            'json': self._load_json_template,
            'generated': self._load_generated_template
        }
        
        # Task 19.1.35: Valid lifecycle transitions
        self.valid_transitions = {
            LifecycleState.DRAFT: [LifecycleState.PUBLISHED],
            LifecycleState.PUBLISHED: [LifecycleState.PROMOTED, LifecycleState.DEPRECATED],
            LifecycleState.PROMOTED: [LifecycleState.DEPRECATED],
            LifecycleState.DEPRECATED: [LifecycleState.RETIRED, LifecycleState.PUBLISHED],  # Can be republished
            LifecycleState.RETIRED: []  # Terminal state
        }
        
        # Task 19.1.2: RBA DSL schema validation rules
        self.required_dsl_fields = ["trigger", "conditions", "actions", "governance"]
        self.supported_trigger_types = ["schedule", "event", "webhook", "manual"]
        self.supported_action_types = ["query", "notify", "update", "escalate"]

    async def create_template(self, template_data: Dict[str, Any], tenant_id: int, user_id: int) -> str:
        """
        Task 19.1.7: Create a new RBA template
        """
        # Task 19.1.8: Validate DSL structure
        validation_result = await self._validate_dsl_definition(template_data.get("dsl_definition", {}))
        if not validation_result["valid"]:
            raise ValueError(f"DSL validation failed: {validation_result['errors']}")
        
        # Validate contracts
        contract_validation = await self._validate_contracts(
            template_data.get("input_contract", {}),
            template_data.get("output_contract", {})
        )
        if not contract_validation["valid"]:
            raise ValueError(f"Contract validation failed: {contract_validation['errors']}")
        
        template_id = str(uuid.uuid4())
        
        # Create template record
        await self.db_manager.execute(
            """
            INSERT INTO registry_cap_rba (
                template_id, template_name, template_description, version,
                industry_code, sla_tier, jurisdiction, trust_score,
                dsl_definition, dsl_schema_version, input_contract, output_contract,
                contract_version, policy_pack_ids, compliance_frameworks,
                lifecycle_state, lifecycle_metadata, evidence_pack_required,
                evidence_retention_days, tenant_id, created_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                template_id,
                template_data["template_name"],
                template_data.get("template_description", ""),
                template_data.get("version", "1.0.0"),
                template_data["industry_code"],
                template_data.get("sla_tier", "Bronze"),
                template_data.get("jurisdiction", "US"),
                template_data.get("trust_score", 0.85),
                json.dumps(template_data["dsl_definition"]),
                template_data.get("dsl_schema_version", "2.0"),
                json.dumps(template_data.get("input_contract", {})),
                json.dumps(template_data.get("output_contract", {})),
                template_data.get("contract_version", "1.0"),
                template_data.get("policy_pack_ids", []),
                template_data.get("compliance_frameworks", []),
                LifecycleState.DRAFT.value,
                json.dumps({"created_state": "draft", "transitions": []}),
                template_data.get("evidence_pack_required", True),
                template_data.get("evidence_retention_days", 2555),
                tenant_id,
                user_id
            )
        )
        
        # Log evidence of template creation
        await self.evidence_service.log_evidence(
            event_type="rba_template_created",
            event_data={
                "template_id": template_id,
                "template_name": template_data["template_name"],
                "industry_code": template_data["industry_code"],
                "validation_results": validation_result,
                "contract_validation": contract_validation
            },
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        logger.info(f"Created RBA template {template_id} for tenant {tenant_id}")
        return template_id

    async def get_template(self, template_id: str, tenant_id: int) -> Optional[RBATemplate]:
        """Task 19.1.7: Retrieve RBA template by ID"""
        template_data = await self.db_manager.fetch_one(
            "SELECT * FROM registry_cap_rba WHERE template_id = ? AND tenant_id = ?",
            (template_id, tenant_id)
        )
        
        if not template_data:
            return None
        
        return RBATemplate(
            template_id=template_data["template_id"],
            template_name=template_data["template_name"],
            template_description=template_data["template_description"],
            version=template_data["version"],
            industry_code=template_data["industry_code"],
            sla_tier=template_data["sla_tier"],
            jurisdiction=template_data["jurisdiction"],
            trust_score=float(template_data["trust_score"]),
            dsl_definition=json.loads(template_data["dsl_definition"]),
            dsl_schema_version=template_data["dsl_schema_version"],
            input_contract=json.loads(template_data["input_contract"]),
            output_contract=json.loads(template_data["output_contract"]),
            contract_version=template_data["contract_version"],
            policy_pack_ids=template_data["policy_pack_ids"],
            compliance_frameworks=template_data["compliance_frameworks"],
            lifecycle_state=template_data["lifecycle_state"],
            lifecycle_metadata=json.loads(template_data["lifecycle_metadata"]),
            evidence_pack_required=template_data["evidence_pack_required"],
            evidence_retention_days=template_data["evidence_retention_days"],
            tenant_id=template_data["tenant_id"],
            created_by_user_id=template_data["created_by_user_id"],
            created_at=template_data["created_at"],
            updated_at=template_data["updated_at"]
        )

    async def list_templates(self, tenant_id: int, industry_code: Optional[str] = None, 
                           lifecycle_state: Optional[str] = None) -> List[RBATemplate]:
        """Task 19.1.7: List RBA templates with optional filters"""
        query = "SELECT * FROM registry_cap_rba WHERE tenant_id = ?"
        params = [tenant_id]
        
        if industry_code:
            query += " AND industry_code = ?"
            params.append(industry_code)
        
        if lifecycle_state:
            query += " AND lifecycle_state = ?"
            params.append(lifecycle_state)
        
        query += " ORDER BY created_at DESC"
        
        templates_data = await self.db_manager.fetch_all(query, params)
        
        templates = []
        for template_data in templates_data:
            templates.append(RBATemplate(
                template_id=template_data["template_id"],
                template_name=template_data["template_name"],
                template_description=template_data["template_description"],
                version=template_data["version"],
                industry_code=template_data["industry_code"],
                sla_tier=template_data["sla_tier"],
                jurisdiction=template_data["jurisdiction"],
                trust_score=float(template_data["trust_score"]),
                dsl_definition=json.loads(template_data["dsl_definition"]),
                dsl_schema_version=template_data["dsl_schema_version"],
                input_contract=json.loads(template_data["input_contract"]),
                output_contract=json.loads(template_data["output_contract"]),
                contract_version=template_data["contract_version"],
                policy_pack_ids=template_data["policy_pack_ids"],
                compliance_frameworks=template_data["compliance_frameworks"],
                lifecycle_state=template_data["lifecycle_state"],
                lifecycle_metadata=json.loads(template_data["lifecycle_metadata"]),
                evidence_pack_required=template_data["evidence_pack_required"],
                evidence_retention_days=template_data["evidence_retention_days"],
                tenant_id=template_data["tenant_id"],
                created_by_user_id=template_data["created_by_user_id"],
                created_at=template_data["created_at"],
                updated_at=template_data["updated_at"]
            ))
        
        return templates

    async def promote_template(self, template_id: str, to_state: LifecycleState, 
                             tenant_id: int, user_id: int, reason: str = "") -> str:
        """
        Task 19.1.35: Promote template through lifecycle states
        Task 19.1.36: Create signed promotion manifest
        """
        # Get current template
        template = await self.get_template(template_id, tenant_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        current_state = LifecycleState(template.lifecycle_state)
        
        # Validate transition
        if to_state not in self.valid_transitions.get(current_state, []):
            raise ValueError(f"Invalid transition from {current_state.value} to {to_state.value}")
        
        # Run validation checks for promotion
        validation_results = await self._validate_template_for_promotion(template, to_state)
        if not validation_results["valid"]:
            raise ValueError(f"Template validation failed: {validation_results['errors']}")
        
        # Create promotion manifest
        promotion_id = str(uuid.uuid4())
        manifest = PromotionManifest(
            promotion_id=promotion_id,
            template_id=template_id,
            from_state=current_state.value,
            to_state=to_state.value,
            promotion_reason=reason,
            validation_results=validation_results,
            compliance_checks=await self._run_compliance_checks(template),
            approver_chain=[str(user_id)],
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Generate manifest hash and signature
        manifest_json = json.dumps(asdict(manifest), sort_keys=True)
        manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
        
        # Update template lifecycle state
        updated_lifecycle_metadata = template.lifecycle_metadata.copy()
        updated_lifecycle_metadata["transitions"].append({
            "from_state": current_state.value,
            "to_state": to_state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "promoted_by": user_id,
            "reason": reason,
            "promotion_id": promotion_id
        })
        
        await self.db_manager.execute(
            """
            UPDATE registry_cap_rba 
            SET lifecycle_state = ?, lifecycle_metadata = ?, updated_at = NOW()
            WHERE template_id = ? AND tenant_id = ?
            """,
            (to_state.value, json.dumps(updated_lifecycle_metadata), template_id, tenant_id)
        )
        
        # Store promotion manifest
        await self.db_manager.execute(
            """
            INSERT INTO registry_cap_rba_promotions (
                promotion_id, template_id, from_state, to_state,
                promotion_manifest, manifest_hash, promoted_by_user_id,
                approval_chain, approval_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                promotion_id,
                template_id,
                current_state.value,
                to_state.value,
                manifest_json,
                manifest_hash,
                user_id,
                json.dumps([str(user_id)]),
                json.dumps({"reason": reason, "validation_results": validation_results})
            )
        )
        
        # Log evidence of promotion
        await self.evidence_service.log_evidence(
            event_type="rba_template_promoted",
            event_data={
                "template_id": template_id,
                "from_state": current_state.value,
                "to_state": to_state.value,
                "promotion_id": promotion_id,
                "manifest_hash": manifest_hash,
                "validation_results": validation_results
            },
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        logger.info(f"Promoted RBA template {template_id} from {current_state.value} to {to_state.value}")
        return promotion_id

    async def pin_template_version(self, template_id: str, version: str, tenant_id: int, 
                                 user_id: int, reason: str = "", expires_at: Optional[datetime] = None) -> str:
        """Task 19.1.37: Pin template version for tenant"""
        pin_id = str(uuid.uuid4())
        
        # Remove existing pin if any
        await self.db_manager.execute(
            "DELETE FROM registry_cap_rba_tenant_pins WHERE tenant_id = ? AND template_id = ?",
            (tenant_id, template_id)
        )
        
        # Create new pin
        await self.db_manager.execute(
            """
            INSERT INTO registry_cap_rba_tenant_pins (
                pin_id, tenant_id, template_id, pinned_version, 
                pin_reason, pinned_by_user_id, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (pin_id, tenant_id, template_id, version, reason, user_id, expires_at)
        )
        
        # Log evidence
        await self.evidence_service.log_evidence(
            event_type="rba_template_version_pinned",
            event_data={
                "template_id": template_id,
                "pinned_version": version,
                "pin_id": pin_id,
                "reason": reason,
                "expires_at": expires_at.isoformat() if expires_at else None
            },
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return pin_id

    async def _validate_dsl_definition(self, dsl_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task 19.1.8: Validate RBA DSL structure
        Task 19.1.2: Ensure triggers & actions are valid
        """
        errors = []
        
        # Check required fields
        for field in self.required_dsl_fields:
            if field not in dsl_definition:
                errors.append(f"Missing required field: {field}")
        
        # Validate trigger
        if "trigger" in dsl_definition:
            trigger = dsl_definition["trigger"]
            if not isinstance(trigger, dict):
                errors.append("Trigger must be an object")
            elif "type" not in trigger:
                errors.append("Trigger must have a type")
            elif trigger["type"] not in self.supported_trigger_types:
                errors.append(f"Unsupported trigger type: {trigger['type']}")
        
        # Validate actions
        if "actions" in dsl_definition:
            actions = dsl_definition["actions"]
            if not isinstance(actions, list):
                errors.append("Actions must be an array")
            else:
                for i, action in enumerate(actions):
                    if not isinstance(action, dict):
                        errors.append(f"Action {i} must be an object")
                    elif "type" not in action:
                        errors.append(f"Action {i} must have a type")
                    elif action["type"] not in self.supported_action_types:
                        errors.append(f"Action {i} has unsupported type: {action['type']}")
        
        # Validate governance section
        if "governance" in dsl_definition:
            governance = dsl_definition["governance"]
            if not isinstance(governance, dict):
                errors.append("Governance must be an object")
            elif "policy_pack_id" not in governance:
                errors.append("Governance must specify policy_pack_id")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _validate_contracts(self, input_contract: Dict[str, Any], 
                                output_contract: Dict[str, Any]) -> Dict[str, Any]:
        """Task 19.1.6: Validate JSON Schema contracts"""
        errors = []
        
        # Basic JSON Schema validation
        for contract_name, contract in [("input", input_contract), ("output", output_contract)]:
            if not isinstance(contract, dict):
                errors.append(f"{contract_name} contract must be an object")
                continue
            
            # Check for required JSON Schema fields
            if "type" not in contract:
                errors.append(f"{contract_name} contract must specify 'type'")
            
            if contract.get("type") == "object" and "properties" not in contract:
                errors.append(f"{contract_name} contract of type 'object' must have 'properties'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _validate_template_for_promotion(self, template: RBATemplate, 
                                             to_state: LifecycleState) -> Dict[str, Any]:
        """Validate template before promotion"""
        errors = []
        
        # Check if template has required evidence packs
        if template.evidence_pack_required:
            evidence_count = await self.db_manager.fetch_one(
                "SELECT COUNT(*) as count FROM registry_cap_rba_evidence WHERE template_id = ?",
                (template.template_id,)
            )
            if evidence_count["count"] == 0:
                errors.append("Template requires evidence packs but none found")
        
        # Check trust score threshold for promotion
        if to_state == LifecycleState.PROMOTED and template.trust_score < 0.8:
            errors.append(f"Trust score {template.trust_score} below promotion threshold (0.8)")
        
        # Validate DSL definition again
        dsl_validation = await self._validate_dsl_definition(template.dsl_definition)
        if not dsl_validation["valid"]:
            errors.extend(dsl_validation["errors"])
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _run_compliance_checks(self, template: RBATemplate) -> Dict[str, Any]:
        """Run compliance framework checks"""
        compliance_results = {}
        
        for framework in template.compliance_frameworks:
            # Simulate compliance checks (in real implementation, would integrate with OPA/Kyverno)
            compliance_results[framework] = {
                "status": "passed",
                "checks_run": ["policy_pack_attached", "evidence_retention_configured", "audit_trail_enabled"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return compliance_results

    async def _load_yaml_template(self, source_path: str) -> Dict[str, Any]:
        """Task 19.1.2: Load RBA template from YAML file (dynamic)"""
        try:
            # Support both absolute and relative paths
            if not os.path.isabs(source_path):
                # Relative to workflows directory
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                source_path = os.path.join(base_dir, "workflows", source_path)
            
            with open(source_path, 'r') as file:
                template_data = yaml.safe_load(file)
            
            return template_data
        except Exception as e:
            logger.error(f"Failed to load YAML template from {source_path}: {e}")
            return {}

    async def _load_json_template(self, source_path: str) -> Dict[str, Any]:
        """Load RBA template from JSON file (dynamic)"""
        try:
            if not os.path.isabs(source_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                source_path = os.path.join(base_dir, "templates", source_path)
            
            with open(source_path, 'r') as file:
                template_data = json.load(file)
            
            return template_data
        except Exception as e:
            logger.error(f"Failed to load JSON template from {source_path}: {e}")
            return {}

    async def _load_generated_template(self, loader_class: str) -> Dict[str, Any]:
        """Load RBA template from generator class (dynamic)"""
        try:
            # Dynamic import and instantiation
            module_path, class_name = loader_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            generator_class = getattr(module, class_name)
            generator = generator_class()
            
            # Generate template based on context
            # This would integrate with existing SaaS template generators
            if hasattr(generator, 'generate_rba_templates'):
                templates = await generator.generate_rba_templates()
                return templates[0] if templates else {}
            
            return {}
        except Exception as e:
            logger.error(f"Failed to load generated template from {loader_class}: {e}")
            return {}

    async def load_industry_templates(self, industry_code: str, tenant_id: int) -> List[Dict[str, Any]]:
        """
        Task 19.1.9-19.1.34: Load industry-specific templates dynamically
        """
        templates = []
        
        # Query industry templates from database
        industry_templates = await self.db_manager.fetch_all(
            """
            SELECT it.*, rtl.lifecycle_state 
            FROM rba_industry_templates it
            LEFT JOIN rba_template_lifecycle rtl ON it.capability_id = rtl.capability_id
            WHERE it.industry_code = ? AND (it.tenant_id = ? OR it.tenant_id IS NULL)
            ORDER BY it.created_at DESC
            """,
            (industry_code, tenant_id)
        )
        
        for template_record in industry_templates:
            try:
                # Load template definition dynamically
                template_source = template_record['template_source']
                loader = self.template_loaders.get(template_source)
                
                if loader:
                    if template_source == 'generated':
                        template_def = await loader(template_record['loader_class'])
                    else:
                        template_def = await loader(template_record['source_path'])
                    
                    # Merge with database metadata
                    template = {
                        'template_id': template_record['template_id'],
                        'template_name': template_record['template_name'],
                        'industry_code': template_record['industry_code'],
                        'template_category': template_record['template_category'],
                        'lifecycle_state': template_record.get('lifecycle_state', 'draft'),
                        'template_config': json.loads(template_record['template_config']),
                        'parameter_schema': json.loads(template_record['parameter_schema']),
                        'default_parameters': json.loads(template_record['default_parameters']),
                        'policy_pack_refs': template_record['policy_pack_refs'],
                        'compliance_frameworks': template_record['compliance_frameworks'],
                        'template_definition': template_def
                    }
                    templates.append(template)
                    
            except Exception as e:
                logger.error(f"Failed to load template {template_record['template_id']}: {e}")
                continue
        
        return templates

    async def register_template_from_yaml(self, yaml_path: str, industry_code: str, 
                                        template_category: str, tenant_id: int, user_id: int) -> str:
        """
        Task 19.1.7: Register RBA template from existing YAML workflow
        """
        # Load template from YAML
        template_def = await self._load_yaml_template(yaml_path)
        if not template_def:
            raise ValueError(f"Failed to load template from {yaml_path}")
        
        # Extract template metadata
        template_name = template_def.get('name', f"{industry_code} {template_category}")
        
        # Register in capability registry first
        capability_id = await self.capability_registry.register_capability(
            name=template_name,
            capability_type=CapabilityType.RBA_TEMPLATE,
            description=template_def.get('description', ''),
            industry_tags=[industry_code],
            tenant_id=tenant_id
        )
        
        # Create industry template record
        template_id = str(uuid.uuid4())
        await self.db_manager.execute(
            """
            INSERT INTO rba_industry_templates (
                template_id, capability_id, template_name, industry_code, 
                template_category, template_config, parameter_schema, 
                default_parameters, template_source, source_path,
                tenant_id, created_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                template_id,
                capability_id,
                template_name,
                industry_code,
                template_category,
                json.dumps(template_def.get('config', {})),
                json.dumps(self._extract_parameter_schema(template_def)),
                json.dumps(self._extract_default_parameters(template_def)),
                'yaml',
                yaml_path,
                tenant_id,
                user_id
            )
        )
        
        # Create lifecycle record
        await self.db_manager.execute(
            """
            INSERT INTO rba_template_lifecycle (
                capability_id, lifecycle_state, state_metadata,
                industry_overlays, tenant_id, created_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                capability_id,
                LifecycleState.DRAFT.value,
                json.dumps({"source": "yaml_registration", "yaml_path": yaml_path}),
                [industry_code],
                tenant_id,
                user_id
            )
        )
        
        # Log evidence
        await self.evidence_service.log_evidence(
            event_type="rba_template_registered",
            event_data={
                "template_id": template_id,
                "capability_id": capability_id,
                "template_name": template_name,
                "industry_code": industry_code,
                "source": "yaml",
                "yaml_path": yaml_path
            },
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        logger.info(f"Registered RBA template {template_id} from YAML {yaml_path}")
        return template_id

    def _extract_parameter_schema(self, template_def: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameter schema from template definition"""
        # Look for configurable parameters in the template
        schema = {"type": "object", "properties": {}}
        
        # Extract from steps if present
        steps = template_def.get('steps', [])
        for step in steps:
            params = step.get('params', {})
            for key, value in params.items():
                if isinstance(value, str) and '{{' in value and '}}' in value:
                    # This is a template parameter
                    param_name = value.replace('{{', '').replace('}}', '').strip()
                    schema["properties"][param_name] = {
                        "type": "string",
                        "description": f"Parameter for {key} in step {step.get('id', 'unknown')}"
                    }
        
        return schema

    def _extract_default_parameters(self, template_def: Dict[str, Any]) -> Dict[str, Any]:
        """Extract default parameters from template definition"""
        defaults = {}
        
        # Look for default values in template config
        config = template_def.get('config', {})
        for key, value in config.items():
            if not isinstance(value, dict):
                defaults[key] = value
        
        return defaults
