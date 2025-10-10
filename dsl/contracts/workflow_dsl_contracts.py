"""
Workflow DSL Contracts
Task 9.3.4: Implement workflow DSL contracts

Declarative definition of end-to-end flows with governance fields embedded.
Provides contract validation, versioning, and compliance enforcement.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import hashlib
import jsonschema
from jsonschema import Draft7Validator

logger = logging.getLogger(__name__)

class ContractType(Enum):
    """DSL contract types"""
    WORKFLOW = "workflow"
    STEP = "step"
    CONNECTOR = "connector"
    POLICY = "policy"
    EVIDENCE = "evidence"

class ContractStatus(Enum):
    """Contract lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class ValidationLevel(Enum):
    """Contract validation levels"""
    STRICT = "strict"      # All fields required, no additional properties
    STANDARD = "standard"  # Required fields enforced, additional allowed
    LENIENT = "lenient"    # Minimal validation, mostly warnings

@dataclass
class GovernanceFields:
    """Mandatory governance fields for all DSL contracts"""
    tenant_id: int
    region_id: str
    policy_pack_id: Optional[str] = None
    compliance_frameworks: List[str] = field(default_factory=list)
    sla_tier: str = "T2"
    data_classification: str = "internal"
    retention_days: int = 2555  # 7 years default
    
    def validate(self) -> Dict[str, Any]:
        """Validate governance fields"""
        errors = []
        
        if not self.tenant_id or self.tenant_id <= 0:
            errors.append("tenant_id must be a positive integer")
        
        if not self.region_id:
            errors.append("region_id is required")
        
        if self.sla_tier not in ["T0", "T1", "T2"]:
            errors.append("sla_tier must be T0, T1, or T2")
        
        if self.data_classification not in ["public", "internal", "confidential", "restricted"]:
            errors.append("Invalid data_classification")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

@dataclass
class ContractMetadata:
    """Contract metadata and versioning information"""
    contract_id: str
    contract_name: str
    contract_type: ContractType
    version: str
    status: ContractStatus
    
    # Authorship and lifecycle
    created_by: str
    created_at: datetime
    updated_at: datetime
    description: str
    tags: List[str] = field(default_factory=list)
    
    # Dependencies and relationships
    depends_on: List[str] = field(default_factory=list)
    used_by: List[str] = field(default_factory=list)
    
    # Validation and quality
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    quality_score: float = 0.0
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        result['contract_type'] = self.contract_type.value
        result['status'] = self.status.value
        result['validation_level'] = self.validation_level.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result

@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_id: str
    step_name: str
    step_type: str  # query, decision, ml_decision, agent_call, notify, governance
    
    # Step configuration
    parameters: Dict[str, Any]
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    # Execution configuration
    timeout_ms: int = 30000
    retry_count: int = 3
    fallback_step: Optional[str] = None
    
    # Governance
    governance_required: bool = True
    evidence_capture: bool = True
    policy_checks: List[str] = field(default_factory=list)
    
    def validate(self) -> Dict[str, Any]:
        """Validate step definition"""
        errors = []
        warnings = []
        
        if not self.step_id:
            errors.append("step_id is required")
        
        if not self.step_name:
            errors.append("step_name is required")
        
        valid_step_types = ["query", "decision", "ml_decision", "agent_call", "notify", "governance"]
        if self.step_type not in valid_step_types:
            errors.append(f"step_type must be one of: {valid_step_types}")
        
        if self.timeout_ms <= 0:
            errors.append("timeout_ms must be positive")
        
        if self.retry_count < 0:
            warnings.append("retry_count should be non-negative")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

@dataclass
class WorkflowContract:
    """Complete workflow DSL contract definition"""
    # Contract metadata
    metadata: ContractMetadata
    governance: GovernanceFields
    
    # Workflow definition
    workflow_name: str
    workflow_description: str
    workflow_version: str
    
    # Steps and flow
    steps: List[WorkflowStep]
    step_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Execution configuration
    execution_mode: str = "sequential"  # sequential, parallel, conditional
    max_execution_time_ms: int = 300000  # 5 minutes default
    
    # Input/Output contracts
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    error_handling: Dict[str, Any] = field(default_factory=dict)
    rollback_strategy: str = "none"  # none, partial, full
    
    # Monitoring and observability
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    sla_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Dict[str, Any]:
        """Comprehensive contract validation"""
        errors = []
        warnings = []
        
        # Validate metadata
        if not self.metadata:
            errors.append("Contract metadata is required")
        
        # Validate governance
        if not self.governance:
            errors.append("Governance fields are required")
        else:
            gov_validation = self.governance.validate()
            if not gov_validation["valid"]:
                errors.extend(gov_validation["errors"])
        
        # Validate workflow basics
        if not self.workflow_name:
            errors.append("workflow_name is required")
        
        if not self.steps:
            errors.append("At least one workflow step is required")
        
        # Validate steps
        step_ids = set()
        for step in self.steps:
            step_validation = step.validate()
            if not step_validation["valid"]:
                errors.extend([f"Step {step.step_id}: {error}" for error in step_validation["errors"]])
            
            warnings.extend([f"Step {step.step_id}: {warning}" for warning in step_validation.get("warnings", [])])
            
            # Check for duplicate step IDs
            if step.step_id in step_ids:
                errors.append(f"Duplicate step_id: {step.step_id}")
            step_ids.add(step.step_id)
        
        # Validate step dependencies
        for step_id, deps in self.step_dependencies.items():
            if step_id not in step_ids:
                errors.append(f"Step dependency references unknown step: {step_id}")
            
            for dep in deps:
                if dep not in step_ids:
                    errors.append(f"Step {step_id} depends on unknown step: {dep}")
        
        # Validate execution mode
        valid_modes = ["sequential", "parallel", "conditional"]
        if self.execution_mode not in valid_modes:
            errors.append(f"execution_mode must be one of: {valid_modes}")
        
        # Validate schemas if provided
        if self.input_schema:
            try:
                Draft7Validator.check_schema(self.input_schema)
            except Exception as e:
                errors.append(f"Invalid input_schema: {e}")
        
        if self.output_schema:
            try:
                Draft7Validator.check_schema(self.output_schema)
            except Exception as e:
                errors.append(f"Invalid output_schema: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "step_count": len(self.steps),
            "governance_compliant": self.governance.validate()["valid"] if self.governance else False
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary for storage/transmission"""
        return {
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "governance": asdict(self.governance) if self.governance else {},
            "workflow_name": self.workflow_name,
            "workflow_description": self.workflow_description,
            "workflow_version": self.workflow_version,
            "steps": [asdict(step) for step in self.steps],
            "step_dependencies": self.step_dependencies,
            "execution_mode": self.execution_mode,
            "max_execution_time_ms": self.max_execution_time_ms,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "error_handling": self.error_handling,
            "rollback_strategy": self.rollback_strategy,
            "monitoring_config": self.monitoring_config,
            "sla_requirements": self.sla_requirements
        }
    
    def get_contract_hash(self) -> str:
        """Generate hash for contract integrity verification"""
        contract_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(contract_str.encode()).hexdigest()

class WorkflowDSLContractManager:
    """
    Workflow DSL Contract Manager
    Task 9.3.4: Declarative definition of end-to-end flows
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Contract registry and cache
        self.contract_cache = {}  # contract_id -> WorkflowContract
        self.contract_templates = {}
        
        # Validation configurations
        self.validation_rules = {
            "governance_required": True,
            "schema_validation_enabled": True,
            "dependency_validation_enabled": True,
            "sla_validation_enabled": True
        }
        
        # Industry-specific contract templates
        self.industry_templates = {
            "SaaS": self._get_saas_contract_templates(),
            "Banking": self._get_banking_contract_templates(),
            "Insurance": self._get_insurance_contract_templates()
        }
        
    async def initialize(self) -> bool:
        """Initialize the contract manager"""
        try:
            self.logger.info("🚀 Initializing Workflow DSL Contract Manager...")
            
            # Create database tables
            await self._create_contract_tables()
            
            # Load existing contracts
            await self._load_contract_cache()
            
            # Initialize industry templates
            await self._initialize_industry_templates()
            
            self.logger.info("✅ Workflow DSL Contract Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Contract Manager: {e}")
            return False
    
    async def create_contract(self, 
                            workflow_name: str,
                            workflow_description: str,
                            steps: List[Dict[str, Any]],
                            governance: Dict[str, Any],
                            created_by: str,
                            contract_type: ContractType = ContractType.WORKFLOW,
                            validation_level: ValidationLevel = ValidationLevel.STANDARD) -> str:
        """
        Create a new workflow DSL contract
        """
        try:
            contract_id = str(uuid.uuid4())
            
            # Create metadata
            metadata = ContractMetadata(
                contract_id=contract_id,
                contract_name=workflow_name,
                contract_type=contract_type,
                version="1.0.0",
                status=ContractStatus.DRAFT,
                created_by=created_by,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                description=workflow_description,
                validation_level=validation_level
            )
            
            # Create governance fields
            gov_fields = GovernanceFields(
                tenant_id=governance["tenant_id"],
                region_id=governance["region_id"],
                policy_pack_id=governance.get("policy_pack_id"),
                compliance_frameworks=governance.get("compliance_frameworks", []),
                sla_tier=governance.get("sla_tier", "T2"),
                data_classification=governance.get("data_classification", "internal"),
                retention_days=governance.get("retention_days", 2555)
            )
            
            # Create workflow steps
            workflow_steps = []
            for step_data in steps:
                step = WorkflowStep(
                    step_id=step_data["step_id"],
                    step_name=step_data["step_name"],
                    step_type=step_data["step_type"],
                    parameters=step_data.get("parameters", {}),
                    inputs=step_data.get("inputs", []),
                    outputs=step_data.get("outputs", []),
                    timeout_ms=step_data.get("timeout_ms", 30000),
                    retry_count=step_data.get("retry_count", 3),
                    fallback_step=step_data.get("fallback_step"),
                    governance_required=step_data.get("governance_required", True),
                    evidence_capture=step_data.get("evidence_capture", True),
                    policy_checks=step_data.get("policy_checks", [])
                )
                workflow_steps.append(step)
            
            # Create contract
            contract = WorkflowContract(
                metadata=metadata,
                governance=gov_fields,
                workflow_name=workflow_name,
                workflow_description=workflow_description,
                workflow_version="1.0.0",
                steps=workflow_steps
            )
            
            # Validate contract
            validation_result = contract.validate()
            if not validation_result["valid"]:
                raise ValueError(f"Contract validation failed: {validation_result['errors']}")
            
            # Store contract
            await self._store_contract(contract)
            
            # Update cache
            self.contract_cache[contract_id] = contract
            
            self.logger.info(f"✅ Contract created: {workflow_name} (ID: {contract_id})")
            return contract_id
            
        except Exception as e:
            self.logger.error(f"❌ Contract creation failed: {e}")
            raise
    
    async def get_contract(self, contract_id: str) -> Optional[WorkflowContract]:
        """Get contract by ID"""
        try:
            # Check cache first
            if contract_id in self.contract_cache:
                return self.contract_cache[contract_id]
            
            # Query database
            if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    result = await conn.fetchrow("""
                        SELECT * FROM workflow_contracts WHERE contract_id = $1
                    """, contract_id)
                    
                    if result:
                        contract = self._row_to_contract(result)
                        self.contract_cache[contract_id] = contract
                        return contract
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get contract {contract_id}: {e}")
            return None
    
    async def validate_contract(self, contract_id: str) -> Dict[str, Any]:
        """Validate an existing contract"""
        try:
            contract = await self.get_contract(contract_id)
            if not contract:
                return {"valid": False, "errors": ["Contract not found"]}
            
            return contract.validate()
            
        except Exception as e:
            self.logger.error(f"❌ Contract validation failed: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    async def execute_contract_validation(self, 
                                        contract_id: str, 
                                        input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against contract's input schema"""
        try:
            contract = await self.get_contract(contract_id)
            if not contract:
                return {"valid": False, "errors": ["Contract not found"]}
            
            # Validate against input schema if defined
            if contract.input_schema:
                try:
                    validator = Draft7Validator(contract.input_schema)
                    errors = list(validator.iter_errors(input_data))
                    
                    return {
                        "valid": len(errors) == 0,
                        "errors": [error.message for error in errors],
                        "contract_id": contract_id
                    }
                except Exception as e:
                    return {"valid": False, "errors": [f"Schema validation error: {e}"]}
            
            # If no schema defined, basic validation
            return {"valid": True, "errors": [], "warnings": ["No input schema defined"]}
            
        except Exception as e:
            self.logger.error(f"❌ Contract input validation failed: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    async def list_contracts(self, 
                           tenant_id: int = None,
                           contract_type: ContractType = None,
                           status: ContractStatus = None) -> List[ContractMetadata]:
        """List contracts with optional filters"""
        try:
            contracts = []
            
            if self.pool_manager and hasattr(self.pool_manager, 'postgres_pool'):
                async with self.pool_manager.postgres_pool.acquire() as conn:
                    # Build dynamic query
                    conditions = []
                    params = []
                    param_count = 0
                    
                    if tenant_id:
                        param_count += 1
                        conditions.append(f"governance->>'tenant_id' = ${param_count}")
                        params.append(str(tenant_id))
                    
                    if contract_type:
                        param_count += 1
                        conditions.append(f"contract_type = ${param_count}")
                        params.append(contract_type.value)
                    
                    if status:
                        param_count += 1
                        conditions.append(f"status = ${param_count}")
                        params.append(status.value)
                    
                    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                    
                    query = f"""
                        SELECT metadata FROM workflow_contracts 
                        {where_clause}
                        ORDER BY created_at DESC
                    """
                    
                    results = await conn.fetch(query, *params)
                    
                    for row in results:
                        metadata_dict = json.loads(row['metadata'])
                        metadata = self._dict_to_metadata(metadata_dict)
                        contracts.append(metadata)
            
            return contracts
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list contracts: {e}")
            return []
    
    async def get_industry_templates(self, industry: str) -> Dict[str, Any]:
        """Get industry-specific contract templates"""
        try:
            if industry in self.industry_templates:
                return {
                    "industry": industry,
                    "templates": self.industry_templates[industry],
                    "count": len(self.industry_templates[industry])
                }
            else:
                return {
                    "industry": industry,
                    "templates": {},
                    "count": 0
                }
        except Exception as e:
            self.logger.error(f"❌ Failed to get industry templates: {e}")
            return {"industry": industry, "templates": {}, "count": 0}
    
    # Private helper methods
    
    async def _create_contract_tables(self):
        """Create database tables for contract storage"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        async with self.pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_contracts (
                    contract_id UUID PRIMARY KEY,
                    contract_name VARCHAR(255) NOT NULL,
                    contract_type VARCHAR(50) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    metadata JSONB NOT NULL,
                    governance JSONB NOT NULL,
                    contract_definition JSONB NOT NULL,
                    contract_hash VARCHAR(64) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    INDEX(contract_name),
                    INDEX(contract_type),
                    INDEX(status),
                    INDEX((governance->>'tenant_id')),
                    INDEX(created_at)
                )
            """)
    
    async def _load_contract_cache(self):
        """Load existing contracts into cache"""
        try:
            contracts = await self.list_contracts()
            for metadata in contracts:
                contract = await self.get_contract(metadata.contract_id)
                if contract:
                    self.contract_cache[metadata.contract_id] = contract
        except Exception as e:
            self.logger.warning(f"Failed to load contract cache: {e}")
    
    async def _initialize_industry_templates(self):
        """Initialize industry-specific templates"""
        # Templates would be loaded/registered here
        pass
    
    async def _store_contract(self, contract: WorkflowContract):
        """Store contract in database"""
        if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
            return
        
        async with self.pool_manager.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflow_contracts 
                (contract_id, contract_name, contract_type, version, status,
                 metadata, governance, contract_definition, contract_hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (contract_id) DO UPDATE SET
                contract_definition = EXCLUDED.contract_definition,
                contract_hash = EXCLUDED.contract_hash,
                updated_at = NOW()
            """, 
            contract.metadata.contract_id,
            contract.metadata.contract_name,
            contract.metadata.contract_type.value,
            contract.metadata.version,
            contract.metadata.status.value,
            json.dumps(contract.metadata.to_dict()),
            json.dumps(asdict(contract.governance)),
            json.dumps(contract.to_dict()),
            contract.get_contract_hash())
    
    def _row_to_contract(self, row) -> WorkflowContract:
        """Convert database row to WorkflowContract object"""
        contract_dict = json.loads(row['contract_definition'])
        
        # Reconstruct metadata
        metadata_dict = json.loads(row['metadata'])
        metadata = self._dict_to_metadata(metadata_dict)
        
        # Reconstruct governance
        governance_dict = json.loads(row['governance'])
        governance = GovernanceFields(**governance_dict)
        
        # Reconstruct steps
        steps = []
        for step_data in contract_dict.get("steps", []):
            step = WorkflowStep(**step_data)
            steps.append(step)
        
        # Create contract
        contract = WorkflowContract(
            metadata=metadata,
            governance=governance,
            workflow_name=contract_dict["workflow_name"],
            workflow_description=contract_dict["workflow_description"],
            workflow_version=contract_dict["workflow_version"],
            steps=steps,
            step_dependencies=contract_dict.get("step_dependencies", {}),
            execution_mode=contract_dict.get("execution_mode", "sequential"),
            max_execution_time_ms=contract_dict.get("max_execution_time_ms", 300000),
            input_schema=contract_dict.get("input_schema", {}),
            output_schema=contract_dict.get("output_schema", {}),
            error_handling=contract_dict.get("error_handling", {}),
            rollback_strategy=contract_dict.get("rollback_strategy", "none"),
            monitoring_config=contract_dict.get("monitoring_config", {}),
            sla_requirements=contract_dict.get("sla_requirements", {})
        )
        
        return contract
    
    def _dict_to_metadata(self, metadata_dict: Dict[str, Any]) -> ContractMetadata:
        """Convert dictionary to ContractMetadata object"""
        return ContractMetadata(
            contract_id=metadata_dict["contract_id"],
            contract_name=metadata_dict["contract_name"],
            contract_type=ContractType(metadata_dict["contract_type"]),
            version=metadata_dict["version"],
            status=ContractStatus(metadata_dict["status"]),
            created_by=metadata_dict["created_by"],
            created_at=datetime.fromisoformat(metadata_dict["created_at"]),
            updated_at=datetime.fromisoformat(metadata_dict["updated_at"]),
            description=metadata_dict["description"],
            tags=metadata_dict.get("tags", []),
            depends_on=metadata_dict.get("depends_on", []),
            used_by=metadata_dict.get("used_by", []),
            validation_level=ValidationLevel(metadata_dict.get("validation_level", "standard")),
            quality_score=metadata_dict.get("quality_score", 0.0),
            usage_count=metadata_dict.get("usage_count", 0)
        )
    
    # Industry-specific template methods
    
    def _get_saas_contract_templates(self) -> Dict[str, Any]:
        """SaaS industry contract templates"""
        return {
            "subscription_lifecycle": {
                "description": "SaaS subscription lifecycle workflow",
                "steps": [
                    {"step_id": "validate_subscription", "step_type": "query"},
                    {"step_id": "check_billing", "step_type": "decision"},
                    {"step_id": "update_status", "step_type": "governance"}
                ]
            },
            "churn_prediction": {
                "description": "Customer churn prediction workflow",
                "steps": [
                    {"step_id": "gather_usage_data", "step_type": "query"},
                    {"step_id": "predict_churn", "step_type": "ml_decision"},
                    {"step_id": "notify_success_team", "step_type": "notify"}
                ]
            }
        }
    
    def _get_banking_contract_templates(self) -> Dict[str, Any]:
        """Banking industry contract templates"""
        return {
            "transaction_processing": {
                "description": "Banking transaction processing workflow",
                "steps": [
                    {"step_id": "validate_transaction", "step_type": "governance"},
                    {"step_id": "check_limits", "step_type": "decision"},
                    {"step_id": "process_payment", "step_type": "query"}
                ]
            }
        }
    
    def _get_insurance_contract_templates(self) -> Dict[str, Any]:
        """Insurance industry contract templates"""
        return {
            "claim_processing": {
                "description": "Insurance claim processing workflow",
                "steps": [
                    {"step_id": "validate_claim", "step_type": "governance"},
                    {"step_id": "assess_risk", "step_type": "ml_decision"},
                    {"step_id": "approve_claim", "step_type": "decision"}
                ]
            }
        }
