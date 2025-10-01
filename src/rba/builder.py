"""
RBA Builder - Task 12.1 Implementation
=====================================

SOP-first, policy-aware, orchestrator-anchored builder for deterministic workflows.

Task Reference:
- 12.1-T01: Define Builder design principles (SOP-first, policy-aware, orchestrator-anchored)
- 12.1-T02: Draft detailed wireframes for builder UI
- 12.1-T03: Define DSL grammar for SOP encoding
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

from ..services.connection_pool_manager import pool_manager

class WorkflowState(Enum):
    """Workflow lifecycle states - Task 12.1-T24"""
    DRAFT = "draft"
    REVIEW = "review" 
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class BlockType(Enum):
    """DSL Block primitives - Task 12.1-T05"""
    QUERY = "query"
    TRANSFORM = "transform"
    DECISION = "decision"
    ACTION = "action"
    NOTIFY = "notify"
    GOVERNANCE = "governance"

@dataclass
class WorkflowBlock:
    """
    Block metadata schema - Task 12.1-T06
    Registry-ready block definition with compliance tags
    """
    id: str
    type: BlockType
    params: Dict[str, Any]
    overlay_refs: List[str] = None
    compliance_tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.overlay_refs is None:
            self.overlay_refs = []
        if self.compliance_tags is None:
            self.compliance_tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class RBAWorkflow:
    """
    Complete RBA Workflow definition
    Task 12.1-T03: DSL grammar for SOP encoding
    """
    workflow_id: str
    name: str
    description: str
    industry: str = "SaaS"  # Default to SaaS as requested
    automation_type: str = "RBA"
    version: str = "1.0.0"
    state: WorkflowState = WorkflowState.DRAFT
    blocks: List[WorkflowBlock] = None
    governance: Dict[str, Any] = None
    sla_config: Dict[str, Any] = None
    tenant_id: Optional[int] = None
    created_by: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.blocks is None:
            self.blocks = []
        if self.governance is None:
            self.governance = {
                "policy_packs": [],
                "evidence_required": True,
                "override_allowed": False
            }
        if self.sla_config is None:
            self.sla_config = {
                "timeout_minutes": 30,
                "tier": "T1"
            }
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RBABuilder:
    """
    SOP-driven No-Code Builder for RBA workflows
    
    Design Principles (Task 12.1-T01):
    - SOP-first: Workflows encode Standard Operating Procedures
    - Policy-aware: Governance baked in from design time  
    - Orchestrator-anchored: All workflows route through Routing Orchestrator
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pool_manager = pool_manager
        
        # Task 12.1-T07: SaaS block library
        self._initialize_saas_blocks()
        
    def _initialize_saas_blocks(self):
        """
        Task 12.1-T07: Create SaaS block library 
        (pipeline hygiene, comp plan adjustments)
        """
        self.saas_blocks = {
            "pipeline_hygiene_check": {
                "type": BlockType.QUERY,
                "template": {
                    "source": "postgres",
                    "resource": "opportunities",
                    "filters": [
                        {"field": "last_activity_date", "op": "<", "value": "{{hygiene_threshold_days}} days ago"},
                        {"field": "stage", "op": "not in", "value": ["Closed Won", "Closed Lost"]}
                    ]
                },
                "compliance_tags": ["pipeline_hygiene", "revenue_ops"]
            },
            "quota_compliance_check": {
                "type": BlockType.DECISION,
                "template": {
                    "condition": "{{current_coverage}} < {{minimum_coverage_ratio}}",
                    "true_action": "escalate_to_manager",
                    "false_action": "continue_workflow"
                },
                "compliance_tags": ["quota_management", "revenue_ops"]
            },
            "forecast_approval_gate": {
                "type": BlockType.GOVERNANCE,
                "template": {
                    "policy_id": "forecast_approval_policy",
                    "evidence_required": True,
                    "approver_role": "sales_manager",
                    "sod_required": True
                },
                "compliance_tags": ["forecast_governance", "sox_compliance"]
            }
        }
        
    async def create_workflow(
        self, 
        name: str, 
        description: str,
        tenant_id: int,
        user_id: int,
        industry: str = "SaaS"
    ) -> RBAWorkflow:
        """
        Create new RBA workflow in DRAFT state
        Task 12.1-T24: Implement draft → review state machine
        """
        try:
            workflow = RBAWorkflow(
                workflow_id=str(uuid.uuid4()),
                name=name,
                description=description,
                industry=industry,
                tenant_id=tenant_id,
                created_by=user_id,
                state=WorkflowState.DRAFT
            )
            
            # Store in database
            await self._store_workflow(workflow)
            
            self.logger.info(f"Created RBA workflow: {workflow.workflow_id}")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            raise
    
    async def add_block(
        self,
        workflow_id: str,
        block_type: BlockType,
        params: Dict[str, Any],
        compliance_tags: List[str] = None
    ) -> WorkflowBlock:
        """
        Add block to existing workflow
        Task 12.1-T20: Build block addition APIs
        """
        try:
            # Get workflow
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            # Create new block
            block = WorkflowBlock(
                id=str(uuid.uuid4()),
                type=block_type,
                params=params,
                compliance_tags=compliance_tags or []
            )
            
            # Add to workflow
            workflow.blocks.append(block)
            
            # Update workflow in database
            await self._store_workflow(workflow)
            
            self.logger.info(f"Added block to workflow: {workflow_id}")
            return block
            
        except Exception as e:
            self.logger.error(f"Failed to add block: {e}")
            raise

    async def apply_template(self, workflow_id: str, template_id: str):
        """
        Apply template to workflow
        Task 12.1-T09: Template application to workflows
        """
        try:
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            # Get template blocks based on template_id
            # First check if it's a UUID from the capability registry
            template_blocks = []
            if len(template_id) > 20:  # Likely a UUID
                # Look up template in capability registry to determine type
                if self.pool_manager and self.pool_manager.postgres_pool:
                    async with self.pool_manager.postgres_pool.acquire() as conn:
                        template_row = await conn.fetchrow(
                            "SELECT name, description FROM ro_capabilities WHERE id = $1",
                            template_id
                        )
                        if template_row:
                            template_name = template_row['name'].lower()
                            if 'pipeline' in template_name and 'hygiene' in template_name:
                                template_blocks = self._get_pipeline_hygiene_template_blocks()
                            elif 'forecast' in template_name and 'approval' in template_name:
                                template_blocks = self._get_forecast_approval_template_blocks()
                            else:
                                # Default to pipeline hygiene for unknown templates
                                template_blocks = self._get_pipeline_hygiene_template_blocks()
                        else:
                            # Template not found, use default
                            template_blocks = self._get_pipeline_hygiene_template_blocks()
                else:
                    # No database connection, use default
                    template_blocks = self._get_pipeline_hygiene_template_blocks()
            else:
                # Legacy string-based template IDs
                if "pipeline_hygiene" in template_id:
                    template_blocks = self._get_pipeline_hygiene_template_blocks()
                elif "forecast_approval" in template_id:
                    template_blocks = self._get_forecast_approval_template_blocks()
                else:
                    raise ValueError(f"Unknown template: {template_id}")
            
            # Add template blocks to workflow
            workflow.blocks.extend(template_blocks)
            
            # Update workflow
            await self._store_workflow(workflow)
            
            self.logger.info(f"Applied template {template_id} to workflow: {workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply template: {e}")
            raise

    async def get_workflow(self, workflow_id: str) -> Optional[RBAWorkflow]:
        """Get workflow by ID"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                self.logger.warning("No database connection - using mock data")
                return self._get_mock_workflow(workflow_id)
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM rba_workflows WHERE workflow_id = $1", 
                    workflow_id
                )
                
                if row:
                    return self._row_to_workflow(row)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow: {e}")
            return None

    async def list_workflows(
        self, 
        tenant_id: int, 
        user_id: Optional[int] = None,
        state: Optional[str] = None,
        industry: Optional[str] = None
    ) -> List[RBAWorkflow]:
        """List workflows with filtering"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                self.logger.warning("No database connection - using mock data")
                return self._get_mock_workflows()
            
            # Build query with filters
            query = "SELECT * FROM rba_workflows WHERE tenant_id = $1"
            params = [tenant_id]
            
            if user_id:
                query += f" AND created_by = ${len(params) + 1}"
                params.append(user_id)
            
            if state:
                query += f" AND state = ${len(params) + 1}"
                params.append(state)
                
            query += " ORDER BY created_at DESC"
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [self._row_to_workflow(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to list workflows: {e}")
            return []

    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> Optional[RBAWorkflow]:
        """Update workflow"""
        try:
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                return None
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(workflow, key):
                    if key == 'state' and isinstance(value, str):
                        setattr(workflow, key, WorkflowState(value))
                    else:
                        setattr(workflow, key, value)
            
            # Store updated workflow
            await self._store_workflow(workflow)
            
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to update workflow: {e}")
            return None

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow"""
        try:
            if not self.pool_manager or not self.pool_manager.postgres_pool:
                self.logger.warning("No database connection - mock delete")
                return True
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM rba_workflows WHERE workflow_id = $1",
                    workflow_id
                )
                return result == "DELETE 1"
                
        except Exception as e:
            self.logger.error(f"Failed to delete workflow: {e}")
            return False

    async def publish_workflow(self, workflow_id: str) -> Optional[RBAWorkflow]:
        """Publish workflow to production"""
        try:
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                return None
            
            # Change state to published
            workflow.state = WorkflowState.PUBLISHED
            
            # Store updated workflow
            await self._store_workflow(workflow)
            
            self.logger.info(f"Published workflow: {workflow_id}")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to publish workflow: {e}")
            return None

    # Helper methods
    def _row_to_workflow(self, row) -> RBAWorkflow:
        """Convert database row to RBAWorkflow"""
        # Parse blocks from JSON if they exist
        blocks = []
        if row.get('blocks'):
            try:
                blocks_data = row['blocks'] if isinstance(row['blocks'], list) else json.loads(row['blocks'])
                blocks = []
                for block_data in blocks_data:
                    try:
                        # Handle different block data structures
                        if isinstance(block_data, dict):
                            # Ensure required fields exist
                            # Handle both old and new formats
                            block_type_str = block_data.get('type') or block_data.get('block_type', 'QUERY')
                            block_dict = {
                                'id': block_data.get('id', block_data.get('block_id', str(uuid.uuid4()))),
                                'type': BlockType(block_type_str),
                                'params': block_data.get('params', {}),
                                'overlay_refs': block_data.get('overlay_refs', []),
                                'compliance_tags': block_data.get('compliance_tags', []),
                                'metadata': block_data.get('metadata', {})
                            }
                            blocks.append(WorkflowBlock(**block_dict))
                    except Exception as block_error:
                        self.logger.warning(f"Skipping invalid block: {block_error}")
                        continue
            except (json.JSONDecodeError, TypeError) as e:
                self.logger.warning(f"Failed to parse blocks JSON: {e}")
                blocks = []
        
        return RBAWorkflow(
            workflow_id=str(row['workflow_id']),
            name=row['name'],
            description=row['description'],
            industry=row.get('industry', 'SaaS'),
            automation_type=row.get('automation_type', 'RBA'),
            version=row.get('version', '1.0.0'),
            state=WorkflowState(row.get('state', 'draft')),
            tenant_id=row['tenant_id'],
            created_by=row['created_by'],
            created_at=row['created_at'],
            blocks=blocks,
            governance=self._parse_json_field(row.get('governance', {})),
            sla_config=self._parse_json_field(row.get('sla_config', {}))
        )

    def _parse_json_field(self, field):
        """Parse JSON field that might be a string or already parsed dict"""
        if isinstance(field, str):
            try:
                return json.loads(field)
            except (json.JSONDecodeError, TypeError):
                return {}
        elif isinstance(field, dict):
            return field
        else:
            return {}

    def _get_mock_workflow(self, workflow_id: str) -> RBAWorkflow:
        """Get mock workflow for testing"""
        return RBAWorkflow(
            workflow_id=workflow_id,
            name="Mock Pipeline Hygiene Workflow",
            description="Sample pipeline hygiene workflow for testing",
            industry="SaaS",
            tenant_id=1300,
            created_by=1319,
            state=WorkflowState.DRAFT
        )

    def _get_mock_workflows(self) -> List[RBAWorkflow]:
        """Get mock workflows for testing"""
        return [
            RBAWorkflow(
                workflow_id="mock-pipeline-hygiene",
                name="Pipeline Hygiene Workflow",
                description="Automated pipeline health monitoring",
                industry="SaaS",
                tenant_id=1300,
                created_by=1319,
                state=WorkflowState.PUBLISHED
            ),
            RBAWorkflow(
                workflow_id="mock-forecast-approval",
                name="Forecast Approval Workflow",
                description="Automated forecast variance analysis",
                industry="SaaS",
                tenant_id=1300,
                created_by=1319,
                state=WorkflowState.APPROVED
            )
        ]

    def _get_pipeline_hygiene_template_blocks(self) -> List[WorkflowBlock]:
        """Get pipeline hygiene template blocks"""
        return [
            WorkflowBlock(
                id=str(uuid.uuid4()),
                type=BlockType.QUERY,
                params=self.saas_blocks["pipeline_hygiene_check"]["template"],
                compliance_tags=self.saas_blocks["pipeline_hygiene_check"]["compliance_tags"]
            ),
            WorkflowBlock(
                id=str(uuid.uuid4()),
                type=BlockType.DECISION,
                params=self.saas_blocks["quota_compliance_check"]["template"],
                compliance_tags=self.saas_blocks["quota_compliance_check"]["compliance_tags"]
            )
        ]

    def _get_forecast_approval_template_blocks(self) -> List[WorkflowBlock]:
        """Get forecast approval template blocks"""
        return [
            WorkflowBlock(
                id=str(uuid.uuid4()),
                type=BlockType.GOVERNANCE,
                params=self.saas_blocks["forecast_approval_gate"]["template"],
                compliance_tags=self.saas_blocks["forecast_approval_gate"]["compliance_tags"]
            )
        ]
    
    async def create_saas_pipeline_hygiene_template(self, tenant_id: int, user_id: int) -> RBAWorkflow:
        """
        Task 19.3-T03: SaaS template: Pipeline hygiene + quota compliance
        
        Creates the official SaaS template as specified in the task sheet
        """
        try:
            # Create base workflow
            workflow = await self.create_workflow(
                name="SaaS Pipeline Hygiene + Quota Compliance",
                description="Automated pipeline hygiene monitoring with quota compliance checks for SaaS revenue operations",
                tenant_id=tenant_id,
                user_id=user_id,
                industry="SaaS"
            )
            
            # Add pipeline hygiene check block
            hygiene_block = await self.add_block(
                workflow.workflow_id,
                BlockType.QUERY,
                {
                    "source": "postgres",
                    "resource": "opportunities", 
                    "filters": [
                        {"field": "last_activity_date", "op": "<", "value": "15 days ago"},
                        {"field": "stage", "op": "not in", "value": ["Closed Won", "Closed Lost"]},
                        {"field": "tenant_id", "op": "=", "value": "{{tenant_id}}"}
                    ],
                    "select": ["opportunity_id", "account_name", "owner_name", "stage", "amount"]
                },
                compliance_tags=["pipeline_hygiene", "revenue_ops", "saas"]
            )
            
            # Add quota compliance check
            quota_block = await self.add_block(
                workflow.workflow_id,
                BlockType.DECISION,
                {
                    "condition": "{{pipeline_coverage}} < {{minimum_coverage_ratio}}",
                    "true_action": {
                        "type": "notify",
                        "params": {
                            "channel": "slack",
                            "recipient": "#revenue-ops",
                            "message": "🚨 Pipeline hygiene alert: {{stale_opportunities_count}} opportunities stale >15 days. Coverage: {{pipeline_coverage}}x (Target: {{minimum_coverage_ratio}}x)"
                        }
                    },
                    "false_action": {
                        "type": "notify", 
                        "params": {
                            "channel": "slack",
                            "recipient": "#revenue-ops",
                            "message": "✅ Pipeline hygiene check passed. Coverage: {{pipeline_coverage}}x"
                        }
                    }
                },
                compliance_tags=["quota_compliance", "revenue_ops", "saas"]
            )
            
            # Add governance block
            governance_block = await self.add_block(
                workflow.workflow_id,
                BlockType.GOVERNANCE,
                {
                    "policy_id": "saas_pipeline_governance",
                    "evidence_required": True,
                    "action": "workflow_complete",
                    "evidence_fields": ["stale_opportunities", "coverage_ratio", "notification_sent"]
                },
                compliance_tags=["governance", "audit_trail", "sox_compliance"]
            )
            
            self.logger.info(f"Created SaaS Pipeline Hygiene template: {workflow.workflow_id}")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to create SaaS pipeline hygiene template: {e}")
            raise
    
    async def create_saas_forecast_approval_template(self, tenant_id: int, user_id: int) -> RBAWorkflow:
        """
        Task 19.3-T04: SaaS template: Forecast approval governance
        
        Creates the official SaaS forecast approval template as specified in the task sheet
        """
        try:
            # Create base workflow
            workflow = await self.create_workflow(
                name="SaaS Forecast Approval Governance",
                description="Automated forecast approval workflow with governance controls for SaaS CRO trust",
                tenant_id=tenant_id,
                user_id=user_id,
                industry="SaaS"
            )
            
            # Add forecast data fetch block
            forecast_fetch_block = await self.add_block(
                workflow.workflow_id,
                BlockType.QUERY,
                {
                    "source": "postgres",
                    "resource": "forecasts",
                    "filters": [
                        {"field": "status", "op": "=", "value": "pending_approval"},
                        {"field": "tenant_id", "op": "=", "value": "{{tenant_id}}"},
                        {"field": "forecast_period", "op": "=", "value": "{{current_quarter}}"}
                    ],
                    "select": ["forecast_id", "forecast_amount", "submitter_id", "variance_pct", "confidence_score"]
                },
                compliance_tags=["forecast_management", "revenue_ops", "saas"]
            )
            
            # Add approval decision logic
            approval_decision_block = await self.add_block(
                workflow.workflow_id,
                BlockType.DECISION,
                {
                    "condition": "{{variance_pct}} > 10 OR {{confidence_score}} < 0.8",
                    "true_action": {
                        "type": "governance",
                        "params": {
                            "policy_id": "forecast_escalation_policy",
                            "required_approver": "cro",
                            "sod_check": True,
                            "evidence_required": True,
                            "justification_required": True
                        }
                    },
                    "false_action": {
                        "type": "action",
                        "params": {
                            "action": "auto_approve",
                            "update_status": "approved",
                            "notify_submitter": True
                        }
                    }
                },
                compliance_tags=["forecast_approval", "governance", "cro_oversight"]
            )
            
            # Add final governance block
            governance_block = await self.add_block(
                workflow.workflow_id,
                BlockType.GOVERNANCE,
                {
                    "policy_id": "saas_forecast_governance",
                    "evidence_required": True,
                    "action": "workflow_complete",
                    "evidence_fields": ["forecast_id", "approval_decision", "approver_id", "variance_pct", "justification"],
                    "override_ledger": True
                },
                compliance_tags=["governance", "audit_trail", "cro_trust", "sox_compliance"]
            )
            
            self.logger.info(f"Created SaaS Forecast Approval template: {workflow.workflow_id}")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to create SaaS forecast approval template: {e}")
            raise
    
    async def _validate_block_compliance(self, block: WorkflowBlock):
        """
        Task 12.1-T36: Build policy guardrails at authoring time
        Block non-compliant blocks - must fail closed
        """
        try:
            # Check if governance tags are present for sensitive operations
            if block.type in [BlockType.ACTION, BlockType.DECISION]:
                if not block.compliance_tags:
                    raise ValueError(f"Block {block.id} of type {block.type} requires compliance tags")
            
            # Validate SaaS-specific compliance requirements
            if "revenue_ops" in block.compliance_tags:
                if "governance" not in [tag for tag in block.compliance_tags]:
                    # Add governance requirement for revenue ops blocks
                    block.compliance_tags.append("audit_trail")
            
            # Policy-aware validation - fail closed approach
            if block.type == BlockType.GOVERNANCE:
                required_fields = ["policy_id", "evidence_required"]
                for field in required_fields:
                    if field not in block.params:
                        raise ValueError(f"Governance block {block.id} missing required field: {field}")
            
            self.logger.debug(f"Block {block.id} passed compliance validation")
            
        except Exception as e:
            self.logger.error(f"Block compliance validation failed: {e}")
            raise
    
    async def _store_workflow(self, workflow: RBAWorkflow):
        """Store workflow in database with tenant isolation"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                query = """
                INSERT INTO rba_workflows (
                    workflow_id, name, description, industry, automation_type, 
                    version, state, tenant_id, created_by, created_at, 
                    blocks, governance, sla_config
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """
                # Convert blocks to JSON-serializable format
                blocks_json = []
                for block in workflow.blocks:
                    block_dict = {
                        'id': block.id,
                        'type': block.type.value,  # Convert enum to string
                        'params': block.params,
                        'overlay_refs': block.overlay_refs or [],
                        'compliance_tags': block.compliance_tags or [],
                        'metadata': block.metadata or {}
                    }
                    blocks_json.append(block_dict)
                
                await conn.execute(
                    query,
                    workflow.workflow_id, workflow.name, workflow.description,
                    workflow.industry, workflow.automation_type, workflow.version,
                    workflow.state.value, workflow.tenant_id, workflow.created_by,
                    workflow.created_at, json.dumps(blocks_json),
                    json.dumps(workflow.governance), json.dumps(workflow.sla_config)
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store workflow: {e}")
            raise
    
    async def _get_workflow(self, workflow_id: str) -> RBAWorkflow:
        """Retrieve workflow from database"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                query = "SELECT * FROM rba_workflows WHERE workflow_id = $1"
                row = await conn.fetchrow(query, workflow_id)
                
                if not row:
                    raise ValueError(f"Workflow {workflow_id} not found")
                
                # Convert back to workflow object
                blocks = [WorkflowBlock(**block_data) for block_data in json.loads(row['blocks'])]
                
                return RBAWorkflow(
                    workflow_id=row['workflow_id'],
                    name=row['name'],
                    description=row['description'],
                    industry=row['industry'],
                    automation_type=row['automation_type'],
                    version=row['version'],
                    state=WorkflowState(row['state']),
                    blocks=blocks,
                    governance=json.loads(row['governance']),
                    sla_config=json.loads(row['sla_config']),
                    tenant_id=row['tenant_id'],
                    created_by=row['created_by'],
                    created_at=row['created_at']
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow: {e}")
            raise
    
    async def _update_workflow(self, workflow: RBAWorkflow):
        """Update workflow in database"""
        try:
            async with self.pool_manager.postgres_pool.acquire() as conn:
                query = """
                UPDATE rba_workflows SET 
                    blocks = $1, governance = $2, sla_config = $3, 
                    state = $4, updated_at = $5
                WHERE workflow_id = $6 AND tenant_id = $7
                """
                await conn.execute(
                    query,
                    json.dumps([asdict(b) for b in workflow.blocks]),
                    json.dumps(workflow.governance),
                    json.dumps(workflow.sla_config),
                    workflow.state.value,
                    datetime.utcnow(),
                    workflow.workflow_id,
                    workflow.tenant_id
                )
                
        except Exception as e:
            self.logger.error(f"Failed to update workflow: {e}")
            raise

    def to_dsl(self, workflow: RBAWorkflow) -> str:
        """
        Convert workflow to DSL YAML format
        Task 12.1-T23: Support drag-drop → auto-generate DSL code
        """
        try:
            dsl_dict = {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "module": "RBA",
                "automation_type": workflow.automation_type,
                "version": workflow.version,
                "metadata": {
                    "description": workflow.description,
                    "industry_focus": [workflow.industry],
                    "sla_tier": workflow.sla_config.get("tier", "T1"),
                    "compliance_tags": list(set([tag for block in workflow.blocks for tag in block.compliance_tags]))
                },
                "governance": workflow.governance,
                "steps": []
            }
            
            # Convert blocks to DSL steps
            for block in workflow.blocks:
                step = {
                    "id": block.id,
                    "type": block.type.value,
                    "params": block.params,
                    "governance": {
                        "compliance_tags": block.compliance_tags,
                        "evidence_capture": True
                    }
                }
                dsl_dict["steps"].append(step)
            
            return yaml.dump(dsl_dict, default_flow_style=False, sort_keys=False)
            
        except Exception as e:
            self.logger.error(f"Failed to convert workflow to DSL: {e}")
            raise
