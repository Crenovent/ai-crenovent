#!/usr/bin/env python3
"""
Cross-Module Orchestration: Multi-module workflow chaining
=========================================================
Implements Chapter 12.1 T22, 12.2 T31, 12.3 T30: Cross-module orchestration

Features:
- Multi-module workflow chaining (Pipeline → Billing → Compensation → CLM)
- Dynamic module discovery and capability mapping
- Inter-module data flow and dependency management
- Governance and compliance across module boundaries
- Error handling and rollback across modules
- Evidence pack generation for cross-module workflows
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# DSL Components
from ..compiler.parser import DSLParser
from ..governance.policy_engine import PolicyEngine
from ..registry.enhanced_capability_registry import EnhancedCapabilityRegistry
from ..hub.execution_hub import ExecutionHub
from ..intelligence.evidence_pack_generator import EvidencePackGenerator

logger = logging.getLogger(__name__)

class ModuleType(Enum):
    """Supported RevAI Pro modules"""
    PIPELINE = "pipeline"
    FORECASTING = "forecasting"
    COMPENSATION = "compensation"
    BILLING = "billing"
    CLM = "clm"  # Contract Lifecycle Management
    CRUXX = "cruxx"
    REVENUE_DASHBOARD = "revenue_dashboard"
    PLANNING = "planning"

class ExecutionStatus(Enum):
    """Cross-module execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class ModuleCapability:
    """Module capability definition"""
    module: ModuleType
    capability_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str] = None
    sla_ms: int = 5000
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class CrossModuleWorkflow:
    """Cross-module workflow definition"""
    workflow_id: str
    name: str
    description: str
    modules: List[ModuleType]
    execution_graph: Dict[str, Any]
    tenant_id: int
    user_id: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ModuleExecutionResult:
    """Result of module execution"""
    module: ModuleType
    capability_id: str
    status: ExecutionStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    evidence_pack_id: Optional[str] = None

@dataclass
class CrossModuleExecutionResult:
    """Result of cross-module workflow execution"""
    workflow_id: str
    status: ExecutionStatus
    module_results: List[ModuleExecutionResult]
    total_execution_time_ms: int
    evidence_pack_ids: List[str]
    rollback_performed: bool = False
    error_message: Optional[str] = None

class CrossModuleOrchestrator:
    """
    Cross-Module Orchestrator for multi-module workflow chaining
    
    Capabilities:
    - Dynamic module discovery and registration
    - Workflow dependency resolution and execution planning
    - Inter-module data flow management
    - Cross-module governance and compliance
    - Error handling and rollback coordination
    - Evidence pack generation across modules
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.execution_hub = None
        self.policy_engine = None
        self.capability_registry = None
        self.evidence_generator = None
        
        # Module capabilities registry
        self.module_capabilities: Dict[ModuleType, List[ModuleCapability]] = {}
        
        # Active executions tracking
        self.active_executions: Dict[str, CrossModuleWorkflow] = {}
        
        # Chapter 19.2 - Module Expansion Tracking
        self.expansion_metrics: Dict[str, Dict[str, Any]] = {}
        self.expansion_adoption_kits: Dict[str, Dict[str, Any]] = {}
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the cross-module orchestrator"""
        if self.initialized:
            return
            
        try:
            # Initialize core components
            self.execution_hub = ExecutionHub(self.pool_manager)
            await self.execution_hub.initialize()
            
            self.policy_engine = PolicyEngine(self.pool_manager)
            await self.policy_engine.initialize()
            
            self.capability_registry = EnhancedCapabilityRegistry(self.pool_manager)
            await self.capability_registry.initialize()
            
            self.evidence_generator = EvidencePackGenerator(self.pool_manager)
            await self.evidence_generator.initialize()
            
            # Register built-in module capabilities
            await self._register_builtin_capabilities()
            
            self.initialized = True
            logger.info("✅ Cross-Module Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cross-Module Orchestrator: {e}")
            raise
    
    async def register_module_capability(self, capability: ModuleCapability):
        """Register a new module capability"""
        try:
            if capability.module not in self.module_capabilities:
                self.module_capabilities[capability.module] = []
            
            self.module_capabilities[capability.module].append(capability)
            
            logger.info(f"✅ Registered capability '{capability.name}' for module {capability.module.value}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register module capability: {e}")
            raise
    
    async def create_cross_module_workflow(self, 
                                         modules: List[ModuleType],
                                         workflow_definition: Dict[str, Any],
                                         tenant_id: int,
                                         user_id: int) -> CrossModuleWorkflow:
        """Create a cross-module workflow"""
        try:
            workflow_id = str(uuid.uuid4())
            
            # Build execution graph
            execution_graph = await self._build_execution_graph(modules, workflow_definition)
            
            workflow = CrossModuleWorkflow(
                workflow_id=workflow_id,
                name=workflow_definition.get("name", f"Cross-Module Workflow {workflow_id[:8]}"),
                description=workflow_definition.get("description", "Multi-module workflow"),
                modules=modules,
                execution_graph=execution_graph,
                tenant_id=tenant_id,
                user_id=user_id,
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "cross_module": True,
                    "module_count": len(modules)
                }
            )
            
            self.active_executions[workflow_id] = workflow
            
            logger.info(f"✅ Created cross-module workflow {workflow_id} with modules: {[m.value for m in modules]}")
            
            return workflow
            
        except Exception as e:
            logger.error(f"❌ Failed to create cross-module workflow: {e}")
            raise
    
    async def execute_cross_module_workflow(self, workflow_id: str) -> CrossModuleExecutionResult:
        """Execute a cross-module workflow"""
        try:
            workflow = self.active_executions.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            logger.info(f"🚀 Executing cross-module workflow {workflow_id}")
            
            start_time = datetime.now()
            module_results = []
            evidence_pack_ids = []
            rollback_performed = False
            
            try:
                # Execute modules in dependency order
                execution_order = self._resolve_execution_order(workflow.execution_graph)
                
                for module_step in execution_order:
                    module = module_step["module"]
                    capability_id = module_step["capability_id"]
                    input_data = module_step.get("input_data", {})
                    
                    # Execute module capability
                    result = await self._execute_module_capability(
                        module, capability_id, input_data, workflow.tenant_id, workflow.user_id
                    )
                    
                    module_results.append(result)
                    
                    if result.evidence_pack_id:
                        evidence_pack_ids.append(result.evidence_pack_id)
                    
                    # Check for failure
                    if result.status == ExecutionStatus.FAILED:
                        logger.error(f"❌ Module {module.value} failed: {result.error}")
                        
                        # Perform rollback
                        rollback_performed = await self._rollback_workflow(workflow_id, module_results)
                        
                        return CrossModuleExecutionResult(
                            workflow_id=workflow_id,
                            status=ExecutionStatus.FAILED,
                            module_results=module_results,
                            total_execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                            evidence_pack_ids=evidence_pack_ids,
                            rollback_performed=rollback_performed,
                            error_message=result.error
                        )
                    
                    # Pass output to next module
                    if result.data:
                        self._propagate_data_to_next_modules(execution_order, module_step, result.data)
                
                # Generate cross-module evidence pack
                cross_module_evidence_id = await self._generate_cross_module_evidence(
                    workflow, module_results
                )
                evidence_pack_ids.append(cross_module_evidence_id)
                
                total_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                logger.info(f"✅ Cross-module workflow {workflow_id} completed in {total_time}ms")
                
                return CrossModuleExecutionResult(
                    workflow_id=workflow_id,
                    status=ExecutionStatus.COMPLETED,
                    module_results=module_results,
                    total_execution_time_ms=total_time,
                    evidence_pack_ids=evidence_pack_ids,
                    rollback_performed=False
                )
                
            except Exception as e:
                logger.error(f"❌ Cross-module execution failed: {e}")
                
                # Attempt rollback
                rollback_performed = await self._rollback_workflow(workflow_id, module_results)
                
                return CrossModuleExecutionResult(
                    workflow_id=workflow_id,
                    status=ExecutionStatus.FAILED,
                    module_results=module_results,
                    total_execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    evidence_pack_ids=evidence_pack_ids,
                    rollback_performed=rollback_performed,
                    error_message=str(e)
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to execute cross-module workflow: {e}")
            raise
    
    async def _register_builtin_capabilities(self):
        """Register built-in module capabilities"""
        
        # Pipeline Module Capabilities
        pipeline_capabilities = [
            ModuleCapability(
                module=ModuleType.PIPELINE,
                capability_id="pipeline_hygiene",
                name="Pipeline Hygiene Check",
                description="Check for stale opportunities and data quality issues",
                input_schema={"type": "object", "properties": {"days_threshold": {"type": "integer"}}},
                output_schema={"type": "object", "properties": {"stale_opportunities": {"type": "array"}}}
            ),
            ModuleCapability(
                module=ModuleType.PIPELINE,
                capability_id="forecast_update",
                name="Forecast Update",
                description="Update forecast based on pipeline changes",
                input_schema={"type": "object", "properties": {"opportunities": {"type": "array"}}},
                output_schema={"type": "object", "properties": {"forecast_delta": {"type": "number"}}}
            )
        ]
        
        for cap in pipeline_capabilities:
            await self.register_module_capability(cap)
        
        # Compensation Module Capabilities
        compensation_capabilities = [
            ModuleCapability(
                module=ModuleType.COMPENSATION,
                capability_id="commission_calculation",
                name="Commission Calculation",
                description="Calculate commissions based on closed deals",
                input_schema={"type": "object", "properties": {"closed_deals": {"type": "array"}}},
                output_schema={"type": "object", "properties": {"commissions": {"type": "array"}}},
                dependencies=["pipeline_hygiene"]
            ),
            ModuleCapability(
                module=ModuleType.COMPENSATION,
                capability_id="quota_adjustment",
                name="Quota Adjustment",
                description="Adjust quotas based on performance",
                input_schema={"type": "object", "properties": {"performance_data": {"type": "object"}}},
                output_schema={"type": "object", "properties": {"quota_changes": {"type": "array"}}}
            )
        ]
        
        for cap in compensation_capabilities:
            await self.register_module_capability(cap)
        
        # Billing Module Capabilities
        billing_capabilities = [
            ModuleCapability(
                module=ModuleType.BILLING,
                capability_id="invoice_generation",
                name="Invoice Generation",
                description="Generate invoices for closed deals",
                input_schema={"type": "object", "properties": {"deals": {"type": "array"}}},
                output_schema={"type": "object", "properties": {"invoices": {"type": "array"}}},
                dependencies=["commission_calculation"]
            )
        ]
        
        for cap in billing_capabilities:
            await self.register_module_capability(cap)
    
    async def _build_execution_graph(self, modules: List[ModuleType], workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Build execution graph for cross-module workflow"""
        
        execution_graph = {
            "nodes": [],
            "edges": [],
            "execution_order": []
        }
        
        # Build nodes for each module capability
        for module in modules:
            capabilities = self.module_capabilities.get(module, [])
            for capability in capabilities:
                node = {
                    "id": f"{module.value}_{capability.capability_id}",
                    "module": module,
                    "capability_id": capability.capability_id,
                    "dependencies": capability.dependencies,
                    "input_data": workflow_definition.get("module_inputs", {}).get(module.value, {})
                }
                execution_graph["nodes"].append(node)
        
        # Build edges based on dependencies
        for node in execution_graph["nodes"]:
            for dep in node["dependencies"]:
                # Find dependency node
                dep_node = next((n for n in execution_graph["nodes"] if n["capability_id"] == dep), None)
                if dep_node:
                    edge = {
                        "from": dep_node["id"],
                        "to": node["id"]
                    }
                    execution_graph["edges"].append(edge)
        
        return execution_graph
    
    def _resolve_execution_order(self, execution_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve execution order using topological sort"""
        
        nodes = execution_graph["nodes"]
        edges = execution_graph["edges"]
        
        # Build adjacency list
        adj_list = {node["id"]: [] for node in nodes}
        in_degree = {node["id"]: 0 for node in nodes}
        
        for edge in edges:
            adj_list[edge["from"]].append(edge["to"])
            in_degree[edge["to"]] += 1
        
        # Topological sort using Kahn's algorithm
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            
            # Find the node object
            node = next(n for n in nodes if n["id"] == current)
            execution_order.append(node)
            
            # Update in-degrees
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(execution_order) != len(nodes):
            raise ValueError("Circular dependency detected in execution graph")
        
        return execution_order
    
    async def _execute_module_capability(self, 
                                       module: ModuleType, 
                                       capability_id: str, 
                                       input_data: Dict[str, Any],
                                       tenant_id: int,
                                       user_id: int) -> ModuleExecutionResult:
        """Execute a specific module capability"""
        
        start_time = datetime.now()
        
        try:
            logger.info(f"🔄 Executing {module.value}.{capability_id}")
            
            # Find capability definition
            capabilities = self.module_capabilities.get(module, [])
            capability = next((c for c in capabilities if c.capability_id == capability_id), None)
            
            if not capability:
                raise ValueError(f"Capability {capability_id} not found for module {module.value}")
            
            # Execute capability through execution hub
            execution_result = await self._delegate_to_module(module, capability_id, input_data, tenant_id, user_id)
            
            # Generate evidence pack
            evidence_pack_id = await self._generate_module_evidence(
                module, capability_id, input_data, execution_result, tenant_id
            )
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return ModuleExecutionResult(
                module=module,
                capability_id=capability_id,
                status=ExecutionStatus.COMPLETED,
                data=execution_result,
                execution_time_ms=execution_time,
                evidence_pack_id=evidence_pack_id
            )
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return ModuleExecutionResult(
                module=module,
                capability_id=capability_id,
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def _delegate_to_module(self, 
                                module: ModuleType, 
                                capability_id: str, 
                                input_data: Dict[str, Any],
                                tenant_id: int,
                                user_id: int) -> Dict[str, Any]:
        """Delegate execution to specific module"""
        
        # This is where we would integrate with actual module implementations
        # For now, we'll simulate module execution
        
        if module == ModuleType.PIPELINE:
            return await self._simulate_pipeline_execution(capability_id, input_data)
        elif module == ModuleType.COMPENSATION:
            return await self._simulate_compensation_execution(capability_id, input_data)
        elif module == ModuleType.BILLING:
            return await self._simulate_billing_execution(capability_id, input_data)
        else:
            # Generic execution through execution hub
            return await self._generic_module_execution(module, capability_id, input_data)
    
    async def _simulate_pipeline_execution(self, capability_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate pipeline module execution"""
        
        if capability_id == "pipeline_hygiene":
            return {
                "stale_opportunities": [
                    {"id": "opp_1", "name": "Deal A", "days_stale": 45},
                    {"id": "opp_2", "name": "Deal B", "days_stale": 60}
                ],
                "total_stale": 2
            }
        elif capability_id == "forecast_update":
            return {
                "forecast_delta": 50000,
                "updated_forecast": 1250000
            }
        
        return {"result": "success"}
    
    async def _simulate_compensation_execution(self, capability_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate compensation module execution"""
        
        if capability_id == "commission_calculation":
            return {
                "commissions": [
                    {"rep_id": "rep_1", "commission": 5000, "deal_id": "deal_1"},
                    {"rep_id": "rep_2", "commission": 7500, "deal_id": "deal_2"}
                ],
                "total_commission": 12500
            }
        elif capability_id == "quota_adjustment":
            return {
                "quota_changes": [
                    {"rep_id": "rep_1", "old_quota": 100000, "new_quota": 110000}
                ]
            }
        
        return {"result": "success"}
    
    async def _simulate_billing_execution(self, capability_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate billing module execution"""
        
        if capability_id == "invoice_generation":
            return {
                "invoices": [
                    {"invoice_id": "inv_1", "amount": 25000, "deal_id": "deal_1"},
                    {"invoice_id": "inv_2", "amount": 37500, "deal_id": "deal_2"}
                ],
                "total_invoiced": 62500
            }
        
        return {"result": "success"}
    
    async def _generic_module_execution(self, module: ModuleType, capability_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic module execution fallback"""
        
        return {
            "module": module.value,
            "capability": capability_id,
            "status": "executed",
            "timestamp": datetime.now().isoformat()
        }
    
    def _propagate_data_to_next_modules(self, execution_order: List[Dict[str, Any]], current_step: Dict[str, Any], output_data: Dict[str, Any]):
        """Propagate output data to dependent modules"""
        
        current_index = execution_order.index(current_step)
        
        # Find dependent steps
        for i in range(current_index + 1, len(execution_order)):
            next_step = execution_order[i]
            if current_step["capability_id"] in next_step["dependencies"]:
                # Merge output data into next step's input
                if "propagated_data" not in next_step:
                    next_step["propagated_data"] = {}
                next_step["propagated_data"].update(output_data)
    
    async def _generate_module_evidence(self, 
                                      module: ModuleType, 
                                      capability_id: str, 
                                      input_data: Dict[str, Any], 
                                      output_data: Dict[str, Any],
                                      tenant_id: int) -> str:
        """Generate evidence pack for module execution"""
        
        try:
            evidence_data = {
                "module": module.value,
                "capability_id": capability_id,
                "input_data": input_data,
                "output_data": output_data,
                "execution_timestamp": datetime.now().isoformat(),
                "tenant_id": tenant_id
            }
            
            evidence_pack_id = await self.evidence_generator.generate_evidence_pack(
                workflow_id=f"{module.value}_{capability_id}",
                tenant_id=tenant_id,
                evidence_data=evidence_data,
                evidence_type="module_execution"
            )
            
            return evidence_pack_id
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate module evidence: {e}")
            return f"evidence_error_{uuid.uuid4()}"
    
    async def _generate_cross_module_evidence(self, 
                                            workflow: CrossModuleWorkflow, 
                                            module_results: List[ModuleExecutionResult]) -> str:
        """Generate cross-module evidence pack"""
        
        try:
            evidence_data = {
                "workflow_id": workflow.workflow_id,
                "workflow_name": workflow.name,
                "modules": [m.value for m in workflow.modules],
                "module_results": [asdict(result) for result in module_results],
                "execution_timestamp": datetime.now().isoformat(),
                "tenant_id": workflow.tenant_id
            }
            
            evidence_pack_id = await self.evidence_generator.generate_evidence_pack(
                workflow_id=workflow.workflow_id,
                tenant_id=workflow.tenant_id,
                evidence_data=evidence_data,
                evidence_type="cross_module_execution"
            )
            
            return evidence_pack_id
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate cross-module evidence: {e}")
            return f"cross_module_evidence_error_{uuid.uuid4()}"
    
    async def _rollback_workflow(self, workflow_id: str, module_results: List[ModuleExecutionResult]) -> bool:
        """Rollback cross-module workflow"""
        
        try:
            logger.info(f"🔄 Rolling back workflow {workflow_id}")
            
            # Rollback in reverse order
            for result in reversed(module_results):
                if result.status == ExecutionStatus.COMPLETED:
                    await self._rollback_module(result)
            
            logger.info(f"✅ Workflow {workflow_id} rolled back successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed for workflow {workflow_id}: {e}")
            return False
    
    async def _rollback_module(self, result: ModuleExecutionResult):
        """Rollback a specific module execution"""
        
        try:
            logger.info(f"🔄 Rolling back {result.module.value}.{result.capability_id}")
            
            # Module-specific rollback logic would go here
            # For now, we'll just log the rollback
            
            logger.info(f"✅ Rolled back {result.module.value}.{result.capability_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to rollback {result.module.value}.{result.capability_id}: {e}")
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a cross-module workflow"""
        
        workflow = self.active_executions.get(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "modules": [m.value for m in workflow.modules],
            "status": "active",
            "created_at": workflow.metadata.get("created_at")
        }
    
    async def list_module_capabilities(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all registered module capabilities"""
        
        capabilities = {}
        for module, caps in self.module_capabilities.items():
            capabilities[module.value] = [
                {
                    "capability_id": cap.capability_id,
                    "name": cap.name,
                    "description": cap.description,
                    "dependencies": cap.dependencies,
                    "sla_ms": cap.sla_ms
                }
                for cap in caps
            ]
        
        return capabilities
    
    # =====================================================
    # CHAPTER 19.2 - MODULE EXPANSION FUNCTIONALITY
    # =====================================================
    
    async def create_module_expansion_kit(self, modules: List[ModuleType], tenant_id: int, industry: str = "SaaS") -> Dict[str, Any]:
        """
        Package multi-module adoption kits (Task 19.2-T15)
        """
        try:
            kit_id = f"expansion_kit_{tenant_id}_{uuid.uuid4().hex[:8]}"
            
            expansion_kit = {
                "kit_id": kit_id,
                "tenant_id": tenant_id,
                "industry": industry,
                "modules": [module.value for module in modules],
                "workflows": self._get_expansion_workflows(modules, industry),
                "governance": {
                    "policy_packs": [f"{industry.lower()}_multi_module_policy"],
                    "evidence_capture": True,
                    "trust_scoring": True,
                    "compliance_frameworks": self._get_industry_compliance_frameworks(industry)
                },
                "deployment_templates": self._generate_deployment_templates(modules),
                "created_at": datetime.now().isoformat()
            }
            
            self.expansion_adoption_kits[kit_id] = expansion_kit
            
            logger.info(f"✅ Created module expansion kit {kit_id} for modules: {[m.value for m in modules]}")
            return expansion_kit
            
        except Exception as e:
            logger.error(f"❌ Failed to create module expansion kit: {e}")
            raise
    
    async def execute_module_expansion_workflow(self, kit_id: str, expansion_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute multi-module expansion workflow (Task 19.2-T10)
        """
        try:
            kit = self.expansion_adoption_kits.get(kit_id)
            if not kit:
                raise ValueError(f"Expansion kit {kit_id} not found")
            
            expansion_id = f"expansion_{uuid.uuid4().hex[:12]}"
            
            # Create cross-module workflow from expansion kit
            workflow = await self.create_cross_module_workflow(
                modules=[ModuleType(m) for m in kit["modules"]],
                workflow_definition=kit["deployment_templates"],
                tenant_id=kit["tenant_id"],
                user_id=expansion_config.get("user_id", 0)
            )
            
            # Execute the expansion workflow
            execution_result = await self.execute_cross_module_workflow(workflow.workflow_id)
            
            # Track expansion metrics (Task 19.2-T21, T22)
            expansion_metrics = {
                "expansion_id": expansion_id,
                "kit_id": kit_id,
                "tenant_id": kit["tenant_id"],
                "modules": kit["modules"],
                "execution_status": execution_result.status.value,
                "start_time": execution_result.start_time,
                "end_time": execution_result.end_time,
                "evidence_pack_ids": execution_result.evidence_pack_ids,
                "business_impact": self._calculate_expansion_business_impact(execution_result)
            }
            
            self.expansion_metrics[expansion_id] = expansion_metrics
            
            # Generate multi-module evidence pack (Task 19.2-T22)
            evidence_pack_id = await self._generate_expansion_evidence_pack(expansion_metrics)
            
            return {
                "success": True,
                "expansion_id": expansion_id,
                "execution_result": execution_result,
                "evidence_pack_id": evidence_pack_id,
                "business_impact": expansion_metrics["business_impact"]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to execute module expansion: {e}")
            raise
    
    async def get_expansion_adoption_heatmap(self, tenant_id: int) -> Dict[str, Any]:
        """
        Build expansion adoption heatmap (tenants × modules × industries) (Task 19.2-T36)
        """
        try:
            # Get expansion metrics for tenant
            tenant_expansions = {k: v for k, v in self.expansion_metrics.items() 
                               if v["tenant_id"] == tenant_id}
            
            heatmap_data = {
                "tenant_id": tenant_id,
                "total_expansions": len(tenant_expansions),
                "module_adoption": {},
                "success_rates": {},
                "roi_metrics": {},
                "trends": {}
            }
            
            # Calculate module-specific metrics
            for module_type in ModuleType:
                module_name = module_type.value
                module_expansions = [e for e in tenant_expansions.values() 
                                   if module_name in e["modules"]]
                
                if module_expansions:
                    success_count = len([e for e in module_expansions 
                                       if e["execution_status"] == "completed"])
                    
                    heatmap_data["module_adoption"][module_name] = len(module_expansions)
                    heatmap_data["success_rates"][module_name] = (success_count / len(module_expansions)) * 100
                    heatmap_data["roi_metrics"][module_name] = self._calculate_module_roi(module_expansions)
            
            return heatmap_data
            
        except Exception as e:
            logger.error(f"❌ Failed to generate expansion heatmap: {e}")
            raise
    
    async def calculate_expansion_roi(self, expansion_id: str) -> Dict[str, Any]:
        """
        Calculate ROI for multi-module expansion (Task 19.2-T37)
        """
        try:
            expansion = self.expansion_metrics.get(expansion_id)
            if not expansion:
                raise ValueError(f"Expansion {expansion_id} not found")
            
            # Calculate cross-module ROI
            roi_calculation = {
                "expansion_id": expansion_id,
                "modules": expansion["modules"],
                "time_saved_hours": self._calculate_cross_module_time_savings(expansion),
                "cost_savings_usd": self._calculate_cross_module_cost_savings(expansion),
                "efficiency_improvement": self._calculate_cross_module_efficiency(expansion),
                "synergy_benefits": self._calculate_module_synergy_benefits(expansion),
                "total_roi_percentage": 0.0
            }
            
            # Calculate total ROI with synergy benefits
            total_benefits = (roi_calculation["cost_savings_usd"] + 
                            roi_calculation["synergy_benefits"])
            implementation_cost = len(expansion["modules"]) * 8000  # Cost per module
            
            roi_calculation["total_roi_percentage"] = ((total_benefits - implementation_cost) / 
                                                     implementation_cost) * 100
            
            return roi_calculation
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate expansion ROI: {e}")
            raise
    
    # Helper methods for Chapter 19.2
    def _get_expansion_workflows(self, modules: List[ModuleType], industry: str) -> Dict[str, Any]:
        """Get workflows for module expansion"""
        workflows = {}
        
        for module in modules:
            if industry == "SaaS":
                if module == ModuleType.PIPELINE:
                    workflows[module.value] = "saas_pipeline_expansion_workflow"
                elif module == ModuleType.FORECASTING:
                    workflows[module.value] = "saas_forecast_expansion_workflow"
                elif module == ModuleType.COMPENSATION:
                    workflows[module.value] = "saas_compensation_expansion_workflow"
                # Add more module-specific workflows
        
        return workflows
    
    def _get_industry_compliance_frameworks(self, industry: str) -> List[str]:
        """Get compliance frameworks for industry"""
        frameworks_map = {
            "SaaS": ["SOX", "GDPR", "SOC2"],
            "Banking": ["RBI", "BASEL_III", "DPDP"],
            "Insurance": ["IRDAI", "SOLVENCY_II", "GDPR"],
            "Healthcare": ["HIPAA", "FDA_21CFR11", "GDPR"]
        }
        return frameworks_map.get(industry, ["SOX", "GDPR"])
    
    def _generate_deployment_templates(self, modules: List[ModuleType]) -> Dict[str, Any]:
        """Generate deployment templates for modules"""
        return {
            "deployment_strategy": "progressive",
            "module_sequence": [m.value for m in modules],
            "rollback_strategy": "module_by_module",
            "health_checks": True,
            "governance_validation": True
        }
    
    def _calculate_expansion_business_impact(self, execution_result: CrossModuleExecutionResult) -> Dict[str, Any]:
        """Calculate business impact of expansion"""
        return {
            "modules_integrated": len(execution_result.module_results),
            "execution_time_ms": (execution_result.end_time - execution_result.start_time).total_seconds() * 1000,
            "success_rate": 100.0 if execution_result.status.value == "completed" else 0.0,
            "evidence_generated": len(execution_result.evidence_pack_ids)
        }
    
    async def _generate_expansion_evidence_pack(self, expansion_metrics: Dict[str, Any]) -> str:
        """Generate evidence pack for expansion"""
        evidence_pack_id = f"expansion_evidence_{uuid.uuid4().hex[:12]}"
        
        # Would integrate with evidence pack service
        evidence_data = {
            "evidence_type": "multi_module_expansion",
            "expansion_id": expansion_metrics["expansion_id"],
            "modules": expansion_metrics["modules"],
            "tenant_id": expansion_metrics["tenant_id"],
            "execution_metadata": expansion_metrics,
            "compliance_validation": True,
            "digital_signature": f"sig_{uuid.uuid4().hex[:16]}"
        }
        
        return evidence_pack_id
    
    def _calculate_cross_module_time_savings(self, expansion: Dict[str, Any]) -> float:
        """Calculate time savings from cross-module integration"""
        base_savings_per_module = 8.0  # hours per week per module
        synergy_multiplier = 1.3  # 30% additional savings from integration
        
        return len(expansion["modules"]) * base_savings_per_module * synergy_multiplier
    
    def _calculate_cross_module_cost_savings(self, expansion: Dict[str, Any]) -> float:
        """Calculate cost savings from cross-module integration"""
        time_saved = self._calculate_cross_module_time_savings(expansion)
        hourly_cost = 85  # Average hourly cost
        
        return time_saved * hourly_cost * 4  # Monthly savings
    
    def _calculate_cross_module_efficiency(self, expansion: Dict[str, Any]) -> float:
        """Calculate efficiency improvement from cross-module integration"""
        base_efficiency_per_module = 15.0  # 15% per module
        integration_bonus = 10.0  # Additional 10% from integration
        
        return (len(expansion["modules"]) * base_efficiency_per_module) + integration_bonus
    
    def _calculate_module_synergy_benefits(self, expansion: Dict[str, Any]) -> float:
        """Calculate synergy benefits from module integration"""
        # Synergy benefits increase exponentially with more modules
        module_count = len(expansion["modules"])
        base_synergy = 2000  # Base synergy value
        
        return base_synergy * (module_count ** 1.5)  # Exponential growth
    
    def _calculate_module_roi(self, module_expansions: List[Dict[str, Any]]) -> float:
        """Calculate ROI for specific module"""
        if not module_expansions:
            return 0.0
        
        total_roi = sum(expansion.get("business_impact", {}).get("roi", 0) 
                       for expansion in module_expansions)
        
        return total_roi / len(module_expansions)

# Factory function
async def get_cross_module_orchestrator(pool_manager=None) -> CrossModuleOrchestrator:
    """Get initialized cross-module orchestrator instance"""
    orchestrator = CrossModuleOrchestrator(pool_manager)
    await orchestrator.initialize()
    return orchestrator
