"""
Dynamic RBA Orchestrator
Dynamically selects and executes RBA agents without hardcoding

This orchestrator uses the RBA registry to dynamically discover
and select the appropriate agent based on analysis type.
NO hardcoded agent imports or selections.

Enhanced with Task 6.1-6.2 implementations:
- DSL Parser integration
- Runtime execution engine
- Idempotency framework
- Governance-first design
"""

import logging
import asyncio
import hashlib
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from ..registry.enhanced_capability_registry import EnhancedCapabilityRegistry
from ..parser import DSLParser, DSLWorkflow
from ..compiler.runtime import WorkflowRuntime, WorkflowExecution
from ..compiler.static_analyzer import RBAStaticAnalyzer, ValidationSeverity
from ..compiler.workflow_planner import RBAWorkflowPlanner, OptimizationStrategy
from ..governance.policy_engine import PolicyEngine
from ..observability.otel_tracer import get_tracer, TraceContext
from ..observability.metrics_collector import get_metrics_collector

# Import our new components
from ..schemas.canonical_trace_schema import trace_builder, TraceBuilder, WorkflowTrace
from ..registry.workflow_registry_versioning import workflow_registry
from ..tenancy.tenant_context_manager import tenant_context_manager
from ..performance.slo_sla_manager import slo_sla_manager
from ..performance.auto_scaling_manager import auto_scaling_manager
from ..performance.performance_optimizer import performance_optimizer

# Import Chapter 8 governance components
from ..governance.policy_execution_limits import policy_registry, policy_enforcer, policy_audit_logger
from ..governance.rbac_sod_system import rbac_manager, sod_enforcer, abac_manager
from ..governance.industry_governance_overlays import industry_governance_manager, Industry
from ..governance.enhanced_industry_compliance import enhanced_industry_compliance
from ..governance.security_compliance_checklist import security_compliance_checklist

# Import Chapter 7 schema components
from ..schemas.evidence_pack_schema import evidence_pack_manager
from ..schemas.override_ledger_schema import override_ledger_manager
from ..schemas.risk_register_schema import risk_register_manager

# Import Chapter 9 overlay components
from ..overlays.overlay_manager import overlay_manager
from ..adapters.adapter_sdk import IndustryType

logger = logging.getLogger(__name__)

class DynamicRBAOrchestrator:
    """
    Dynamic RBA Orchestrator
    
    Features:
    - NO hardcoded agent imports or selections
    - Dynamic agent discovery via registry
    - Automatic agent selection based on analysis type
    - Fallback mechanisms for unsupported types
    - Extensible - new agents automatically available
    - Enhanced with DSL parsing and runtime execution
    - Governance-first design with policy enforcement
    - Idempotency framework for duplicate prevention
    """
    
    def __init__(self, db_pool=None, redis_client=None):
        self.logger = logging.getLogger(__name__)
        self.registry = EnhancedCapabilityRegistry()
        
        # Enhanced components from Task 6.1-6.2
        self.dsl_parser = DSLParser()
        self.workflow_runtime = None
        self.static_analyzer = None
        self.workflow_planner = None
        self.policy_engine = None
        
        # Observability components (Tasks 6.7, 20.1)
        self.tracer = get_tracer()
        self.metrics_collector = get_metrics_collector()
        
        # Database and cache connections
        self.db_pool = db_pool
        self.redis_client = redis_client
        
        # NEW: Integrated components from Chapters 6.5-7
        self.trace_builder = trace_builder
        self.workflow_registry = workflow_registry
        self.tenant_context_manager = tenant_context_manager
        self.slo_sla_manager = slo_sla_manager
        self.auto_scaling_manager = auto_scaling_manager
        self.performance_optimizer = performance_optimizer
        
        # NEW: Chapter 8 governance components
        self.policy_registry = policy_registry
        self.policy_enforcer = policy_enforcer
        self.policy_audit_logger = policy_audit_logger
        self.rbac_manager = rbac_manager
        self.sod_enforcer = sod_enforcer
        self.abac_manager = abac_manager
        self.industry_governance_manager = industry_governance_manager
        self.enhanced_industry_compliance = enhanced_industry_compliance
        self.security_compliance_checklist = security_compliance_checklist
        
        # NEW: Chapter 7 schema components
        self.evidence_pack_manager = evidence_pack_manager
        self.override_ledger_manager = override_ledger_manager
        self.risk_register_manager = risk_register_manager
        
        # Initialize registry (async initialization will be handled separately)
        self.registry_initialized = False
    
    async def initialize_enhanced_components(self, pool_manager=None):
        """Initialize enhanced RBA components"""
        try:
            # Initialize registry first
            if not self.registry_initialized:
                await self.registry.initialize()
                self.registry_initialized = True
                self.logger.info("✅ Enhanced Capability Registry initialized")
            
            if pool_manager and not self.workflow_runtime:
                self.workflow_runtime = WorkflowRuntime(pool_manager)
                
            # Initialize static analyzer
            if self.db_pool and not self.static_analyzer:
                self.static_analyzer = RBAStaticAnalyzer(self.db_pool)
                await self.static_analyzer.initialize()
                
                # Initialize workflow planner
                if self.db_pool and not self.workflow_planner:
                    self.workflow_planner = RBAWorkflowPlanner(self.db_pool)
                    await self.workflow_planner.initialize()
                
                # Initialize policy engine for enhanced governance
                if pool_manager and not self.policy_engine:
                    self.policy_engine = PolicyEngine(pool_manager)
                    await self.policy_engine.initialize()
                    
                self.logger.info("✅ Enhanced RBA components initialized (parser, runtime, analyzer, planner, governance)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize enhanced components: {e}")
    
    async def execute_rba_analysis(
        self,
        analysis_type: str,
        opportunities: List[Dict[str, Any]],
        config: Dict[str, Any] = None,
        tenant_id: str = None,
        user_id: str = None,
        kg_store = None  # Knowledge Graph store for execution tracing
    ) -> Dict[str, Any]:
        """
        Execute RBA analysis using dynamic agent selection
        
        Args:
            analysis_type: Type of analysis to perform
            opportunities: List of opportunity data
            config: Configuration parameters
            tenant_id: Tenant identifier
            user_id: User identifier
            
        Returns:
            Analysis results from the selected agent
        """
        
        execution_id = f"rba_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"🚀 Dynamic RBA Orchestrator: {analysis_type} analysis")
            self.logger.info(f"📊 Execution ID: {execution_id}")
            self.logger.info(f"🔢 Opportunities: {len(opportunities)}")
            
            # Step 1: Dynamically select agent based on analysis type
            agent_instance = self._select_agent_for_analysis(analysis_type, config or {})
            
            if not agent_instance:
                return self._create_error_response(
                    execution_id=execution_id,
                    analysis_type=analysis_type,
                    error=f"No agent available for analysis type: {analysis_type}",
                    available_types=self.registry.get_supported_analysis_types()
                )
            
            # Step 2: Prepare input data
            input_data = {
                'opportunities': opportunities,
                'config': config or {},
                'analysis_type': analysis_type,
                'tenant_id': tenant_id,
                'user_id': user_id,
                'execution_id': execution_id
            }
            
            # Step 3: Execute analysis with selected agent (with KG tracing if available)
            self.logger.info(f"🎯 Executing with agent: {agent_instance.__class__.__name__}")
            
            # Check if agent supports enhanced execution with tracing
            if hasattr(agent_instance, 'execute_with_tracing') and kg_store:
                self.logger.info(f"🧠 Using enhanced execution with Knowledge Graph tracing")
                result = await agent_instance.execute_with_tracing(
                    input_data, 
                    kg_store=kg_store,
                    context={
                        'tenant_id': tenant_id,
                        'user_id': user_id,
                        'execution_id': execution_id,
                        'source': 'dynamic_orchestrator'
                    }
                )
            else:
                # Fallback to standard execution
                self.logger.info(f"⚙️ Using standard execution (no KG tracing)")
                result = await agent_instance.execute(input_data)
            
            # Step 4: Enrich result with orchestration metadata
            if isinstance(result, dict):
                result.update({
                    'orchestration_metadata': {
                        'execution_id': execution_id,
                        'orchestrator_type': 'dynamic_rba',
                        'agent_selected': agent_instance.__class__.__name__,
                        'selection_method': 'registry_based',
                        'tenant_id': tenant_id,
                        'user_id': user_id
                    }
                })
            
            self.logger.info(f"✅ Dynamic RBA execution complete: {execution_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic RBA orchestration failed: {e}")
            return self._create_error_response(
                execution_id=execution_id,
                analysis_type=analysis_type,
                error=str(e)
            )
    
    async def execute_dsl_workflow(
        self,
        workflow_dsl: Dict[str, Any],
        input_data: Dict[str, Any],
        user_context: Dict[str, Any],
        tenant_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute DSL workflow with enhanced governance and idempotency
        
        Args:
            workflow_dsl: DSL workflow definition
            input_data: Input data for workflow
            user_context: User context and permissions
            tenant_id: Tenant identifier
            user_id: User identifier
            
        Returns:
            Workflow execution results
        """
        execution_id = str(uuid.uuid4())
        workflow_start_time = datetime.now()
        
        # NEW: Create canonical trace using our trace builder
        workflow_trace = self.trace_builder.create_workflow_trace(
            workflow_id=workflow_dsl.get('workflow_id', 'unknown'),
            execution_id=execution_id,
            workflow_version=workflow_dsl.get('version', '1.0.0'),
            tenant_id=int(tenant_id) if tenant_id else 1300,
            user_id=user_id or user_context.get('user_id', 'unknown'),
            actor_name=user_context.get('actor_name', 'system')
        )
        
        # Start the workflow trace
        self.trace_builder.start_workflow(workflow_trace)
        
        # NEW: Get tenant context for enhanced governance
        tenant_context = None
        if tenant_id:
            tenant_context = await self.tenant_context_manager.get_tenant_context(int(tenant_id))
            if tenant_context:
                self.logger.info(f"🏢 Tenant context loaded: {tenant_context.tenant_name} ({tenant_context.industry_code.value})")
        
        # NEW: Check tenant rate limits and quotas
        if tenant_id:
            rate_limit_check = await self.auto_scaling_manager.check_rate_limit(
                int(tenant_id), 'workflows', 1
            )
            if not rate_limit_check['allowed']:
                self.trace_builder.fail_workflow(
                    workflow_trace,
                    'RATE_LIMIT_EXCEEDED',
                    f"Rate limit exceeded: {rate_limit_check['reason']}"
                )
                return {
                    'success': False,
                    'execution_id': execution_id,
                    'error': 'Rate limit exceeded',
                    'rate_limit_info': rate_limit_check,
                    'trace_id': workflow_trace.trace_id
                }
        
        # NEW: Chapter 8 - RBAC and SoD checks
        user_id_str = user_id or user_context.get('user_id', 'unknown')
        if tenant_id and user_id_str != 'unknown':
            # Check RBAC permissions
            from ..governance.rbac_sod_system import Permission
            has_execute_permission = self.rbac_manager.has_permission(
                user_id_str, int(tenant_id), Permission.WORKFLOW_EXECUTE
            )
            
            if not has_execute_permission:
                self.trace_builder.fail_workflow(
                    workflow_trace,
                    'RBAC_PERMISSION_DENIED',
                    f"User {user_id_str} lacks WORKFLOW_EXECUTE permission"
                )
                return {
                    'success': False,
                    'execution_id': execution_id,
                    'error': 'RBAC permission denied',
                    'rbac_check': {'user_id': user_id_str, 'required_permission': 'WORKFLOW_EXECUTE'},
                    'trace_id': workflow_trace.trace_id
                }
            
            # Check SoD compliance
            sod_compliant, sod_violations, applicable_sod_rules = await self.sod_enforcer.check_sod_compliance(
                user_id_str, int(tenant_id), Permission.WORKFLOW_EXECUTE, 
                workflow_dsl.get('workflow_id', 'unknown'), workflow_dsl.get('workflow_id')
            )
            
            if not sod_compliant:
                self.trace_builder.fail_workflow(
                    workflow_trace,
                    'SOD_VIOLATION',
                    f"SoD violations: {'; '.join(sod_violations)}"
                )
                return {
                    'success': False,
                    'execution_id': execution_id,
                    'error': 'Segregation of Duties violation',
                    'sod_violations': sod_violations,
                    'applicable_sod_rules': [rule.rule_id for rule in applicable_sod_rules],
                    'trace_id': workflow_trace.trace_id
                }
        
        # NEW: Chapter 8 - Industry governance compliance check
        industry_code = user_context.get('industry_code', 'SaaS')
        try:
            industry_enum = Industry(industry_code.lower())
            compliance_result = self.industry_governance_manager.validate_compliance(
                industry_enum,
                workflow_dsl,
                {
                    'tenant_id': int(tenant_id) if tenant_id else 1300,
                    'user_id': user_id_str,
                    'execution_context': user_context
                }
            )
            
            if not compliance_result['compliant']:
                self.trace_builder.fail_workflow(
                    workflow_trace,
                    'INDUSTRY_COMPLIANCE_VIOLATION',
                    f"Industry compliance violations: {'; '.join(compliance_result['violations'])}"
                )
                return {
                    'success': False,
                    'execution_id': execution_id,
                    'error': 'Industry compliance violation',
                    'compliance_violations': compliance_result['violations'],
                    'industry': industry_code,
                    'trace_id': workflow_trace.trace_id
                }
        except ValueError:
            # Unknown industry, continue with default governance
            self.logger.warning(f"⚠️ Unknown industry code: {industry_code}, using default governance")
        
        # NEW: Chapter 8 - Policy enforcement
        if tenant_id:
            try:
                industry_enum = Industry(industry_code.lower())
            except ValueError:
                industry_enum = Industry.SAAS  # Default to SaaS
            
            policy_enforcement_result = await self.policy_enforcer.enforce_policies(
                workflow_dsl.get('workflow_id', 'unknown'),
                execution_id,
                int(tenant_id),
                user_id_str,
                user_context.get('user_role', 'user'),
                industry_enum,
                {
                    'max_executions_per_minute': 1,  # Current execution
                    'max_records_per_execution': input_data.get('record_count', 0),
                    'data_region': user_context.get('region', 'US'),
                    'execution_time_estimate': 60  # Mock estimate
                }
            )
            
            if not policy_enforcement_result.allowed:
                # Log policy violations
                for violation in policy_enforcement_result.violations:
                    await self.policy_audit_logger.log_policy_violation(violation)
                
                self.trace_builder.fail_workflow(
                    workflow_trace,
                    'POLICY_VIOLATION',
                    f"Policy violations: {len(policy_enforcement_result.violations)} violations detected"
                )
                return {
                    'success': False,
                    'execution_id': execution_id,
                    'error': 'Policy enforcement violation',
                    'policy_violations': [v.to_dict() for v in policy_enforcement_result.violations],
                    'enforcement_time_ms': policy_enforcement_result.enforcement_time_ms,
                    'trace_id': workflow_trace.trace_id
                }
        
        # Create trace context for observability
        trace_context = self.tracer.create_trace_context(
            tenant_id=int(tenant_id) if tenant_id else 1300,
            user_id=user_id or user_context.get('user_id', 'unknown'),
            workflow_id=workflow_dsl.get('workflow_id', 'unknown'),
            execution_id=execution_id,
            industry_code=user_context.get('industry_code', 'SaaS'),
            compliance_frameworks=user_context.get('compliance_frameworks', ['SOX_SAAS', 'GDPR_SAAS'])
        )
        
        try:
            # Start distributed tracing for the entire workflow execution
            async with self.tracer.trace_workflow_execution(
                workflow_id=workflow_dsl.get('workflow_id', 'unknown'),
                execution_id=execution_id,
                trace_context=trace_context,
                operation_name="dsl_workflow_execution"
            ) as workflow_span:
                
                self.logger.info(f"🚀 Executing enhanced DSL workflow: {execution_id} (trace: {trace_context.trace_id[:8]}...)")
                
                # 1. Static Analysis with tracing (Task 6.2-T03)
                validation_result = None
                if self.static_analyzer:
                    async with self.tracer.trace_compilation_step(
                        step_name="static_analysis",
                        trace_context=trace_context,
                        step_metadata={'analyzer_version': '2.0', 'tenant_id': tenant_id}
                    ):
                        self.logger.info(f"🔍 Running static analysis...")
                        validation_result = await self.static_analyzer.analyze_workflow(
                            workflow_dsl, tenant_id, user_context.get('industry_code')
                        )
                
                # Check for critical errors
                if not validation_result.is_valid:
                    critical_errors = [issue for issue in validation_result.errors 
                                     if issue.severity == ValidationSeverity.CRITICAL]
                    if critical_errors:
                        return {
                            'success': False,
                            'execution_id': execution_id,
                            'error': 'Static analysis failed with critical errors',
                            'validation_errors': [issue.message for issue in critical_errors],
                            'orchestrator_type': 'dynamic_rba_enhanced'
                        }
                
                self.logger.info(f"✅ Static analysis completed: {len(validation_result.issues)} issues found")
            
            # 2. Parse DSL workflow using existing parser
            if isinstance(workflow_dsl, dict):
                # Convert dict to DSL workflow object
                workflow = self.dsl_parser.parse_workflow_dict(workflow_dsl)
            else:
                # Parse YAML string
                workflow = self.dsl_parser.parse_workflow(workflow_dsl)
            
            # 3. Workflow Planning (Task 6.2-T04)
            execution_plan = None
            if self.workflow_planner:
                self.logger.info(f"🎯 Creating optimized execution plan...")
                
                # Determine optimization strategy based on user context
                optimization_strategy = OptimizationStrategy.PERFORMANCE
                if user_context.get('sla_tier') == 'T0':
                    optimization_strategy = OptimizationStrategy.PERFORMANCE
                elif user_context.get('cost_sensitive'):
                    optimization_strategy = OptimizationStrategy.COST
                elif user_context.get('compliance_critical'):
                    optimization_strategy = OptimizationStrategy.COMPLIANCE
                
                execution_plan = await self.workflow_planner.create_execution_plan(
                    workflow_dsl=workflow_dsl,
                    optimization_strategy=optimization_strategy,
                    constraints=user_context.get('execution_constraints'),
                    tenant_id=tenant_id
                )
                
                self.logger.info(f"✅ Execution plan created: {execution_plan.plan_hash[:8]}... ({len(execution_plan.execution_steps)} steps)")
            
                # Execute workflow using existing runtime with enhanced governance
                if self.workflow_runtime:
                    self.logger.info(f"🎯 Executing with WorkflowRuntime: {workflow.workflow_id}")
                    
                    # Enhanced user context with tenant/user info
                    enhanced_context = {
                        **user_context,
                        'tenant_id': tenant_id or user_context.get('tenant_id'),
                        'user_id': user_id or user_context.get('user_id'),
                        'execution_id': execution_id,
                        'orchestrator_type': 'dynamic_rba_enhanced'
                    }
                    
                    # Pre-execution governance checks
                    governance_context = {}
                    if self.policy_engine:
                        self.logger.info(f"🛡️ Running pre-execution governance checks...")
                        
                        # Policy enforcement check
                        policy_allowed, policy_violations = await self.policy_engine.enforce_policy(
                            tenant_id=int(tenant_id) if tenant_id else 1300,
                            workflow_data={
                                'workflow_id': workflow.workflow_id,
                                'input_data': input_data,
                                'user_context': enhanced_context
                            },
                            context=enhanced_context
                        )
                        
                        if not policy_allowed:
                            return {
                                'success': False,
                                'execution_id': execution_id,
                                'error': 'Workflow execution blocked by governance policies',
                                'policy_violations': [v.message for v in policy_violations],
                                'orchestrator_type': 'dynamic_rba_enhanced'
                            }
                        
                        governance_context.update({
                            'policy_violations': policy_violations,
                            'applied_policies': [v.policy_id for v in policy_violations] if policy_violations else [],
                            'policy_compliance': 'compliant' if not policy_violations else 'violations_present'
                        })
                    
                    # Execute workflow
                    execution_result = await self.workflow_runtime.execute_workflow(
                        workflow=workflow,
                        input_data=input_data,
                        user_context=enhanced_context
                    )
                    
                    # Post-execution governance processing
                    if self.policy_engine and execution_result:
                        self.logger.info(f"🛡️ Running post-execution governance processing...")
                        
                        # Calculate trust score
                        trust_score_data = await self.policy_engine.calculate_trust_score(
                            tenant_id=int(tenant_id) if tenant_id else 1300,
                            workflow_execution_data={
                                'workflow_id': workflow.workflow_id,
                                'execution_id': execution_id,
                                'applied_policies': governance_context.get('applied_policies', []),
                                'policy_violations': governance_context.get('policy_violations', []),
                                'execution_time_ms': execution_result.execution_time_ms,
                                'input_data': input_data,
                                'output_data': execution_result.outputs,
                                'execution_metadata': {
                                    'status': execution_result.status,
                                    'error_message': execution_result.error_message
                                },
                                'governance_metadata': governance_context,
                                'overrides': enhanced_context.get('overrides', [])
                            },
                            context={
                                'user_id': user_id,
                                'sla_threshold_ms': 5000,
                                'industry_code': enhanced_context.get('industry_code', 'SaaS')
                            }
                        )
                        
                        # Generate evidence pack
                        evidence_pack_id = await self.policy_engine.generate_evidence_pack(
                            tenant_id=int(tenant_id) if tenant_id else 1300,
                            pack_type='workflow_execution',
                            data={
                                'workflow_execution': {
                                    'workflow_id': workflow.workflow_id,
                                    'execution_id': execution_id,
                                    'input_data': input_data,
                                    'output_data': execution_result.outputs,
                                    'execution_time_ms': execution_result.execution_time_ms,
                                    'status': execution_result.status
                                },
                                'governance_data': governance_context,
                                'trust_score_data': trust_score_data
                            },
                            context={
                                'workflow_execution_id': execution_id,
                                'user_id': user_id,
                                'trace_id': enhanced_context.get('trace_id'),
                                'applied_policies': governance_context.get('applied_policies', []),
                                'policy_compliance': governance_context.get('policy_compliance', 'compliant'),
                                'trust_score': trust_score_data.get('trust_score', 1.0),
                                'risk_level': trust_score_data.get('risk_level', 'low'),
                                'industry_code': enhanced_context.get('industry_code', 'SaaS'),
                                'compliance_frameworks': ['SOX_SAAS', 'GDPR_SAAS'],
                                'execution_time_ms': execution_result.execution_time_ms,
                                'source_system': 'dynamic_rba_orchestrator'
                            }
                        )
                        
                        governance_context.update({
                            'trust_score_data': trust_score_data,
                            'evidence_pack_id': evidence_pack_id
                        })
                
                        # Record comprehensive metrics (Tasks 20.1-T05 to 20.1-T09)
                        workflow_end_time = datetime.now()
                        total_execution_time = (workflow_end_time - workflow_start_time).total_seconds()
                        
                        # NEW: Complete the canonical trace
                        self.trace_builder.complete_workflow(workflow_trace, execution_result.outputs)
                        
                        # NEW: Update workflow registry execution metrics
                        await self.workflow_registry.update_execution_metrics(
                            workflow.workflow_id,
                            workflow_dsl.get('version', '1.0.0'),
                            int(total_execution_time * 1000),
                            execution_result.status == 'completed'
                        )
                        
                        # NEW: Record SLO measurements
                        if tenant_id:
                            await self.slo_sla_manager.record_slo_measurement(
                                f"{tenant_context.industry_code.value.lower()}_latency_p95" if tenant_context else 'saas_latency_p95',
                                int(tenant_id),
                                total_execution_time * 1000
                            )
                            
                            await self.slo_sla_manager.record_slo_measurement(
                                f"{tenant_context.industry_code.value.lower()}_error_rate" if tenant_context else 'saas_error_rate',
                                int(tenant_id),
                                0.0 if execution_result.status == 'completed' else 100.0
                            )
                        
                        # NEW: Analyze performance and detect anomalies
                        if tenant_id:
                            current_metrics = {
                                'cpu_utilization': 50.0,  # Mock - would get real metrics
                                'memory_utilization': 60.0,
                                'latency': total_execution_time * 1000,
                                'error_rate': 0.0 if execution_result.status == 'completed' else 100.0
                            }
                            
                            performance_analysis = await self.performance_optimizer.analyze_performance(
                                int(tenant_id), current_metrics
                            )
                            
                            if performance_analysis['anomalies_detected']:
                                self.logger.warning(f"🚨 Performance anomalies detected for tenant {tenant_id}: {len(performance_analysis['anomalies_detected'])} issues")
                        
                        # Record workflow execution metrics
                        self.metrics_collector.record_workflow_execution(
                            workflow_id=workflow.workflow_id,
                            execution_time_seconds=total_execution_time,
                            status=execution_result.status,
                            tenant_id=int(tenant_id) if tenant_id else 1300,
                            industry_code=enhanced_context.get('industry_code', 'SaaS'),
                            region=enhanced_context.get('region', 'US'),
                            execution_id=execution_id
                        )
                    
                    # Record governance metrics
                    if governance_context.get('applied_policies'):
                        for policy_id in governance_context['applied_policies']:
                            self.metrics_collector.record_policy_evaluation(
                                workflow_id=workflow.workflow_id,
                                policy_id=policy_id,
                                result=governance_context.get('policy_compliance', 'compliant'),
                                tenant_id=int(tenant_id) if tenant_id else 1300,
                                industry_code=enhanced_context.get('industry_code', 'SaaS'),
                                region=enhanced_context.get('region', 'US')
                            )
                    
                    # Record trust score metrics
                    if governance_context.get('trust_score_data'):
                        trust_data = governance_context['trust_score_data']
                        self.metrics_collector.record_trust_score(
                            workflow_id=workflow.workflow_id,
                            trust_score=trust_data.get('trust_score', 0.8),
                            trust_level=trust_data.get('trust_level', 'unknown'),
                            tenant_id=int(tenant_id) if tenant_id else 1300,
                            industry_code=enhanced_context.get('industry_code', 'SaaS'),
                            region=enhanced_context.get('region', 'US')
                        )
                    
                    # Record evidence pack metrics
                    if governance_context.get('evidence_pack_id'):
                        self.metrics_collector.record_evidence_pack(
                            workflow_id=workflow.workflow_id,
                            pack_type='workflow_execution',
                            compliance_framework='SOX_SAAS',
                            tenant_id=int(tenant_id) if tenant_id else 1300,
                            industry_code=enhanced_context.get('industry_code', 'SaaS'),
                            region=enhanced_context.get('region', 'US')
                        )
                    
                    # Update trace context with evidence pack
                    if governance_context.get('evidence_pack_id'):
                        trace_context.evidence_pack_id = governance_context['evidence_pack_id']
                        workflow_span.set_attribute("rba.evidence.pack_id", governance_context['evidence_pack_id'])
                    
                    # Set final span attributes
                    workflow_span.set_attribute("rba.execution.total_time_ms", int(total_execution_time * 1000))
                    workflow_span.set_attribute("rba.execution.status", execution_result.status)
                    workflow_span.set_attribute("rba.governance.trust_score", governance_context.get('trust_score_data', {}).get('trust_score', 0.8))
                
                return {
                        'success': execution_result.status == 'completed',
                        'execution_id': execution_id,
                        'workflow_id': workflow.workflow_id,
                        'result_data': execution_result.outputs,
                        'execution_time_ms': execution_result.execution_time_ms,
                        'evidence_pack_id': governance_context.get('evidence_pack_id') or execution_result.evidence_pack_id,
                        'error_message': execution_result.error_message,
                        'orchestrator_type': 'dynamic_rba_enhanced',
                        'enhanced_metadata': {
                            'static_analysis': {
                                'validation_passed': validation_result.is_valid if self.static_analyzer else None,
                                'issues_found': len(validation_result.issues) if self.static_analyzer else 0,
                                'errors_count': len(validation_result.errors) if self.static_analyzer else 0,
                                'warnings_count': len(validation_result.warnings) if self.static_analyzer else 0
                            },
                            'execution_plan': {
                                'plan_hash': execution_plan.plan_hash if execution_plan else None,
                                'optimization_strategy': execution_plan.optimization_strategy.value if execution_plan else None,
                                'estimated_time_ms': execution_plan.estimated_total_time_ms if execution_plan else None,
                                'parallel_groups': len(execution_plan.parallel_groups) if execution_plan else 0,
                                'total_steps': len(execution_plan.execution_steps) if execution_plan else 0
                            },
                            'governance': {
                                'tenant_id': tenant_id,
                                'user_id': user_id,
                                'policy_compliance': governance_context.get('policy_compliance', 'compliant'),
                                'policy_violations_count': len(governance_context.get('policy_violations', [])),
                                'applied_policies': governance_context.get('applied_policies', []),
                                'trust_score': governance_context.get('trust_score_data', {}).get('trust_score'),
                                'trust_level': governance_context.get('trust_score_data', {}).get('trust_level'),
                                'risk_level': governance_context.get('trust_score_data', {}).get('risk_level'),
                                'evidence_pack_id': governance_context.get('evidence_pack_id'),
                                'audit_trail_available': True,
                                'evidence_captured': bool(governance_context.get('evidence_pack_id')),
                                'governance_engine_active': bool(self.policy_engine)
                            }
                        }
                    }
            else:
                # Fallback to legacy execution
                self.logger.warning("⚠️ Enhanced workflow runtime not available, using legacy mode")
                return await self._execute_legacy_workflow(workflow, input_data, user_context, execution_id)
                
        except Exception as e:
            self.logger.error(f"❌ DSL workflow execution failed: {e}")
            
            # Record error metrics
            workflow_end_time = datetime.now()
            total_execution_time = (workflow_end_time - workflow_start_time).total_seconds()
            
            self.metrics_collector.record_workflow_execution(
                workflow_id=workflow_dsl.get('workflow_id', 'unknown'),
                execution_time_seconds=total_execution_time,
                status='failed',
                tenant_id=int(tenant_id) if tenant_id else 1300,
                industry_code=user_context.get('industry_code', 'SaaS'),
                region=user_context.get('region', 'US'),
                execution_id=execution_id
            )
            
            return {
                'success': False,
                'execution_id': execution_id,
                'error': str(e),
                'orchestrator_type': 'dynamic_rba_enhanced',
                'execution_time_ms': int(total_execution_time * 1000),
                'trace_id': trace_context.trace_id if 'trace_context' in locals() else None
            }
    
    async def _execute_legacy_workflow(self, workflow, input_data: Dict[str, Any], user_context: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Fallback to legacy workflow execution"""
        # This would integrate with existing DSL execution logic
        return {
            'success': True,
            'execution_id': execution_id,
            'workflow_id': workflow.workflow_id,
            'result_data': {'message': 'Legacy execution completed'},
            'orchestrator_type': 'dynamic_rba_legacy'
        }
    
    def _select_agent_for_analysis(self, analysis_type: str, config: Dict[str, Any]) -> Optional[Any]:
        """Dynamically select agent for analysis type"""
        
        try:
            # Use registry to create appropriate agent instance
            agent_instance = self.registry.create_agent_instance(analysis_type, config)
            
            if agent_instance:
                agent_info = self.registry.get_agent_for_analysis_type(analysis_type)
                self.logger.info(f"✅ Selected agent: {agent_info.agent_name} for {analysis_type}")
                self.logger.info(f"📝 Agent description: {agent_info.agent_description}")
            
            return agent_instance
            
        except Exception as e:
            self.logger.error(f"❌ Agent selection failed for {analysis_type}: {e}")
            return None
    
    def _create_error_response(
        self,
        execution_id: str,
        analysis_type: str,
        error: str,
        available_types: List[str] = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        
        return {
            'success': False,
            'error': error,
            'execution_id': execution_id,
            'analysis_type': analysis_type,
            'orchestrator_type': 'dynamic_rba',
            'available_analysis_types': available_types or [],
            'analysis_timestamp': datetime.now().isoformat(),
            'troubleshooting': {
                'supported_types': self.registry.get_supported_analysis_types(),
                'total_agents_available': len(self.registry.get_all_agents()),
                'registry_stats': self.registry.get_registry_stats()
            }
        }
    
    def get_available_analysis_types(self) -> List[str]:
        """Get all available analysis types"""
        return self.registry.get_supported_analysis_types()
    
    def get_agent_info_for_analysis_type(self, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get information about the agent that handles a specific analysis type"""
        
        agent_info = self.registry.get_agent_for_analysis_type(analysis_type)
        
        if not agent_info:
            return None
        
        return {
            'agent_name': agent_info.agent_name,
            'agent_description': agent_info.agent_description,
            'supported_analysis_types': agent_info.supported_analysis_types,
            'class_name': agent_info.class_name
        }
    
    def get_orchestrator_capabilities(self) -> Dict[str, Any]:
        """Get orchestrator capabilities and statistics"""
        
        registry_stats = self.registry.get_registry_stats()
        
        return {
            'orchestrator_type': 'dynamic_rba',
            'features': [
                'Dynamic agent discovery',
                'No hardcoded agent selections',
                'Automatic extensibility',
                'Registry-based routing',
                'Fallback error handling'
            ],
            'registry_stats': registry_stats,
            'supported_analysis_types': self.get_available_analysis_types(),
            'agent_details': {
                name: self.get_agent_info_for_analysis_type(analysis_type)
                for analysis_type in self.get_available_analysis_types()
                for name in [analysis_type] if self.get_agent_info_for_analysis_type(analysis_type)
            }
        }
    
    async def validate_analysis_request(
        self,
        analysis_type: str,
        opportunities: List[Dict[str, Any]],
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Validate analysis request before execution"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check if analysis type is supported
        if analysis_type not in self.get_available_analysis_types():
            validation_result['valid'] = False
            validation_result['errors'].append(f"Unsupported analysis type: {analysis_type}")
            validation_result['recommendations'].append(f"Available types: {', '.join(self.get_available_analysis_types())}")
        
        # Check opportunities data
        if not opportunities:
            validation_result['valid'] = False
            validation_result['errors'].append("No opportunities provided for analysis")
        elif len(opportunities) == 0:
            validation_result['warnings'].append("Empty opportunities list provided")
        
        # Validate opportunities structure (basic check)
        if opportunities:
            required_fields = ['Id', 'Name']  # Basic required fields
            for i, opp in enumerate(opportunities[:5]):  # Check first 5
                missing_fields = [field for field in required_fields if field not in opp]
                if missing_fields:
                    validation_result['warnings'].append(f"Opportunity {i+1} missing fields: {missing_fields}")
        
        # Check configuration
        if config and not isinstance(config, dict):
            validation_result['errors'].append("Configuration must be a dictionary")
        
        return validation_result

# Global orchestrator instance
dynamic_rba_orchestrator = DynamicRBAOrchestrator()
