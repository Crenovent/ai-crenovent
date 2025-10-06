"""
Enhanced Plan Synthesizer - Chapter 15.4 Implementation
=======================================================
Tasks 15.4.4-15.4.8: Plan synthesis algorithms & hybrid planning

Implements intelligent plan synthesis for routing orchestrator:
- Hybrid workflow construction (RBA + RBIA + AALA)
- Cost estimation and SLA projection
- Policy overlay enforcement
- Capability dependency resolution
- Dynamic plan optimization
- Evidence pack generation
"""

import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid

logger = logging.getLogger(__name__)

class PlanType(Enum):
    """Plan synthesis types"""
    SINGLE_CAPABILITY = "single_capability"
    HYBRID_WORKFLOW = "hybrid_workflow"
    MULTI_STEP_PLAN = "multi_step_plan"

class ExecutionStrategy(Enum):
    """Execution strategies for plans"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"

class StepType(Enum):
    """Types of plan steps"""
    CAPABILITY_EXECUTION = "capability_execution"
    DECISION_POINT = "decision_point"
    PARALLEL_BRANCH = "parallel_branch"
    LOOP = "loop"

class AutomationType(Enum):
    """Automation types"""
    RBA = "RBA"
    RBIA = "RBIA"
    AALA = "AALA"

@dataclass
class CapabilityRequirement:
    """Capability requirement for plan synthesis"""
    capability_name: str
    capability_type: str
    automation_type: AutomationType
    required_inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    sla_requirements: Dict[str, Any]
    cost_constraints: Dict[str, Any]
    compliance_requirements: List[str]

@dataclass
class PlanStep:
    """Individual step in execution plan"""
    step_id: str
    step_name: str
    step_order: int
    step_type: StepType
    capability_requirement: Optional[CapabilityRequirement]
    automation_type: AutomationType
    
    # Configuration
    step_configuration: Dict[str, Any]
    input_mapping: Dict[str, Any]
    output_mapping: Dict[str, Any]
    
    # Dependencies
    depends_on_steps: List[str]
    parallel_group: Optional[str]
    
    # Cost & Performance
    estimated_cost: float
    estimated_duration_ms: int
    timeout_ms: int
    
    # Error Handling
    retry_policy: Dict[str, Any]
    fallback_step_id: Optional[str]
    error_handling_strategy: str
    
    # Governance
    policy_requirements: List[str]
    compliance_checks: List[str]

@dataclass
class ExecutionPlan:
    """Complete execution plan"""
    plan_id: str
    plan_name: str
    plan_type: PlanType
    execution_strategy: ExecutionStrategy
    
    # Plan Content
    steps: List[PlanStep]
    plan_definition: Dict[str, Any]
    plan_hash: str
    
    # Estimates
    estimated_total_cost: float
    estimated_total_duration_ms: int
    estimated_resource_usage: Dict[str, Any]
    
    # Quality & Risk
    quality_score: float
    risk_score: float
    target_sla_tier: str
    
    # Status
    synthesis_status: str
    validation_status: str
    approval_status: str
    
    # Governance
    evidence_pack_id: Optional[str]
    compliance_attestation: Dict[str, Any]
    
    # Timestamps
    synthesized_at: datetime
    validated_at: Optional[datetime]
    approved_at: Optional[datetime]

class EnhancedPlanSynthesizer:
    """
    Enhanced Plan Synthesizer for Routing Orchestrator
    Tasks 15.4.4-15.4.8: Intelligent plan synthesis with hybrid workflows
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Dynamic configuration
        self.synthesis_config = self._load_synthesis_config()
        self.cost_models = self._load_cost_models()
        self.sla_templates = self._load_sla_templates()
        
        # Capability registry integration
        self.capability_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Plan optimization settings
        self.optimization_strategies = self._load_optimization_strategies()
        
    def _load_synthesis_config(self) -> Dict[str, Any]:
        """Load dynamic synthesis configuration"""
        return {
            "max_plan_steps": 20,
            "max_parallel_branches": 5,
            "default_timeout_ms": 300000,  # 5 minutes
            "cost_optimization_enabled": True,
            "sla_optimization_enabled": True,
            "hybrid_workflow_threshold": 0.7,
            "quality_score_threshold": 0.8,
            "risk_score_threshold": 0.3,
            "auto_approval_threshold": 0.9
        }
    
    def _load_cost_models(self) -> Dict[str, Dict[str, Any]]:
        """Load cost estimation models"""
        return {
            "RBA": {
                "base_cost_per_step": 0.05,
                "complexity_multiplier": 1.2,
                "data_volume_factor": 0.001,
                "sla_tier_multipliers": {"T0": 2.0, "T1": 1.5, "T2": 1.0}
            },
            "RBIA": {
                "base_cost_per_step": 0.25,
                "ml_inference_cost": 0.15,
                "model_complexity_factor": 1.8,
                "data_volume_factor": 0.005,
                "sla_tier_multipliers": {"T0": 2.5, "T1": 1.8, "T2": 1.0}
            },
            "AALA": {
                "base_cost_per_step": 1.50,
                "agent_complexity_factor": 2.5,
                "token_cost_per_1k": 0.002,
                "reasoning_overhead": 0.75,
                "sla_tier_multipliers": {"T0": 3.0, "T1": 2.2, "T2": 1.0}
            }
        }
    
    def _load_sla_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load SLA templates for different tiers"""
        return {
            "T0": {
                "max_response_time_ms": 5000,
                "availability_slo": 99.99,
                "error_rate_threshold": 0.01,
                "concurrent_executions": 100
            },
            "T1": {
                "max_response_time_ms": 15000,
                "availability_slo": 99.9,
                "error_rate_threshold": 0.05,
                "concurrent_executions": 50
            },
            "T2": {
                "max_response_time_ms": 30000,
                "availability_slo": 99.5,
                "error_rate_threshold": 0.1,
                "concurrent_executions": 20
            }
        }
    
    def _load_optimization_strategies(self) -> Dict[str, Any]:
        """Load plan optimization strategies"""
        return {
            "cost_optimization": {
                "prefer_rba_over_aala": True,
                "batch_similar_operations": True,
                "cache_intermediate_results": True,
                "parallel_independent_steps": True
            },
            "performance_optimization": {
                "minimize_sequential_dependencies": True,
                "optimize_data_flow": True,
                "preload_required_data": True,
                "use_circuit_breakers": True
            },
            "reliability_optimization": {
                "add_retry_policies": True,
                "include_fallback_steps": True,
                "validate_intermediate_results": True,
                "implement_checkpoints": True
            }
        }
    
    async def synthesize_plan(
        self, 
        intent_classification: Dict[str, Any],
        available_capabilities: List[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """
        Task 15.4.4: Synthesize execution plan from intent and capabilities
        """
        try:
            context = context or {}
            plan_id = str(uuid.uuid4())
            
            self.logger.info(f"🔧 Synthesizing plan for intent: {intent_classification.get('category')}")
            
            # Determine plan type and strategy
            plan_type, execution_strategy = await self._determine_plan_characteristics(
                intent_classification, available_capabilities, context
            )
            
            # Build capability requirements
            capability_requirements = await self._build_capability_requirements(
                intent_classification, context
            )
            
            # Select and score capabilities
            selected_capabilities = await self._select_capabilities(
                capability_requirements, available_capabilities, context
            )
            
            # Generate plan steps
            plan_steps = await self._generate_plan_steps(
                selected_capabilities, capability_requirements, execution_strategy, context
            )
            
            # Optimize plan
            optimized_steps = await self._optimize_plan(plan_steps, context)
            
            # Calculate estimates
            cost_estimate = await self._calculate_cost_estimate(optimized_steps, context)
            duration_estimate = await self._calculate_duration_estimate(optimized_steps, context)
            resource_estimate = await self._calculate_resource_estimate(optimized_steps, context)
            
            # Calculate quality and risk scores
            quality_score = await self._calculate_quality_score(optimized_steps, selected_capabilities)
            risk_score = await self._calculate_risk_score(optimized_steps, context)
            
            # Generate plan definition and hash
            plan_definition = await self._generate_plan_definition(optimized_steps, context)
            plan_hash = self._calculate_plan_hash(plan_definition)
            
            # Determine approval status
            approval_status = await self._determine_approval_status(
                cost_estimate, risk_score, quality_score, context
            )
            
            # Create execution plan
            execution_plan = ExecutionPlan(
                plan_id=plan_id,
                plan_name=f"Plan for {intent_classification.get('category', 'Unknown')}",
                plan_type=plan_type,
                execution_strategy=execution_strategy,
                steps=optimized_steps,
                plan_definition=plan_definition,
                plan_hash=plan_hash,
                estimated_total_cost=cost_estimate,
                estimated_total_duration_ms=duration_estimate,
                estimated_resource_usage=resource_estimate,
                quality_score=quality_score,
                risk_score=risk_score,
                target_sla_tier=context.get("sla_tier", "T2"),
                synthesis_status="synthesized",
                validation_status="pending",
                approval_status=approval_status,
                evidence_pack_id=None,
                compliance_attestation={},
                synthesized_at=datetime.now(timezone.utc),
                validated_at=None,
                approved_at=None
            )
            
            self.logger.info(f"✅ Plan synthesized: {plan_id} ({len(optimized_steps)} steps, ${cost_estimate:.2f}, {duration_estimate}ms)")
            return execution_plan
            
        except Exception as e:
            self.logger.error(f"❌ Plan synthesis failed: {e}")
            raise
    
    async def _determine_plan_characteristics(
        self, 
        intent_classification: Dict[str, Any],
        available_capabilities: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Tuple[PlanType, ExecutionStrategy]:
        """Determine plan type and execution strategy"""
        try:
            automation_route = intent_classification.get("automation_route", "RBA")
            confidence_score = intent_classification.get("confidence_score", 0.5)
            
            # Determine plan type
            if len(available_capabilities) == 1:
                plan_type = PlanType.SINGLE_CAPABILITY
            elif confidence_score >= self.synthesis_config["hybrid_workflow_threshold"]:
                plan_type = PlanType.HYBRID_WORKFLOW
            else:
                plan_type = PlanType.MULTI_STEP_PLAN
            
            # Determine execution strategy
            if automation_route == "AALA" or plan_type == PlanType.HYBRID_WORKFLOW:
                execution_strategy = ExecutionStrategy.CONDITIONAL
            elif len(available_capabilities) > 3:
                execution_strategy = ExecutionStrategy.PARALLEL
            else:
                execution_strategy = ExecutionStrategy.SEQUENTIAL
            
            return plan_type, execution_strategy
            
        except Exception as e:
            self.logger.error(f"❌ Error determining plan characteristics: {e}")
            return PlanType.SINGLE_CAPABILITY, ExecutionStrategy.SEQUENTIAL
    
    async def _build_capability_requirements(
        self, 
        intent_classification: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[CapabilityRequirement]:
        """Build capability requirements from intent classification"""
        try:
            requirements = []
            
            category = intent_classification.get("category", "unknown")
            automation_route = intent_classification.get("automation_route", "RBA")
            extracted_params = intent_classification.get("extracted_parameters", {})
            compliance_flags = intent_classification.get("compliance_flags", [])
            
            # Build primary capability requirement
            primary_requirement = CapabilityRequirement(
                capability_name=intent_classification.get("recommended_capability", f"{category}_capability"),
                capability_type=f"{automation_route}_TEMPLATE" if automation_route == "RBA" else f"{automation_route}_MODEL",
                automation_type=AutomationType(automation_route),
                required_inputs=self._build_required_inputs(extracted_params, context),
                expected_outputs=self._build_expected_outputs(category, context),
                sla_requirements=self._build_sla_requirements(context),
                cost_constraints=self._build_cost_constraints(context),
                compliance_requirements=compliance_flags
            )
            requirements.append(primary_requirement)
            
            # Add supporting capabilities for hybrid workflows
            if automation_route in ["RBIA", "AALA"]:
                supporting_requirements = await self._build_supporting_requirements(
                    category, automation_route, context
                )
                requirements.extend(supporting_requirements)
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"❌ Error building capability requirements: {e}")
            return []
    
    def _build_required_inputs(self, extracted_params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Build required inputs from extracted parameters"""
        inputs = {
            "tenant_id": context.get("tenant_id"),
            "user_id": context.get("user_id"),
            "execution_context": context
        }
        
        # Add extracted parameters
        if extracted_params:
            inputs.update(extracted_params)
        
        return inputs
    
    def _build_expected_outputs(self, category: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build expected outputs based on category"""
        output_templates = {
            "arr_analysis": {
                "arr_metrics": "object",
                "growth_trends": "array",
                "variance_analysis": "object",
                "recommendations": "array"
            },
            "churn_prediction": {
                "churn_risk_scores": "array",
                "at_risk_accounts": "array",
                "retention_recommendations": "array",
                "health_scores": "object"
            },
            "comp_plan": {
                "compensation_structure": "object",
                "quota_allocations": "array",
                "performance_metrics": "object",
                "approval_workflow": "object"
            },
            "qbr": {
                "account_summary": "object",
                "business_review": "object",
                "growth_opportunities": "array",
                "action_items": "array"
            }
        }
        
        return output_templates.get(category, {"results": "object", "metadata": "object"})
    
    def _build_sla_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build SLA requirements from context"""
        sla_tier = context.get("sla_tier", "T2")
        return self.sla_templates.get(sla_tier, self.sla_templates["T2"])
    
    def _build_cost_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build cost constraints from context"""
        return {
            "max_cost_per_execution": context.get("max_cost_per_execution", 100.0),
            "budget_category": context.get("budget_category", "operations"),
            "cost_center": context.get("cost_center", "revenue_operations")
        }
    
    async def _build_supporting_requirements(
        self, 
        category: str, 
        automation_route: str,
        context: Dict[str, Any]
    ) -> List[CapabilityRequirement]:
        """Build supporting capability requirements for hybrid workflows"""
        try:
            supporting_requirements = []
            
            # Add data validation capability for RBIA/AALA
            if automation_route in ["RBIA", "AALA"]:
                data_validation_req = CapabilityRequirement(
                    capability_name="Data Validation Engine",
                    capability_type="RBA_TEMPLATE",
                    automation_type=AutomationType.RBA,
                    required_inputs={"data_sources": "array", "validation_rules": "object"},
                    expected_outputs={"validation_results": "object", "data_quality_score": "number"},
                    sla_requirements={"max_response_time_ms": 5000},
                    cost_constraints={"max_cost_per_execution": 5.0},
                    compliance_requirements=["DATA_QUALITY"]
                )
                supporting_requirements.append(data_validation_req)
            
            # Add evidence collection for compliance-sensitive categories
            if category in ["arr_analysis", "comp_plan"]:
                evidence_req = CapabilityRequirement(
                    capability_name="Evidence Collector",
                    capability_type="RBA_TEMPLATE",
                    automation_type=AutomationType.RBA,
                    required_inputs={"execution_data": "object", "compliance_frameworks": "array"},
                    expected_outputs={"evidence_pack": "object", "compliance_attestation": "object"},
                    sla_requirements={"max_response_time_ms": 10000},
                    cost_constraints={"max_cost_per_execution": 8.0},
                    compliance_requirements=["SOX", "AUDIT_TRAIL"]
                )
                supporting_requirements.append(evidence_req)
            
            return supporting_requirements
            
        except Exception as e:
            self.logger.error(f"❌ Error building supporting requirements: {e}")
            return []
    
    async def _select_capabilities(
        self,
        capability_requirements: List[CapabilityRequirement],
        available_capabilities: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select best matching capabilities for requirements"""
        try:
            selected_capabilities = []
            
            for requirement in capability_requirements:
                best_match = None
                best_score = 0.0
                
                for capability in available_capabilities:
                    score = await self._score_capability_match(requirement, capability, context)
                    if score > best_score:
                        best_score = score
                        best_match = capability
                
                if best_match and best_score >= 0.5:  # Minimum match threshold
                    selected_capabilities.append({
                        **best_match,
                        "match_score": best_score,
                        "requirement": asdict(requirement)
                    })
                else:
                    self.logger.warning(f"⚠️ No suitable capability found for requirement: {requirement.capability_name}")
            
            return selected_capabilities
            
        except Exception as e:
            self.logger.error(f"❌ Error selecting capabilities: {e}")
            return []
    
    async def _score_capability_match(
        self,
        requirement: CapabilityRequirement,
        capability: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Score how well a capability matches a requirement"""
        try:
            score_factors = []
            
            # Name similarity
            if requirement.capability_name.lower() in capability.get("capability_name", "").lower():
                score_factors.append(0.3)
            
            # Type match
            if requirement.capability_type == capability.get("capability_type"):
                score_factors.append(0.2)
            
            # Automation type match
            if requirement.automation_type.value in capability.get("capability_type", ""):
                score_factors.append(0.2)
            
            # Trust score
            trust_score = capability.get("trust_score", 0.5)
            score_factors.append(trust_score * 0.15)
            
            # SLA compatibility
            capability_sla = capability.get("sla_tier", "T2")
            required_sla = context.get("sla_tier", "T2")
            if capability_sla <= required_sla:  # T0 > T1 > T2
                score_factors.append(0.1)
            
            # Cost compatibility
            capability_cost = capability.get("estimated_cost_per_execution", 0.0)
            max_cost = requirement.cost_constraints.get("max_cost_per_execution", 100.0)
            if capability_cost <= max_cost:
                score_factors.append(0.05)
            
            return sum(score_factors)
            
        except Exception as e:
            self.logger.error(f"❌ Error scoring capability match: {e}")
            return 0.0
    
    async def _generate_plan_steps(
        self,
        selected_capabilities: List[Dict[str, Any]],
        capability_requirements: List[CapabilityRequirement],
        execution_strategy: ExecutionStrategy,
        context: Dict[str, Any]
    ) -> List[PlanStep]:
        """Generate plan steps from selected capabilities"""
        try:
            steps = []
            step_order = 1
            
            for i, capability in enumerate(selected_capabilities):
                requirement = capability_requirements[i] if i < len(capability_requirements) else None
                
                step = PlanStep(
                    step_id=str(uuid.uuid4()),
                    step_name=f"Execute {capability.get('capability_name', 'Unknown')}",
                    step_order=step_order,
                    step_type=StepType.CAPABILITY_EXECUTION,
                    capability_requirement=requirement,
                    automation_type=AutomationType(capability.get("capability_type", "RBA_TEMPLATE").split("_")[0]),
                    step_configuration=self._build_step_configuration(capability, context),
                    input_mapping=self._build_input_mapping(capability, requirement),
                    output_mapping=self._build_output_mapping(capability, requirement),
                    depends_on_steps=self._build_dependencies(steps, execution_strategy),
                    parallel_group=self._determine_parallel_group(execution_strategy, step_order),
                    estimated_cost=capability.get("estimated_cost_per_execution", 0.0),
                    estimated_duration_ms=capability.get("avg_execution_time_ms", 30000),
                    timeout_ms=self._calculate_timeout(capability, context),
                    retry_policy=self._build_retry_policy(capability, context),
                    fallback_step_id=None,
                    error_handling_strategy="fail_fast",
                    policy_requirements=requirement.compliance_requirements if requirement else [],
                    compliance_checks=self._build_compliance_checks(capability, context)
                )
                
                steps.append(step)
                step_order += 1
            
            return steps
            
        except Exception as e:
            self.logger.error(f"❌ Error generating plan steps: {e}")
            return []
    
    def _build_step_configuration(self, capability: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Build step configuration"""
        return {
            "capability_id": capability.get("capability_meta_id"),
            "version": capability.get("version", "1.0.0"),
            "execution_mode": "production",
            "tenant_id": context.get("tenant_id"),
            "user_id": context.get("user_id")
        }
    
    def _build_input_mapping(self, capability: Dict[str, Any], requirement: Optional[CapabilityRequirement]) -> Dict[str, Any]:
        """Build input mapping for step"""
        if requirement:
            return requirement.required_inputs
        return {"default_inputs": True}
    
    def _build_output_mapping(self, capability: Dict[str, Any], requirement: Optional[CapabilityRequirement]) -> Dict[str, Any]:
        """Build output mapping for step"""
        if requirement:
            return requirement.expected_outputs
        return {"default_outputs": True}
    
    def _build_dependencies(self, existing_steps: List[PlanStep], execution_strategy: ExecutionStrategy) -> List[str]:
        """Build step dependencies based on execution strategy"""
        if execution_strategy == ExecutionStrategy.SEQUENTIAL and existing_steps:
            return [existing_steps[-1].step_id]
        elif execution_strategy == ExecutionStrategy.CONDITIONAL and len(existing_steps) >= 2:
            return [existing_steps[-2].step_id]  # Skip immediate predecessor for conditional
        return []
    
    def _determine_parallel_group(self, execution_strategy: ExecutionStrategy, step_order: int) -> Optional[str]:
        """Determine parallel group for step"""
        if execution_strategy == ExecutionStrategy.PARALLEL:
            return f"parallel_group_{(step_order - 1) // 3 + 1}"  # Group every 3 steps
        return None
    
    def _calculate_timeout(self, capability: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Calculate timeout for step"""
        base_timeout = capability.get("avg_execution_time_ms", 30000)
        sla_tier = context.get("sla_tier", "T2")
        
        # Add buffer based on SLA tier
        buffer_multipliers = {"T0": 1.5, "T1": 2.0, "T2": 3.0}
        return int(base_timeout * buffer_multipliers.get(sla_tier, 3.0))
    
    def _build_retry_policy(self, capability: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Build retry policy for step"""
        return {
            "max_retries": 3,
            "retry_delay_ms": 1000,
            "exponential_backoff": True,
            "retry_on_errors": ["timeout", "temporary_failure", "rate_limit"]
        }
    
    def _build_compliance_checks(self, capability: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Build compliance checks for step"""
        checks = []
        
        # Add industry-specific checks
        if context.get("industry_code") == "SAAS":
            checks.extend(["SAAS_BUSINESS_RULES", "DATA_PRIVACY"])
        
        # Add SLA tier checks
        sla_tier = context.get("sla_tier", "T2")
        if sla_tier in ["T0", "T1"]:
            checks.append("SLA_COMPLIANCE")
        
        return checks
    
    async def _optimize_plan(self, steps: List[PlanStep], context: Dict[str, Any]) -> List[PlanStep]:
        """Optimize plan steps for cost, performance, and reliability"""
        try:
            optimized_steps = steps.copy()
            
            # Cost optimization
            if self.optimization_strategies["cost_optimization"]["prefer_rba_over_aala"]:
                optimized_steps = await self._optimize_for_cost(optimized_steps)
            
            # Performance optimization
            if self.optimization_strategies["performance_optimization"]["parallel_independent_steps"]:
                optimized_steps = await self._optimize_for_performance(optimized_steps)
            
            # Reliability optimization
            if self.optimization_strategies["reliability_optimization"]["add_retry_policies"]:
                optimized_steps = await self._optimize_for_reliability(optimized_steps)
            
            return optimized_steps
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing plan: {e}")
            return steps
    
    async def _optimize_for_cost(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Optimize plan for cost efficiency"""
        # Sort steps by cost efficiency (results per dollar)
        for step in steps:
            if step.estimated_cost > 50.0 and step.automation_type == AutomationType.AALA:
                # Consider downgrading to RBIA if possible
                step.automation_type = AutomationType.RBIA
                step.estimated_cost *= 0.6  # RBIA typically 60% of AALA cost
        
        return steps
    
    async def _optimize_for_performance(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Optimize plan for performance"""
        # Identify independent steps that can run in parallel
        independent_steps = []
        for step in steps:
            if not step.depends_on_steps:
                independent_steps.append(step)
        
        # Group independent steps for parallel execution
        if len(independent_steps) > 1:
            for i, step in enumerate(independent_steps):
                step.parallel_group = f"parallel_batch_1"
        
        return steps
    
    async def _optimize_for_reliability(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Optimize plan for reliability"""
        for step in steps:
            # Add circuit breaker for external dependencies
            if "external" in step.step_name.lower():
                step.step_configuration["circuit_breaker"] = {
                    "failure_threshold": 3,
                    "recovery_timeout": 60,
                    "enabled": True
                }
            
            # Enhance retry policy for critical steps
            if step.automation_type == AutomationType.AALA:
                step.retry_policy["max_retries"] = 5
                step.retry_policy["retry_delay_ms"] = 2000
        
        return steps
    
    async def _calculate_cost_estimate(self, steps: List[PlanStep], context: Dict[str, Any]) -> float:
        """Calculate total cost estimate for plan"""
        try:
            total_cost = 0.0
            
            for step in steps:
                base_cost = step.estimated_cost
                
                # Apply SLA tier multiplier
                sla_tier = context.get("sla_tier", "T2")
                automation_type = step.automation_type.value
                multiplier = self.cost_models.get(automation_type, {}).get("sla_tier_multipliers", {}).get(sla_tier, 1.0)
                
                step_cost = base_cost * multiplier
                total_cost += step_cost
            
            return round(total_cost, 2)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating cost estimate: {e}")
            return 0.0
    
    async def _calculate_duration_estimate(self, steps: List[PlanStep], context: Dict[str, Any]) -> int:
        """Calculate total duration estimate for plan"""
        try:
            # Check for parallel execution
            parallel_groups = {}
            sequential_duration = 0
            
            for step in steps:
                if step.parallel_group:
                    if step.parallel_group not in parallel_groups:
                        parallel_groups[step.parallel_group] = []
                    parallel_groups[step.parallel_group].append(step.estimated_duration_ms)
                else:
                    sequential_duration += step.estimated_duration_ms
            
            # Calculate parallel group durations (max of each group)
            parallel_duration = sum(max(group_durations) for group_durations in parallel_groups.values())
            
            return sequential_duration + parallel_duration
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating duration estimate: {e}")
            return 300000  # Default 5 minutes
    
    async def _calculate_resource_estimate(self, steps: List[PlanStep], context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource usage estimate"""
        try:
            return {
                "cpu_cores": len(steps) * 0.5,
                "memory_mb": len(steps) * 256,
                "storage_mb": len(steps) * 10,
                "network_mb": len(steps) * 5,
                "concurrent_executions": min(len(steps), 10)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating resource estimate: {e}")
            return {}
    
    async def _calculate_quality_score(self, steps: List[PlanStep], capabilities: List[Dict[str, Any]]) -> float:
        """Calculate plan quality score"""
        try:
            if not capabilities:
                return 0.5
            
            # Average trust scores of selected capabilities
            trust_scores = [cap.get("trust_score", 0.5) for cap in capabilities]
            avg_trust = sum(trust_scores) / len(trust_scores)
            
            # Plan complexity factor (fewer steps = higher quality for simple tasks)
            complexity_factor = max(0.5, 1.0 - (len(steps) - 1) * 0.1)
            
            # Optimization factor
            optimization_factor = 0.9 if len(steps) > 1 else 1.0
            
            return min(1.0, avg_trust * complexity_factor * optimization_factor)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating quality score: {e}")
            return 0.5
    
    async def _calculate_risk_score(self, steps: List[PlanStep], context: Dict[str, Any]) -> float:
        """Calculate plan risk score"""
        try:
            risk_factors = []
            
            # Complexity risk
            complexity_risk = min(0.5, len(steps) * 0.05)
            risk_factors.append(complexity_risk)
            
            # Automation type risk (AALA > RBIA > RBA)
            automation_risks = {"AALA": 0.3, "RBIA": 0.2, "RBA": 0.1}
            avg_automation_risk = sum(automation_risks.get(step.automation_type.value, 0.1) for step in steps) / len(steps)
            risk_factors.append(avg_automation_risk)
            
            # External dependency risk
            external_deps = sum(1 for step in steps if "external" in step.step_name.lower())
            external_risk = min(0.3, external_deps * 0.1)
            risk_factors.append(external_risk)
            
            return min(1.0, sum(risk_factors))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk score: {e}")
            return 0.3
    
    async def _generate_plan_definition(self, steps: List[PlanStep], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete plan definition"""
        return {
            "version": "1.0.0",
            "steps": [asdict(step) for step in steps],
            "execution_metadata": {
                "tenant_id": context.get("tenant_id"),
                "user_id": context.get("user_id"),
                "industry_code": context.get("industry_code"),
                "sla_tier": context.get("sla_tier"),
                "generated_at": datetime.now(timezone.utc).isoformat()
            },
            "optimization_applied": list(self.optimization_strategies.keys()),
            "governance": {
                "compliance_frameworks": context.get("compliance_frameworks", []),
                "policy_requirements": list(set(req for step in steps for req in step.policy_requirements)),
                "evidence_required": True
            }
        }
    
    def _calculate_plan_hash(self, plan_definition: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of plan definition"""
        plan_json = json.dumps(plan_definition, sort_keys=True)
        return hashlib.sha256(plan_json.encode()).hexdigest()
    
    async def _determine_approval_status(
        self, 
        cost_estimate: float, 
        risk_score: float, 
        quality_score: float,
        context: Dict[str, Any]
    ) -> str:
        """Determine if plan requires approval"""
        try:
            # Auto-approve if low cost, low risk, high quality
            if (cost_estimate < 25.0 and 
                risk_score < 0.2 and 
                quality_score >= self.synthesis_config["auto_approval_threshold"]):
                return "auto_approved"
            
            # Require approval for high-cost or high-risk plans
            if cost_estimate > 100.0 or risk_score > 0.5:
                return "pending"
            
            # Default to auto-approved for moderate plans
            return "auto_approved"
            
        except Exception as e:
            self.logger.error(f"❌ Error determining approval status: {e}")
            return "pending"

# Singleton instance
_enhanced_plan_synthesizer = None

def get_enhanced_plan_synthesizer(pool_manager=None) -> EnhancedPlanSynthesizer:
    """Get singleton instance of Enhanced Plan Synthesizer"""
    global _enhanced_plan_synthesizer
    if _enhanced_plan_synthesizer is None:
        _enhanced_plan_synthesizer = EnhancedPlanSynthesizer(pool_manager)
    return _enhanced_plan_synthesizer
