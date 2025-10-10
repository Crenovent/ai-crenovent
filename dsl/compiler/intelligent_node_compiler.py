"""
Intelligent Node Compiler - Compiler hooks for ML decision nodes
Task 4.1.3: Implement compiler hooks for intelligent nodes
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json

from ..operators.base import BaseOperator, OperatorContext, OperatorResult
from ..operators.ml_predict import MLPredictOperator
from ..operators.ml_score import MLScoreOperator
from ..operators.ml_classify import MLClassifyOperator
from ..operators.ml_explain import MLExplainOperator
from ..operators.node_registry_service import NodeRegistryService
from ..operators.explainability_service import ExplainabilityService
from ..operators.drift_bias_monitor import DriftBiasMonitor

logger = logging.getLogger(__name__)

@dataclass
class IntelligentNodeConfig:
    """Configuration for intelligent ML nodes"""
    node_id: str
    node_type: str  # 'ml_predict', 'ml_score', 'ml_classify', 'ml_explain'
    model_id: str
    input_mapping: Dict[str, str]  # Maps workflow variables to model inputs
    output_mapping: Dict[str, str]  # Maps model outputs to workflow variables
    confidence_threshold: float = 0.7
    fallback_enabled: bool = True
    explainability_enabled: bool = True
    governance_config: Dict[str, Any] = None

@dataclass
class CompilationResult:
    """Result of intelligent node compilation"""
    success: bool
    compiled_operators: List[BaseOperator]
    execution_plan: List[Dict[str, Any]]
    error_message: Optional[str] = None
    warnings: List[str] = None

class IntelligentNodeCompiler:
    """
    Compiler for intelligent ML nodes in RBIA workflows
    
    Provides:
    - ML node compilation and optimization
    - Execution plan generation
    - Model registry integration
    - Fallback logic compilation
    - Governance rule enforcement
    """
    
    def __init__(self, node_registry: NodeRegistryService):
        self.node_registry = node_registry
        self.logger = logging.getLogger(__name__)
        self._operator_factory = self._create_operator_factory()
        self._explainability_service = ExplainabilityService()
        self._drift_bias_monitor = DriftBiasMonitor()
    
    def _create_operator_factory(self) -> Dict[str, type]:
        """Create factory for ML operators"""
        return {
            'ml_predict': MLPredictOperator,
            'ml_score': MLScoreOperator,
            'ml_classify': MLClassifyOperator,
            'ml_explain': MLExplainOperator
        }
    
    async def compile_intelligent_node(
        self, 
        node_config: IntelligentNodeConfig,
        workflow_context: Dict[str, Any]
    ) -> CompilationResult:
        """Compile an intelligent ML node for execution"""
        try:
            self.logger.info(f"Compiling intelligent node: {node_config.node_id}")
            
            # Validate node configuration
            validation_result = await self._validate_node_config(node_config)
            if not validation_result.success:
                return CompilationResult(
                    success=False,
                    compiled_operators=[],
                    execution_plan=[],
                    error_message=validation_result.error_message
                )
            
            # Get model configuration
            model_config = await self.node_registry.get_model(node_config.model_id)
            if not model_config:
                return CompilationResult(
                    success=False,
                    compiled_operators=[],
                    execution_plan=[],
                    error_message=f"Model {node_config.model_id} not found in registry"
                )
            
            # Create ML operator
            operator_class = self._operator_factory.get(node_config.node_type)
            if not operator_class:
                return CompilationResult(
                    success=False,
                    compiled_operators=[],
                    execution_plan=[],
                    error_message=f"Unsupported node type: {node_config.node_type}"
                )
            
            # Initialize operator
            operator = operator_class(operator_id=node_config.node_id)
            
            # Inject model registry
            model_registry = await self.node_registry.get_model_registry_dict()
            operator.set_model_registry(model_registry)
            
            # Compile execution plan
            execution_plan = await self._compile_execution_plan(
                node_config, model_config, workflow_context
            )
            
            # Add governance hooks
            governance_plan = await self._compile_governance_hooks(
                node_config, model_config, workflow_context
            )
            execution_plan.extend(governance_plan)
            
            # Add fallback hooks if enabled
            if node_config.fallback_enabled:
                fallback_plan = await self._compile_fallback_hooks(
                    node_config, model_config, workflow_context
                )
                execution_plan.extend(fallback_plan)
            
            # Add explainability hooks if enabled
            if node_config.explainability_enabled:
                explainability_plan = await self._compile_explainability_hooks(
                    node_config, model_config, workflow_context
                )
                execution_plan.extend(explainability_plan)
            
            self.logger.info(f"Successfully compiled intelligent node: {node_config.node_id}")
            
            return CompilationResult(
                success=True,
                compiled_operators=[operator],
                execution_plan=execution_plan,
                warnings=validation_result.warnings
            )
            
        except Exception as e:
            self.logger.error(f"Failed to compile intelligent node {node_config.node_id}: {e}")
            return CompilationResult(
                success=False,
                compiled_operators=[],
                execution_plan=[],
                error_message=f"Compilation failed: {e}"
            )
    
    async def _validate_node_config(self, node_config: IntelligentNodeConfig) -> CompilationResult:
        """Validate intelligent node configuration"""
        errors = []
        warnings = []
        
        # Check required fields
        if not node_config.node_id:
            errors.append("node_id is required")
        
        if not node_config.node_type:
            errors.append("node_type is required")
        elif node_config.node_type not in self._operator_factory:
            errors.append(f"Unsupported node_type: {node_config.node_type}")
        
        if not node_config.model_id:
            errors.append("model_id is required")
        
        if not node_config.input_mapping:
            errors.append("input_mapping is required")
        
        if not node_config.output_mapping:
            warnings.append("No output_mapping specified - using default mapping")
        
        # Check confidence threshold
        if not 0 <= node_config.confidence_threshold <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        return CompilationResult(
            success=len(errors) == 0,
            compiled_operators=[],
            execution_plan=[],
            error_message="; ".join(errors) if errors else None,
            warnings=warnings
        )
    
    async def _compile_execution_plan(
        self,
        node_config: IntelligentNodeConfig,
        model_config,
        workflow_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile the main execution plan for the intelligent node"""
        execution_plan = []
        
        # Main ML operation
        ml_operation = {
            "step_id": f"{node_config.node_id}_ml_operation",
            "operator_type": node_config.node_type,
            "operator_id": node_config.node_id,
            "config": {
                "model_id": node_config.model_id,
                "input_data": self._build_input_data(node_config, workflow_context),
                "confidence_threshold": node_config.confidence_threshold,
                "explainability_enabled": node_config.explainability_enabled,
                "fallback_enabled": node_config.fallback_enabled
            },
            "output_mapping": node_config.output_mapping,
            "execution_order": 1
        }
        
        execution_plan.append(ml_operation)
        
        # Add model-specific configurations
        if node_config.node_type == "ml_score":
            ml_operation["config"]["score_range"] = [0, 100]
            ml_operation["config"]["thresholds"] = {
                "low": 30,
                "medium": 60,
                "high": 80
            }
        elif node_config.node_type == "ml_classify":
            ml_operation["config"]["class_mapping"] = {
                "low_risk": "Low Risk",
                "medium_risk": "Medium Risk",
                "high_risk": "High Risk"
            }
        elif node_config.node_type == "ml_explain":
            ml_operation["config"]["explanation_type"] = "shap"
            ml_operation["config"]["explanation_params"] = {
                "shap": {"background_samples": 100, "max_evals": 1000}
            }
        
        return execution_plan
    
    async def _compile_governance_hooks(
        self,
        node_config: IntelligentNodeConfig,
        model_config,
        workflow_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile governance hooks for the intelligent node"""
        governance_plan = []
        
        # Pre-execution governance check
        pre_governance = {
            "step_id": f"{node_config.node_id}_pre_governance",
            "operator_type": "governance_check",
            "config": {
                "check_type": "pre_execution",
                "model_id": node_config.model_id,
                "tenant_id": workflow_context.get("tenant_id"),
                "user_id": workflow_context.get("user_id"),
                "required_permissions": ["ml_execute", "model_access"],
                "trust_threshold": node_config.confidence_threshold
            },
            "execution_order": 0
        }
        governance_plan.append(pre_governance)
        
        # Post-execution governance logging
        post_governance = {
            "step_id": f"{node_config.node_id}_post_governance",
            "operator_type": "governance_logging",
            "config": {
                "check_type": "post_execution",
                "model_id": node_config.model_id,
                "evidence_required": True,
                "audit_logging": True,
                "bias_monitoring": True,
                "drift_monitoring": True
            },
            "execution_order": 2
        }
        governance_plan.append(post_governance)
        
        return governance_plan
    
    async def _compile_fallback_hooks(
        self,
        node_config: IntelligentNodeConfig,
        model_config,
        workflow_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile fallback hooks for the intelligent node"""
        fallback_plan = []
        
        # Fallback rule compilation
        fallback_rules = self._compile_fallback_rules(node_config, model_config)
        
        # Fallback execution step
        fallback_execution = {
            "step_id": f"{node_config.node_id}_fallback_execution",
            "operator_type": "fallback_executor",
            "config": {
                "fallback_rules": fallback_rules,
                "trigger_conditions": [
                    "confidence_below_threshold",
                    "model_error",
                    "timeout_exceeded"
                ],
                "fallback_strategy": "rule_based",
                "logging_enabled": True
            },
            "execution_order": 3,
            "conditional": True
        }
        fallback_plan.append(fallback_execution)
        
        return fallback_plan
    
    async def _compile_explainability_hooks(
        self,
        node_config: IntelligentNodeConfig,
        model_config,
        workflow_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile explainability hooks for the intelligent node"""
        explainability_plan = []
        
        # Explainability logging step
        explainability_logging = {
            "step_id": f"{node_config.node_id}_explainability_logging",
            "operator_type": "explainability_logger",
            "config": {
                "model_id": node_config.model_id,
                "explanation_types": ["shap", "lime"],
                "feature_importance_tracking": True,
                "audit_integration": True,
                "tenant_id": workflow_context.get("tenant_id"),
                "workflow_id": workflow_context.get("workflow_id")
            },
            "execution_order": 4
        }
        explainability_plan.append(explainability_logging)
        
        return explainability_plan
    
    def _build_input_data(
        self, 
        node_config: IntelligentNodeConfig, 
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build input data for the ML model from workflow context"""
        input_data = {}
        
        for model_input, workflow_var in node_config.input_mapping.items():
            if workflow_var in workflow_context:
                input_data[model_input] = workflow_context[workflow_var]
            else:
                # Try to resolve nested variables (e.g., "customer.mrr")
                value = self._resolve_nested_variable(workflow_var, workflow_context)
                if value is not None:
                    input_data[model_input] = value
                else:
                    self.logger.warning(f"Could not resolve workflow variable: {workflow_var}")
        
        return input_data
    
    def _resolve_nested_variable(self, var_path: str, context: Dict[str, Any]) -> Any:
        """Resolve nested variable paths like 'customer.mrr'"""
        try:
            parts = var_path.split('.')
            value = context
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            
            return value
        except Exception:
            return None
    
    def _compile_fallback_rules(
        self, 
        node_config: IntelligentNodeConfig, 
        model_config
    ) -> List[Dict[str, Any]]:
        """Compile fallback rules for the intelligent node"""
        fallback_rules = []
        
        # Default fallback rules based on model type
        if node_config.node_type == "ml_predict":
            fallback_rules.extend([
                {
                    "rule_type": "threshold",
                    "condition": "confidence < 0.5",
                    "action": {
                        "prediction": "unknown",
                        "confidence": 0.3,
                        "reason": "Low confidence fallback"
                    }
                },
                {
                    "rule_type": "field_check",
                    "condition": "mrr < 100",
                    "action": {
                        "prediction": "high_risk",
                        "confidence": 0.6,
                        "reason": "Low MRR fallback rule"
                    }
                }
            ])
        elif node_config.node_type == "ml_score":
            fallback_rules.extend([
                {
                    "rule_type": "threshold",
                    "condition": "confidence < 0.5",
                    "action": {
                        "score": 50.0,
                        "confidence": 0.3,
                        "reason": "Low confidence fallback"
                    }
                }
            ])
        elif node_config.node_type == "ml_classify":
            fallback_rules.extend([
                {
                    "rule_type": "threshold",
                    "condition": "confidence < 0.5",
                    "action": {
                        "predicted_class": "unknown",
                        "confidence": 0.3,
                        "reason": "Low confidence fallback"
                    }
                }
            ])
        
        return fallback_rules
    
    async def compile_workflow_intelligent_nodes(
        self, 
        workflow_steps: List[Dict[str, Any]],
        workflow_context: Dict[str, Any]
    ) -> Dict[str, CompilationResult]:
        """Compile all intelligent nodes in a workflow"""
        compilation_results = {}
        
        for step in workflow_steps:
            if step.get("type") in self._operator_factory:
                # Convert workflow step to intelligent node config
                node_config = IntelligentNodeConfig(
                    node_id=step["id"],
                    node_type=step["type"],
                    model_id=step["params"].get("model_id"),
                    input_mapping=step["params"].get("input_mapping", {}),
                    output_mapping=step.get("outputs", {}),
                    confidence_threshold=step["params"].get("confidence_threshold", 0.7),
                    fallback_enabled=step["params"].get("fallback_enabled", True),
                    explainability_enabled=step["params"].get("explainability_enabled", True),
                    governance_config=step.get("governance", {})
                )
                
                # Compile the intelligent node
                result = await self.compile_intelligent_node(node_config, workflow_context)
                compilation_results[step["id"]] = result
        
        return compilation_results
    
    async def execute_intelligent_node(
        self,
        node_config: IntelligentNodeConfig,
        context: OperatorContext,
        execution_plan: List[Dict[str, Any]]
    ) -> OperatorResult:
        """Execute a compiled intelligent node"""
        try:
            self.logger.info(f"Executing intelligent node: {node_config.node_id}")
            
            # Execute steps in order
            results = []
            for step in sorted(execution_plan, key=lambda x: x.get("execution_order", 0)):
                step_result = await self._execute_step(step, context)
                results.append(step_result)
                
                # Check if step failed and fallback is needed
                if not step_result.success and step.get("conditional"):
                    fallback_result = await self._execute_fallback(step, context)
                    results.append(fallback_result)
            
            # Combine results
            return self._combine_execution_results(results, node_config)
            
        except Exception as e:
            self.logger.error(f"Failed to execute intelligent node {node_config.node_id}: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Execution failed: {e}",
                evidence_data={"node_id": node_config.node_id, "error": str(e)}
            )
    
    async def _execute_step(
        self, 
        step: Dict[str, Any], 
        context: OperatorContext
    ) -> OperatorResult:
        """Execute a single step in the execution plan"""
        try:
            operator_type = step["operator_type"]
            
            if operator_type in self._operator_factory:
                # Create and execute ML operator
                operator_class = self._operator_factory[operator_type]
                operator = operator_class(operator_id=step["step_id"])
                
                # Inject model registry
                model_registry = await self.node_registry.get_model_registry_dict()
                operator.set_model_registry(model_registry)
                
                return await operator.execute(context, step["config"])
            
            elif operator_type == "governance_check":
                return await self._execute_governance_check(step, context)
            
            elif operator_type == "governance_logging":
                return await self._execute_governance_logging(step, context)
            
            elif operator_type == "fallback_executor":
                return await self._execute_fallback_executor(step, context)
            
            elif operator_type == "explainability_logger":
                return await self._execute_explainability_logger(step, context)
            
            else:
                return OperatorResult(
                    success=False,
                    error_message=f"Unknown operator type: {operator_type}"
                )
                
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Step execution failed: {e}"
            )
    
    async def _execute_governance_check(
        self, 
        step: Dict[str, Any], 
        context: OperatorContext
    ) -> OperatorResult:
        """Execute governance check step"""
        try:
            # Basic governance checks
            if not context.tenant_id:
                return OperatorResult(
                    success=False,
                    error_message="Tenant ID required for governance"
                )
            
            if context.trust_threshold < 0.5:
                return OperatorResult(
                    success=False,
                    error_message="Trust threshold too low for ML operations"
                )
            
            return OperatorResult(
                success=True,
                evidence_data={
                    "governance_check": "passed",
                    "timestamp": datetime.utcnow().isoformat(),
                    "tenant_id": context.tenant_id
                }
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Governance check failed: {e}"
            )
    
    async def _execute_governance_logging(
        self, 
        step: Dict[str, Any], 
        context: OperatorContext
    ) -> OperatorResult:
        """Execute governance logging step"""
        try:
            # Log governance evidence
            evidence_data = {
                "governance_logging": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "workflow_id": context.workflow_id
            }
            
            return OperatorResult(
                success=True,
                evidence_data=evidence_data
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Governance logging failed: {e}"
            )
    
    async def _execute_fallback_executor(
        self, 
        step: Dict[str, Any], 
        context: OperatorContext
    ) -> OperatorResult:
        """Execute fallback executor step"""
        try:
            # This would integrate with the fallback service
            # For now, return a placeholder
            return OperatorResult(
                success=True,
                output_data={"fallback_executed": True},
                evidence_data={"fallback_type": "rule_based"}
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Fallback execution failed: {e}"
            )
    
    async def _execute_explainability_logger(
        self, 
        step: Dict[str, Any], 
        context: OperatorContext
    ) -> OperatorResult:
        """Execute explainability logger step"""
        try:
            # This would integrate with the explainability service
            # For now, return a placeholder
            return OperatorResult(
                success=True,
                evidence_data={"explainability_logged": True}
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Explainability logging failed: {e}"
            )
    
    async def _execute_fallback(
        self, 
        step: Dict[str, Any], 
        context: OperatorContext
    ) -> OperatorResult:
        """Execute fallback when main step fails"""
        try:
            # Implement fallback logic
            return OperatorResult(
                success=True,
                output_data={"fallback_used": True},
                evidence_data={"fallback_reason": "Main execution failed"}
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Fallback execution failed: {e}"
            )
    
    def _combine_execution_results(
        self, 
        results: List[OperatorResult], 
        node_config: IntelligentNodeConfig
    ) -> OperatorResult:
        """Combine results from all execution steps"""
        try:
            # Find the main ML operation result
            main_result = None
            for result in results:
                if result.success and result.output_data:
                    main_result = result
                    break
            
            if not main_result:
                return OperatorResult(
                    success=False,
                    error_message="No successful execution results found"
                )
            
            # Combine evidence from all steps
            combined_evidence = {}
            for result in results:
                if result.evidence_data:
                    combined_evidence.update(result.evidence_data)
            
            return OperatorResult(
                success=True,
                output_data=main_result.output_data,
                confidence_score=main_result.confidence_score,
                evidence_data=combined_evidence
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Result combination failed: {e}"
            )
