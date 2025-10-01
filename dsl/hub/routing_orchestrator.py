 #!/usr/bin/env python3
"""
Clean Routing Orchestrator - Enterprise Grade
============================================

Clean, focused routing orchestrator following build plan architecture.
YAML-first, no hardcoded logic, proper separation of concerns.
"""

import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Import clean components
from .yaml_workflow_loader import YAMLWorkflowLoader, get_workflow_loader
from .intent_parser_v2 import CleanIntentParser, ParsedIntent
from .parameter_mapper_v2 import CleanParameterMapper, get_parameter_mapper
from .dsl_execution_engine import DSLExecutionEngine, get_execution_engine

logger = logging.getLogger(__name__)

@dataclass
class RoutingResult:
    """Clean routing result structure"""
    request_id: str
    success: bool
    workflow_category: str
    confidence: float
    execution_result: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None

class CleanRoutingOrchestrator:
    """
    Clean Routing Orchestrator
    
    Enterprise-grade routing with:
    - YAML-first workflow loading (single source of truth)
    - Dynamic intent parsing (no hardcoded workflows)
    - Real DSL execution (no mocks)
    - Clean architecture (separation of concerns)
    - Build plan compliance
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize clean components
        self.workflow_loader = get_workflow_loader()
        self.intent_parser = CleanIntentParser(pool_manager, self.workflow_loader)
        self.parameter_mapper = get_parameter_mapper()
        self.execution_engine = get_execution_engine(pool_manager)
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time_ms": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the clean routing orchestrator"""
        try:
            # Initialize all components
            if not self.workflow_loader.initialize():
                self.logger.error("❌ Failed to initialize workflow loader")
                return False
            
            if not await self.execution_engine.initialize():
                self.logger.error("❌ Failed to initialize execution engine")
                return False
            
            available_workflows = self.workflow_loader.get_available_workflows()
            self.logger.info(f"✅ Clean Routing Orchestrator initialized with {len(available_workflows)} workflows")
            self.logger.info(f"📋 Available workflows: {available_workflows}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Clean Routing Orchestrator: {e}")
            return False
    
    async def route_request(
        self, 
        user_input: str,
        tenant_id: str = "1300",
        user_id: str = "1319",
        context_data: Dict[str, Any] = None
    ) -> RoutingResult:
        """
        Single entry point for all routing requests
        
        This is the main method that handles the complete flow:
        1. Parse intent + extract parameters
        2. Load workflow from YAML
        3. Map parameters to configuration
        4. Execute workflow with real DSL engine
        
        Args:
            user_input: Natural language request
            tenant_id: Tenant identifier
            user_id: User identifier
            context_data: Additional context
            
        Returns:
            RoutingResult with execution results
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🚀 Processing routing request: '{user_input[:50]}...' (ID: {request_id})")
            self.metrics["total_requests"] += 1
            
            # Step 1: Parse intent and extract parameters
            self.logger.info("🔍 Step 1: Parsing intent and extracting parameters")
            parsed_intent = await self.intent_parser.parse_intent(user_input, tenant_id)
            
            self.logger.info(f"✅ Intent parsed - Category: {parsed_intent.workflow_category}, Confidence: {parsed_intent.confidence}")
            self.logger.info(f"📊 Extracted parameters: {parsed_intent.parameters}")
            
            # Step 2: Load workflow from YAML (single source of truth)
            self.logger.info("📁 Step 2: Loading workflow from YAML")
            workflow_def = self.workflow_loader.load_workflow(parsed_intent.workflow_category)
            
            if not workflow_def:
                error_msg = f"Workflow not found in YAML: {parsed_intent.workflow_category}"
                self.logger.error(f"❌ {error_msg}")
                self.metrics["failed_executions"] += 1
                return RoutingResult(
                    request_id=request_id,
                    success=False,
                    workflow_category=parsed_intent.workflow_category,
                    confidence=parsed_intent.confidence,
                    execution_result={},
                    error=error_msg
                )
            
            self.logger.info(f"✅ Workflow loaded: {workflow_def.get('name', 'Unknown')}")
            
            # Step 3: Map parameters to workflow configuration
            self.logger.info("🔧 Step 3: Mapping parameters to workflow configuration")
            mapped_config = self.parameter_mapper.map_parameters(
                parsed_intent.parameters,
                parsed_intent.workflow_category
            )
            
            # Add context data to mapped config
            if context_data:
                mapped_config.update(context_data)
            
            # Step 4: Execute workflow with real DSL engine
            self.logger.info("⚡ Step 4: Executing workflow with DSL engine")
            execution_result = await self.execution_engine.execute_workflow(
                workflow_def=workflow_def,
                mapped_config=mapped_config,
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update metrics
            if execution_result.get("success", False):
                self.metrics["successful_executions"] += 1
                self.logger.info(f"✅ Workflow execution completed successfully in {execution_time_ms}ms")
            else:
                self.metrics["failed_executions"] += 1
                self.logger.error(f"❌ Workflow execution failed: {execution_result.get('error', 'Unknown error')}")
            
            # Update average execution time
            total_executions = self.metrics["successful_executions"] + self.metrics["failed_executions"]
            if total_executions > 0:
                self.metrics["avg_execution_time_ms"] = (
                    (self.metrics["avg_execution_time_ms"] * (total_executions - 1) + execution_time_ms) / total_executions
                )
            
            return RoutingResult(
                request_id=request_id,
                success=execution_result.get("success", False),
                workflow_category=parsed_intent.workflow_category,
                confidence=parsed_intent.confidence,
                execution_result=execution_result,
                error=execution_result.get("error"),
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"❌ Routing request failed: {e}")
            self.metrics["failed_executions"] += 1
            
            return RoutingResult(
                request_id=request_id,
                success=False,
                workflow_category="unknown",
                confidence=0.0,
                execution_result={},
                error=str(e)
            )
    
    def get_available_workflows(self) -> Dict[str, Any]:
        """Get information about available workflows"""
        try:
            categories = self.workflow_loader.get_available_workflows()
            workflows_info = {}
            
            for category in categories:
                metadata = self.workflow_loader.get_workflow_metadata(category)
                workflows_info[category] = {
                    "name": metadata.name if metadata else category,
                    "description": metadata.description if metadata else f"Execute {category} workflow",
                    "automation_type": metadata.automation_type if metadata else "RBA",
                    "version": metadata.version if metadata else "1.0"
                }
            
            return {
                "total_workflows": len(categories),
                "workflows": workflows_info,
                "loader_status": "active"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get available workflows: {e}")
            return {"total_workflows": 0, "workflows": {}, "loader_status": "error", "error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_executions"] / max(1, self.metrics["total_requests"])
            ) * 100,
            "engine_stats": self.execution_engine.get_execution_stats()
        }
    
    def reload_workflows(self) -> bool:
        """Reload workflows from YAML files"""
        try:
            success = self.workflow_loader.reload_workflows()
            if success:
                self.logger.info("✅ Workflows reloaded successfully")
            else:
                self.logger.error("❌ Failed to reload workflows")
            return success
        except Exception as e:
            self.logger.error(f"❌ Error reloading workflows: {e}")
            return False

# Global instance for backward compatibility
_clean_orchestrator = None

def get_clean_orchestrator(pool_manager=None) -> CleanRoutingOrchestrator:
    """Get global clean orchestrator instance"""
    global _clean_orchestrator
    if _clean_orchestrator is None:
        _clean_orchestrator = CleanRoutingOrchestrator(pool_manager)
    return _clean_orchestrator

# Backward compatibility wrapper
class RoutingOrchestrator(CleanRoutingOrchestrator):
    """
    Backward compatibility wrapper for existing code
    
    This allows existing code to continue working while using the clean architecture.
    """
    
    async def route_request_legacy(self, *args, **kwargs):
        """Legacy method wrapper"""
        return await self.route_request(*args, **kwargs)
