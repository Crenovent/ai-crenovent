#!/usr/bin/env python3
"""
Integration Orchestrator - Connects all system components
Ensures proper connectivity between DSL, Knowledge Graph, Routing, and Governance
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from dsl.orchestrator import DSLOrchestrator
from dsl.hub.routing_orchestrator import RoutingOrchestrator
from dsl.hub.execution_hub import ExecutionHub
from dsl.knowledge.kg_store import KnowledgeGraphStore
from dsl.governance.multi_tenant_taxonomy import MultiTenantEnforcementEngine
from dsl.intelligence.trust_scoring_engine import TrustScoringEngine
from dsl.intelligence.sla_monitoring_system import SLAMonitoringSystem
from dsl.capability_registry.dynamic_capability_orchestrator import DynamicCapabilityOrchestrator
try:
    from src.services.connection_pool_manager import ConnectionPoolManager
except ImportError:
    from services.connection_pool_manager import ConnectionPoolManager

logger = logging.getLogger(__name__)

class IntegrationOrchestrator:
    """
    Master orchestrator that connects all system components
    Ensures proper initialization, connectivity, and error handling
    """
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.dsl_orchestrator = None
        self.routing_orchestrator = None
        self.execution_hub = None
        self.kg_store = None
        self.tenant_engine = None
        self.trust_engine = None
        self.sla_monitor = None
        self.capability_orchestrator = None
        
        # Integration status
        self.initialized = False
        self.health_status = {}
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all system components with proper connectivity"""
        
        try:
            self.logger.info("🚀 Starting Integration Orchestrator initialization...")
            
            # 1. Initialize Knowledge Graph Store (foundation)
            self.logger.info("📊 Initializing Knowledge Graph Store...")
            self.kg_store = KnowledgeGraphStore(self.pool_manager)
            await self.kg_store.initialize()
            
            # 2. Initialize Multi-Tenant Enforcement Engine
            self.logger.info("🔒 Initializing Multi-Tenant Enforcement...")
            self.tenant_engine = MultiTenantEnforcementEngine(self.pool_manager)
            # Engine has internal initialization
            
            # 3. Initialize DSL Orchestrator
            self.logger.info("⚙️ Initializing DSL Orchestrator...")
            self.dsl_orchestrator = DSLOrchestrator(self.pool_manager)
            
            # 4. Initialize Routing Orchestrator with LLM/NLP
            self.logger.info("🧠 Initializing Routing Orchestrator with LLM...")
            self.routing_orchestrator = RoutingOrchestrator(
                pool_manager=self.pool_manager,
                dsl_orchestrator=self.dsl_orchestrator
            )
            await self.routing_orchestrator.initialize()
            
            # 5. Initialize Execution Hub
            self.logger.info("🎯 Initializing Execution Hub...")
            self.execution_hub = ExecutionHub(self.pool_manager)
            await self.execution_hub.initialize()
            
            # 6. Initialize Intelligence Systems
            self.logger.info("🤖 Initializing Trust Scoring Engine...")
            self.trust_engine = TrustScoringEngine(self.pool_manager)
            await self.trust_engine.initialize()
            
            self.logger.info("📈 Initializing SLA Monitoring System...")
            self.sla_monitor = SLAMonitoringSystem(self.pool_manager)
            await self.sla_monitor.initialize()
            
            # 7. Initialize Dynamic Capability Orchestrator
            self.logger.info("🔄 Initializing Dynamic Capability Orchestrator...")
            self.capability_orchestrator = DynamicCapabilityOrchestrator(self.pool_manager)
            await self.capability_orchestrator.initialize()
            
            # 8. Establish cross-component connectivity
            await self._establish_connectivity()
            
            # 9. Run health checks
            health_results = await self._run_health_checks()
            
            self.initialized = True
            
            initialization_result = {
                "status": "success",
                "initialized_at": datetime.utcnow().isoformat(),
                "components": {
                    "kg_store": "initialized",
                    "tenant_engine": "initialized", 
                    "dsl_orchestrator": "initialized",
                    "routing_orchestrator": "initialized",
                    "execution_hub": "initialized",
                    "trust_engine": "initialized",
                    "sla_monitor": "initialized",
                    "capability_orchestrator": "initialized"
                },
                "health_checks": health_results,
                "llm_nlp_enabled": True,
                "conversational_mode": True
            }
            
            self.logger.info("✅ Integration Orchestrator initialization complete!")
            return initialization_result
            
        except Exception as e:
            self.logger.error(f"❌ Integration Orchestrator initialization failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "initialized_components": self._get_initialized_components()
            }
    
    async def _establish_connectivity(self):
        """Establish connectivity between components"""
        
        # Connect DSL Orchestrator with Knowledge Graph
        if hasattr(self.dsl_orchestrator, 'kg_store'):
            self.dsl_orchestrator.kg_store = self.kg_store
            
        # Connect Routing Orchestrator with DSL Orchestrator
        if hasattr(self.routing_orchestrator, 'dsl_orchestrator'):
            self.routing_orchestrator.dsl_orchestrator = self.dsl_orchestrator
            
        # Connect Execution Hub with all orchestrators
        if hasattr(self.execution_hub, 'dsl_orchestrator'):
            self.execution_hub.dsl_orchestrator = self.dsl_orchestrator
        if hasattr(self.execution_hub, 'routing_orchestrator'):
            self.execution_hub.routing_orchestrator = self.routing_orchestrator
            
        self.logger.info("🔗 Component connectivity established")
    
    async def _run_health_checks(self) -> Dict[str, Any]:
        """Run health checks on all components"""
        
        health_results = {}
        
        try:
            # Knowledge Graph Store health
            kg_stats = await self.kg_store.get_knowledge_stats("1300")  # Test tenant
            health_results["kg_store"] = {
                "status": "healthy",
                "entities_count": kg_stats.get("total_entities", 0)
            }
        except Exception as e:
            health_results["kg_store"] = {"status": "error", "error": str(e)}
        
        try:
            # Routing Orchestrator health (test LLM integration)
            test_result = await self.routing_orchestrator.route_request(
                user_input="test system health",
                tenant_id=1300,
                user_id=1319,
                context_data={"test": True}
            )
            health_results["routing_orchestrator"] = {
                "status": "healthy",
                "llm_enabled": True,
                "test_confidence": test_result.parsed_intent.confidence
            }
        except Exception as e:
            health_results["routing_orchestrator"] = {"status": "error", "error": str(e)}
        
        try:
            # Execution Hub health
            hub_health = await self.execution_hub.health_check()
            health_results["execution_hub"] = hub_health
        except Exception as e:
            health_results["execution_hub"] = {"status": "error", "error": str(e)}
        
        try:
            # Trust Engine health
            trust_overview = await self.trust_engine.get_tenant_trust_overview(1300)
            health_results["trust_engine"] = {
                "status": "healthy",
                "capabilities_tracked": len(trust_overview.get("capability_scores", []))
            }
        except Exception as e:
            health_results["trust_engine"] = {"status": "error", "error": str(e)}
        
        return health_results
    
    def _get_initialized_components(self) -> List[str]:
        """Get list of successfully initialized components"""
        components = []
        if self.kg_store: components.append("kg_store")
        if self.tenant_engine: components.append("tenant_engine")
        if self.dsl_orchestrator: components.append("dsl_orchestrator")
        if self.routing_orchestrator: components.append("routing_orchestrator")
        if self.execution_hub: components.append("execution_hub")
        if self.trust_engine: components.append("trust_engine")
        if self.sla_monitor: components.append("sla_monitor")
        if self.capability_orchestrator: components.append("capability_orchestrator")
        return components
    
    async def route_and_execute(
        self,
        user_input: str,
        tenant_id: int,
        user_id: int,
        context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end processing:
        1. Route user input using LLM/NLP
        2. Execute via appropriate automation type (RBA/RBIA/AALA)
        3. Capture knowledge and update trust scores
        """
        
        if not self.initialized:
            raise RuntimeError("Integration Orchestrator not initialized")
        
        try:
            # 1. Route request using LLM/NLP
            self.logger.info(f"🧠 Routing request: '{user_input}' for tenant {tenant_id}")
            routing_result = await self.routing_orchestrator.route_request(
                user_input=user_input,
                tenant_id=tenant_id,
                user_id=user_id,
                context_data=context_data or {}
            )
            
            # 2. Execute the routed workflow
            self.logger.info(f"⚙️ Executing {routing_result.selected_capability.capability_type} workflow")
            execution_result = await self.routing_orchestrator.execute_routing_result(routing_result)
            
            # 3. Capture execution knowledge
            if execution_result.status == "completed":
                await self.execution_hub.capture_execution_knowledge({
                    "workflow_id": routing_result.selected_capability.capability_id,
                    "user_id": user_id,
                    "tenant_id": str(tenant_id),
                    "execution_time_ms": execution_result.execution_time_ms,
                    "success": True,
                    "input_data": {"user_input": user_input},
                    "output_data": execution_result.final_result,
                    "user_context": context_data or {}
                })
            
            # 4. Update trust scores
            if self.trust_engine:
                await self.trust_engine.calculate_trust_score(
                    routing_result.selected_capability.capability_id,
                    tenant_id
                )
            
            return {
                "status": "success",
                "routing_result": {
                    "intent_type": routing_result.parsed_intent.intent_type.value,
                    "confidence": routing_result.parsed_intent.confidence,
                    "selected_capability": routing_result.selected_capability.name,
                    "automation_type": routing_result.selected_capability.capability_type
                },
                "execution_result": {
                    "status": execution_result.status,
                    "execution_time_ms": execution_result.execution_time_ms,
                    "output": execution_result.output_data or {}
                },
                "llm_reasoning": routing_result.parsed_intent.llm_reasoning
            }
            
        except Exception as e:
            self.logger.error(f"❌ End-to-end processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "user_input": user_input
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        if not self.initialized:
            return {"status": "not_initialized"}
        
        # Get fresh health checks
        health_results = await self._run_health_checks()
        
        return {
            "status": "operational" if self.initialized else "initializing",
            "initialized": self.initialized,
            "components": self._get_initialized_components(),
            "health_checks": health_results,
            "capabilities": {
                "llm_nlp_enabled": True,
                "conversational_mode": True,
                "multi_tenant_isolation": True,
                "knowledge_graph_enabled": True,
                "trust_scoring_enabled": True,
                "sla_monitoring_enabled": True,
                "dynamic_capabilities": True
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        
        self.logger.info("🛑 Shutting down Integration Orchestrator...")
        
        # Shutdown components in reverse order
        if self.execution_hub:
            await self.execution_hub.shutdown()
        
        if self.sla_monitor and hasattr(self.sla_monitor, 'shutdown'):
            await self.sla_monitor.shutdown()
        
        self.initialized = False
        self.logger.info("✅ Integration Orchestrator shutdown complete")

# Global instance for API access
_integration_orchestrator: Optional[IntegrationOrchestrator] = None

async def get_integration_orchestrator() -> IntegrationOrchestrator:
    """Get or create the global integration orchestrator"""
    global _integration_orchestrator
    
    if _integration_orchestrator is None:
        try:
            from src.services.connection_pool_manager import ConnectionPoolManager
        except ImportError:
            from services.connection_pool_manager import ConnectionPoolManager
            
        pool_manager = ConnectionPoolManager()
        await pool_manager.initialize()
        
        _integration_orchestrator = IntegrationOrchestrator(pool_manager)
        await _integration_orchestrator.initialize()
    
    return _integration_orchestrator

async def initialize_system() -> Dict[str, Any]:
    """Initialize the complete integrated system"""
    orchestrator = await get_integration_orchestrator()
    return await orchestrator.get_system_status()
