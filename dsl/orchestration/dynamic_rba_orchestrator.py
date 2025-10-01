"""
Dynamic RBA Orchestrator
Dynamically selects and executes RBA agents without hardcoding

This orchestrator uses the RBA registry to dynamically discover
and select the appropriate agent based on analysis type.
NO hardcoded agent imports or selections.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..registry.rba_agent_registry import rba_registry

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
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.registry = rba_registry
        
        # Initialize registry
        if not self.registry.initialize():
            self.logger.error("❌ Failed to initialize RBA registry")
    
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
