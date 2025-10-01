"""
Dynamic RBA Agent Registry
Automatically discovers and registers RBA agents without hardcoding

This registry dynamically discovers all RBA agents and provides
a clean interface for the orchestrator to select the right agent
based on analysis type, without any hardcoded dependencies.
"""

import logging
import importlib
import inspect
import os
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RBAAgentInfo:
    """Information about a registered RBA agent"""
    agent_name: str
    agent_type: str
    agent_description: str
    supported_analysis_types: List[str]
    class_name: str
    module_path: str
    agent_class: Type
    priority: int = 100  # Lower number = higher priority

class RBAAgentRegistry:
    """
    Dynamic RBA Agent Registry
    
    Features:
    - Automatic agent discovery (no hardcoding)
    - Dynamic agent loading and instantiation
    - Analysis type to agent mapping
    - Agent metadata management
    - Priority-based agent selection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._agents: Dict[str, RBAAgentInfo] = {}
        self._analysis_type_mapping: Dict[str, List[str]] = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the registry by discovering all RBA agents"""
        
        if self._initialized:
            return True
            
        try:
            self.logger.info("🔍 Initializing RBA Agent Registry...")
            
            # Discover agents from the RBA module
            self._discover_rba_agents()
            
            # Build analysis type mappings
            self._build_analysis_type_mappings()
            
            self._initialized = True
            
            self.logger.info(f"✅ RBA Registry initialized: {len(self._agents)} agents registered")
            self.logger.info(f"📊 Analysis types supported: {len(self._analysis_type_mapping)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RBA registry: {e}")
            return False
    
    def _discover_rba_agents(self):
        """Automatically discover RBA agents"""
        
        try:
            # Import the RBA agents module
            from ..operators.rba import (
                SandbaggingRBAAgent,
                StaleDealsRBAAgent,
                MissingFieldsRBAAgent,
                ActivityTrackingRBAAgent,
                DuplicateDetectionRBAAgent,
                OwnerlessDealsRBAAgent,
                QuarterEndDumpingRBAAgent,
                DealRiskScoringRBAAgent,
                DealsAtRiskRBAAgent,
                StageVelocityRBAAgent,
                PipelineSummaryRBAAgent,
                DataQualityRBAAgent,
                ForecastAlignmentRBAAgent,
                CoverageAnalysisRBAAgent,
                HealthOverviewRBAAgent
            )
            from ..operators.rba.onboarding_rba_agent import OnboardingRBAAgent
            from ..operators.rba.smart_location_mapper import SmartLocationMapperRBAAgent
            
            # List of agent classes to register
            agent_classes = [
                SandbaggingRBAAgent,
                StaleDealsRBAAgent,
                MissingFieldsRBAAgent,
                ActivityTrackingRBAAgent,
                DuplicateDetectionRBAAgent,
                OwnerlessDealsRBAAgent,
                QuarterEndDumpingRBAAgent,
                DealRiskScoringRBAAgent,
                DealsAtRiskRBAAgent,
                StageVelocityRBAAgent,
                PipelineSummaryRBAAgent,
                DataQualityRBAAgent,
                ForecastAlignmentRBAAgent,
                CoverageAnalysisRBAAgent,
                HealthOverviewRBAAgent,
                OnboardingRBAAgent,
                SmartLocationMapperRBAAgent
            ]
            
            # Register each agent
            for agent_class in agent_classes:
                self._register_agent_class(agent_class)
                
        except Exception as e:
            self.logger.error(f"Failed to discover RBA agents: {e}")
    
    def _register_agent_class(self, agent_class: Type):
        """Register a single agent class"""
        
        try:
            # Get agent metadata
            if hasattr(agent_class, 'get_agent_metadata'):
                metadata = agent_class.get_agent_metadata()
            else:
                # Fallback to class attributes
                # Handle both property and class attribute for SUPPORTED_ANALYSIS_TYPES
                supported_types = []
                if hasattr(agent_class, 'SUPPORTED_ANALYSIS_TYPES'):
                    types_attr = getattr(agent_class, 'SUPPORTED_ANALYSIS_TYPES')
                    if isinstance(types_attr, property):
                        # It's a property, create instance to get value
                        try:
                            instance = agent_class()
                            supported_types = instance.SUPPORTED_ANALYSIS_TYPES
                        except:
                            supported_types = [getattr(agent_class, 'AGENT_NAME', agent_class.__name__)]
                    elif isinstance(types_attr, list):
                        # It's a class attribute list
                        supported_types = types_attr
                    else:
                        supported_types = [getattr(agent_class, 'AGENT_NAME', agent_class.__name__)]
                
                metadata = {
                    'agent_name': getattr(agent_class, 'AGENT_NAME', agent_class.__name__),
                    'agent_type': getattr(agent_class, 'AGENT_TYPE', 'RBA'),
                    'agent_description': getattr(agent_class, 'AGENT_DESCRIPTION', 'No description'),
                    'supported_analysis_types': supported_types,
                    'class_name': agent_class.__name__,
                    'module_path': agent_class.__module__
                }
            
            # Create agent info
            agent_info = RBAAgentInfo(
                agent_name=metadata['agent_name'],
                agent_type=metadata['agent_type'],
                agent_description=metadata['agent_description'],
                supported_analysis_types=metadata['supported_analysis_types'],
                class_name=metadata['class_name'],
                module_path=metadata['module_path'],
                agent_class=agent_class
            )
            
            # Register the agent
            self._agents[agent_info.agent_name] = agent_info
            
            self.logger.info(f"✅ Registered RBA agent: {agent_info.agent_name} ({len(agent_info.supported_analysis_types)} analysis types)")
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_class.__name__}: {e}")
    
    def _build_analysis_type_mappings(self):
        """Build mappings from analysis types to agents"""
        
        self._analysis_type_mapping.clear()
        
        for agent_name, agent_info in self._agents.items():
            for analysis_type in agent_info.supported_analysis_types:
                if analysis_type not in self._analysis_type_mapping:
                    self._analysis_type_mapping[analysis_type] = []
                
                self._analysis_type_mapping[analysis_type].append(agent_name)
        
        self.logger.info(f"📊 Built analysis type mappings: {list(self._analysis_type_mapping.keys())}")
    
    def get_agent_for_analysis_type(self, analysis_type: str) -> Optional[RBAAgentInfo]:
        """Get the best agent for a specific analysis type"""
        
        if not self._initialized:
            self.initialize()
        
        if analysis_type not in self._analysis_type_mapping:
            self.logger.warning(f"⚠️ No agent found for analysis type: {analysis_type}")
            return None
        
        # Get agents that support this analysis type
        agent_names = self._analysis_type_mapping[analysis_type]
        
        if not agent_names:
            return None
        
        # Return the first (highest priority) agent
        # In the future, we could implement priority-based selection
        agent_name = agent_names[0]
        return self._agents.get(agent_name)
    
    def create_agent_instance(self, analysis_type: str, config: Dict[str, Any] = None) -> Optional[Any]:
        """Create an agent instance for the specified analysis type"""
        
        agent_info = self.get_agent_for_analysis_type(analysis_type)
        
        if not agent_info:
            self.logger.error(f"❌ Cannot create agent for analysis type: {analysis_type}")
            return None
        
        try:
            # Create agent instance
            agent_instance = agent_info.agent_class(config or {})
            
            self.logger.info(f"✅ Created {agent_info.agent_name} agent for {analysis_type}")
            return agent_instance
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create agent {agent_info.agent_name}: {e}")
            return None
    
    def get_all_agents(self) -> Dict[str, RBAAgentInfo]:
        """Get all registered agents"""
        
        if not self._initialized:
            self.initialize()
            
        return self._agents.copy()
    
    def get_supported_analysis_types(self) -> List[str]:
        """Get all supported analysis types"""
        
        if not self._initialized:
            self.initialize()
            
        return list(self._analysis_type_mapping.keys())
    
    def get_agent_info(self, agent_name: str) -> Optional[RBAAgentInfo]:
        """Get information about a specific agent"""
        
        if not self._initialized:
            self.initialize()
            
        return self._agents.get(agent_name)
    
    def register_custom_agent(self, agent_class: Type, priority: int = 100):
        """Register a custom agent class"""
        
        try:
            self._register_agent_class(agent_class)
            
            # Update priority if specified
            if hasattr(agent_class, 'AGENT_NAME'):
                agent_name = agent_class.AGENT_NAME
                if agent_name in self._agents:
                    self._agents[agent_name].priority = priority
            
            # Rebuild mappings
            self._build_analysis_type_mappings()
            
            self.logger.info(f"✅ Custom agent registered: {agent_class.__name__}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register custom agent: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        if not self._initialized:
            self.initialize()
            
        total_analysis_types = sum(len(info.supported_analysis_types) for info in self._agents.values())
        
        return {
            'total_agents': len(self._agents),
            'total_analysis_types': len(self._analysis_type_mapping),
            'total_agent_capabilities': total_analysis_types,
            'agents': {name: {
                'analysis_types': info.supported_analysis_types,
                'description': info.agent_description
            } for name, info in self._agents.items()},
            'analysis_type_coverage': self._analysis_type_mapping
        }

# Global registry instance
rba_registry = RBAAgentRegistry()
