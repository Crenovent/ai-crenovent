"""
RBA Agent Auto-Discovery System
===============================
Automatically discovers and registers all RBA agents in the system.
Provides dynamic loading, configuration management, and metadata extraction.
"""

import os
import importlib
import inspect
import logging
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentMetadata:
    """Metadata for a discovered RBA agent"""
    name: str
    class_name: str
    module_path: str
    category: str
    description: str
    supported_analysis_types: List[str]
    config_schema: Optional[Dict[str, Any]] = None
    result_schema: Optional[Dict[str, Any]] = None

class RBAAgentDiscovery:
    """
    Auto-discovery system for RBA agents.
    
    Features:
    - Scans RBA agent directory for *_rba_agent.py files
    - Extracts agent metadata and configuration schemas
    - Provides dynamic loading and instantiation
    - Caches discovery results for performance
    """
    
    def __init__(self, agents_directory: str = None):
        self.agents_directory = agents_directory or "dsl/operators/rba"
        self.discovered_agents: Dict[str, AgentMetadata] = {}
        self.loaded_classes: Dict[str, Type] = {}
        self._discovery_cache_valid = False
    
    def discover_agents(self, force_refresh: bool = False) -> Dict[str, AgentMetadata]:
        """
        Discover all RBA agents in the agents directory.
        
        Args:
            force_refresh: Force re-discovery even if cache is valid
            
        Returns:
            Dictionary of agent_name -> AgentMetadata
        """
        if not force_refresh and self._discovery_cache_valid:
            return self.discovered_agents
        
        logger.info(f"🔍 Discovering RBA agents in {self.agents_directory}")
        
        self.discovered_agents.clear()
        self.loaded_classes.clear()
        
        # Get absolute path to agents directory
        agents_path = Path(self.agents_directory)
        if not agents_path.exists():
            logger.error(f"❌ Agents directory not found: {agents_path}")
            return {}
        
        # Scan for *_rba_agent.py files
        agent_files = list(agents_path.glob("*_rba_agent.py"))
        logger.info(f"📁 Found {len(agent_files)} potential agent files")
        
        for agent_file in agent_files:
            try:
                self._discover_agent_from_file(agent_file)
            except Exception as e:
                logger.error(f"❌ Failed to discover agent from {agent_file}: {e}")
        
        self._discovery_cache_valid = True
        logger.info(f"✅ Discovered {len(self.discovered_agents)} RBA agents")
        
        return self.discovered_agents
    
    def _discover_agent_from_file(self, agent_file: Path) -> None:
        """
        Discover agent metadata from a single file.
        
        Args:
            agent_file: Path to the agent file
        """
        # Convert file path to module path
        try:
            relative_path = agent_file.relative_to(Path.cwd())
        except ValueError:
            # Handle case where file is not relative to cwd
            relative_path = agent_file.relative_to(Path(self.agents_directory).parent.parent)
        module_path = str(relative_path).replace(os.sep, '.').replace('.py', '')
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find RBA agent classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_rba_agent_class(obj, module_path):
                    metadata = self._extract_agent_metadata(obj, module_path)
                    if metadata:
                        self.discovered_agents[metadata.name] = metadata
                        self.loaded_classes[metadata.name] = obj
                        logger.info(f"📋 Discovered agent: {metadata.name} ({metadata.category})")
        
        except Exception as e:
            logger.error(f"❌ Failed to import module {module_path}: {e}")
    
    def _is_rba_agent_class(self, cls: Type, module_path: str) -> bool:
        """
        Check if a class is a valid RBA agent.
        
        Args:
            cls: Class to check
            module_path: Module path for the class
            
        Returns:
            True if it's a valid RBA agent class
        """
        # Skip the base class itself
        if cls.__name__ == 'BaseRBAAgent':
            return False
        
        # Check if it's defined in this module (not imported)
        if cls.__module__ != module_path:
            return False
        
        # Check if it has RBA agent characteristics
        has_agent_name = hasattr(cls, 'AGENT_NAME')
        has_execute_method = hasattr(cls, 'execute')
        ends_with_agent = cls.__name__.endswith('Agent')
        
        return has_agent_name and has_execute_method and ends_with_agent
    
    def _extract_agent_metadata(self, cls: Type, module_path: str) -> Optional[AgentMetadata]:
        """
        Extract metadata from an RBA agent class.
        
        Args:
            cls: RBA agent class
            module_path: Module path for the class
            
        Returns:
            AgentMetadata if extraction successful, None otherwise
        """
        try:
            # Extract basic metadata
            agent_name = getattr(cls, 'AGENT_NAME', cls.__name__.lower())
            description = getattr(cls, 'AGENT_DESCRIPTION', cls.__doc__ or f"{cls.__name__} agent")
            supported_types = getattr(cls, 'SUPPORTED_ANALYSIS_TYPES', [agent_name])
            
            # Determine category from agent name or class attributes
            category = self._determine_category(agent_name, cls)
            
            # Try to extract configuration schema
            config_schema = None
            result_schema = None
            
            try:
                # Create a temporary instance to extract schemas
                temp_instance = cls({})
                if hasattr(temp_instance, 'get_config_schema'):
                    config_schema = temp_instance.get_config_schema()
                if hasattr(temp_instance, 'get_result_schema'):
                    result_schema = temp_instance.get_result_schema()
            except Exception as e:
                logger.warning(f"⚠️ Could not extract schemas from {agent_name}: {e}")
            
            return AgentMetadata(
                name=agent_name,
                class_name=cls.__name__,
                module_path=module_path,
                category=category,
                description=description.strip(),
                supported_analysis_types=supported_types,
                config_schema=config_schema,
                result_schema=result_schema
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to extract metadata from {cls.__name__}: {e}")
            return None
    
    def _determine_category(self, agent_name: str, cls: Type) -> str:
        """
        Determine the category of an agent based on its name and characteristics.
        
        Args:
            agent_name: Name of the agent
            cls: Agent class
            
        Returns:
            Category string
        """
        # Check for explicit category attribute
        if hasattr(cls, 'AGENT_CATEGORY'):
            return getattr(cls, 'AGENT_CATEGORY')
        
        # Infer from agent name
        name_lower = agent_name.lower()
        
        if any(keyword in name_lower for keyword in ['missing', 'duplicate', 'ownerless', 'data_quality', 'activity_tracking']):
            return 'data_quality'
        elif any(keyword in name_lower for keyword in ['risk', 'sandbagging', 'deals_at_risk', 'stale']):
            return 'risk_analysis'
        elif any(keyword in name_lower for keyword in ['velocity', 'stage', 'conversion', 'forecast']):
            return 'velocity_analysis'
        elif any(keyword in name_lower for keyword in ['pipeline', 'coverage', 'summary', 'health']):
            return 'performance_analysis'
        else:
            return 'general'
    
    def get_agent_by_name(self, agent_name: str) -> Optional[Type]:
        """
        Get an agent class by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent class if found, None otherwise
        """
        if not self._discovery_cache_valid:
            self.discover_agents()
        
        return self.loaded_classes.get(agent_name)
    
    def get_agents_by_category(self, category: str) -> Dict[str, AgentMetadata]:
        """
        Get all agents in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            Dictionary of agent_name -> AgentMetadata for the category
        """
        if not self._discovery_cache_valid:
            self.discover_agents()
        
        return {
            name: metadata 
            for name, metadata in self.discovered_agents.items()
            if metadata.category == category
        }
    
    def create_agent_instance(self, agent_name: str, config: Dict[str, Any] = None) -> Optional[Any]:
        """
        Create an instance of an agent by name.
        
        Args:
            agent_name: Name of the agent to create
            config: Configuration for the agent
            
        Returns:
            Agent instance if successful, None otherwise
        """
        agent_class = self.get_agent_by_name(agent_name)
        if not agent_class:
            logger.error(f"❌ Agent not found: {agent_name}")
            return None
        
        try:
            return agent_class(config or {})
        except Exception as e:
            logger.error(f"❌ Failed to create agent instance {agent_name}: {e}")
            return None
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the discovery results.
        
        Returns:
            Summary dictionary with counts and categories
        """
        if not self._discovery_cache_valid:
            self.discover_agents()
        
        categories = {}
        for metadata in self.discovered_agents.values():
            category = metadata.category
            if category not in categories:
                categories[category] = []
            categories[category].append(metadata.name)
        
        return {
            'total_agents': len(self.discovered_agents),
            'categories': categories,
            'agent_names': list(self.discovered_agents.keys())
        }
    
    def generate_config_schemas_for_ui(self) -> Dict[str, Any]:
        """
        Generate configuration schemas for UI generation.
        
        Returns:
            Dictionary of agent_name -> UI config schema
        """
        if not self._discovery_cache_valid:
            self.discover_agents()
        
        ui_schemas = {}
        
        for agent_name, metadata in self.discovered_agents.items():
            try:
                agent_instance = self.create_agent_instance(agent_name)
                if agent_instance and hasattr(agent_instance, 'get_config_schema_for_ui'):
                    ui_schemas[agent_name] = agent_instance.get_config_schema_for_ui()
                else:
                    # Fallback to basic schema
                    ui_schemas[agent_name] = {
                        'agent_name': agent_name,
                        'category': metadata.category,
                        'description': metadata.description,
                        'parameters': {}
                    }
            except Exception as e:
                logger.error(f"❌ Failed to generate UI schema for {agent_name}: {e}")
        
        return ui_schemas

# Global discovery instance
_discovery_instance = None

def get_discovery_instance() -> RBAAgentDiscovery:
    """Get the global discovery instance"""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = RBAAgentDiscovery()
    return _discovery_instance

def discover_all_agents(force_refresh: bool = False) -> Dict[str, AgentMetadata]:
    """Convenience function to discover all agents"""
    return get_discovery_instance().discover_agents(force_refresh)

def get_agent_class(agent_name: str) -> Optional[Type]:
    """Convenience function to get an agent class"""
    return get_discovery_instance().get_agent_by_name(agent_name)

def create_agent(agent_name: str, config: Dict[str, Any] = None) -> Optional[Any]:
    """Convenience function to create an agent instance"""
    return get_discovery_instance().create_agent_instance(agent_name, config)
