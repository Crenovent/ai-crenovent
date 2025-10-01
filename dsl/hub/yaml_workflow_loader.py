#!/usr/bin/env python3
"""
YAML Workflow Loader - Single Source of Truth
============================================

Loads all workflows from YAML files with no hardcoded fallbacks.
Implements Build Plan requirement for YAML-first architecture.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WorkflowMetadata:
    """Metadata about a workflow from YAML"""
    name: str
    description: str
    version: str
    automation_type: str
    industry: str

class YAMLWorkflowLoader:
    """
    YAML-First Workflow Loader
    
    Single source of truth for all workflow definitions.
    No hardcoded workflows, no fallbacks - pure YAML.
    """
    
    def __init__(self, workflows_dir: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Set default workflows directory
        if workflows_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.workflows_dir = os.path.join(current_dir, "..", "..", "workflows")
        else:
            self.workflows_dir = workflows_dir
            
        self.yaml_file = os.path.join(self.workflows_dir, "saas_rba_workflows.yaml")
        
        # Cache for loaded workflows
        self._workflows_cache = {}
        self._metadata_cache = {}
        self._loaded = False
    
    def initialize(self) -> bool:
        """Initialize the workflow loader and validate YAML file"""
        try:
            if not os.path.exists(self.yaml_file):
                self.logger.error(f"❌ YAML workflow file not found: {self.yaml_file}")
                return False
            
            # Load and validate YAML structure
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                all_workflows = yaml.safe_load(f)
            
            if not all_workflows or not isinstance(all_workflows, dict):
                self.logger.error("❌ Invalid YAML structure - expected dictionary of workflows")
                return False
            
            self._workflows_cache = all_workflows
            self._loaded = True
            
            # Build metadata cache
            for workflow_name, workflow_def in all_workflows.items():
                if isinstance(workflow_def, dict):
                    self._metadata_cache[workflow_name] = WorkflowMetadata(
                        name=workflow_def.get('name', workflow_name),
                        description=workflow_def.get('description', ''),
                        version=workflow_def.get('version', '1.0'),
                        automation_type=workflow_def.get('automation_type', 'RBA'),
                        industry=workflow_def.get('industry', 'SaaS')
                    )
            
            self.logger.info(f"✅ YAML Workflow Loader initialized with {len(all_workflows)} workflows")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize YAML Workflow Loader: {e}")
            return False
    
    def get_available_workflows(self) -> List[str]:
        """Get list of all available workflow categories"""
        if not self._loaded:
            self.initialize()
        
        # Return workflow categories (remove _workflow suffix)
        categories = []
        for workflow_name in self._workflows_cache.keys():
            if workflow_name.endswith('_workflow'):
                category = workflow_name[:-9]  # Remove '_workflow'
                categories.append(category)
            else:
                categories.append(workflow_name)
        
        return categories
    
    def load_workflow(self, workflow_category: str) -> Optional[Dict[str, Any]]:
        """
        Load workflow definition from YAML
        
        Args:
            workflow_category: Category like 'pipeline_hygiene_stale_deals'
            
        Returns:
            Workflow definition dict or None if not found
        """
        if not self._loaded:
            if not self.initialize():
                return None
        
        # Try exact match first
        if workflow_category in self._workflows_cache:
            self.logger.info(f"✅ Loaded workflow: {workflow_category}")
            return self._workflows_cache[workflow_category]
        
        # Try with _workflow suffix
        workflow_name = f"{workflow_category}_workflow"
        if workflow_name in self._workflows_cache:
            self.logger.info(f"✅ Loaded workflow: {workflow_name}")
            return self._workflows_cache[workflow_name]
        
        self.logger.warning(f"⚠️ Workflow not found: {workflow_category}")
        return None
    
    def get_workflow_metadata(self, workflow_category: str) -> Optional[WorkflowMetadata]:
        """Get metadata for a workflow"""
        if not self._loaded:
            if not self.initialize():
                return None
        
        # Try exact match first
        if workflow_category in self._metadata_cache:
            return self._metadata_cache[workflow_category]
        
        # Try with _workflow suffix
        workflow_name = f"{workflow_category}_workflow"
        if workflow_name in self._metadata_cache:
            return self._metadata_cache[workflow_name]
        
        return None
    
    def reload_workflows(self) -> bool:
        """Reload workflows from YAML file"""
        self._workflows_cache.clear()
        self._metadata_cache.clear()
        self._loaded = False
        return self.initialize()
    
    def get_workflows_by_type(self, automation_type: str) -> List[str]:
        """Get workflows filtered by automation type (RBA, RBIA, AALA)"""
        workflows = []
        for workflow_name, metadata in self._metadata_cache.items():
            if metadata.automation_type == automation_type:
                # Return category name (remove _workflow suffix)
                if workflow_name.endswith('_workflow'):
                    category = workflow_name[:-9]
                    workflows.append(category)
                else:
                    workflows.append(workflow_name)
        
        return workflows
    
    def validate_workflow_structure(self, workflow_def: Dict[str, Any]) -> bool:
        """Validate that a workflow has the required structure"""
        required_fields = ['name', 'steps']
        
        for field in required_fields:
            if field not in workflow_def:
                self.logger.error(f"❌ Missing required field '{field}' in workflow")
                return False
        
        if not isinstance(workflow_def['steps'], list):
            self.logger.error("❌ Workflow 'steps' must be a list")
            return False
        
        if len(workflow_def['steps']) == 0:
            self.logger.error("❌ Workflow must have at least one step")
            return False
        
        return True

# Global instance for easy access
_workflow_loader = None

def get_workflow_loader() -> YAMLWorkflowLoader:
    """Get global workflow loader instance"""
    global _workflow_loader
    if _workflow_loader is None:
        _workflow_loader = YAMLWorkflowLoader()
        _workflow_loader.initialize()
    return _workflow_loader
