"""
Policy Snippet Library - Task 6.5.76
====================================

Reusable governance (thresholds, kill-switch, Assisted)
- Copy/paste safe policy snippets
- Common governance patterns
- Compliance-ready templates
"""

from typing import Dict, List, Any, Optional
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PolicySnippetLibrary:
    """Library of reusable policy snippets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.snippets = self._load_snippets()
    
    def _load_snippets(self) -> Dict[str, Any]:
        """Load policy snippets from YAML file"""
        try:
            snippet_file = Path(__file__).parent.parent / "rules" / "policy_snippets.yaml"
            with open(snippet_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load policy snippets: {e}")
            return {}
    
    def get_snippet(self, snippet_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific policy snippet"""
        return self.snippets.get(snippet_name)
    
    def get_snippets_by_category(self, category: str) -> Dict[str, Any]:
        """Get all snippets in a category"""
        result = {}
        for name, snippet in self.snippets.items():
            if name.startswith(category):
                result[name] = snippet
        return result
    
    def list_available_snippets(self) -> List[str]:
        """List all available snippet names"""
        return list(self.snippets.keys())
    
    def apply_snippet(self, base_config: Dict[str, Any], snippet_name: str) -> Dict[str, Any]:
        """Apply a snippet to base configuration"""
        snippet = self.get_snippet(snippet_name)
        if not snippet:
            raise ValueError(f"Snippet '{snippet_name}' not found")
        
        # Deep merge snippet into base config
        result = base_config.copy()
        self._deep_merge(result, snippet)
        return result
    
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        """Deep merge overlay into base dictionary"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get_compliance_snippet(self, framework: str) -> Optional[Dict[str, Any]]:
        """Get compliance-specific snippet"""
        mapping = {
            "SOX": "sox_compliance",
            "GDPR": "gdpr_compliance",
            "CCPA": "gdpr_compliance",  # Similar requirements
            "DPDP": "gdpr_compliance"   # Similar requirements
        }
        
        snippet_name = mapping.get(framework)
        return self.get_snippet(snippet_name) if snippet_name else None
    
    def get_risk_snippet(self, risk_level: str) -> Optional[Dict[str, Any]]:
        """Get risk-appropriate snippet"""
        mapping = {
            "high": "confidence_high_risk",
            "medium": "confidence_medium_risk", 
            "low": "confidence_low_risk"
        }
        
        snippet_name = mapping.get(risk_level)
        return self.get_snippet(snippet_name) if snippet_name else None
    
    def get_sla_snippet(self, priority: str) -> Optional[Dict[str, Any]]:
        """Get SLA-appropriate snippet"""
        mapping = {
            "critical": "sla_critical",
            "important": "sla_important",
            "standard": "sla_standard"
        }
        
        snippet_name = mapping.get(priority)
        return self.get_snippet(snippet_name) if snippet_name else None
