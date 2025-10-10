"""
Task 6.3.81: Accessibility checks on error/alert UIs
WCAG AA compliance for error and alert interfaces
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class AccessibilityResult:
    """Accessibility check result"""
    component_id: str
    wcag_level: str
    violations: List[Dict[str, Any]]
    score: float
    compliant: bool

class AccessibilityChecker:
    """
    Accessibility checker for error/alert UIs
    Task 6.3.81: WCAG AA compliance for inclusivity
    """
    
    def __init__(self):
        self.wcag_rules = {
            "color_contrast": {"min_ratio": 4.5, "weight": 0.3},
            "keyboard_navigation": {"required": True, "weight": 0.2}, 
            "screen_reader": {"required": True, "weight": 0.2},
            "focus_indicators": {"required": True, "weight": 0.15},
            "alt_text": {"required": True, "weight": 0.15}
        }
    
    def check_component(self, component_config: Dict[str, Any]) -> AccessibilityResult:
        """Check component for WCAG AA compliance"""
        violations = []
        total_score = 0.0
        
        for rule, criteria in self.wcag_rules.items():
            if rule not in component_config:
                violations.append({
                    "rule": rule,
                    "severity": "error",
                    "message": f"Missing {rule} configuration"
                })
            else:
                rule_score = self._check_rule(rule, component_config[rule], criteria)
                total_score += rule_score * criteria["weight"]
        
        return AccessibilityResult(
            component_id=component_config.get("id", "unknown"),
            wcag_level="AA",
            violations=violations,
            score=total_score,
            compliant=len(violations) == 0 and total_score >= 0.8
        )
    
    def _check_rule(self, rule: str, value: Any, criteria: Dict[str, Any]) -> float:
        """Check individual WCAG rule"""
        if rule == "color_contrast" and isinstance(value, (int, float)):
            return 1.0 if value >= criteria["min_ratio"] else 0.0
        elif criteria.get("required") and value:
            return 1.0
        return 0.0

# Global accessibility checker
accessibility_checker = AccessibilityChecker()
