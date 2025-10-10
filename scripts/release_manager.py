"""
Release Management Tools - Task 6.5.80
======================================

Release process: version matrix (DSL↔compiler↔runtime), changelogs
- Predictable rollouts with weekly cadence
- Version compatibility tracking
- Automated changelog generation
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class ReleaseType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    HOTFIX = "hotfix"

class ComponentType(Enum):
    DSL_GRAMMAR = "dsl_grammar"
    COMPILER = "compiler"
    RUNTIME = "runtime"
    POLICY_PACKS = "policy_packs"

@dataclass
class Version:
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def bump(self, release_type: ReleaseType) -> 'Version':
        if release_type == ReleaseType.MAJOR:
            return Version(self.major + 1, 0, 0)
        elif release_type == ReleaseType.MINOR:
            return Version(self.major, self.minor + 1, 0)
        else:  # PATCH or HOTFIX
            return Version(self.major, self.minor, self.patch + 1)

@dataclass
class ComponentVersion:
    component: ComponentType
    version: Version
    compatible_versions: Dict[ComponentType, List[Version]]
    release_date: datetime
    status: str  # "current", "supported", "eol"

@dataclass
class ReleaseInfo:
    version: Version
    release_type: ReleaseType
    release_date: datetime
    changelog: Dict[str, List[str]]  # "added", "changed", "fixed", etc.
    components: List[ComponentVersion]

class ReleaseManager:
    """Manages RBIA release process and version compatibility"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_versions = self._initialize_current_versions()
        self.compatibility_matrix = self._initialize_compatibility_matrix()
    
    def _initialize_current_versions(self) -> Dict[ComponentType, Version]:
        """Initialize current component versions"""
        return {
            ComponentType.DSL_GRAMMAR: Version(2, 0, 0),
            ComponentType.COMPILER: Version(2, 0, 0),
            ComponentType.RUNTIME: Version(2, 0, 0),
            ComponentType.POLICY_PACKS: Version(1, 5, 0)
        }
    
    def _initialize_compatibility_matrix(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize version compatibility matrix"""
        return {
            "2.0.0": {
                "dsl_grammar": ["2.0.0"],
                "compiler": ["2.0.0", "1.9.0"],
                "runtime": ["2.0.0", "1.9.0"],
                "policy_packs": ["1.5.0", "1.4.0"]
            },
            "1.9.0": {
                "dsl_grammar": ["1.9.0", "1.8.0"],
                "compiler": ["1.9.0"],
                "runtime": ["1.9.0", "1.8.0"],
                "policy_packs": ["1.4.0", "1.3.0"]
            }
        }
    
    def check_compatibility(self, component_versions: Dict[ComponentType, Version]) -> Tuple[bool, List[str]]:
        """Check if component versions are compatible"""
        issues = []
        
        dsl_version = str(component_versions.get(ComponentType.DSL_GRAMMAR, Version(0, 0, 0)))
        compiler_version = str(component_versions.get(ComponentType.COMPILER, Version(0, 0, 0)))
        runtime_version = str(component_versions.get(ComponentType.RUNTIME, Version(0, 0, 0)))
        
        # Check DSL-Compiler compatibility
        if dsl_version in self.compatibility_matrix:
            compatible_compilers = self.compatibility_matrix[dsl_version].get("compiler", [])
            if compiler_version not in compatible_compilers:
                issues.append(f"DSL {dsl_version} not compatible with Compiler {compiler_version}")
        
        # Check Compiler-Runtime compatibility
        if compiler_version in self.compatibility_matrix:
            compatible_runtimes = self.compatibility_matrix[compiler_version].get("runtime", [])
            if runtime_version not in compatible_runtimes:
                issues.append(f"Compiler {compiler_version} not compatible with Runtime {runtime_version}")
        
        return len(issues) == 0, issues
    
    def plan_release(self, component: ComponentType, release_type: ReleaseType) -> ReleaseInfo:
        """Plan a new release for a component"""
        current_version = self.current_versions[component]
        new_version = current_version.bump(release_type)
        
        # Calculate release date (next Friday for regular releases)
        today = datetime.now()
        days_ahead = 4 - today.weekday()  # Friday is 4
        if days_ahead <= 0:  # Target next Friday
            days_ahead += 7
        release_date = today + timedelta(days=days_ahead)
        
        # Generate changelog template
        changelog = {
            "added": [],
            "changed": [],
            "deprecated": [],
            "removed": [],
            "fixed": [],
            "security": []
        }
        
        return ReleaseInfo(
            version=new_version,
            release_type=release_type,
            release_date=release_date,
            changelog=changelog,
            components=[]
        )
    
    def generate_changelog(self, release_info: ReleaseInfo) -> str:
        """Generate formatted changelog"""
        changelog = f"## Version {release_info.version} ({release_info.release_date.strftime('%Y-%m-%d')})\n\n"
        
        for section, items in release_info.changelog.items():
            if items:
                changelog += f"### {section.title()}\n"
                for item in items:
                    changelog += f"- {item}\n"
                changelog += "\n"
        
        return changelog
    
    def get_migration_path(self, from_version: Version, to_version: Version, component: ComponentType) -> List[str]:
        """Get migration steps between versions"""
        steps = []
        
        if component == ComponentType.DSL_GRAMMAR:
            if from_version.major < to_version.major:
                steps.extend([
                    "Run DSL compatibility checker",
                    "Update grammar syntax to new version",
                    "Add required governance metadata",
                    "Validate with new compiler",
                    "Deploy with gradual rollout"
                ])
            elif from_version.minor < to_version.minor:
                steps.extend([
                    "Review new features documentation",
                    "Update workflows to use new features",
                    "Test with current compiler",
                    "Deploy incrementally"
                ])
        
        elif component == ComponentType.COMPILER:
            steps.extend([
                "Backup current compiler configuration",
                "Install new compiler version",
                "Run regression test suite",
                "Validate compilation outputs",
                "Monitor for errors post-deployment"
            ])
        
        return steps
    
    def get_support_status(self, version: Version) -> str:
        """Get support status for a version"""
        current = self.current_versions[ComponentType.COMPILER]  # Use compiler as reference
        
        if version.major == current.major and version.minor == current.minor:
            return "current"
        elif version.major == current.major and version.minor == current.minor - 1:
            return "supported"
        elif version.major == current.major and version.minor == current.minor - 2:
            return "security_only"
        else:
            return "eol"
    
    def export_version_matrix(self) -> Dict[str, Any]:
        """Export current version compatibility matrix"""
        return {
            "current_versions": {
                component.value: str(version) 
                for component, version in self.current_versions.items()
            },
            "compatibility_matrix": self.compatibility_matrix,
            "support_policy": {
                "current": "Full support, active development",
                "supported": "Security fixes and critical bugs",
                "security_only": "Security fixes only",
                "eol": "No support - migration required"
            },
            "release_schedule": "Weekly on Fridays",
            "generated_at": datetime.now().isoformat()
        }
