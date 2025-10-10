"""
Task 7.3-T16: Implement Diff & Changelog generation
Clear change history with AST diff and auto-generated Markdown notes
"""

import json
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field

import yaml
from deepdiff import DeepDiff
from packaging import version as pkg_version


class ChangeType(Enum):
    """Types of changes detected"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MOVED = "moved"
    TYPE_CHANGED = "type_changed"


class ChangeImpact(Enum):
    """Impact level of changes"""
    BREAKING = "breaking"
    FEATURE = "feature"
    BUGFIX = "bugfix"
    DOCUMENTATION = "documentation"
    INTERNAL = "internal"


class ChangeSeverity(Enum):
    """Severity of changes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class Change:
    """Individual change detected between versions"""
    change_id: str
    change_type: ChangeType
    path: str
    old_value: Any = None
    new_value: Any = None
    description: str = ""
    impact: ChangeImpact = ChangeImpact.INTERNAL
    severity: ChangeSeverity = ChangeSeverity.MEDIUM
    breaking: bool = False
    migration_required: bool = False
    migration_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert change to dictionary"""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type.value,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "description": self.description,
            "impact": self.impact.value,
            "severity": self.severity.value,
            "breaking": self.breaking,
            "migration_required": self.migration_required,
            "migration_notes": self.migration_notes
        }


@dataclass
class VersionDiff:
    """Complete diff between two workflow versions"""
    from_version: str
    to_version: str
    from_version_id: str
    to_version_id: str
    workflow_name: str
    changes: List[Change] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    breaking_changes: List[Change] = field(default_factory=list)
    migration_required: bool = False
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Calculate summary and breaking changes"""
        self.summary = {
            "total_changes": len(self.changes),
            "added": len([c for c in self.changes if c.change_type == ChangeType.ADDED]),
            "removed": len([c for c in self.changes if c.change_type == ChangeType.REMOVED]),
            "modified": len([c for c in self.changes if c.change_type == ChangeType.MODIFIED]),
            "breaking_changes": len([c for c in self.changes if c.breaking]),
            "migration_required": len([c for c in self.changes if c.migration_required])
        }
        
        self.breaking_changes = [c for c in self.changes if c.breaking]
        self.migration_required = any(c.migration_required for c in self.changes)
    
    def get_changes_by_impact(self, impact: ChangeImpact) -> List[Change]:
        """Get changes by impact level"""
        return [c for c in self.changes if c.impact == impact]
    
    def get_changes_by_severity(self, severity: ChangeSeverity) -> List[Change]:
        """Get changes by severity level"""
        return [c for c in self.changes if c.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert diff to dictionary"""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "from_version_id": self.from_version_id,
            "to_version_id": self.to_version_id,
            "workflow_name": self.workflow_name,
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.summary,
            "breaking_changes": [c.to_dict() for c in self.breaking_changes],
            "migration_required": self.migration_required,
            "generated_at": self.generated_at.isoformat()
        }


@dataclass
class Changelog:
    """Generated changelog for a version"""
    version: str
    version_id: str
    workflow_name: str
    release_date: datetime
    changes: List[Change] = field(default_factory=list)
    breaking_changes: List[Change] = field(default_factory=list)
    migration_notes: List[str] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Generate Markdown changelog"""
        lines = []
        
        # Header
        lines.append(f"# Changelog - {self.workflow_name} v{self.version}")
        lines.append("")
        lines.append(f"**Release Date:** {self.release_date.strftime('%Y-%m-%d')}")
        lines.append("")
        
        # Summary
        if self.changes:
            summary = {
                "added": len([c for c in self.changes if c.change_type == ChangeType.ADDED]),
                "modified": len([c for c in self.changes if c.change_type == ChangeType.MODIFIED]),
                "removed": len([c for c in self.changes if c.change_type == ChangeType.REMOVED]),
                "breaking": len(self.breaking_changes)
            }
            
            lines.append("## Summary")
            lines.append("")
            lines.append(f"- **{summary['added']}** additions")
            lines.append(f"- **{summary['modified']}** modifications")
            lines.append(f"- **{summary['removed']}** removals")
            if summary['breaking'] > 0:
                lines.append(f"- **{summary['breaking']}** breaking changes ⚠️")
            lines.append("")
        
        # Breaking Changes
        if self.breaking_changes:
            lines.append("## ⚠️ Breaking Changes")
            lines.append("")
            for change in self.breaking_changes:
                lines.append(f"- **{change.path}**: {change.description}")
                if change.migration_notes:
                    lines.append(f"  - Migration: {change.migration_notes}")
            lines.append("")
        
        # Changes by category
        categories = {
            ChangeImpact.FEATURE: ("✨ New Features", []),
            ChangeImpact.BUGFIX: ("🐛 Bug Fixes", []),
            ChangeImpact.DOCUMENTATION: ("📚 Documentation", []),
            ChangeImpact.INTERNAL: ("🔧 Internal Changes", [])
        }
        
        for change in self.changes:
            if change.impact in categories:
                categories[change.impact][1].append(change)
        
        for impact, (title, changes) in categories.items():
            if changes:
                lines.append(f"## {title}")
                lines.append("")
                for change in changes:
                    lines.append(f"- **{change.path}**: {change.description}")
                lines.append("")
        
        # Migration Notes
        if self.migration_notes:
            lines.append("## 📋 Migration Guide")
            lines.append("")
            for note in self.migration_notes:
                lines.append(f"- {note}")
            lines.append("")
        
        # Contributors
        if self.contributors:
            lines.append("## 👥 Contributors")
            lines.append("")
            for contributor in self.contributors:
                lines.append(f"- {contributor}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert changelog to dictionary"""
        return {
            "version": self.version,
            "version_id": self.version_id,
            "workflow_name": self.workflow_name,
            "release_date": self.release_date.isoformat(),
            "changes": [c.to_dict() for c in self.changes],
            "breaking_changes": [c.to_dict() for c in self.breaking_changes],
            "migration_notes": self.migration_notes,
            "contributors": self.contributors
        }


class WorkflowDiffAnalyzer:
    """Analyzes differences between workflow versions"""
    
    def __init__(self):
        self.breaking_change_patterns = self._load_breaking_change_patterns()
        self.change_classifiers = self._load_change_classifiers()
    
    def compare_versions(
        self,
        from_content: Union[str, Dict[str, Any]],
        to_content: Union[str, Dict[str, Any]],
        from_version: str,
        to_version: str,
        from_version_id: str,
        to_version_id: str,
        workflow_name: str
    ) -> VersionDiff:
        """Compare two workflow versions and generate diff"""
        
        # Parse content if strings
        from_data = self._parse_content(from_content)
        to_data = self._parse_content(to_content)
        
        # Perform deep diff
        diff = DeepDiff(from_data, to_data, ignore_order=True, report_type='list')
        
        # Convert to Change objects
        changes = self._convert_diff_to_changes(diff, from_data, to_data)
        
        # Classify changes
        for change in changes:
            self._classify_change(change, from_data, to_data)
        
        return VersionDiff(
            from_version=from_version,
            to_version=to_version,
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            workflow_name=workflow_name,
            changes=changes
        )
    
    def _parse_content(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse content to dictionary"""
        if isinstance(content, dict):
            return content
        elif isinstance(content, str):
            try:
                # Try YAML first
                return yaml.safe_load(content)
            except yaml.YAMLError:
                try:
                    # Try JSON
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Return as plain text
                    return {"content": content}
        else:
            return {"content": str(content)}
    
    def _convert_diff_to_changes(
        self,
        diff: Dict[str, Any],
        from_data: Dict[str, Any],
        to_data: Dict[str, Any]
    ) -> List[Change]:
        """Convert DeepDiff result to Change objects"""
        changes = []
        
        # Handle dictionary item additions
        if 'dictionary_item_added' in diff:
            for item in diff['dictionary_item_added']:
                path = self._clean_path(item['path'])
                change = Change(
                    change_id=str(uuid.uuid4()),
                    change_type=ChangeType.ADDED,
                    path=path,
                    new_value=item['value'],
                    description=f"Added {path}"
                )
                changes.append(change)
        
        # Handle dictionary item removals
        if 'dictionary_item_removed' in diff:
            for item in diff['dictionary_item_removed']:
                path = self._clean_path(item['path'])
                change = Change(
                    change_id=str(uuid.uuid4()),
                    change_type=ChangeType.REMOVED,
                    path=path,
                    old_value=item['value'],
                    description=f"Removed {path}"
                )
                changes.append(change)
        
        # Handle value changes
        if 'values_changed' in diff:
            for item in diff['values_changed']:
                path = self._clean_path(item['path'])
                change = Change(
                    change_id=str(uuid.uuid4()),
                    change_type=ChangeType.MODIFIED,
                    path=path,
                    old_value=item['old_value'],
                    new_value=item['new_value'],
                    description=f"Modified {path}"
                )
                changes.append(change)
        
        # Handle type changes
        if 'type_changes' in diff:
            for item in diff['type_changes']:
                path = self._clean_path(item['path'])
                change = Change(
                    change_id=str(uuid.uuid4()),
                    change_type=ChangeType.TYPE_CHANGED,
                    path=path,
                    old_value=item['old_value'],
                    new_value=item['new_value'],
                    description=f"Changed type of {path} from {item['old_type']} to {item['new_type']}"
                )
                changes.append(change)
        
        # Handle iterable item additions
        if 'iterable_item_added' in diff:
            for item in diff['iterable_item_added']:
                path = self._clean_path(item['path'])
                change = Change(
                    change_id=str(uuid.uuid4()),
                    change_type=ChangeType.ADDED,
                    path=path,
                    new_value=item['value'],
                    description=f"Added item to {path}"
                )
                changes.append(change)
        
        # Handle iterable item removals
        if 'iterable_item_removed' in diff:
            for item in diff['iterable_item_removed']:
                path = self._clean_path(item['path'])
                change = Change(
                    change_id=str(uuid.uuid4()),
                    change_type=ChangeType.REMOVED,
                    path=path,
                    old_value=item['value'],
                    description=f"Removed item from {path}"
                )
                changes.append(change)
        
        return changes
    
    def _clean_path(self, path: str) -> str:
        """Clean and normalize path string"""
        # Remove root prefix and brackets
        cleaned = re.sub(r"^root\['?([^']*)'?\]", r"\1", path)
        cleaned = re.sub(r"\['?([^']*)'?\]", r".\1", cleaned)
        cleaned = re.sub(r"^\.", "", cleaned)
        return cleaned or "root"
    
    def _classify_change(
        self,
        change: Change,
        from_data: Dict[str, Any],
        to_data: Dict[str, Any]
    ) -> None:
        """Classify change impact and severity"""
        
        # Check for breaking changes
        if self._is_breaking_change(change):
            change.breaking = True
            change.impact = ChangeImpact.BREAKING
            change.severity = ChangeSeverity.HIGH
            change.migration_required = True
            change.migration_notes = self._get_migration_notes(change)
        
        # Classify by path patterns
        path_lower = change.path.lower()
        
        # Workflow structure changes
        if any(keyword in path_lower for keyword in ['workflow', 'steps', 'parameters']):
            if change.change_type == ChangeType.REMOVED:
                change.impact = ChangeImpact.BREAKING
                change.breaking = True
                change.severity = ChangeSeverity.HIGH
            elif change.change_type == ChangeType.ADDED:
                change.impact = ChangeImpact.FEATURE
                change.severity = ChangeSeverity.MEDIUM
            else:
                change.impact = ChangeImpact.FEATURE
                change.severity = ChangeSeverity.MEDIUM
        
        # Documentation changes
        elif any(keyword in path_lower for keyword in ['description', 'documentation', 'readme', 'comment']):
            change.impact = ChangeImpact.DOCUMENTATION
            change.severity = ChangeSeverity.LOW
        
        # Version changes
        elif 'version' in path_lower:
            change.impact = ChangeImpact.INTERNAL
            change.severity = ChangeSeverity.LOW
        
        # Configuration changes
        elif any(keyword in path_lower for keyword in ['config', 'settings', 'options']):
            change.impact = ChangeImpact.FEATURE
            change.severity = ChangeSeverity.MEDIUM
        
        # Default classification
        else:
            change.impact = ChangeImpact.INTERNAL
            change.severity = ChangeSeverity.MEDIUM
        
        # Generate better description
        change.description = self._generate_change_description(change)
    
    def _is_breaking_change(self, change: Change) -> bool:
        """Determine if a change is breaking"""
        path_lower = change.path.lower()
        
        # Breaking change patterns
        breaking_patterns = [
            # Parameter removals or type changes
            (r'parameters\..*', ChangeType.REMOVED),
            (r'parameters\..*\.type', ChangeType.MODIFIED),
            (r'parameters\..*\.required', ChangeType.MODIFIED),
            
            # Step removals or major modifications
            (r'steps\..*', ChangeType.REMOVED),
            (r'steps\..*\.type', ChangeType.MODIFIED),
            
            # Workflow structure changes
            (r'workflow\.name', ChangeType.MODIFIED),
            (r'workflow\.type', ChangeType.MODIFIED),
            
            # Output format changes
            (r'outputs\..*\.type', ChangeType.MODIFIED),
            (r'outputs\..*', ChangeType.REMOVED),
        ]
        
        for pattern, change_type in breaking_patterns:
            if re.match(pattern, path_lower) and change.change_type == change_type:
                return True
        
        # Type changes are often breaking
        if change.change_type == ChangeType.TYPE_CHANGED:
            return True
        
        # Required parameter additions
        if (change.change_type == ChangeType.ADDED and 
            'parameters' in path_lower and 
            isinstance(change.new_value, dict) and 
            change.new_value.get('required', False)):
            return True
        
        return False
    
    def _get_migration_notes(self, change: Change) -> str:
        """Generate migration notes for breaking changes"""
        path_lower = change.path.lower()
        
        if change.change_type == ChangeType.REMOVED:
            if 'parameters' in path_lower:
                return f"Remove usage of parameter '{change.path}' from workflow calls"
            elif 'steps' in path_lower:
                return f"Update workflow logic to handle removal of step '{change.path}'"
            else:
                return f"Update code that depends on '{change.path}'"
        
        elif change.change_type == ChangeType.MODIFIED:
            if 'type' in path_lower:
                return f"Update '{change.path}' from {change.old_value} to {change.new_value}"
            else:
                return f"Review and update usage of '{change.path}'"
        
        elif change.change_type == ChangeType.ADDED and 'required' in str(change.new_value):
            return f"Add required parameter '{change.path}' to all workflow calls"
        
        return f"Review changes to '{change.path}' and update accordingly"
    
    def _generate_change_description(self, change: Change) -> str:
        """Generate human-readable change description"""
        path_parts = change.path.split('.')
        
        if change.change_type == ChangeType.ADDED:
            if 'parameters' in change.path:
                return f"Added parameter '{path_parts[-1]}'"
            elif 'steps' in change.path:
                return f"Added workflow step '{path_parts[-1]}'"
            else:
                return f"Added '{path_parts[-1]}'"
        
        elif change.change_type == ChangeType.REMOVED:
            if 'parameters' in change.path:
                return f"Removed parameter '{path_parts[-1]}'"
            elif 'steps' in change.path:
                return f"Removed workflow step '{path_parts[-1]}'"
            else:
                return f"Removed '{path_parts[-1]}'"
        
        elif change.change_type == ChangeType.MODIFIED:
            if isinstance(change.old_value, str) and isinstance(change.new_value, str):
                return f"Changed '{path_parts[-1]}' from '{change.old_value}' to '{change.new_value}'"
            else:
                return f"Modified '{path_parts[-1]}'"
        
        elif change.change_type == ChangeType.TYPE_CHANGED:
            return f"Changed type of '{path_parts[-1]}' from {type(change.old_value).__name__} to {type(change.new_value).__name__}"
        
        return f"Updated '{change.path}'"
    
    def _load_breaking_change_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for detecting breaking changes"""
        return [
            {"pattern": r"parameters\..*", "type": "removed", "breaking": True},
            {"pattern": r"steps\..*", "type": "removed", "breaking": True},
            {"pattern": r"outputs\..*", "type": "removed", "breaking": True},
            {"pattern": r".*\.type", "type": "modified", "breaking": True},
            {"pattern": r".*\.required", "type": "modified", "breaking": True},
        ]
    
    def _load_change_classifiers(self) -> Dict[str, Any]:
        """Load change classification rules"""
        return {
            "documentation": ["description", "documentation", "readme", "comment"],
            "configuration": ["config", "settings", "options", "metadata"],
            "workflow": ["workflow", "steps", "parameters", "outputs"],
            "internal": ["version", "id", "created_at", "updated_at"]
        }


class ChangelogGenerator:
    """Generates changelogs from version diffs"""
    
    def __init__(self):
        self.diff_analyzer = WorkflowDiffAnalyzer()
    
    def generate_changelog(
        self,
        version_diff: VersionDiff,
        contributors: Optional[List[str]] = None,
        custom_notes: Optional[List[str]] = None
    ) -> Changelog:
        """Generate changelog from version diff"""
        
        changelog = Changelog(
            version=version_diff.to_version,
            version_id=version_diff.to_version_id,
            workflow_name=version_diff.workflow_name,
            release_date=datetime.now(timezone.utc),
            changes=version_diff.changes,
            breaking_changes=version_diff.breaking_changes,
            contributors=contributors or []
        )
        
        # Generate migration notes
        migration_notes = []
        for change in version_diff.breaking_changes:
            if change.migration_notes:
                migration_notes.append(change.migration_notes)
        
        if custom_notes:
            migration_notes.extend(custom_notes)
        
        changelog.migration_notes = migration_notes
        
        return changelog
    
    def generate_changelog_from_versions(
        self,
        from_content: Union[str, Dict[str, Any]],
        to_content: Union[str, Dict[str, Any]],
        from_version: str,
        to_version: str,
        from_version_id: str,
        to_version_id: str,
        workflow_name: str,
        contributors: Optional[List[str]] = None,
        custom_notes: Optional[List[str]] = None
    ) -> Changelog:
        """Generate changelog by comparing two versions"""
        
        # Generate diff
        version_diff = self.diff_analyzer.compare_versions(
            from_content=from_content,
            to_content=to_content,
            from_version=from_version,
            to_version=to_version,
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            workflow_name=workflow_name
        )
        
        # Generate changelog
        return self.generate_changelog(version_diff, contributors, custom_notes)


# Example usage
def example_diff_and_changelog():
    """Example of diff and changelog generation"""
    
    # Example workflow versions
    v1_content = """
    workflow:
      name: "pipeline_hygiene"
      version: "1.0.0"
      description: "Basic pipeline hygiene checks"
      parameters:
        - name: "threshold_days"
          type: "integer"
          required: true
          default: 30
      steps:
        - name: "check_stale_opportunities"
          type: "query"
          params:
            table: "opportunities"
            filters:
              - "updated_at < NOW() - INTERVAL '{threshold_days} days'"
    """
    
    v2_content = """
    workflow:
      name: "pipeline_hygiene"
      version: "2.0.0"
      description: "Enhanced pipeline hygiene with notifications"
      parameters:
        - name: "threshold_days"
          type: "integer"
          required: true
          default: 30
        - name: "notification_channel"
          type: "string"
          required: true
      steps:
        - name: "check_stale_opportunities"
          type: "query"
          params:
            table: "opportunities"
            filters:
              - "updated_at < NOW() - INTERVAL '{threshold_days} days'"
        - name: "send_notification"
          type: "notify"
          params:
            channel: "{notification_channel}"
            message: "Found stale opportunities"
    """
    
    # Create analyzer and generate diff
    analyzer = WorkflowDiffAnalyzer()
    diff = analyzer.compare_versions(
        from_content=v1_content,
        to_content=v2_content,
        from_version="1.0.0",
        to_version="2.0.0",
        from_version_id="version-1",
        to_version_id="version-2",
        workflow_name="Pipeline Hygiene"
    )
    
    print("=== VERSION DIFF ===")
    print(f"Changes: {len(diff.changes)}")
    print(f"Breaking changes: {len(diff.breaking_changes)}")
    print(f"Migration required: {diff.migration_required}")
    
    for change in diff.changes:
        print(f"- {change.change_type.value}: {change.description}")
        if change.breaking:
            print(f"  ⚠️ BREAKING: {change.migration_notes}")
    
    # Generate changelog
    generator = ChangelogGenerator()
    changelog = generator.generate_changelog(
        diff,
        contributors=["RevOps Team", "Platform Team"],
        custom_notes=["Test thoroughly before deploying to production"]
    )
    
    print("\n=== CHANGELOG MARKDOWN ===")
    print(changelog.to_markdown())
    
    return diff, changelog


if __name__ == "__main__":
    example_diff_and_changelog()
