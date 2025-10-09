"""
Template Versioning System
Task 4.2.2: Template versioning with backward compatibility and migration support
"""

import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sqlite3
from enum import Enum
import copy

logger = logging.getLogger(__name__)

class VersionStatus(Enum):
    """Version status types"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    BETA = "beta"

class VersionCompatibility(Enum):
    """Backward compatibility levels"""
    BREAKING = "breaking"  # Major version change
    COMPATIBLE = "compatible"  # Minor version change
    PATCH = "patch"  # Patch version change

@dataclass
class TemplateVersion:
    """Template version information"""
    version_id: str
    template_id: str
    version_number: str  # Semantic versioning: major.minor.patch
    status: VersionStatus
    compatibility: VersionCompatibility
    template_config: Dict[str, Any]
    changelog: List[str]
    breaking_changes: List[str]
    migration_notes: str
    created_by: int
    created_at: datetime
    deprecated_at: Optional[datetime] = None
    checksum: str = ""
    parent_version_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VersionComparisonResult:
    """Result of version comparison"""
    old_version: str
    new_version: str
    compatibility: VersionCompatibility
    added_fields: List[str]
    removed_fields: List[str]
    modified_fields: List[str]
    breaking_changes: List[str]
    migration_required: bool
    auto_migration_possible: bool

class TemplateVersioningSystem:
    """
    Comprehensive versioning system for templates
    
    Features:
    - Semantic versioning (major.minor.patch)
    - Backward compatibility tracking
    - Version comparison and diff
    - Migration path generation
    - Version rollback support
    - Checksum verification
    """
    
    def __init__(self, db_path: str = "template_versioning.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.version_cache: Dict[str, List[TemplateVersion]] = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize versioning database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create template versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS template_versions (
                    version_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    version_number TEXT NOT NULL,
                    status TEXT NOT NULL,
                    compatibility TEXT NOT NULL,
                    template_config TEXT NOT NULL,
                    changelog TEXT NOT NULL,
                    breaking_changes TEXT NOT NULL,
                    migration_notes TEXT,
                    created_by INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deprecated_at TIMESTAMP,
                    checksum TEXT NOT NULL,
                    parent_version_id TEXT,
                    metadata TEXT,
                    UNIQUE(template_id, version_number)
                )
            ''')
            
            # Create version dependencies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_dependencies (
                    dependency_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    depends_on_template_id TEXT NOT NULL,
                    depends_on_version TEXT NOT NULL,
                    dependency_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (version_id) REFERENCES template_versions(version_id)
                )
            ''')
            
            # Create version migrations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_migrations (
                    migration_id TEXT PRIMARY KEY,
                    from_version_id TEXT NOT NULL,
                    to_version_id TEXT NOT NULL,
                    migration_script TEXT NOT NULL,
                    auto_migration BOOLEAN DEFAULT FALSE,
                    tested BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (from_version_id) REFERENCES template_versions(version_id),
                    FOREIGN KEY (to_version_id) REFERENCES template_versions(version_id)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_template ON template_versions(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_status ON template_versions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_versions_created_at ON template_versions(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dependencies_version ON version_dependencies(version_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_migrations_from ON version_migrations(from_version_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_migrations_to ON version_migrations(to_version_id)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Template versioning database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize versioning database: {e}")
            raise
    
    def _calculate_checksum(self, template_config: Dict[str, Any]) -> str:
        """Calculate checksum for template configuration"""
        config_str = json.dumps(template_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _parse_version(self, version_number: str) -> Tuple[int, int, int]:
        """Parse semantic version number"""
        try:
            parts = version_number.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except Exception as e:
            raise ValueError(f"Invalid version number format: {version_number}")
    
    def _increment_version(
        self,
        current_version: str,
        compatibility: VersionCompatibility
    ) -> str:
        """Increment version number based on compatibility"""
        major, minor, patch = self._parse_version(current_version)
        
        if compatibility == VersionCompatibility.BREAKING:
            return f"{major + 1}.0.0"
        elif compatibility == VersionCompatibility.COMPATIBLE:
            return f"{major}.{minor + 1}.0"
        else:  # PATCH
            return f"{major}.{minor}.{patch + 1}"
    
    async def create_version(
        self,
        template_id: str,
        template_config: Dict[str, Any],
        changelog: List[str],
        breaking_changes: List[str],
        migration_notes: str,
        created_by: int,
        compatibility: VersionCompatibility = VersionCompatibility.COMPATIBLE,
        parent_version_id: Optional[str] = None
    ) -> TemplateVersion:
        """Create a new template version"""
        
        try:
            version_id = str(uuid.uuid4())
            
            # Get latest version to determine new version number
            latest_version = await self.get_latest_version(template_id)
            
            if latest_version:
                version_number = self._increment_version(
                    latest_version.version_number,
                    compatibility
                )
            else:
                version_number = "1.0.0"
            
            # Calculate checksum
            checksum = self._calculate_checksum(template_config)
            
            # Create version object
            version = TemplateVersion(
                version_id=version_id,
                template_id=template_id,
                version_number=version_number,
                status=VersionStatus.DRAFT,
                compatibility=compatibility,
                template_config=template_config,
                changelog=changelog,
                breaking_changes=breaking_changes,
                migration_notes=migration_notes,
                created_by=created_by,
                created_at=datetime.utcnow(),
                checksum=checksum,
                parent_version_id=parent_version_id or (latest_version.version_id if latest_version else None)
            )
            
            # Store in database
            await self._store_version(version)
            
            # Clear cache
            if template_id in self.version_cache:
                del self.version_cache[template_id]
            
            self.logger.info(f"Created version {version_number} for template {template_id}")
            
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to create version: {e}")
            raise
    
    async def _store_version(self, version: TemplateVersion):
        """Store version in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO template_versions (
                    version_id, template_id, version_number, status, compatibility,
                    template_config, changelog, breaking_changes, migration_notes,
                    created_by, deprecated_at, checksum, parent_version_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version.version_id,
                version.template_id,
                version.version_number,
                version.status.value,
                version.compatibility.value,
                json.dumps(version.template_config),
                json.dumps(version.changelog),
                json.dumps(version.breaking_changes),
                version.migration_notes,
                version.created_by,
                version.deprecated_at.isoformat() if version.deprecated_at else None,
                version.checksum,
                version.parent_version_id,
                json.dumps(version.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store version: {e}")
            raise
    
    async def get_version(
        self,
        template_id: str,
        version_number: str
    ) -> Optional[TemplateVersion]:
        """Get specific version of a template"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM template_versions
                WHERE template_id = ? AND version_number = ?
            ''', (template_id, version_number))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_version(row)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get version: {e}")
            return None
    
    async def get_latest_version(
        self,
        template_id: str,
        status: Optional[VersionStatus] = None
    ) -> Optional[TemplateVersion]:
        """Get latest version of a template"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if status:
                cursor.execute('''
                    SELECT * FROM template_versions
                    WHERE template_id = ? AND status = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                ''', (template_id, status.value))
            else:
                cursor.execute('''
                    SELECT * FROM template_versions
                    WHERE template_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                ''', (template_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_version(row)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest version: {e}")
            return None
    
    async def get_all_versions(
        self,
        template_id: str,
        status: Optional[VersionStatus] = None
    ) -> List[TemplateVersion]:
        """Get all versions of a template"""
        try:
            # Check cache
            cache_key = f"{template_id}_{status.value if status else 'all'}"
            if cache_key in self.version_cache:
                return self.version_cache[cache_key]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if status:
                cursor.execute('''
                    SELECT * FROM template_versions
                    WHERE template_id = ? AND status = ?
                    ORDER BY created_at DESC
                ''', (template_id, status.value))
            else:
                cursor.execute('''
                    SELECT * FROM template_versions
                    WHERE template_id = ?
                    ORDER BY created_at DESC
                ''', (template_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            versions = [self._row_to_version(row) for row in rows]
            
            # Cache results
            self.version_cache[cache_key] = versions
            
            return versions
            
        except Exception as e:
            self.logger.error(f"Failed to get all versions: {e}")
            return []
    
    def _row_to_version(self, row) -> TemplateVersion:
        """Convert database row to TemplateVersion object"""
        return TemplateVersion(
            version_id=row[0],
            template_id=row[1],
            version_number=row[2],
            status=VersionStatus(row[3]),
            compatibility=VersionCompatibility(row[4]),
            template_config=json.loads(row[5]),
            changelog=json.loads(row[6]),
            breaking_changes=json.loads(row[7]),
            migration_notes=row[8],
            created_by=row[9],
            created_at=datetime.fromisoformat(row[10]),
            deprecated_at=datetime.fromisoformat(row[11]) if row[11] else None,
            checksum=row[12],
            parent_version_id=row[13],
            metadata=json.loads(row[14]) if row[14] else {}
        )
    
    async def update_version_status(
        self,
        version_id: str,
        new_status: VersionStatus
    ) -> bool:
        """Update version status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if new_status == VersionStatus.DEPRECATED:
                cursor.execute('''
                    UPDATE template_versions
                    SET status = ?, deprecated_at = ?
                    WHERE version_id = ?
                ''', (new_status.value, datetime.utcnow().isoformat(), version_id))
            else:
                cursor.execute('''
                    UPDATE template_versions
                    SET status = ?
                    WHERE version_id = ?
                ''', (new_status.value, version_id))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Updated version {version_id} status to {new_status.value}")
            
            # Clear cache
            self.version_cache.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update version status: {e}")
            return False
    
    async def compare_versions(
        self,
        template_id: str,
        old_version: str,
        new_version: str
    ) -> VersionComparisonResult:
        """Compare two versions and identify differences"""
        try:
            old = await self.get_version(template_id, old_version)
            new = await self.get_version(template_id, new_version)
            
            if not old or not new:
                raise ValueError(f"Version not found for comparison")
            
            # Compare configurations
            old_config = old.template_config
            new_config = new.template_config
            
            # Identify added fields
            added_fields = []
            for key in new_config:
                if key not in old_config:
                    added_fields.append(key)
            
            # Identify removed fields
            removed_fields = []
            for key in old_config:
                if key not in new_config:
                    removed_fields.append(key)
            
            # Identify modified fields
            modified_fields = []
            for key in old_config:
                if key in new_config and old_config[key] != new_config[key]:
                    modified_fields.append(key)
            
            # Determine compatibility
            breaking_changes = new.breaking_changes
            migration_required = len(breaking_changes) > 0 or len(removed_fields) > 0
            auto_migration_possible = len(removed_fields) == 0  # Can only auto-migrate if no fields removed
            
            return VersionComparisonResult(
                old_version=old_version,
                new_version=new_version,
                compatibility=new.compatibility,
                added_fields=added_fields,
                removed_fields=removed_fields,
                modified_fields=modified_fields,
                breaking_changes=breaking_changes,
                migration_required=migration_required,
                auto_migration_possible=auto_migration_possible
            )
            
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {e}")
            raise
    
    async def validate_version_integrity(
        self,
        version_id: str
    ) -> Tuple[bool, str]:
        """Validate version integrity using checksum"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT template_config, checksum FROM template_versions
                WHERE version_id = ?
            ''', (version_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return False, "Version not found"
            
            template_config = json.loads(row[0])
            stored_checksum = row[1]
            
            # Calculate current checksum
            current_checksum = self._calculate_checksum(template_config)
            
            if current_checksum == stored_checksum:
                return True, "Integrity verified"
            else:
                return False, f"Checksum mismatch: expected {stored_checksum}, got {current_checksum}"
                
        except Exception as e:
            self.logger.error(f"Failed to validate version integrity: {e}")
            return False, str(e)
    
    async def get_version_history(
        self,
        template_id: str
    ) -> List[Dict[str, Any]]:
        """Get version history with changelog"""
        try:
            versions = await self.get_all_versions(template_id)
            
            history = []
            for version in versions:
                history.append({
                    "version_number": version.version_number,
                    "status": version.status.value,
                    "compatibility": version.compatibility.value,
                    "created_at": version.created_at.isoformat(),
                    "created_by": version.created_by,
                    "changelog": version.changelog,
                    "breaking_changes": version.breaking_changes,
                    "deprecated_at": version.deprecated_at.isoformat() if version.deprecated_at else None
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get version history: {e}")
            return []
    
    async def create_migration_script(
        self,
        from_version_id: str,
        to_version_id: str,
        migration_script: str,
        auto_migration: bool = False
    ) -> str:
        """Create migration script between versions"""
        try:
            migration_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO version_migrations (
                    migration_id, from_version_id, to_version_id,
                    migration_script, auto_migration, tested
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                migration_id,
                from_version_id,
                to_version_id,
                migration_script,
                auto_migration,
                False
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created migration script from {from_version_id} to {to_version_id}")
            
            return migration_id
            
        except Exception as e:
            self.logger.error(f"Failed to create migration script: {e}")
            raise
    
    async def get_migration_path(
        self,
        template_id: str,
        from_version: str,
        to_version: str
    ) -> List[Dict[str, Any]]:
        """Get migration path between two versions"""
        try:
            # Get all versions
            all_versions = await self.get_all_versions(template_id)
            
            # Sort versions by semantic version
            sorted_versions = sorted(
                all_versions,
                key=lambda v: self._parse_version(v.version_number)
            )
            
            # Find start and end indices
            start_idx = next((i for i, v in enumerate(sorted_versions) if v.version_number == from_version), None)
            end_idx = next((i for i, v in enumerate(sorted_versions) if v.version_number == to_version), None)
            
            if start_idx is None or end_idx is None:
                raise ValueError(f"Version not found in path")
            
            # Build migration path
            migration_path = []
            
            if start_idx < end_idx:  # Upgrade path
                for i in range(start_idx, end_idx):
                    current = sorted_versions[i]
                    next_ver = sorted_versions[i + 1]
                    
                    migration_path.append({
                        "from_version": current.version_number,
                        "to_version": next_ver.version_number,
                        "compatibility": next_ver.compatibility.value,
                        "breaking_changes": next_ver.breaking_changes,
                        "migration_notes": next_ver.migration_notes
                    })
            else:  # Downgrade path
                for i in range(start_idx, end_idx, -1):
                    current = sorted_versions[i]
                    prev_ver = sorted_versions[i - 1]
                    
                    migration_path.append({
                        "from_version": current.version_number,
                        "to_version": prev_ver.version_number,
                        "compatibility": "downgrade",
                        "breaking_changes": current.breaking_changes,
                        "migration_notes": f"Downgrading from {current.version_number} to {prev_ver.version_number}"
                    })
            
            return migration_path
            
        except Exception as e:
            self.logger.error(f"Failed to get migration path: {e}")
            return []
    
    async def get_version_statistics(
        self,
        template_id: str
    ) -> Dict[str, Any]:
        """Get version statistics for a template"""
        try:
            versions = await self.get_all_versions(template_id)
            
            stats = {
                "total_versions": len(versions),
                "active_versions": len([v for v in versions if v.status == VersionStatus.ACTIVE]),
                "deprecated_versions": len([v for v in versions if v.status == VersionStatus.DEPRECATED]),
                "draft_versions": len([v for v in versions if v.status == VersionStatus.DRAFT]),
                "beta_versions": len([v for v in versions if v.status == VersionStatus.BETA]),
                "breaking_changes_count": sum(len(v.breaking_changes) for v in versions),
                "latest_version": versions[0].version_number if versions else None,
                "oldest_version": versions[-1].version_number if versions else None,
                "version_timeline": [
                    {
                        "version": v.version_number,
                        "created_at": v.created_at.isoformat(),
                        "status": v.status.value
                    }
                    for v in versions
                ]
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get version statistics: {e}")
            return {}
