# ai-crenovent/dsl/schemas/migration_generator.py
"""
Task 7.2-T45: Author migration scripts (up/down) - Safe evolution
Dynamic migration script generator for database schema evolution
"""

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MigrationType(Enum):
    """Types of database migrations"""
    CREATE_TABLE = "create_table"
    ALTER_TABLE = "alter_table"
    DROP_TABLE = "drop_table"
    CREATE_INDEX = "create_index"
    DROP_INDEX = "drop_index"
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    MODIFY_COLUMN = "modify_column"
    CREATE_CONSTRAINT = "create_constraint"
    DROP_CONSTRAINT = "drop_constraint"
    CREATE_RLS_POLICY = "create_rls_policy"
    DROP_RLS_POLICY = "drop_rls_policy"
    INSERT_DATA = "insert_data"
    UPDATE_DATA = "update_data"
    DELETE_DATA = "delete_data"

@dataclass
class MigrationStep:
    """Individual migration step"""
    step_id: str
    migration_type: MigrationType
    description: str
    up_sql: str
    down_sql: str
    dependencies: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    rollback_notes: Optional[str] = None
    validation_sql: Optional[str] = None

@dataclass
class Migration:
    """Complete migration with metadata"""
    migration_id: str
    version: str
    title: str
    description: str
    author: str
    created_at: datetime
    steps: List[MigrationStep] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    risk_assessment: str = "low"
    estimated_duration: str = "< 1 minute"
    rollback_strategy: str = "automatic"

class MigrationGenerator:
    """
    Task 7.2-T45: Author migration scripts (up/down)
    Generates safe database migration scripts with rollback capability
    """
    
    def __init__(self, migrations_dir: str = "ai-crenovent/database/migrations"):
        self.migrations_dir = migrations_dir
        self.migrations: Dict[str, Migration] = {}
        
        # Ensure migrations directory exists
        os.makedirs(migrations_dir, exist_ok=True)
        
    def create_migration(self, title: str, description: str, author: str = "System") -> Migration:
        """Create a new migration"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        migration_id = f"{timestamp}_{self._slugify(title)}"
        version = f"v{timestamp}"
        
        migration = Migration(
            migration_id=migration_id,
            version=version,
            title=title,
            description=description,
            author=author,
            created_at=datetime.now(timezone.utc)
        )
        
        self.migrations[migration_id] = migration
        return migration
    
    def _slugify(self, text: str) -> str:
        """Convert text to slug format"""
        return re.sub(r'[^a-zA-Z0-9]+', '_', text.lower()).strip('_')
    
    def add_create_table_step(self, migration: Migration, table_name: str, 
                             columns_ddl: str, constraints_ddl: str = "") -> None:
        """Add create table migration step"""
        step_id = f"create_table_{table_name}"
        
        up_sql = f"""
-- Create table {table_name}
CREATE TABLE IF NOT EXISTS {table_name} (
{columns_ddl}
{constraints_ddl}
);
"""
        
        down_sql = f"""
-- Drop table {table_name}
DROP TABLE IF EXISTS {table_name} CASCADE;
"""
        
        validation_sql = f"""
-- Validate table {table_name} exists
SELECT 1 FROM information_schema.tables 
WHERE table_name = '{table_name}' AND table_schema = current_schema();
"""
        
        step = MigrationStep(
            step_id=step_id,
            migration_type=MigrationType.CREATE_TABLE,
            description=f"Create table {table_name}",
            up_sql=up_sql.strip(),
            down_sql=down_sql.strip(),
            validation_sql=validation_sql.strip(),
            risk_level="medium"
        )
        
        migration.steps.append(step)
    
    def add_add_column_step(self, migration: Migration, table_name: str, 
                           column_name: str, column_definition: str) -> None:
        """Add column addition migration step"""
        step_id = f"add_column_{table_name}_{column_name}"
        
        up_sql = f"""
-- Add column {column_name} to {table_name}
ALTER TABLE {table_name} 
ADD COLUMN IF NOT EXISTS {column_name} {column_definition};
"""
        
        down_sql = f"""
-- Remove column {column_name} from {table_name}
ALTER TABLE {table_name} 
DROP COLUMN IF EXISTS {column_name};
"""
        
        validation_sql = f"""
-- Validate column {column_name} exists in {table_name}
SELECT 1 FROM information_schema.columns 
WHERE table_name = '{table_name}' 
  AND column_name = '{column_name}' 
  AND table_schema = current_schema();
"""
        
        step = MigrationStep(
            step_id=step_id,
            migration_type=MigrationType.ADD_COLUMN,
            description=f"Add column {column_name} to table {table_name}",
            up_sql=up_sql.strip(),
            down_sql=down_sql.strip(),
            validation_sql=validation_sql.strip(),
            risk_level="low",
            rollback_notes="Column will be dropped. Ensure no data loss is acceptable."
        )
        
        migration.steps.append(step)
    
    def add_create_index_step(self, migration: Migration, table_name: str, 
                             index_name: str, columns: List[str], 
                             unique: bool = False, index_type: str = "BTREE") -> None:
        """Add create index migration step"""
        step_id = f"create_index_{index_name}"
        
        unique_clause = "UNIQUE " if unique else ""
        columns_clause = ", ".join(columns)
        
        up_sql = f"""
-- Create index {index_name} on {table_name}
CREATE {unique_clause}INDEX IF NOT EXISTS {index_name} 
ON {table_name} USING {index_type} ({columns_clause});
"""
        
        down_sql = f"""
-- Drop index {index_name}
DROP INDEX IF EXISTS {index_name};
"""
        
        validation_sql = f"""
-- Validate index {index_name} exists
SELECT 1 FROM pg_indexes 
WHERE indexname = '{index_name}' AND schemaname = current_schema();
"""
        
        step = MigrationStep(
            step_id=step_id,
            migration_type=MigrationType.CREATE_INDEX,
            description=f"Create index {index_name} on {table_name}({columns_clause})",
            up_sql=up_sql.strip(),
            down_sql=down_sql.strip(),
            validation_sql=validation_sql.strip(),
            risk_level="low"
        )
        
        migration.steps.append(step)
    
    def add_rls_policy_step(self, migration: Migration, table_name: str, 
                           policy_name: str, command: str, using_expression: str) -> None:
        """Add RLS policy migration step"""
        step_id = f"create_rls_policy_{policy_name}"
        
        up_sql = f"""
-- Enable RLS on {table_name}
ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;

-- Create RLS policy {policy_name}
CREATE POLICY {policy_name} ON {table_name}
FOR {command}
USING ({using_expression});
"""
        
        down_sql = f"""
-- Drop RLS policy {policy_name}
DROP POLICY IF EXISTS {policy_name} ON {table_name};

-- Note: RLS remains enabled on table for safety
"""
        
        validation_sql = f"""
-- Validate RLS policy {policy_name} exists
SELECT 1 FROM pg_policies 
WHERE policyname = '{policy_name}' 
  AND tablename = '{table_name}' 
  AND schemaname = current_schema();
"""
        
        step = MigrationStep(
            step_id=step_id,
            migration_type=MigrationType.CREATE_RLS_POLICY,
            description=f"Create RLS policy {policy_name} on {table_name}",
            up_sql=up_sql.strip(),
            down_sql=down_sql.strip(),
            validation_sql=validation_sql.strip(),
            risk_level="medium",
            rollback_notes="RLS policy will be dropped but RLS remains enabled on table"
        )
        
        migration.steps.append(step)
    
    def add_insert_data_step(self, migration: Migration, table_name: str, 
                            data_description: str, insert_sql: str, 
                            delete_condition: str) -> None:
        """Add data insertion migration step"""
        step_id = f"insert_data_{table_name}_{self._slugify(data_description)}"
        
        up_sql = f"""
-- Insert {data_description} into {table_name}
{insert_sql}
"""
        
        down_sql = f"""
-- Remove {data_description} from {table_name}
DELETE FROM {table_name} WHERE {delete_condition};
"""
        
        step = MigrationStep(
            step_id=step_id,
            migration_type=MigrationType.INSERT_DATA,
            description=f"Insert {data_description} into {table_name}",
            up_sql=up_sql.strip(),
            down_sql=down_sql.strip(),
            risk_level="low"
        )
        
        migration.steps.append(step)
    
    def generate_migration_file(self, migration: Migration) -> str:
        """Generate complete migration file content"""
        content_lines = []
        
        # Header
        content_lines.extend([
            "-- =====================================================",
            f"-- MIGRATION: {migration.title}",
            f"-- Version: {migration.version}",
            f"-- ID: {migration.migration_id}",
            f"-- Author: {migration.author}",
            f"-- Created: {migration.created_at.isoformat()}",
            f"-- Risk Assessment: {migration.risk_assessment}",
            f"-- Estimated Duration: {migration.estimated_duration}",
            f"-- Rollback Strategy: {migration.rollback_strategy}",
            "-- =====================================================",
            "",
            f"-- Description: {migration.description}",
            ""
        ])
        
        # Dependencies
        if migration.dependencies:
            content_lines.append("-- Dependencies:")
            for dep in migration.dependencies:
                content_lines.append(f"--   - {dep}")
            content_lines.append("")
        
        # Tags
        if migration.tags:
            content_lines.append(f"-- Tags: {', '.join(migration.tags)}")
            content_lines.append("")
        
        # Migration steps (UP)
        content_lines.extend([
            "-- =====================================================",
            "-- UP MIGRATION",
            "-- =====================================================",
            ""
        ])
        
        for i, step in enumerate(migration.steps, 1):
            content_lines.extend([
                f"-- Step {i}: {step.description}",
                f"-- Risk Level: {step.risk_level}",
                ""
            ])
            
            if step.validation_sql:
                content_lines.extend([
                    "-- Validation check",
                    "DO $$",
                    "BEGIN",
                    f"    IF NOT EXISTS ({step.validation_sql.replace('SELECT 1', 'SELECT 1')}) THEN",
                    "        -- Proceed with migration step",
                    "        NULL;",
                    "    END IF;",
                    "END $$;",
                    ""
                ])
            
            content_lines.extend([
                step.up_sql,
                "",
                "-- Verify step completion",
                f"-- {step.validation_sql if step.validation_sql else 'Manual verification required'}",
                "",
                "-- " + "="*50,
                ""
            ])
        
        # Rollback instructions (DOWN)
        content_lines.extend([
            "",
            "-- =====================================================",
            "-- DOWN MIGRATION (ROLLBACK)",
            "-- =====================================================",
            "-- IMPORTANT: Review rollback steps carefully before execution",
            "-- Execute steps in REVERSE order for proper rollback",
            ""
        ])
        
        for i, step in enumerate(reversed(migration.steps), 1):
            content_lines.extend([
                f"-- Rollback Step {i}: Reverse {step.description}",
                f"-- Risk Level: {step.risk_level}",
                ""
            ])
            
            if step.rollback_notes:
                content_lines.extend([
                    f"-- ROLLBACK NOTES: {step.rollback_notes}",
                    ""
                ])
            
            content_lines.extend([
                "/*",
                step.down_sql,
                "*/",
                "",
                "-- " + "="*50,
                ""
            ])
        
        # Footer
        content_lines.extend([
            "",
            "-- =====================================================",
            "-- MIGRATION COMPLETE",
            f"-- {migration.title} - {migration.version}",
            "-- ====================================================="
        ])
        
        return "\n".join(content_lines)
    
    def save_migration(self, migration: Migration) -> str:
        """Save migration to file"""
        filename = f"{migration.migration_id}.sql"
        file_path = os.path.join(self.migrations_dir, filename)
        
        content = self.generate_migration_file(migration)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Migration saved: {file_path}")
        return file_path
    
    def create_sample_tenant_migration(self) -> Migration:
        """Create sample migration for tenant metadata table"""
        migration = self.create_migration(
            title="Create Tenant Metadata Table",
            description="Initial migration to create tenant_metadata table with RLS policies for multi-tenant isolation",
            author="RevOps Team"
        )
        
        # Add create table step
        columns_ddl = """    tenant_id INTEGER PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    industry_code VARCHAR(10) NOT NULL CHECK (industry_code IN ('SaaS', 'BANK', 'INSUR', 'ECOMM', 'FS', 'IT')),
    region_code VARCHAR(10) NOT NULL CHECK (region_code IN ('US', 'EU', 'IN', 'APAC')),
    compliance_requirements JSONB DEFAULT '[]',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'offboarding')),
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    metadata JSONB DEFAULT '{}'"""
        
        self.add_create_table_step(migration, "tenant_metadata", columns_ddl)
        
        # Add indexes
        self.add_create_index_step(migration, "tenant_metadata", "idx_tenant_industry", ["industry_code"])
        self.add_create_index_step(migration, "tenant_metadata", "idx_tenant_region", ["region_code"])
        self.add_create_index_step(migration, "tenant_metadata", "idx_tenant_status", ["status"])
        
        # Add sample data
        sample_data_sql = """
INSERT INTO tenant_metadata (tenant_id, tenant_name, industry_code, region_code, compliance_requirements) 
VALUES 
    (1000, 'Demo SaaS Company', 'SaaS', 'US', '["SOX", "GDPR"]'),
    (1001, 'Sample Bank Corp', 'BANK', 'US', '["SOX", "RBI"]'),
    (1002, 'Test Insurance Ltd', 'INSUR', 'EU', '["GDPR", "IRDAI"]')
ON CONFLICT (tenant_id) DO NOTHING;
"""
        
        self.add_insert_data_step(
            migration, 
            "tenant_metadata", 
            "sample tenant data", 
            sample_data_sql,
            "tenant_id IN (1000, 1001, 1002)"
        )
        
        migration.risk_assessment = "medium"
        migration.estimated_duration = "2-3 minutes"
        migration.tags = ["initial", "tenant", "multi-tenant", "rls"]
        
        return migration

# Example usage
if __name__ == "__main__":
    # Create migration generator
    generator = MigrationGenerator()
    
    # Create sample migration
    migration = generator.create_sample_tenant_migration()
    
    # Save migration file
    file_path = generator.save_migration(migration)
    
    print(f"✅ Task 7.2-T45 completed: Migration script generated at {file_path}")
    print(f"Migration ID: {migration.migration_id}")
    print(f"Steps: {len(migration.steps)}")
