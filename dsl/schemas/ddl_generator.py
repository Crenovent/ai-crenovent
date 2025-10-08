# ai-crenovent/dsl/schemas/ddl_generator.py
"""
Task 7.2-T44: Generate DDL (create table/index/rls) - Build-ready SQL
Dynamic DDL generator for RevOps ontology to database schema mapping
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DataType(Enum):
    """PostgreSQL data types"""
    UUID = "UUID"
    BIGINT = "BIGINT"
    INTEGER = "INTEGER"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    TIMESTAMPTZ = "TIMESTAMPTZ"
    JSONB = "JSONB"
    DECIMAL = "DECIMAL"
    DATE = "DATE"

class IndexType(Enum):
    """PostgreSQL index types"""
    BTREE = "BTREE"
    GIN = "GIN"
    GIST = "GIST"
    HASH = "HASH"

@dataclass
class Column:
    """Database column definition"""
    name: str
    data_type: DataType
    size: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    default_value: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_table: Optional[str] = None
    foreign_column: Optional[str] = None
    check_constraint: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class Index:
    """Database index definition"""
    name: str
    table_name: str
    columns: List[str]
    index_type: IndexType = IndexType.BTREE
    unique: bool = False
    partial_condition: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class RLSPolicy:
    """Row Level Security policy definition"""
    name: str
    table_name: str
    command: str  # ALL, SELECT, INSERT, UPDATE, DELETE
    role: Optional[str] = None
    using_expression: str = "true"
    check_expression: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class Table:
    """Database table definition"""
    name: str
    columns: List[Column] = field(default_factory=list)
    indexes: List[Index] = field(default_factory=list)
    rls_policies: List[RLSPolicy] = field(default_factory=list)
    comment: Optional[str] = None
    partition_by: Optional[str] = None
    inherits: Optional[str] = None

class DDLGenerator:
    """
    Task 7.2-T44: Generate DDL (create table/index/rls)
    Dynamic DDL generator for RevOps database schemas
    """
    
    def __init__(self, schema_name: str = "public"):
        self.schema_name = schema_name
        self.tables: Dict[str, Table] = {}
        self.sequences: List[str] = []
        self.extensions: List[str] = ["uuid-ossp", "pgcrypto"]
        
    def add_table(self, table: Table) -> None:
        """Add a table definition"""
        self.tables[table.name] = table
        
    def add_extension(self, extension_name: str) -> None:
        """Add PostgreSQL extension requirement"""
        if extension_name not in self.extensions:
            self.extensions.append(extension_name)
    
    def generate_column_ddl(self, column: Column) -> str:
        """Generate DDL for a single column"""
        ddl_parts = [f'    {column.name}']
        
        # Data type with size/precision
        if column.data_type == DataType.VARCHAR and column.size:
            ddl_parts.append(f'{column.data_type.value}({column.size})')
        elif column.data_type == DataType.DECIMAL and column.precision and column.scale:
            ddl_parts.append(f'{column.data_type.value}({column.precision},{column.scale})')
        else:
            ddl_parts.append(column.data_type.value)
        
        # Primary key
        if column.is_primary_key:
            ddl_parts.append('PRIMARY KEY')
        
        # Nullable/Not null
        if not column.nullable:
            ddl_parts.append('NOT NULL')
        
        # Default value
        if column.default_value:
            if column.data_type in [DataType.UUID] and column.default_value == "gen_random_uuid()":
                ddl_parts.append(f'DEFAULT {column.default_value}')
            elif column.data_type == DataType.TIMESTAMPTZ and column.default_value == "NOW()":
                ddl_parts.append('DEFAULT NOW()')
            elif column.data_type == DataType.BOOLEAN:
                ddl_parts.append(f'DEFAULT {column.default_value}')
            elif column.data_type == DataType.JSONB:
                ddl_parts.append(f"DEFAULT '{column.default_value}'")
            else:
                ddl_parts.append(f"DEFAULT '{column.default_value}'")
        
        # Check constraint
        if column.check_constraint:
            ddl_parts.append(f'CHECK ({column.check_constraint})')
        
        return ' '.join(ddl_parts)
    
    def generate_table_ddl(self, table: Table) -> str:
        """Generate DDL for a complete table"""
        ddl_lines = []
        
        # Table comment
        if table.comment:
            ddl_lines.append(f"-- {table.comment}")
        
        # Create table statement
        create_stmt = f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table.name} ("
        ddl_lines.append(create_stmt)
        
        # Columns
        column_ddls = []
        foreign_keys = []
        
        for column in table.columns:
            column_ddls.append(self.generate_column_ddl(column))
            
            # Collect foreign key constraints
            if column.is_foreign_key and column.foreign_table and column.foreign_column:
                fk_name = f"fk_{table.name}_{column.name}"
                fk_constraint = f"    CONSTRAINT {fk_name} FOREIGN KEY ({column.name}) REFERENCES {self.schema_name}.{column.foreign_table}({column.foreign_column}) ON DELETE CASCADE"
                foreign_keys.append(fk_constraint)
        
        # Add all column definitions
        ddl_lines.extend(column_ddls)
        
        # Add foreign key constraints
        if foreign_keys:
            ddl_lines.append("")  # Empty line for readability
            ddl_lines.extend(foreign_keys)
        
        # Close table definition
        ddl_lines.append(");")
        ddl_lines.append("")  # Empty line
        
        # Partition by clause
        if table.partition_by:
            ddl_lines.append(f"-- Partition table by {table.partition_by}")
            ddl_lines.append(f"-- ALTER TABLE {self.schema_name}.{table.name} PARTITION BY {table.partition_by};")
            ddl_lines.append("")
        
        return "\n".join(ddl_lines)
    
    def generate_index_ddl(self, index: Index) -> str:
        """Generate DDL for an index"""
        ddl_lines = []
        
        if index.comment:
            ddl_lines.append(f"-- {index.comment}")
        
        # Build index statement
        unique_clause = "UNIQUE " if index.unique else ""
        columns_clause = ", ".join(index.columns)
        
        index_stmt = f"CREATE {unique_clause}INDEX IF NOT EXISTS {index.name} ON {self.schema_name}.{index.table_name}"
        
        if index.index_type != IndexType.BTREE:
            index_stmt += f" USING {index.index_type.value}"
        
        index_stmt += f" ({columns_clause})"
        
        if index.partial_condition:
            index_stmt += f" WHERE {index.partial_condition}"
        
        index_stmt += ";"
        
        ddl_lines.append(index_stmt)
        ddl_lines.append("")
        
        return "\n".join(ddl_lines)
    
    def generate_rls_policy_ddl(self, policy: RLSPolicy) -> str:
        """Generate DDL for RLS policy"""
        ddl_lines = []
        
        if policy.comment:
            ddl_lines.append(f"-- {policy.comment}")
        
        # Enable RLS on table first
        ddl_lines.append(f"ALTER TABLE {self.schema_name}.{policy.table_name} ENABLE ROW LEVEL SECURITY;")
        
        # Create policy
        policy_stmt = f"CREATE POLICY {policy.name} ON {self.schema_name}.{policy.table_name}"
        policy_stmt += f" FOR {policy.command}"
        
        if policy.role:
            policy_stmt += f" TO {policy.role}"
        
        policy_stmt += f" USING ({policy.using_expression})"
        
        if policy.check_expression:
            policy_stmt += f" WITH CHECK ({policy.check_expression})"
        
        policy_stmt += ";"
        
        ddl_lines.append(policy_stmt)
        ddl_lines.append("")
        
        return "\n".join(ddl_lines)
    
    def generate_complete_ddl(self) -> str:
        """Generate complete DDL script"""
        ddl_sections = []
        
        # Header
        ddl_sections.append("-- =====================================================")
        ddl_sections.append("-- REVOPS ONTOLOGY DATABASE SCHEMA DDL")
        ddl_sections.append(f"-- Generated on: {datetime.now(timezone.utc).isoformat()}")
        ddl_sections.append(f"-- Schema: {self.schema_name}")
        ddl_sections.append("-- Task 7.2-T44: Generate DDL (create table/index/rls)")
        ddl_sections.append("-- =====================================================")
        ddl_sections.append("")
        
        # Extensions
        ddl_sections.append("-- Required PostgreSQL extensions")
        for ext in self.extensions:
            ddl_sections.append(f'CREATE EXTENSION IF NOT EXISTS "{ext}";')
        ddl_sections.append("")
        
        # Schema creation
        if self.schema_name != "public":
            ddl_sections.append(f"-- Create schema")
            ddl_sections.append(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")
            ddl_sections.append("")
        
        # Tables
        ddl_sections.append("-- =====================================================")
        ddl_sections.append("-- TABLES")
        ddl_sections.append("-- =====================================================")
        ddl_sections.append("")
        
        for table_name, table in self.tables.items():
            ddl_sections.append(self.generate_table_ddl(table))
        
        # Indexes
        ddl_sections.append("-- =====================================================")
        ddl_sections.append("-- INDEXES")
        ddl_sections.append("-- =====================================================")
        ddl_sections.append("")
        
        for table in self.tables.values():
            for index in table.indexes:
                ddl_sections.append(self.generate_index_ddl(index))
        
        # RLS Policies
        ddl_sections.append("-- =====================================================")
        ddl_sections.append("-- ROW LEVEL SECURITY POLICIES")
        ddl_sections.append("-- =====================================================")
        ddl_sections.append("")
        
        for table in self.tables.values():
            for policy in table.rls_policies:
                ddl_sections.append(self.generate_rls_policy_ddl(policy))
        
        return "\n".join(ddl_sections)
    
    def save_ddl_to_file(self, file_path: str) -> None:
        """Save generated DDL to file"""
        ddl_content = self.generate_complete_ddl()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(ddl_content)
        
        logger.info(f"DDL generated and saved to: {file_path}")

def create_sample_revops_schema() -> DDLGenerator:
    """Create sample RevOps schema for demonstration"""
    generator = DDLGenerator(schema_name="revops")
    
    # Tenant metadata table
    tenant_table = Table(
        name="tenant_metadata",
        comment="Multi-tenant isolation and metadata - Task 7.2-T04"
    )
    
    tenant_table.columns = [
        Column("tenant_id", DataType.INTEGER, is_primary_key=True, nullable=False),
        Column("tenant_name", DataType.VARCHAR, size=255, nullable=False),
        Column("industry_code", DataType.VARCHAR, size=10, nullable=False, 
               check_constraint="industry_code IN ('SaaS', 'BANK', 'INSUR', 'ECOMM', 'FS', 'IT')"),
        Column("region_code", DataType.VARCHAR, size=10, nullable=False,
               check_constraint="region_code IN ('US', 'EU', 'IN', 'APAC')"),
        Column("compliance_requirements", DataType.JSONB, default_value="[]"),
        Column("status", DataType.VARCHAR, size=20, default_value="active",
               check_constraint="status IN ('active', 'suspended', 'offboarding')"),
        Column("created_at", DataType.TIMESTAMPTZ, default_value="NOW()", nullable=False),
        Column("updated_at", DataType.TIMESTAMPTZ, default_value="NOW()", nullable=False),
        Column("metadata", DataType.JSONB, default_value="{}")
    ]
    
    # Add indexes for tenant table
    tenant_table.indexes = [
        Index("idx_tenant_industry", "tenant_metadata", ["industry_code"]),
        Index("idx_tenant_region", "tenant_metadata", ["region_code"]),
        Index("idx_tenant_status", "tenant_metadata", ["status"])
    ]
    
    generator.add_table(tenant_table)
    
    # Account table - Task 7.2-T05
    account_table = Table(
        name="accounts",
        comment="CRM canonical account entity - Task 7.2-T05"
    )
    
    account_table.columns = [
        Column("account_id", DataType.UUID, default_value="gen_random_uuid()", is_primary_key=True),
        Column("tenant_id", DataType.INTEGER, nullable=False, is_foreign_key=True, 
               foreign_table="tenant_metadata", foreign_column="tenant_id"),
        Column("external_id", DataType.VARCHAR, size=255, comment="Source system ID"),
        Column("account_name", DataType.VARCHAR, size=255, nullable=False),
        Column("account_type", DataType.VARCHAR, size=50,
               check_constraint="account_type IN ('prospect', 'customer', 'partner', 'competitor')"),
        Column("industry", DataType.VARCHAR, size=100),
        Column("annual_revenue", DataType.DECIMAL, precision=15, scale=2),
        Column("employee_count", DataType.INTEGER),
        Column("website", DataType.VARCHAR, size=255),
        Column("billing_address", DataType.JSONB),
        Column("shipping_address", DataType.JSONB),
        Column("created_at", DataType.TIMESTAMPTZ, default_value="NOW()", nullable=False),
        Column("updated_at", DataType.TIMESTAMPTZ, default_value="NOW()", nullable=False),
        Column("created_by_user_id", DataType.INTEGER),
        Column("is_deleted", DataType.BOOLEAN, default_value="false")
    ]
    
    # Add RLS policy for tenant isolation
    account_table.rls_policies = [
        RLSPolicy(
            name="tenant_isolation_accounts",
            table_name="accounts",
            command="ALL",
            using_expression="tenant_id = current_setting('app.tenant_id')::integer",
            comment="Enforce tenant isolation for accounts"
        )
    ]
    
    # Add indexes
    account_table.indexes = [
        Index("idx_accounts_tenant", "accounts", ["tenant_id"]),
        Index("idx_accounts_external_id", "accounts", ["tenant_id", "external_id"], unique=True),
        Index("idx_accounts_name", "accounts", ["account_name"]),
        Index("idx_accounts_type", "accounts", ["account_type"]),
        Index("idx_accounts_created", "accounts", ["created_at"])
    ]
    
    generator.add_table(account_table)
    
    # Opportunity table - Task 7.2-T06
    opportunity_table = Table(
        name="opportunities",
        comment="Deal flow canonical entity - Task 7.2-T06"
    )
    
    opportunity_table.columns = [
        Column("opportunity_id", DataType.UUID, default_value="gen_random_uuid()", is_primary_key=True),
        Column("tenant_id", DataType.INTEGER, nullable=False, is_foreign_key=True,
               foreign_table="tenant_metadata", foreign_column="tenant_id"),
        Column("account_id", DataType.UUID, nullable=False, is_foreign_key=True,
               foreign_table="accounts", foreign_column="account_id"),
        Column("external_id", DataType.VARCHAR, size=255),
        Column("opportunity_name", DataType.VARCHAR, size=255, nullable=False),
        Column("stage", DataType.VARCHAR, size=100, nullable=False),
        Column("amount", DataType.DECIMAL, precision=15, scale=2),
        Column("probability", DataType.DECIMAL, precision=5, scale=2,
               check_constraint="probability >= 0 AND probability <= 100"),
        Column("close_date", DataType.DATE),
        Column("owner_user_id", DataType.INTEGER),
        Column("lead_source", DataType.VARCHAR, size=100),
        Column("next_step", DataType.TEXT),
        Column("description", DataType.TEXT),
        Column("created_at", DataType.TIMESTAMPTZ, default_value="NOW()", nullable=False),
        Column("updated_at", DataType.TIMESTAMPTZ, default_value="NOW()", nullable=False),
        Column("is_deleted", DataType.BOOLEAN, default_value="false")
    ]
    
    # RLS policy for opportunities
    opportunity_table.rls_policies = [
        RLSPolicy(
            name="tenant_isolation_opportunities",
            table_name="opportunities",
            command="ALL",
            using_expression="tenant_id = current_setting('app.tenant_id')::integer"
        )
    ]
    
    # Indexes for opportunities
    opportunity_table.indexes = [
        Index("idx_opportunities_tenant", "opportunities", ["tenant_id"]),
        Index("idx_opportunities_account", "opportunities", ["account_id"]),
        Index("idx_opportunities_stage", "opportunities", ["stage"]),
        Index("idx_opportunities_close_date", "opportunities", ["close_date"]),
        Index("idx_opportunities_owner", "opportunities", ["owner_user_id"]),
        Index("idx_opportunities_amount", "opportunities", ["amount"]),
        Index("idx_opportunities_external_id", "opportunities", ["tenant_id", "external_id"], unique=True)
    ]
    
    generator.add_table(opportunity_table)
    
    return generator

# Example usage
if __name__ == "__main__":
    # Create sample schema
    generator = create_sample_revops_schema()
    
    # Generate and save DDL
    output_path = "ai-crenovent/database/generated_revops_schema.sql"
    generator.save_ddl_to_file(output_path)
    
    print(f"✅ Task 7.2-T44 completed: DDL generated at {output_path}")
    print(f"Generated {len(generator.tables)} tables with indexes and RLS policies")
