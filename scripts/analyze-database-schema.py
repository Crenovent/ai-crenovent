#!/usr/bin/env python3
"""
Comprehensive Database Schema Analysis
Analyzes all tables, columns, and relationships used by the onboarding agent and related components
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.append('src')

async def analyze_complete_database_schema():
    """Analyze the complete database schema used by the application"""
    try:
        from src.services.connection_pool_manager import pool_manager
        
        # Initialize pool manager
        await pool_manager.initialize()
        
        schema_analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'database_info': {},
            'tables': {},
            'relationships': [],
            'onboarding_requirements': {}
        }
        
        async with pool_manager.postgres_pool.acquire() as conn:
            # Get database information
            db_info = await conn.fetchrow("""
                SELECT 
                    current_database() as database_name,
                    current_user as current_user,
                    version() as postgres_version
            """)
            
            schema_analysis['database_info'] = dict(db_info)
            print(f"🗄️ Database: {db_info['database_name']}")
            print(f"👤 User: {db_info['current_user']}")
            print(f"🐘 PostgreSQL: {db_info['postgres_version'][:50]}...")
            
            # Get all tables in the database
            tables = await conn.fetch("""
                SELECT table_name, table_type
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            print(f"\n📋 Found {len(tables)} tables:")
            for table in tables:
                print(f"   - {table['table_name']} ({table['table_type']})")
            
            # Analyze each table
            for table in tables:
                table_name = table['table_name']
                print(f"\n🔍 Analyzing table: {table_name}")
                
                # Get table schema
                columns = await conn.fetch("""
                    SELECT 
                        column_name, 
                        data_type, 
                        is_nullable, 
                        column_default,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale
                    FROM information_schema.columns 
                    WHERE table_name = $1 AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, table_name)
                
                # Get table constraints
                constraints = await conn.fetch("""
                    SELECT 
                        constraint_name,
                        constraint_type,
                        column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.constraint_column_usage ccu 
                        ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.table_name = $1 AND tc.table_schema = 'public'
                """, table_name)
                
                # Get foreign key relationships
                foreign_keys = await conn.fetch("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_name = $1
                        AND tc.table_schema = 'public'
                """, table_name)
                
                # Get row count
                try:
                    row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                except:
                    row_count = 0
                
                # Get sample data if table has rows
                sample_data = []
                if row_count > 0:
                    try:
                        sample_rows = await conn.fetch(f"SELECT * FROM {table_name} LIMIT 3")
                        for row in sample_rows:
                            sample_record = {}
                            for key, value in row.items():
                                if isinstance(value, str) and len(value) > 100:
                                    sample_record[key] = value[:100] + "..."
                                else:
                                    sample_record[key] = str(value) if value is not None else None
                            sample_data.append(sample_record)
                    except Exception as e:
                        sample_data = [{"error": f"Could not fetch sample: {str(e)}"}]
                
                # Store table analysis
                schema_analysis['tables'][table_name] = {
                    'type': table['table_type'],
                    'row_count': row_count,
                    'columns': [
                        {
                            'name': col['column_name'],
                            'type': col['data_type'],
                            'nullable': col['is_nullable'] == 'YES',
                            'default': col['column_default'],
                            'max_length': col['character_maximum_length'],
                            'precision': col['numeric_precision'],
                            'scale': col['numeric_scale']
                        }
                        for col in columns
                    ],
                    'constraints': [
                        {
                            'name': const['constraint_name'],
                            'type': const['constraint_type'],
                            'column': const['column_name']
                        }
                        for const in constraints
                    ],
                    'foreign_keys': [
                        {
                            'column': fk['column_name'],
                            'references_table': fk['foreign_table_name'],
                            'references_column': fk['foreign_column_name']
                        }
                        for fk in foreign_keys
                    ],
                    'sample_data': sample_data
                }
                
                print(f"   📊 {len(columns)} columns, {row_count} rows")
                for col in columns[:5]:  # Show first 5 columns
                    nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                    print(f"      - {col['column_name']}: {col['data_type']} {nullable}")
                if len(columns) > 5:
                    print(f"      ... and {len(columns) - 5} more columns")
        
        # Analyze onboarding-specific requirements
        await analyze_onboarding_requirements(schema_analysis)
        
        await pool_manager.close()
        
        # Save analysis to file
        with open('database_schema_analysis.json', 'w') as f:
            json.dump(schema_analysis, f, indent=2, default=str)
        
        print(f"\n💾 Complete analysis saved to: database_schema_analysis.json")
        
        return schema_analysis
        
    except Exception as e:
        print(f"❌ Error analyzing database schema: {e}")
        import traceback
        traceback.print_exc()
        return None

async def analyze_onboarding_requirements(schema_analysis: Dict[str, Any]):
    """Analyze specific requirements for the onboarding agent"""
    print(f"\n🎯 Analyzing Onboarding Agent Requirements...")
    
    # Check if users table exists and analyze its structure
    users_table = schema_analysis['tables'].get('users')
    if users_table:
        print(f"✅ Users table found with {users_table['row_count']} records")
        
        # Analyze users table columns
        user_columns = {col['name']: col for col in users_table['columns']}
        
        required_columns = [
            'user_id', 'username', 'email', 'tenant_id', 'profile', 
            'reports_to', 'is_activated', 'password', 'access_token', 
            'refresh_token', 'expiration_date', 'created_at', 'updated_at'
        ]
        
        missing_columns = []
        existing_columns = []
        
        for col_name in required_columns:
            if col_name in user_columns:
                existing_columns.append({
                    'name': col_name,
                    'type': user_columns[col_name]['type'],
                    'nullable': user_columns[col_name]['nullable']
                })
            else:
                missing_columns.append(col_name)
        
        schema_analysis['onboarding_requirements'] = {
            'users_table_exists': True,
            'users_table_row_count': users_table['row_count'],
            'required_columns': required_columns,
            'existing_columns': existing_columns,
            'missing_columns': missing_columns,
            'profile_column_analysis': analyze_profile_column(users_table),
            'tenant_support': 'tenant_id' in user_columns,
            'hierarchy_support': 'reports_to' in user_columns
        }
        
        print(f"   📋 Required columns: {len(required_columns)}")
        print(f"   ✅ Existing columns: {len(existing_columns)}")
        print(f"   ❌ Missing columns: {len(missing_columns)}")
        
        if missing_columns:
            print(f"   🚨 Missing columns: {', '.join(missing_columns)}")
        
        # Analyze profile column structure
        if 'profile' in user_columns:
            profile_analysis = analyze_profile_column(users_table)
            print(f"   📄 Profile column: {profile_analysis['type']} - {profile_analysis['description']}")
    else:
        print(f"❌ Users table not found!")
        schema_analysis['onboarding_requirements'] = {
            'users_table_exists': False,
            'error': 'Users table not found in database'
        }
    
    # Check for other related tables
    related_tables = ['strategic_account_plans', 'user_roles', 'permissions', 'tenants']
    found_related = []
    missing_related = []
    
    for table_name in related_tables:
        if table_name in schema_analysis['tables']:
            found_related.append(table_name)
        else:
            missing_related.append(table_name)
    
    schema_analysis['onboarding_requirements']['related_tables'] = {
        'found': found_related,
        'missing': missing_related
    }
    
    print(f"   🔗 Related tables found: {', '.join(found_related) if found_related else 'None'}")
    print(f"   🔗 Related tables missing: {', '.join(missing_related) if missing_related else 'None'}")

def analyze_profile_column(users_table: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the profile column structure"""
    profile_col = None
    for col in users_table['columns']:
        if col['name'] == 'profile':
            profile_col = col
            break
    
    if not profile_col:
        return {'exists': False}
    
    analysis = {
        'exists': True,
        'type': profile_col['type'],
        'nullable': profile_col['nullable'],
        'description': 'JSON column storing user profile data'
    }
    
    # Analyze sample profile data
    if users_table['sample_data']:
        sample_profiles = []
        for sample in users_table['sample_data']:
            if 'profile' in sample and sample['profile']:
                try:
                    if isinstance(sample['profile'], str):
                        profile_data = json.loads(sample['profile'])
                        sample_profiles.append(list(profile_data.keys()))
                except:
                    pass
        
        if sample_profiles:
            # Find common profile fields
            all_fields = set()
            for profile_fields in sample_profiles:
                all_fields.update(profile_fields)
            
            analysis['sample_fields'] = list(all_fields)
            analysis['common_fields'] = list(all_fields)  # Simplified for now
    
    return analysis

def generate_schema_creation_sql(schema_analysis: Dict[str, Any]) -> str:
    """Generate SQL to create missing tables and columns"""
    sql_statements = []
    
    onboarding_req = schema_analysis.get('onboarding_requirements', {})
    
    if not onboarding_req.get('users_table_exists', False):
        # Create users table
        sql_statements.append("""
-- Create users table for onboarding agent
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    tenant_id INTEGER NOT NULL,
    reports_to BIGINT REFERENCES users(user_id),
    is_activated BOOLEAN DEFAULT TRUE,
    profile JSONB,
    password VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    expiration_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_tenant_id ON users(tenant_id);
CREATE INDEX idx_users_reports_to ON users(reports_to);
CREATE INDEX idx_users_profile ON users USING GIN(profile);
        """.strip())
    
    elif onboarding_req.get('missing_columns'):
        # Add missing columns
        for col_name in onboarding_req['missing_columns']:
            column_def = get_column_definition(col_name)
            sql_statements.append(f"ALTER TABLE users ADD COLUMN {col_name} {column_def};")
    
    return "\n\n".join(sql_statements)

def get_column_definition(column_name: str) -> str:
    """Get the SQL definition for a column"""
    column_definitions = {
        'user_id': 'BIGINT PRIMARY KEY',
        'username': 'VARCHAR(255) NOT NULL',
        'email': 'VARCHAR(255) UNIQUE NOT NULL',
        'tenant_id': 'INTEGER NOT NULL',
        'reports_to': 'BIGINT REFERENCES users(user_id)',
        'is_activated': 'BOOLEAN DEFAULT TRUE',
        'profile': 'JSONB',
        'password': 'VARCHAR(255) NOT NULL',
        'access_token': 'TEXT',
        'refresh_token': 'TEXT',
        'expiration_date': 'DATE',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
    }
    
    return column_definitions.get(column_name, 'TEXT')

async def main():
    """Main analysis function"""
    print("🔍 Comprehensive Database Schema Analysis")
    print("=" * 60)
    
    schema_analysis = await analyze_complete_database_schema()
    
    if schema_analysis:
        print(f"\n📊 Analysis Summary:")
        print(f"   🗄️ Database: {schema_analysis['database_info']['database_name']}")
        print(f"   📋 Tables found: {len(schema_analysis['tables'])}")
        
        # Generate SQL for missing components
        sql_creation = generate_schema_creation_sql(schema_analysis)
        if sql_creation:
            print(f"\n🔧 SQL Creation Script:")
            print("=" * 40)
            print(sql_creation)
            
            # Save SQL to file
            with open('create_missing_schema.sql', 'w') as f:
                f.write(sql_creation)
            print(f"\n💾 SQL script saved to: create_missing_schema.sql")
        
        # Print onboarding requirements summary
        onboarding_req = schema_analysis.get('onboarding_requirements', {})
        if onboarding_req:
            print(f"\n🎯 Onboarding Agent Requirements:")
            print(f"   ✅ Users table exists: {onboarding_req.get('users_table_exists', False)}")
            if onboarding_req.get('users_table_exists'):
                print(f"   📊 User records: {onboarding_req.get('users_table_row_count', 0)}")
                print(f"   🏢 Tenant support: {onboarding_req.get('tenant_support', False)}")
                print(f"   👥 Hierarchy support: {onboarding_req.get('hierarchy_support', False)}")
                
                missing_cols = onboarding_req.get('missing_columns', [])
                if missing_cols:
                    print(f"   ❌ Missing columns: {', '.join(missing_cols)}")
                else:
                    print(f"   ✅ All required columns present")
        
        print(f"\n✅ Analysis complete!")
    else:
        print(f"\n❌ Analysis failed!")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(main())