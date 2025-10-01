#!/usr/bin/env python3
"""
Azure Database Configuration for Intelligence System
===================================================

This module configures the connection to Azure PostgreSQL database used by your Node.js backend.
It integrates with the existing database schema and ensures proper multi-tenant isolation.

The connection configuration matches your Node.js backend's database setup.
"""

import os
import asyncio
import asyncpg
import logging
from typing import Dict, Any, Optional
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

logger = logging.getLogger(__name__)

class AzureDatabaseConfig:
    """
    Azure database configuration manager that integrates with your existing system
    """
    
    def __init__(self):
        self.credential = None
        self.secret_client = None
        self.connection_config = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize Azure Key Vault connection and retrieve database secrets"""
        try:
            logger.info("🔐 Initializing Azure Database Configuration...")
            
            # Initialize Azure credentials
            self.credential = DefaultAzureCredential()
            
            # Get Key Vault URI from environment
            key_vault_uri = os.getenv('AZURE_KEY_VAULTS_URI')
            if not key_vault_uri:
                logger.warning("⚠️ AZURE_KEY_VAULTS_URI not found, using environment variables")
                return await self._initialize_from_env()
            
            # Initialize Key Vault client
            self.secret_client = SecretClient(vault_url=key_vault_uri, credential=self.credential)
            
            # Retrieve database secrets
            await self._retrieve_database_secrets()
            
            self.initialized = True
            logger.info("✅ Azure Database Configuration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure Database Configuration: {e}")
            logger.info("🔄 Falling back to environment variables...")
            return await self._initialize_from_env()
    
    async def _retrieve_database_secrets(self):
        """Retrieve database connection secrets from Azure Key Vault"""
        try:
            # Define the secrets we need (matching your Node.js backend)
            secret_names = {
                'host': 'PostgresHost',
                'database': 'PostgresDB', 
                'user': 'PostgresDBUser',
                'password': 'PostgresDBPassword',
                # Also get the user database credentials
                'user_host': 'PostgresCrenoventUserHost',
                'user_database': 'PostgresCrenoventUserDB',
                'user_user': 'PostgresCrenoventUser', 
                'user_password': 'PostgresCrenoventUserDBPassword'
            }
            
            # Retrieve secrets
            secrets = {}
            for key, secret_name in secret_names.items():
                try:
                    secret = self.secret_client.get_secret(secret_name)
                    secrets[key] = secret.value
                    logger.debug(f"✅ Retrieved secret: {secret_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not retrieve secret {secret_name}: {e}")
            
            # Configure primary database connection (main data)
            if all(k in secrets for k in ['host', 'database', 'user', 'password']):
                self.connection_config['primary'] = {
                    'host': secrets['host'],
                    'database': secrets['database'],
                    'user': secrets['user'],
                    'password': secrets['password'],
                    'port': 5432,
                    'ssl': 'require'
                }
                logger.info("✅ Primary database configuration loaded from Key Vault")
            
            # Configure user database connection (user management)
            if all(k in secrets for k in ['user_host', 'user_database', 'user_user', 'user_password']):
                self.connection_config['user'] = {
                    'host': secrets['user_host'],
                    'database': secrets['user_database'],
                    'user': secrets['user_user'],
                    'password': secrets['user_password'],
                    'port': 5432,
                    'ssl': 'require'
                }
                logger.info("✅ User database configuration loaded from Key Vault")
            
            # If we don't have complete configs, fall back to environment
            if not self.connection_config:
                logger.warning("⚠️ No complete database configurations found in Key Vault")
                await self._initialize_from_env()
                
        except Exception as e:
            logger.error(f"❌ Error retrieving database secrets: {e}")
            await self._initialize_from_env()
    
    async def _initialize_from_env(self):
        """Initialize database configuration from environment variables"""
        try:
            logger.info("🔧 Initializing database configuration from environment variables...")
            
            # Primary database configuration
            primary_config = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'database': os.getenv('POSTGRES_DB', 'crenovent'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', ''),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'ssl': os.getenv('POSTGRES_SSL', 'prefer')
            }
            
            # User database configuration
            user_config = {
                'host': os.getenv('POSTGRES_CRENOVENT_USER_HOST', primary_config['host']),
                'database': os.getenv('POSTGRES_CRENOVENT_USER_DB', 'crenovent_users'),
                'user': os.getenv('POSTGRES_CRENOVENT_USER', primary_config['user']),
                'password': os.getenv('POSTGRES_CRENOVENT_USER_PASSWORD', primary_config['password']),
                'port': int(os.getenv('POSTGRES_CRENOVENT_USER_PORT', '5432')),
                'ssl': os.getenv('POSTGRES_CRENOVENT_USER_SSL', 'prefer')
            }
            
            # Validate configurations
            if primary_config['password']:
                self.connection_config['primary'] = primary_config
                logger.info("✅ Primary database configuration loaded from environment")
            
            if user_config['password']:
                self.connection_config['user'] = user_config
                logger.info("✅ User database configuration loaded from environment")
            
            # If still no config, use defaults for development
            if not self.connection_config:
                logger.warning("⚠️ No database configuration found, using development defaults")
                self.connection_config['primary'] = {
                    'host': 'localhost',
                    'database': 'crenovent',
                    'user': 'postgres', 
                    'password': 'password',
                    'port': 5432,
                    'ssl': 'prefer'
                }
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing from environment: {e}")
            return False
    
    def get_connection_string(self, db_type: str = 'primary') -> str:
        """Get PostgreSQL connection string for specified database type"""
        if not self.initialized:
            raise RuntimeError("Database configuration not initialized")
        
        config = self.connection_config.get(db_type)
        if not config:
            raise ValueError(f"Database configuration '{db_type}' not found")
        
        return (
            f"postgresql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
            f"?sslmode={config['ssl']}"
        )
    
    def get_asyncpg_config(self, db_type: str = 'primary') -> Dict[str, Any]:
        """Get asyncpg connection configuration"""
        if not self.initialized:
            raise RuntimeError("Database configuration not initialized")
        
        config = self.connection_config.get(db_type)
        if not config:
            raise ValueError(f"Database configuration '{db_type}' not found")
        
        return {
            'host': config['host'],
            'port': config['port'],
            'user': config['user'],
            'password': config['password'],
            'database': config['database'],
            'ssl': config['ssl']
        }
    
    async def test_connection(self, db_type: str = 'primary') -> bool:
        """Test database connection"""
        try:
            config = self.get_asyncpg_config(db_type)
            
            # Test connection
            conn = await asyncpg.connect(**config)
            
            # Test query
            result = await conn.fetchval('SELECT 1')
            
            await conn.close()
            
            if result == 1:
                logger.info(f"✅ Database connection test successful for '{db_type}'")
                return True
            else:
                logger.error(f"❌ Database connection test failed for '{db_type}'")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database connection test error for '{db_type}': {e}")
            return False
    
    async def ensure_intelligence_schema(self, db_type: str = 'primary'):
        """Ensure intelligence system database schema exists"""
        try:
            logger.info(f"🔧 Ensuring intelligence schema exists in '{db_type}' database...")
            
            config = self.get_asyncpg_config(db_type)
            conn = await asyncpg.connect(**config)
            
            # Create intelligence schema if it doesn't exist
            await conn.execute("CREATE SCHEMA IF NOT EXISTS intelligence")
            
            # Create tables for intelligence system
            # Trust scores table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS intelligence.capability_trust_scores (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    capability_id TEXT NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    overall_score DECIMAL(3,2) NOT NULL,
                    trust_level TEXT NOT NULL,
                    execution_score DECIMAL(3,2),
                    performance_score DECIMAL(3,2),
                    compliance_score DECIMAL(3,2),
                    business_impact_score DECIMAL(3,2),
                    factors JSONB,
                    sample_size INTEGER,
                    confidence_interval JSONB,
                    recommendations TEXT[],
                    calculated_at TIMESTAMPTZ DEFAULT NOW(),
                    valid_until TIMESTAMPTZ,
                    
                    CONSTRAINT trust_score_check CHECK (overall_score >= 0.0 AND overall_score <= 1.0),
                    CONSTRAINT trust_level_check CHECK (trust_level IN ('critical', 'high', 'medium', 'low', 'untrusted'))
                )
            """)
            
            # SLA reports table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS intelligence.capability_sla_reports (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    capability_id TEXT NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    sla_tier TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    availability_percentage DECIMAL(5,2),
                    average_latency_ms DECIMAL(10,2),
                    error_rate_percentage DECIMAL(5,2),
                    sla_compliance_percentage DECIMAL(5,2),
                    breaches_count INTEGER,
                    total_breach_duration_minutes INTEGER,
                    reporting_period_start TIMESTAMPTZ,
                    reporting_period_end TIMESTAMPTZ,
                    measurements JSONB,
                    estimated_business_impact JSONB,
                    cost_attribution JSONB,
                    recommended_actions TEXT[],
                    escalations_triggered TEXT[],
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    CONSTRAINT sla_tier_check CHECK (sla_tier IN ('T0', 'T1', 'T2')),
                    CONSTRAINT sla_status_check CHECK (overall_status IN ('meeting', 'at_risk', 'breached', 'recovering'))
                )
            """)
            
            # SLA alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS intelligence.capability_sla_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    alert_id TEXT NOT NULL UNIQUE,
                    capability_id TEXT NOT NULL,
                    tenant_id INTEGER NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    measured_value DECIMAL(10,2),
                    target_value DECIMAL(10,2),
                    status TEXT NOT NULL,
                    message TEXT,
                    recommended_actions TEXT[],
                    triggered_at TIMESTAMPTZ DEFAULT NOW(),
                    acknowledged_at TIMESTAMPTZ,
                    resolved_at TIMESTAMPTZ,
                    
                    CONSTRAINT alert_type_check CHECK (alert_type IN ('sla_breach', 'sla_at_risk')),
                    CONSTRAINT severity_check CHECK (severity IN ('critical', 'warning', 'info'))
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trust_scores_capability_tenant 
                ON intelligence.capability_trust_scores(capability_id, tenant_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sla_reports_capability_tenant 
                ON intelligence.capability_sla_reports(capability_id, tenant_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sla_alerts_triggered 
                ON intelligence.capability_sla_alerts(triggered_at DESC)
            """)
            
            await conn.close()
            
            logger.info(f"✅ Intelligence schema ensured in '{db_type}' database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error ensuring intelligence schema: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database configuration information (without sensitive data)"""
        info = {
            'initialized': self.initialized,
            'configured_databases': list(self.connection_config.keys()),
            'azure_keyvault_enabled': self.secret_client is not None
        }
        
        for db_type, config in self.connection_config.items():
            info[f'{db_type}_database'] = {
                'host': config['host'],
                'database': config['database'],
                'user': config['user'],
                'port': config['port'],
                'ssl': config['ssl']
            }
        
        return info

# Global instance
azure_db_config = AzureDatabaseConfig()

async def initialize_azure_database():
    """Initialize the Azure database configuration"""
    return await azure_db_config.initialize()

def get_database_config():
    """Get the initialized database configuration"""
    return azure_db_config
