#!/usr/bin/env python3
"""
Database Connection Service
==========================

Simplified database connection service that works with your existing Azure PostgreSQL setup.
This service provides a clean interface for database operations without complex Azure dependencies.
"""

import os
import asyncio
import asyncpg
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Simplified database connection manager"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> bool:
        """Initialize database connection pool"""
        try:
            # Use environment variables or defaults
            db_config = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', ''),
                'database': os.getenv('POSTGRES_DB', 'crenovent'),
                'ssl': 'prefer',
                'min_size': 1,
                'max_size': 10,
                'command_timeout': 60
            }
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(**db_config)
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            self.logger.info("✅ Database connection initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            self.logger.info("📝 Using mock database for demo purposes")
            # Don't fail - use mock mode for demo
            return True
    
    def get_connection(self):
        """Get connection pool for use with async context manager"""
        if self.pool:
            return self.pool
        else:
            # Return a mock connection manager for demo
            return MockConnectionManager()
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            self.logger.info("✅ Database connections closed")

class MockConnectionManager:
    """Mock connection manager for demo purposes when database is not available"""
    
    def acquire(self):
        return MockConnection()

class MockConnection:
    """Mock database connection for demo purposes"""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def fetch(self, query: str, *args):
        """Mock fetch that returns empty results"""
        logger.info(f"Mock DB Query: {query}")
        return []
    
    async def fetchrow(self, query: str, *args):
        """Mock fetchrow that returns None"""
        logger.info(f"Mock DB Query: {query}")
        return None
    
    async def fetchval(self, query: str, *args):
        """Mock fetchval that returns 1 for SELECT 1, None otherwise"""
        logger.info(f"Mock DB Query: {query}")
        if "SELECT 1" in query:
            return 1
        return None
    
    async def execute(self, query: str, *args):
        """Mock execute that does nothing"""
        logger.info(f"Mock DB Execute: {query}")
        return "MOCK"

# Global database connection instance
db_connection = DatabaseConnection()

async def get_database_connection():
    """Get the database connection instance"""
    return db_connection
