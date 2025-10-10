"""
Centralized Connection Pool Manager
Manages all database connections efficiently with proper pooling
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
import asyncpg
from openai import AsyncAzureOpenAI
from datetime import datetime, timedelta
from src.services.azure_database_config import get_database_config, initialize_azure_database

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class ConnectionPoolManager:
    """
    Singleton connection pool manager for efficient resource management
    - Single PostgreSQL connection pool shared across all services
    - Single Azure OpenAI client instance
    - Connection monitoring and health checks
    - Automatic reconnection on failures
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectionPoolManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.postgres_pool: Optional[asyncpg.Pool] = None
            self.openai_client: Optional[AsyncAzureOpenAI] = None
            self.fabric_service = None
            self.logger = logging.getLogger(__name__)
            
            # Enhanced pool configuration optimized for Azure PostgreSQL
            self.pool_config = {
                'min_size': int(os.getenv('POSTGRES_POOL_MIN_SIZE', '1')),  # Reduced to prevent Azure limits
                'max_size': int(os.getenv('POSTGRES_POOL_MAX_SIZE', '4')),   # Reduced for Azure compatibility
                'command_timeout': int(os.getenv('POSTGRES_COMMAND_TIMEOUT', '120')),  # Longer timeout for Azure
                'ssl': 'require',  # Required for Azure PostgreSQL
                'max_queries': 1000,  # Reduced to prevent connection exhaustion
                'max_inactive_connection_lifetime': 600,  # 10 minutes - longer for Azure
                'setup': self._setup_connection,  # Connection setup callback
                'init': self._init_connection,  # Connection initialization callback
                # Azure-specific optimizations
                'server_settings': {
                    'application_name': 'crenovent_ai_service',
                    'tcp_keepalives_idle': '600',  # 10 minutes
                    'tcp_keepalives_interval': '30',  # 30 seconds
                    'tcp_keepalives_count': '3'  # 3 retries
                }
            }
            
            # Circuit breaker state
            self.circuit_breaker = {
                'state': 'closed',  # closed, open, half-open
                'failure_count': 0,
                'failure_threshold': 5,
                'recovery_timeout': 60,
                'last_failure_time': None
            }
            
            # Performance metrics
            self.metrics = {
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0,
                'avg_response_time': 0.0,
                'last_health_check': None,
                'connection_errors': 0
            }
            
            # Server settings
            self.server_settings = {
                'application_name': 'crenovent_ai_service_pool',
                'search_path': 'public'
            }
            
            # Health monitoring
            self.pool_stats = {
                'total_connections': 0,
                'active_connections': 0,
                'idle_connections': 0,
                'last_health_check': None,
                'connection_errors': 0,
                'total_queries': 0,
                'avg_query_time': 0.0
            }
            
            ConnectionPoolManager._initialized = True
    
    async def _setup_connection(self, connection):
        """Setup callback for new connections - Azure optimized"""
        try:
            # Azure PostgreSQL optimized settings
            await connection.execute("SET application_name = 'crenovent_ai_service'")
            await connection.execute("SET search_path = public")
            await connection.execute("SET timezone = 'UTC'")
            # Longer timeouts for Azure
            await connection.execute("SET statement_timeout = '600s'")  # 10 minutes
            await connection.execute("SET idle_in_transaction_session_timeout = '300s'")  # 5 minutes
            # Azure-specific connection settings
            await connection.execute("SET tcp_keepalives_idle = 600")  # 10 minutes
            await connection.execute("SET tcp_keepalives_interval = 30")  # 30 seconds
            await connection.execute("SET tcp_keepalives_count = 3")  # 3 retries
            self.logger.debug("Azure-optimized connection setup completed")
        except Exception as e:
            self.logger.warning(f"Connection setup warning (non-critical): {e}")
    
    async def _init_connection(self, connection):
        """Initialize callback for connections - minimal testing"""
        try:
            # Minimal connection test to avoid interference
            await connection.fetchval("SELECT 1")
            self.logger.debug("Connection validated")
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            raise
    
    async def initialize(self) -> bool:
        """Initialize all connection pools and clients"""
        try:
            # Initialize Azure database configuration first
            await initialize_azure_database()
            
            await self._initialize_postgres_pool()
            await self._initialize_openai_client()
            await self._initialize_fabric_service()
            self.logger.info(" All services initialized - PostgreSQL, OpenAI, and Fabric working!")
            
            # Start health monitoring (now passive)
            asyncio.create_task(self._health_monitor())
            
            self.logger.info(" Connection Pool Manager initialized successfully")
            self.logger.info(f" PostgreSQL Pool: {self.pool_config['min_size']}-{self.pool_config['max_size']} connections")
            
            return True
            
        except Exception as e:
            self.logger.error(f" Connection Pool Manager initialization failed: {e}")
            return False
    
    async def reset_pool_if_closed(self):
        """Reset the connection pool if it's closed - Azure PostgreSQL fix"""
        try:
            if not self.postgres_pool or self.postgres_pool.is_closing():
                self.logger.warning(" Connection pool is closed, reinitializing...")
                await self._initialize_postgres_pool()
                self.logger.info(" Connection pool reinitialized successfully")
                return True
            return False
        except Exception as e:
            self.logger.error(f" Failed to reset connection pool: {e}")
            return False
    
    async def _initialize_postgres_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            # Get Azure database configuration
            azure_db_config = get_database_config()
            
            # Try to use Azure database configuration first
            if azure_db_config.initialized:
                try:
                    # Use primary database configuration
                    db_config = azure_db_config.get_asyncpg_config('primary')
                    
                    # Merge configs carefully to avoid SSL conflicts
                    merged_config = {
                        'host': db_config['host'],
                        'port': db_config['port'],
                        'user': db_config['user'],
                        'password': db_config['password'],
                        'database': db_config['database'],
                        'ssl': db_config['ssl'],
                        'min_size': self.pool_config['min_size'],
                        'max_size': self.pool_config['max_size'],
                        'command_timeout': self.pool_config['command_timeout'],
                        'server_settings': self.pool_config['server_settings']
                    }
                    
                    self.postgres_pool = await asyncpg.create_pool(**merged_config)
                    self.logger.info(" Using Azure database configuration")
                except Exception as e:
                    self.logger.warning(f" Azure database config failed, falling back to environment: {e}")
                    await self._initialize_postgres_pool_from_env()
                    return
            else:
                # Fall back to environment variables
                await self._initialize_postgres_pool_from_env()
                return
            
            # Test the pool
            async with self.postgres_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                # Test pgvector extension (optional, don't fail if not available)
                try:
                    await conn.fetchval("SELECT '[1,2,3]'::vector")
                    self.logger.info(" pgvector extension available")
                except Exception as e:
                    self.logger.warning(f" pgvector extension not available: {e}")
            
            # Ensure intelligence schema exists
            if azure_db_config.initialized:
                await azure_db_config.ensure_intelligence_schema('primary')
            
            self.logger.info(" PostgreSQL connection pool initialized")
            
        except Exception as e:
            self.logger.error(f" PostgreSQL pool initialization failed: {e}")
            raise
    
    async def _initialize_postgres_pool_from_env(self):
        """Initialize PostgreSQL pool from environment variables (fallback)"""
        # Use DATABASE_URL if available, otherwise individual params
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            self.postgres_pool = await asyncpg.create_pool(
                database_url,
                **self.pool_config
            )
        else:
            self.postgres_pool = await asyncpg.create_pool(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                user=os.getenv('POSTGRES_USER'),
                password=os.getenv('POSTGRES_PASSWORD'),
                database=os.getenv('POSTGRES_DB'),  # Changed from POSTGRES_DATABASE
                **self.pool_config
            )
        self.logger.info(" Using environment variable database configuration")
    
    async def _initialize_openai_client(self):
        """Initialize Azure OpenAI client"""
        try:
            if self.openai_client:
                await self.openai_client.close()
            
            self.openai_client = AsyncAzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
            )
            
            # Test the client
            test_response = await self.openai_client.embeddings.create(
                model=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small'),
                input=["connection test"]
            )
            
            self.logger.info(" Azure OpenAI client initialized")
            
        except Exception as e:
            self.logger.error(f" Azure OpenAI client initialization failed: {e}")
            raise
    
    async def _initialize_fabric_service(self):
        """Initialize Fabric service (optional)"""
        try:
            from src.services.fabric_service import FabricService
            
            if self.fabric_service:
                await self.fabric_service.close()
            
            self.fabric_service = FabricService()
            await self.fabric_service.initialize()
            
            self.logger.info(" Fabric service initialized")
            
        except Exception as e:
            self.logger.error(f" FABRIC SERVICE INITIALIZATION FAILED: {e}")
            self.logger.error(" This means workflows will use MOCK DATA instead of real Salesforce data!")
            import traceback
            self.logger.error(f" Full error traceback: {traceback.format_exc()}")
            # Don't raise - Fabric is optional, system can work without it
            self.fabric_service = None
    
    def get_connection(self):
        """Get connection pool for use with 'async with' context manager"""
        if not self.postgres_pool:
            raise RuntimeError("PostgreSQL pool not initialized")
        return self.postgres_pool
    
    async def get_postgres_connection(self) -> asyncpg.Connection:
        """Get PostgreSQL connection from pool with automatic recovery"""
        # Check if pool needs reset
        await self.reset_pool_if_closed()
        
        if not self.postgres_pool:
            # Try to reinitialize the pool
            self.logger.warning(" PostgreSQL pool not available, attempting to reinitialize...")
            success = await self.initialize()
            if not success:
                raise RuntimeError("PostgreSQL pool not initialized and reinitialize failed")
        
        try:
            connection = await self.postgres_pool.acquire()
            self.pool_stats['total_queries'] += 1
            return connection
            
        except Exception as e:
            self.pool_stats['connection_errors'] += 1
            self.logger.error(f" Failed to acquire PostgreSQL connection: {e}")
            
            # If pool is closed, try to reset and reinitialize
            if "pool is closed" in str(e).lower() or "pool is closing" in str(e).lower():
                self.logger.warning(" Pool is closed/closing, attempting to reset...")
                try:
                    await self.reset_pool_if_closed()
                    if self.postgres_pool:
                        connection = await self.postgres_pool.acquire()
                        self.pool_stats['total_queries'] += 1
                        return connection
                except Exception as reinit_error:
                    self.logger.error(f" Pool reset failed: {reinit_error}")
            
            raise
    
    def release_postgres_connection(self, connection: asyncpg.Connection):
        """Release PostgreSQL connection back to pool"""
        try:
            self.postgres_pool.release(connection)
        except Exception as e:
            self.logger.error(f" Failed to release PostgreSQL connection: {e}")
    
    async def execute_postgres_query(self, query: str, *args) -> Any:
        """Execute PostgreSQL query with automatic connection management and recovery"""
        start_time = datetime.now()
        
        # Check if pool needs reset
        await self.reset_pool_if_closed()
        
        # Ensure pool is available
        if not self.postgres_pool:
            await self.initialize()
        
        try:
            async with self.postgres_pool.acquire() as conn:
                try:
                    result = await conn.fetch(query, *args)
                    
                    # Update stats
                    query_time = (datetime.now() - start_time).total_seconds()
                    self.pool_stats['total_queries'] += 1
                    
                    # Update average query time (rolling average)
                    current_avg = self.pool_stats['avg_query_time']
                    total_queries = self.pool_stats['total_queries']
                    self.pool_stats['avg_query_time'] = (
                        (current_avg * (total_queries - 1) + query_time) / total_queries
                    )
                    
                    return result
                    
                except Exception as e:
                    self.pool_stats['connection_errors'] += 1
                    self.logger.error(f" PostgreSQL query failed: {e}")
                    raise
        except Exception as e:
            # If pool is closed, try to recover
            if "pool is closed" in str(e).lower() or "pool is closing" in str(e).lower():
                self.logger.warning(" Pool closed during query, attempting recovery...")
                await self.reset_pool_if_closed()
                if self.postgres_pool:
                    # Retry once
                    async with self.postgres_pool.acquire() as conn:
                        return await conn.fetch(query, *args)
            raise
    
    async def execute_postgres_fetchrow(self, query: str, *args) -> Any:
        """Execute PostgreSQL query and return single row"""
        async with self.postgres_pool.acquire() as conn:
            try:
                result = await conn.fetchrow(query, *args)
                self.pool_stats['total_queries'] += 1
                return result
            except Exception as e:
                self.pool_stats['connection_errors'] += 1
                self.logger.error(f" PostgreSQL fetchrow failed: {e}")
                raise
    
    async def execute_postgres_fetchval(self, query: str, *args) -> Any:
        """Execute PostgreSQL query and return single value"""
        async with self.postgres_pool.acquire() as conn:
            try:
                result = await conn.fetchval(query, *args)
                self.pool_stats['total_queries'] += 1
                return result
            except Exception as e:
                self.pool_stats['connection_errors'] += 1
                self.logger.error(f" PostgreSQL fetchval failed: {e}")
                raise
    
    async def create_openai_embedding(self, text: str, model: str = None) -> list:
        """Create embedding using shared OpenAI client"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            embedding_model = model or os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small')
            
            response = await self.openai_client.embeddings.create(
                model=embedding_model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f" OpenAI embedding creation failed: {e}")
            raise
    
    async def chat_completion(self, messages: list, model: str = None, **kwargs) -> Any:
        """Create chat completion using shared OpenAI client"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            chat_model = model or os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'gpt-4o-mini')
            
            response = await self.openai_client.chat.completions.create(
                model=chat_model,
                messages=messages,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f" OpenAI chat completion failed: {e}")
            raise
    
    def get_fabric_service(self):
        """Get shared Fabric service instance"""
        return self.fabric_service
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics"""
        if self.postgres_pool:
            try:
                # Safely get pool stats - different asyncpg versions have different attributes
                total_conns = getattr(self.postgres_pool, '_holders', [])
                total_connections = len(total_conns) if total_conns else 0
                
                # Try different attribute names for asyncpg compatibility
                active_connections = (
                    getattr(self.postgres_pool, '_working_count', None) or
                    getattr(self.postgres_pool, '_working', None) or
                    getattr(self.postgres_pool, 'get_size', lambda: 0)()
                )
                
                idle_connections = (
                    getattr(self.postgres_pool, '_free_count', None) or
                    getattr(self.postgres_pool, '_queue_size', None) or
                    0
                )
                
                self.pool_stats.update({
                    'total_connections': total_connections,
                    'active_connections': active_connections if isinstance(active_connections, int) else 0,
                    'idle_connections': idle_connections if isinstance(idle_connections, int) else 0,
                    'last_health_check': datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Could not get detailed pool stats: {e}")
                self.pool_stats.update({
                    'total_connections': 'available',
                    'active_connections': 'available',
                    'idle_connections': 'available',
                    'last_health_check': datetime.now().isoformat()
                })
        
        return self.pool_stats.copy()
    
    async def _health_monitor(self):
        """Background health monitoring task - DISABLED to prevent pool closure"""
        # CRITICAL: Health monitor disabled due to Azure PostgreSQL connection issues
        # The health monitor was causing connection pool closures during active use
        self.logger.info(" Health monitor disabled to prevent Azure PostgreSQL connection issues")
        
        # Instead of continuous monitoring, we'll do passive monitoring
        while True:
            try:
                # Much longer sleep to reduce interference
                await asyncio.sleep(300)  # Check every 5 minutes instead of 1 minute
                
                # Only update stats, don't test connections during active use
                self.pool_stats['last_health_check'] = datetime.now()
                
                # Log stats every 30 minutes instead of 10
                if datetime.now().minute % 30 == 0:
                    try:
                        stats = await self.get_pool_stats()
                        self.logger.info(f" Pool Stats: Connections available, "
                                       f"{stats['total_queries']} queries processed")
                    except Exception as stats_error:
                        self.logger.debug(f"Stats collection skipped: {stats_error}")
                
            except Exception as e:
                self.logger.debug(f"Health monitor error (non-critical): {e}")
                # Don't increment connection_errors for health monitor issues
    
    async def close(self):
        """Close all connections and cleanup"""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
                self.logger.info(" PostgreSQL pool closed")
            
            if self.openai_client:
                await self.openai_client.close()
                self.logger.info(" OpenAI client closed")
            
            if self.fabric_service:
                await self.fabric_service.close()
                self.logger.info(" Fabric service closed")
            
        except Exception as e:
            self.logger.error(f" Error during cleanup: {e}")

# Global instance
pool_manager = ConnectionPoolManager()
