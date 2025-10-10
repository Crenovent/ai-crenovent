"""
RBA Connector SDK Core - Task 10.1-T03
Implements HTTP client, auth, retries, backoff, and idempotency for external system integration.
Provides reusable base for CRM, Billing, CLM, and ERP connectors with governance-first design.
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import backoff
import jwt
from urllib.parse import urljoin, urlencode
import logging

logger = logging.getLogger(__name__)

class AuthType(Enum):
    """Supported authentication types"""
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    HMAC = "hmac"
    BASIC = "basic"
    BEARER = "bearer"

class ErrorType(Enum):
    """Error classification for retry logic"""
    TRANSIENT = "transient"          # Network issues, 5xx errors
    PERMANENT = "permanent"          # 4xx errors, auth failures
    POLICY = "policy"               # Governance violations
    DATA = "data"                   # Data validation errors
    RATE_LIMIT = "rate_limit"       # Rate limiting

@dataclass
class ConnectorConfig:
    """Connector configuration with governance metadata"""
    name: str
    base_url: str
    auth_type: AuthType
    tenant_id: str
    region: str = "us-east-1"
    
    # Authentication credentials
    auth_config: Dict[str, Any] = field(default_factory=dict)
    
    # Rate limiting
    rate_limit_per_second: int = 10
    rate_limit_per_minute: int = 600
    rate_limit_per_hour: int = 10000
    
    # Retry configuration
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    
    # Timeouts
    connect_timeout: int = 30
    read_timeout: int = 60
    total_timeout: int = 300
    
    # Governance
    policy_id: str = ""
    compliance_tags: List[str] = field(default_factory=list)
    evidence_capture: bool = True
    
    # Headers
    default_headers: Dict[str, str] = field(default_factory=dict)
    user_agent: str = "RBA-Connector-SDK/1.0"

@dataclass
class RequestContext:
    """Request context with tenant isolation and governance"""
    tenant_id: str
    user_id: str
    execution_id: str
    idempotency_key: Optional[str] = None
    policy_id: Optional[str] = None
    evidence_capture: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConnectorResponse:
    """Standardized connector response"""
    success: bool
    status_code: int
    data: Any
    headers: Dict[str, str]
    error_message: Optional[str] = None
    error_type: Optional[ErrorType] = None
    response_time_ms: int = 0
    idempotency_key: Optional[str] = None
    evidence_id: Optional[str] = None

class RateLimiter:
    """Token bucket rate limiter with tenant isolation"""
    
    def __init__(self, per_second: int = 10, per_minute: int = 600, per_hour: int = 10000):
        self.per_second = per_second
        self.per_minute = per_minute
        self.per_hour = per_hour
        
        # Token buckets per tenant
        self.buckets: Dict[str, Dict[str, Dict]] = {}
        
    async def acquire(self, tenant_id: str, tokens: int = 1) -> bool:
        """
        Acquire tokens from rate limiter
        
        Args:
            tenant_id: Tenant ID for isolation
            tokens: Number of tokens to acquire
            
        Returns:
            bool: True if tokens acquired, False if rate limited
        """
        now = time.time()
        
        if tenant_id not in self.buckets:
            self.buckets[tenant_id] = {
                "second": {"tokens": self.per_second, "last_refill": now},
                "minute": {"tokens": self.per_minute, "last_refill": now},
                "hour": {"tokens": self.per_hour, "last_refill": now}
            }
        
        tenant_buckets = self.buckets[tenant_id]
        
        # Refill buckets based on time elapsed
        self._refill_bucket(tenant_buckets["second"], self.per_second, now, 1.0)
        self._refill_bucket(tenant_buckets["minute"], self.per_minute, now, 60.0)
        self._refill_bucket(tenant_buckets["hour"], self.per_hour, now, 3600.0)
        
        # Check if we can acquire tokens from all buckets
        for bucket in tenant_buckets.values():
            if bucket["tokens"] < tokens:
                return False
        
        # Acquire tokens from all buckets
        for bucket in tenant_buckets.values():
            bucket["tokens"] -= tokens
        
        return True
    
    def _refill_bucket(self, bucket: Dict, capacity: int, now: float, interval: float):
        """Refill token bucket based on elapsed time"""
        elapsed = now - bucket["last_refill"]
        tokens_to_add = int(elapsed / interval * capacity)
        
        if tokens_to_add > 0:
            bucket["tokens"] = min(capacity, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now

class AuthManager:
    """Authentication manager supporting multiple auth types"""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.auth_cache: Dict[str, Dict] = {}
    
    async def get_auth_headers(self, context: RequestContext) -> Dict[str, str]:
        """
        Get authentication headers for request
        
        Args:
            context: Request context
            
        Returns:
            Dict: Authentication headers
        """
        if self.config.auth_type == AuthType.OAUTH2:
            return await self._get_oauth2_headers(context)
        elif self.config.auth_type == AuthType.API_KEY:
            return self._get_api_key_headers()
        elif self.config.auth_type == AuthType.HMAC:
            return self._get_hmac_headers(context)
        elif self.config.auth_type == AuthType.BASIC:
            return self._get_basic_auth_headers()
        elif self.config.auth_type == AuthType.BEARER:
            return self._get_bearer_headers()
        else:
            return {}
    
    async def _get_oauth2_headers(self, context: RequestContext) -> Dict[str, str]:
        """Get OAuth2 access token headers"""
        cache_key = f"{context.tenant_id}:oauth2"
        
        # Check cache for valid token
        if cache_key in self.auth_cache:
            cached = self.auth_cache[cache_key]
            if cached["expires_at"] > datetime.now(timezone.utc):
                return {"Authorization": f"Bearer {cached['access_token']}"}
        
        # Get new access token
        token_data = await self._fetch_oauth2_token()
        
        # Cache token
        self.auth_cache[cache_key] = {
            "access_token": token_data["access_token"],
            "expires_at": datetime.now(timezone.utc) + timedelta(seconds=token_data.get("expires_in", 3600))
        }
        
        return {"Authorization": f"Bearer {token_data['access_token']}"}
    
    async def _fetch_oauth2_token(self) -> Dict[str, Any]:
        """Fetch OAuth2 access token"""
        token_url = self.config.auth_config["token_url"]
        client_id = self.config.auth_config["client_id"]
        client_secret = self.config.auth_config["client_secret"]
        scope = self.config.auth_config.get("scope", "")
        
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"OAuth2 token fetch failed: {response.status}")
    
    def _get_api_key_headers(self) -> Dict[str, str]:
        """Get API key headers"""
        api_key = self.config.auth_config["api_key"]
        header_name = self.config.auth_config.get("header_name", "X-API-Key")
        
        return {header_name: api_key}
    
    def _get_hmac_headers(self, context: RequestContext) -> Dict[str, str]:
        """Get HMAC signature headers"""
        secret = self.config.auth_config["secret"]
        timestamp = str(int(time.time()))
        
        # Create signature payload
        payload = f"{context.execution_id}:{timestamp}:{context.tenant_id}"
        signature = hashlib.hmac.new(
            secret.encode(), 
            payload.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-Timestamp": timestamp,
            "X-Signature": signature,
            "X-Execution-ID": context.execution_id
        }
    
    def _get_basic_auth_headers(self) -> Dict[str, str]:
        """Get Basic authentication headers"""
        import base64
        
        username = self.config.auth_config["username"]
        password = self.config.auth_config["password"]
        
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {credentials}"}
    
    def _get_bearer_headers(self) -> Dict[str, str]:
        """Get Bearer token headers"""
        token = self.config.auth_config["token"]
        return {"Authorization": f"Bearer {token}"}

class ConnectorSDK:
    """
    Core Connector SDK implementing Task 10.1-T03
    Provides HTTP client with auth, retries, backoff, and governance
    """
    
    def __init__(self, config: ConnectorConfig):
        """
        Initialize Connector SDK
        
        Args:
            config: Connector configuration
        """
        self.config = config
        self.auth_manager = AuthManager(config)
        self.rate_limiter = RateLimiter(
            config.rate_limit_per_second,
            config.rate_limit_per_minute,
            config.rate_limit_per_hour
        )
        
        # Session with connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Metrics tracking
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "rate_limited": 0,
            "retries_total": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize HTTP session"""
        timeout = aiohttp.ClientTimeout(
            connect=self.config.connect_timeout,
            sock_read=self.config.read_timeout,
            total=self.config.total_timeout
        )
        
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "User-Agent": self.config.user_agent,
                **self.config.default_headers
            }
        )
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def request(self, method: str, endpoint: str, context: RequestContext,
                     data: Optional[Dict] = None, params: Optional[Dict] = None,
                     headers: Optional[Dict] = None) -> ConnectorResponse:
        """
        Make HTTP request with governance, retries, and rate limiting
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            context: Request context with tenant isolation
            data: Request body data
            params: Query parameters
            headers: Additional headers
            
        Returns:
            ConnectorResponse: Standardized response
        """
        start_time = time.time()
        
        # Validate tenant isolation
        if context.tenant_id != self.config.tenant_id:
            raise ValueError("Tenant mismatch in request context")
        
        # Generate idempotency key if not provided
        if not context.idempotency_key:
            context.idempotency_key = self._generate_idempotency_key(
                method, endpoint, data, params, context
            )
        
        # Check rate limiting
        if not await self.rate_limiter.acquire(context.tenant_id):
            self.metrics["rate_limited"] += 1
            return ConnectorResponse(
                success=False,
                status_code=429,
                data=None,
                headers={},
                error_message="Rate limit exceeded",
                error_type=ErrorType.RATE_LIMIT,
                idempotency_key=context.idempotency_key
            )
        
        # Execute request with retries
        response = await self._execute_with_retries(
            method, endpoint, context, data, params, headers
        )
        
        # Calculate response time
        response.response_time_ms = int((time.time() - start_time) * 1000)
        
        # Update metrics
        self.metrics["requests_total"] += 1
        if response.success:
            self.metrics["requests_success"] += 1
        else:
            self.metrics["requests_failed"] += 1
        
        # Generate evidence if required
        if context.evidence_capture:
            response.evidence_id = await self._generate_evidence(
                method, endpoint, context, response
            )
        
        return response
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        base=1.0,
        max_value=60.0
    )
    async def _execute_with_retries(self, method: str, endpoint: str, 
                                   context: RequestContext,
                                   data: Optional[Dict] = None,
                                   params: Optional[Dict] = None,
                                   headers: Optional[Dict] = None) -> ConnectorResponse:
        """Execute HTTP request with exponential backoff retries"""
        
        # Build full URL
        url = urljoin(self.config.base_url, endpoint)
        
        # Prepare headers
        request_headers = {}
        request_headers.update(await self.auth_manager.get_auth_headers(context))
        
        # Add governance headers
        request_headers.update({
            "X-Tenant-ID": context.tenant_id,
            "X-Execution-ID": context.execution_id,
            "X-Policy-ID": context.policy_id or self.config.policy_id,
            "X-Idempotency-Key": context.idempotency_key
        })
        
        if headers:
            request_headers.update(headers)
        
        # Prepare request data
        json_data = None
        if data and method.upper() in ["POST", "PUT", "PATCH"]:
            json_data = data
            request_headers["Content-Type"] = "application/json"
        
        try:
            async with self.session.request(
                method=method.upper(),
                url=url,
                json=json_data,
                params=params,
                headers=request_headers
            ) as response:
                
                # Read response data
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()
                
                # Classify error type
                error_type = None
                error_message = None
                
                if response.status >= 400:
                    error_type = self._classify_error(response.status)
                    error_message = f"HTTP {response.status}: {response.reason}"
                
                return ConnectorResponse(
                    success=response.status < 400,
                    status_code=response.status,
                    data=response_data,
                    headers=dict(response.headers),
                    error_message=error_message,
                    error_type=error_type,
                    idempotency_key=context.idempotency_key
                )
                
        except asyncio.TimeoutError:
            self.metrics["retries_total"] += 1
            raise
        except aiohttp.ClientError as e:
            self.metrics["retries_total"] += 1
            logger.error(f"Request failed: {e}")
            raise
    
    def _classify_error(self, status_code: int) -> ErrorType:
        """Classify HTTP error for retry logic"""
        if status_code == 429:
            return ErrorType.RATE_LIMIT
        elif status_code in [401, 403]:
            return ErrorType.PERMANENT  # Auth errors
        elif status_code in [400, 404, 422]:
            return ErrorType.DATA  # Client errors
        elif status_code >= 500:
            return ErrorType.TRANSIENT  # Server errors
        else:
            return ErrorType.PERMANENT
    
    def _generate_idempotency_key(self, method: str, endpoint: str,
                                 data: Optional[Dict], params: Optional[Dict],
                                 context: RequestContext) -> str:
        """Generate idempotency key for request"""
        key_data = {
            "tenant_id": context.tenant_id,
            "method": method.upper(),
            "endpoint": endpoint,
            "data": data or {},
            "params": params or {},
            "execution_id": context.execution_id
        }
        
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()
    
    async def _generate_evidence(self, method: str, endpoint: str,
                               context: RequestContext,
                               response: ConnectorResponse) -> str:
        """Generate evidence pack for governance"""
        evidence_id = str(uuid.uuid4())
        
        evidence_data = {
            "evidence_id": evidence_id,
            "connector_name": self.config.name,
            "tenant_id": context.tenant_id,
            "execution_id": context.execution_id,
            "method": method.upper(),
            "endpoint": endpoint,
            "status_code": response.status_code,
            "success": response.success,
            "response_time_ms": response.response_time_ms,
            "idempotency_key": context.idempotency_key,
            "policy_id": context.policy_id or self.config.policy_id,
            "compliance_tags": self.config.compliance_tags,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Store evidence (would integrate with evidence service)
        logger.info(f"Generated evidence pack: {evidence_id}")
        
        return evidence_id
    
    # Convenience methods for common HTTP operations
    
    async def get(self, endpoint: str, context: RequestContext,
                 params: Optional[Dict] = None, headers: Optional[Dict] = None) -> ConnectorResponse:
        """GET request"""
        return await self.request("GET", endpoint, context, params=params, headers=headers)
    
    async def post(self, endpoint: str, context: RequestContext,
                  data: Optional[Dict] = None, headers: Optional[Dict] = None) -> ConnectorResponse:
        """POST request"""
        return await self.request("POST", endpoint, context, data=data, headers=headers)
    
    async def put(self, endpoint: str, context: RequestContext,
                 data: Optional[Dict] = None, headers: Optional[Dict] = None) -> ConnectorResponse:
        """PUT request"""
        return await self.request("PUT", endpoint, context, data=data, headers=headers)
    
    async def patch(self, endpoint: str, context: RequestContext,
                   data: Optional[Dict] = None, headers: Optional[Dict] = None) -> ConnectorResponse:
        """PATCH request"""
        return await self.request("PATCH", endpoint, context, data=data, headers=headers)
    
    async def delete(self, endpoint: str, context: RequestContext,
                    headers: Optional[Dict] = None) -> ConnectorResponse:
        """DELETE request"""
        return await self.request("DELETE", endpoint, context, headers=headers)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics"""
        return self.metrics.copy()

# Example connector implementations

class SalesforceConnector(ConnectorSDK):
    """Salesforce CRM connector implementation"""
    
    def __init__(self, tenant_id: str, client_id: str, client_secret: str, 
                 instance_url: str = "https://login.salesforce.com"):
        config = ConnectorConfig(
            name="salesforce",
            base_url=f"{instance_url}/services/data/v58.0/",
            auth_type=AuthType.OAUTH2,
            tenant_id=tenant_id,
            auth_config={
                "token_url": f"{instance_url}/services/oauth2/token",
                "client_id": client_id,
                "client_secret": client_secret
            },
            compliance_tags=["CRM", "GDPR", "SOX"],
            policy_id="salesforce_connector_policy_v1"
        )
        super().__init__(config)
    
    async def query_soql(self, soql: str, context: RequestContext) -> ConnectorResponse:
        """Execute SOQL query"""
        params = {"q": soql}
        return await self.get("query/", context, params=params)
    
    async def create_record(self, sobject: str, data: Dict, context: RequestContext) -> ConnectorResponse:
        """Create Salesforce record"""
        endpoint = f"sobjects/{sobject}/"
        return await self.post(endpoint, context, data=data)
    
    async def update_record(self, sobject: str, record_id: str, data: Dict, context: RequestContext) -> ConnectorResponse:
        """Update Salesforce record"""
        endpoint = f"sobjects/{sobject}/{record_id}"
        return await self.patch(endpoint, context, data=data)

class StripeConnector(ConnectorSDK):
    """Stripe billing connector implementation"""
    
    def __init__(self, tenant_id: str, api_key: str):
        config = ConnectorConfig(
            name="stripe",
            base_url="https://api.stripe.com/v1/",
            auth_type=AuthType.BEARER,
            tenant_id=tenant_id,
            auth_config={"token": api_key},
            compliance_tags=["Billing", "PCI-DSS", "GDPR"],
            policy_id="stripe_connector_policy_v1"
        )
        super().__init__(config)
    
    async def create_customer(self, customer_data: Dict, context: RequestContext) -> ConnectorResponse:
        """Create Stripe customer"""
        return await self.post("customers", context, data=customer_data)
    
    async def create_subscription(self, subscription_data: Dict, context: RequestContext) -> ConnectorResponse:
        """Create Stripe subscription"""
        return await self.post("subscriptions", context, data=subscription_data)
    
    async def list_invoices(self, customer_id: str, context: RequestContext) -> ConnectorResponse:
        """List customer invoices"""
        params = {"customer": customer_id}
        return await self.get("invoices", context, params=params)

# Example usage
async def example_usage():
    """Example of using the Connector SDK"""
    
    # Create Salesforce connector
    sf_connector = SalesforceConnector(
        tenant_id="tenant_1300",
        client_id="your_client_id",
        client_secret="your_client_secret"
    )
    
    # Create request context
    context = RequestContext(
        tenant_id="tenant_1300",
        user_id="user_123",
        execution_id=str(uuid.uuid4()),
        policy_id="salesforce_query_policy"
    )
    
    async with sf_connector:
        # Query opportunities
        response = await sf_connector.query_soql(
            "SELECT Id, Name, StageName FROM Opportunity LIMIT 10",
            context
        )
        
        if response.success:
            print(f"Found {len(response.data['records'])} opportunities")
        else:
            print(f"Query failed: {response.error_message}")
        
        # Get connector metrics
        metrics = sf_connector.get_metrics()
        print(f"Connector metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(example_usage())
