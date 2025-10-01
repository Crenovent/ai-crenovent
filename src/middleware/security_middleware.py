"""
Security Middleware for Enterprise AI Service
Handles authentication, authorization, rate limiting, and security headers
"""

import os
import jwt
import time
import hashlib
from typing import Optional, Dict, Any
from functools import wraps
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque

class SecurityConfig:
    """Security configuration settings"""
    
    def __init__(self):
        self.jwt_secret = os.getenv('JWT_SECRET', 'your-secret-key')
        self.jwt_algorithm = 'HS256'
        self.jwt_expiry_hours = int(os.getenv('JWT_EXPIRY_HOURS', '24'))
        self.rate_limit_requests = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))  # 1 hour
        self.max_request_size = int(os.getenv('MAX_REQUEST_SIZE', '10485760'))  # 10MB
        self.allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,https://yourdomain.com').split(',')
        self.enable_csrf_protection = os.getenv('ENABLE_CSRF_PROTECTION', 'true').lower() == 'true'

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.blocked_ips = defaultdict(float)
        self.config = SecurityConfig()
    
    def is_rate_limited(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited"""
        now = time.time()
        window = self.config.rate_limit_window
        
        # Check if IP is temporarily blocked
        if identifier in self.blocked_ips:
            if now < self.blocked_ips[identifier]:
                return True, {
                    'error': 'IP temporarily blocked',
                    'retry_after': int(self.blocked_ips[identifier] - now),
                    'block_type': 'temporary'
                }
            else:
                del self.blocked_ips[identifier]
        
        # Clean old requests
        requests = self.requests[identifier]
        while requests and requests[0] < now - window:
            requests.popleft()
        
        # Check rate limit
        if len(requests) >= self.config.rate_limit_requests:
            # Temporarily block aggressive users
            self.blocked_ips[identifier] = now + 300  # 5 minutes
            return True, {
                'error': 'Rate limit exceeded',
                'limit': self.config.rate_limit_requests,
                'window': window,
                'retry_after': 300
            }
        
        # Record this request
        requests.append(now)
        
        return False, {
            'remaining': self.config.rate_limit_requests - len(requests),
            'reset_time': int(now + window)
        }

class TokenValidator:
    """JWT token validation and user context extraction"""
    
    def __init__(self):
        self.config = SecurityConfig()
        self.token_cache = {}  # Simple token cache
    
    def validate_token(self, token: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Validate JWT token and extract user context"""
        try:
            # Check cache first
            if token in self.token_cache:
                cached_data, expiry = self.token_cache[token]
                if time.time() < expiry:
                    return True, cached_data
                else:
                    del self.token_cache[token]
            
            # Decode and validate JWT
            payload = jwt.decode(
                token, 
                self.config.jwt_secret, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Extract user context
            user_context = {
                'user_id': payload.get('user_id'),
                'tenant_id': payload.get('tenant_id'),
                'role': payload.get('role'),
                'permissions': payload.get('permissions', []),
                'segment': payload.get('segment'),
                'region': payload.get('region'),
                'expires_at': payload.get('exp')
            }
            
            # Cache for 5 minutes
            self.token_cache[token] = (user_context, time.time() + 300)
            
            return True, user_context
            
        except jwt.ExpiredSignatureError:
            return False, {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return False, {'error': 'Invalid token'}
        except Exception as e:
            return False, {'error': f'Token validation failed: {str(e)}'}

class SecurityMiddleware:
    """Comprehensive security middleware"""
    
    def __init__(self):
        self.config = SecurityConfig()
        self.rate_limiter = RateLimiter()
        self.token_validator = TokenValidator()
        
    def get_client_ip(self, request_headers: Dict[str, str]) -> str:
        """Extract client IP from headers"""
        # Check common proxy headers
        ip = (
            request_headers.get('x-forwarded-for', '').split(',')[0].strip() or
            request_headers.get('x-real-ip', '') or
            request_headers.get('cf-connecting-ip', '') or
            request_headers.get('x-client-ip', '') or
            '127.0.0.1'
        )
        return ip
    
    def validate_request_size(self, content_length: int) -> tuple[bool, Optional[str]]:
        """Validate request size"""
        if content_length > self.config.max_request_size:
            return False, f"Request too large. Max size: {self.config.max_request_size} bytes"
        return True, None
    
    def validate_cors(self, origin: str) -> bool:
        """Validate CORS origin"""
        if not origin:
            return True  # Allow requests without origin header
        return origin in self.config.allowed_origins
    
    def generate_security_headers(self) -> Dict[str, str]:
        """Generate security headers"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    def validate_csrf_token(self, request_headers: Dict[str, str], user_context: Dict[str, Any]) -> bool:
        """Validate CSRF token"""
        if not self.config.enable_csrf_protection:
            return True
        
        csrf_token = request_headers.get('x-csrf-token', '')
        if not csrf_token:
            return False
        
        # Generate expected CSRF token
        expected_token = self.generate_csrf_token(user_context)
        return csrf_token == expected_token
    
    def generate_csrf_token(self, user_context: Dict[str, Any]) -> str:
        """Generate CSRF token"""
        user_id = str(user_context.get('user_id', ''))
        tenant_id = str(user_context.get('tenant_id', ''))
        timestamp = str(int(time.time() // 300))  # 5-minute window
        
        token_data = f"{user_id}:{tenant_id}:{timestamp}:{self.config.jwt_secret}"
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    async def process_request(self, 
                            method: str, 
                            path: str, 
                            headers: Dict[str, str], 
                            content_length: int = 0) -> tuple[bool, Dict[str, Any]]:
        """Main security processing function"""
        
        # Extract client information
        client_ip = self.get_client_ip(headers)
        user_agent = headers.get('user-agent', '')
        origin = headers.get('origin', '')
        
        # Basic security validations
        security_result = {
            'client_ip': client_ip,
            'user_agent': user_agent,
            'security_headers': self.generate_security_headers(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # 1. Validate request size
        size_valid, size_error = self.validate_request_size(content_length)
        if not size_valid:
            security_result['error'] = size_error
            security_result['status_code'] = 413
            return False, security_result
        
        # 2. Validate CORS
        if not self.validate_cors(origin):
            security_result['error'] = 'CORS validation failed'
            security_result['status_code'] = 403
            return False, security_result
        
        # 3. Rate limiting
        rate_limited, rate_info = self.rate_limiter.is_rate_limited(client_ip)
        if rate_limited:
            security_result.update(rate_info)
            security_result['status_code'] = 429
            return False, security_result
        
        security_result.update(rate_info)
        
        # 4. JWT Authentication (skip for public endpoints)
        public_endpoints = ['/health', '/docs', '/openapi.json']
        if path not in public_endpoints:
            auth_header = headers.get('authorization', '')
            if not auth_header.startswith('Bearer '):
                security_result['error'] = 'Missing or invalid authorization header'
                security_result['status_code'] = 401
                return False, security_result
            
            token = auth_header[7:]  # Remove 'Bearer '
            token_valid, user_context = self.token_validator.validate_token(token)
            
            if not token_valid:
                security_result['error'] = user_context.get('error', 'Token validation failed')
                security_result['status_code'] = 401
                return False, security_result
            
            security_result['user_context'] = user_context
            
            # 5. CSRF Protection for state-changing requests
            if method in ['POST', 'PUT', 'DELETE', 'PATCH']:
                if not self.validate_csrf_token(headers, user_context):
                    security_result['error'] = 'CSRF token validation failed'
                    security_result['status_code'] = 403
                    return False, security_result
        
        return True, security_result

# Singleton instance
security_middleware = SecurityMiddleware()

def require_auth(permissions: list = None):
    """Decorator for endpoints requiring authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would be implemented in the specific framework (FastAPI, Flask, etc.)
            pass
        return wrapper
    return decorator

def rate_limit(requests_per_hour: int = 100):
    """Decorator for rate limiting specific endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Implementation would depend on the framework
            pass
        return wrapper
    return decorator
