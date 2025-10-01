#!/usr/bin/env python3
"""
JWT Authentication Service - EXACT MATCH TO BACKEND PATTERN
===========================================================

Matches crenovent-backend/middlewares.js authenticateJWT function exactly.
Supports multi-tenant cookie reading, legacy fallback, and your JWT format.
"""

import jwt
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, status
import os

logger = logging.getLogger(__name__)

# JWT Configuration - matches your backend exactly
JWT_SECRET = os.getenv('JWT_SECRET', 'your-jwt-secret-key')
JWT_ALGORITHM = 'HS256'

def parse_request_context(request: Request):
    """
    Parse request context to determine cookie prefix (matches your backend pattern)
    Based on crenovent-backend/utils/multiTenant.js parseRequestContext
    """
    try:
        # Simple hostname-based logic (can be enhanced based on your needs)
        host = request.headers.get('host', '')
        if 'prod' in host or 'production' in host:
            return {'cookiePrefix': 'prod', 'hostname': host}
        else:
            return {'cookiePrefix': 'dev', 'hostname': host}
    except Exception:
        return {'cookiePrefix': 'legacy', 'hostname': 'unknown'}

class JWTAuthService:
    """JWT Authentication service matching your backend pattern exactly"""
    
    def __init__(self):
        self.secret = JWT_SECRET
        self.algorithm = JWT_ALGORITHM
        self.logger = logging.getLogger(__name__)
    
    async def get_current_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Extract and validate JWT token using EXACT same pattern as your backend
        Matches crenovent-backend/middlewares.js authenticateJWT function
        """
        try:
            # 🌐 MULTI-TENANT COOKIE READING (matches your backend exactly)
            context = parse_request_context(request)
            cookie_prefix = context['cookiePrefix']
            
            self.logger.info(f"🍪 AI Service: Looking for auth cookies with prefix: {cookie_prefix}")
            
            # Try multi-tenant cookie names first, then fall back to legacy
            jwt_cookie_name = f"{cookie_prefix}_jwt_token"
            legacy_jwt_cookie_name = 'jwtToken'
            
            token = request.cookies.get(jwt_cookie_name)
            
            if not token:
                # Fall back to legacy cookie name for backward compatibility
                token = request.cookies.get(legacy_jwt_cookie_name)
                if token:
                    self.logger.info(f"🔄 AI Service: Found token using legacy cookie name: {legacy_jwt_cookie_name}")
            else:
                self.logger.info(f"✅ AI Service: Found token using multi-tenant cookie name: {jwt_cookie_name}")
            
            # Also try Authorization header as fallback
            if not token:
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    self.logger.info("🔄 AI Service: Found token in Authorization header")
            
            if not token:
                self.logger.warning("❌ AI Service: No JWT token found in cookies or headers")
                return None
                
            # Decode and validate token (matches your backend format)
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            
            # Log the raw payload for debugging
            self.logger.info(f"🔍 AI Service: Raw JWT payload keys: {list(payload.keys())}")
            self.logger.info(f"🔍 AI Service: JWT payload sample: user_id={payload.get('user_id')}, id={payload.get('id')}, tenant_id={payload.get('tenant_id')}, email={payload.get('email')}")
            
            # Extract tenant_id from profile if not directly available
            profile = payload.get('profile', {})
            if isinstance(profile, dict):
                tenant_from_profile = profile.get('tenant_id')
            else:
                tenant_from_profile = None
            
            # Return user information matching the exact JWT structure from Node.js backend
            user_data = {
                'user_id': payload.get('user_id') or payload.get('id'),  # Handle both formats
                'username': payload.get('username'),
                'email': payload.get('email'),
                'role': payload.get('role', 'user'),
                'role_name': payload.get('role', 'user'),  # Use role field for role_name
                'role_id': payload.get('role_id'),
                'tenant_id': payload.get('tenant_id') or tenant_from_profile,  # Try both locations
                'tenant_uuid': payload.get('tenant_uuid'),
                'profile': profile,
                'is_activated': payload.get('is_activated', True),
                'teams': payload.get('teams', []),
                'has_teams': payload.get('has_teams', False),
                'isLoggedIn': payload.get('isLoggedIn', True)
            }
            
            self.logger.info(f"✅ AI Service: JWT validated for user: {user_data.get('user_id')} (tenant: {user_data.get('tenant_id')}, email: {user_data.get('email')})")
            return user_data
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("❌ AI Service: JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"❌ AI Service: Invalid JWT token - {str(e)}")
            self.logger.warning(f"❌ AI Service: Token received: {token[:50]}...")
            self.logger.warning(f"❌ AI Service: Using secret: {self.secret[:20]}...")
            return None
        except Exception as e:
            self.logger.error(f"❌ AI Service: JWT validation error: {e}")
            self.logger.error(f"❌ AI Service: Token: {token[:50]}...")
            return None
    
    def create_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """
        Create a JWT token matching your backend format exactly
        """
        payload = {
            'id': user_data.get('user_id'),
            'username': user_data.get('username'),
            'email': user_data.get('email'),
            'role': user_data.get('role', 'user'),
            'role_id': user_data.get('role_id'),
            'tenant_id': user_data.get('tenant_id'),
            'exp': datetime.utcnow() + timedelta(minutes=15),  # 15 min like your backend
            'iat': datetime.utcnow(),
            'issuer': 'crenovent-backend',  # Match your backend issuer
            'audience': 'crenovent-frontend'  # Match your backend audience
        }
        
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

# Global JWT service instance
jwt_service = JWTAuthService()

async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to get current user from JWT token
    Uses your exact authentication pattern
    """
    return await jwt_service.get_current_user(request)

async def require_auth(request: Request) -> Dict[str, Any]:
    """
    Require authentication and return user data, raise HTTPException if not authenticated
    Matches your backend error format
    """
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "Authentication required",
                "isLoggedIn": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    return user

def validate_tenant_access(user: Dict[str, Any], requested_tenant_id: int) -> bool:
    """
    Validate that user has access to the requested tenant
    """
    user_tenant_id = user.get('tenant_id')
    
    # Admin users can access any tenant (if you have admin roles)
    if user.get('role_name') == 'admin':
        return True
    
    # Regular users can only access their own tenant
    return user_tenant_id == requested_tenant_id